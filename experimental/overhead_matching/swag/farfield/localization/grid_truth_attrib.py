"""Diagnostic: attribute truth-mass loss in the eager grid filter to
individual measurements and motion steps.  Uses privileged truth; never an
evaluation.  Writes one JSONL row per measurement / motion step."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    grid_filter as gf,
    odometry_profiles,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--odometry_profile",
                   default="epson_mg570_calibrated_planar_v1")
    p.add_argument("--cell_m", type=float, default=100.0)
    p.add_argument("--n_heading", type=int, default=36)
    p.add_argument("--pi0", type=float, default=0.2)
    p.add_argument("--matcher_recall", type=float, default=0.5)
    p.add_argument("--identity_share", default="constant")
    p.add_argument("--identity_within", default="odds")
    p.add_argument("--heading_rw_deg", type=float, default=1.0)
    p.add_argument("--diffusion_m", type=float, default=5.0)
    p.add_argument("--range_cap", type=int, default=1)
    p.add_argument("--range_softness", type=float, default=0.25)
    p.add_argument("--init_truth_sigma_m", type=float, default=0.0)
    p.add_argument("--release_schedule", default=None,
                   help="natural joint mode: one whole-track factor per release")
    p.add_argument("--joint_slack", type=int, default=1)
    p.add_argument("--joint_temper", type=float, default=1.0)
    p.add_argument("--heading_rw_deg_joint", type=float, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = export_ingest.load(Path(args.input_dir))
    data.odometry, _ = odometry_profiles.derive(
        Path(args.input_dir), data, args.odometry_profile, noise_seed=0)
    n_kf = len(data.truth)
    catalog = data.catalog
    sigma_pos = float(catalog.position_sigma_m[0])
    region = export_ingest.prior_box(data, 0.0)
    grid = gf.Grid(region.east_min_m, region.east_max_m,
                   region.north_min_m, region.north_max_m, args.cell_m)
    belief = gf.GridBelief(grid, args.n_heading, args.device)
    truth = {t.keyframe_idx: t for t in data.truth}
    odometry = {o.keyframe_idx: o for o in data.odometry}

    # truth heading = yaw0 + integrated odometry yaw; yaw0 fit to GPS course
    yaw = np.zeros(n_kf)
    for k in range(1, n_kf):
        yaw[k] = yaw[k - 1] + odometry[k].delta_yaw_cw_rad
    course = np.radians([truth[k].course_world_cw_deg for k in range(n_kf)])
    diff = course - yaw
    yaw0 = math.atan2(np.sin(diff).mean(), np.cos(diff).mean())
    heading = (yaw + yaw0) % (2 * math.pi)
    binw = 2 * math.pi / args.n_heading

    if args.init_truth_sigma_m > 0:
        pose0 = truth[0]
        d2 = ((belief.cell_east - pose0.east_m) ** 2
              + (belief.cell_north - pose0.north_m) ** 2)
        g = torch.exp(-0.5 * d2 / args.init_truth_sigma_m ** 2)
        belief.belief = (g / g.sum() / args.n_heading).expand(
            args.n_heading, -1).reshape(belief.belief.shape).clone()

    cat_e = torch.tensor(catalog.east_m, dtype=torch.float32,
                         device=args.device)
    cat_n = torch.tensor(catalog.north_m, dtype=torch.float32,
                         device=args.device)
    cat_e_np, cat_n_np = catalog.east_m, catalog.north_m
    cache = {}

    def cand(tid):
        if tid not in cache:
            table = data.tables[tid]
            lw = gf.identity_log_weights(
                table, catalog, args.matcher_recall, args.identity_share,
                args.identity_within)
            w = np.exp(lw)
            endorsed = ~gf.filter_lib._surprise_mask(  # noqa: SLF001
                table, gf.filter_lib._clipped_log_lr(table, catalog))  # noqa
            idx = np.nonzero(endorsed)[0]
            tail = float(w[~endorsed].sum())
            it = torch.as_tensor(idx, device=args.device)
            cache[tid] = (cat_e[it], cat_n[it], torch.tensor(
                w[idx], dtype=torch.float32, device=args.device), tail, idx,
                w[idx])
        return cache[tid]

    masks = gf.truth_masks(grid, data.truth, (500.0,))

    def truth_mass(msg, k):
        idx, frac = masks[(k, 500.0)]
        marg = msg.sum(dim=0).reshape(-1)
        return float((marg[torch.as_tensor(idx, device=args.device)]
                      * torch.as_tensor(frac, device=args.device)).sum())

    def state_of(east, north, hdg):
        j = min(max(int((east - grid.east_min) // grid.cell_m), 0),
                grid.n_east - 1)
        i = min(max(int((north - grid.north_min) // grid.cell_m), 0),
                grid.n_north - 1)
        h = int(round(hdg / binw)) % args.n_heading
        return h, i, j

    def dominant(tid, east, north, hdg, bearing_rad, kappa, rmax):
        """Per-candidate contribution at one state (numpy)."""
        _, _, _, tail, idx, w = cand(tid)
        de = cat_e_np[idx] - east
        dn = cat_n_np[idx] - north
        dist = np.hypot(de, dn)
        wb = np.arctan2(de, dn)
        sd = np.maximum(dist, 1.0)
        gate = w.copy()
        if rmax is not None and args.range_cap:
            ex = np.maximum(dist - rmax, 0.0)
            gate = gate * np.exp(-0.5 * (ex / (args.range_softness * rmax)) ** 2)
        var = (1.0 / kappa + (sigma_pos / sd) ** 2 + binw * binw / 12.0
               + (grid.cell_m / math.sqrt(12.0) / sd) ** 2)
        ke = 1.0 / var
        res = (wb - (hdg + bearing_rad) + math.pi) % (2 * math.pi) - math.pi
        from scipy.special import i0e
        contrib = np.exp(ke * (np.cos(res) - 1.0)) / i0e(ke) * gate
        if contrib.size == 0:
            return None
        b = int(np.argmax(contrib))
        return {"id": catalog.landmark_ids[idx[b]], "w": float(w[b]),
                "dist_m": float(dist[b]), "res_deg": float(np.degrees(res[b])),
                "gate": float(gate[b] / max(w[b], 1e-30)),
                "contrib": float(contrib[b]), "n_end": int(idx.size)}

    by_kf = {}
    for m in data.measurements:
        by_kf.setdefault(m.anchor_keyframe_idx, []).append(m)
    rw = math.radians(args.heading_rw_deg)
    rows = []
    out = open(args.out, "w")
    east_c, north_c = grid.centers()

    if args.release_schedule:
        from experimental.overhead_matching.swag.farfield.localization import (
            release_schedule as rs_lib)
        from scipy.special import i0e
        releases = rs_lib.load_sidecar(Path(args.release_schedule), data)
        at = {}
        for r in releases:
            at.setdefault(r.release_keyframe_idx, []).append(r)

        def joint_dominant(tid, epochs, east, north, hdg):
            _, _, _, tail, idx, w = cand(tid)
            de0 = cat_e_np[idx]
            dn0 = cat_n_np[idx]
            logp = np.zeros(idx.size)
            worst = np.zeros(idx.size)
            for (d_fwd, d_left, d_head, br, base_var, rmax, hsv, psm) in epochs:
                pe = east + d_fwd * math.sin(hdg) - d_left * math.cos(hdg)
                pn = north + d_fwd * math.cos(hdg) + d_left * math.sin(hdg)
                de = de0 - pe
                dn = dn0 - pn
                dist = np.hypot(de, dn)
                sd = np.maximum(dist, 1.0)
                var = (base_var + (sigma_pos / sd) ** 2 + binw * binw / 12
                       + (grid.cell_m / math.sqrt(12) / sd) ** 2 + hsv
                       + (psm / sd) ** 2)
                ke = 1.0 / var
                res = (np.arctan2(de, dn) - (hdg + d_head + br) + math.pi) \
                    % (2 * math.pi) - math.pi
                term = ke * (np.cos(res) - 1.0) - np.log(i0e(ke))
                if rmax is not None and args.range_cap:
                    ex = np.maximum(dist - rmax, 0.0)
                    term = term + np.log(np.exp(
                        -0.5 * (ex / (args.range_softness * rmax)) ** 2) + 1e-30)
                logp += args.joint_temper * term
                worst = np.minimum(worst, term)
            if idx.size == 0:
                return None
            b = int(np.argmax(np.log(w + 1e-300) + logp))
            de = de0[b] - east
            dn = dn0[b] - north
            return {"id": catalog.landmark_ids[idx[b]], "w": float(w[b]),
                    "dist_m": float(math.hypot(de, dn)),
                    "logp": float(logp[b]), "worst_epoch": float(worst[b]),
                    "n_end": int(idx.size)}

        for k in range(n_kf):
            tp = truth[k]
            if k > 0:
                before = truth_mass(belief.belief, k - 1)
                belief.motion(odometry[k], 1.0, rw, args.diffusion_m)
                belief.renormalize()
                after = truth_mass(belief.belief, k)
                out.write(json.dumps({"kind": "motion", "kf": k,
                                      "tm_before": before,
                                      "tm_after": after}) + "\n")
            for rel in at.get(k, ()):
                ce, cn, cw, tail, _, _ = cand(rel.tracklet_id)
                anchors = sorted({m.anchor_keyframe_idx
                                  for m in rel.measurements})
                relp = gf.relative_epoch_poses(odometry, anchors, k)
                epochs = []
                for m in rel.measurements:
                    d_fwd, d_left, d_head = relp[m.anchor_keyframe_idx]
                    kappa = min(float(m.kappa), gf.MAX_KAPPA)
                    hsv = 0.0
                    psv = 0.0
                    if args.joint_slack:
                        for step in range(m.anchor_keyframe_idx + 1, k + 1):
                            dlt = odometry[step]
                            hsv += rw ** 2 + dlt.sigma_yaw_rad ** 2
                            psv += args.diffusion_m ** 2 + dlt.sigma_m ** 2
                    epochs.append((d_fwd, d_left, d_head,
                                   math.radians(m.bearing_forward_cw_deg),
                                   1.0 / kappa,
                                   m.range_max_m if args.range_cap else None,
                                   hsv, math.sqrt(psv)))
                L = belief.track_joint_likelihood(
                    epochs, ce, cn, cw, sigma_pos, args.pi0, tail,
                    range_softness=args.range_softness,
                    temper=args.joint_temper)
                h, i, j = state_of(tp.east_m, tp.north_m, heading[k])
                hs = [(h + d) % args.n_heading for d in (-1, 0, 1)]
                i0, i1 = max(i - 1, 0), min(i + 2, grid.n_north)
                j0, j1 = max(j - 1, 0), min(j + 2, grid.n_east)
                l_truth = float(L[hs][:, i0:i1, j0:j1].max())
                floor = args.pi0 + (1 - args.pi0) * tail
                expect = float((belief.belief * L).sum())
                tm_before = truth_mass(belief.belief, k)
                post = belief.belief * L
                am = int(torch.argmax(post))
                ah, rest = divmod(am, grid.n_north * grid.n_east)
                ai, aj = divmod(rest, grid.n_east)
                belief.belief = gf._normalized(post)  # noqa: SLF001
                tm_after = truth_mass(belief.belief, k)
                out.write(json.dumps({
                    "kind": "meas", "kf": k,
                    "track": rel.tracklet_id.split("#")[-1],
                    "n_epochs": len(epochs),
                    "span_kf": k - min(anchors),
                    "bearing": rel.measurements[-1].bearing_forward_cw_deg,
                    "kappa": float(np.median([e[4] ** -1 for e in epochs])),
                    "rmax": min([m.range_max_m or 1e9
                                 for m in rel.measurements]),
                    "tail": tail, "l_truth": l_truth,
                    "l_truth_exact": float(L[h, i, j]),
                    "l_floor": floor, "l_expect": expect,
                    "l_max": float(L.max()),
                    "factor_truth": l_truth / expect,
                    "tm_before": tm_before, "tm_after": tm_after,
                    "dom_truth": joint_dominant(
                        rel.tracklet_id, epochs, tp.east_m, tp.north_m,
                        heading[k]),
                    "argmax_post": {
                        "east": float(east_c[aj]),
                        "north": float(north_c[ai]),
                        "dist_to_truth_m": math.hypot(
                            east_c[aj] - tp.east_m, north_c[ai] - tp.north_m),
                        "heading_deg": math.degrees(ah * binw),
                        "dom": joint_dominant(
                            rel.tracklet_id, epochs, float(east_c[aj]),
                            float(north_c[ai]), ah * binw)},
                }) + "\n")
            if k % 40 == 0 or k == n_kf - 1:
                print(f"kf {k:4d} truth mass500 "
                      f"{truth_mass(belief.belief, k):.4f}")
        out.close()
        return
    for k in range(n_kf):
        tp = truth[k]
        if k > 0:
            before = truth_mass(belief.belief, k - 1)
            belief.motion(odometry[k], 1.0, rw, args.diffusion_m)
            belief.renormalize()
            after = truth_mass(belief.belief, k)
            out.write(json.dumps({"kind": "motion", "kf": k,
                                  "tm_before": before, "tm_after": after})
                      + "\n")
        for m in by_kf.get(k, ()):
            ce, cn, cw, tail, _, _ = cand(m.tracklet_id)
            kappa = min(float(m.kappa), gf.MAX_KAPPA)
            L = belief.track_likelihood(
                math.radians(m.bearing_forward_cw_deg), 1.0 / kappa,
                ce, cn, cw, sigma_pos, args.pi0, tail,
                range_max_m=m.range_max_m if args.range_cap else None,
                range_softness=args.range_softness)
            h, i, j = state_of(tp.east_m, tp.north_m, heading[k])
            hs = [(h + d) % args.n_heading for d in (-1, 0, 1)]
            i0, i1 = max(i - 1, 0), min(i + 2, grid.n_north)
            j0, j1 = max(j - 1, 0), min(j + 2, grid.n_east)
            l_truth_exact = float(L[h, i, j])
            l_truth = float(L[hs][:, i0:i1, j0:j1].max())
            floor = args.pi0 + (1 - args.pi0) * tail
            expect = float((belief.belief * L).sum())
            l_max = float(L.max())
            tm_before = truth_mass(belief.belief, k)
            post = belief.belief * L
            am = int(torch.argmax(post))
            ah, rest = divmod(am, grid.n_north * grid.n_east)
            ai, aj = divmod(rest, grid.n_east)
            belief.belief = gf._normalized(post)  # noqa: SLF001
            tm_after = truth_mass(belief.belief, k)
            br = math.radians(m.bearing_forward_cw_deg)
            row = {
                "kind": "meas", "kf": k,
                "track": m.tracklet_id.split("#")[-1],
                "bearing": m.bearing_forward_cw_deg, "kappa": kappa,
                "rmax": m.range_max_m, "tail": tail,
                "l_truth": l_truth, "l_truth_exact": l_truth_exact,
                "l_floor": floor, "l_expect": expect, "l_max": l_max,
                "factor_truth": l_truth / expect,
                "tm_before": tm_before, "tm_after": tm_after,
                "dom_truth": dominant(
                    m.tracklet_id, tp.east_m, tp.north_m, heading[k], br,
                    kappa, m.range_max_m),
                "argmax_post": {
                    "east": float(east_c[aj]), "north": float(north_c[ai]),
                    "dist_to_truth_m": math.hypot(
                        east_c[aj] - tp.east_m, north_c[ai] - tp.north_m),
                    "heading_deg": math.degrees(ah * binw),
                    "dom": dominant(
                        m.tracklet_id, float(east_c[aj]), float(north_c[ai]),
                        ah * binw, br, kappa, m.range_max_m)},
            }
            out.write(json.dumps(row) + "\n")
        if k % 40 == 0 or k == n_kf - 1:
            print(f"kf {k:4d} truth mass500 {truth_mass(belief.belief, k):.4f}")
    out.close()


if __name__ == "__main__":
    main()
