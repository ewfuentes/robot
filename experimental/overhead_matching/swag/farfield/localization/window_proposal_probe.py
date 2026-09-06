"""Run the window-joint proposal at chosen keyframes of a localization_inputs
export and report where the truth ranks — a standalone check of the generator
(no filter run).

  bazel run //experimental/overhead_matching/swag/farfield/localization:window_proposal_probe -- \
      --input_dir /data/farfield_matching/artifacts/localization_inputs/flevoland_polder/<ver> \
      --keyframes 123 319
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    structs,
    window_proposal,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--keyframes", type=int, nargs="+", required=True)
    parser.add_argument("--window_keyframes", type=int, default=20)
    parser.add_argument("--max_tracklets", type=int, default=8)
    parser.add_argument("--tolerance_deg", type=float, default=1.5)
    parser.add_argument("--max_outliers", type=int, default=2)
    parser.add_argument("--budget", type=int, default=25000)
    parser.add_argument("--debug_truth", action="store_true")
    args = parser.parse_args()

    data = export_ingest.load(Path(args.input_dir))
    truth = {t.keyframe_idx: t for t in data.truth}
    config = structs.ProposalConfig(
        generator="window_joint", window_joint_keyframes=args.window_keyframes,
        window_joint_max_tracklets=args.max_tracklets,
        window_joint_rms_tolerance_deg=args.tolerance_deg,
        window_joint_max_outlier_tracklets=args.max_outliers)
    rng = np.random.default_rng(0)
    for kf in args.keyframes:
        tracks = window_proposal.collect_tracks(
            data.measurements, data.tables, data.catalog, config, kf)
        east = np.asarray(data.catalog.east_m, float)
        north = np.asarray(data.catalog.north_m, float)
        print(f"\n-- kf {kf} window tracklets (cap m, K, epochs, anchors):")
        for t in tracks:
            print(f"   {t.tracklet_id.split('#')[-1]:6s} cap {t.cap_m:7.0f} K {len(t.cand_idx):5d} "
                  f"epochs {len(t.epochs)} at {[e[0] for e in t.epochs]} caps {[int(e[3]) for e in t.epochs]}")
        bbox = (east.min() - 2000, east.max() + 2000, north.min() - 2000, north.max() + 2000)
        import itertools
        n_gen = min(config.window_joint_resection_tracklets, len(tracks))
        for trip in itertools.combinations(range(n_gen), 3):
            k_ref, chosen = window_proposal.reference_epochs([tracks[i] for i in trip])
            poses, idents, n_total, n_pruned = window_proposal.resect_snapshot(
                [tracks[i] for i in trip], east, north, bbox,
                config.window_joint_max_tuples, np.random.default_rng(0), epochs=chosen)
            print(f"   triple {trip} @kf {k_ref} (offsets {[e[0]-k_ref for e in chosen]}): {n_total:.3g} tuples, {n_pruned} after prune, {len(poses)} snapshot fixes")
        tr0 = truth.get(kf)
        if tr0 is not None and args.debug_truth:
            import dataclasses
            rel = window_proposal.relative_poses(data.odometry, kf, config.window_joint_keyframes)
            pose_t = np.array([[tr0.east_m, tr0.north_m, math.radians(tr0.course_world_cw_deg)]])
            # truth identity per track = best in-cap row by window rms at the truth pose
            ident = {}
            for t in tracks:
                arr = window_proposal._epoch_arrays(t, rel)
                u, v, dtheta, bearing, kappa, cap, yv = arr
                px, py = window_proposal._platform_at(pose_t, u, v)
                best = (None, 9.0)
                for j in t.cand_idx:
                    res = (np.arctan2(east[j] - px, north[j] - py) - pose_t[0, 2] - dtheta - bearing + math.pi) % (2 * math.pi) - math.pi
                    rng_ok = (np.hypot(east[j] - px, north[j] - py) <= cap * 1.25).all()
                    rms = float(np.sqrt(np.mean(res ** 2)))
                    if rng_ok and rms < best[1]:
                        best = (int(j), rms)
                ident[t.tracklet_id] = best
            for trip in itertools.combinations(range(n_gen), 3):
                tr3 = [tracks[i] for i in trip]
                ids = [ident[t.tracklet_id] for t in tr3]
                if any(i[0] is None or math.degrees(i[1]) > 2.5 for i in ids):
                    print(f"   [truth] triple {trip}: not truth-consistent {[round(math.degrees(i[1]),1) if i[0] is not None else None for i in ids]}")
                    continue
                k_ref, chosen = window_proposal.reference_epochs(tr3)
                offsets = [math.hypot(rel[e[0]][0] - rel[k_ref][0], rel[e[0]][1] - rel[k_ref][1]) for e in chosen]
                narrow = [dataclasses.replace(t, cand_idx=np.array([i[0]]), cand_w=np.array([1.0])) for t, i in zip(tr3, ids)]
                poses, idents, n_total, n_pruned = window_proposal.resect_snapshot(
                    narrow, east, north, bbox, 10, np.random.default_rng(0), epochs=chosen, offsets_m=offsets)
                trk = truth.get(k_ref)
                msg = f"   [truth] triple {trip} @kf {k_ref} offsets {[round(o) for o in offsets]} m, tuple prune {n_pruned}/{n_total}, fixes {len(poses)}"
                if len(poses) and trk is not None:
                    d = np.hypot(poses[:, 0] - trk.east_m, poses[:, 1] - trk.north_m)
                    cur = window_proposal.to_current_frame(poses, rel[k_ref])
                    ref = window_proposal.refine(cur, idents, tr3, rel, east, north)
                    own = window_proposal.own_track_rms(ref, idents, tr3, rel, east, north, math.radians(config.window_joint_rms_tolerance_deg))
                    d2 = np.hypot(ref[:, 0] - tr0.east_m, ref[:, 1] - tr0.north_m)
                    msg += f"; nearest fix to truth@k_ref {d.min():.0f} m; after move+refine nearest {d2.min():.0f} m, own rms/tol {np.round(own[np.argmin(d2)], 2)}"
                print(msg)
        if tr0 is not None:
            sc = window_proposal.incumbent_score(
                np.array([tr0.east_m]), np.array([tr0.north_m]),
                np.array([math.radians(tr0.course_world_cw_deg)]),
                data.measurements, data.odometry, data.tables, data.catalog, config, kf)
            if sc is not None:
                rel = window_proposal.relative_poses(data.odometry, kf, config.window_joint_keyframes)
                per = []
                for t in tracks:
                    arr = window_proposal._epoch_arrays(t, rel)
                    u, v, dtheta, bearing, kappa, cap, yv = arr
                    px, py = window_proposal._platform_at(np.array([[tr0.east_m, tr0.north_m, math.radians(tr0.course_world_cw_deg)]]), u, v)
                    best = None
                    for j in t.cand_idx:
                        res = np.arctan2(east[j] - px, north[j] - py) - math.radians(tr0.course_world_cw_deg) - dtheta - bearing
                        res = (res + math.pi) % (2 * math.pi) - math.pi
                        rms = float(np.sqrt(np.mean(res ** 2)))
                        if best is None or rms < best:
                            best = rms
                    per.append((round(math.degrees(best), 1), round(math.degrees(window_proposal.track_tolerance(arr, math.radians(config.window_joint_rms_tolerance_deg))), 1)))
                print(f"   truth pose: consistent {int(sc.n_consistent[0])}/{len(tracks)}, (best-any-row rms, tolerance) per track (deg) {per}")
        t0 = time.time()
        result = window_proposal.propose(
            data.measurements, data.odometry, data.tables, data.catalog, config,
            event_id=0, keyframe_idx=kf, trigger="probe",
            particle_budget=args.budget, rng=rng)
        elapsed = time.time() - t0
        tr = truth.get(kf)
        print(f"\n== kf {kf}: {result.n_tracklets_considered} tracklets, "
              f"{result.n_combinations_total:.3g} triples, "
              f"{result.n_combinations_enumerated:.3g} after cap prune, "
              f"{len(result.hypotheses)} hypotheses kept, "
              f"tier mass {result.represented_compatibility_mass:.2f}, {elapsed:.1f} s")
        if not result.hypotheses:
            continue
        hyps = result.hypotheses
        if tr is None:
            continue
        dist = np.array([math.hypot(h.east_m - tr.east_m, h.north_m - tr.north_m)
                         for h in hyps])
        dhead = np.array([abs(math.degrees(
            (h.heading_rad - math.radians(tr.course_world_cw_deg) + math.pi)
            % (2 * math.pi) - math.pi)) for h in hyps])
        near = (dist < 150) & (dhead < 5)
        order = np.argsort([-h.compatibility_mass for h in hyps])
        rank = [i for i, idx in enumerate(order) if near[idx]]
        mass_near = sum(h.compatibility_mass for h, n in zip(hyps, near) if n)
        print(f"   near-truth hypotheses: {int(near.sum())}, best rank "
              f"{rank[0] + 1 if rank else None} of {len(hyps)}, mass share on truth "
              f"{mass_near:.3f}; nearest {dist.min():.0f} m")
        for i in order[:8]:
            h = hyps[i]
            print(f"   mass {h.compatibility_mass:.3f} tracks {len(h.tracklet_ids)} "
                  f"rms {math.degrees(h.residual_rad):.2f} deg  "
                  f"{dist[i]:7.0f} m from truth, dheading {dhead[i]:5.1f} deg")


if __name__ == "__main__":
    main()
