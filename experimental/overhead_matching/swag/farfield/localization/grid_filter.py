"""Exact grid-HMM localization over a localization_inputs export (scratch).

Histogram filter over (heading, north, east) cells consuming the SAME
per-epoch bearing mixture the particle filter applies (design doc §5.3,
independent-epoch marginalization — no association persistence yet):

    p(z | cell, heading) = pi0/2pi
        + (1 - pi0) * sum_j p(j | appearance) * vM(delta_j; kappa_eff_j)

with p(j | appearance) from filter._identity_log_weights and kappa_eff from
LandmarkCatalog.kappa_eff, so the observation model matches the PF exactly.
Exact inference: no particles, no depletion, no proposals, no seeds.

Motion is the PF's rotate-then-move increment applied per heading bin:
heading advances by a wrapped-Gaussian circular kernel (delta_yaw with a
configurable scale on the export's sigma_yaw, which may overstate the true
course noise), position advances per-bin through a sub-cell displacement
accumulator (integer shifts only — repeated bilinear shifts of ~3 m steps
against 100-200 m cells would inject artificial diffusion far above the
modeled odometry noise).

Run:
  bazel run //experimental/overhead_matching/swag/farfield/localization:grid_filter -- \
    --input_dir /data/farfield_matching/artifacts/localization_inputs/<ds>/<ver> \
    --like_run /data/farfield_matching/runs/<scenario>/<run>  # box + pi0 parity
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter as filter_lib,
)

MAX_KAPPA = filter_lib.MAX_KAPPA


class Grid:
    """Cell geometry: index 0 of each axis at the box minimum."""

    def __init__(self, east_min, east_max, north_min, north_max, cell_m):
        self.cell_m = float(cell_m)
        self.east_min = float(east_min)
        self.north_min = float(north_min)
        self.n_east = int(math.ceil((east_max - east_min) / cell_m))
        self.n_north = int(math.ceil((north_max - north_min) / cell_m))

    def centers(self):
        east = self.east_min + (np.arange(self.n_east) + 0.5) * self.cell_m
        north = self.north_min + (np.arange(self.n_north) + 0.5) * self.cell_m
        return east, north


def shift2d(t: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """Integer zero-fill shift of the trailing two (north, east) axes."""
    if di == 0 and dj == 0:
        return t
    out = torch.zeros_like(t)
    ni, nj = t.shape[-2], t.shape[-1]
    if abs(di) >= ni or abs(dj) >= nj:
        return out
    src_i = slice(max(0, -di), ni - max(0, di))
    dst_i = slice(max(0, di), ni - max(0, -di))
    src_j = slice(max(0, -dj), nj - max(0, dj))
    dst_j = slice(max(0, dj), nj - max(0, -dj))
    out[..., dst_i, dst_j] = t[..., src_i, src_j]
    return out


def heading_kernel(shift_rad: float, sigma_rad: float, n_heading: int):
    """Wrapped-Gaussian weights over integer bin offsets (sums to 1)."""
    binw = 2.0 * math.pi / n_heading
    half = min(n_heading // 2,
               int(math.ceil((4.0 * sigma_rad + abs(shift_rad)) / binw)) + 1)
    offsets = np.arange(-half, half + 1)
    sigma = max(sigma_rad, 1e-6)
    weights = np.zeros(offsets.shape)
    for wrap in (-1, 0, 1):
        delta = offsets * binw - shift_rad + wrap * 2.0 * math.pi
        weights += np.exp(-0.5 * (delta / sigma) ** 2)
    return offsets, weights / weights.sum()


def gaussian_taps(sigma_cells: float):
    half = max(1, int(math.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offsets / max(sigma_cells, 1e-6)) ** 2)
    return offsets, weights / weights.sum()


class GridBelief:
    """Linear-space belief over (n_heading, n_north, n_east) on `device`.

    The per-epoch mixture likelihood is floored at pi0/2pi, so one update's
    dynamic range is bounded and linear float32 with per-keyframe
    renormalization is safe (same argument as the PF's damage cap, exact
    here).
    """

    def __init__(self, grid: Grid, n_heading: int, device: str):
        self.grid = grid
        self.n_heading = n_heading
        self.device = device
        self.belief = torch.full(
            (n_heading, grid.n_north, grid.n_east),
            1.0 / (n_heading * grid.n_north * grid.n_east),
            dtype=torch.float32, device=device)
        east, north = grid.centers()
        jj, ii = np.meshgrid(east, north)  # (n_north, n_east)
        self.cell_east = torch.tensor(
            jj.ravel(), dtype=torch.float32, device=device)
        self.cell_north = torch.tensor(
            ii.ravel(), dtype=torch.float32, device=device)
        bins = 2.0 * math.pi * np.arange(n_heading) / n_heading
        self.bin_rad = bins
        # Motion accumulators (applied when they cross a cell/bin fraction).
        self.pending_yaw = 0.0
        self.pending_yaw_var = 0.0
        self.pending_de = np.zeros(n_heading)
        self.pending_dn = np.zeros(n_heading)
        self.pending_pos_var = 0.0
        self.leaked_mass = 0.0

    def motion(self, delta, yaw_sigma_scale: float, heading_rw_rad: float,
               diffusion_m: float):
        binw = 2.0 * math.pi / self.n_heading
        sigma_h = math.hypot(delta.sigma_yaw_rad * yaw_sigma_scale,
                             heading_rw_rad)
        self.pending_yaw += delta.delta_yaw_cw_rad
        self.pending_yaw_var += sigma_h * sigma_h
        if math.sqrt(self.pending_yaw_var) >= 0.35 * binw:
            offsets, weights = heading_kernel(
                self.pending_yaw, math.sqrt(self.pending_yaw_var),
                self.n_heading)
            acc = torch.zeros_like(self.belief)
            for offset, weight in zip(offsets, weights):
                # heading' = heading + offset*binw  =>  read from bin (b-offset)
                acc += weight * torch.roll(self.belief, int(offset), dims=0)
            self.belief = acc
            self.pending_yaw = 0.0
            self.pending_yaw_var = 0.0
        elif abs(self.pending_yaw) >= 0.5 * binw:
            steps = int(round(self.pending_yaw / binw))
            self.belief = torch.roll(self.belief, steps, dims=0)
            self.pending_yaw -= steps * binw

        # Rotate-then-move with the post-update heading of each bin.
        sin_h = np.sin(self.bin_rad)
        cos_h = np.cos(self.bin_rad)
        self.pending_de += delta.forward_m * sin_h - delta.left_m * cos_h
        self.pending_dn += delta.forward_m * cos_h + delta.left_m * sin_h
        cell = self.grid.cell_m
        for b in range(self.n_heading):
            dj = int(round(self.pending_de[b] / cell))
            di = int(round(self.pending_dn[b] / cell))
            if di or dj:
                self.belief[b] = shift2d(self.belief[b], di, dj)
                self.pending_de[b] -= dj * cell
                self.pending_dn[b] -= di * cell

        self.pending_pos_var += delta.sigma_m ** 2 + diffusion_m ** 2
        if math.sqrt(self.pending_pos_var) >= 0.25 * cell:
            sigma_cells = math.sqrt(self.pending_pos_var) / cell
            offsets, weights = gaussian_taps(sigma_cells)
            for axis in (-2, -1):
                acc = torch.zeros_like(self.belief)
                for offset, weight in zip(offsets, weights):
                    shifted = shift2d(
                        self.belief,
                        int(offset) if axis == -2 else 0,
                        int(offset) if axis == -1 else 0)
                    acc += weight * shifted
                self.belief = acc
            self.pending_pos_var = 0.0

    def track_likelihood(self, epochs, cand_east: torch.Tensor,
                         cand_north: torch.Tensor, cand_weight: torch.Tensor,
                         sigma_pos: float, pi0: float, tail_mass: float,
                         quantization_comp: bool = True,
                         chunk: int = 256) -> torch.Tensor:
        """Track-joint §5.3 mixture over `epochs` = [(z_rad, base_var)].

        Marginalizes the track's identity ONCE outside the product over its
        epochs (the outer-sum form the PF's association persistence
        estimates by sampling):

            L = pi0 (1/2pi)^n + (1-pi0) [sum_j w_j prod_i vM(z_i; kappa_i)
                                         + tail (1/2pi)^n]

        Each epoch's bearing arrives pre-rotated into the current frame and
        `base_var` carries its measurement variance plus accumulated yaw
        drift since its keyframe. A single epoch reproduces the PF's
        independent-epoch mixture exactly.

        `quantization_comp` marginalizes the within-cell/within-bin pose
        instead of evaluating at the center: a kappa~700 bearing (sigma 2
        deg) evaluated at wide bin centers kills every pose whose true
        heading is a few degrees off-center — truth included. Boxcar
        variances add: heading binw^2/12, position (cell/sqrt(12)/r)^2.
        """
        n_cells = self.cell_east.shape[0]
        n_heading = self.n_heading
        two_pi = 2.0 * math.pi
        # Everything is scaled by (2pi)^n — the likelihood RELATIVE to the
        # null density — so long tracks cannot underflow float32; the
        # constant cancels in the telescoping ratio and renormalization.
        floor = pi0 + (1.0 - pi0) * tail_mass
        like = torch.full((n_heading, n_cells), floor,
                          dtype=torch.float32, device=self.device)
        binw = two_pi / n_heading
        for start in range(0, cand_east.shape[0], chunk):
            sl = slice(start, min(start + chunk, cand_east.shape[0]))
            d_east = cand_east[sl][None, :] - self.cell_east[:, None]
            d_north = cand_north[sl][None, :] - self.cell_north[:, None]
            rng = torch.sqrt(d_east * d_east + d_north * d_north)
            wb = torch.atan2(d_east, d_north)  # compass CW world bearing
            safe_rng = torch.clamp(rng, min=1.0)
            quant_var = 0.0
            if quantization_comp:
                quant_var = (binw * binw / 12.0
                             + (self.grid.cell_m / math.sqrt(12.0)
                                / safe_rng) ** 2)
            k_effs = []
            log_denom = None
            for _, base_var in epochs:
                var = base_var + (sigma_pos / safe_rng) ** 2 + quant_var
                k_eff = 1.0 / var
                k_effs.append(k_eff)
                term = torch.log(torch.special.i0e(k_eff))
                log_denom = term if log_denom is None else log_denom + term
            cos_wb = torch.cos(wb)
            sin_wb = torch.sin(wb)
            weight = cand_weight[sl][None, :] * (1.0 - pi0)
            for b in range(n_heading):
                exponent = -log_denom
                for (z_rad, _), k_eff in zip(epochs, k_effs):
                    angle = self.bin_rad[b] + z_rad
                    cos_delta = (cos_wb * math.cos(angle)
                                 + sin_wb * math.sin(angle))
                    exponent = exponent + k_eff * (cos_delta - 1.0)
                like[b] += (torch.exp(exponent) * weight).sum(dim=1)
        return like.view(n_heading, self.grid.n_north, self.grid.n_east)

    def renormalize(self):
        total = self.belief.sum()
        self.leaked_mass = 1.0 - float(total)
        self.belief /= total

    def position_marginal(self) -> torch.Tensor:
        return self.belief.sum(dim=0).reshape(-1)


def truth_masks(grid: Grid, truth, radii, subgrid: int = 8):
    """Per (keyframe, radius): flat cell indices + within-radius area
    fraction, from a subgrid so radii below the cell size stay meaningful."""
    east, north = grid.centers()
    sub = (np.arange(subgrid) + 0.5) / subgrid - 0.5
    sub_e, sub_n = np.meshgrid(sub * grid.cell_m, sub * grid.cell_m)
    masks = {}
    for pose in truth:
        for radius in radii:
            reach = radius + 0.75 * grid.cell_m
            j_sel = np.nonzero(np.abs(east - pose.east_m) <= reach)[0]
            i_sel = np.nonzero(np.abs(north - pose.north_m) <= reach)[0]
            idx, frac = [], []
            for i in i_sel:
                for j in j_sel:
                    de = east[j] + sub_e - pose.east_m
                    dn = north[i] + sub_n - pose.north_m
                    inside = float(
                        (de * de + dn * dn <= radius * radius).mean())
                    if inside > 0.0:
                        idx.append(i * grid.n_east + j)
                        frac.append(inside)
            masks[(pose.keyframe_idx, radius)] = (
                np.asarray(idx, dtype=np.int64),
                np.asarray(frac, dtype=np.float32))
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--like_run", default=None,
                        help="run dir whose run_manifest.json supplies the "
                             "init box, pi0, matcher_recall, heading rw")
    parser.add_argument("--cell_m", type=float, default=200.0)
    parser.add_argument("--n_heading", type=int, default=18)
    parser.add_argument("--yaw_sigma_scale", type=float, default=1.0,
                        help="scale on the export's per-step sigma_yaw_rad "
                             "(it may overstate the true course noise)")
    parser.add_argument("--heading_rw_deg", type=float, default=1.0)
    parser.add_argument("--diffusion_m", type=float, default=5.0,
                        help="extra per-keyframe position diffusion")
    parser.add_argument("--pi0", type=float, default=None,
                        help="default: like_run manifest value, else 0.2")
    parser.add_argument("--matcher_recall", type=float, default=None)
    parser.add_argument("--init_truth_sigma_m", type=float, default=0.0,
                        help="diagnostic only: Gaussian init at truth with "
                             "this sigma (0 = uniform evaluation init)")
    parser.add_argument("--kappa_scale", type=float, default=1.0,
                        help="scale on each measurement's kappa; the export "
                             "kappas (~sigma 2 deg) are overconfident vs "
                             "truth-course residuals (~14 deg median)")
    parser.add_argument("--max_track_epochs", type=int, default=8,
                        help="track_joint window: joint over at most this "
                             "many trailing epochs (bounds compute and the "
                             "yaw-compensation span)")
    parser.add_argument("--track_joint", type=int, default=0,
                        help="1: marginalize each track's identity once "
                             "across its epochs (exact analog of PF "
                             "association persistence); 0: independent-"
                             "epoch mixture")
    parser.add_argument("--quantization_comp", type=int, default=1,
                        help="1: marginalize within-cell/within-bin pose "
                             "(inflate bearing variance); 0: evaluate at "
                             "cell/bin centers")
    parser.add_argument("--tail", choices=("exact", "uniform"),
                        default="exact",
                        help="exact: all catalog candidates; uniform: only "
                             "matcher-endorsed candidates, unendorsed mass "
                             "folded into a uniform-bearing floor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="summary JSON path")
    args = parser.parse_args()

    data = export_ingest.load(Path(args.input_dir))
    catalog = data.catalog
    sigma_pos = float(catalog.position_sigma_m[0])

    if args.like_run:
        manifest = json.loads(
            (Path(args.like_run) / "run_manifest.json").read_text())
        fc = manifest["filter_config"]
        init = fc["init"]
        assert init["kind"] == "UniformBoxInit", init["kind"]
        box = (init["east_min_m"], init["east_max_m"],
               init["north_min_m"], init["north_max_m"])
        if args.pi0 is None:
            args.pi0 = fc["pi0"]
        if args.matcher_recall is None:
            args.matcher_recall = fc["matcher_recall"]
    else:
        region = export_ingest.region_box(data, 0.0)
        box = (region.east_min_m, region.east_max_m,
               region.north_min_m, region.north_max_m)
    if args.pi0 is None:
        args.pi0 = 0.2
    if args.matcher_recall is None:
        args.matcher_recall = 0.5

    grid = Grid(*box, args.cell_m)
    belief = GridBelief(grid, args.n_heading, args.device)
    if args.init_truth_sigma_m > 0.0:
        pose0 = data.truth[0]
        d2 = ((belief.cell_east - pose0.east_m) ** 2
              + (belief.cell_north - pose0.north_m) ** 2)
        gauss = torch.exp(-0.5 * d2 / args.init_truth_sigma_m ** 2)
        belief.belief = (gauss / gauss.sum() / args.n_heading).expand(
            args.n_heading, -1).reshape(belief.belief.shape).clone()
    n_states = args.n_heading * grid.n_north * grid.n_east
    print(f"grid: {grid.n_east} x {grid.n_north} x {args.n_heading} = "
          f"{n_states / 1e6:.1f}M states, cell {grid.cell_m:g} m; "
          f"pi0={args.pi0} recall={args.matcher_recall} "
          f"yaw_scale={args.yaw_sigma_scale} rw={args.heading_rw_deg} deg "
          f"tail={args.tail}")

    cat_east = torch.tensor(catalog.east_m, dtype=torch.float32,
                            device=args.device)
    cat_north = torch.tensor(catalog.north_m, dtype=torch.float32,
                             device=args.device)

    # Per-table identity posteriors (PF-identical), cached per tracklet.
    weight_cache = {}

    def candidate_set(tracklet_id):
        if tracklet_id not in weight_cache:
            table = data.tables[tracklet_id]
            log_w = filter_lib._identity_log_weights(
                table, catalog, args.matcher_recall)
            weights = np.exp(log_w)
            if args.tail == "exact":
                idx = np.arange(catalog.n)
                tail_mass = 0.0
            else:
                endorsed = ~filter_lib._surprise_mask(
                    table, filter_lib._clipped_log_lr(table, catalog))
                idx = np.nonzero(endorsed)[0]
                tail_mass = float(weights[~endorsed].sum())
            idx_t = torch.as_tensor(idx, device=args.device)
            weight_cache[tracklet_id] = (
                cat_east[idx_t],
                cat_north[idx_t],
                torch.tensor(weights[idx], dtype=torch.float32,
                             device=args.device),
                tail_mass)
        return weight_cache[tracklet_id]

    by_keyframe = {}
    for meas in data.measurements:
        by_keyframe.setdefault(meas.anchor_keyframe_idx, []).append(meas)
    odometry = {item.keyframe_idx: item for item in data.odometry}

    radii = (50.0, 100.0, 250.0, 500.0, 1000.0)
    masks = truth_masks(grid, data.truth, radii)
    truth_by_kf = {pose.keyframe_idx: pose for pose in data.truth}
    east_centers, north_centers = grid.centers()

    heading_rw_rad = math.radians(args.heading_rw_deg)
    series = {radius: [] for radius in radii}
    start_time = time.time()
    cum_yaw = 0.0       # rotates stored epochs into the current frame
    cum_yaw_var = 0.0   # inflates stored epochs by accumulated drift
    track_epochs = {}   # tracklet_id -> [(z_rad, kappa, cum_yaw, cum_var)]
    for keyframe in range(data.n_keyframes):
        if keyframe > 0:
            delta = odometry[keyframe]
            belief.motion(delta, args.yaw_sigma_scale, heading_rw_rad,
                          args.diffusion_m)
            cum_yaw += delta.delta_yaw_cw_rad
            cum_yaw_var += (math.hypot(
                delta.sigma_yaw_rad * args.yaw_sigma_scale,
                heading_rw_rad) ** 2)
        for meas in by_keyframe.get(keyframe, ()):
            c_east, c_north, c_weight, tail_mass = candidate_set(
                meas.tracklet_id)
            kappa_z = min(float(meas.kappa) * args.kappa_scale, MAX_KAPPA)
            z_rad = math.radians(meas.bearing_forward_cw_deg)

            def epoch_view(record):
                z_i, kappa_i, yaw_i, var_i = record
                return (z_i - (cum_yaw - yaw_i),
                        1.0 / kappa_i + (cum_yaw_var - var_i))

            history = track_epochs.setdefault(meas.tracklet_id, []) \
                if args.track_joint else []
            prev = [epoch_view(record)
                    for record in history[-(args.max_track_epochs - 1):]]
            new = prev + [(z_rad, 1.0 / kappa_z)]
            like = belief.track_likelihood(
                new, c_east, c_north, c_weight, sigma_pos,
                args.pi0, tail_mass,
                quantization_comp=bool(args.quantization_comp))
            if prev:
                like = like / belief.track_likelihood(
                    prev, c_east, c_north, c_weight, sigma_pos,
                    args.pi0, tail_mass,
                    quantization_comp=bool(args.quantization_comp))
            belief.belief *= like
            # Renormalize per measurement, not just per keyframe: many
            # measurements in one keyframe otherwise drive the linear
            # float32 belief to zero (pohang went NaN).
            belief.belief /= belief.belief.sum()
            if args.track_joint:
                history.append((z_rad, kappa_z, cum_yaw, cum_yaw_var))
        belief.renormalize()

        marginal = belief.position_marginal()
        for radius in radii:
            idx, frac = masks[(keyframe, radius)]
            if idx.size:
                cells = marginal[torch.tensor(idx, device=args.device)]
                mass = float(
                    (cells * torch.tensor(frac, device=args.device)).sum())
            else:
                mass = 0.0
            series[radius].append(mass)
        if keyframe % 20 == 0 or keyframe == data.n_keyframes - 1:
            best = int(torch.argmax(marginal))
            pose = truth_by_kf[keyframe]
            map_err = math.hypot(
                east_centers[best % grid.n_east] - pose.east_m,
                north_centers[best // grid.n_east] - pose.north_m)
            print(f"kf {keyframe:4d}  mass500 {series[500.0][-1]:.4f}  "
                  f"mass100 {series[100.0][-1]:.4f}  map_err "
                  f"{map_err:8.1f} m  n_meas "
                  f"{len(by_keyframe.get(keyframe, ()))}  "
                  f"({time.time() - start_time:.0f}s)")

    summary = {}
    for radius in radii:
        values = np.asarray(series[radius])
        area = float(np.sum(0.5 * (values[:-1] + values[1:])))
        summary[f"tn_mass_{radius:g}"] = area / (data.n_keyframes - 1)
    print("time-normalized mass:",
          {key: round(value, 4) for key, value in summary.items()},
          f"({time.time() - start_time:.0f}s total)")
    if args.out:
        payload = {
            "config": {key: value for key, value in vars(args).items()},
            "grid": {"n_east": grid.n_east, "n_north": grid.n_north,
                     "n_heading": args.n_heading, "cell_m": grid.cell_m,
                     "box": box},
            "summary": summary,
            "mass_by_keyframe": {f"{radius:g}": series[radius]
                                 for radius in radii},
        }
        Path(args.out).write_text(json.dumps(payload, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
