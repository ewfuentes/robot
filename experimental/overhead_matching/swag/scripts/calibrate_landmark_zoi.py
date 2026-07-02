"""Calibrate the zero/one-inflated likelihood-ratio landmark observation model.

The landmark stream of ``SafaPlusZeroOneInflatedLandmarkAggregator`` maps the
per-row normalized residual ``r = 1 - sim/row_max`` through a frozen log-LR
lookup:  log p(r|true) - log p(r|negative).  We estimate p(r|true) and
p(r|negative) from a calibration similarity matrix (true (pano,patch) pairs vs
randomly sampled negative patches), decomposing each population into

  * the discrete event r == 0  (true patch is the row argmax),
  * the discrete event r == 1  (landmark missed the patch),
  * a binned continuous interior over (0, 1).

Output JSON contains the calibration fields consumed directly by
``SafaPlusZeroOneInflatedLandmarkAggregatorConfig`` plus diagnostics
(per-regime masses, discrimination score). Calibrate once on a reference city
and freeze across cities (matching the single-sigma_lm methodology).
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.scripts.calibrate_sigma import (
    compute_samples,
)

EPS = 1e-6


def build_calibration(
    r_true: np.ndarray,
    r_neg: np.ndarray,
    n_interior_bins: int,
    log_lr_clip: float,
):
    """Build zero/one-inflated log-LR calibration from true/negative residuals.

    Returns (calibration_dict, diagnostics_dict). Smoothing: a Laplace
    pseudo-count is added to every regime/bin of both populations so empty
    cells give finite, bounded log-LRs; values are clipped to +/- log_lr_clip.
    """
    edges = np.linspace(0.0, 1.0, n_interior_bins + 1)

    def _counts(r):
        at0 = int(np.sum(r <= EPS))
        at1 = int(np.sum(r >= 1.0 - EPS))
        interior = r[(r > EPS) & (r < 1.0 - EPS)]
        hist, _ = np.histogram(interior, bins=edges)
        return at0, at1, hist

    t0, t1, t_hist = _counts(r_true)
    n0, n1, n_hist = _counts(r_neg)

    # One smoothing cell per regime (2 endpoints + n_interior_bins).
    n_cells = 2 + n_interior_bins
    nt, nn = len(r_true), len(r_neg)

    def _logp(count, total):
        return np.log((count + 1.0) / (total + n_cells))

    # Endpoint events: compare probability mass directly.
    lr_zero = float(np.clip(_logp(t0, nt) - _logp(n0, nn), -log_lr_clip, log_lr_clip))
    lr_one = float(np.clip(_logp(t1, nt) - _logp(n1, nn), -log_lr_clip, log_lr_clip))

    # Interior: compare *densities* (per-bin mass / bin width) so the lookup is
    # a proper density ratio; bin width is constant here but keep it explicit.
    widths = np.diff(edges)
    lr_interior = []
    for k in range(n_interior_bins):
        lp_t = _logp(int(t_hist[k]), nt) - np.log(widths[k])
        lp_n = _logp(int(n_hist[k]), nn) - np.log(widths[k])
        lr_interior.append(float(np.clip(lp_t - lp_n, -log_lr_clip, log_lr_clip)))

    calibration = {
        "landmark_log_lr_zero": lr_zero,
        "landmark_log_lr_one": lr_one,
        "landmark_interior_edges": [float(e) for e in edges],
        "landmark_interior_log_lr": lr_interior,
    }
    diagnostics = {
        "n_true": int(nt),
        "n_negative": int(nn),
        "true_mass_zero": t0 / nt,
        "true_mass_one": t1 / nt,
        "true_mass_interior": 1.0 - (t0 + t1) / nt,
        "neg_mass_zero": n0 / nn,
        "neg_mass_one": n1 / nn,
        "neg_mass_interior": 1.0 - (n0 + n1) / nn,
        "n_interior_bins": n_interior_bins,
        "log_lr_clip": log_lr_clip,
    }
    return calibration, diagnostics


def _discrimination(calibration, r_true, r_neg):
    """D = E_true[log LR] - E_neg[log LR]: mean belief-update gap (bigger=better)."""
    edges = np.asarray(calibration["landmark_interior_edges"])
    inner_lr = np.asarray(calibration["landmark_interior_log_lr"])

    def log_lr(r):
        idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(inner_lr) - 1)
        out = inner_lr[idx]
        out = np.where(r <= EPS, calibration["landmark_log_lr_zero"], out)
        out = np.where(r >= 1.0 - EPS, calibration["landmark_log_lr_one"], out)
        return out

    return float(log_lr(r_true).mean() - log_lr(r_neg).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--similarity-matrix-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt")
    parser.add_argument(
        "--dataset-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle")
    parser.add_argument("--landmark-version", type=str, default="v4_202001")
    parser.add_argument("--num-negative-samples-per-pano", type=int, default=200)
    parser.add_argument("--negative-sample-seed", type=int, default=0)
    parser.add_argument("--n-interior-bins", type=int, default=20)
    parser.add_argument("--log-lr-clip", type=float, default=6.0)
    parser.add_argument(
        "--include-semipositive", type=lambda s: s.lower() != "false", default=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    sim_mat = _load_similarity_matrix(Path(args.similarity_matrix_path))
    nan_frac = float(torch.isnan(sim_mat).float().mean())
    print(f"matrix {tuple(sim_mat.shape)} dtype={sim_mat.dtype} nan_frac={nan_frac:.4f}")

    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
        should_load_images=False, should_load_landmarks=False,
        landmark_version=args.landmark_version)
    dataset = vd.VigorDataset(Path(args.dataset_path), dataset_config)
    pano_metadata = dataset._panorama_metadata

    rd = compute_samples(
        sim_mat, pano_metadata,
        include_semipositive=args.include_semipositive,
        num_negative_samples_per_pano=args.num_negative_samples_per_pano,
        rng_seed=args.negative_sample_seed,
        exclude_constant_rows=True, residual_form="normalized")
    r_true = rd["residuals_pair"].numpy()
    r_neg = rd["residuals_negative"].numpy()

    calibration, diagnostics = build_calibration(
        r_true, r_neg, args.n_interior_bins, args.log_lr_clip)
    sigma_mle = float(np.sqrt(np.mean(r_true ** 2)))
    D = _discrimination(calibration, r_true, r_neg)

    print("\nregime masses (true / negative):")
    print(f"  r==0 : {diagnostics['true_mass_zero']:.4f} / {diagnostics['neg_mass_zero']:.4f}"
          f"   -> log_lr_zero = {calibration['landmark_log_lr_zero']:+.3f}")
    print(f"  r==1 : {diagnostics['true_mass_one']:.4f} / {diagnostics['neg_mass_one']:.4f}"
          f"   -> log_lr_one  = {calibration['landmark_log_lr_one']:+.3f}")
    print(f"  interior: {diagnostics['true_mass_interior']:.4f} / "
          f"{diagnostics['neg_mass_interior']:.4f}")
    print(f"\nsigma_mle (gaussian baseline) = {sigma_mle:.4f}")
    print(f"discrimination D (this calibration) = {D:.3f}")

    out = {
        "similarity_matrix_path": str(args.similarity_matrix_path),
        "dataset_path": str(args.dataset_path),
        "calibration": calibration,
        "diagnostics": {**diagnostics, "sigma_mle_gaussian": sigma_mle,
                        "discrimination_D": D},
    }
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote calibration to {output_path}")


if __name__ == "__main__":
    main()
