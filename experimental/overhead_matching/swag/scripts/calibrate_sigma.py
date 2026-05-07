"""Calibrate σ_img for the SAFA Gaussian-on-residuals image-stream aggregator.

σ is used by ``wag_observation_log_likelihood_from_similarity_matrix`` (the
single-matrix aggregator) as the std of a Gaussian on score residuals
``r = sim_max - sim``. We compute the half-normal MLE on per-pair residuals
from a calibration similarity matrix.

Aggregation is per-pair: each (pano, true_patch) is its own positive sample.

Outputs (under --output-path, prefixed with --name-prefix if given):
- ``sigma_calibration.json`` - σ_MLE + summary stats
- ``residual_histogram.png`` - residual distribution, true vs negative
  (linear + log panels)
- ``similarity_distribution.png`` - raw similarity, true vs negative
  (linear + log panels)
- ``samples.pt`` - per-pair residuals + raw similarities (true & negative)
  for downstream analysis
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np
import matplotlib.pyplot as plt

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)


def compute_samples(
    similarity_matrix: torch.Tensor,
    panorama_metadata,
    include_semipositive: bool = True,
    num_negative_samples_per_pano: int = 0,
    rng_seed: int = 0,
    exclude_constant_rows: bool = True,
):
    """Collect per-pair residuals and raw similarities for true and negative pairs.

    Rows where every finite similarity is identical (e.g. the all-zero rows in
    the landmark Hungarian matrix) are uninformative — they would assign every
    patch the same likelihood, contributing only a constant offset while
    polluting the residual/similarity histograms with degenerate spikes.
    They are excluded by default.

    Returns a dict with keys:
        residuals_pair, residuals_negative, sims_pair, sims_negative,
        valid_pano_idx, num_excluded_constant_rows, num_excluded_no_truepatch.
    """
    residuals_pair = []
    residuals_negative = []
    sims_pair = []
    sims_negative = []
    valid_indices = []
    n_constant = 0
    n_no_truepatch = 0

    rng = np.random.default_rng(rng_seed)
    num_patches = similarity_matrix.shape[1]

    for pano_idx in range(similarity_matrix.shape[0]):
        sim_row = similarity_matrix[pano_idx]
        finite_mask = torch.isfinite(sim_row)
        if not finite_mask.any():
            n_constant += 1
            continue
        finite_sims = sim_row[finite_mask]
        sim_max = finite_sims.max().item()
        sim_min = finite_sims.min().item()
        if exclude_constant_rows and sim_max == sim_min:
            n_constant += 1
            continue

        row = panorama_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            true_idxs.extend(row["semipositive_satellite_idxs"])
        if not true_idxs:
            n_no_truepatch += 1
            continue

        true_sims = sim_row[true_idxs]
        true_finite = true_sims[torch.isfinite(true_sims)]
        if len(true_finite) == 0:
            n_no_truepatch += 1
            continue

        for ts in true_finite.tolist():
            sims_pair.append(ts)
            residuals_pair.append(sim_max - ts)
        valid_indices.append(pano_idx)

        if num_negative_samples_per_pano > 0:
            true_set = set(true_idxs)
            k = num_negative_samples_per_pano
            candidates = rng.integers(0, num_patches, size=k * 2)
            candidates = [int(c) for c in candidates if int(c) not in true_set]
            candidates = candidates[:k]
            cand_t = torch.tensor(candidates, dtype=torch.long)
            neg_sims = sim_row[cand_t]
            neg_finite = neg_sims[torch.isfinite(neg_sims)]
            for ns in neg_finite.tolist():
                sims_negative.append(ns)
                residuals_negative.append(sim_max - ns)

    def _t(xs):
        return torch.tensor(xs, dtype=torch.float64)

    return {
        "residuals_pair": _t(residuals_pair),
        "residuals_negative": _t(residuals_negative),
        "sims_pair": _t(sims_pair),
        "sims_negative": _t(sims_negative),
        "valid_pano_idx": torch.tensor(valid_indices, dtype=torch.long),
        "num_excluded_constant_rows": n_constant,
        "num_excluded_no_truepatch": n_no_truepatch,
    }


def _draw_two_pop_histogram(
    ax_density, ax_log, true_vals, neg_vals, *, bins, xmax, xmin, vlines,
    xlabel, title, fits_density=None, fits_count=None,
):
    """Two-panel histogram for two populations.

    Left panel: normalized density (each population integrates to 1) — exposes
    distribution shape independent of sample size.
    Right panel: raw counts on log-y — exposes abundance / sample-size gap.

    `fits_density` / `fits_count` are lists of ``(xs, ys, color, ls, label)``
    overlays drawn on the density / count panel respectively.
    """
    # Density panel
    ax_density.hist(true_vals, bins=bins, density=True, color="steelblue",
                    alpha=0.65, label=f"true (n={len(true_vals)})")
    ax_density.hist(neg_vals, bins=bins, density=True, histtype="step",
                    color="darkorange", lw=1.5,
                    label=f"negative (n={len(neg_vals)})")
    if fits_density is not None:
        for xs, ys, color, ls, label in fits_density:
            ax_density.plot(xs, ys, color=color, ls=ls, lw=2, label=label)
    for x, color, ls, label in vlines:
        ax_density.axvline(x, color=color, ls=ls, alpha=0.7, label=label)
    ax_density.set_xlim(xmin, xmax)
    ax_density.set_xlabel(xlabel)
    ax_density.set_ylabel("density")
    ax_density.set_title(f"{title} (density)")
    ax_density.legend(loc="upper right", fontsize=7)

    # Log-count panel
    ax_log.hist(true_vals, bins=bins, color="steelblue", alpha=0.65,
                label=f"true (n={len(true_vals)})")
    ax_log.hist(neg_vals, bins=bins, histtype="step", color="darkorange",
                lw=1.5, label=f"negative (n={len(neg_vals)})")
    if fits_count is not None:
        for xs, ys, color, ls, label in fits_count:
            ax_log.plot(xs, ys, color=color, ls=ls, lw=2, label=label)
    for x, color, ls, label in vlines:
        ax_log.axvline(x, color=color, ls=ls, alpha=0.7, label=label)
    ax_log.set_xlim(xmin, xmax)
    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel("count (log)")
    ax_log.set_yscale("log")
    ax_log.set_ylim(bottom=0.5)
    ax_log.set_title(f"{title} (log count)")
    ax_log.legend(loc="upper right", fontsize=7)


def plot_residual_histogram(
    residuals_pair: torch.Tensor,
    residuals_negative: torch.Tensor,
    sigma_mle_pair: float,
    output_path: Path,
) -> None:
    r_pair = residuals_pair.numpy()
    r_neg = residuals_negative.numpy()
    pool = [r_pair, r_neg]
    xmax = float(max(np.quantile(p, 0.999) for p in pool)) * 1.05
    bins = np.linspace(0.0, xmax, 81)

    xs = np.linspace(0.0, xmax, 400)
    bin_w = bins[1] - bins[0]
    n_pair = len(r_pair)

    def _half_normal_pdf(sigma):
        return (2.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * (xs / sigma) ** 2
        )

    fits_density = [
        (xs, _half_normal_pdf(sigma_mle_pair), "C3", "-",
         f"HalfNormal(σ={sigma_mle_pair:.4f}) [MLE per-pair]"),
    ]
    fits_count = [
        (xs, _half_normal_pdf(sigma_mle_pair) * bin_w * n_pair,
         "C3", "-", f"HalfNormal(σ={sigma_mle_pair:.4f}) [MLE per-pair]"),
    ]
    vlines = [
        (sigma_mle_pair, "C3", "--", None),
        (np.median(r_pair), "k", ":", f"true median={np.median(r_pair):.4f}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _draw_two_pop_histogram(
        axes[0], axes[1], r_pair, r_neg,
        bins=bins, xmin=0.0, xmax=xmax, vlines=vlines,
        fits_density=fits_density, fits_count=fits_count,
        xlabel="residual = sim_max - sim_t",
        title="Per-pair residuals (true vs negative)",
    )
    fig.suptitle(f"σ_MLE per-pair = {sigma_mle_pair:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_similarity_distribution(
    sims_pair: torch.Tensor,
    sims_negative: torch.Tensor,
    output_path: Path,
) -> None:
    s_pair = sims_pair.numpy()
    s_neg = sims_negative.numpy()
    lo = float(min(s_pair.min(), s_neg.min()))
    hi = float(max(np.quantile(s_pair, 0.999), np.quantile(s_neg, 0.999)))
    pad = 0.02 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 81)

    vlines = [
        (s_pair.mean(), "steelblue", "--", f"true mean={s_pair.mean():.3f}"),
        (s_neg.mean(), "darkorange", "--", f"neg mean={s_neg.mean():.3f}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _draw_two_pop_histogram(
        axes[0], axes[1], s_pair, s_neg,
        bins=bins, xmin=lo - pad, xmax=hi + pad, vlines=vlines,
        xlabel="raw similarity sim_t",
        title="Per-pair similarity (true vs negative)",
    )
    fig.suptitle(
        f"(true−neg mean) = {s_pair.mean()-s_neg.mean():.3f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--similarity-matrix-path",
        type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/safa_dinov3_wag_chicago.pt")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle")
    parser.add_argument("--landmark-version", type=str, default="v4_202001")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--include-semipositive", type=lambda s: s.lower() != "false",
        default=True)
    parser.add_argument(
        "--num-negative-samples-per-pano", type=int, default=200,
        help="Per-pano random negative-patch samples to overlay on plots. "
             "0 disables the overlay.")
    parser.add_argument("--negative-sample-seed", type=int, default=0)
    parser.add_argument(
        "--name-prefix", type=str, default="",
        help="Filename prefix for outputs (e.g. 'image' or 'landmark') so "
             "multiple matrices can write into the same output directory")
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = (args.name_prefix + "_") if args.name_prefix else ""
    with open(output_path / f"{prefix}args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading similarity matrix from {args.similarity_matrix_path}")
    sim_mat = _load_similarity_matrix(Path(args.similarity_matrix_path))
    print(f"  shape: {tuple(sim_mat.shape)}, dtype: {sim_mat.dtype}")

    print(f"Loading dataset metadata from {args.dataset_path}")
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=args.landmark_version,
    )
    vigor_dataset = vd.VigorDataset(Path(args.dataset_path), dataset_config)
    pano_metadata = vigor_dataset._panorama_metadata
    assert sim_mat.shape[0] == len(pano_metadata)
    assert sim_mat.shape[1] == len(vigor_dataset._satellite_metadata)

    print("Computing per-pair residuals and similarity samples...")
    rd = compute_samples(
        sim_mat, pano_metadata,
        include_semipositive=args.include_semipositive,
        num_negative_samples_per_pano=args.num_negative_samples_per_pano,
        rng_seed=args.negative_sample_seed,
        exclude_constant_rows=True,
    )
    residuals_pair = rd["residuals_pair"]
    residuals_neg = rd["residuals_negative"]
    sims_pair = rd["sims_pair"]
    sims_neg = rd["sims_negative"]
    valid_pano_idx = rd["valid_pano_idx"]
    print(f"  excluded {rd['num_excluded_constant_rows']} constant/all-equal rows")
    print(f"  excluded {rd['num_excluded_no_truepatch']} rows with no finite true patch")
    print(f"  contributing panos: {len(valid_pano_idx)} / {sim_mat.shape[0]}")
    print(f"  positive (pano,true) pairs: {len(residuals_pair)}")
    if len(residuals_neg) > 0:
        print(f"  sampled negative pairs: {len(residuals_neg)}")

    r_pair = residuals_pair.numpy()
    sigma_mle_pair = float(np.sqrt(np.mean(r_pair ** 2)))
    sigma_median_pair = float(np.median(r_pair))
    print(f"  σ_MLE per-pair = {sigma_mle_pair:.6f}  "
          f"(median residual = {sigma_median_pair:.6f})")

    print("Writing plots and JSON summary...")
    plot_residual_histogram(
        residuals_pair, residuals_neg,
        sigma_mle_pair,
        output_path / f"{prefix}residual_histogram.png",
    )
    plot_similarity_distribution(
        sims_pair, sims_neg,
        output_path / f"{prefix}similarity_distribution.png",
    )
    torch.save({
        "residuals_pair": residuals_pair,
        "residuals_negative": residuals_neg,
        "sims_pair": sims_pair,
        "sims_negative": sims_neg,
    }, output_path / f"{prefix}samples.pt")

    summary = {
        "similarity_matrix_path": str(args.similarity_matrix_path),
        "dataset_path": str(args.dataset_path),
        "include_semipositive": args.include_semipositive,
        "num_panoramas_total": int(sim_mat.shape[0]),
        "num_pano_truepatch_pairs": int(len(residuals_pair)),
        "num_negative_samples": int(len(residuals_neg)),
        "num_excluded_constant_rows": int(rd["num_excluded_constant_rows"]),
        "num_excluded_no_truepatch": int(rd["num_excluded_no_truepatch"]),
        "num_panoramas_contributing": int(len(valid_pano_idx)),
        "sigma_mle_per_pair": sigma_mle_pair,
        "sigma_median_per_pair": sigma_median_pair,
    }
    with open(output_path / f"{prefix}sigma_calibration.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Outputs in {output_path} (prefix={prefix or '<none>'})")
    print(f"  σ_MLE per-pair = {sigma_mle_pair:.4f}")


if __name__ == "__main__":
    main()
