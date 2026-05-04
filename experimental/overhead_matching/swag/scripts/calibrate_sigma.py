"""Calibrate the observation-likelihood sigma from a precomputed similarity matrix.

Sigma is used by ``wag_observation_log_likelihood_from_similarity_matrix`` (the
single-matrix aggregator) as the std of a Gaussian on score residuals
``r = sim_max - sim``.  The entropy-adaptive aggregator instead uses sigma as
softmax temperature (``log_softmax(sim / sigma)``).  This script computes
several principled calibrations on a validation similarity matrix and emits
diagnostic plots so a single value can be chosen with the tradeoff visible.

Aggregation is per-pair: each (pano, true_patch) is its own positive sample.

Outputs (under --output-path, prefixed with --name-prefix if given):
- ``sigma_calibration.json`` - computed sigma values + summary stats
- ``residual_histogram.png`` - residual distribution, true vs negative
  (linear + log panels)
- ``similarity_distribution.png`` - raw similarity, true vs negative
  (linear + log panels)
- ``nll_vs_sigma.png`` - per-pair NLL of true patches as a function of sigma
- ``residuals.pt`` - residuals tensor for downstream analysis
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

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
    patch the same likelihood, contributing only a constant offset to the NLL
    while polluting the residual/similarity histograms with degenerate spikes.
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


def build_per_pano_cache(
    similarity_matrix: torch.Tensor,
    panorama_metadata,
    valid_pano_idx: torch.Tensor,
    include_semipositive: bool,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-extract (sims_finite, true_local_idx) per pano for fast scalar-σ NLL.

    Iterating panos and rebuilding these tensors per σ-evaluation dominates the
    cost of a Brent-method scalar minimizer.  Cache them once.
    """
    sim_mat = similarity_matrix.to(device)
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for pano_idx in valid_pano_idx.tolist():
        sim_row = sim_mat[pano_idx]
        finite_mask = torch.isfinite(sim_row)
        sims = sim_row[finite_mask]

        finite_idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
        finite_to_local = -torch.ones(
            sim_row.shape[0], dtype=torch.long, device=device
        )
        finite_to_local[finite_idx] = torch.arange(len(finite_idx), device=device)

        row = panorama_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            true_idxs.extend(row["semipositive_satellite_idxs"])
        true_idxs_t = torch.tensor(true_idxs, dtype=torch.long, device=device)
        local_true = finite_to_local[true_idxs_t]
        local_true = local_true[local_true >= 0]
        if len(local_true) == 0:
            continue
        cache.append((sims, local_true))
    return cache


def scalar_nll_softmax(sigma: float, cache, dtype=torch.float64) -> float:
    """Per-pair NLL of true patches under softmax(sim / σ) at a single σ."""
    total = 0.0
    pair_count = 0
    inv_sigma = 1.0 / sigma
    for sims, local_true in cache:
        logits = sims.to(dtype) * inv_sigma
        log_norm = torch.logsumexp(logits, dim=0)
        log_p_true = logits[local_true] - log_norm
        total += float(-log_p_true.sum().item())
        pair_count += int(local_true.numel())
    return total / max(pair_count, 1)


def scalar_nll_gaussian_residual(sigma: float, cache, dtype=torch.float64) -> float:
    """Per-pair NLL under the Gaussian-on-residuals model at a single σ."""
    total = 0.0
    pair_count = 0
    inv_sigma2 = 1.0 / (sigma * sigma)
    for sims, local_true in cache:
        sims = sims.to(dtype)
        sim_max = sims.max()
        residuals = sim_max - sims
        log_unnorm = -0.5 * residuals * residuals * inv_sigma2
        log_norm = torch.logsumexp(log_unnorm, dim=0)
        log_p_true = log_unnorm[local_true] - log_norm
        total += float(-log_p_true.sum().item())
        pair_count += int(local_true.numel())
    return total / max(pair_count, 1)


def nll_gaussian_residual_form_pair(
    similarity_matrix: torch.Tensor,
    panorama_metadata,
    valid_pano_idx: torch.Tensor,
    sigmas: torch.Tensor,
    include_semipositive: bool,
    device: torch.device,
) -> torch.Tensor:
    """Per-pair NLL under the Gaussian-on-residuals model.

    Each patch's unnormalized log-likelihood is
    ``-0.5 * ((sim_max - sim) / sigma)^2`` (constants cancel under categorical
    normalization).  Returns mean per-(pano,true_patch) NLL of the true patch.
    """
    nll = torch.zeros_like(sigmas)
    pair_count = 0
    sim_mat = similarity_matrix.to(device)
    sigmas_dev = sigmas.to(device)
    inv_sigma2 = 1.0 / (sigmas_dev ** 2)

    for pano_idx in valid_pano_idx.tolist():
        sim_row = sim_mat[pano_idx]
        finite_mask = torch.isfinite(sim_row)
        sims = sim_row[finite_mask]
        sim_max = sims.max()

        row = panorama_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            true_idxs.extend(row["semipositive_satellite_idxs"])
        true_idxs_t = torch.tensor(true_idxs, dtype=torch.long, device=device)
        true_sims = sim_row[true_idxs_t]
        true_finite = true_sims[torch.isfinite(true_sims)]
        if len(true_finite) == 0:
            continue
        true_residuals = sim_max - true_finite  # (T,)

        residuals = sim_max - sims  # (M,)
        sq = residuals ** 2
        log_unnorm = -0.5 * inv_sigma2.unsqueeze(1) * sq.unsqueeze(0)  # (K, M)
        log_norm = torch.logsumexp(log_unnorm, dim=1)  # (K,)
        log_unnorm_true = (
            -0.5 * inv_sigma2.unsqueeze(1) * (true_residuals ** 2).unsqueeze(0)
        )  # (K, T)
        log_p_true = log_unnorm_true - log_norm.unsqueeze(1)  # (K, T)
        nll = nll + (-log_p_true.sum(dim=1).detach().cpu())
        pair_count += int(len(true_finite))

    return nll / max(pair_count, 1)


def nll_softmax_form_pair(
    similarity_matrix: torch.Tensor,
    panorama_metadata,
    valid_pano_idx: torch.Tensor,
    sigmas: torch.Tensor,
    include_semipositive: bool,
    device: torch.device,
) -> torch.Tensor:
    """Per-pair NLL under the EA softmax model ``log_softmax(sim / sigma)``."""
    nll = torch.zeros_like(sigmas)
    pair_count = 0
    sim_mat = similarity_matrix.to(device)
    sigmas_dev = sigmas.to(device)

    for pano_idx in valid_pano_idx.tolist():
        sim_row = sim_mat[pano_idx]
        finite_mask = torch.isfinite(sim_row)
        sims = sim_row[finite_mask]

        finite_idx = torch.nonzero(finite_mask, as_tuple=False).squeeze(1)
        finite_to_local = -torch.ones(
            sim_row.shape[0], dtype=torch.long, device=device
        )
        finite_to_local[finite_idx] = torch.arange(len(finite_idx), device=device)

        row = panorama_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            true_idxs.extend(row["semipositive_satellite_idxs"])
        true_idxs_t = torch.tensor(true_idxs, dtype=torch.long, device=device)
        local_true = finite_to_local[true_idxs_t]
        local_true = local_true[local_true >= 0]
        if len(local_true) == 0:
            continue

        logits = sims.unsqueeze(0) / sigmas_dev.unsqueeze(1)  # (K, M)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        true_log_probs = log_probs[:, local_true]  # (K, T)
        nll = nll + (-true_log_probs.sum(dim=1).detach().cpu())
        pair_count += int(len(local_true))

    return nll / max(pair_count, 1)


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
    bin_w = bins[1] - bins[0]

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
    sigma_nll_gauss_pair: float,
    sigma_nll_soft_pair: float,
    output_path: Path,
) -> None:
    r_pair = residuals_pair.numpy()
    r_neg = residuals_negative.numpy()
    pool = [r_pair, r_neg]
    xmax = float(max(np.quantile(p, 0.999) for p in pool)) * 1.05
    bins = np.linspace(0.0, xmax, 81)

    # HalfNormal: pdf for the density panel; pdf * bin_w * n_pair for counts.
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
        (xs, _half_normal_pdf(sigma_nll_gauss_pair), "C0", "-",
         f"HalfNormal(σ={sigma_nll_gauss_pair:.4f}) [NLL Gaussian]"),
    ]
    fits_count = [
        (xs, _half_normal_pdf(sigma_mle_pair) * bin_w * n_pair,
         "C3", "-", f"HalfNormal(σ={sigma_mle_pair:.4f}) [MLE per-pair]"),
        (xs, _half_normal_pdf(sigma_nll_gauss_pair) * bin_w * n_pair,
         "C0", "-", f"HalfNormal(σ={sigma_nll_gauss_pair:.4f}) [NLL Gaussian]"),
    ]
    vlines = [
        (sigma_mle_pair, "C3", "--", None),
        (sigma_nll_gauss_pair, "C0", "--", None),
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
    fig.suptitle(
        f"σ candidates: MLE={sigma_mle_pair:.4f}  "
        f"NLL_gauss={sigma_nll_gauss_pair:.4f}  "
        f"NLL_softmax={sigma_nll_soft_pair:.4f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_similarity_distribution(
    sims_pair: torch.Tensor,
    sims_negative: torch.Tensor,
    sigma_softmax_pair: float,
    output_path: Path,
) -> None:
    s_pair = sims_pair.numpy()
    s_neg = sims_negative.numpy()
    lo = float(min(s_pair.min(), s_neg.min()))
    hi = float(max(np.quantile(s_pair, 0.999), np.quantile(s_neg, 0.999)))
    pad = 0.02 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 81)
    gap_in_sigma = (s_pair.mean() - s_neg.mean()) / sigma_softmax_pair

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
        f"σ_softmax (NLL per-pair) = {sigma_softmax_pair:.4f}    "
        f"(true−neg mean) = {s_pair.mean()-s_neg.mean():.3f} = "
        f"{gap_in_sigma:.2f} σ"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_nll_vs_sigma(
    sigmas: torch.Tensor,
    nll_gauss_pair: torch.Tensor,
    nll_soft_pair: torch.Tensor,
    sigma_mle_pair: float,
    output_path: Path,
) -> None:
    s = sigmas.numpy()
    g = nll_gauss_pair.numpy()
    sm = nll_soft_pair.numpy()
    g_best = int(np.argmin(g))
    s_best = int(np.argmin(sm))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    ax = axes[0]
    ax.plot(s, g, "C0-", lw=2,
            label=f"NLL [σ*={s[g_best]:.4f}]")
    ax.axvline(s[g_best], color="C0", ls=":", alpha=0.7)
    ax.axvline(sigma_mle_pair, color="C3", ls="--", alpha=0.6,
               label=f"σ_MLE={sigma_mle_pair:.4f}")
    ax.set_xlabel("σ"); ax.set_ylabel("mean per-pair NLL")
    ax.set_title("Single-matrix (Gaussian-on-residuals)")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(s, sm, "C2-", lw=2,
            label=f"NLL [σ*={s[s_best]:.4f}]")
    ax.axvline(s[s_best], color="C2", ls=":", alpha=0.7)
    ax.set_xlabel("σ"); ax.set_ylabel("mean per-pair NLL")
    ax.set_title("Entropy-adaptive softmax")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Per-pair NLL vs σ (lower is better)")
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
    parser.add_argument("--sigma-min", type=float, default=0.005)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--sigma-num", type=int, default=80)
    parser.add_argument(
        "--include-semipositive", type=lambda s: s.lower() != "false",
        default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
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

    device = torch.device(args.device)

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

    sigmas = torch.tensor(
        np.logspace(np.log10(args.sigma_min), np.log10(args.sigma_max),
                    args.sigma_num),
        dtype=torch.float64,
    )

    print(f"Sweeping σ over {len(sigmas)} log-spaced values "
          f"in [{args.sigma_min}, {args.sigma_max}]...")
    print("  computing Gaussian-on-residuals per-pair NLL...")
    nll_g_pair = nll_gaussian_residual_form_pair(
        sim_mat.double(), pano_metadata, valid_pano_idx, sigmas,
        include_semipositive=args.include_semipositive, device=device,
    )
    print("  computing softmax per-pair NLL...")
    nll_s_pair = nll_softmax_form_pair(
        sim_mat.double(), pano_metadata, valid_pano_idx, sigmas,
        include_semipositive=args.include_semipositive, device=device,
    )

    g_best = int(torch.argmin(nll_g_pair))
    s_best = int(torch.argmin(nll_s_pair))
    sigma_nll_gauss_grid = float(sigmas[g_best])
    sigma_nll_soft_grid = float(sigmas[s_best])
    print(f"  [grid] σ min Gaussian-form NLL = {sigma_nll_gauss_grid:.6f} "
          f"(NLL={float(nll_g_pair[g_best]):.4f})")
    print(f"  [grid] σ min softmax-form NLL  = {sigma_nll_soft_grid:.6f} "
          f"(NLL={float(nll_s_pair[s_best]):.4f})")

    print("Refining via Brent's method (scipy.optimize.minimize_scalar)...")
    cache = build_per_pano_cache(
        sim_mat, pano_metadata, valid_pano_idx,
        include_semipositive=args.include_semipositive, device=device,
    )

    def _nll_g(sig):
        return scalar_nll_gaussian_residual(float(sig), cache)

    def _nll_s(sig):
        return scalar_nll_softmax(float(sig), cache)

    bracket = (args.sigma_min, args.sigma_max)
    res_g = minimize_scalar(
        _nll_g, bounds=bracket, method="bounded",
        options={"xatol": 1e-5, "maxiter": 200},
    )
    res_s = minimize_scalar(
        _nll_s, bounds=bracket, method="bounded",
        options={"xatol": 1e-5, "maxiter": 200},
    )
    sigma_nll_gauss = float(res_g.x)
    sigma_nll_soft = float(res_s.x)
    nll_gauss_min = float(res_g.fun)
    nll_soft_min = float(res_s.fun)
    print(f"  [brent] σ min Gaussian-form NLL = {sigma_nll_gauss:.6f} "
          f"(NLL={nll_gauss_min:.4f}, evals={res_g.nfev}, "
          f"converged={res_g.success})")
    print(f"  [brent] σ min softmax-form NLL  = {sigma_nll_soft:.6f} "
          f"(NLL={nll_soft_min:.4f}, evals={res_s.nfev}, "
          f"converged={res_s.success})")

    print("Writing plots and JSON summary...")
    plot_residual_histogram(
        residuals_pair, residuals_neg,
        sigma_mle_pair, sigma_nll_gauss, sigma_nll_soft,
        output_path / f"{prefix}residual_histogram.png",
    )
    plot_similarity_distribution(
        sims_pair, sims_neg, sigma_nll_soft,
        output_path / f"{prefix}similarity_distribution.png",
    )
    plot_nll_vs_sigma(
        sigmas, nll_g_pair, nll_s_pair, sigma_mle_pair,
        output_path / f"{prefix}nll_vs_sigma.png",
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
        "sigma_nll_gaussian_form_pair": sigma_nll_gauss,
        "sigma_nll_softmax_form_pair": sigma_nll_soft,
        "min_nll_gaussian_form_pair": float(nll_g_pair[g_best]),
        "min_nll_softmax_form_pair": float(nll_s_pair[s_best]),
        "sigma_sweep": sigmas.tolist(),
        "nll_gaussian_form_pair_sweep": nll_g_pair.tolist(),
        "nll_softmax_form_pair_sweep": nll_s_pair.tolist(),
    }
    with open(output_path / f"{prefix}sigma_calibration.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Outputs in {output_path} (prefix={prefix or '<none>'})")
    print(f"  σ_MLE per-pair = {sigma_mle_pair:.4f}")
    print(f"  σ_NLL_gaussian = {sigma_nll_gauss:.4f}")
    print(f"  σ_NLL_softmax  = {sigma_nll_soft:.4f}")


if __name__ == "__main__":
    main()
