"""Paper figure: normalized true-patch residual histogram vs the calibrated
HalfNormal(σ) used by the landmark stream of SafaPlusNormalizedLandmarkAggregator.

Single clean density panel (Seattle by default): the empirical distribution of
the per-pair normalized residual ``r_norm = 1 - sim_t / row_max`` for true
(pano, patch) pairs, overlaid with the half-normal PDF whose σ is the per-pair
MLE ``σ = sqrt(mean(r_norm²))``.

Reuses ``calibrate_sigma.compute_samples`` so the numbers match the calibration.
"""

from pathlib import Path
import argparse

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np
import matplotlib.pyplot as plt

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.scripts.calibrate_sigma import (
    compute_samples,
)


def _half_normal_pdf(xs: np.ndarray, sigma: float) -> np.ndarray:
    return (2.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs / sigma) ** 2)


def plot_paper_figure(
    residuals_pair: torch.Tensor,
    sigmas: list[tuple[float, str]],
    output_path: Path,
    *,
    city: str,
) -> None:
    """Plot the residual histogram with one or more HalfNormal(σ) overlays.

    ``sigmas`` is a list of ``(σ, label)``; each is drawn in a different color
    with a matching dashed vertical at ``r = σ``.
    """
    r = residuals_pair.numpy()
    xmax = 1.0  # r_norm is bounded in [0, 1]
    bins = np.linspace(0.0, xmax, 61)
    xs = np.linspace(0.0, xmax, 400)

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.hist(
        r, bins=bins, density=True, color="steelblue", alpha=0.7,
        edgecolor="white", linewidth=0.3,
        label=f"true residuals (n={len(r):,})",
    )
    palette = ["C3", "C2", "C1", "C4", "C5"]
    for i, (sigma, label) in enumerate(sigmas):
        color = palette[i % len(palette)]
        display = label or rf"HalfNormal($\sigma$={sigma:.3f})"
        ax.plot(xs, _half_normal_pdf(xs, sigma), color=color, lw=2.5, label=display)
        ax.axvline(sigma, color=color, ls="--", lw=1.5, alpha=0.6)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"normalized residual $r_{\mathrm{norm}} = 1 - \mathrm{sim}_t / \mathrm{row\,max}$")
    ax.set_ylabel("density")
    ax.set_title(f"{city}: true-patch residuals vs calibrated $\\sigma$")
    ax.legend(loc="upper center", frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--similarity-matrix-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt")
    parser.add_argument(
        "--dataset-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle")
    parser.add_argument("--city", type=str, default="Seattle")
    parser.add_argument("--landmark-version", type=str, default="v4_202001")
    parser.add_argument(
        "--include-semipositive", type=lambda s: s.lower() != "false",
        default=True)
    parser.add_argument(
        "--output-path", type=str,
        default="/data/overhead_matching/evaluation/results/"
                "260501_v6_safa_plus_norm_lm/landmark_residual_exploration/"
                "paper_sigma_fit_seattle")
    parser.add_argument(
        "--sigma", type=str, action="append", default=[],
        help=("HalfNormal σ overlay(s). Repeat to overlay multiple. Each value "
              "is either a bare number (e.g. 0.4673) or number:label "
              "(e.g. 0.737:previous). When omitted, uses the per-pair MLE."))
    args = parser.parse_args()

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

    print("Computing normalized per-pair residuals...")
    rd = compute_samples(
        sim_mat, pano_metadata,
        include_semipositive=args.include_semipositive,
        num_negative_samples_per_pano=0,
        exclude_constant_rows=True,
        residual_form="normalized",
    )
    residuals_pair = rd["residuals_pair"]
    r = residuals_pair.numpy()
    sigma_mle = float(np.sqrt(np.mean(r ** 2)))
    print(f"  pairs: {len(r):,}   σ_MLE (normalized) = {sigma_mle:.6f}   "
          f"median = {np.median(r):.6f}")

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sigma:
        sigmas: list[tuple[float, str]] = []
        for spec in args.sigma:
            if ":" in spec:
                num_str, label = spec.split(":", 1)
                sigma_val = float(num_str)
                sigmas.append((sigma_val, rf"$\sigma$={sigma_val:.3f} ({label})"))
            else:
                sigma_val = float(spec)
                sigmas.append((sigma_val, rf"$\sigma$={sigma_val:.3f}"))
    else:
        sigmas = [(sigma_mle, rf"HalfNormal($\sigma_{{\mathrm{{lm}}}}$={sigma_mle:.3f})")]

    plot_paper_figure(residuals_pair, sigmas, output_path, city=args.city)
    print(f"\nWrote {output_path.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
