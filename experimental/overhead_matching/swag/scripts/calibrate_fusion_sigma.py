"""Joint per-pair NLL calibration of (σ_img, σ_lm) under EA fused softmax.

Sweeps a 2D grid of σ values for the image and landmark streams, computes the
mean per-pair NLL of true patches under the entropy-adaptive fused posterior
(matching :class:`EntropyAdaptiveAggregator`), and emits a heatmap + summary.

This complements ``calibrate_sigma.py``: that script calibrates each stream
independently; this one tells you whether independent calibration sits at the
joint NLL minimum or whether the streams should be co-tuned.

Outputs:
- ``fusion_nll_heatmap.png`` - joint NLL surface with marked optimum and
  marginal per-stream optima for comparison
- ``fusion_calibration.json`` - σ grids, full NLL surface, summary statistics
"""

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np
import matplotlib.pyplot as plt

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)


def _peak_sharpness(log_p: torch.Tensor) -> torch.Tensor:
    """``log_p.max(-1) - log_p.mean(-1)`` over the last axis, ignoring -inf.

    Matches ``EntropyAdaptiveAggregator._compute_confidence``.
    """
    finite = torch.isfinite(log_p)
    safe_for_sum = torch.where(finite, log_p, torch.zeros_like(log_p))
    count = finite.sum(dim=-1).clamp(min=1).to(log_p.dtype)
    mean = safe_for_sum.sum(dim=-1) / count
    safe_for_max = torch.where(
        finite, log_p, torch.full_like(log_p, float("-inf"))
    )
    maxv = safe_for_max.max(dim=-1).values
    return maxv - mean


def _raw_peak_sharpness_1d(sim: torch.Tensor) -> torch.Tensor:
    """``max(sim) - mean(sim)`` over a 1-D tensor, clamped at 0, ignoring -inf."""
    finite = sim[torch.isfinite(sim)]
    if finite.numel() < 2:
        return torch.tensor(0.0, dtype=sim.dtype, device=sim.device)
    return (finite.max() - finite.mean()).clamp(min=0.0)


def _raw_max_1d(sim: torch.Tensor) -> torch.Tensor:
    """``max(sim, 0)`` over the finite entries; 0 when no finite entries."""
    finite = sim[torch.isfinite(sim)]
    if finite.numel() == 0:
        return torch.tensor(0.0, dtype=sim.dtype, device=sim.device)
    return finite.max().clamp(min=0.0)


def fused_nll_grid(
    img_sim_mat: torch.Tensor,
    lm_sim_mat: torch.Tensor,
    panorama_metadata,
    sigmas_img: torch.Tensor,
    sigmas_lm: torch.Tensor,
    include_semipositive: bool,
    device: torch.device,
    exclude_constant_rows: bool = True,
    landmark_pano_confidence_threshold: float | None = None,
    landmark_pair_sim_threshold: float | None = None,
    alpha_mode: str = "peak_sharpness",
):
    """Compute per-pair NLL on a 2D (σ_img, σ_lm) grid under the EA fusion.

    For each pano, build ``log_softmax(sim_img / σ_img)`` and
    ``log_softmax(sim_lm / σ_lm)`` for every grid σ, blend them with the EA
    confidence-weighted ``α``, score the true patches, accumulate.

    Optional landmark-confidence filters (mutually exclusive in spirit but can
    be combined):
      * ``landmark_pano_confidence_threshold``: drop panos whose landmark
        ``row_max`` is at or below this value (per-pano filter — Option 1).
      * ``landmark_pair_sim_threshold``: drop (pano, true_patch) pairs whose
        landmark sim_t is at or below this value (per-pair filter — Option 3).

    Returns the NLL surface (K_img × K_lm), the per-stream marginal NLLs, and
    counts.
    """
    K_img = len(sigmas_img)
    K_lm = len(sigmas_lm)
    nll_fused = torch.zeros(K_img, K_lm, dtype=torch.float64)
    nll_img_only = torch.zeros(K_img, dtype=torch.float64)
    nll_lm_only = torch.zeros(K_lm, dtype=torch.float64)
    pair_count = 0
    pano_count = 0
    n_excluded_constant = 0
    n_excluded_no_truepatch = 0
    n_excluded_pano_confidence = 0
    n_excluded_pair_sim = 0

    sigmas_img_dev = sigmas_img.to(device)
    sigmas_lm_dev = sigmas_lm.to(device)
    img_mat = img_sim_mat.to(device)
    lm_mat = lm_sim_mat.to(device)
    eps = 1e-12

    for pano_idx in range(img_mat.shape[0]):
        img_row = img_mat[pano_idx]
        lm_row = lm_mat[pano_idx]
        img_finite = torch.isfinite(img_row)
        lm_finite = torch.isfinite(lm_row)
        if not img_finite.any() or not lm_finite.any():
            n_excluded_constant += 1
            continue
        if exclude_constant_rows:
            i_finite = img_row[img_finite]
            l_finite = lm_row[lm_finite]
            if (i_finite.max() == i_finite.min()) or (
                l_finite.max() == l_finite.min()
            ):
                n_excluded_constant += 1
                continue
        if landmark_pano_confidence_threshold is not None:
            if lm_row[lm_finite].max().item() <= landmark_pano_confidence_threshold:
                n_excluded_pano_confidence += 1
                continue

        row = panorama_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            true_idxs.extend(row["semipositive_satellite_idxs"])
        if not true_idxs:
            n_excluded_no_truepatch += 1
            continue
        true_idxs_t = torch.tensor(true_idxs, dtype=torch.long, device=device)
        true_finite_mask = (
            torch.isfinite(img_row[true_idxs_t])
            & torch.isfinite(lm_row[true_idxs_t])
        )
        true_idxs_t = true_idxs_t[true_finite_mask]
        if len(true_idxs_t) == 0:
            n_excluded_no_truepatch += 1
            continue
        if landmark_pair_sim_threshold is not None:
            keep_mask = lm_row[true_idxs_t] > landmark_pair_sim_threshold
            true_idxs_t = true_idxs_t[keep_mask]
            if len(true_idxs_t) == 0:
                n_excluded_pair_sim += 1
                continue

        # log-softmax over each grid σ.
        img_logits = img_row.unsqueeze(0) / sigmas_img_dev.unsqueeze(1)  # (Ki, M)
        lm_logits = lm_row.unsqueeze(0) / sigmas_lm_dev.unsqueeze(1)  # (Kl, M)
        log_p_img = torch.log_softmax(img_logits, dim=-1)  # (Ki, M)
        log_p_lm = torch.log_softmax(lm_logits, dim=-1)  # (Kl, M)

        if alpha_mode == "peak_sharpness":
            img_conf = _peak_sharpness(log_p_img)  # (Ki,)
            lm_conf = _peak_sharpness(log_p_lm)  # (Kl,)
            alpha = img_conf.unsqueeze(1) / (
                img_conf.unsqueeze(1) + lm_conf.unsqueeze(0) + eps
            )  # (Ki, Kl)
        elif alpha_mode == "raw_peak_sharpness":
            img_raw_conf = _raw_peak_sharpness_1d(img_row)  # scalar
            lm_raw_conf = _raw_peak_sharpness_1d(lm_row)  # scalar
            alpha_s = (img_raw_conf / (img_raw_conf + lm_raw_conf + eps)).item()
            alpha = torch.full((K_img, K_lm), alpha_s, device=device)
        elif alpha_mode == "raw_max":
            img_raw_conf = _raw_max_1d(img_row)  # scalar
            lm_raw_conf = _raw_max_1d(lm_row)  # scalar
            alpha_s = (img_raw_conf / (img_raw_conf + lm_raw_conf + eps)).item()
            alpha = torch.full((K_img, K_lm), alpha_s, device=device)
        else:
            raise ValueError(f"Unknown alpha_mode {alpha_mode!r}")

        log_p_img_t = log_p_img[:, true_idxs_t]  # (Ki, T)
        log_p_lm_t = log_p_lm[:, true_idxs_t]  # (Kl, T)
        # Fused log-prob at true patches: (Ki, Kl, T)
        fused_t = (
            alpha.unsqueeze(-1) * log_p_img_t.unsqueeze(1)
            + (1 - alpha).unsqueeze(-1) * log_p_lm_t.unsqueeze(0)
        )

        nll_fused += (-fused_t.sum(dim=-1).detach().cpu().double())
        nll_img_only += (-log_p_img_t.sum(dim=-1).detach().cpu().double())
        nll_lm_only += (-log_p_lm_t.sum(dim=-1).detach().cpu().double())
        pair_count += int(len(true_idxs_t))
        pano_count += 1

    if pair_count == 0:
        raise RuntimeError("No panos contributed pairs.")
    return {
        "nll_fused": nll_fused / pair_count,
        "nll_img_only": nll_img_only / pair_count,
        "nll_lm_only": nll_lm_only / pair_count,
        "pair_count": pair_count,
        "pano_count": pano_count,
        "num_excluded_constant": n_excluded_constant,
        "num_excluded_no_truepatch": n_excluded_no_truepatch,
        "num_excluded_pano_confidence": n_excluded_pano_confidence,
        "num_excluded_pair_sim": n_excluded_pair_sim,
    }


def plot_heatmap(
    sigmas_img: torch.Tensor,
    sigmas_lm: torch.Tensor,
    nll_fused: torch.Tensor,
    sigma_img_star: float,
    sigma_lm_star: float,
    sigma_img_marginal_star: float,
    sigma_lm_marginal_star: float,
    output_path: Path,
    pair_count: int,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    si = sigmas_img.numpy()
    sl = sigmas_lm.numpy()

    # pcolormesh on log-scaled axes shows actual σ values on the ticks
    pcm = ax.pcolormesh(si, sl, nll_fused.numpy().T, shading="auto", cmap="viridis")
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label("mean per-pair NLL")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_img")
    ax.set_ylabel("σ_lm")

    # Joint optimum
    ax.scatter([sigma_img_star], [sigma_lm_star], marker="*", color="white",
               edgecolor="red", s=260, lw=1.5, zorder=5,
               label=f"joint min ({sigma_img_star:.4f}, {sigma_lm_star:.4f})")

    # Marginal-NLL optima from this same grid (each stream's softmax NLL min)
    ax.scatter([sigma_img_marginal_star], [sigma_lm_marginal_star],
               marker="o", facecolor="none", edgecolor="white", s=180, lw=2,
               zorder=5,
               label=f"marginal min ({sigma_img_marginal_star:.4f}, "
                     f"{sigma_lm_marginal_star:.4f})")

    ax.set_title(f"EA fusion NLL surface (n={pair_count} pairs, Seattle)"
                 f"{title_suffix}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--img-sim-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/safa_dinov3_wag_chicago.pt")
    parser.add_argument(
        "--lm-sim-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt")
    parser.add_argument(
        "--dataset-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle")
    parser.add_argument("--landmark-version", type=str, default="v4_202001")
    parser.add_argument("--output-path", type=str, required=True)
    # Defaults bracket the per-stream NLL_softmax optima we already found
    # (image ≈ 0.049, landmark ≈ 0.130).
    parser.add_argument("--sigma-img-min", type=float, default=0.01)
    parser.add_argument("--sigma-img-max", type=float, default=0.4)
    parser.add_argument("--sigma-lm-min", type=float, default=0.02)
    parser.add_argument("--sigma-lm-max", type=float, default=0.8)
    parser.add_argument("--sigma-num", type=int, default=25)
    parser.add_argument(
        "--include-semipositive", type=lambda s: s.lower() != "false",
        default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--landmark-pano-confidence-threshold", type=float, default=None,
        help="Option 1: drop panos whose landmark row_max ≤ this value. "
             "Calibrates fusion only on panos where landmarks made a confident bid.")
    parser.add_argument(
        "--landmark-pair-sim-threshold", type=float, default=None,
        help="Option 3: drop (pano,true_patch) pairs whose landmark sim_t ≤ this. "
             "Calibrates only on pairs the landmark already scored highly.")
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Suffix appended to output filenames (e.g. 'pano_filter_1.0').")
    parser.add_argument(
        "--alpha-mode", type=str, default="peak_sharpness",
        choices=["peak_sharpness", "raw_peak_sharpness", "raw_max"],
        help="α formula for EA fusion. 'peak_sharpness' is the deployed EA "
             "default (σ-dependent). 'raw_peak_sharpness' uses max-mean of raw "
             "similarities. 'raw_max' uses raw row max. Last two are σ-independent.")
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "fusion_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    print(f"Loading image similarity matrix from {args.img_sim_path}")
    img_sim = _load_similarity_matrix(Path(args.img_sim_path)).double()
    print(f"  shape: {tuple(img_sim.shape)}")
    print(f"Loading landmark similarity matrix from {args.lm_sim_path}")
    lm_sim = _load_similarity_matrix(Path(args.lm_sim_path)).double()
    print(f"  shape: {tuple(lm_sim.shape)}")
    assert img_sim.shape == lm_sim.shape, (
        f"shape mismatch: img {tuple(img_sim.shape)} vs lm {tuple(lm_sim.shape)}")

    print(f"Loading dataset metadata from {args.dataset_path}")
    cfg = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=args.landmark_version,
    )
    ds = vd.VigorDataset(Path(args.dataset_path), cfg)
    pano_metadata = ds._panorama_metadata
    assert img_sim.shape[0] == len(pano_metadata)

    sigmas_img = torch.tensor(
        np.logspace(np.log10(args.sigma_img_min), np.log10(args.sigma_img_max),
                    args.sigma_num),
        dtype=torch.float64,
    )
    sigmas_lm = torch.tensor(
        np.logspace(np.log10(args.sigma_lm_min), np.log10(args.sigma_lm_max),
                    args.sigma_num),
        dtype=torch.float64,
    )

    print(f"Sweeping {args.sigma_num}x{args.sigma_num} grid "
          f"(σ_img ∈ [{args.sigma_img_min}, {args.sigma_img_max}], "
          f"σ_lm ∈ [{args.sigma_lm_min}, {args.sigma_lm_max}]) ...")
    res = fused_nll_grid(
        img_sim, lm_sim, pano_metadata, sigmas_img, sigmas_lm,
        include_semipositive=args.include_semipositive,
        device=device, exclude_constant_rows=True,
        landmark_pano_confidence_threshold=args.landmark_pano_confidence_threshold,
        landmark_pair_sim_threshold=args.landmark_pair_sim_threshold,
        alpha_mode=args.alpha_mode,
    )
    nll_fused = res["nll_fused"]
    nll_img_only = res["nll_img_only"]
    nll_lm_only = res["nll_lm_only"]
    pair_count = res["pair_count"]
    pano_count = res["pano_count"]
    print(f"  contributing panos: {pano_count}")
    print(f"  contributing (pano, true_patch) pairs: {pair_count}")
    print(f"  excluded constant-row panos:           {res['num_excluded_constant']}")
    print(f"  excluded no-truepatch panos:           {res['num_excluded_no_truepatch']}")
    print(f"  excluded by pano-confidence filter:    {res['num_excluded_pano_confidence']}")
    print(f"  excluded by pair-sim filter:           {res['num_excluded_pair_sim']}")

    flat = int(torch.argmin(nll_fused))
    bi, bl = flat // args.sigma_num, flat % args.sigma_num
    sigma_img_star = float(sigmas_img[bi])
    sigma_lm_star = float(sigmas_lm[bl])
    nll_fused_min = float(nll_fused[bi, bl])

    bi_marg = int(torch.argmin(nll_img_only))
    bl_marg = int(torch.argmin(nll_lm_only))
    sigma_img_marg = float(sigmas_img[bi_marg])
    sigma_lm_marg = float(sigmas_lm[bl_marg])

    print(f"  joint NLL min:    σ_img={sigma_img_star:.4f}  "
          f"σ_lm={sigma_lm_star:.4f}  NLL={nll_fused_min:.4f}")
    print(f"  marginal img NLL: σ_img={sigma_img_marg:.4f}  "
          f"NLL={float(nll_img_only[bi_marg]):.4f}")
    print(f"  marginal lm NLL:  σ_lm={sigma_lm_marg:.4f}  "
          f"NLL={float(nll_lm_only[bl_marg]):.4f}")
    nll_at_marg = float(nll_fused[bi_marg, bl_marg])
    print(f"  fused NLL at marginal optima: {nll_at_marg:.4f}  "
          f"(Δ from joint = {nll_at_marg - nll_fused_min:.4f})")

    suffix = ("_" + args.output_suffix) if args.output_suffix else ""
    title_pieces = [f"\nα = {args.alpha_mode}"]
    if args.landmark_pano_confidence_threshold is not None:
        title_pieces.append(
            f"  | pano filter: lm row_max > {args.landmark_pano_confidence_threshold}")
    if args.landmark_pair_sim_threshold is not None:
        title_pieces.append(
            f"  | pair filter: lm sim_t > {args.landmark_pair_sim_threshold}")
    title_suffix = "".join(title_pieces)
    plot_heatmap(
        sigmas_img, sigmas_lm, nll_fused,
        sigma_img_star, sigma_lm_star,
        sigma_img_marg, sigma_lm_marg,
        output_path / f"fusion_nll_heatmap{suffix}.png",
        pair_count,
        title_suffix=title_suffix,
    )

    summary = {
        "img_sim_path": str(args.img_sim_path),
        "lm_sim_path": str(args.lm_sim_path),
        "num_pairs": int(pair_count),
        "num_excluded_constant": int(res["num_excluded_constant"]),
        "num_excluded_no_truepatch": int(res["num_excluded_no_truepatch"]),
        "sigma_img_grid": sigmas_img.tolist(),
        "sigma_lm_grid": sigmas_lm.tolist(),
        "fused_nll_grid": nll_fused.tolist(),
        "img_only_nll_grid": nll_img_only.tolist(),
        "lm_only_nll_grid": nll_lm_only.tolist(),
        "sigma_img_joint_star": sigma_img_star,
        "sigma_lm_joint_star": sigma_lm_star,
        "min_fused_nll": nll_fused_min,
        "sigma_img_marginal_star": sigma_img_marg,
        "sigma_lm_marginal_star": sigma_lm_marg,
        "fused_nll_at_marginal_star": nll_at_marg,
    }
    with open(output_path / f"fusion_calibration{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Outputs in {output_path}")


if __name__ == "__main__":
    main()
