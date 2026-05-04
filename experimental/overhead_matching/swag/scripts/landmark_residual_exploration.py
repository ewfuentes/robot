"""Cross-city exploration of landmark-similarity residuals.

For each of Chicago / Seattle / NewYork / post-hurricane-ian, load the v6
Hungarian landmark similarity matrix, walk the panorama metadata to gather
per-pair (pano, true_patch) samples + sampled negatives, and build the per-
pair residual statistics under two parameterizations:

  - raw residual:        ``r_raw  = sim_max - sim_t``
  - normalized residual: ``r_norm = 1 - sim_t / sim_max``  (the user's
    proposed normalization for the new fusion aggregator)

Emits five figures into ``--output-path`` and prints σ_MLE / σ_median per
city. The headline plot is ``normalized_true_residual_overlay.png`` — if
the four cities overlap closely, the normalization is transfer-friendly and
we can pick a single σ_lm.

All-zero (or otherwise constant) landmark rows are excluded from per-pair
sampling — they're handled by an image-only fall-through in the aggregator
itself.
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np
import matplotlib.pyplot as plt

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)


CITY_DEFAULTS: dict[str, dict] = {
    "Chicago": {
        "dataset_path": "/data/overhead_matching/datasets/VIGOR/Chicago",
        "lm_sim_path": "/data/overhead_matching/datasets/VIGOR/Chicago/"
                       "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt",
        "landmark_version": "v4_202001",
    },
    "Seattle": {
        "dataset_path": "/data/overhead_matching/datasets/VIGOR/Seattle",
        "lm_sim_path": "/data/overhead_matching/datasets/VIGOR/Seattle/"
                       "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt",
        "landmark_version": "v4_202001",
    },
    "NewYork": {
        "dataset_path": "/data/overhead_matching/datasets/VIGOR/NewYork",
        "lm_sim_path": "/data/overhead_matching/datasets/VIGOR/NewYork/"
                       "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt",
        "landmark_version": "v4_202001",
    },
    "post_hurricane_ian": {
        "dataset_path": "/data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian",
        "lm_sim_path": "/data/overhead_matching/datasets/VIGOR/mapillary/"
                       "post_hurricane_ian/similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt",
        "landmark_version": "post_hurricane_ian_v1_220101",
    },
}

CITY_COLORS = {
    "Chicago": "C0",
    "Seattle": "C1",
    "NewYork": "C2",
    "post_hurricane_ian": "C3",
}


def load_city(city: str) -> tuple[torch.Tensor, "pd.DataFrame"]:
    """Load (lm_sim, panorama_metadata) for a city using its defaults."""
    cfg = CITY_DEFAULTS[city]
    print(f"  [{city}] loading lm matrix from {cfg['lm_sim_path']}")
    lm_sim = _load_similarity_matrix(Path(cfg["lm_sim_path"]))
    ds_cfg = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=cfg["landmark_version"],
    )
    ds = vd.VigorDataset(Path(cfg["dataset_path"]), ds_cfg)
    pano_meta = ds._panorama_metadata
    assert lm_sim.shape[0] == len(pano_meta), (
        f"[{city}] sim rows {lm_sim.shape[0]} != panoramas {len(pano_meta)}"
    )
    return lm_sim, pano_meta


def gather_samples(
    lm_sim: torch.Tensor,
    pano_meta,
    include_semipositive: bool = True,
    num_negatives_per_pano: int = 200,
    rng_seed: int = 0,
) -> dict:
    """Walk panoramas and return per-pair true / negative similarity samples.

    Returns a dict with keys: row_max (per pano), num_zero_rows, num_constant_rows,
    num_kept_rows, true_sim, true_residual_raw, true_residual_norm,
    negative_residual_raw, negative_residual_norm.
    """
    rng = np.random.default_rng(rng_seed)
    num_panos = lm_sim.shape[0]
    num_patches = lm_sim.shape[1]

    row_max_all = torch.full((num_panos,), float("nan"), dtype=torch.float64)
    true_sim, true_r_raw, true_r_norm = [], [], []
    neg_r_raw, neg_r_norm = [], []
    n_zero_rows = 0
    n_constant_rows = 0
    n_kept_rows = 0
    n_no_truepatch = 0

    for pano_idx in range(num_panos):
        sim_row = lm_sim[pano_idx]
        finite_mask = torch.isfinite(sim_row)
        if not finite_mask.any():
            n_zero_rows += 1
            continue
        finite_vals = sim_row[finite_mask]
        rmax = finite_vals.max().item()
        rmin = finite_vals.min().item()
        row_max_all[pano_idx] = rmax
        if rmax == 0.0:
            n_zero_rows += 1
            continue
        if rmax == rmin:
            n_constant_rows += 1
            continue

        row = pano_meta.iloc[pano_idx]
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
            true_sim.append(ts)
            true_r_raw.append(rmax - ts)
            true_r_norm.append(1.0 - ts / rmax)

        n_kept_rows += 1

        # Sample negatives (random patches that aren't true).
        if num_negatives_per_pano > 0:
            true_set = set(true_idxs)
            cand = rng.integers(0, num_patches, size=num_negatives_per_pano * 2)
            cand = [int(c) for c in cand if int(c) not in true_set][:num_negatives_per_pano]
            cand_t = torch.tensor(cand, dtype=torch.long)
            neg_sims = sim_row[cand_t]
            neg_finite = neg_sims[torch.isfinite(neg_sims)]
            for ns in neg_finite.tolist():
                neg_r_raw.append(rmax - ns)
                neg_r_norm.append(1.0 - ns / rmax)

    return {
        "row_max_all": row_max_all,
        "n_zero_rows": n_zero_rows,
        "n_constant_rows": n_constant_rows,
        "n_no_truepatch": n_no_truepatch,
        "n_kept_rows": n_kept_rows,
        "true_sim": np.asarray(true_sim, dtype=np.float64),
        "true_r_raw": np.asarray(true_r_raw, dtype=np.float64),
        "true_r_norm": np.asarray(true_r_norm, dtype=np.float64),
        "neg_r_raw": np.asarray(neg_r_raw, dtype=np.float64),
        "neg_r_norm": np.asarray(neg_r_norm, dtype=np.float64),
    }


def plot_row_max_distribution(per_city: dict[str, dict], out: Path) -> None:
    """Histogram of row-max(lm_sim) across rows, one curve per city."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0.0, 5.0, 60)
    for city, samples in per_city.items():
        rmax = samples["row_max_all"].numpy()
        rmax = rmax[np.isfinite(rmax)]
        ax.hist(
            rmax, bins=bins, histtype="step", lw=1.5,
            color=CITY_COLORS[city],
            label=f"{city} "
                  f"(0-rows={samples['n_zero_rows']}, "
                  f"const={samples['n_constant_rows']}, "
                  f"kept={samples['n_kept_rows']})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("row_max(lm_sim) — landmark similarity matrix")
    ax.set_ylabel("count (log)")
    ax.set_title("Per-city distribution of landmark row-max\n"
                 "(rows with row_max=0 are dropped by the new aggregator)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_raw_true_sim_overlay(per_city: dict[str, dict], out: Path) -> None:
    """Per-pair true similarity (raw, no normalization), one curve per city."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.linspace(0.0, 4.0, 80)
    for city, samples in per_city.items():
        ts = samples["true_sim"]
        ax.hist(
            ts, bins=bins, histtype="step", lw=1.5,
            color=CITY_COLORS[city],
            label=f"{city} (n={len(ts)})  "
                  f"median={np.median(ts):.3f}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("raw landmark similarity at true patches  sim_t")
    ax.set_ylabel("count (log)")
    ax.set_title("True-patch raw landmark similarity, per city\n"
                 "(absolute scale differs across cities — motivates normalization)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_normalized_true_residual_overlay(per_city: dict[str, dict], out: Path) -> None:
    """Headline plot: per-city normalized-residual density at true patches."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    bins = np.linspace(0.0, 1.0, 60)
    for ax, scale_name in zip(axes, ["density", "log count"]):
        for city, samples in per_city.items():
            r = samples["true_r_norm"]
            ax.hist(
                r, bins=bins,
                density=(scale_name == "density"),
                histtype="step", lw=1.6,
                color=CITY_COLORS[city],
                label=f"{city} (n={len(r)})  σ_MLE={float(np.sqrt(np.mean(r**2))):.3f}  "
                      f"med={float(np.median(r)):.3f}",
            )
        ax.set_xlabel(r"$r_{\rm norm} = 1 - \mathrm{sim}_t / \mathrm{sim}_{\max}$")
        if scale_name == "log count":
            ax.set_yscale("log")
            ax.set_ylabel("count (log)")
        else:
            ax.set_ylabel("density")
        ax.set_title(f"Normalized true-patch residual ({scale_name})")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "HEADLINE: per-city overlap of true-patch residuals AFTER /row_max normalization."
        " Tight overlap → σ_lm transfers."
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_true_vs_negative_per_city(per_city: dict[str, dict], out: Path) -> None:
    """2×2 grid (one per city) of true vs negative normalized residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(0.0, 1.0, 60)
    for ax, (city, samples) in zip(axes.flatten(), per_city.items()):
        tr = samples["true_r_norm"]
        nr = samples["neg_r_norm"]
        ax.hist(tr, bins=bins, color="steelblue", alpha=0.65,
                label=f"true (n={len(tr)})")
        ax.hist(nr, bins=bins, histtype="step", color="darkorange", lw=1.5,
                label=f"negative sample (n={len(nr)})")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
        ax.set_xlabel(r"$r_{\rm norm} = 1 - \mathrm{sim}/\mathrm{sim}_{\max}$")
        ax.set_ylabel("count (log)")
        ax.set_title(f"{city} — separability under /row_max normalization")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-city true vs negative normalized residuals")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_sigma_calibration(per_city: dict[str, dict], out: Path) -> tuple[float, float]:
    """Bar chart of σ_MLE and σ_median (on r_norm) per city. Return cross-city
    averaged σ_MLE (used as the σ_lm candidate) plus the across-city ratio
    max/min (for the ±25% transferability check)."""
    cities = list(per_city.keys())
    sigma_mles, sigma_medians = [], []
    for city in cities:
        r = per_city[city]["true_r_norm"]
        sigma_mles.append(float(np.sqrt(np.mean(r ** 2))))
        sigma_medians.append(float(np.median(r)))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cities))
    w = 0.4
    ax.bar(x - w / 2, sigma_mles, width=w, color="C0", label="σ_MLE")
    ax.bar(x + w / 2, sigma_medians, width=w, color="C2", label="σ_median")
    for i, (m, med) in enumerate(zip(sigma_mles, sigma_medians)):
        ax.text(i - w / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, med + 0.005, f"{med:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=15)
    ax.set_ylabel("σ on r_norm")
    avg_mle = float(np.mean(sigma_mles))
    spread = float(np.max(sigma_mles) / max(np.min(sigma_mles), 1e-9))
    ax.set_title(
        f"σ on normalized residual r_norm = 1 − sim_t/sim_max\n"
        f"avg σ_MLE = {avg_mle:.3f}    max/min = {spread:.2f}× "
        f"(target < 1.25 for transferability)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return avg_mle, spread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--cities", type=str, default="Chicago,Seattle,NewYork,post_hurricane_ian"
    )
    parser.add_argument("--num-negatives-per-pano", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    print(f"Exploring {len(cities)} cities: {cities}")
    per_city: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    for city in cities:
        print(f"--- {city} ---")
        lm_sim, pano_meta = load_city(city)
        samples = gather_samples(
            lm_sim, pano_meta,
            num_negatives_per_pano=args.num_negatives_per_pano,
            rng_seed=args.seed,
        )
        per_city[city] = samples
        sigma_mle = float(np.sqrt(np.mean(samples["true_r_norm"] ** 2)))
        sigma_median = float(np.median(samples["true_r_norm"]))
        sigma_mle_raw = float(np.sqrt(np.mean(samples["true_r_raw"] ** 2)))
        summary[city] = {
            "n_panoramas_total": int(lm_sim.shape[0]),
            "n_zero_or_const_rows": int(samples["n_zero_rows"] + samples["n_constant_rows"]),
            "n_kept_rows": int(samples["n_kept_rows"]),
            "n_truepatch_pairs": int(len(samples["true_r_norm"])),
            "sigma_mle_norm_residual": sigma_mle,
            "sigma_median_norm_residual": sigma_median,
            "sigma_mle_raw_residual": sigma_mle_raw,
        }
        print(f"  rows total={lm_sim.shape[0]}  "
              f"zero+const={samples['n_zero_rows'] + samples['n_constant_rows']}  "
              f"kept={samples['n_kept_rows']}  pairs={len(samples['true_r_norm'])}")
        print(f"  σ_MLE  (norm r) = {sigma_mle:.4f}  "
              f"σ_med = {sigma_median:.4f}  σ_MLE (raw r) = {sigma_mle_raw:.4f}")

    print("\nWriting plots and summary...")
    plot_row_max_distribution(per_city, output_path / "row_max_distribution.png")
    plot_raw_true_sim_overlay(per_city, output_path / "raw_true_sim_overlay.png")
    plot_normalized_true_residual_overlay(
        per_city, output_path / "normalized_true_residual_overlay.png"
    )
    plot_true_vs_negative_per_city(
        per_city, output_path / "true_vs_negative_per_city.png"
    )
    avg_mle, spread = plot_sigma_calibration(
        per_city, output_path / "sigma_calibration.png"
    )

    summary["__cross_city__"] = {
        "avg_sigma_mle_norm_residual": avg_mle,
        "max_over_min_spread": spread,
        "cities": cities,
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Outputs in {output_path}")
    print(f"  Cross-city avg σ_MLE (norm) = {avg_mle:.4f}    "
          f"max/min spread = {spread:.2f}×")
    if spread < 1.25:
        print("  → spread within target (<1.25×) — proceed to Phase B with this σ_lm.")
    else:
        print("  → spread above target; normalization may not transfer cleanly.")


if __name__ == "__main__":
    main()
