import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import common.torch.load_torch_deps
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path

    from experimental.overhead_matching.swag.data import vigor_dataset as vd
    return Path, np, pd, torch, vd


@app.cell
def _(mo):
    # CLI args via marimo app args
    args = mo.cli_args()
    return (args,)


@app.cell
def _(Path, args):
    img_sim_path = Path(args.get("img-sim-path", "/tmp/similarities/baseline_chicago.pt"))
    lm_sim_path = Path(args.get("lm-sim-path", "/tmp/similarities/proper_noun_embedding_chicago.pt"))
    dataset_path = Path(args.get("dataset-path", "/data/overhead_matching/datasets/VIGOR/Chicago"))
    landmark_version = args.get("landmark-version", "v4_202001")

    print(f"Image sim path: {img_sim_path}")
    print(f"Landmark sim path: {lm_sim_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Landmark version: {landmark_version}")
    return dataset_path, img_sim_path, landmark_version, lm_sim_path


@app.cell
def _(dataset_path, landmark_version, vd):
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=landmark_version,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, _config)
    panorama_metadata = vigor_dataset._panorama_metadata
    print(f"Loaded dataset with {len(panorama_metadata)} panoramas and "
          f"{len(vigor_dataset._satellite_metadata)} satellite patches")
    return panorama_metadata, vigor_dataset


@app.cell
def _(torch):
    def load_similarity_matrix(path):
        """Load a similarity matrix from a file.
        Handles both raw tensor format and dict format (with 'similarity' key).
        """
        data = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(data, dict):
            if "similarity" in data:
                return data["similarity"]
            raise ValueError(
                f"Similarity matrix file {path} is a dict but has no 'similarity' key. "
                f"Available keys: {list(data.keys())}"
            )
        return data
    return (load_similarity_matrix,)


@app.cell
def _(img_sim_path, lm_sim_path, load_similarity_matrix):
    img_sim = load_similarity_matrix(img_sim_path)
    lm_sim = load_similarity_matrix(lm_sim_path)

    print(f"Image similarity matrix shape: {img_sim.shape}")
    print(f"Landmark similarity matrix shape: {lm_sim.shape}")

    assert img_sim.shape[0] == lm_sim.shape[0], (
        f"Panorama count mismatch: img={img_sim.shape[0]} vs lm={lm_sim.shape[0]}")
    return img_sim, lm_sim


@app.cell
def _(panorama_metadata, pd, torch):
    SIGMA = 0.25

    def compute_diagnostics(sim_matrix, pano_metadata, sigma=SIGMA):
        """Compute per-panorama diagnostic metrics for a similarity matrix."""
        num_panos = sim_matrix.shape[0]
        records = []

        for pano_idx in range(num_panos):
            sim_row = sim_matrix[pano_idx]
            valid_mask = torch.isfinite(sim_row)
            if valid_mask.sum() == 0:
                records.append({
                    "pano_idx": pano_idx, "entropy": float("nan"),
                    "peak_sharpness": float("nan"), "max_sim": float("nan"),
                    "true_rank": float("nan"), "best_true_sim": float("nan"),
                    "has_data": False,
                })
                continue

            valid_sims = sim_row[valid_mask]

            # 1. Entropy of softmax(sim / sigma)
            logits = valid_sims / sigma
            log_probs = logits - torch.logsumexp(logits, dim=0)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum().item()

            # 2. Peak sharpness: max_sim - 2nd_max_sim
            finite_sorted = torch.sort(sim_row, descending=True).values
            finite_sorted = finite_sorted[torch.isfinite(finite_sorted)]
            peak_sharpness = (finite_sorted[0] - finite_sorted[1]).item() if len(finite_sorted) >= 2 else 0.0

            # 3. Max similarity
            max_sim = finite_sorted[0].item() if len(finite_sorted) > 0 else float("nan")

            # 4. Rank of true patch
            row = pano_metadata.iloc[pano_idx]
            true_idxs = list(row["positive_satellite_idxs"]) + list(row["semipositive_satellite_idxs"])
            if true_idxs:
                true_sims = sim_row[true_idxs]
                finite_true = true_sims[torch.isfinite(true_sims)]
                if len(finite_true) > 0:
                    best_true_sim = finite_true.max().item()
                    rank = (sim_row > best_true_sim).sum().item() + 1
                else:
                    best_true_sim, rank = float("nan"), float("nan")
            else:
                best_true_sim, rank = float("nan"), float("nan")

            records.append({
                "pano_idx": pano_idx, "entropy": entropy,
                "peak_sharpness": peak_sharpness, "max_sim": max_sim,
                "true_rank": rank, "best_true_sim": best_true_sim, "has_data": True,
            })

        return pd.DataFrame.from_records(records)

    print(f"Using sigma={SIGMA} for entropy computation")
    return SIGMA, compute_diagnostics


@app.cell
def _(compute_diagnostics, img_sim, lm_sim, panorama_metadata):
    img_diag = compute_diagnostics(img_sim, panorama_metadata)
    lm_diag = compute_diagnostics(lm_sim, panorama_metadata)
    print(f"Image diagnostics: {len(img_diag)} panos, {img_diag['has_data'].sum()} with data")
    print(f"Landmark diagnostics: {len(lm_diag)} panos, {lm_diag['has_data'].sum()} with data")
    return img_diag, lm_diag


@app.cell
def _(img_diag, lm_diag, mo, np, panorama_metadata, pd):
    combined = pd.DataFrame({
        "pano_idx": img_diag["pano_idx"], "pano_id": panorama_metadata["pano_id"].values,
        "img_entropy": img_diag["entropy"], "img_peak_sharpness": img_diag["peak_sharpness"],
        "img_max_sim": img_diag["max_sim"], "img_true_rank": img_diag["true_rank"],
        "img_best_true_sim": img_diag["best_true_sim"], "img_has_data": img_diag["has_data"],
        "lm_entropy": lm_diag["entropy"], "lm_peak_sharpness": lm_diag["peak_sharpness"],
        "lm_max_sim": lm_diag["max_sim"], "lm_true_rank": lm_diag["true_rank"],
        "lm_best_true_sim": lm_diag["best_true_sim"], "lm_has_data": lm_diag["has_data"],
    })
    both_valid = combined[combined["img_has_data"] & combined["lm_has_data"]].copy()
    both_valid["winner"] = np.where(
        both_valid["img_true_rank"] < both_valid["lm_true_rank"], "image",
        np.where(both_valid["img_true_rank"] > both_valid["lm_true_rank"], "landmark", "tie"))
    print(f"Panoramas with data in BOTH sources: {len(both_valid)}")
    print(both_valid["winner"].value_counts().to_string())
    mo.output.replace(mo.md(f"**{len(both_valid)}** panoramas have data in both sources"))
    return both_valid, combined


@app.cell
def _(both_valid, mo, pd):
    def _rank_stats(ranks, name):
        ranks = ranks.dropna()
        return {
            "Source": name, "Count": len(ranks),
            "Top-1": (ranks == 1).sum(), "Top-1 %": f"{100*(ranks == 1).mean():.1f}%",
            "Top-5": (ranks <= 5).sum(), "Top-5 %": f"{100*(ranks <= 5).mean():.1f}%",
            "Top-10": (ranks <= 10).sum(), "Top-10 %": f"{100*(ranks <= 10).mean():.1f}%",
            "Top-100": (ranks <= 100).sum(), "Top-100 %": f"{100*(ranks <= 100).mean():.1f}%",
            "Median Rank": f"{ranks.median():.0f}", "Mean Rank": f"{ranks.mean():.1f}",
            "Mean Recip Rank": f"{(1.0 / ranks).mean():.4f}",
        }
    summary_df = pd.DataFrame([
        _rank_stats(both_valid["img_true_rank"], "Image"),
        _rank_stats(both_valid["lm_true_rank"], "Landmark"),
    ])
    mo.output.replace(mo.ui.table(summary_df, label="Rank Summary"))
    return (summary_df,)


@app.cell
def _(both_valid, mo, pd):
    def _source_stats(df, prefix, name):
        return {
            "Source": name,
            "Mean Entropy": f"{df[f'{prefix}_entropy'].mean():.3f}",
            "Mean Peak Sharpness": f"{df[f'{prefix}_peak_sharpness'].mean():.4f}",
            "Mean Max Sim": f"{df[f'{prefix}_max_sim'].mean():.4f}",
            "Mean Best True Sim": f"{df[f'{prefix}_best_true_sim'].mean():.4f}",
        }
    dist_df = pd.DataFrame([
        _source_stats(both_valid, "img", "Image"),
        _source_stats(both_valid, "lm", "Landmark"),
    ])
    mo.output.replace(mo.ui.table(dist_df, label="Distribution Statistics"))
    return (dist_df,)


@app.cell
def _(both_valid, mo, np):
    import matplotlib as _matplotlib
    _matplotlib.style.use("ggplot")
    import matplotlib.pyplot as _plt

    _fig, _axes = _plt.subplots(2, 3, figsize=(18, 10))

    _ax = _axes[0, 0]
    _colors = {"image": "blue", "landmark": "orange", "tie": "gray"}
    for _w, _c in _colors.items():
        _mask = both_valid["winner"] == _w
        _ax.scatter(both_valid.loc[_mask, "img_true_rank"], both_valid.loc[_mask, "lm_true_rank"],
                    c=_c, alpha=0.3, s=10, label=_w)
    _ax.set_xlabel("Image Rank"); _ax.set_ylabel("Landmark Rank")
    _ax.set_title("Image Rank vs Landmark Rank"); _ax.set_xscale("log"); _ax.set_yscale("log")
    _ax.plot([1, 1e5], [1, 1e5], "k--", alpha=0.3); _ax.legend(fontsize=8)

    _ax = _axes[0, 1]
    _bins = np.logspace(0, 5, 50)
    _ax.hist(both_valid["img_true_rank"].dropna(), bins=_bins, alpha=0.5, label="Image")
    _ax.hist(both_valid["lm_true_rank"].dropna(), bins=_bins, alpha=0.5, label="Landmark")
    _ax.set_xscale("log"); _ax.set_xlabel("Rank"); _ax.set_title("Rank Distribution"); _ax.legend()

    _ax = _axes[0, 2]
    for _label, _col, _color in [("Image", "img_true_rank", "blue"), ("Landmark", "lm_true_rank", "orange")]:
        _ranks = both_valid[_col].dropna().sort_values().values
        _ax.step(_ranks, np.arange(1, len(_ranks)+1)/len(_ranks), label=_label, color=_color)
    _ax.set_xscale("log"); _ax.set_xlabel("Rank"); _ax.set_ylabel("Fraction <= Rank")
    _ax.set_title("Rank ECDF"); _ax.legend()

    _ax = _axes[1, 0]
    _ax.scatter(both_valid["img_entropy"], both_valid["lm_entropy"], alpha=0.2, s=10)
    _ax.set_xlabel("Image Entropy"); _ax.set_ylabel("Landmark Entropy"); _ax.set_title("Entropy")

    _ax = _axes[1, 1]
    _ax.scatter(both_valid["img_peak_sharpness"], both_valid["lm_peak_sharpness"], alpha=0.2, s=10)
    _ax.set_xlabel("Image Peak Sharpness"); _ax.set_ylabel("Landmark Peak Sharpness")
    _ax.set_title("Peak Sharpness")

    _ax = _axes[1, 2]
    _diff = (both_valid["img_true_rank"] - both_valid["lm_true_rank"]).dropna().clip(-500, 500)
    _ax.hist(_diff, bins=100, alpha=0.7); _ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    _ax.set_xlabel("Img Rank - Lm Rank"); _ax.set_title("Rank Difference")

    _plt.tight_layout()
    mo.mpl.interactive(_plt.gcf())
    return


@app.cell
def _(both_valid, mo, np):
    import matplotlib.pyplot as _plt
    _fig2, _axes2 = _plt.subplots(1, 2, figsize=(14, 5))

    _ax = _axes2[0]
    for _label, _col, _color in [("Image", "img_true_rank", "blue"), ("Landmark", "lm_true_rank", "orange")]:
        _rr = (1.0 / both_valid[_col]).dropna().sort_values().values
        _ax.step(_rr, np.arange(1, len(_rr)+1)/len(_rr), label=_label, color=_color)
    _ax.set_xlabel("Reciprocal Rank"); _ax.set_title("Reciprocal Rank ECDF"); _ax.legend()

    _ax = _axes2[1]
    _oracle = np.minimum(both_valid["img_true_rank"], both_valid["lm_true_rank"])
    for _label, _ranks, _color in [("Image", both_valid["img_true_rank"], "blue"),
                                    ("Landmark", both_valid["lm_true_rank"], "orange"),
                                    ("Oracle", _oracle, "green")]:
        _vals = _ranks.dropna().sort_values().values
        _ax.step(_vals, np.arange(1, len(_vals)+1)/len(_vals), label=_label, color=_color)
    _ax.set_xscale("log"); _ax.set_xlabel("Rank"); _ax.set_title("Oracle Best-of-Both"); _ax.legend()

    _plt.tight_layout()
    mo.mpl.interactive(_plt.gcf())
    return


@app.cell
def _(both_valid, np, summary_df):
    print("=" * 70)
    print("FUSION DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Total panoramas analyzed: {len(both_valid)}")
    print(summary_df.to_string(index=False))

    _oracle_rank = np.minimum(both_valid["img_true_rank"], both_valid["lm_true_rank"])
    print(f"Oracle (min rank):")
    for _k in [1, 5, 10, 100]:
        print(f"  Top-{_k}: {(_oracle_rank <= _k).sum()} ({100*(_oracle_rank <= _k).mean():.1f}%)")
    print(f"  Mean RR: {(1.0 / _oracle_rank).mean():.4f}")

    for _w, _c in both_valid["winner"].value_counts().items():
        print(f"  {_w}: {_c} ({100*_c/len(both_valid):.1f}%)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
