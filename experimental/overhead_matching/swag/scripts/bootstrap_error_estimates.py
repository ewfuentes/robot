import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import torch

    from pathlib import Path

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.style.use('ggplot')
    return Path, mo, np, pd, plt, torch


@app.cell
def _(Path):
    result_paths = [
        Path("/data/overhead_matching/evaluation/results/20250815_swag_semantic/NewYork/all_chicago_sat_dino_embedding_mat_pano_dino_sam"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8"),
        Path("/data/overhead_matching/evaluation/results/20250707_dino_features/NewYork/all_chicago_dino_project_512"),
    ]
    return (result_paths,)


@app.cell
def _(pd, result_paths, torch):
    def load_results_from_path(path):
        error = torch.load(path / "error.pt")

        return {
            'path_idx': int(path.parts[-1]),
            "model": path.parts[-2],
            "error_m": error[-1].item()
        }

    def load_results_from_all_paths(path):
        path_dirs = sorted(path.glob("0*"))
        return [load_results_from_path(p) for p in path_dirs]

    _records = []
    for _rp in result_paths:
        _records.extend(load_results_from_all_paths(_rp))

    results_df = pd.DataFrame.from_records(_records)    
    return (results_df,)


@app.cell
def _(mo, np, plt, results_df, torch):
    def bootstrap_estimate(values, fn, num_replicates=10000):
        out = []
        for i in range(num_replicates):
            # Randomly sample a population
            idxs = torch.randint(len(values), (len(values),))

            # compute a statistic
            out.append(fn(values[idxs].reshape(-1, 1)))
        # return the computed estimates
        return np.array(out)

    plt.figure(figsize=(12,6))
    query_pts = np.logspace(-3, 3.33, 200).reshape(1, -1)
    for _model_name, _df in results_df.groupby('model'):
        bootstrap_estimates = bootstrap_estimate(_df["error_m"].values, lambda x: (x < query_pts).sum(axis=0) / x.shape[0])
        _mean = bootstrap_estimates.mean(axis=0)
        _ci = np.percentile(bootstrap_estimates, [2.5, 97.5],  axis=0)
        plt.fill_between(query_pts.squeeze(), _ci[0, :], _ci[1, :], alpha=0.33)
        plt.plot(query_pts.squeeze(), _mean, label=_model_name)
    plt.legend()
    plt.xlabel('Final Position Error (m)')
    plt.ylabel("Fraction")
    plt.xlim(0, 30)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
