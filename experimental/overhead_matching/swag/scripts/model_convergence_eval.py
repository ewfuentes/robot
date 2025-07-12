import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import common.torch.load_torch_deps
    import torch
    import itertools
    from pprint import pprint
    import numpy as np
    import seaborn as sns
    import math
    import tqdm

    import re

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use("ggplot")

    from experimental.overhead_matching.swag.data import vigor_dataset, satellite_embedding_database
    from experimental.overhead_matching.swag.evaluation import evaluate_swag
    import experimental.overhead_matching.swag.model.patch_embedding
    from common.torch.load_and_save_models import load_model
    from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig, SatellitePatchConfig
    from common.math import haversine
    from pathlib import Path
    from typing import NamedTuple
    import seaborn

    import importlib
    importlib.reload(evaluate_swag)
    importlib.reload(haversine)
    return (
        NamedTuple,
        Path,
        SatellitePatchConfig,
        WagConfig,
        alt,
        common,
        evaluate_swag,
        experimental,
        haversine,
        importlib,
        itertools,
        load_model,
        math,
        matplotlib,
        mo,
        np,
        pd,
        plt,
        pprint,
        re,
        satellite_embedding_database,
        seaborn,
        sns,
        torch,
        tqdm,
        vigor_dataset,
    )


@app.cell
def _(Path, pd, torch, tqdm):
    # Plot statistics about path evaluation

    def process_path(path: Path):
        error_path = path / 'error.pt'
        var_path = path / 'var.pt'
        error_data = torch.load(error_path)
        var_data = torch.load(var_path)
        return error_data, var_data

    def process_eval_results(path: Path):
        out = []
        for p in tqdm.tqdm(sorted(path.glob("[0-9]*"))):
            try:
                error_m, var_sq_m = process_path(p)
                for i, (e, v) in enumerate(zip(error_m.detach().cpu().tolist(), var_sq_m.detach().cpu().tolist())):
                    out.append({
                        'path_idx': int(p.stem),
                        'error_m': e,
                        'var_sq_m': v,
                        "datapoint_index": i,
                        'model': path.stem,
                    })
            except:
                ...
        return pd.DataFrame.from_records(out)

    _base_path = Path('/home/ekf/scratch/crossview/evaluations/')

    _results_paths = sorted(_base_path.iterdir())

    path_dfs = []
    for _p in _results_paths:
        path_dfs.append(process_eval_results(_p))
    path_df = pd.concat(path_dfs)
    return path_df, path_dfs, process_eval_results, process_path


@app.cell
def _(path_df, plt, sns):
    # Plot error_m
    plt.figure(figsize=(6,4))
    sns.lineplot(data=path_df, x="datapoint_index", y="error_m", hue="path_idx", palette=None, legend=None, alpha=0.1)
    plt.title("Error over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(path_df, plt, sns):
    # Plot var_sq_m
    plt.figure(figsize=(6,4))
    sns.lineplot(data=path_df, x="datapoint_index", y="var_sq_m", hue="path_idx", palette=None, legend=None, alpha=0.1)
    plt.title("Variance over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo, path_df, plt, sns):
    plt.figure(figsize=(8,4))
    path_lengths = path_df.groupby("path_idx")["datapoint_index"].max()
    shortest_last_index = path_lengths.min()
    print(f"Shortest path ends at index {shortest_last_index}")
    plt.axvline(shortest_last_index, color="black", linestyle="--", label="End of shortest path")

    sns.lineplot(
        data=path_df,
        x="datapoint_index",
        y="error_m",
        estimator="median",  
        errorbar=("ci", 95),
        hue="model"
    )

    plt.title("Median Error over Time with 95% CI Intervals")
    plt.ylabel("Error (m)")
    plt.xlabel("Timestep")
    plt.grid(True)
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return path_lengths, shortest_last_index


@app.cell
def _(path_df, plt, sns):
    # Bin size
    bin_size = 100
    # Create a new column for the bin label
    path_df["index_bin"] = (path_df["datapoint_index"] // bin_size) * bin_size

    plt.figure(figsize=(12,4))
    sns.boxplot(
        data=path_df,
        x="index_bin",
        y="error_m",
        color="lightblue",
        showfliers=False  # optionally hide outliers for clarity
    )

    plt.title("Error Distribution Over Time (Boxplots)")
    plt.xlabel("Timestep")
    plt.ylabel("Error (m)")
    plt.xticks(rotation=90)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()
    return (bin_size,)


@app.cell
def _(path_df, pd, plt, shortest_last_index):
    thresholds = [10, 100, 1000]
    result = []
    grouped = path_df.groupby("datapoint_index")
    for t in thresholds:
        tmp = grouped.apply(lambda g: (g["error_m"] < t).mean()).reset_index(name="proportion")
        tmp["threshold"] = t
        result.append(tmp)
    all_thresh_df = pd.concat(result, ignore_index=True)
    plt.figure(figsize=(8,4))
    plt.axvline(shortest_last_index, color="black", linestyle="--", label="End of shortest path")

    for t in thresholds:
        subset = all_thresh_df[all_thresh_df["threshold"]==t]
        plt.plot(
            subset["datapoint_index"],
            subset["proportion"],
            label=f"< {t} m"
        )
    plt.xlabel("Timestep")
    plt.ylabel("Proportion with error_m below threshold")
    plt.title("Proportion of Paths Below Error Thresholds")
    plt.ylim(0,1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return all_thresh_df, grouped, result, subset, t, thresholds, tmp


@app.cell
def _(np, path_df, pd):
    def flattened_df_to_ndarray(
        dataframe: pd.DataFrame,
        data_key: str,
        index_key: str = "datapoint_index",
        path_key: str = "path_idx",
        model_key: str = "model",
    ) -> dict[str, np.ndarray]:
        out_arrays = {}
        path_lengths = path_df.groupby(path_key)[index_key].max()
        shortest_last_index = path_lengths.min()

        for model_name in dataframe[model_key].unique():
            model_df = dataframe[dataframe[model_key] == model_name]
            num_paths = len(model_df[path_key].unique())
            out_arr = np.ones((num_paths, shortest_last_index)) * np.nan
            for _, row in model_df.iterrows():
                if row[index_key] < shortest_last_index:
                    out_arr[row[path_key], row[index_key]] = row[data_key]
            assert not np.any(np.isnan(out_arr))
            out_arrays[model_name] = out_arr
        return out_arrays

    error_m_np_arrays = flattened_df_to_ndarray(path_df, "error_m")


    return error_m_np_arrays, flattened_df_to_ndarray


@app.cell
def _(error_m_np_arrays, matplotlib, np, plt):
    percentile_cutoff = np.asarray([5, 50, 95])
    line_styles = ['-', '--', '-.']

    cmap = matplotlib.colormaps.get_cmap('hsv')
    error_percentiles = {k: np.percentile(v, percentile_cutoff, axis=0) for k, v in error_m_np_arrays.items()}
    fig, ax = plt.subplots()
    for model_i, (model, percentiles) in enumerate(error_percentiles.items()):
        for i, (ls, percentile) in enumerate(zip(line_styles, percentile_cutoff)):
            ax.plot(percentiles[i], c=cmap(model_i), ls=ls, label=f"{model} {percentile}%")
    plt.xlabel("Timestep")
    plt.ylabel("Error (m)")
    plt.title("Error Percentiles over Timesteps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (
        ax,
        cmap,
        error_percentiles,
        fig,
        i,
        line_styles,
        ls,
        model,
        model_i,
        percentile,
        percentile_cutoff,
        percentiles,
    )


if __name__ == "__main__":
    app.run()
