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
    import seaborn
    import math

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
        torch,
        vigor_dataset,
    )


@app.cell
def _(Path, evaluate_swag, load_model):
    def get_top_k_results(model_partial_path, dataset):
        sat_model = load_model(Path(f"{model_partial_path}_satellite"), device='cuda')
        pano_model = load_model(Path(f"{model_partial_path}_panorama"), device='cuda')

        df, all_similarity = evaluate_swag.evaluate_prediction_top_k(sat_model=sat_model, pano_model=pano_model, dataset=dataset)
        del sat_model, pano_model, all_similarity
        return df
    return (get_top_k_results,)


@app.cell
def _(NamedTuple, Path, itertools, vigor_dataset):
    class ModelConfig(NamedTuple):
        lr_schedule: bool
        negative_mining: bool
        pos_semipos: bool

    model_paths = {}
    idx=59
    for lr_schedule, negative_mining, pos_semipos in itertools.product(*[[False, True]]*3):
        model_paths[ModelConfig(lr_schedule, negative_mining, pos_semipos)] = f"/data/overhead_matching/models/20250616_8_way_experiment/all_chicago_lr_schedule_{lr_schedule}_negative_mining_{negative_mining}_pos_semipos_{pos_semipos}/{idx:04d}"

    # model_paths = {
    #     ModelConfig(False, False, False): "/data/overhead_matching/models/all_chicago_model/0240",
    #     'w_semi_pos': "/data/overhead_matching/models/all_chicago_model_w_semipos/0080",
    # }

    dataset_paths = {
        'chicago': '/data/overhead_matching/datasets/VIGOR/Chicago',
        # 'sanfrancisco': '/data/overhead_matching/datasets/VIGOR/SanFrancisco'
        "newyork": '/data/overhead_matching/datasets/VIGOR/NewYork'
    }

    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=1e-6,
        satellite_patch_size=(320, 320),
        panorama_size=(320,640),
        factor=1.0
    )

    datasets = {k: vigor_dataset.VigorDataset(Path(v), dataset_config) for k, v in dataset_paths.items()}
    return (
        ModelConfig,
        dataset_config,
        dataset_paths,
        datasets,
        idx,
        lr_schedule,
        model_paths,
        negative_mining,
        pos_semipos,
    )


@app.cell
def _(datasets, get_top_k_results, itertools, model_paths, pd):
    dfs = []
    for (model_name, model_path), (data_name, _dataset) in itertools.product(model_paths.items(), datasets.items()):
        print(model_name, data_name)
        df = get_top_k_results(model_path, _dataset)
        df["dataset"] = data_name
        df["model"] = [model_name]*len(df)
        dfs.append(df)
    df = pd.concat(dfs)
    return data_name, df, dfs, model_name, model_path


@app.cell
def _(df, mo, plt, seaborn):
    df["plot_label"] = [str(x) for x in df["model"]]

    seaborn.displot(data=df, kind='ecdf', x='k_value', col='dataset', hue='plot_label')
    plt.suptitle("top K CDF by model")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(Path, itertools, pd, torch):
    # Plot statistics about path evaluation

    def process_path(path: Path):
        error_path = path / 'error.pt'
        var_path = path / 'var.pt'
        error_data = torch.load(error_path)
        var_data = torch.load(var_path)
        return error_data[-1].item(), var_data[-1].item()

    def process_eval_results(path: Path):
        out = []
        for p in sorted(path.glob("[0-9]*")):
            try:
                final_error_m, var_sq_m = process_path(p)
                out.append({
                    'path_idx': int(p.stem),
                    'final_error_m': final_error_m,
                    'var_sq_m': var_sq_m,
                    'model': path.stem,
                })
            except:
                ...
        return pd.DataFrame.from_records(out)

    base_path = Path('/data/overhead_matching/evaluation/results/20250616_8_way_experiment_fixed/')

    path_dfs = []
    for _lr_schedule, _negative_mining, _pos_semipos in itertools.product(*[[False, True]]*3):
        _model_name = f"all_chicago_lr_schedule_{_lr_schedule}_negative_mining_{_negative_mining}_pos_semipos_{_pos_semipos}"
        path_dfs.append(process_eval_results(base_path / _model_name))

    path_df = pd.concat(path_dfs)
    return base_path, path_df, path_dfs, process_eval_results, process_path


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.displot(data=path_df, x='final_error_m', kind='ecdf', hue='model')
    plt.title("Final Error CDF across 100 paths in New York")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.scatterplot(data=path_df, x="final_error_m", y="var_sq_m", hue="model")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, model_paths):
    path_slider = mo.ui.slider(0, 99)
    model_selector = mo.ui.dropdown({str(x):x for x in model_paths.keys()})
    mo.vstack([path_slider, model_selector])
    return model_selector, path_slider


@app.cell
def _(Path, mo, model_selector, path_slider, plt, torch):
    print(path_slider.value)
    print(model_selector.value)
    _v = model_selector.value
    plt.figure()
    _base_path = Path('/data/overhead_matching/evaluation/results/20250616_8_way_experiment')
    _model_name = f"all_chicago_lr_schedule_{_v.lr_schedule}_negative_mining_{_v.negative_mining}_pos_semipos_{_v.pos_semipos}"
    _error_path = _base_path / _model_name / f"{path_slider.value:07d}" /'error.pt'
    _error_data = torch.load(_error_path)

    plt.figure(figsize=(12, 4))
    plt.plot(_error_data)
    plt.xlabel("Panorama Idx")
    plt.ylabel("Error (m)")
    plt.title(f'{_v}\n Path {path_slider.value}')
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(NamedTuple, Path, pd, re, torch):
    # Mixture MCL Evaluation

    class MixtureSettings(NamedTuple):
        frac: float
        phantom_counts: float

    def extract_mixture_info(p):
        m = re.match(r"mcl_frac_([0-9.]+)_phantom_counts_([0-9.]+)", p.name)
        if not m:
            print(f"Could not parse settings from: {p}")
            return

        out = []

        settings = MixtureSettings(frac=float(m.group(1)), phantom_counts=float(m.group(2)))
        for eval_path in sorted(p.glob("[0-9]*")):
            error = torch.load(eval_path / "error.pt")
            var = torch.load(eval_path / "var.pt")
            out.append({
                "path_id": int(eval_path.name),
                "dual_mcl_frac": settings.frac,
                "phantom_counts": settings.phantom_counts,
                "final_error_m": error[-1].item(),
                "final_var_sq_m": var[-1].item(),
                "settings": settings})

        return pd.DataFrame.from_records(out)



    _experiment_folder = Path('/data/overhead_matching/evaluation/results/mixture_mcl_sweep_fixed/')
    _dfs = []
    for p in _experiment_folder.glob("mcl_frac*"):
        _dfs.append(extract_mixture_info(p))

    mixture_mcl_df = pd.concat(_dfs).reset_index(drop=True)
    return MixtureSettings, extract_mixture_info, mixture_mcl_df, p


@app.cell
def _(mixture_mcl_df):
    mixture_mcl_df
    return


@app.cell
def _(mixture_mcl_df, mo, plt, seaborn):
    plt.figure()
    seaborn.boxenplot(data=mixture_mcl_df, x="dual_mcl_frac", y="final_error_m", hue="phantom_counts", legend="full", palette='tab10')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
