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

    from experimental.overhead_matching.swag.data import vigor_dataset, satellite_embedding_database
    from experimental.overhead_matching.swag.evaluation import evaluate_swag
    import experimental.overhead_matching.swag.model.patch_embedding
    from common.torch.load_and_save_models import load_model
    from pathlib import Path
    from typing import NamedTuple
    import seaborn

    import importlib
    importlib.reload(evaluate_swag)
    return (
        NamedTuple,
        Path,
        alt,
        common,
        evaluate_swag,
        experimental,
        importlib,
        itertools,
        load_model,
        mo,
        np,
        pd,
        pprint,
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
def _(NamedTuple, Path, get_top_k_results, itertools, pd, vigor_dataset):
    class ModelConfig(NamedTuple):
        lr_schedule: bool
        negative_mining: bool
        pos_semipos: bool

    checkpoint_idx = [59, 59, 59, 59, 59, 59, 60, 60]
    model_paths = {}
    for idx, (lr_schedule, negative_mining, pos_semipos) in zip(checkpoint_idx, itertools.product(*[[False, True]]*3)):
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

    results = {}

    dfs = []
    for (model_name, model_path), (data_name, dataset) in itertools.product(model_paths.items(), datasets.items()):
        print(model_name, data_name)
        df = get_top_k_results(model_path, dataset)
        df["dataset"] = data_name
        df["model"] = [model_name]*len(df)
        dfs.append(df)
    df = pd.concat(dfs)
    return (
        ModelConfig,
        checkpoint_idx,
        data_name,
        dataset,
        dataset_config,
        dataset_paths,
        datasets,
        df,
        dfs,
        idx,
        lr_schedule,
        model_name,
        model_path,
        model_paths,
        negative_mining,
        pos_semipos,
        results,
    )


@app.cell
def _(df, mo, seaborn):
    df["plot_label"] = [str(x) for x in df["model"]]
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use("ggplot")
    seaborn.displot(data=df, kind='ecdf', x='k_value', col='dataset', hue='plot_label')
    plt.suptitle("top K CDF by model")
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return matplotlib, plt


@app.cell
def _(Path, itertools, pd, torch):
    # Plot statistics about path evaluation

    def process_path(path: Path):
        error_path = path / 'error.pt'
        error_data = torch.load(error_path)
        return error_data[-1].item()

    def process_eval_results(path: Path):
        out = []
        for p in sorted(path.glob("[0-9]*")):
            out.append({
                'path_idx': int(p.stem),
                'final_error': process_path(p),
                'model': path.stem,
            })
        return pd.DataFrame.from_records(out)

    base_path = Path('/data/overhead_matching/evaluation/results/20250616_8_way_experiment')

    path_dfs = []
    for _lr_schedule, _negative_mining, _pos_semipos in itertools.product(*[[False, True]]*3):
        _model_name = f"all_chicago_lr_schedule_{_lr_schedule}_negative_mining_{_negative_mining}_pos_semipos_{_pos_semipos}"
        path_dfs.append(process_eval_results(base_path / _model_name))

    path_df = pd.concat(path_dfs)
    return base_path, path_df, path_dfs, process_eval_results, process_path


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.displot(data=path_df, x='final_error', kind='ecdf', hue='model')
    plt.title("Final Error CDF across 100 paths in New York")
    plt.tight_layout()
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


if __name__ == "__main__":
    app.run()
