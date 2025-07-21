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
    import json
    import msgspec

    import re

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use("ggplot")

    from experimental.overhead_matching.swag.data import vigor_dataset, satellite_embedding_database
    from experimental.overhead_matching.swag.evaluation import evaluate_swag
    from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding

    from common.torch import load_and_save_models
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
        haversine,
        importlib,
        itertools,
        json,
        load_and_save_models,
        math,
        matplotlib,
        mo,
        msgspec,
        np,
        patch_embedding,
        pd,
        plt,
        pprint,
        re,
        satellite_embedding_database,
        seaborn,
        swag_patch_embedding,
        torch,
        vigor_dataset,
    )


@app.cell
def _(
    Path,
    evaluate_swag,
    json,
    load_and_save_models,
    msgspec,
    patch_embedding,
    swag_patch_embedding,
    torch,
    vigor_dataset,
):
    def load_model(path, device='cuda'):
        print(path)
        try:
            model = load_and_save_models.load_model(path, device=device)
            model.patch_dims
            model.model_input_from_batch
        except Exception as e:
            print("Failed to load model", e)
            training_config_path = path.parent / "config.json"
            training_config_json = json.loads(training_config_path.read_text())
            model_config_json = training_config_json["sat_model_config"] if 'satellite' in path.name else training_config_json["pano_model_config"]
            config = msgspec.json.decode(json.dumps(model_config_json), type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)

            model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
            model_type = patch_embedding.WagPatchEmbedding if isinstance(config, patch_embedding.WagPatchEmbeddingConfig) else swag_patch_embedding.SwagPatchEmbedding
            model = model_type(config)
            model.load_state_dict(model_weights)
            model = model.to(device)
        return model

    def get_top_k_results(model_partial_path, dataset_path):
        sat_model = load_model(Path(f"{model_partial_path}_satellite"), device='cuda')
        pano_model = load_model(Path(f"{model_partial_path}_panorama"), device='cuda')

        dataset_config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=1e-6,
            satellite_patch_size=sat_model.patch_dims,
            panorama_size=pano_model.patch_dims,
            factor=1.0
        )
        dataset = vigor_dataset.VigorDataset(dataset_path, dataset_config)

        df, all_similarity = evaluate_swag.evaluate_prediction_top_k(sat_model=sat_model, pano_model=pano_model, dataset=dataset)
        del sat_model, pano_model, all_similarity
        return df
    return get_top_k_results, load_model


@app.cell
def _(Path, pprint):
    model_paths = {}
    idx=59
    # _base_path = Path("/data/overhead_matching/models/all_chicago_sat_embedding_pano_wag")
    # for _p in sorted(_base_path.iterdir()):
    #     model_paths[_p.name] = _p / f"{idx:04d}"

    model_paths = {
        "all_chicago_sat_embedding_pano_wag": Path("/data/overhead_matching/models/all_chicago_sat_embedding_pano_wag/0059"),
        "all_chicago_dino_project_1024": Path("/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_1024/0059"),
        "all_chicago_sat_dino_embedding_mat_pano_dino_sam": Path("/data/overhead_matching/models/20250719_swag_model/all_chicago_sat_dino_embedding_mat_pano_dino_sam/0059")
    }

    pprint(model_paths)

    dataset_paths = {
        'chicago': Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        # 'sanfrancisco': '/data/overhead_matching/datasets/VIGOR/SanFrancisco'
        "newyork": Path('/data/overhead_matching/datasets/VIGOR/NewYork')
    }
    return dataset_paths, idx, model_paths


@app.cell
def _(dataset_paths, get_top_k_results, itertools, model_paths, pd):
    dfs = []
    for (model_name, model_path), (data_name, _dataset_path) in itertools.product(
            model_paths.items(), dataset_paths.items()):
        print(model_name, data_name)
        df = get_top_k_results(model_path, _dataset_path)
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
def _(Path, pd, torch):
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

    # _base_path = Path('/data/overhead_matching/evaluation/results/20250707_dino_features')

    # _results_paths = sorted(_base_path.iterdir())
    _results_paths = [
        Path('/data/overhead_matching/evaluation/results/all_chicago_sat_embedding_pano_wag'),
        Path('/data/overhead_matching/evaluation/results/20250707_dino_features/all_chicago_dino_project_512'),
        Path('/data/overhead_matching/evaluation/results/20250719_swag_model/all_chicago_sat_dino_embedding_mat_pano_dino_sam'),
    ]

    path_dfs = []
    for _p in _results_paths:
        path_dfs.append(process_eval_results(_p))
    # for _lr_schedule, _negative_mining, _pos_semipos in itertools.product(*[[False, True]]*3):
    #     _model_name = f"all_chicago_lr_schedule_{_lr_schedule}_negative_mining_{_negative_mining}_pos_semipos_{_pos_semipos}"
    #     path_dfs.append(process_eval_results(base_path / _model_name))

    path_df = pd.concat(path_dfs)
    return path_df, path_dfs, process_eval_results, process_path


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.displot(data=path_df, x='final_error_m', kind='ecdf', hue='model', palette="tab10")
    plt.title("Final Error CDF across 100 paths in New York")
    plt.tight_layout()
    plt.xlim(-1, 50)
    plt.ylim(-0.05, 1.05)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.scatterplot(data=path_df, x="final_error_m", y="var_sq_m", hue="model", palette="Paired")
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
def _(Path, evaluate_swag, load_model, vigor_dataset):
    def get_similarity_matrix(model_path: Path, dataset_path: Path):
        sat_model = load_model(Path(f"{model_path}_satellite"))
        pano_model = load_model(Path(f"{model_path}_panorama"))
        dataset_config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=1e-6,
            satellite_patch_size=sat_model.patch_dims,
            panorama_size=pano_model.patch_dims,
            factor=1.0,
            panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                dataset_key=dataset_path.name,
                model_type="panorama",
                hash_and_key=pano_model.cache_info()),
            satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                dataset_key=dataset_path.name,
                model_type="satellite",
                hash_and_key=sat_model.cache_info())
        )
        dataset = vigor_dataset.VigorDataset(dataset_path, dataset_config)
        return dataset, evaluate_swag.compute_cached_similarity_matrix(sat_model, pano_model, dataset, device='cuda', use_cached_similarity=True)
    return (get_similarity_matrix,)


@app.cell
def _(dataset_paths, get_similarity_matrix, model_paths, pd, torch):
    _sim_dfs = []
    for _model_name, _model_path in model_paths.items():
        print(_model_name)
        dataset, all_similarity = get_similarity_matrix(
            model_paths[_model_name],
            dataset_paths["newyork"])
        _max_sims = torch.max(all_similarity, dim=1).values.cpu()
        _positive_sims = all_similarity[torch.arange(len(dataset._panorama_metadata)), dataset._panorama_metadata["satellite_idx"]]
        _sim_df = pd.DataFrame(_max_sims.cpu() - _positive_sims.cpu(), columns=["sim_diff_from_max"])
        _sim_df["model_name"] = _model_name
        _sim_dfs.append(_sim_df)
    sim_df = pd.concat(_sim_dfs)
    return all_similarity, dataset, sim_df


@app.cell
def _(mo, np, plt, seaborn, sim_df):
    def compute_approx(sigma):
        _x = np.linspace(0, 1.5, 1000)
        _dx = _x[1]
        _sigma = 0.1
        _y =  np.exp(-_x**2 / (2.0 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        _y_sum = np.cumsum(_y) * _dx * 2.0
        return _x, _y_sum, _y



    plt.figure()

    seaborn.displot(data=sim_df, kind='ecdf', x='sim_diff_from_max', hue='model_name', palette='tab10')
    for _sigma in [0.03, 0.1, 0.2, 0.3, 0.4, 0.5]:
        _x, _approx_sum, _approx = compute_approx(_sigma)
        plt.plot(_x, _approx_sum, label=rf'$\sigma$={_sigma}', linestyle='--')

    plt.legend(loc='upper right')
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return (compute_approx,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
