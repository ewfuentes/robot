import marimo

__generated_with = "0.14.17"
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
        Path,
        WagConfig,
        evaluate_swag,
        haversine,
        itertools,
        json,
        load_and_save_models,
        mo,
        msgspec,
        np,
        patch_embedding,
        pd,
        plt,
        pprint,
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
            print("Creating model with config:", config)
            model = model_type(config)
            model.load_state_dict(model_weights)
            model = model.to(device)
        return model

    def get_top_k_results(model_partial_path, dataset_path):
        sat_model = load_model(Path(f"{model_partial_path}_satellite"), device='cuda')
        pano_model = load_model(Path(f"{model_partial_path}_panorama"), device='cuda')

        dataset_config = vigor_dataset.VigorDatasetConfig(
            satellite_patch_size=sat_model.patch_dims,
            panorama_size=pano_model.patch_dims,
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
    _base = Path("/data/overhead_matching/models")
    _fixed_base = _base / "20250806_swag_model_fixed"
    model_paths = {
        "all_chicago_dino_project_512": _base / "20250707_dino_features/all_chicago_dino_project_512",
        "all_chicago_sat_dino_pano_dino_agg_small_attn_8": _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_attn_8',
        "all_chicago_sat_dino_pano_dino_agg_small_batch_size_128": _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_batch_size_128',
        "all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5": _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5',
        "all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20": _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20',
        "all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128": _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128',
        "all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5": _fixed_base / "all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5",
        "all_chicago_sat_dino_embedding_mat_pano_dino_sam": _base / "20250815_swag_semantic" / "all_chicago_sat_dino_embedding_mat_pano_dino_sam",
    }

    """
    drwxrwxr-x 17 erick erick 4096 Aug  6 20:06 all_chicago_sat_dino_pano_dino_agg_small_attn_8
    drwxrwsr-x 17 erick erick 4096 Aug  6 19:50 all_chicago_sat_dino_pano_dino_agg_small_attn_8_layers_1
    drwxrwxr-x 17 erick erick 4096 Aug  7 18:22 all_chicago_sat_dino_pano_dino_agg_small_batch_size_128
    drwxrwxr-x 17 erick erick 4096 Aug  7 19:55 all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5
    drwxrwxr-x 17 erick erick 4096 Aug  7 17:34 all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20
    drwxrwxr-x 17 erick erick 4096 Aug  7 18:54 all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128
    """

    pprint(model_paths)

    dataset_paths = {
        'chicago': Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        # 'sanfrancisco': '/data/overhead_matching/datasets/VIGOR/SanFrancisco'
        "newyork": Path('/data/overhead_matching/datasets/VIGOR/NewYork')
    }
    return dataset_paths, model_paths


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
    return (df,)


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
                    'location': 'NewYork',
                    'path_idx': int(p.stem),
                    'final_error_m': final_error_m,
                    'var_sq_m': var_sq_m,
                    'model': p.parts[-2],
                })
            except:
                ...
        return pd.DataFrame.from_records(out)

    # _base_path = Path('/data/overhead_matching/evaluation/results/20250707_dino_features')

    # _results_paths = sorted(_base_path.iterdir())
    _results_paths = [
        Path('/data/overhead_matching/evaluation/results/20250707_dino_features/NewYork/all_chicago_dino_project_512/'),

        # Path('/data/overhead_matching/evaluation/results/20250719_swag_model/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8'),
        Path('/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8'),
        # Path('/data/overhead_matching/evaluation/results/20250719_swag_model/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8_avg_neg_0p25'),
        # Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8_layers_1/"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_batch_size_128/"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5/"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20/"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128/"),
        Path("/data/overhead_matching/evaluation/results/20250806_swag_model_fixed/NewYork/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5/"),
    ]

    path_dfs = []
    for _p in _results_paths:
        path_dfs.append(process_eval_results(_p))

    path_df = pd.concat(path_dfs)
    return (path_df,)


@app.cell
def _(path_df):
    path_df
    return


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.displot(data=path_df, x='final_error_m', kind='ecdf', hue='model', palette="tab10", col='location')
    plt.suptitle("Final Error CDF across 100 paths")
    plt.tight_layout()
    # plt.xlim(-1, 3
    plt.ylim(-0.05, 1.05)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, path_df, plt, seaborn):
    seaborn.scatterplot(data=path_df, x="final_error_m", y="var_sq_m", hue="model", palette="Paired")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(Path, evaluate_swag, load_model, vigor_dataset):
    def get_similarity_matrix(model_path: Path, dataset_path: Path, checkpoint_idx=59):
        sat_model = load_model(model_path / f"{checkpoint_idx:04d}_satellite")
        pano_model = load_model(model_path / f"{checkpoint_idx:04d}_panorama")
        dataset_config = vigor_dataset.VigorDatasetConfig(
            satellite_patch_size=sat_model.patch_dims,
            panorama_size=pano_model.patch_dims,
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
def _(Path, model_paths):
    def get_max_checkpoint(model_path: Path):
        checkpoints = [int(x.name.split('_')[0]) for x in model_path.glob("*_satellite")]
        return max(checkpoints)

    get_max_checkpoint(model_paths['all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5'])

    return (get_max_checkpoint,)


@app.cell
def _(
    dataset_paths,
    get_max_checkpoint,
    get_similarity_matrix,
    model_paths,
    pd,
    torch,
):
    _sim_dfs = []
    for _model_name, _model_path in model_paths.items():
        print(_model_name)
        dataset, all_similarity = get_similarity_matrix(
            model_paths[_model_name],
            dataset_paths["newyork"],
            checkpoint_idx=get_max_checkpoint(model_paths[_model_name])
        )
        _max_sims = torch.max(all_similarity, dim=1).values.cpu()
        _positive_sims = all_similarity[torch.arange(len(dataset._panorama_metadata)), dataset._panorama_metadata["satellite_idx"]]
        _sim_df = pd.DataFrame(_max_sims.cpu() - _positive_sims.cpu(), columns=["sim_diff_from_max"])
        _sim_df["model_name"] = _model_name
        _sim_dfs.append(_sim_df)
    sim_df = pd.concat(_sim_dfs)
    return (sim_df,)


@app.cell
def _(np):
    def compute_approx(sigma):
        _x = np.linspace(0, 1.5, 10000)
        _dx = _x[1]
        _y =  np.exp(-_x**2 / (2.0 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        _y_sum = np.cumsum(_y) * _dx * 2.0
        return _x, _y_sum, _y
    return (compute_approx,)


@app.cell
def _(sim_df):
    sim_df
    return


@app.cell
def _(compute_approx, mo, plt, seaborn, sim_df):
    plt.figure()

    seaborn.displot(data=sim_df, kind='ecdf', x='sim_diff_from_max', hue='model_name', palette='tab10')
    for _sigma in [0.03, 0.1, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5]:
        _x, _approx_sum, _approx = compute_approx(_sigma)
        plt.plot(_x, _approx_sum, label=rf'$\sigma$={_sigma}', linestyle='--')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('/tmp/sigma.png')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(r"""# Compute Probability Mass in Region""")
    return


@app.cell
def _(Path, WagConfig, evaluate_swag, haversine, json, torch, vigor_dataset):
    import experimental.overhead_matching.swag.scripts.evaluate_model_on_paths as emop
    import experimental.overhead_matching.swag.filter.particle_filter as pf
    from google.protobuf import text_format
    import tqdm

    def compute_particle_history(path_dir: Path, dataset: vigor_dataset.VigorDataset):
        path_eval_args = json.loads((path_dir.parent / "args.json").read_text())
        aux_info = json.loads((path_dir / "other_info.json").read_text())

        gt_path_pano_indices = torch.load(path_dir / "path.pt")

        # dataset, sat_model, pano_model, paths_data = emop.construct_path_eval_inputs_from_args(
        #     sat_model_path=path_eval_args["sat_path"],
        #     pano_model_path=path_eval_args["pano_path"],
        #     dataset_path=path_eval_args["dataset_path"],
        #     paths_path=path_eval_args["paths_path"],
        #     panorama_neighbor_radius_deg=path_eval_args["panorama_neighbor_radius_deg"],
        #     device="cuda:0")

        wag_config = text_format.Parse((path_dir.parent / "wag_config.pbtxt").read_text(), WagConfig())

        path_similarity_values = torch.load(path_dir / "similarity.pt")

        inference_result = evaluate_swag.construct_inputs_and_evaluate_path(
            device="cuda:0",
            generator_seed=aux_info["seed"],
            path=gt_path_pano_indices,
            vigor_dataset=dataset,
            path_similarity_values=path_similarity_values,
            wag_config=wag_config,
            return_intermediates=False)

        obs_likelihood = pf.wag_observation_log_likelihood_from_similarity_matrix(
            path_similarity_values, wag_config.sigma_obs_prob_from_sim)

        return gt_path_pano_indices, path_similarity_values, obs_likelihood, inference_result.particle_history

    def compute_particle_mass_near_robot(panorama_positions, particle_history, radius_m):
        dist_m = haversine.find_d_on_unit_circle(panorama_positions.unsqueeze(1), particle_history) * vigor_dataset.EARTH_RADIUS_M
        is_in_range = dist_m < radius_m
        return torch.sum(is_in_range, axis=-1) / particle_history.shape[1]

    def process_path(path_dir: Path, dataset: vigor_dataset.VigorDataset):
        pano_indices, path_similarity, obs_likelihood, particle_history = compute_particle_history(path_dir, dataset)
        particle_history = particle_history.cuda()
        panorama_positions = dataset.get_panorama_positions(pano_indices).cuda()
        particle_mass_near_robot = compute_particle_mass_near_robot(panorama_positions, particle_history, radius_m=50.0)

        return particle_mass_near_robot, path_similarity, obs_likelihood, pano_indices


    root_dir = Path('/data/overhead_matching/evaluation/results/20250719_swag_model/NewYork/all_chicago_sat_dino_pano_dino')

    _path_eval_args = json.loads((root_dir / "args.json").read_text())
    path_dataset = vigor_dataset.VigorDataset(
        _path_eval_args["dataset_path"],
        vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None))

    for path_idx in tqdm.tqdm(range(3, 4)):
        particle_mass_near_robot, path_similarity, obs_likelihood, pano_indices = process_path(root_dir / f"{path_idx:07d}", path_dataset)

        # plt.figure(figsize=(12, 6))
        # plt.plot(particle_mass_near_robot.cpu().numpy(), label='prob. mass near robot')
        # plt.plot(torch.exp(obs_likelihood).cpu().numpy(), label='obs likelihood near robot')
        # plt.yscale('log')
        # plt.title(f'path {path_idx:07d}')
        # plt.legend()
        # plt.xlim(0, 50)
        # plt.savefig(f'/tmp/{path_idx:07d}.png')
        # plt.close()
    # mo.mpl.interactive(plt.gcf())
    return (root_dir,)


@app.cell
def _(Path, get_similarity_matrix, json, root_dir):
    def process_root(root_path: Path):
        path_eval_args = json.loads((root_path / "args.json").read_text())
        return get_similarity_matrix(Path(path_eval_args["sat_path"]).parent / "0059", Path(path_eval_args["dataset_path"]))

    sim_dataset, similarity_matrix = process_root(root_dir)
    return sim_dataset, similarity_matrix


@app.cell
def _(np, sim_dataset):
    pos_mask = np.zeros((len(sim_dataset._panorama_metadata), len(sim_dataset._satellite_metadata)), dtype=bool)
    semipos_mask = np.zeros((len(sim_dataset._panorama_metadata), len(sim_dataset._satellite_metadata)), dtype=bool)
    for _idx, _row in sim_dataset._panorama_metadata.iterrows():
        pos_idxs = _row.positive_satellite_idxs
        semipos_idxs = _row.semipositive_satellite_idxs
        pos_mask[_idx, pos_idxs] = True
        semipos_mask[_idx, semipos_idxs] = True


    pos_semipos_mask = np.logical_or(pos_mask, semipos_mask)
    neg_mask = np.logical_not(pos_semipos_mask)
    return neg_mask, pos_mask, pos_semipos_mask, semipos_mask


@app.cell
def _(
    neg_mask,
    pos_mask,
    pos_semipos_mask,
    semipos_mask,
    similarity_matrix,
    torch,
):
    sigma = 0.2

    sim_matrix = (torch.max(similarity_matrix, dim=-1).values.unsqueeze(-1) - similarity_matrix).cpu()
    # sim_matrix = 1 - similarity_matrix.cpu()

    pos_values = sim_matrix[pos_mask]
    semipos_values = sim_matrix[semipos_mask]
    pos_semipos_values = sim_matrix[pos_semipos_mask]
    neg_values = sim_matrix[neg_mask][::10000]
    return neg_values, pos_semipos_values


@app.cell
def _(neg_values):
    neg_values.shape
    return


@app.cell
def _(bins, compute_approx, mo, neg_values, plt, pos_semipos_values):
    _x, _y_sum, _y = compute_approx(0.17)



    plt.figure()
    # plt.hist(pos_values, density=True, bins=100, alpha=0.5, label='positive')
    # plt.hist(semipos_values, density=True, bins=100, alpha=0.5, label='semipos')
    plt.hist(pos_semipos_values, density=True, bins=bins, alpha=0.5, label='pos_semipos')
    plt.hist(neg_values, density=True, bins=bins, alpha=0.5, label='negative')
    plt.plot(_x, _y)
    plt.legend()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(Path, get_similarity_matrix, itertools, mo, np, plt, torch):
    _model_paths = [
        Path("/data/overhead_matching/models/20250719_swag_model/all_chicago_sat_dino_pano_dino_agg_small_attn_8"),
        Path("/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512"),
        Path("/data/overhead_matching/models/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5")
    ]

    _dataset_paths = [
        Path("/data/overhead_matching/datasets/VIGOR/Chicago"),
        Path("/data/overhead_matching/datasets/VIGOR/NewYork"),
        Path("/data/overhead_matching/datasets/VIGOR/SanFrancisco"),
        Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    ]

    _bins = np.linspace(0, 2, 100)

    plt.figure(figsize=(16, 8))
    for _plot_idx, (_mp, _dp) in enumerate(itertools.product(_model_paths, _dataset_paths)):

        print(_mp, _dp)
        _dataset, _sim_matrix = get_similarity_matrix(_mp / "0059", _dp)
        _pos_mask = np.zeros((len(_dataset._panorama_metadata), len(_dataset._satellite_metadata)), dtype=bool)
        _semipos_mask = np.zeros((len(_dataset._panorama_metadata), len(_dataset._satellite_metadata)), dtype=bool)
        for _idx, _row in _dataset._panorama_metadata.iterrows():
            _pos_idxs = _row.positive_satellite_idxs
            _semipos_idxs = _row.semipositive_satellite_idxs
            _pos_mask[_idx, _pos_idxs] = True
            _semipos_mask[_idx, _semipos_idxs] = True

        _sim_matrix = (torch.max(_sim_matrix, dim=-1).values.unsqueeze(-1) - _sim_matrix).cpu()

        _pos_semipos_mask = np.logical_or(_pos_mask, _semipos_mask)
        _neg_mask = np.logical_not(_pos_semipos_mask)

        _pos_values = _sim_matrix[_pos_mask]
        _semipos_values = _sim_matrix[_semipos_mask]
        _pos_semipos_values = _sim_matrix[_pos_semipos_mask]
        _neg_values = _sim_matrix[_neg_mask][::10000]
        plt.subplot(2, 4, _plot_idx+1)
        if _plot_idx < 4:
            plt.title(f"{_dp.name}")
        if _plot_idx % 4 == 0:
            plt.ylabel(f"{_mp.name[12:]}")

        plt.hist(_pos_semipos_values, density=True, bins=_bins, alpha=0.5, label='pos_semipos')
        plt.hist(_neg_values, density=True, bins=_bins, alpha=0.5, label='negative')
        plt.ylim(0, 4)
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(Path, get_similarity_matrix, itertools, mo, np, plt):
    _fixed_base = Path("/data/overhead_matching/models/20250806_swag_model_fixed")

    _model_paths = [
        # Path("/data/overhead_matching/models/20250719_swag_model/all_chicago_sat_dino_pano_dino_agg_small_attn_8"),
        Path("/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512"),
        _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_attn_8',
        _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_batch_size_128',
        _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5',
        _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20',
        _fixed_base / 'all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128',
    ]

    _dataset_paths = [
        Path("/data/overhead_matching/datasets/VIGOR/Chicago"),
        Path("/data/overhead_matching/datasets/VIGOR/NewYork"),
        Path("/data/overhead_matching/datasets/VIGOR/SanFrancisco"),
        Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    ]

    _bins = np.linspace(-1, 1, 200)

    plt.figure(figsize=(16, 24))
    for _plot_idx, (_mp, _dp) in enumerate(itertools.product(_model_paths, _dataset_paths)):

        print(_mp, _dp)
        _dataset, _sim_matrix = get_similarity_matrix(_mp, _dp, checkpoint_idx=59)
        _pos_mask = np.zeros((len(_dataset._panorama_metadata), len(_dataset._satellite_metadata)), dtype=bool)
        _semipos_mask = np.zeros((len(_dataset._panorama_metadata), len(_dataset._satellite_metadata)), dtype=bool)
        for _idx, _row in _dataset._panorama_metadata.iterrows():
            _pos_idxs = _row.positive_satellite_idxs
            _semipos_idxs = _row.semipositive_satellite_idxs
            _pos_mask[_idx, _pos_idxs] = True
            _semipos_mask[_idx, _semipos_idxs] = True

        _sim_matrix = _sim_matrix.cpu()

        _pos_semipos_mask = np.logical_or(_pos_mask, _semipos_mask)
        _neg_mask = np.logical_not(_pos_semipos_mask)

        _pos_values = _sim_matrix[_pos_mask]
        _semipos_values = _sim_matrix[_semipos_mask]
        _pos_semipos_values = _sim_matrix[_pos_semipos_mask]
        _neg_values = _sim_matrix[_neg_mask][::10000]
        plt.subplot(len(_model_paths), len(_dataset_paths), _plot_idx+1)
        if _plot_idx < len(_dataset_paths):
            plt.title(f"{_dp.name}")
        if _plot_idx % len(_dataset_paths) == 0:
            _label_parts = _mp.name.split('_')[2:]
            _label = [_label_parts[0]]
            for _p in _label_parts[1:]:
                if len(_label[-1]) + len(_p) > 40:
                    _label.append(_p)
                else:
                    _label[-1] += "_" + _p
            _label = "\n".join(_label)
            plt.ylabel(f"{_label}")

        plt.hist(_pos_values, density=True, bins=_bins, alpha=0.5, label='pos')
        plt.hist(_semipos_values, density=True, bins=_bins, alpha=0.5, label='semipos')
        plt.hist(_neg_values, density=True, bins=_bins, alpha=0.5, label='negative')
        plt.ylim(0, 4)
    plt.legend()
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
