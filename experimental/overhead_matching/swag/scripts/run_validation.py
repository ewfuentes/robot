import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from experimental.overhead_matching.swag.scripts import (
        train as T
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from experimental.overhead_matching.swag.model import (
        patch_embedding as pe,
        swag_patch_embedding as spe
    )
    import common.torch.load_torch_deps
    import torch
    import common.torch.load_and_save_models as lsm
    from pathlib import Path
    import pandas as pd
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json
    import msgspec
    return Path, T, json, lsm, mo, msgspec, pd, pe, plt, sns, spe, torch, vd


@app.cell
def _(Path):
    model_paths = [
        Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512'),
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_attn_8'),
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_batch_size_512_hard_negative_20_lr_3e-4_warmup_5'),
        Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dino_embedding_mat_pano_dino_sam'),
        Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_pano_dinov3'),

    ]

    validation_set_paths = [
        Path("/data/overhead_matching/datasets/VIGOR/Chicago"),
        Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    ]
    return model_paths, validation_set_paths


@app.cell
def _(model_paths):
    model_paths
    return


@app.cell
def _(json, lsm, msgspec, pe, spe, torch):
    def load_model(path, device='cuda'):
        print(path)
        try:
            model = lsm.load_model(path, device=device, skip_consistent_output_check=True)
            model.patch_dims
            model.model_input_from_batch
            if isinstance(model, spe.SwagPatchEmbedding):
                model._extractor_by_name
            print('loading from saved model')
        except Exception as e:
            print("failed to load model, trying to load from weights")
            print(e)
            training_config_path = path.parent / "config.json"
            training_config_json = json.loads(training_config_path.read_text())
            model_config_json = training_config_json["sat_model_config"] if 'satellite' in path.name else training_config_json["pano_model_config"]
            print(model_config_json)
            config = msgspec.json.decode(json.dumps(model_config_json), type=pe.WagPatchEmbeddingConfig | spe.SwagPatchEmbeddingConfig)

            model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
            model_type = (
                pe.WagPatchEmbedding if isinstance(config, pe.WagPatchEmbeddingConfig) else spe.SwagPatchEmbedding)
            model = model_type(config)
            if isinstance(config, spe.SwagPatchEmbeddingConfig):
                to_add = {}
                for k, v in model_weights.items():
                    if k.startswith('_feature_map_extractor'):
                        to_add_key = f"_extractor_by_name._{k}"
                        to_add[to_add_key] = v
                    elif k.startswith('_semantic_token_extractor'):
                        to_add_key = f"_extractor_by_name._{k}"
                        to_add[to_add_key] = v
                    elif k.startswith("_feature_token_projection"):
                        to_add_key = '.'.join(["_projection_by_name", '__feature_map_extractor'] + k.split('.')[1:])
                        to_add[to_add_key] = v
                    elif k.startswith("_semantic_token_projection"):
                        to_add_key = '.'.join(
                            ["_projection_by_name", '__semantic_token_extractor'] + k.split('.')[1:])
                        to_add[to_add_key] = v
                    elif k == "_feature_token_marker":
                        to_add_key = "_token_marker_by_name.__feature_map_extractor"
                        to_add[to_add_key] = v
                    elif k == "_semantic_token_marker":
                        to_add_key = "_token_marker_by_name.__semantic_token_extractor"
                        to_add[to_add_key] = v

                model_weights |= to_add

            model.load_state_dict(model_weights)
            model = model.to(device)
        return model
    return (load_model,)


@app.cell
def _(T, load_model, mo, vd):
    @mo.persistent_cache
    def compute_validation_metrics(model_path, dataset_path, checkpoint, factor, landmark_version="v1"):
        print(f'working on: {model_path.name} {dataset_path.name} {checkpoint}')
        _sat_model = load_model(
            model_path / f"{checkpoint:04d}_satellite", device='cuda:0').eval()
        _pano_model = load_model(
            model_path / f"{checkpoint:04d}_panorama", device='cuda:0').eval()


        _dataset_config = vd.VigorDatasetConfig(
            panorama_tensor_cache_info=vd.TensorCacheInfo(
                dataset_keys=[dataset_path.name],
                model_type="panorama",
                landmark_version=landmark_version,
                extractor_info=_pano_model.cache_info()),
            satellite_tensor_cache_info=vd.TensorCacheInfo(
                dataset_keys=[dataset_path.name],
                model_type="satellite",
                landmark_version=landmark_version,
                extractor_info=_sat_model.cache_info()),
            panorama_size=_pano_model.patch_dims,
            satellite_patch_size=_sat_model.patch_dims,
            factor=factor,
            landmark_version=landmark_version
        )
        _dataset = vd.VigorDataset(dataset_path, _dataset_config)

        return T.compute_validation_metrics(
            sat_model=_sat_model,
            pano_model=_pano_model,
            validation_datasets={"": _dataset})
    return (compute_validation_metrics,)


@app.cell
def _(compute_validation_metrics, model_paths, pd, validation_set_paths):
    _records = []
    FACTOR=0.3
    for _dp in validation_set_paths:
        _dataset = None
        print(f'Dataset: {_dp.name}')
        for _mp in model_paths:
            _checkpoints = [int(x.name.split("_")[0]) for x in sorted(_mp.glob("[0-9]*_satellite"))]
            print(_mp.name, _checkpoints)
            for _checkpoint in _checkpoints:
                _metrics = compute_validation_metrics(_mp, _dp, _checkpoint, FACTOR)
                for k, v in _metrics.items():
                    _records.append({
                        'model': _mp.name,
                        'epoch': _checkpoint,
                        'dataset': _dp.name,
                        'metric': k,
                        'value': v})

    validation_df = pd.DataFrame.from_records(_records)
    return (validation_df,)


@app.cell
def _(validation_df):
    validation_df
    return


@app.cell
def _(mo, plt, sns, validation_df):
    # grid = sns.FacetGrid(to_plot_df, col='metric', col_wrap=4, hue='dataset_and_model')
    # grid.map(sns.lineplot, 'epoch', 'value')
    # grid.set_titles(col_template='{col_name}')
    # grid.add_legend()

    _grid_layout = [
        ['positive_mean_recip_rank', 'max_pos_semi_pos_recip_rank', None, None],
        ['pos_recall@1', 'pos_recall@5', 'pos_recall@10', 'pos_recall@100'],
        ['any pos_semipos_recall@1', 'any pos_semipos_recall@5', 'any pos_semipos_recall@10', 'any pos_semipos_recall@100'],
        [None, 'all pos_semipos_recall@5', 'all pos_semipos_recall@10', 'all pos_semipos_recall@100'],
    ]

    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(16, 16))
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = validation_df["metric"] == f"/{_metric}"
            _to_plot_df = validation_df[_mask]
            if _i == 0 and _j == 0:
                sns.lineplot(_to_plot_df, x='epoch', y='value', hue='model', style='dataset', ax=_ax)    
                legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='epoch', y='value', hue='model', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    print([l.get_label() for l in _axs[0, 0].lines])
    _fig.legend(*legend_objs, loc='upper right')
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(validation_df):
    validation_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
