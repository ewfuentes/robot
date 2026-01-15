import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import json
    import msgspec
    from experimental.overhead_matching.swag.model import (
        patch_embedding as pe,
        swag_patch_embedding as spe,
    )
    from common.torch import load_and_save_models
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd,
        satellite_embedding_database as sed,
    )
    from experimental.overhead_matching.swag.evaluation import (
        evaluate_swag as es
    )
    import itertools
    import torch
    return (
        Path,
        es,
        itertools,
        json,
        load_and_save_models,
        mo,
        msgspec,
        pe,
        spe,
        torch,
        vd,
    )


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    return (plt,)


@app.cell
def _():
    return


@app.cell
def _(json, load_and_save_models, msgspec, pe, spe, torch):
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
            print(model_config_json)
            config = msgspec.json.decode(json.dumps(model_config_json), type=pe.WagPatchEmbeddingConfig | spe.SwagPatchEmbeddingConfig)

            model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
            model_type = pe.WagPatchEmbedding if isinstance(config, pe.WagPatchEmbeddingConfig) else spe.SwagPatchEmbedding
            model = model_type(config)
            model.load_state_dict(model_weights)
            model = model.to(device)
        return model
    return (load_model,)


@app.cell
def _(Path):
    dataset_paths = [
        Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        Path('/data/overhead_matching/datasets/VIGOR/NewYork'),
        Path('/data/overhead_matching/datasets/VIGOR/SanFrancisco'),
        Path('/data/overhead_matching/datasets/VIGOR/Seattle'),
    ]

    model_paths = [
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5/'),
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_attn_8/'),
        Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512'),
    ]
    return dataset_paths, model_paths


@app.cell
def _(dataset_paths, es, itertools, load_model, model_paths, vd):

    def get_max_checkpoint(path):
        dirs = path.glob("*_satellite")
        return max([int(x.name.split('_')[0]) for x in dirs])

    for _dataset_path, _model_path in itertools.product(dataset_paths, model_paths):
        _checkpoint = get_max_checkpoint(_model_path)
        _sat_model = load_model(_model_path / f"{_checkpoint:04d}_satellite")
        _pano_model = load_model(_model_path / f"{_checkpoint:04d}_panorama")
        _landmark_version = "v3"
        _dataset = vd.VigorDataset(
            _dataset_path,
            config=vd.VigorDatasetConfig(
                satellite_tensor_cache_info=vd.TensorCacheInfo(
                    dataset_keys=[_dataset_path.name],
                    model_type="satellite",
                    landmark_version=_landmark_version,
                    extractor_info=_sat_model.cache_info()),
                panorama_tensor_cache_info=vd.TensorCacheInfo(
                    dataset_keys=[_dataset_path.name],
                    model_type="panorama",
                    landmark_version=_landmark_version,
                    extractor_info=_pano_model.cache_info()),
                satellite_patch_size=_sat_model.patch_dims,
                panorama_size=_pano_model.patch_dims,
                landmark_version=_landmark_version))

        sim_matrix = es.compute_cached_similarity_matrix(
            sat_model=_sat_model,
            pano_model=_pano_model,
            dataset=_dataset,
            device='cuda:0',
            use_cached_similarity=True)
    return (get_max_checkpoint,)


@app.cell
def _(
    dataset_paths,
    es,
    get_max_checkpoint,
    itertools,
    load_model,
    model_paths,
    torch,
    vd,
):
    hist_counts = {}

    for _dataset_path, _model_path in itertools.product(dataset_paths, model_paths):
        _sim_matrices = []
        _checkpoint = get_max_checkpoint(_model_path)
        _sat_model = load_model(_model_path / f"{_checkpoint:04d}_satellite")
        _pano_model = load_model(_model_path / f"{_checkpoint:04d}_panorama")
        _landmark_version = "v3"
        _dataset = vd.VigorDataset(
            _dataset_path,
            config=vd.VigorDatasetConfig(
                satellite_tensor_cache_info=vd.TensorCacheInfo(
                    dataset_keys=[_dataset_path.name],
                    model_type="satellite",
                    landmark_version=_landmark_version,
                    extractor_info=_sat_model.cache_info()),
                panorama_tensor_cache_info=vd.TensorCacheInfo(
                    dataset_keys=[_dataset_path.name],
                    model_type="panorama",
                    landmark_version=_landmark_version,
                    extractor_info=_pano_model.cache_info()),
                satellite_patch_size=_sat_model.patch_dims,
                panorama_size=_pano_model.patch_dims,
                landmark_version=_landmark_version))

        _sim_matrix = es.compute_cached_similarity_matrix(
            sat_model=_sat_model,
            pano_model=_pano_model,
            dataset=_dataset,
            device='cuda:0',
            use_cached_similarity=True).cpu()

        _pos_mask = torch.zeros_like(_sim_matrix, dtype=torch.bool)
        _semipos_mask = torch.zeros_like(_pos_mask)
        for _pano_idx, _row in _dataset._panorama_metadata.iterrows():
            _pos_sat_idxs = _row.positive_satellite_idxs
            _semipos_sat_idxs = _row.semipositive_satellite_idxs
            _pos_mask[_pano_idx, _pos_sat_idxs] = True
            _semipos_mask[_pano_idx, _semipos_sat_idxs] = True
        _neg_mask = torch.logical_not(_pos_mask | _semipos_mask)

        _pos_similarities = _sim_matrix[..., _pos_mask]
        _semipos_similarities = _sim_matrix[..., _semipos_mask]
        _neg_similarities = _sim_matrix[..., _neg_mask]

        _neg_counts, bin_edges = torch.histogram(_neg_similarities, range=(-1, 1), bins=200, density=True)
        _semipos_counts, bin_edges = torch.histogram(_semipos_similarities, range=(-1, 1), bins=200, density=True)
        _pos_counts, bin_edges = torch.histogram(_pos_similarities, range=(-1, 1), bins=200, density=True)

        hist_counts[(_dataset_path.name, _model_path.name)] = {
            'positive': _pos_counts,
            'negative': _neg_counts,
            'semipositive': _semipos_counts,
        }
    return bin_edges, hist_counts


@app.cell
def _(plt):
    def plot_hist(bin_edges, pos_counts, semipos_counts, neg_counts):
        plt.bar(bin_edges[:-1], pos_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='pos', alpha=0.33)
        plt.bar(bin_edges[:-1], semipos_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='semipos', alpha=0.33)
        plt.bar(bin_edges[:-1], neg_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='neg', alpha=0.33)
    return (plot_hist,)


@app.cell
def _(
    Path,
    bin_edges,
    dataset_paths,
    hist_counts,
    itertools,
    mo,
    model_paths,
    plot_hist,
    plt,
):
    num_models = len(model_paths)

    _dataset_paths = [
        Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        Path('/data/overhead_matching/datasets/VIGOR/NewYork'),
        Path('/data/overhead_matching/datasets/VIGOR/SanFrancisco'),
        Path('/data/overhead_matching/datasets/VIGOR/Seattle'),
    ]

    plt.figure(figsize=(12, 3 * num_models))
    for _i, (_model_path, _dataset_path) in enumerate(itertools.product(model_paths, _dataset_paths)):
        print(_dataset_path.name, _model_path.name)
        plt.subplot(num_models, len(dataset_paths), _i+1)
        counts = hist_counts[(_dataset_path.name, _model_path.name)]
        plot_hist(bin_edges, counts['positive'], counts["semipositive"], counts["negative"])
        if _i < len(_dataset_paths):
            plt.title(_dataset_path.name)
    
        if _i % len(_dataset_paths) > 0:
            plt.gca().tick_params(labelleft=False, axis='y', length=0.0)
        if _i % len(_dataset_paths) == 0:
            _model_parts = _model_path.name.split('_')
            _label = [_model_parts[2]]
            for _p in _model_parts[3:]:
                if len(_label[-1] + '_' + _p) < 30:
                    _label[-1] += '_' + _p
                else:
                    _label.append(_p)
            _label = '\n'.join(_label)
            plt.ylabel(_label)
        plt.ylim(0, 6)
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
