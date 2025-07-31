import marimo

__generated_with = "0.11.9"
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
        sed,
        spe,
        torch,
        vd,
    )


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    return matplotlib, plt


@app.cell
def _():
    return


@app.cell
def _(json, load_and_save_models, model_path, msgspec, pe, spe, torch):
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

    sat_model = load_model(model_path / "0059_satellite").cuda()
    pano_model = load_model(model_path / "0059_panorama").cuda()
    return load_model, pano_model, sat_model


@app.cell
def _(Path):
    dataset_paths = [
        Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        Path('/data/overhead_matching/datasets/VIGOR/NewYork'),
        Path('/data/overhead_matching/datasets/VIGOR/SanFrancisco'),
        Path('/data/overhead_matching/datasets/VIGOR/Seattle'),
    ]

    checkpoints = [0, 10, 20, 30, 40, 50, 59]

    model_paths = [
        Path('/data/overhead_matching/models/20250719_swag_model/all_chicago_sat_dino_pano_dino_agg_small_attn_8'),
        Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512'),
    ]




    return checkpoints, dataset_paths, model_paths


@app.cell
def _(checkpoints, dataset_paths, es, itertools, load_model, model_paths, vd):
    for _dataset_path, _model_path in itertools.product(dataset_paths, model_paths):
        for _checkpoint in checkpoints:
            _sat_model = load_model(_model_path / f"{_checkpoint:04d}_satellite")
            _pano_model = load_model(_model_path / f"{_checkpoint:04d}_panorama")
            _dataset = vd.VigorDataset(
                _dataset_path,
                config=vd.VigorDatasetConfig(
                    satellite_tensor_cache_info=vd.TensorCacheInfo(
                        dataset_key=_dataset_path.name,
                        model_type="satellite",
                        hash_and_key=_sat_model.cache_info()),
                    panorama_tensor_cache_info=vd.TensorCacheInfo(
                        dataset_key=_dataset_path.name,
                        model_type="panorama",
                        hash_and_key=_pano_model.cache_info()),
                    satellite_patch_size=_sat_model.patch_dims,
                    panorama_size=_pano_model.patch_dims))

            sim_matrix = es.compute_cached_similarity_matrix(
                sat_model=_sat_model,
                pano_model=_pano_model,
                dataset=_dataset,
                device='cuda:0',
                use_cached_similarity=True)
    return (sim_matrix,)


@app.cell
def _(
    checkpoints,
    dataset_paths,
    es,
    itertools,
    load_model,
    model_paths,
    torch,
    vd,
):
    hist_counts = {}

    for _dataset_path, _model_path in itertools.product(dataset_paths, model_paths):
        _sim_matrices = []
        for _checkpoint in checkpoints:
            _sat_model = load_model(_model_path / f"{_checkpoint:04d}_satellite")
            _pano_model = load_model(_model_path / f"{_checkpoint:04d}_panorama")
            _dataset = vd.VigorDataset(
                _dataset_path,
                config=vd.VigorDatasetConfig(
                    satellite_tensor_cache_info=vd.TensorCacheInfo(
                        dataset_key=_dataset_path.name,
                        model_type="satellite",
                        hash_and_key=_sat_model.cache_info()),
                    panorama_tensor_cache_info=vd.TensorCacheInfo(
                        dataset_key=_dataset_path.name,
                        model_type="panorama",
                        hash_and_key=_pano_model.cache_info()),
                    satellite_patch_size=_sat_model.patch_dims,
                    panorama_size=_pano_model.patch_dims))

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

            hist_counts[(_dataset_path.name, _model_path.name, _checkpoint)] = {
                'positive': _pos_counts,
                'negative': _neg_counts,
                'semipositive': _semipos_counts,
            }

    return bin_edges, hist_counts


@app.cell
def _(plt):

    def plot_hist(bin_edges, pos_counts, semipos_counts, neg_counts):
        plt.bar(bin_edges[:-1], pos_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='pos', alpha=0.5)
        plt.bar(bin_edges[:-1], semipos_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='semipos', alpha=0.5)
        plt.bar(bin_edges[:-1], neg_counts, align='edge', width=bin_edges[1]-bin_edges[0], label='neg', alpha=0.5)


    return (plot_hist,)


@app.cell
def _(
    bin_edges,
    checkpoints,
    dataset_paths,
    hist_counts,
    itertools,
    model_paths,
    plot_hist,
    plt,
):
    for _dataset_path in dataset_paths:
        plt.figure(figsize=(12, 6))
        for _i, (_model_path, _checkpoint) in enumerate(itertools.product(model_paths, checkpoints)):
            plt.subplot(2, 7, _i+1)
            counts = hist_counts[(_dataset_path.name, _model_path.name, _checkpoint)]
            plot_hist(bin_edges, counts['positive'], counts["semipositive"], counts["negative"])
            if _i < 7:
                plt.title(f'Epoch: {_checkpoint}')
            if _i % 7 > 0:
                plt.gca().tick_params(labelleft=False, axis='y', length=0.0)
            if _i % 7 == 0:
                plt.ylabel(_model_path.name[12:])
            plt.ylim(0, 6)
            plt.suptitle(_dataset_path.name)
        plt.tight_layout()
    plt.show()
    # mo.mpl.interactive(plt.gcf())
    return (counts,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
