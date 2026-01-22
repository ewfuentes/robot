import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path

    from common.torch import (
        load_torch_deps,
        load_and_save_models
    )
    from experimental.overhead_matching.swag.model import (
        patch_embedding as pe,
        swag_patch_embedding as spe,
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from experimental.overhead_matching.swag.evaluation import (
        evaluate_swag as es
    )
    import torch
    import json
    import msgspec

    import matplotlib
    import matplotlib.pyplot as plt

    import seaborn as sns

    import pandas as pd
    import altair as alt
    return Path, alt, es, json, mo, msgspec, pd, pe, spe, torch, vd


@app.cell
def _(Path):
    dataset_path = Path('/data/overhead_matching/datasets/VIGOR/NewYork')
    checkpoint = 130
    model_path = Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_batch_size_512_hard_negative_20_lr_3e-4_warmup_5/')
    return checkpoint, dataset_path, model_path


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(checkpoint, load_model, model_path):
    sat_model = load_model(model_path / f"{checkpoint:04d}_satellite")
    pano_model = load_model(model_path / f"{checkpoint:04d}_panorama")
    return pano_model, sat_model


@app.cell(hide_code=True)
def _(dataset_path, pano_model, sat_model, vd):
    landmark_version = "v3"
    dataset = vd.VigorDataset(
        dataset_path,
        vd.VigorDatasetConfig(
            satellite_patch_size=sat_model.patch_dims,
            panorama_size=pano_model.patch_dims,
            satellite_tensor_cache_info=vd.TensorCacheInfo(
                dataset_keys=[dataset_path.name],
                model_type="satellite",
                landmark_version=landmark_version,
                extractor_info=sat_model.cache_info()),
            panorama_tensor_cache_info=vd.TensorCacheInfo(
                dataset_keys=[dataset_path.name],
                model_type="panorama",
                landmark_version=landmark_version,
                extractor_info=pano_model.cache_info()),
            landmark_version=landmark_version))
    return (dataset,)


@app.cell(hide_code=True)
def _(dataset, es, pano_model, sat_model):
    similarity = es.compute_cached_similarity_matrix(
        pano_model=pano_model, sat_model=sat_model, dataset=dataset, use_cached_similarity=True, device='cuda:0')
    return (similarity,)


@app.cell(hide_code=True)
def _(dataset, pd, similarity, torch):
    # Compute panorama statistics
    def compute_panorama_statistics(dataset, similarity):
        records = []
        num_panos = len(dataset._panorama_metadata)
        sat_idxs = torch.zeros((num_panos, 5), dtype=torch.int32)
        valid_mask = torch.zeros((num_panos, 5), dtype=torch.bool)

        for pano_idx, metadata in dataset._panorama_metadata.iterrows():
            assert len(metadata.positive_satellite_idxs) <= 1
            assert len(metadata.semipositive_satellite_idxs) <= 4
            for sat_idx in metadata.positive_satellite_idxs:
                sat_idxs[pano_idx, 0] = sat_idx
                valid_mask[pano_idx, 0] = True

            for i, sat_idx in enumerate(metadata.semipositive_satellite_idxs):
                sat_idxs[pano_idx, i+1] = sat_idx
                valid_mask[pano_idx, i+1] = True

        row_idxs = torch.arange(num_panos).reshape(-1, 1).repeat((1, 5))
        sat_sims = similarity[row_idxs, sat_idxs]
        sat_sims[~valid_mask] = torch.nan
        bool_mat = similarity[:, None, :] >= sat_sims[:, :, None]
        ranks = bool_mat.sum(dim=-1, dtype=torch.int32)

        sorted_sims = torch.sort(similarity)

        for pano_idx in range(num_panos):
            num_valid_semipositive = valid_mask[pano_idx, 1:].sum()
            records.append({
                'pano_idx': pano_idx,
                'positive_sat_sim': sat_sims[pano_idx, 0].item() if valid_mask[pano_idx, 1] else None,
                'positive_rank': ranks[pano_idx, 0].item() if valid_mask[pano_idx, 0] else None,
                'positive_sat_sim_list': sat_sims[pano_idx, :1].cpu().numpy(),
                'positive_rank_list': ranks[pano_idx, :1].cpu().numpy(),
                'semipositive_ranks': ranks[pano_idx, 1:1+num_valid_semipositive].cpu().numpy(),
                'semipositive_sat_sims': sat_sims[pano_idx, 1:1+num_valid_semipositive].cpu().numpy(),
                'top_10_sats': sorted_sims.indices[pano_idx, -10:].cpu().numpy(),
                'top_10_sims': sorted_sims.values[pano_idx, -10:].cpu().numpy()
            })

        return pd.DataFrame.from_records(records)


    # dataset._panorama_metadata.positive_satellite_idxs
    metrics_df = compute_panorama_statistics(dataset, similarity)
    return (metrics_df,)


@app.cell(hide_code=True)
def _(dataset, metrics_df):
    to_plot_df = metrics_df.join(dataset._panorama_metadata)
    return (to_plot_df,)


@app.cell(hide_code=True)
def _(alt, mo, to_plot_df):
    chart = mo.ui.altair_chart(alt.Chart(to_plot_df).mark_point().encode(
        x='lon',
        y='lat',
        color='positive_rank'
    ).properties(height=1000, width=1000))

    mo.vstack([chart])
    return (chart,)


@app.cell(hide_code=True)
def _(chart, dataset, mo, torch, vd):
    selected_df = chart.value
    _pano_metadata = selected_df.iloc[0]
    _pano = mo.image(_pano_metadata.path)

    _positive_sat_idxs = selected_df.iloc[0].positive_satellite_idxs
    _semipositive_sat_idxs = selected_df.iloc[0].semipositive_satellite_idxs

    def load_sat_image(sat_idx, similarity, rank, img_size):
        sat_metadata = dataset._satellite_metadata.loc[sat_idx]
        sat_path = sat_metadata.path
        img, _ = vd.load_image(sat_path,resize_shape=None)

        bbox = (img.shape[1] // 4, img.shape[2] // 4, img.shape[1] * 3 // 4, img.shape[2] * 3 // 4)
        img[:, bbox[0]:bbox[2], bbox[1]-1:bbox[1]+1] = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        img[:, bbox[0]:bbox[2], bbox[3]-1:bbox[3]+1] = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        img[:, bbox[0]-1:bbox[0]+1, bbox[1]:bbox[3]] = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)
        img[:, bbox[2]-1:bbox[2]+1, bbox[1]:bbox[3]] = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1)

        pano_in_image_y = int(_pano_metadata["web_mercator_y"] - sat_metadata["web_mercator_y"]) + img.shape[1] // 2
        pano_in_image_x = int(_pano_metadata["web_mercator_x"] - sat_metadata["web_mercator_x"]) + img.shape[2] // 2

        size = 10
        img[:, pano_in_image_y-size:pano_in_image_y+size, pano_in_image_x-size:pano_in_image_x+size] = torch.tensor([0.0, 1.0, 1.0]).reshape(3, 1, 1)

        return mo.vstack([mo.image(img.permute(1, 2, 0), height=img_size, width=img_size),
                          f"sim:{similarity:0.3f} rank: {rank} offset: ({pano_in_image_y:0.1f}, {pano_in_image_x:0.1f})"])

    print(selected_df.iloc[0].path)
    mo.vstack([
        selected_df,
        _pano,
        mo.hstack([
            mo.hstack([load_sat_image(_idx, _pano_metadata.positive_sat_sim_list[_i], _pano_metadata.positive_rank_list[_i], 300) for _i, _idx in enumerate(_positive_sat_idxs)]),
            mo.hstack([load_sat_image(_idx, _pano_metadata.semipositive_sat_sims[_i], _pano_metadata.semipositive_ranks[_i], 300) for _i, _idx in enumerate(_semipositive_sat_idxs)]),
        ],widths=[1, 4]),
        mo.hstack([load_sat_image(_idx, _pano_metadata.top_10_sims[_i], 10-_i, 250) for _i, _idx in enumerate(_pano_metadata.top_10_sats[:5])], wrap=True),
        mo.hstack([load_sat_image(_idx, _pano_metadata.top_10_sims[_i+5], 5-_i, 250) for _i, _idx in enumerate(_pano_metadata.top_10_sats[5:])], wrap=True),
    ])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
