import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd,
        alphaearth_registry as ar
    )

    import matplotlib
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import scipy.ndimage

    import common.torch.load_torch_deps
    import torch

    matplotlib.style.use('ggplot')
    return Path, ar, mo, np, plt, torch, vd


@app.cell
def _(Path, ar, vd):
    _base_path = Path('/data/overhead_matching/datasets/alphaearth/')
    registry = ar.AlphaEarthRegistry(_base_path, version='v1')

    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago')
    _dataset = vd.VigorDataset(
        _dataset_path,
        vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None))
    dataset = _dataset.get_sat_patch_view()
    return dataset, registry


@app.cell
def _(registry):
    all_features, _ = registry.query(41.884608, -87.6419839, patch_size=(2000, 6000), zoom_level=20)
    return (all_features,)


@app.cell
def _(all_features, np):
    _flat_features = all_features.reshape(-1, 64)
    _flat_features = _flat_features[~np.isnan(_flat_features[:, 0])]
    _centered_features = _flat_features - _flat_features.mean(axis=0, keepdims=True)
    _feature_cov = np.cov(_centered_features.T)
    feature_eig = np.linalg.eig(_feature_cov)

    return (feature_eig,)


@app.cell
def _(all_features, feature_eig, np):
    projections = np.einsum("hwf,fd->hwd", all_features, feature_eig.eigenvectors[:, 0:3])
    # vis[np.isnan(vis)] = 0.0
    min = np.nanmin(projections, axis=(0, 1)).reshape(1, 1, -1)
    max = np.nanmax(projections, axis=(0, 1)).reshape(1, 1, -1)
    vis = (projections - min) / (max - min)

    return max, min, vis


@app.cell
def _(mo, plt, vis):
    plt.figure(figsize=(8, 8))
    plt.imshow(vis)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(dataset, registry):
    item = dataset[14872]

    features, position = registry.query(
        lat_deg=item.satellite_metadata["lat"],
        lon_deg=item.satellite_metadata["lon"],
        patch_size=(10, 10),
        zoom_level=item.satellite_metadata["zoom_level"])
    return features, item, position


@app.cell
def _(feature_eig, features, max, min, np, plt):
    plt.figure()
    _patch_proj = np.einsum("hwf,fd->hwd", features, feature_eig.eigenvectors[:, 0:3])
    _patch_proj = (_patch_proj - min) / (max - min)
    plt.imshow(_patch_proj)
    return


@app.cell
def _(item, np):
    patch_rows = np.arange(item.satellite_metadata["original_shape"][0]) - item.satellite_metadata["original_shape"][0] // 2 + 0.5
    patch_cols = np.arange(item.satellite_metadata["original_shape"][1]) - item.satellite_metadata["original_shape"][1] // 2 + 0.5

    patch_rows, patch_cols = np.meshgrid(patch_rows, patch_cols, indexing='ij')
    patch_rows += item.satellite_metadata["web_mercator_y"]
    patch_cols += item.satellite_metadata["web_mercator_x"]
    return patch_cols, patch_rows


@app.cell
def _(position):
    position.shape
    return


@app.cell
def _(np, patch_cols, patch_rows, position):
    feature_span = (
        (position[-1, 0, 0] - position[0, 0, 0]) / 2.0,
        (position[0, -1, 1] - position[0, 0, 1]) / 2.0)
    feature_center = (
        (position[-1, 0, 0] + position[0, 0, 0]) / 2.0,
        (position[0, -1, 1] + position[0, 0, 1]) / 2.0)
    patch_rows_in_features = (patch_rows - feature_center[0]) / feature_span[0]
    patch_cols_in_features = (patch_cols - feature_center[1]) / feature_span[1]
    idxs_in_features = np.stack([patch_cols_in_features, patch_rows_in_features], axis=-1)
    return (idxs_in_features,)


@app.cell
def _(idxs_in_features):
    idxs_in_features[:2, :2, :]
    return


@app.cell
def _(features, idxs_in_features, torch):
    torch_features = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).cuda()
    torch_grid = torch.from_numpy(idxs_in_features).unsqueeze(0).cuda()
    return torch_features, torch_grid


@app.cell
def _(torch, torch_features, torch_grid):
    resampled = torch.nn.functional.grid_sample(torch_features, torch_grid, mode='nearest', align_corners=False).squeeze().cpu().permute(1, 2, 0).numpy()
    return (resampled,)


@app.cell
def _(resampled):
    resampled.shape
    return


@app.cell
def _(feature_eig, item, max, min, mo, np, plt, resampled):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(item.satellite.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(item.satellite.permute(1, 2, 0))
    _patch_proj = np.einsum("hwf,fd->hwd", resampled, feature_eig.eigenvectors[:, 0:3])
    _patch_proj = (_patch_proj - min) / (max - min)
    plt.imshow(_patch_proj, alpha=0.95)
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(dataset, np):

    mask = np.logical_and(np.logical_and(dataset.dataset._satellite_metadata.lat > 41.8805,
                                         dataset.dataset._satellite_metadata.lat < 41.881),
                          np.logical_and(dataset.dataset._satellite_metadata.lon > -87.618,
                                         dataset.dataset._satellite_metadata.lon < -87.617)
                         )

    dataset.dataset._satellite_metadata.loc[mask]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
