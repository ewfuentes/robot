import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from experimental.overhead_matching.swag.model import (
        swag_patch_embedding as spe
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    from pathlib import Path
    import tqdm
    import functools

    import torch
    return Path, mo, plt, spe, torch, tqdm, vd


@app.cell
def _():
    import hdbscan
    return (hdbscan,)


@app.cell
def _():
    import numpy as np
    return


@app.cell
def _(Path, vd):
    dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago')

    dataset = vd.VigorDataset(
        dataset_path,
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
        )
    )
    return dataset, dataset_path


@app.cell
def _(spe):
    dino_extractor = spe.DinoFeatureExtractor(spe.DinoFeatureMapExtractorConfig(
        model_str='dinov3_vitb16')).cuda().eval()
    return (dino_extractor,)


@app.cell
def _(dino_extractor, spe, torch, tqdm):
    def sat_model_input_from_batch(batch):
        return spe.ModelInput(
            image=batch.satellite,
            metadata=batch.satellite_metadata,
            cached_tensors=None)

    def pano_model_input_from_batch(batch):
        return spe.ModelInput(
            image=batch.panorama,
            metadata=batch.panorama_metadata,
            cached_tensors=None)

    def compute_dino_features(dataloader, model_input_from_batch):
        d = dino_extractor.output_dim
        mean_accum = torch.zeros((d,), device='cuda', dtype=torch.float64)
        cov_accum = torch.zeros((d, d), device='cuda', dtype=torch.float64)
        num_tokens = 0
        for batch in tqdm.tqdm(dataloader):
            model_input = model_input_from_batch(batch).to('cuda')
            with torch.no_grad():
                output = dino_extractor(model_input)
                tokens = output.features.reshape(-1, dino_extractor.output_dim)
                num_tokens += tokens.shape[0]
                mean_accum += torch.sum(tokens, 0)
                cov_accum += tokens.T @ tokens
        return mean_accum, cov_accum, num_tokens

    return (
        compute_dino_features,
        pano_model_input_from_batch,
        sat_model_input_from_batch,
    )


@app.cell
def _(
    Path,
    compute_dino_features,
    dataset,
    dataset_path,
    pano_model_input_from_batch,
    sat_model_input_from_batch,
    torch,
    vd,
):
    sat_dataset = dataset.get_sat_patch_view()
    pano_dataset = dataset.get_pano_view()

    _sat_dataloader = vd.get_dataloader(sat_dataset, num_workers=16, batch_size=32)
    _pano_dataloader = vd.get_dataloader(pano_dataset, num_workers=8, batch_size=32)

    saved_accums_path = Path(f'/home/erick/scratch/overhead_matching/dino_features_{dataset_path.name}.pt')

    if not saved_accums_path.exists():
        sat_mean_accum, sat_cov_accum, sat_num_tokens = compute_dino_features(_sat_dataloader, sat_model_input_from_batch)
        pano_mean_accum, pano_cov_accum, pano_num_tokens = compute_dino_features(_pano_dataloader, pano_model_input_from_batch)

        torch.save({
            'sat': {
                'mean_accum': sat_mean_accum,
                'cov_accum': sat_cov_accum,
                'num_tokens': sat_num_tokens},
            'pano': {
                'mean_accum': pano_mean_accum,
                'cov_accum': pano_cov_accum,
                'num_tokens': pano_num_tokens}},
            saved_accums_path)
    else:
        loaded_data = torch.load(saved_accums_path)
        sat_mean_accum = loaded_data["sat"]["mean_accum"]
        sat_cov_accum = loaded_data["sat"]["cov_accum"]
        sat_num_tokens = loaded_data["sat"]["num_tokens"]

        pano_mean_accum = loaded_data["pano"]["mean_accum"]
        pano_cov_accum = loaded_data["pano"]["cov_accum"]
        pano_num_tokens = loaded_data["pano"]["num_tokens"]


    return (
        pano_cov_accum,
        pano_dataset,
        pano_mean_accum,
        pano_num_tokens,
        sat_cov_accum,
        sat_mean_accum,
        sat_num_tokens,
    )


@app.cell
def _(
    pano_cov_accum,
    pano_mean_accum,
    pano_num_tokens,
    sat_cov_accum,
    sat_mean_accum,
    sat_num_tokens,
):
    pano_mean = pano_mean_accum.cpu() / pano_num_tokens
    pano_cov = pano_cov_accum.cpu() / pano_num_tokens - pano_mean[:, None] @ pano_mean[None, :]
    sat_mean = sat_mean_accum.cpu() / sat_num_tokens
    sat_cov = sat_cov_accum.cpu() / sat_num_tokens - sat_mean[:, None] @ sat_mean[None, :]

    all_mean = (pano_mean_accum + sat_mean_accum).cpu() / (pano_num_tokens + sat_num_tokens)
    all_cov = (pano_cov_accum + sat_cov_accum).cpu() / (pano_num_tokens + sat_num_tokens)
    all_cov -= all_mean[:, None] @ all_mean[None, :]
    return all_cov, all_mean, pano_cov, sat_cov


@app.cell
def _(all_cov, pano_cov, sat_cov, torch):
    sat_svd = torch.linalg.svd(sat_cov)
    pano_svd = torch.linalg.svd(pano_cov)
    all_svd = torch.linalg.svd(all_cov)
    return all_svd, pano_svd, sat_svd


@app.cell
def _(all_svd, mo, pano_svd, plt, sat_svd, torch):
    sat_svd_sum = torch.cumsum(sat_svd.S, 0)
    pano_svd_sum = torch.cumsum(pano_svd.S, 0)
    all_svd_sum = torch.cumsum(all_svd.S, 0)

    plt.plot(sat_svd_sum / sat_svd_sum[-1], label='sat')
    plt.plot(pano_svd_sum / pano_svd_sum[-1], label='pano')
    plt.plot(all_svd_sum / all_svd_sum[-1], label='all')
    plt.legend()

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(dino_extractor, torch, tqdm):
    def compute_embeddings(dataloader, get_model_input):
        embeddings = []
        for batch in tqdm.tqdm(dataloader):
            model_input = get_model_input(batch).to('cuda')
            image_size = torch.tensor(model_input.image.shape[2:]) // 16
            output = dino_extractor(model_input).features
            embeddings.append(output.reshape(-1, *image_size, dino_extractor.output_dim))
        return torch.concat(embeddings, dim=0)
    return (compute_embeddings,)


@app.cell
def _(
    compute_embeddings,
    dataset,
    pano_model_input_from_batch,
    sat_model_input_from_batch,
    torch,
    vd,
):
    pano_embeddings = compute_embeddings(
        vd.get_dataloader(
            torch.utils.data.Subset(dataset.get_pano_view(), list(range(256))),
            batch_size=16),
        pano_model_input_from_batch).cpu()
    sat_embeddings = compute_embeddings(
        vd.get_dataloader(
            torch.utils.data.Subset(dataset.get_sat_patch_view(), list(range(256))),
            batch_size=32),
        sat_model_input_from_batch).cpu()
    return pano_embeddings, sat_embeddings


@app.cell
def _(
    compute_embeddings,
    dataset,
    pano_model_input_from_batch,
    sat_model_input_from_batch,
    torch,
    vd,
):
    holdout_pano_embeddings = compute_embeddings(
        vd.get_dataloader(
            torch.utils.data.Subset(dataset.get_pano_view(), list(range(256, 320))),
            batch_size=16),
        pano_model_input_from_batch).cpu()
    holdout_sat_embeddings = compute_embeddings(
        vd.get_dataloader(
            torch.utils.data.Subset(dataset.get_sat_patch_view(), list(range(256, 320))),
            batch_size=32),
        sat_model_input_from_batch).cpu()
    return holdout_pano_embeddings, holdout_sat_embeddings


@app.cell
def _(all_mean, all_svd, torch):
    def in_pca(embeddings):
        return torch.einsum("i,ij,bhwj->bhwi", all_svd.S, all_svd.Vh, embeddings-all_mean)
    return (in_pca,)


@app.cell
def _(
    holdout_pano_embeddings,
    holdout_sat_embeddings,
    in_pca,
    pano_embeddings,
    sat_embeddings,
):
    pano_in_pca = in_pca(pano_embeddings)
    sat_in_pca = in_pca(sat_embeddings)
    holdout_pano_embeddings_in_pca = in_pca(holdout_pano_embeddings)
    holdout_sat_embeddings_in_pca = in_pca(holdout_sat_embeddings)
    return pano_in_pca, sat_in_pca


@app.cell
def _(pano_in_pca, sat_in_pca, torch):
    num_components = 40

    all_in_pca_subset = torch.cat([pano_in_pca[:num_components], sat_in_pca[:num_components]], dim=1).T
    return (all_in_pca_subset,)


@app.cell
def _(all_in_pca_subset):
    all_in_pca_subset.shape
    return


@app.cell
def _(pano_in_pca):
    pano_in_pca.min()
    return


@app.cell
def _(pano_in_pca):
    pano_in_pca.max()
    return


@app.cell
def _(mo, pano_dataset, pano_in_pca, plt):
    starting_dim=3
    min_val = pano_in_pca[..., starting_dim:starting_dim+3].min()
    max_val = pano_in_pca[..., starting_dim:starting_dim+3].max()
    def norm_embeddings(embeddings):
        return (embeddings - min_val) / (max_val - min_val)



    plt.figure(figsize=(12, 10))
    for i in range(3):
        plt.subplot(3,2,2*i+1)
        plt.imshow(pano_dataset[i+10].panorama.permute((1, 2, 0)))
        plt.subplot(3,2,2*i+2)
        plt.imshow(norm_embeddings(pano_in_pca[i+10, ..., starting_dim:starting_dim+1]))
    # plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(hdbscan, pano_in_pca):

    pano_idx = 10
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(pano_in_pca[pano_idx, ..., 1:41].reshape(-1, 40))
    return clusterer, pano_idx


@app.cell
def _(clusterer, mo):
    slider = mo.ui.slider(start=-1, stop=clusterer.labels_.max(), value=0, label='cluster id', show_value=True)
    return (slider,)


@app.cell
def _(clusterer, mo, pano_dataset, pano_idx, pano_in_pca, plt, slider, torch):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.imshow(pano_dataset[pano_idx].panorama.permute(1, 2, 0))

    _img_size = pano_in_pca.shape[1:-1]

    _mask = (clusterer.labels_ == slider.value).reshape(*_img_size)

    _alpha = torch.tensor(clusterer.probabilities_).reshape(*_img_size)
    _alpha[~_mask] = 0


    plt.imshow(_mask, alpha=_alpha, extent=(-0.5, 2048.5, 1024.5, -0.5))
    plt.subplot(212)
    plt.imshow(_mask, extent=(-0.5, 2048.5, 1024.5, -0.5))

    mo.vstack([mo.mpl.interactive(fig), slider])
    return


@app.cell
def _(clusterer, plt):
    plt.hist(clusterer.probabilities_)
    return


app._unparsable_cell(
    r"""
    mport sklearn
    all_embedded = sklearn.manifold.TSNE().fit_transform(all_in_pca)
    """,
    column=None, disabled=True, hide_code=False, name="_"
)


@app.cell
def _(all_embedded, mo, plt):
    plt.figure()
    plt.plot(all_embedded[:, 0], all_embedded[:, 1], '.')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(all_embedded):
    all_embedded.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
