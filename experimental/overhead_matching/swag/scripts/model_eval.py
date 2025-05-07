import marimo

__generated_with = "0.11.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import common.torch.load_torch_deps
    import torch

    from experimental.overhead_matching.swag.data import vigor_dataset, satellite_embedding_database
    from experimental.overhead_matching.swag.evaluation import evaluate_swag
    import experimental.overhead_matching.swag.model.patch_embedding
    from common.torch.load_and_save_models import load_model
    from pathlib import Path
    return (
        Path,
        alt,
        common,
        evaluate_swag,
        experimental,
        load_model,
        mo,
        pd,
        satellite_embedding_database,
        torch,
        vigor_dataset,
    )


@app.cell
def _(
    Path,
    evaluate_swag,
    load_model,
    satellite_embedding_database,
    vigor_dataset,
):
    sat_model = load_model(Path("/data/overhead_matching/models/all_chicago_model/0020_satellite"), device='cuda')
    pano_model = load_model(Path("/data/overhead_matching/models/all_chicago_model/0020_panorama"), device='cuda')

    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=1e-6,
        satellite_patch_size=(320, 320),
        panorama_size=(320,640),
        factor=0.25
    )

    dataset = vigor_dataset.VigorDataset(Path("/data/overhead_matching/datasets/VIGOR/Chicago/"), dataset_config)
    dataset_loader = vigor_dataset.get_dataloader(dataset, batch_size=5)

    sat_dataset = dataset.get_sat_patch_view()
    sat_loader = vigor_dataset.get_dataloader(sat_dataset, batch_size=5)
    sat_db = satellite_embedding_database.build_satellite_embedding_database(sat_model, sat_loader)

    evaluate_swag.evaluate_prediction_top_k(sat_db, dataset_loader, pano_model)
    return (
        dataset,
        dataset_config,
        dataset_loader,
        pano_model,
        sat_dataset,
        sat_db,
        sat_loader,
        sat_model,
    )


@app.cell
def _(pano_embeddings):
    pano_embeddings.shape
    return


@app.cell
def _(sat_db):
    sat_db.shape
    return


@app.cell
def _(pano_embeddings, sat_db, satellite_embedding_database):
    import numpy as np
    np.set_printoptions(linewidth=200)
    # `similarities` is a n_panos x n_sats matrix where `similarities[i][j]` corresponds to the similarity between the
    # `i`th panorama and the `j`th satellite patch
    similarities = satellite_embedding_database.calculate_cos_similarity_against_database(
        pano_embeddings, sat_db).cpu().numpy()
    return np, similarities


@app.cell
def _(np, similarities):
    idxs = np.argsort(similarities, axis=1)
    return (idxs,)


@app.cell
def _(idxs):
    idxs
    return


@app.cell
def _(dataset):
    dataset._panorama_metadata
    return


@app.cell
def _(mo):
    alpha = mo.ui.slider(start=0, stop=50, step=1, value=1, label="alpha")
    m = mo.ui.slider(start=-1, stop=1, step=0.01, value=0, label="m")
    return alpha, m


@app.cell
def _(alpha, alt, m, mo, np, pd):
    import numpy

    x = np.linspace(-1, 1, 100)
    y = np.log(1 + np.exp(alpha.value* (x - m.value))) / alpha.value

    chart = mo.ui.altair_chart(alt.Chart(pd.DataFrame({'x': x, 'y':y})).mark_line().encode(x="x", y='y'))
    mo.vstack([chart, mo.hstack([alpha, m])])
    return chart, numpy, x, y


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
