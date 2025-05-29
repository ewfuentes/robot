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
    def get_top_k_results(model_partial_path, dataset_path):
        sat_model = load_model(Path(f"{model_partial_path}_satellite"), device='cuda')
        pano_model = load_model(Path(f"{model_partial_path}_panorama"), device='cuda')

        dataset_config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=1e-6,
            satellite_patch_size=(320, 320),
            panorama_size=(320,640),
            factor=1.0
        )

        dataset = vigor_dataset.VigorDataset(Path(dataset_path), dataset_config)
        dataset_loader = vigor_dataset.get_dataloader(dataset, batch_size=128)

        sat_dataset = dataset.get_sat_patch_view()
        sat_loader = vigor_dataset.get_dataloader(sat_dataset, batch_size=128)
        sat_db = satellite_embedding_database.build_satellite_embedding_database(sat_model, sat_loader)

        return evaluate_swag.evaluate_prediction_top_k(sat_db, dataset_loader, pano_model)
    return (get_top_k_results,)


@app.cell
def _(get_top_k_results):
    print('computing chicago top k')
    chicago_top_k_results = get_top_k_results(
        "/data/overhead_matching/models/all_chicago_model/0240",
        "/data/overhead_matching/datasets/VIGOR/Chicago",
    )
    print('computing sf top k')
    sanfrancisco_top_k_results = get_top_k_results(
        "/data/overhead_matching/models/all_chicago_model/0240",
        "/data/overhead_matching/datasets/VIGOR/SanFrancisco/",
    )
    print('computing new york top k')
    newyork_top_k_results = get_top_k_results(
        "/data/overhead_matching/models/all_chicago_model/0240",
        "/data/overhead_matching/datasets/VIGOR/NewYork/",
    )
    return (
        chicago_top_k_results,
        newyork_top_k_results,
        sanfrancisco_top_k_results,
    )


@app.cell
def _(
    chicago_top_k_results,
    newyork_top_k_results,
    plt,
    sanfrancisco_top_k_results,
):
    fig = plt.figure()
    ax = plt.subplot(311)
    plt.hist(chicago_top_k_results["k_value"], bins=50)
    plt.title('Chicago')
    plt.ylabel('Count')
    plt.subplot(312)
    plt.hist(newyork_top_k_results["k_value"], bins=50)
    plt.title('New York')
    plt.ylabel('Count')
    plt.subplot(313)
    plt.hist(sanfrancisco_top_k_results["k_value"], bins=50)
    plt.title("San Francisco")
    plt.ylabel('Count')
    plt.xlabel("K Value")
    plt.suptitle("K Value Distribution")
    plt.tight_layout()
    fig
    return ax, fig


@app.cell
def _(pano_embeddings, sat_db, satellite_embedding_database):
    import numpy as np
    np.set_printoptions(linewidth=200)
    satellite_embedding_database.calculate_cos_similarity_against_database(pano_embeddings, sat_db).cpu().numpy()
    return (np,)


@app.cell
def _(alt, mo, pano_embeddings, pd):
    def  make_pano_plot():
        d = {
            'row': [],
            'dim': [],
            'data': [],
        }
        for row_idx, row in enumerate(pano_embeddings.cpu().numpy()):
            for dim_idx, entry in enumerate(row):
                d['row'].append(str(row_idx))
                d['dim'].append(dim_idx)
                d['data'].append(entry)
        pano_df = pd.DataFrame(d)
    
        chart = mo.ui.altair_chart(alt.Chart(pano_df).mark_point().encode(x="dim", y="data", color="row"))
        return mo.vstack([chart])
    make_pano_plot()
    return (make_pano_plot,)


@app.cell
def _(alt, mo, pd, sat_db):
    def make_sat_db_plot():
        d = {
            'row': [],
            'dim': [],
            'data': [],
        }
        for row_idx, row in enumerate(sat_db.cpu().numpy()):
            for dim_idx, entry in enumerate(row):
                d['row'].append(str(row_idx))
                d['dim'].append(dim_idx)
                d['data'].append(entry)
        sat_df = pd.DataFrame(d)
    
        chart = mo.ui.altair_chart(alt.Chart(sat_df).mark_point().encode(x="dim", y="data", color="row"))
        return mo.vstack([chart])
    make_sat_db_plot()
    return (make_sat_db_plot,)


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
