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

    import google.protobuf
    print(google.protobuf.__version__)
    print(google.protobuf.__file__)

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
        google,
        itertools,
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
        dataset_loader = vigor_dataset.get_dataloader(dataset.get_pano_view(), batch_size=128)

        sat_dataset = dataset.get_sat_patch_view()
        sat_loader = vigor_dataset.get_dataloader(sat_dataset, batch_size=128)
        sat_db = satellite_embedding_database.build_satellite_db(sat_model, sat_loader)

        return evaluate_swag.evaluate_prediction_top_k(sat_db, dataset_loader, pano_model)
    return (get_top_k_results,)


@app.cell
def _(get_top_k_results, itertools):
    model_paths = {
        'no_semi_pos': "/data/overhead_matching/models/all_chicago_model/0240",
        'w_semi_pos': "/data/overhead_matching/models/all_chicago_model_w_semipos/0080",
    }

    dataset_paths = {
        'chicago': '/data/overhead_matching/datasets/VIGOR/Chicago',
        # 'sanfrancisco': '/data/overhead_matching/datasets/VIGOR/SanFrancisco'
        "newyork": '/data/overhead_matching/datasets/VIGOR/NewYork'
    }


    results = {}

    for (model_name, model_path), (data_name, data_path) in itertools.product(model_paths.items(), dataset_paths.items()):
        print(model_name, data_name)
        results[f"{model_name}-{data_name}"] = get_top_k_results(model_path, data_path)

    return (
        data_name,
        data_path,
        dataset_paths,
        model_name,
        model_path,
        model_paths,
        results,
    )


@app.cell
def _(results):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 6))
    for i, (name, r) in enumerate(results.items()):
        ax = plt.subplot(len(results), 1, i+1)
        plt.hist(r["k_value"], bins=50)
        plt.title(name)
        plt.ylabel('Count')
        plt.yscale('log')

    plt.suptitle("K Value Distribution")
    plt.tight_layout()
    fig
    return ax, fig, i, name, plt, r


if __name__ == "__main__":
    app.run()
