import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
        SemanticLandmarkExtractor, SemanticLandmarkExtractorConfig)
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import numpy as np
    return (
        SemanticLandmarkExtractor,
        SemanticLandmarkExtractorConfig,
        mo,
        np,
        plt,
        vd,
    )


@app.cell
def _(SemanticLandmarkExtractor, SemanticLandmarkExtractorConfig):
    model = SemanticLandmarkExtractor(SemanticLandmarkExtractorConfig())
    return


@app.cell
def _(vd):
    dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Chicago',
                             vd.VigorDatasetConfig(
                                 satellite_tensor_cache_info=None,
                                 panorama_tensor_cache_info=None,
                                 landmark_version="v2"
                             ))
    return (dataset,)


@app.cell
def _(dataset):
    item = dataset[500]
    return


@app.cell
def _(dataset, mo, np, plt):
    num_sat_landmark_idxs = []
    num_pano_landmark_idxs = []
    for _, row in dataset._satellite_metadata.iterrows():
        num_sat_landmark_idxs.append(len(row.landmark_idxs))

    for _, row in dataset._panorama_metadata.iterrows():
        num_pano_landmark_idxs.append(len(row.landmark_idxs))

    bins = np.arange(-0.5, 15.5, 1.0)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(num_sat_landmark_idxs, bins=bins)
    plt.title('satellite')
    plt.yscale('log')
    plt.subplot(122)
    plt.hist(num_pano_landmark_idxs, bins=bins)
    plt.title('panorama')
    plt.yscale('log')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(dataset):
    _mask = [len(x) == 0 for x in dataset._satellite_metadata.landmark_idxs]
    dataset._satellite_metadata[_mask]
    return


@app.cell
def _(dataset):
    dataset._panorama_metadata.iloc[1172]
    return


@app.cell
def _(dataset):
    unique_landmarks = set()
    for i, df in dataset._landmark_metadata.iterrows():
        fields = df.drop(["web_mercator_y", "web_mercator_x", "panorama_idxs", "satellite_idxs",
            "landmark_type", 'element', 'id', 'geometry', 'opening_hours', 'website'])
        fields = fields.dropna()
        fields = fields.to_dict()
        fields = frozenset(fields.items())
        unique_landmarks.add(fields)
    return (unique_landmarks,)


@app.cell
def _(unique_landmarks):
    for obj in unique_landmarks:
        print(dict(obj))
    return


@app.cell
def _(unique_landmarks):
    len(unique_landmarks)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
