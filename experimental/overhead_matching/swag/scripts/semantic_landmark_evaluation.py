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
    from common.ollama import pyollama
    import json
    import tqdm
    import pandas as pd
    import geopandas as gpd
    from pathlib import Path
    import math
    return (
        Path,
        SemanticLandmarkExtractor,
        SemanticLandmarkExtractorConfig,
        gpd,
        json,
        math,
        mo,
        np,
        plt,
        pyollama,
        tqdm,
        vd,
    )


@app.cell
def _():
    from experimental.overhead_matching.swag.model import swag_model_input_output as smio
    return (smio,)


@app.cell
def _(Path, gpd, math):
    def deg2pixel(lat_deg, lon_deg, zoom):
        """Convert lat/lon to global pixel coordinates at given zoom level"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        x = int((lon_deg + 180.0) / 360.0 * n * 256)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256)
        return (x, y)

    def convert_geometry_to_pixels(geometry, zoom=20):
        """Convert geometry coordinates to pixel coordinates"""
        from shapely.ops import transform

        def coord_transform(lon, lat, z=None):
            x, y = deg2pixel(lat, lon, zoom)  # Note: deg2pixel expects (lat, lon)
            return (x, y)

        return transform(coord_transform, geometry)
    zoom_level = 20
    df = gpd.read_file(Path('/data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v3.geojson'))
    df["geometry_px"] = df['geometry'].apply(
        lambda geom: convert_geometry_to_pixels(geom, zoom_level)
    )

    return (df,)


@app.cell
def _(df):
    df_px = df.set_geometry('geometry_px')
    return


app._unparsable_cell(
    r"""
    df_px.
    """,
    name="_"
)


@app.cell
def _(df, mo):
    mo.mpl.interactive(df.plot())
    return


@app.cell
def _(batch):
    batch.satellite_metadata[0]
    return


@app.cell
def _(df):
    df.geometry.crs
    return


@app.cell
def _(df):
    df.geometry
    return


@app.cell
def _(df, np):
    mask = np.logical_and(
        df.element == "way",
        df["addr:city"] == "Chicago")

    mask = np.logical_and(mask,
                         df["addr:street:name"] == "Adams")

    df[mask]
    return


@app.cell
def _(SemanticLandmarkExtractorConfig):
    SemanticLandmarkExtractorConfig()
    return


@app.cell
def _(SemanticLandmarkExtractor, SemanticLandmarkExtractorConfig):
    model = SemanticLandmarkExtractor(SemanticLandmarkExtractorConfig(llm_str='gemma3:4b-it-qat'))
    return (model,)


@app.cell
def _(vd):
    dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Chicago',
                             vd.VigorDatasetConfig(
                                 satellite_tensor_cache_info=None,
                                 panorama_tensor_cache_info=None,
                             ))
    return (dataset,)


@app.cell
def _(dataset, vd):
    dataloader = vd.get_dataloader(dataset, batch_size=5)
    return (dataloader,)


@app.cell
def _(dataloader):
    batch = next(iter(dataloader))
    return (batch,)


@app.cell
def _(batch):
    batch
    return


@app.cell
def _(batch, smio):
    model_input = smio.ModelInput(metadata=batch.satellite_metadata, image=batch.satellite, cached_tensors=None)
    return (model_input,)


@app.cell
def _(model, model_input):
    extractor_output = model(model_input)
    return (extractor_output,)


@app.cell
def _(extractor_output):
    bytes(extractor_output.debug["sentences"][0][0])
    return


@app.cell
def _(batch):
    batch.satellite_metadata
    return


@app.cell
def _(item):
    item.panorama_metadata
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

    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)
    plt.hist(num_sat_landmark_idxs, bins=bins)
    plt.title(f'Satellite (Count: {len(dataset._satellite_metadata)})')
    plt.yscale('log')
    plt.subplot(122, sharey=ax)
    plt.hist(num_pano_landmark_idxs, bins=bins)
    plt.ylim(100, 10000)
    plt.title(f'Panorama (Count: {len(dataset._panorama_metadata)})')
    plt.yscale('log')
    plt.suptitle(f'Chicago Landmarks (Count: {len(dataset._landmark_metadata)})')
    fig.supylabel('Count')
    fig.supxlabel('Number of landmarks')
    plt.tight_layout()
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
def _(dataset, df):
    unique_landmarks = set() 
    check_date_cols = [x for x in dataset._landmark_metadata.columns if x.startswith('check_date')]
    for i, _df in dataset._landmark_metadata.iterrows():
        fields = df.drop([
            "web_mercator_y", "web_mercator_x", "panorama_idxs", "satellite_idxs",
            "landmark_type", 'element', 'id', 'geometry', 'opening_hours', 'website',
            'addr:city', 'addr:state'] + check_date_cols)
        fields = fields.dropna()
        fields = fields.to_dict()
        fields = frozenset(fields.items())
        unique_landmarks.add(fields)
    return (unique_landmarks,)


@app.cell
def _(json, pyollama, tqdm, unique_landmarks):

    descriptions = []
    with pyollama.Ollama('gemma3:4b-it-qat') as ollama:
        for _i, obj in tqdm.tqdm(list(enumerate(unique_landmarks))):
            d = dict(obj)
            # print(dict(obj))
            try:
                prompt = f"Generate a natural language description of this openstreetmap landmark. Only include information relevant for visually identifying the object. For example, don't include payment methods accepted. Don't include any details not derived from the landmark information. Include no other details: {json.dumps(d)}"
                # prompt = f"Generate a brief description for this OpenStreetMap landmark dictionary. The brief description should describe the landmark, only including information in the dictionary relevant to visually identifying the landmark if you were walking by it on the street. Don't include addresses, but include the name if it has one.\n{d}"
                description = ollama(prompt)
                descriptions.append(description)
            except:
                print('failed to serialize', d)
                descriptions.append(None)

            # print(ollama(prompt))
            if _i > 200:
                break

    return


if __name__ == "__main__":
    app.run()
