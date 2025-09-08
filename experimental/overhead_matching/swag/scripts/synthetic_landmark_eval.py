import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from experimental.overhead_matching.swag.data import vigor_dataset as vd
    from experimental.overhead_matching.swag.model import synthetic_landmark_extractor as sle
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    matplotlib.style.use('ggplot')
    return Path, mo, np, plt, sle, vd


@app.cell
def _(Path, vd):
    dataset = vd.VigorDataset(
        Path("/data/overhead_matching/datasets/VIGOR/Chicago/"),
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,

            should_load_images=False        
        ))
    return (dataset,)


@app.cell
def _(sle):
    _config = sle.SyntheticLandmarkExtractorConfig(
        log_grid_spacing=5,
        grid_bounds_px=640,
        embedding_dim=128,
        should_produce_bearing_position_for_pano=False
    )

    model = sle.SyntheticLandmarkExtractor(_config)

    return (model,)


@app.cell
def _(np, plt):
    def plot_text(output, offset, color):
        descriptions = np.packbits(((output.features[~output.mask].numpy() + 1) / 2).astype(np.uint8), axis=-1)
        positions = output.debug["landmark_positions"][0]

        for p, desc in zip(positions, descriptions):
            plt.text(p[1] + offset[1], p[0] + offset[0], f"{desc[:4]}", color=color)
            print(p, desc)
    return


@app.cell
def _(dataset, mo, model, plt, sle):
    _item = dataset[0]
    _sat_input = sle.ModelInput(image=_item.satellite.unsqueeze(0), metadata=[_item.satellite_metadata], cached_tensors=None)
    _pano_input = sle.ModelInput(image=_item.panorama.unsqueeze(0), metadata=[_item.panorama_metadata], cached_tensors=None)
    sat_output = model(_sat_input)
    pano_output = model(_pano_input)

    plt.figure(figsize=(6,6))
    lms = pano_output.debug["landmark_positions"][0]
    plt.plot(lms[:, 1], lms[:, 0], 'bs', label='pano', markersize=5)
    plt.plot(_item.panorama_metadata["web_mercator_x"], _item.panorama_metadata["web_mercator_y"], 'b*', label='pano_loc')

    # plot_text(pano_output, (+10, 3), 'b')

    lms = sat_output.debug["landmark_positions"][0]
    plt.plot(lms[:, 1], lms[:, 0], 'ro', label='sat', markersize=3)
    plt.plot(_item.satellite_metadata["web_mercator_x"], _item.satellite_metadata["web_mercator_y"], 'r*', label='sat_loc')
    # plot_text(sat_output, (-10, 3), 'r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.legend()
    mo.mpl.interactive(plt.gcf())

    return (pano_output,)


@app.cell
def _(np, pano_output):
    np.packbits(((pano_output.features[~pano_output.mask].numpy() + 1) / 2).astype(np.uint8), axis=-1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
