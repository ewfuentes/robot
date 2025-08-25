import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from bokeh.plotting import figure, show
    import numpy as np
    from pathlib import Path

    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )

    from experimental.overhead_matching.swag.model import (
        swag_patch_embedding as spe,
        swag_config_types as sct,
    )
    return Path, figure, np, show, vd


@app.cell
def _(Path, vd):
    dataset = vd.VigorDataset(
        Path('/data/overhead_matching/datasets/VIGOR/NewYork'),
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None
        )
    )
    return (dataset,)


@app.cell
def _(dataset):
    item = dataset[100]
    return (item,)


@app.cell
def _(item):
    item
    return


@app.cell
def _(figure, item, np, show):
    x = np.linspace(0, 2*np.pi, 200)
    y = np.cos(x)

    p = figure(title="simple line example", x_axis_label="x", y_axis_label="cos(x)")
    p.image_rgba(item.satellite.permute(1, 2, 0))


    show(p)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
