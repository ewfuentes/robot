import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import torch
    from pathlib import Path
    from functools import reduce
    import operator
    import numpy as np
    import altair as alt
    import pandas as pd
    import seaborn as sns
    from pprint import pprint, pformat
    from marimo._plugins.ui._impl.altair_chart import ChartSelection
    return (
        ChartSelection,
        Path,
        alt,
        common,
        mo,
        np,
        operator,
        pd,
        pformat,
        pprint,
        reduce,
        sns,
        torch,
    )


@app.cell
def _():
    from experimental.overhead_matching.learned.model import clevr_transformer, clevr_tokenizer
    from experimental.overhead_matching.learned.data import clevr_dataset
    from experimental.overhead_matching.learned.scripts import (train_clevr_transformer as tct, learning_utils as lu)
    from common.torch.load_and_save_models import load_model
    return (
        clevr_dataset,
        clevr_tokenizer,
        clevr_transformer,
        load_model,
        lu,
        tct,
    )


@app.cell
def _(Path):
    model_path = Path('~/scratch/clevr/clevr_models/use_patch_size_16/best').expanduser()
    # model_path = Path('~/scratch/clevr/clevr_models/use_conv_stem_model/best').expanduser()
    # model_path = Path('~/scratch/clevr/clevr_models/gaussian_output_image_overhead_ego_vec_identical_dataset/best').expanduser()
    eval_dataset_path = Path('~/scratch/clevr/overhead_image_ego_vectorized_val/').expanduser()
    # eval_dataset_path = Path('~/scratch/clevr/more_than_four_objects_val_all_objects_identical/').expanduser()
    return eval_dataset_path, model_path


@app.cell
def _(clevr_dataset, eval_dataset_path, load_model, model_path, tct):
    eval_dataset = clevr_dataset.ClevrDataset(eval_dataset_path, load_overhead=True)
    loader = clevr_dataset.get_dataloader(eval_dataset, batch_size=128)
    model = load_model(model_path, skip_constient_output_check=True).cuda().eval()
    return eval_dataset, loader, model


@app.cell
def _(eval_dataset, loader, lu, model, np):
    df = lu.gather_clevr_model_performance(model, loader, np.random.default_rng(123))
    df['image_filename'] = eval_dataset.overhead_file_paths
    return (df,)


@app.cell
def _(df):
    df['mse'].mean()
    return


@app.cell
def _(alt, df, mo):
    pos_chart = mo.ui.altair_chart(alt.Chart(df).mark_point().encode(x='dx', y='dy'))
    angle_chart = mo.ui.altair_chart(alt.Chart(df.reset_index()).mark_point().encode(x='index', y='dtheta'))
    return angle_chart, pos_chart


@app.cell
def _(eval_dataset_path, mo, pformat):
    def make_views(chart):
        first_row = chart.value.iloc[0] if len(chart.value) else None
        image_path = eval_dataset_path / 'images' / first_row['image_filename'] if first_row is not None else None
        return mo.vstack([
            chart,
            chart.value,
            mo.hstack([mo.image(image_path) if image_path is not None and image_path.exists() else "",
                      mo.plain_text(pformat(first_row["scene"] if first_row is not None else "")),])])
    return (make_views,)


@app.cell
def _(make_views, pos_chart):
    make_views(pos_chart)
    return


@app.cell
def _(angle_chart, make_views):
    make_views(angle_chart)
    return


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, sns):
    ## error vs number of objects
    sns.boxplot(df, x='num_objects', y='mse')
    return


if __name__ == "__main__":
    app.run()
