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
        torch,
    )


@app.cell
def _():
    from experimental.overhead_matching.learned.model import clevr_transformer, clevr_tokenizer
    from experimental.overhead_matching.learned.data import clevr_dataset
    from experimental.overhead_matching.learned.scripts import train_clevr_transformer as tct
    from common.torch.load_and_save_models import load_model
    return clevr_dataset, clevr_tokenizer, clevr_transformer, load_model, tct


@app.cell
def _(Path):
    model_path = Path('/tmp/ego_vec_overhead_16patch/epoch_000220/').expanduser()
    training_dataset_path = Path('~/scratch/clevr/overhead_image_ego_vectorized').expanduser()
    eval_dataset_path = Path('~/scratch/clevr/overhead_image_ego_vectorized_val/').expanduser()
    return eval_dataset_path, model_path, training_dataset_path


@app.cell
def _(clevr_dataset, eval_dataset_path):
    eval_dataset = clevr_dataset.ClevrDataset(eval_dataset_path, load_overhead=True)
    loader = clevr_dataset.get_dataloader(eval_dataset, batch_size=64)
    return eval_dataset, loader


@app.cell
def _(load_model, model_path):
    model = load_model(model_path)
    model = model.cuda()
    model = model.eval()
    return (model,)


@app.cell
def _(clevr_transformer, loader, model, np, tct, torch):
    # Collect the predictions
    rng = np.random.default_rng(1024)
    ego_from_worlds = []
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ego_from_worlds.append(tct.sample_ego_from_world(rng, len(batch)))
            ego_scene_descriptions = tct.project_scene_description_to_ego(
                batch.scene_description["objects"], ego_from_worlds[-1])
            inputs = clevr_transformer.SceneDescription(
                overhead_image=None,
                ego_image=None,
                ego_scene_description=ego_scene_descriptions,
                overhead_scene_description=None
            )
            query_tokens = None
            query_mask = None
            results.append(model(inputs, query_tokens, query_mask))

    results = {
        k: torch.cat([x[k] for x in results]).cpu().numpy()
        for k in results[0]
    }
    ego_from_worlds = np.concatenate(ego_from_worlds)
    return (
        batch,
        ego_from_worlds,
        ego_scene_descriptions,
        i,
        inputs,
        query_mask,
        query_tokens,
        results,
        rng,
    )


@app.cell
def _(ego_from_worlds, eval_dataset, np, pd, results):
    pose_source = 'prediction'

    # compute the errors
    pred_x = results[pose_source][:, 0]
    pred_y = results[pose_source][:, 1]
    pred_cos = results[pose_source][:, 2]
    pred_sin = results[pose_source][:, 3]
    pred_theta = np.arctan2(pred_sin, pred_cos)

    gt_x = ego_from_worlds[:, 0, 2]
    gt_y = ego_from_worlds[:, 1, 2]
    gt_cos = ego_from_worlds[:, 0, 0]
    gt_sin = ego_from_worlds[:, 1, 0]
    gt_theta = np.arctan2(gt_sin, gt_cos)

    dtheta = pred_theta - gt_theta
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi

    error = results[pose_source][:, :2] - ego_from_worlds[:, :2, 2]
    error = np.sqrt(np.sum(error * error, axis=1)).mean()
    print(error)

    df = pd.DataFrame({
        'pred_x': pred_x,
        'pred_y': pred_y,
        'pred_cos': pred_cos,
        'pred_sin': pred_sin,
        'pred_theta': pred_theta,
        'gt_x': gt_x,
        'gt_y': gt_y,
        'gt_cos': gt_cos,
        'gt_sin': gt_sin,
        'gt_theta': gt_theta,
        'dx': pred_x - gt_x,
        'dy': pred_y - gt_y,
        'dtheta': dtheta,
        'scene': [x.scene_description["objects"] for x in eval_dataset],
        'num_objects': [len(x.scene_description["objects"]) for x in eval_dataset],
        'image_filename': eval_dataset.overhead_file_paths,

    })
    return (
        df,
        dtheta,
        error,
        gt_cos,
        gt_sin,
        gt_theta,
        gt_x,
        gt_y,
        pose_source,
        pred_cos,
        pred_sin,
        pred_theta,
        pred_x,
        pred_y,
    )


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
def _():
    return


if __name__ == "__main__":
    app.run()
