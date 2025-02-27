import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch as torch
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
    return clevr_dataset, clevr_tokenizer, clevr_transformer, tct


@app.cell
def _(Path):
    model_path = Path('~/scratch/overhead_matching/models/direct_prediction/001000.pt').expanduser()
    training_dataset_path = Path('~/scratch/overhead_matching/clevr/').expanduser()
    eval_dataset_path = Path('~/scratch/overhead_matching/clevr_validation/').expanduser()
    return eval_dataset_path, model_path, training_dataset_path


@app.cell
def _(clevr_dataset, eval_dataset_path, training_dataset_path):
    vocabulary = clevr_dataset.ClevrDataset(training_dataset_path).vocabulary()
    eval_dataset = clevr_dataset.ClevrDataset(eval_dataset_path)
    loader = clevr_dataset.get_dataloader(eval_dataset, batch_size=128)
    return eval_dataset, loader, vocabulary


@app.cell
def _(clevr_transformer, model_path, operator, reduce, torch, vocabulary):
    EMBEDDING_SIZE=128
    vocabulary_size = reduce(operator.mul, [len(x) for x in vocabulary.values()])
    config = clevr_transformer.ClevrTransformerConfig(
        token_dim=EMBEDDING_SIZE,
        vocabulary_size=vocabulary_size,
        num_encoder_heads=4,
        num_encoder_layers=8,
        num_decoder_heads=4,
        num_decoder_layers=8,
        output_dim=4,
        predict_gaussian=True
    )
    model = clevr_transformer.ClevrTransformer(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.cuda()
    return EMBEDDING_SIZE, config, model, vocabulary_size


@app.cell
def _(EMBEDDING_SIZE, loader, model, np, tct, torch, vocabulary):
    # Collect the predictions

    rng = np.random.default_rng(1024)
    ego_from_worlds = []
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch["objects"]
            ego_from_worlds.append(tct.sample_ego_from_world(rng, len(batch)))
            inputs = tct.clevr_input_from_batch(
                batch, vocabulary, embedding_size=EMBEDDING_SIZE, ego_from_world=ego_from_worlds[-1])
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
        i,
        inputs,
        query_mask,
        query_tokens,
        results,
        rng,
    )


@app.cell
def _(ego_from_worlds, eval_dataset, np, pd, results):
    pose_source = 'decoder_output'

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
        'scene': [x["objects"] for x in eval_dataset],
        'num_objects': [len(x["objects"]) for x in eval_dataset],
        'image_filename': [x["image_filename"] for x in eval_dataset],
    })
    return (
        df,
        dtheta,
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
