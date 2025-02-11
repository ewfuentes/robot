import argparse

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import seaborn as sns
import pandas as pd

from pathlib import Path
import numpy as np

import common.torch as torch

from experimental.overhead_matching.learned.model import (
    clevr_transformer,
    clevr_tokenizer,
)
from experimental.overhead_matching.learned.data import clevr_dataset
from experimental.overhead_matching.learned.scripts import (
    train_clevr_transformer as tct,
)

EMBEDDING_SIZE = 128


def load_model(model_path: Path):
    config = clevr_transformer.ClevrTransformerConfig(
        token_dim=EMBEDDING_SIZE,
        vocabulary_size=96,
        num_encoder_heads=4,
        num_encoder_layers=8,
        num_decoder_heads=4,
        num_decoder_layers=8,
        output_dim=4,
        predict_gaussian=True,
    )

    model = clevr_transformer.ClevrTransformer(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


def main(model_path: Path, training_dataset: Path, dataset_path: Path):
    model = load_model(model_path).cuda().eval()

    vocabulary = clevr_dataset.ClevrDataset(training_dataset).vocabulary()
    eval = clevr_dataset.ClevrDataset(dataset_path)
    loader = clevr_dataset.get_dataloader(eval, batch_size=128)

    rng = np.random.default_rng(1024)
    ego_from_worlds = []
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch["objects"]
            ego_from_worlds.append(tct.sample_ego_from_world(rng, len(batch)))
            inputs = tct.clevr_input_from_batch(
                batch, vocabulary, EMBEDDING_SIZE, ego_from_worlds[-1]
            )
            query_tokens = None
            query_mask = None

            results.append(model(inputs, query_tokens, query_mask).cpu())

    results = torch.cat(results, dim=0).numpy()
    ego_from_worlds = np.concatenate(ego_from_worlds)

    pred_x = results[:, 0]
    pred_y = results[:, 1]
    pred_cos_t = results[:, 2]
    pred_sin_t = results[:, 3]
    pred_t = np.arctan2(pred_sin_t, pred_cos_t)

    gt_x = ego_from_worlds[:, 0, 2]
    gt_y = ego_from_worlds[:, 1, 2]
    gt_cos_t = ego_from_worlds[:, 0, 0]
    gt_sin_t = ego_from_worlds[:, 1, 0]
    gt_t = np.arctan2(gt_sin_t, gt_cos_t)

    dx = pred_x - gt_x
    dy = pred_y - gt_y
    dtheta = pred_t - gt_t
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi

    df = pd.DataFrame(
        {
            "dx (m)": dx,
            "dy (m)": dy,
            "d_cos_t": pred_cos_t - gt_cos_t,
            "d_sin_t": pred_sin_t - gt_sin_t,
            "dtheta (rad)": dtheta,
        }
    )

    g1 = sns.JointGrid(data=df, x="dx (m)", y="dy (m)")
    g1.plot(sns.histplot, sns.histplot)
    plt.suptitle(
        f"Position Errors (($\mu_x={np.mean(dx):0.3f}$, $\sigma_x={np.std(dx): 0.3f}$), ($\sigma_y={np.mean(dy): 0.3f}$, $\sigma_y={np.std(dy):0.3f}$))"
    )

    g2 = sns.JointGrid(data=df, x="d_cos_t", y="d_sin_t")
    g2.plot(sns.histplot, sns.histplot)

    plt.figure()
    sns.histplot(df, x="dtheta (rad)")
    plt.title(f"Angle error ($\mu={np.mean(dtheta):0.3f}$, $\sigma={np.std(dtheta):0.3f}$)")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--training_dataset", required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(Path(args.model), Path(args.training_dataset), Path(args.dataset))
