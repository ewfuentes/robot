import argparse

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import seaborn as sns
import pandas as pd

from pathlib import Path
import numpy as np

import common.torch.load_torch_deps
import torch
from common.torch.load_and_save_models import load_model

from experimental.overhead_matching.learned.model import (
    clevr_transformer,
    clevr_tokenizer,
)
from experimental.overhead_matching.learned.data import clevr_dataset
from experimental.overhead_matching.learned.scripts import (
    train_clevr_transformer as tct,
    learning_utils as lu,
)



def main(model_path: Path, dataset_path: Path):

    dataset = clevr_dataset.ClevrDataset(dataset_path, load_overhead=True)
    loader = clevr_dataset.get_dataloader(dataset, batch_size=128, num_workers=12)

    # _, dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1023))
    # loader = clevr_dataset.get_dataloader(dataset, batch_size=128, num_workers=12)

    model = load_model(model_path, skip_constient_output_check=True).cuda().eval()


    rng = np.random.default_rng(2048)
    df = lu.gather_clevr_model_performance(model, loader, rng)
    print(df['mse'].mean())

    g1 = sns.JointGrid(data=df, x="dx", y="dy")
    g1.plot(sns.histplot, sns.histplot)
    plt.suptitle(
        f"Position Errors (($\mu_x={df['dx'].mean():0.3f}$, $\sigma_x={df['dx'].std(): 0.3f}$), ($\sigma_y={df['dy'].mean(): 0.3f}$, $\sigma_y={df['dy'].std():0.3f}$))"
    )

    plt.figure()
    sns.histplot(df, x="dtheta")
    plt.title(f"Angle error ($\mu={df['dtheta'].mean():0.3f}$, $\sigma={df['dtheta'].std():0.3f}$)")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)

    args = parser.parse_args()

    main(Path(args.model), Path(args.dataset))
