import argparse
import os
import torch
import sys
from typing import NamedTuple


from experimental.beacon_dist.utils import (
    Dataset,
    ReconstructorBatch,
    reconstruction_loss,
    batchify,
)
from experimental.beacon_dist.model import Reconstructor


class TrainConfig(NamedTuple):
    num_epochs: int
    dataset_path: str
    output_dir: str
    random_seed: int


def collate_fn(samples: list[ReconstructorBatch]) -> ReconstructorBatch:
    """Join multiple samples into a single batch"""
    fields = {}
    for f in ReconstructorBatch._fields:
        fields[f] = torch.nested.nested_tensor(
            [getattr(sample, f) for sample in samples]
        )

    return batchify(ReconstructorBatch(**fields))


def train(dataset: Dataset, train_config: TrainConfig):
    # create model
    model = Reconstructor()

    # create dataloader
    rng = torch.Generator()
    rng.manual_seed(train_config.random_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    # Create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch_idx in range(train_config.num_epochs):
        loss = None
        for batch_idx, batch in enumerate(data_loader):

            # Zero gradients
            optim.zero_grad()

            # compute model outputs
            out = model(batch)

            # compute loss
            loss = reconstruction_loss(batch, out)
            loss.backward()

            # take step
            optim.step()
        print(f'End of Epoch {epoch_idx} Loss: {loss}')


def main(train_config: TrainConfig):
    if not os.path.exists(train_config.output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(train_config.output_dir)

    train(Dataset(filename=train_config.dataset_path), train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reconstructor Training script")
    parser.add_argument(
        "--dataset_path", help="path to dataset", type=str, required=True
    )
    parser.add_argument(
        "--output_dir",
        help="path to output directory. It will be created if it doesn't exist",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_epochs",
        help="path to output directory. It will be created if it doesn't exist",
        nargs="?",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--random_seed",
        help="path to output directory. It will be created if it doesn't exist",
        nargs="?",
        type=int,
        # Set an arbitrary initial seed
        default=0xFFBD5654,
    )

    args = parser.parse_args()

    main(TrainConfig(**vars(args)))
