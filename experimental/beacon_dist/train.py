import argparse
import os
import torch
from typing import NamedTuple
import time


from experimental.beacon_dist.utils import (
    Dataset,
    KeypointBatch,
    batchify,
)
from experimental.beacon_dist.model import ConfigurationModel, ConfigurationModelParams


class TrainConfig(NamedTuple):
    num_epochs: int
    dataset_path: str
    output_dir: str
    random_seed: int


def collate_fn(samples: list[KeypointBatch]) -> KeypointBatch:
    """Join multiple samples into a single batch"""
    fields = {}
    for f in KeypointBatch._fields:
        fields[f] = torch.nested.nested_tensor(
            [getattr(sample, f) for sample in samples]
        )

    return batchify(KeypointBatch(**fields))


def train(dataset: Dataset, train_config: TrainConfig):
    # create model
    model = ConfigurationModel(ConfigurationModelParams(
        descriptor_size=256,
        descriptor_embedding_size=256,
        position_encoding_factor=10000,
        num_encoder_heads=4,
        num_encoder_layers=4,
        num_decoder_heads=4,
        num_decoder_layers=4
    )).to('cuda')

    # create dataloader
    rng = torch.Generator()
    rng.manual_seed(train_config.random_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    print(model)

    loss_func = torch.nn.BCEWithLogitsLoss()

    # Create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch_idx in range(train_config.num_epochs):
        loss = None
        for batch_idx, batch in enumerate(data_loader):
            # Zero gradients
            batch = batch.to('cuda')
            optim.zero_grad()

            # compute model outputs
            out = model(batch, torch.ones_like(batch.x).to(torch.bool))

            # compute loss
            loss = loss_func(out, torch.ones_like(out))
            loss.backward()

            # take step
            optim.step()
        print(f'End of Epoch {epoch_idx} Loss: {loss}')


def main(train_config: TrainConfig):
    if not os.path.exists(train_config.output_dir):
        print(f"Creating output directory: {train_config.output_dir}")
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
