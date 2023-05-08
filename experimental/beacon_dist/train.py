import argparse
import os
import torch
from typing import NamedTuple, Callable
import time

import IPython

from experimental.beacon_dist.utils import (
    Dataset,
    KeypointBatch,
    batchify,
    generate_valid_queries,
    generate_invalid_queries,
    valid_configuration_loss,
    get_descriptor_test_dataset,
    get_x_position_test_dataset,
    get_y_position_test_dataset,
    test_dataset_collator,
)
from experimental.beacon_dist.model import ConfigurationModel, ConfigurationModelParams


class TrainConfig(NamedTuple):
    num_epochs: int
    dataset_path: str
    test_dataset: str
    output_dir: str
    random_seed: int
    num_environments_per_batch: int
    num_queries_per_environment: int


def make_collator_fn(num_queries_per_environment: int):
    def __inner__(samples: list[KeypointBatch]) -> KeypointBatch:
        """Join multiple samples into a single batch"""
        fields = {}
        for f in KeypointBatch._fields:
            fields[f] = torch.nested.nested_tensor(
                [getattr(sample, f) for sample in samples]
            )
        batch = batchify(KeypointBatch(**fields))
        batch = KeypointBatch(
            **{
                k: v.repeat_interleave(num_queries_per_environment, dim=0)
                for k, v in batch._asdict().items()
            }
        )

        # This will get called from a dataloader which may spawn multiple processes
        # we have different environments, so we should generate different queries
        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        valid_queries = generate_valid_queries(batch.class_label, rng)
        invalid_queries = generate_invalid_queries(
            batch.class_label, valid_queries, rng
        )

        valid_query_selector = torch.randint(
            low=0,
            high=2,
            size=(batch.x.shape[0], 1),
            dtype=torch.bool,
            generator=rng,
        )
        invalid_query_selector = torch.logical_not(valid_query_selector)
        queries = (
            valid_queries * valid_query_selector
            + invalid_queries * invalid_query_selector
        )
        return batch, queries

    return __inner__


def train(dataset: Dataset, train_config: TrainConfig, collator_fn: Callable):
    # create model
    model = ConfigurationModel(
        ConfigurationModelParams(
            descriptor_size=256,
            descriptor_embedding_size=32,
            position_encoding_factor=10000,
            num_encoder_heads=2,
            num_encoder_layers=2,
            num_decoder_heads=2,
            num_decoder_layers=2,
        )
    ).to("cuda")

    # create dataloader
    torch.manual_seed(train_config.random_seed + 1)
    rng = torch.Generator()
    rng.manual_seed(train_config.random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.90, 0.10], generator=rng
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.num_environments_per_batch,
        shuffle=True,
        collate_fn=collator_fn,
        num_workers=4,
        persistent_workers=True,
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.num_environments_per_batch,
        collate_fn=collator_fn,
    )

    print(model)

    # Create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    for epoch_idx in range(train_config.num_epochs):
        loss = None
        epoch_loss = 0.0
        epoch_start_time = time.time()
        # Train
        for batch_idx, (batch, queries) in enumerate(train_data_loader):
            batch_start_time = time.time()
            # Zero gradients
            batch = batch.to("cuda")
            queries = queries.to("cuda")
            optim.zero_grad()

            # compute model outputs
            model_out = model(batch, queries)

            # compute loss
            loss = valid_configuration_loss(batch.class_label, queries, model_out)
            loss.backward()

            # take step
            optim.step()
            batch_dt = time.time() - batch_start_time
            if batch_idx % 10 == 0:
                # print(f"Batch: {batch_idx} dt: {batch_dt: 0.6f} s Loss: {loss: 0.6f}")
                ...
            epoch_loss += loss.detach().item() * queries.shape[0]

        model.train(False)
        # Evaluation
        validation_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch, queries) in enumerate(test_data_loader):
                batch = batch.to("cuda")
                queries = queries.to("cuda")

                model_out = model(batch, queries)
                validation_loss += valid_configuration_loss(
                    batch.class_label, queries, model_out, reduction="sum"
                )
        model.train(True)

        epoch_dt = time.time() - epoch_start_time
        if epoch_idx % 1 == 0:
            print(
                f"End of Epoch {epoch_idx} dt: {epoch_dt: 0.6f} s Epoch Loss: {epoch_loss: 0.6f}",
                f"Validation Loss: {validation_loss: 0.6f}",
                flush=True,
            )
        if epoch_idx % 100 == 0:
            file_name = os.path.join(
                train_config.output_dir, f"model_{epoch_idx:09}.pt"
            )
            print(f"Saving model: {file_name}", flush=True)
            torch.save(model.state_dict(), file_name)
    model.eval()
    IPython.embed()


def main(train_config: TrainConfig):
    assert (
        train_config.dataset_path is None or train_config.test_dataset is None
    ), "Cannot set dataset_path and test_dataset"
    if not os.path.exists(train_config.output_dir):
        print(f"Creating output directory: {train_config.output_dir}")
        os.makedirs(train_config.output_dir)

    if train_config.dataset_path:
        dataset = Dataset(filename=train_config.dataset_path)
        collator_fn = make_collator_fn(train_config.num_queries_per_environment)
    elif train_config.test_dataset:
        dataset_generator = {
            "x": get_x_position_test_dataset,
            "y": get_y_position_test_dataset,
            "descriptor": get_descriptor_test_dataset,
        }
        dataset = Dataset(data=dataset_generator[train_config.test_dataset]())
        train_config = train_config._replace(num_environments_per_batch=1)
        collator_fn = test_dataset_collator

    train(dataset, train_config, collator_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reconstructor Training script")
    parser.add_argument(
        "--dataset_path",
        help="path to dataset",
        type=str,
    )
    parser.add_argument(
        "--test_dataset",
        help="Launch training with test dataset",
        choices=["descriptor", "x", "y"],
    )
    parser.add_argument(
        "--output_dir",
        help="path to output directory. It will be created if it doesn't exist",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_epochs",
        help="number of epochs",
        nargs="?",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--random_seed",
        help="seed for all random operations",
        nargs="?",
        type=int,
        # Set an arbitrary initial seed
        default=0xFFBD5654,
    )

    parser.add_argument(
        "--num_environments_per_batch",
        help="number_of_environments_per_batch",
        nargs="?",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_queries_per_environment",
        help="Number of queries per environment",
        nargs="?",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    main(TrainConfig(**vars(args)))
