import argparse
import os
import torch
from typing import NamedTuple
import time


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
)
from experimental.beacon_dist.model import ConfigurationModel, ConfigurationModelParams


class TrainConfig(NamedTuple):
    num_epochs: int
    dataset_path: str
    test_dataset: str
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
    model = ConfigurationModel(
        ConfigurationModelParams(
            descriptor_size=256,
            descriptor_embedding_size=16,
            position_encoding_factor=10000,
            num_encoder_heads=2,
            num_encoder_layers=2,
            num_decoder_heads=2,
            num_decoder_layers=2,
        )
    ).to("cuda")

    # create dataloader
    rng = torch.Generator()
    rng.manual_seed(train_config.random_seed)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    print(model)

    # Create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    for epoch_idx in range(train_config.num_epochs):
        loss = None
        for batch_idx, batch in enumerate(data_loader):
            # batch_start_time = time.time()
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
            # batch_time = time.time() - batch_start_time
            # if batch_idx % 10 == 0:
            #     print(f"Batch: {batch_idx} dt: {batch_time: 0.3f} s Loss: {loss}")

        print(f"End of Epoch {epoch_idx} Loss: {loss}", flush=True)
        if epoch_idx % 100 == 0:
            file_name = os.path.join(
                train_config.output_dir, f"model_{epoch_idx:09}.pt"
            )
            print(f"Saving model: {file_name}", flush=True)
            torch.save(model.state_dict(), file_name)


def main(train_config: TrainConfig):
    assert (
        train_config.dataset_path is None or train_config.test_dataset is None
    ), "Cannot set dataset_path and test_dataset"
    if not os.path.exists(train_config.output_dir):
        print(f"Creating output directory: {train_config.output_dir}")
        os.makedirs(train_config.output_dir)

    if train_config.dataset_path:
        dataset = Dataset(filename=train_config.dataset_path)
    elif train_config.test_dataset:
        dataset_generator = {
            'x': get_x_position_test_dataset,
            'y': get_y_position_test_dataset,
            'descriptor': get_descriptor_test_dataset,
        }
        dataset = Dataset(data=dataset_generator[train_config.test_dataset]())

    train(dataset, train_config)


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

    args = parser.parse_args()

    main(TrainConfig(**vars(args)))
