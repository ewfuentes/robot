import argparse
import os
import torch
from typing import NamedTuple, Callable
import time
import tqdm

import IPython

from experimental.beacon_dist.utils import (
    KeypointBatch,
    KeypointPairs,
    batchify,
    generate_valid_queries,
    generate_invalid_queries,
    valid_configuration_loss,
    get_descriptor_test_dataset,
    get_x_position_test_dataset,
    get_y_position_test_dataset,
    test_dataset_collator,
)
import experimental.beacon_dist.multiview_dataset as mvd
from experimental.beacon_dist.model import ConfigurationModel, ConfigurationModelParams


class TrainConfig(NamedTuple):
    num_epochs: int
    dataset_path: list[str]
    test_dataset: str
    output_dir: str
    random_seed: int
    num_environments_per_batch: int
    num_queries_per_environment: int
    data_parallel: bool
    epochs_between_checkpoints: int
    distributed: bool


def make_collator_fn(num_queries_per_environment: int):
    def __collate_keypoint_batch__(samples: list[KeypointBatch]) -> KeypointBatch:
        fields = {}
        for f in KeypointBatch._fields:
            fields[f] = torch.nested.nested_tensor(
                [getattr(sample, f) for sample in samples]
            )
        batch = batchify(KeypointBatch(**fields))
        return KeypointBatch(
            **{
                k: v.repeat_interleave(num_queries_per_environment, dim=0)
                for k, v in batch._asdict().items()
            }
        )

    def __inner__(samples: list[KeypointPairs]) -> KeypointPairs:
        # Work with KeypointPairs
        batch = KeypointPairs(
            context=__collate_keypoint_batch__([x.context for x in samples]),
            query=__collate_keypoint_batch__([x.query for x in samples]),
        )

        # This will get called from a dataloader which may spawn multiple processes
        # we have different environments, so we should generate different queries

        rng = torch.Generator(device="cpu")
        rng.manual_seed(0)
        valid_configs = generate_valid_queries(batch.query.class_label, rng)
        invalid_configs = generate_invalid_queries(
            batch.query.class_label, valid_configs, rng
        )

        valid_config_selector = torch.randint(
            low=0,
            high=2,
            size=(batch.query.x.shape[0], 1),
            dtype=torch.bool,
            generator=rng,
        )
        invalid_config_selector = torch.logical_not(valid_config_selector)
        configs = (
            valid_configs * valid_config_selector
            + invalid_configs * invalid_config_selector
        )
        return batch, configs

    return __inner__

def setup_distributed_training():
    world_size = int(os.environ["SLURM_NTASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    print('World size:', world_size, "rank:", rank)
    print('CPUs allocated', os.environ["SLURM_CPUS_PER_GPU"], 'gpu count:', torch.cuda.device_count())
    # Note that this function expects the MASTER_ADDR and MASTER_PORT environment variables to be set
    torch.distributed.init_process_group(rank=rank, world_size=world_size)
    print('torch rank:', torch.distributed.get_rank(),
          'torch world size:', torch.distributed.get_world_size(),
          'is_init:', torch.distributed.is_initialized())

def is_master():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def train(
    dataset: mvd.MultiviewDataset, train_config: TrainConfig, collator_fn: Callable
):
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

    if train_config.data_parallel:
        model = torch.nn.DataParallel(model)

    if train_config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    if is_master():
        print(model)

    # create dataloader
    torch.manual_seed(train_config.random_seed + 1)
    rng = torch.Generator()
    rng.manual_seed(train_config.random_seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.90, 0.10], generator=rng
    )

    train_sampler = None
    if train_config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_GPU', os.cpu_count()))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.num_environments_per_batch,
        shuffle=(train_sampler is None),
        collate_fn=collator_fn,
        num_workers=max(5, num_cpus),
        prefetch_factor=4,
        persistent_workers=True,
        sampler=train_sampler,
    )

    test_sampler = None
    if train_config.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_config.num_environments_per_batch,
        shuffle=(test_sampler is None),
        collate_fn=collator_fn,
        num_workers=max(5, os.cpu_count() // 2),
        prefetch_factor=4,
        persistent_workers=True,
        sampler=test_sampler,
    )

    # Create the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)
    for epoch_idx in range(train_config.num_epochs):
        loss = None
        epoch_loss = 0.0
        epoch_start_time = time.time()
        # Train
        if train_config.distributed:
            train_sampler.set_epoch(epoch_idx)
        for batch_idx, (batch, configs) in tqdm.tqdm(
            enumerate(train_data_loader), total=len(train_data_loader),
            disable=not is_master()
        ):
            batch_start_time = time.time()
            # Zero gradients
            batch = batch.to("cuda")
            configs = configs.to("cuda")
            optim.zero_grad()

            # compute model outputs
            model_out = model(batch, configs)

            # compute loss
            loss = valid_configuration_loss(batch.query.class_label, configs, model_out)
            loss.backward()

            # take step
            optim.step()
            batch_end_time = time.time()
            batch_dt = batch_end_time - batch_start_time
            if batch_idx % 10 == 0:
                # print(f"Batch: {batch_idx} dt: {batch_dt: 0.6f} s Loss: {loss: 0.6f}")
                ...
            epoch_loss += loss.detach().item() * configs.shape[0]

        model.train(False)
        # Evaluation
        validation_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch, configs) in enumerate(test_data_loader):
                batch = batch.to("cuda")
                configs = configs.to("cuda")

                model_out = model(batch, configs)
                validation_loss += valid_configuration_loss(
                    batch.query.class_label, configs, model_out, reduction="sum"
                )
        model.train(True)

        epoch_dt = time.time() - epoch_start_time
        if epoch_idx % 1 == 0 and is_master():
            print(
                f"End of Epoch {epoch_idx} dt: {epoch_dt: 0.6f} s Epoch Loss: {epoch_loss: 0.6f}",
                f"Validation Loss: {validation_loss: 0.6f}",
                flush=True,
            )
        if (epoch_idx % train_config.epochs_between_checkpoints) == 0 and is_master():
            file_name = os.path.join(
                train_config.output_dir, f"model_{epoch_idx:09}.pt"
            )
            print(f"Saving model: {file_name}", flush=True)
            torch.save(model.module.state_dict()
                    if train_config.distributed or train_config.data_parallel
                    else model.state_dict(), file_name)
    model.eval()
    IPython.embed()


def main(train_config: TrainConfig):
    assert (
        train_config.dataset_path is None or train_config.test_dataset is None
    ), "Cannot set dataset_path and test_dataset"
    if not os.path.exists(train_config.output_dir):
        print(f"Creating output directory: {train_config.output_dir}")
        os.makedirs(train_config.output_dir)

    if train_config.distributed:
        setup_distributed_training()

    if train_config.dataset_path:
        try:
            dataset = mvd.MultiviewDataset(
                mvd.DatasetInputs(
                    file_paths=train_config.dataset_path,
                    index_path=None,
                    data_tables=None,
                )
            )
        except AssertionError:
            dataset = mvd.MultiviewDataset.from_single_view(
                filename=train_config.dataset_path[0]
            )
        collator_fn = make_collator_fn(train_config.num_queries_per_environment)
    elif train_config.test_dataset:
        dataset_generator = {
            "x": get_x_position_test_dataset,
            "y": get_y_position_test_dataset,
            "descriptor": get_descriptor_test_dataset,
        }
        dataset = mvd.MultiviewDataset.from_single_view(
            data=dataset_generator[train_config.test_dataset]()
        )
        train_config = train_config._replace(num_environments_per_batch=1)
        collator_fn = test_dataset_collator

    train(dataset, train_config, collator_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reconstructor Training script")
    parser.add_argument(
        "--dataset_path",
        help="path to dataset",
        nargs="+",
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

    parser.add_argument(
        "--data_parallel",
        help="Wrap the model in a DataParallel Block",
        action="store_true",
    )

    parser.add_argument(
        "--epochs_between_checkpoints",
        help="Number of epoch between checkpoints",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--distributed",
        help="Run the training script in a distributed mode",
        action='store_true',
    )

    args = parser.parse_args()

    main(TrainConfig(**vars(args)))
