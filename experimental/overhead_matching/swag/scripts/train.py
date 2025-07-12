import argparse
import json
import math
import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import itertools
from pathlib import Path
from common.python.serialization import dataclass_to_dict, flatten_dict
from experimental.overhead_matching.swag.data import vigor_dataset
from experimental.overhead_matching.swag.model import patch_embedding
from experimental.overhead_matching.swag.model import swag_patch_embedding
from typing import Union
from dataclasses import dataclass
import tqdm
import msgspec
from pprint import pprint


@dataclass
class LearningRateSchedule:
    initial_lr: float
    lr_step_factor: float
    num_epochs_at_lr: int


@dataclass
class OptimizationConfig:
    num_epochs: int

    # This is the number of examples from which we build triplets
    # This is computed with torch.no_grad, so more examples can
    # be processed in a batch than is possible when we are taking a
    # gradient step.
    batch_size: int

    lr_schedule: LearningRateSchedule

    # Switch from random sampling to hard negative mining after
    # this many epochs
    enable_hard_negative_sampling_after_epoch_idx: int

    random_sample_type: vigor_dataset.HardNegativeMiner.RandomSampleType


ModelConfig = Union[patch_embedding.WagPatchEmbeddingConfig,
                    swag_patch_embedding.SwagPatchEmbeddingConfig]


@dataclass
class TrainConfig:
    opt_config: OptimizationConfig
    sat_model_config: ModelConfig
    pano_model_config: ModelConfig
    output_dir: Path
    tensorboard_output: Path | None
    dataset_path: list[Path]


@dataclass
class Pairs:
    positive_pairs: list[tuple[int, int]]
    negative_pairs: list[tuple[int, int]]
    semipositive_pairs: list[tuple[int, int]]


def enc_hook(obj):
    if isinstance(obj, Path):
        return str(obj)
    else:
        raise ValueError(f"Unhandled Value: {obj}")


def dec_hook(type, obj):
    if type is Path:
        return Path(obj)
    raise ValueError(f"Unhandled type: {type=} {obj=}")


def create_pairs(panorama_metadata, satellite_metadata) -> Pairs:
    # Generate an exhaustive set of triplets where the anchor is a panorama
    # TODO consider creating triplets where a satellite patch is the anchor
    out = Pairs(positive_pairs=[], negative_pairs=[], semipositive_pairs=[])
    for batch_pano_idx in range(len(panorama_metadata)):
        # batch_pano_idx is the index of the panorama in the batch
        # pano_idx is the index of the panorama in the dataset
        pano_idx = panorama_metadata[batch_pano_idx]['index']

        for batch_sat_idx in range(len(satellite_metadata)):
            # batch_sat_idx is the index of the satellite image in the batch
            curr_sat_metadata = satellite_metadata[batch_sat_idx]
            if pano_idx in curr_sat_metadata["positive_panorama_idxs"]:
                out.positive_pairs.append((batch_pano_idx, batch_sat_idx))
            elif pano_idx in curr_sat_metadata["semipositive_panorama_idxs"]:
                out.semipositive_pairs.append((batch_pano_idx, batch_sat_idx))
            else:
                out.negative_pairs.append((batch_pano_idx, batch_sat_idx))
    return out


def train(config: TrainConfig, *, dataset, panorama_model, satellite_model, quiet):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # save config:
    config_json = msgspec.json.encode(config, enc_hook=enc_hook)
    config_dict = json.loads(config_json)
    with open(config.output_dir / "config.json", 'wb') as f:
        f.write(config_json)
    writer = SummaryWriter(
        log_dir=config.tensorboard_output
    )
    # write hyperparameters
    writer.add_hparams(
        flatten_dict(config_dict['opt_config']), {}
    )
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()
    panorama_model.train()
    satellite_model.train()

    print(dataset._satellite_metadata)
    print(dataset._panorama_metadata)
    print(dataset._landmark_metadata)

    opt_config = config.opt_config

    miner = vigor_dataset.HardNegativeMiner(
            batch_size=opt_config.batch_size,
            embedding_dimension=panorama_model.output_dim,
            random_sample_type=opt_config.random_sample_type,
            dataset=dataset)
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_sampler=miner, num_workers=2, persistent_workers=True)

    opt = torch.optim.Adam(
        list(panorama_model.parameters()) + list(satellite_model.parameters()),
        lr=opt_config.lr_schedule.initial_lr
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=opt_config.lr_schedule.num_epochs_at_lr,
            gamma=opt_config.lr_schedule.lr_step_factor)

    torch.set_printoptions(linewidth=200)

    total_batches = 0
    for epoch_idx in tqdm.tqdm(range(opt_config.num_epochs),  desc="Epoch"):
        for batch_idx, batch in enumerate(dataloader):
            pairs = create_pairs(
                batch.panorama_metadata,
                batch.satellite_metadata
            )

            if opt_config.random_sample_type == vigor_dataset.HardNegativeMiner.RandomSampleType.NEAREST:
                pairs = Pairs(
                    positive_pairs=pairs.positive_pairs + pairs.semipositive_pairs,
                    semipositive_pairs=[],
                    negative_pairs=pairs.negative_pairs)

            opt.zero_grad()

            panorama_embeddings = panorama_model(batch.panorama.cuda())
            sat_embeddings = satellite_model(
                swag_patch_embedding.ModelInput(
                    image=batch.satellite.cuda(),
                    metadata=batch.satellite_metadata))

            similarity = torch.einsum("ad,bd->ab", panorama_embeddings, sat_embeddings)

            loss = 0
            pos_rows = [x[0] for x in pairs.positive_pairs]
            pos_cols = [x[1] for x in pairs.positive_pairs]
            pos_similarities = similarity[pos_rows, pos_cols]

            semipos_rows = [x[0] for x in pairs.semipositive_pairs]
            semipos_cols = [x[1] for x in pairs.semipositive_pairs]
            semipos_similarities = similarity[semipos_rows, semipos_cols]

            neg_rows = [x[0] for x in pairs.negative_pairs]
            neg_cols = [x[1] for x in pairs.negative_pairs]
            neg_similarities = similarity[neg_rows, neg_cols]

            # Compute Loss
            POS_WEIGHT = 5
            AVG_POS_SIMILARITY = 0.0
            if len(pairs.positive_pairs):
                pos_loss = torch.log(
                        1 + torch.exp(-POS_WEIGHT * (pos_similarities - AVG_POS_SIMILARITY)))
                pos_loss = torch.mean(pos_loss) / POS_WEIGHT
            else:
                pos_loss = torch.tensor(0)

            SEMIPOS_WEIGHT = 6
            AVG_SEMIPOS_SIMILARITY = 0.3
            if len(pairs.semipositive_pairs):
                semipos_loss = torch.log(
                        1 + torch.exp(-SEMIPOS_WEIGHT * (
                            semipos_similarities - AVG_SEMIPOS_SIMILARITY)))
                semipos_loss = torch.mean(semipos_loss) / SEMIPOS_WEIGHT
            else:
                semipos_loss = torch.tensor(0)

            NEG_WEIGHT = 20
            AVG_NEG_SIMILARITY = 0.7
            if len(pairs.negative_pairs):
                neg_loss = torch.log(
                        1 + torch.exp(NEG_WEIGHT * (neg_similarities - AVG_NEG_SIMILARITY)))
                neg_loss = torch.mean(neg_loss) / NEG_WEIGHT
            else:
                neg_loss = torch.tensor(0)

            loss = pos_loss + neg_loss + semipos_loss
            loss.backward()
            opt.step()

            # Hard Negative Mining
            miner.consume(
                    panorama_embeddings=panorama_embeddings.detach(),
                    satellite_embeddings=sat_embeddings.detach(),
                    batch=batch)

            # Logging
            writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step=total_batches)
            writer.add_scalar("train/num_positive_pairs", len(pairs.positive_pairs), global_step=total_batches)
            writer.add_scalar("train/num_semipos_pairs", len(pairs.semipositive_pairs), global_step=total_batches)
            writer.add_scalar("train/num_neg_pairs", len(pairs.negative_pairs), global_step=total_batches)
            writer.add_scalar("train/loss_pos", pos_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss_semipos", semipos_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss_neg", neg_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss", loss.item(), global_step=total_batches)
            if not quiet:
                print(f"{epoch_idx=:4d} {batch_idx=:4d} lr: {lr_scheduler.get_last_lr()[0]:.2e} " +
                      f" num_pos_pairs: {len(pairs.positive_pairs):3d}" +
                      f" num_semipos_pairs: {len(pairs.semipositive_pairs):3d}" +
                      f" num_neg_pairs: {len(pairs.negative_pairs):3d} {pos_loss.item()=:0.6f}" +
                      f" {semipos_loss.item()=:0.6f} {neg_loss.item()=:0.6f} {loss.item()=:0.6f}", end='\r')
                if batch_idx % 50 == 0:
                    print()

            total_batches += 1
        if not quiet:
            print()
        lr_scheduler.step()

        if epoch_idx >= opt_config.enable_hard_negative_sampling_after_epoch_idx:
            miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE)

        if (epoch_idx % 10 == 0) or (epoch_idx == opt_config.num_epochs - 1):
            # Periodically save the model
            config.output_dir.mkdir(parents=True, exist_ok=True)
            panorama_model_path = config.output_dir / f"{epoch_idx:04d}_panorama"
            satellite_model_path = config.output_dir / f"{epoch_idx:04d}_satellite"

            batch = next(iter(dataloader))
            save_model(panorama_model, panorama_model_path,
                       (batch.panorama[:opt_config.batch_size].cuda(),))
            save_model(satellite_model, satellite_model_path,
                       (batch.satellite[:opt_config.batch_size].cuda(),))


def main(
        dataset_base_path: Path,
        output_base_path: Path,
        train_config_path: Path,
        quiet: bool):
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(file_in.read(), type=TrainConfig, dec_hook=dec_hook)
    pprint(train_config)

    PANORAMA_NEIGHBOR_RADIUS_DEG = 1e-6
    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=PANORAMA_NEIGHBOR_RADIUS_DEG,
        satellite_patch_size=train_config.sat_model_config.patch_dims,
        panorama_size=train_config.pano_model_config.patch_dims,
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
    )
    dataset_paths = [dataset_base_path / p for p in train_config.dataset_path]
    landmark_paths = [dataset_base_path / "landmarks" / (str(p) + ".geojson") for p in train_config.dataset_path]
    dataset = vigor_dataset.VigorDataset(dataset_paths, dataset_config, landmark_paths)

    if isinstance(train_config.sat_model_config, patch_embedding.WagPatchEmbeddingConfig):
        satellite_model = patch_embedding.WagPatchEmbedding(train_config.sat_model_config)
    elif isinstance(train_config.sat_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        satellite_model = swag_patch_embedding.SwagPatchEmbedding(train_config.sat_model_config)

    if isinstance(train_config.pano_model_config, patch_embedding.WagPatchEmbeddingConfig):
        panorama_model = patch_embedding.WagPatchEmbedding(train_config.pano_model_config)
    elif isinstance(train_config.pano_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        panorama_model = swag_patch_embedding.SwagPatchEmbedding(train_config.pano_model_config)

    output_dir = output_base_path / train_config.output_dir
    tensorboard_output = train_config.tensorboard_output
    tensorboard_output = tensorboard_output if tensorboard_output is not None else output_dir / "logs"

    train_config.output_dir = output_dir
    train_config.tensorboard_output = tensorboard_output

    train(
        train_config,
        dataset=dataset,
        panorama_model=panorama_model,
        satellite_model=satellite_model,
        quiet=quiet)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", help="path to dataset", required=True)
    parser.add_argument("--output_base", help="path to output", required=True)
    parser.add_argument("--train_config", help="path to train_config", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    main(
        Path(args.dataset_base),
        Path(args.output_base),
        Path(args.train_config),
        quiet=args.quiet)
