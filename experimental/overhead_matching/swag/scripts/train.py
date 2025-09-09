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
from experimental.overhead_matching.swag.data import (
        vigor_dataset, satellite_embedding_database as sed)
from experimental.overhead_matching.swag.model import (
        patch_embedding, swag_patch_embedding)
from typing import Union
from dataclasses import dataclass
import tqdm
import msgspec
from pprint import pprint
import ipdb


@dataclass
class LearningRateSchedule:
    initial_lr: float
    lr_step_factor: float
    num_epochs_at_lr: int

    warmup_factor: float
    num_warmup_epochs: int


@dataclass
class LossConfig:
    positive_weight: float
    avg_positive_similarity: float

    semipositive_weight: float
    avg_semipositive_similarity: float

    negative_weight: float
    avg_negative_similarity: float


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
    hard_negative_pool_size: int

    random_sample_type: vigor_dataset.HardNegativeMiner.RandomSampleType

    loss_config: LossConfig


ModelConfig = Union[patch_embedding.WagPatchEmbeddingConfig,
                    swag_patch_embedding.SwagPatchEmbeddingConfig]


@dataclass
class DatasetConfig:
    paths: list[Path]
    factor: None | float = 1.0
    should_load_images: bool = True


@dataclass
class TrainConfig:
    opt_config: OptimizationConfig
    sat_model_config: ModelConfig
    pano_model_config: ModelConfig
    output_dir: Path
    tensorboard_output: Path | None
    dataset_config: DatasetConfig
    validation_dataset_configs: list[DatasetConfig]


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


def compute_loss(sat_embeddings, pano_embeddings, pairs, loss_config):
    similarity = torch.einsum("ad,bd->ab", pano_embeddings, sat_embeddings)

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
    POS_WEIGHT = loss_config.positive_weight
    AVG_POS_SIMILARITY = loss_config.avg_positive_similarity
    if len(pairs.positive_pairs):
        pos_loss = torch.log(
                1 + torch.exp(-POS_WEIGHT * (pos_similarities - AVG_POS_SIMILARITY)))
        pos_loss = torch.mean(pos_loss) / POS_WEIGHT
    else:
        pos_loss = torch.tensor(0)

    SEMIPOS_WEIGHT = loss_config.semipositive_weight
    AVG_SEMIPOS_SIMILARITY = loss_config.avg_semipositive_similarity
    if len(pairs.semipositive_pairs):
        semipos_loss = torch.log(
                1 + torch.exp(-SEMIPOS_WEIGHT * (
                    semipos_similarities - AVG_SEMIPOS_SIMILARITY)))
        semipos_loss = torch.mean(semipos_loss) / SEMIPOS_WEIGHT
    else:
        semipos_loss = torch.tensor(0)

    NEG_WEIGHT = loss_config.negative_weight
    AVG_NEG_SIMILARITY = loss_config.avg_negative_similarity
    if len(pairs.negative_pairs):
        neg_loss = torch.log(
                1 + torch.exp(NEG_WEIGHT * (neg_similarities - AVG_NEG_SIMILARITY)))
        neg_loss = torch.mean(neg_loss) / NEG_WEIGHT
    else:
        neg_loss = torch.tensor(0)

    return {
        'loss': pos_loss + neg_loss + semipos_loss,
        'pos_loss': pos_loss,
        'neg_loss': neg_loss,
        'semipos_loss': semipos_loss}


def log_batch_metrics(writer, loss_dict, lr_scheduler, pairs, step_idx, epoch_idx, batch_idx, quiet):
    writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step=step_idx)
    writer.add_scalar("train/num_positive_pairs", len(pairs.positive_pairs), global_step=step_idx)
    writer.add_scalar("train/num_semipos_pairs", len(pairs.semipositive_pairs), global_step=step_idx)
    writer.add_scalar("train/num_neg_pairs", len(pairs.negative_pairs), global_step=step_idx)
    writer.add_scalar("train/loss_pos", loss_dict["pos_loss"].item(), global_step=step_idx)
    writer.add_scalar("train/loss_semipos", loss_dict["semipos_loss"].item(), global_step=step_idx)
    writer.add_scalar("train/loss_neg", loss_dict["neg_loss"].item(), global_step=step_idx)
    writer.add_scalar("train/loss", loss_dict["loss"].item(), global_step=step_idx)
    if not quiet:
        print(f"{epoch_idx=:4d} {batch_idx=:4d} lr: {lr_scheduler.get_last_lr()[0]:.2e} " +
              f" num_pos_pairs: {len(pairs.positive_pairs):3d}" +
              f" num_semipos_pairs: {len(pairs.semipositive_pairs):3d}" +
              f" num_neg_pairs: {len(pairs.negative_pairs):3d}" +
              f" pos_loss: {loss_dict["pos_loss"].item():0.6f}" +
              f" semipos_loss: {loss_dict["semipos_loss"]:0.6f}" +
              f" neg_loss: {loss_dict["neg_loss"].item():0.6f}" +
              f" loss: {loss_dict["loss"].item():0.6f}",
              end='\r')
        if batch_idx % 50 == 0:
            print()


def compute_validation_metrics(
        sat_model,
        pano_model,
        validation_datasets):
    out = {}
    for name, dataset in validation_datasets.items():
        sat_embeddings = sed.build_satellite_db(
            sat_model,
            vigor_dataset.get_dataloader(dataset.get_sat_patch_view(), batch_size=64, num_workers=8))
        pano_embeddings = sed.build_panorama_db(
            pano_model,
            vigor_dataset.get_dataloader(dataset.get_pano_view(), batch_size=64, num_workers=8))
        similarity = pano_embeddings @ sat_embeddings.T
        num_panos = similarity.shape[0]

        invalid_mask = torch.ones((num_panos, 5), dtype=torch.bool)
        sat_idxs = torch.zeros((num_panos, 5), dtype=torch.int32)
        for pano_idx, pano_metadata in dataset._panorama_metadata.iterrows():
            assert len(pano_metadata.positive_satellite_idxs) <= 1
            assert len(pano_metadata.semipositive_satellite_idxs) <= 4

            for col_idx, sat_idx in enumerate(pano_metadata.positive_satellite_idxs):
                sat_idxs[pano_idx, col_idx] = sat_idx
                invalid_mask[pano_idx, col_idx] = False

            for col_idx, sat_idx in enumerate(pano_metadata.semipositive_satellite_idxs):
                sat_idxs[pano_idx, col_idx+1] = sat_idx
                invalid_mask[pano_idx, col_idx+1] = False

        row_idxs = torch.arange(num_panos).reshape(-1, 1).expand(-1, 5)
        pos_semipos_similarities = similarity[row_idxs, sat_idxs]
        pos_semipos_similarities[invalid_mask] = torch.nan

        # Since the invalid entries are set to nan, their ranks are set to zero
        ranks = (similarity[:, None, :] >= pos_semipos_similarities[:, :, None]).sum(dim=-1)

        #  Compute the mean reciprocal rank of the valid positive matches
        positive_recip_ranks = 1.0 / ranks[:, 0]
        positive_recip_ranks[invalid_mask[:, 0]] = torch.nan
        positive_mean_recip_rank = torch.nanmean(positive_recip_ranks)

        # Compute the mean reciprocal rank of the lowest ranked positive/semipositive matches
        # All panoramas should have at least one positive or semipostive match, so the max
        # rank is guaranteed to be greater than zero, to the reciprocal rank should be valid
        max_pos_semi_pos_recip_ranks = 1.0 / ranks.max(dim=-1).values
        max_pos_semi_pos_recip_ranks = torch.nanmean(max_pos_semi_pos_recip_ranks)

        # compute the positive recall @ K
        k_values = [1, 5, 10, 100]
        pos_recall = {f"{name}/pos_recall@{k}": (ranks[~(invalid_mask[:, 0]), 0] <= k).float().mean().item()
                      for k in k_values}

        # compute the positive/semipositive recall @ K
        invalid_mask_cuda = invalid_mask.cuda()
        any_pos_semipos_recall = {
                f"{name}/any pos_semipos_recall@{k}": ((ranks <= k) & (~invalid_mask_cuda)).any(dim=-1).float().mean().item()
                for k in k_values}

        all_pos_semipos_recall = {
                f"{name}/all pos_semipos_recall@{k}": (ranks <= k).all(dim=-1).float().mean().item()
                for k in k_values[1:]}

        out |= ({
            f"{name}/positive_mean_recip_rank": positive_mean_recip_rank.item(),
            f"{name}/max_pos_semi_pos_recip_rank": max_pos_semi_pos_recip_ranks.item()}
            | pos_recall
            | any_pos_semipos_recall
            | all_pos_semipos_recall)
    return out


def log_validation_metrics(writer, validation_metrics, epoch_idx, quiet):
    to_print = []
    for key, value in validation_metrics.items():
        to_print.append(f'{key}: {value:0.3f}')
        writer.add_scalar(f"validation/{key}", value, global_step=epoch_idx)

    if not quiet:
        print(f"epoch_idx: {epoch_idx} {' '.join(to_print)}")


def train(config: TrainConfig, *, dataset, validation_datasets, panorama_model, satellite_model, quiet):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # save config:
    config_json = msgspec.json.encode(config, enc_hook=enc_hook)
    config_dict = json.loads(config_json)
    with open(config.output_dir / "config.json", 'wb') as f:
        f.write(config_json)

    with open(config.output_dir / "satellite_model.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config.sat_model_config, enc_hook=enc_hook))

    with open(config.output_dir / "panorama_model.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config.pano_model_config, enc_hook=enc_hook))

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

    print(f"working with train dataset {len(dataset._satellite_metadata)=}" +
          f" {len(dataset._panorama_metadata)=} {len(dataset._landmark_metadata)=}")
    for name, val_dataset in validation_datasets.items():
        print(f"working with validation dataset {name} {len(val_dataset._satellite_metadata)=}" +
              f" {len(val_dataset._panorama_metadata)=} {len(val_dataset._landmark_metadata)=}")

    opt_config = config.opt_config

    miner = vigor_dataset.HardNegativeMiner(
            batch_size=opt_config.batch_size,
            embedding_dimension=panorama_model.output_dim,
            random_sample_type=opt_config.random_sample_type,
            hard_negative_pool_size=opt_config.hard_negative_pool_size,
            dataset=dataset)
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_sampler=miner, num_workers=24, persistent_workers=True)

    opt = torch.optim.Adam(
        list(panorama_model.parameters()) + list(satellite_model.parameters()),
        lr=opt_config.lr_schedule.initial_lr
    )

    warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        opt,
        factor=opt_config.lr_schedule.warmup_factor,
        total_iters=opt_config.lr_schedule.num_warmup_epochs)
    step_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=opt_config.lr_schedule.num_epochs_at_lr,
            gamma=opt_config.lr_schedule.lr_step_factor)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_lr_scheduler, step_lr_scheduler],
        milestones=[opt_config.lr_schedule.num_warmup_epochs])

    grad_scaler = torch.amp.GradScaler()

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

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                panorama_embeddings = panorama_model(
                        panorama_model.model_input_from_batch(batch).to("cuda"))
                sat_embeddings = satellite_model(
                        satellite_model.model_input_from_batch(batch).to("cuda"))

                loss_dict = compute_loss(
                        pano_embeddings=panorama_embeddings,
                        sat_embeddings=sat_embeddings,
                        pairs=pairs,
                        loss_config=opt_config.loss_config)

            grad_scaler.scale(loss_dict["loss"]).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

            # Hard Negative Mining
            miner.consume(
                    panorama_embeddings=panorama_embeddings.detach(),
                    satellite_embeddings=sat_embeddings.detach(),
                    batch=batch)

            # Logging
            log_batch_metrics(
                    writer=writer,
                    loss_dict=loss_dict,
                    lr_scheduler=lr_scheduler,
                    pairs=pairs,
                    step_idx=total_batches,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    quiet=quiet)

            total_batches += 1
        if not quiet:
            print()
        lr_scheduler.step()

        if epoch_idx >= opt_config.enable_hard_negative_sampling_after_epoch_idx:
            miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE)

        if miner.sample_mode == vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE:
            # Since we are hard negative mining, we want to update the embedding vectors for any
            # satellite patches that were not observed as part of the epoch
            unobserved_patch_dataset = torch.utils.data.Subset(
                dataset.get_sat_patch_view(), list(miner.unobserved_sat_idxs))
            unobserved_dataloader = vigor_dataset.get_dataloader(
                unobserved_patch_dataset, num_workers=8, batch_size=128)

            for batch in tqdm.tqdm(unobserved_dataloader, desc="Unobserved sat batches"):
                with torch.no_grad():
                    sat_embeddings = satellite_model(
                        satellite_model.model_input_from_batch(batch).to("cuda"))
                miner.consume(None, sat_embeddings, batch)

        # compute validation set metrics
        validation_metrics = compute_validation_metrics(
                sat_model=satellite_model,
                pano_model=panorama_model,
                validation_datasets=validation_datasets)
        log_validation_metrics(
                writer=writer,
                validation_metrics=validation_metrics,
                epoch_idx=epoch_idx,
                quiet=quiet)

        if (epoch_idx % 10 == 0) or (epoch_idx == opt_config.num_epochs - 1):
            # Periodically save the model
            config.output_dir.mkdir(parents=True, exist_ok=True)
            panorama_model_path = config.output_dir / f"{epoch_idx:04d}_panorama"
            satellite_model_path = config.output_dir / f"{epoch_idx:04d}_satellite"

            save_dataloader = vigor_dataset.get_dataloader(dataset, batch_size=16)
            batch = next(iter(save_dataloader))
            save_model(panorama_model, panorama_model_path,
                       (panorama_model.model_input_from_batch(batch).to("cuda"),))

            save_model(satellite_model, satellite_model_path,
                       (satellite_model.model_input_from_batch(batch).to("cuda"),))


def main(
        dataset_base_path: Path,
        output_base_path: Path,
        train_config_path: Path,
        quiet: bool):
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(file_in.read(), type=TrainConfig, dec_hook=dec_hook)
    pprint(train_config)

    if isinstance(train_config.sat_model_config, patch_embedding.WagPatchEmbeddingConfig):
        satellite_model = patch_embedding.WagPatchEmbedding(train_config.sat_model_config)
    elif isinstance(train_config.sat_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        satellite_model = swag_patch_embedding.SwagPatchEmbedding(train_config.sat_model_config)

    if isinstance(train_config.pano_model_config, patch_embedding.WagPatchEmbeddingConfig):
        panorama_model = patch_embedding.WagPatchEmbedding(train_config.pano_model_config)
    elif isinstance(train_config.pano_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        panorama_model = swag_patch_embedding.SwagPatchEmbedding(train_config.pano_model_config)

    assert len(train_config.dataset_config.paths) == 1
    dataset_config = vigor_dataset.VigorDatasetConfig(
        satellite_patch_size=train_config.sat_model_config.patch_dims,
        panorama_size=train_config.pano_model_config.patch_dims,
        satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_key=train_config.dataset_config.paths[0],
            model_type="satellite",
            hash_and_key=satellite_model.cache_info()),
        panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_key=train_config.dataset_config.paths[0],
            model_type="panorama",
            hash_and_key=panorama_model.cache_info()),
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
        factor=train_config.dataset_config.factor,
        should_load_images=train_config.dataset_config.should_load_images)

    dataset_paths = [dataset_base_path / p for p in train_config.dataset_config.paths]
    dataset = vigor_dataset.VigorDataset(dataset_paths, dataset_config)

    validation_datasets = {}
    for validation_dataset_config in train_config.validation_dataset_configs:
        assert len(validation_dataset_config.paths) == 1
        validation_dataset_paths = [dataset_base_path / p for p in validation_dataset_config.paths]
        validation_datasets[validation_dataset_paths[0].name] = vigor_dataset.VigorDataset(
            validation_dataset_paths,
            vigor_dataset.VigorDatasetConfig(
                satellite_patch_size=satellite_model.patch_dims,
                panorama_size=panorama_model.patch_dims,
                satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_key=validation_dataset_config.paths[0],
                    model_type="satellite",
                    hash_and_key=satellite_model.cache_info()),
                panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_key=validation_dataset_config.paths[0],
                    model_type="panorama",
                    hash_and_key=panorama_model.cache_info()),
                sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
                factor=validation_dataset_config.factor,
                should_load_images=train_config.dataset_config.should_load_images))

    output_dir = output_base_path / train_config.output_dir
    tensorboard_output = train_config.tensorboard_output
    tensorboard_output = tensorboard_output if tensorboard_output is not None else output_dir / "logs"

    train_config.output_dir = output_dir
    train_config.tensorboard_output = tensorboard_output

    with ipdb.launch_ipdb_on_exception():
        train(
            train_config,
            dataset=dataset,
            validation_datasets=validation_datasets,
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
