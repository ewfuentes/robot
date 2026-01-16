import argparse
import json
import os
import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
from torch.utils.tensorboard import SummaryWriter
import itertools
from pathlib import Path
from common.python.serialization import flatten_dict, msgspec_enc_hook, msgspec_dec_hook
from experimental.overhead_matching.swag.scripts.losses import LossConfig, compute_loss, LossFunctionType, create_losses_from_loss_config_list, InfoNCELossConfig
from experimental.overhead_matching.swag.scripts.distances import DistanceConfig, create_distance_from_config
from experimental.overhead_matching.swag.scripts.pairing import PairingType, create_pairs, create_anchors, Pairs, PairingDataType
from experimental.overhead_matching.swag.data import (
    vigor_dataset, satellite_embedding_database as sed)
from experimental.overhead_matching.swag.model import (
    patch_embedding, swag_patch_embedding)
from experimental.overhead_matching.swag.model.swag_config_types import ExtractorDataRequirement
from experimental.overhead_matching.swag.model.swag_model_input_output import derive_data_requirements_from_model
from experimental.overhead_matching.swag.model.landmark_scheduler import (
    LandmarkDropoutScheduleConfig,
    LandmarkDropoutScheduler,
)
from experimental.overhead_matching.swag.scripts.logging_utils import (
    log_batch_metrics, log_embedding_stats, log_gradient_stats, log_validation_metrics, log_feature_counts)
from experimental.overhead_matching.swag.scripts.model_inspector import ModelInspector
from typing import Union
from dataclasses import dataclass
import pandas as pd
import tqdm
import msgspec
from pprint import pprint
import ipdb
from contextlib import nullcontext
import datetime
import random
import numpy as np
from experimental.overhead_matching.swag.scripts.lr_sweep import LearningRateSweepConfig, run_lr_sweep


def setup_reproducibility(seed: int | None) -> torch.Generator | None:
    """Set up reproducible training by seeding all RNGs.

    Note: This does NOT set torch.backends.cudnn.deterministic to preserve
    training performance. Some minor non-determinism may remain in CUDA operations.

    Args:
        seed: Random seed for reproducibility. If None, uses default non-deterministic behavior.

    Returns:
        A seeded torch.Generator if seed is provided, None otherwise.
    """
    if seed is None:
        return None

    print(f"Setting random seed to {seed} for reproducible training")


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Note: NOT setting cudnn.deterministic to preserve performance
    # Most randomness is controlled, but some CUDA ops may have minor variance

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def debug_log(message: str, log_file: str = "/tmp/training_debug.log"):
    """Write log message to file with timestamp and flush. Also prints to stdout."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_msg = f"[DEBUG {timestamp}] {message}"

    # Print to stdout
    print(log_msg, flush=True)

    # Write to debug log file with flush
    try:
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    except Exception as e:
        print(f"Warning: Failed to write debug log: {e}", flush=True)


@dataclass
class LearningRateSchedule:
    initial_lr: float
    lr_step_factor: float
    num_epochs_at_lr: int

    warmup_factor: float
    num_warmup_epochs: int


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

    lr_sweep_config: LearningRateSweepConfig | None = None


ModelConfig = Union[patch_embedding.WagPatchEmbeddingConfig,
                    swag_patch_embedding.SwagPatchEmbeddingConfig]


@dataclass
class DatasetConfig:
    paths: list[Path]
    landmark_version: str
    factor: None | float = 1.0
    panorama_landmark_radius_px: float = 640
    landmark_correspondence_inflation_factor: float = 1.0
    require_proper_noun_match: bool = False
    pano_gemini_base_path: str | None = None


@dataclass
class TrainConfig:
    opt_config: OptimizationConfig
    sat_model_config: ModelConfig
    pano_model_config: ModelConfig
    distance_model_config: DistanceConfig
    dataset_config: DatasetConfig
    validation_dataset_configs: list[DatasetConfig]
    loss_configs: list[LossConfig]
    output_dir: Path
    tensorboard_output: Path | None
    pano_landmark_dropout_schedules: list[LandmarkDropoutScheduleConfig] = None
    sat_landmark_dropout_schedules: list[LandmarkDropoutScheduleConfig] = None
    seed: int | None = None

@torch.no_grad
def validation_metrics_from_similarity(
    name: str,
    similarity: torch.Tensor,  # num_panos x num_sat
    panorama_metadata: pd.DataFrame,
) -> dict:
    num_panos = similarity.shape[0]

    invalid_mask = torch.ones((num_panos, 5), dtype=torch.bool)
    sat_idxs = torch.zeros((num_panos, 5), dtype=torch.int32)
    for pano_idx, pano_metadata in panorama_metadata.iterrows():
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
    ranks_cuda = ranks.cuda()
    any_pos_semipos_recall = {
        f"{name}/any pos_semipos_recall@{k}": ((ranks_cuda <= k) & (~invalid_mask_cuda)).any(dim=-1).float().mean().item()
        for k in k_values}

    all_pos_semipos_recall = {
        f"{name}/all pos_semipos_recall@{k}": (ranks_cuda <= k).all(dim=-1).float().mean().item()
        for k in k_values[1:]}

    out = ({
        f"{name}/positive_mean_recip_rank": positive_mean_recip_rank.item(),
        f"{name}/max_pos_semi_pos_recip_rank": max_pos_semi_pos_recip_ranks.item()}
        | pos_recall
        | any_pos_semipos_recall
        | all_pos_semipos_recall)
    return out

@torch.no_grad
def compute_validation_metrics(
        sat_model,
        pano_model,
        validation_datasets,
        distance_model: torch.nn.Module,
        quiet: bool = False,
):
    out = {}
    for name, dataset in validation_datasets.items():
        sat_embeddings = sed.build_satellite_db(
            sat_model,
            vigor_dataset.get_dataloader(dataset.get_sat_patch_view(), batch_size=64, num_workers=8),
            verbose=not quiet)
        pano_embeddings = sed.build_panorama_db(
            pano_model,
            vigor_dataset.get_dataloader(dataset.get_pano_view(), batch_size=64, num_workers=8),
            verbose=not quiet)
        similarity = distance_model(
            pano_embeddings_unnormalized=pano_embeddings,
            sat_embeddings_unnormalized=sat_embeddings
        )
        similarity = similarity.to("cpu")
        out |= validation_metrics_from_similarity(name, similarity, panorama_metadata=dataset._panorama_metadata)

    return out


def setup_models_for_training(panorama_model, satellite_model, distance_model):
    """Move models to GPU and set to training mode."""
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()
    panorama_model.train()
    satellite_model.train()
    distance_model = distance_model.cuda()
    distance_model.train()
    return panorama_model, satellite_model, distance_model


def save_checkpoint(
    output_dir: Path,
    checkpoint_name: str,
    panorama_model,
    satellite_model,
    distance_model,
    dataset,
    remove_existing: bool = False,
):
    """Save a checkpoint with the given name prefix."""
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    panorama_model_path = output_dir / f"{checkpoint_name}_panorama"
    satellite_model_path = output_dir / f"{checkpoint_name}_satellite"
    distance_model_path = output_dir / f"{checkpoint_name}_distance"

    if remove_existing:
        for path in [panorama_model_path, satellite_model_path, distance_model_path]:
            if path.exists():
                shutil.rmtree(path)

    save_dataloader = vigor_dataset.get_dataloader(dataset, batch_size=16)
    batch = next(iter(save_dataloader))
    pano_model_input = panorama_model.model_input_from_batch(batch).to("cuda")
    sat_model_input = satellite_model.model_input_from_batch(batch).to("cuda")
    save_model(panorama_model, panorama_model_path, (pano_model_input,))
    save_model(satellite_model, satellite_model_path, (sat_model_input,))
    if sum(param.numel() for param in distance_model.parameters()) > 0:
        sat_emb, _ = satellite_model(sat_model_input)
        pano_emb, _ = panorama_model(pano_model_input)
        save_model(distance_model, distance_model_path, (sat_emb, pano_emb))


def create_training_components(dataset,
                               panorama_model,
                               satellite_model,
                               distance_model,
                               opt_config,
                               generator: torch.Generator | None = None):
    """Create miner, dataloader, and optimizer for training."""
    # Create miner and dataloader
    # Note: embeddings stored on CPU to save GPU memory, transferred to GPU as needed
    miner = vigor_dataset.HardNegativeMiner(
        batch_size=opt_config.batch_size,
        num_pano_embeddings=panorama_model.num_embeddings,
        num_sat_embeddings=satellite_model.num_embeddings,
        distance_model=distance_model,
        embedding_dimension=panorama_model.output_dim,
        random_sample_type=opt_config.random_sample_type,
        hard_negative_pool_size=opt_config.hard_negative_pool_size,
        dataset=dataset,
        device='cpu',
        generator=generator)

    num_workers = int(os.environ.get("MAX_DATALOADER_WORKERS", min(os.cpu_count() // 2, 24)))
    # Pass seed to training dataloader for reproducibility (if generator is provided)
    worker_seed = generator.initial_seed() if generator is not None else None
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_sampler=miner, num_workers=num_workers, persistent_workers=(num_workers > 0),
        worker_seed=worker_seed)

    # Create optimizer
    opt = torch.optim.AdamW(
        list(panorama_model.parameters()) + list(satellite_model.parameters()) +
        list(distance_model.parameters()),
        lr=opt_config.lr_schedule.initial_lr
    )

    return miner, dataloader, opt


def compute_forward_pass_and_loss(batch,
                                  panorama_model,
                                  satellite_model,
                                  distance_model,
                                  pairing_data: PairingDataType,
                                  loss_functions: list[LossFunctionType],
                                  pano_dropout_scheduler=None,
                                  sat_dropout_scheduler=None,
                                  ):
    """Compute forward pass and loss for a batch."""

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        panorama_embeddings, pano_debug = panorama_model(
            panorama_model.model_input_from_batch(batch).to("cuda"),
            pano_dropout_scheduler)
        sat_embeddings, sat_debug = satellite_model(
            satellite_model.model_input_from_batch(batch).to("cuda"),
            sat_dropout_scheduler)

        similarity = distance_model(
            sat_embeddings_unnormalized=sat_embeddings,
            pano_embeddings_unnormalized=panorama_embeddings
        )
        loss_dict = compute_loss(
            pano_embeddings=panorama_embeddings,
            sat_embeddings=sat_embeddings,
            similarity=similarity,
            pairing_data=pairing_data,
            loss_functions=loss_functions,
        )

    return loss_dict, panorama_embeddings, sat_embeddings, {'sat': sat_debug, 'pano': pano_debug}


def train(config: TrainConfig,
          *,
          output_dir: Path,
          dataset,
          validation_datasets,
          panorama_model,
          satellite_model,
          quiet,
          capture_model_data: bool = False,
          num_batches_to_capture: int = 10,
          generator: torch.Generator | None = None):

    output_dir.mkdir(parents=True, exist_ok=True)
    # save config:
    config_json = msgspec.json.encode(config, enc_hook=msgspec_enc_hook)
    config_dict = json.loads(config_json)

    with open(output_dir / "train_config.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config, enc_hook=msgspec_enc_hook))

    writer = SummaryWriter(
        log_dir=config.tensorboard_output
    )
    # write hyperparameters
    writer.add_hparams(
        flatten_dict(config_dict['opt_config']), {},
        run_name="."
    )

    distance_model = create_distance_from_config(config.distance_model_config)
    loss_functions = create_losses_from_loss_config_list(config.loss_configs)
    # Setup models using extracted function
    panorama_model, satellite_model, distance_model = setup_models_for_training(
        panorama_model, satellite_model, distance_model)

    print(f"working with train dataset {len(dataset._satellite_metadata)=}" +
          f" {len(dataset._panorama_metadata)=}" + (" No landmarks" if dataset._landmark_metadata is None else f"{len(dataset._landmark_metadata)=}"))
    for name, val_dataset in validation_datasets.items():
        print(f"working with validation dataset {name} {len(val_dataset._satellite_metadata)=}" +
              f" {len(val_dataset._panorama_metadata)=}" + (" No landmarks" if dataset._landmark_metadata is None else f"{len(dataset._landmark_metadata)=}"))

    opt_config = config.opt_config

    # Create training components using extracted function
    miner, dataloader, opt = create_training_components(
        dataset, panorama_model, satellite_model, distance_model, opt_config, generator)

    # Create landmark dropout schedulers if configured
    pano_dropout_scheduler = None
    if config.pano_landmark_dropout_schedules:
        pano_dropout_scheduler = LandmarkDropoutScheduler(
            config.pano_landmark_dropout_schedules,
            opt_config.num_epochs
        )

    sat_dropout_scheduler = None
    if config.sat_landmark_dropout_schedules:
        sat_dropout_scheduler = LandmarkDropoutScheduler(
            config.sat_landmark_dropout_schedules,
            opt_config.num_epochs
        )

    # Create model inspector if requested
    inspector = None
    if capture_model_data:
        inspector = ModelInspector(
            output_dir=output_dir,
            num_batches_to_capture=num_batches_to_capture)

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

    pairing_type = PairingType.PAIRS

    for loss_config in config.loss_configs:
        if isinstance(loss_config, InfoNCELossConfig):
            pairing_type = PairingType.ANCHOR_SETS

    # Track best model based on validation MRR (higher is better)
    # If no non-training validation datasets, fall back to loss (lower is better)
    best_metric = None
    best_epoch = -1

    # Identify non-training validation datasets for best model selection
    # Convert training paths to strings for comparison with validation dataset names
    training_city_names = [str(p) for p in config.dataset_config.paths]
    non_training_val_datasets = [
        name for name in validation_datasets.keys()
        if name not in training_city_names
    ]
    use_mrr_for_best = len(non_training_val_datasets) > 0
    if use_mrr_for_best:
        print(f"Using MRR for best model selection. Datasets included in average: {non_training_val_datasets}")
    else:
        print(f"No non-training validation datasets found. Using loss for best model selection.")

    total_batches = 0
    for epoch_idx in tqdm.tqdm(range(opt_config.num_epochs),  desc="Epoch", disable=quiet):
        debug_log(f"Starting epoch {epoch_idx}")

        # Update epoch for dropout schedulers
        if pano_dropout_scheduler is not None:
            pano_dropout_scheduler.set_epoch(epoch_idx)
        if sat_dropout_scheduler is not None:
            sat_dropout_scheduler.set_epoch(epoch_idx)
        for batch_idx, batch in enumerate(dataloader):
            match pairing_type:
                case PairingType.PAIRS:
                    pairing_data = create_pairs(
                        batch.panorama_metadata,
                        batch.satellite_metadata
                    )

                    if opt_config.random_sample_type == vigor_dataset.HardNegativeMiner.RandomSampleType.NEAREST:
                        pairing_data = Pairs(
                            positive_pairs=pairing_data.positive_pairs + pairing_data.semipositive_pairs,
                            semipositive_pairs=[],
                            negative_pairs=pairing_data.negative_pairs)
                case PairingType.ANCHOR_SETS:
                    pairing_data = create_anchors(
                        batch.panorama_metadata,
                        batch.satellite_metadata,
                        use_pano_as_anchor=False
                    )
                case _:
                    raise RuntimeError(f"Pairing type not recongnized, {pairing_type}")
            opt.zero_grad()

            # Use extracted function for forward pass and loss
            loss_dict, panorama_embeddings, satellite_embeddings, debug_dict = compute_forward_pass_and_loss(
                batch=batch,
                panorama_model=panorama_model,
                satellite_model=satellite_model,
                distance_model=distance_model,
                pairing_data=pairing_data,
                loss_functions=loss_functions,
                pano_dropout_scheduler=pano_dropout_scheduler,
                sat_dropout_scheduler=sat_dropout_scheduler)

            # Capture model inputs and extractor outputs if inspector is enabled
            if inspector is not None and inspector.should_capture(total_batches):
                pano_input = panorama_model.model_input_from_batch(batch)
                sat_input = satellite_model.model_input_from_batch(batch)
                # Use the extractor outputs returned from the forward pass
                pano_extractor_outputs = debug_dict['pano']
                sat_extractor_outputs = debug_dict['sat']
                inspector.capture(
                    pano_input=pano_input,
                    sat_input=sat_input,
                    pano_extractor_outputs=pano_extractor_outputs,
                    sat_extractor_outputs=sat_extractor_outputs,
                    pairing_data=pairing_data,
                    batch_idx=batch_idx,
                    epoch_idx=epoch_idx,
                    total_batches=total_batches)

            grad_scaler.scale(loss_dict["loss"]).backward()
            grad_scaler.step(opt)
            grad_scaler.update()
            if torch.isnan(loss_dict["loss"]):
                raise RuntimeError("Got NaN loss!")
            # perform checks that all parameters we expect to update have gradients for the first set of batches
            if total_batches < 50:
                for model_name, model in zip(["pano", "sat"], [panorama_model, satellite_model]):
                    for name, param in model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            raise RuntimeError(
                                f"Parameter {name} for model {model_name} requires grad, but had no update.")
                        if param.grad is not None and torch.any(torch.isinf(param.grad)):
                            print(
                                f"Warining: INF was found in parameter gradient: {name} in model {model_name}")

            log_gradient_stats(writer, panorama_model, "panorama", total_batches)
            log_gradient_stats(writer, satellite_model, "satellite", total_batches)
            log_embedding_stats(writer, "pano", panorama_embeddings.detach(), total_batches)
            log_embedding_stats(writer, "sat", satellite_embeddings.detach(), total_batches)
            log_feature_counts(writer, debug_dict['pano'], debug_dict['sat'], total_batches)

            # Hard Negative Mining
            miner.consume(
                panorama_embeddings=panorama_embeddings.detach(),
                satellite_embeddings=satellite_embeddings.detach(),
                batch=batch)

            # Logging
            log_batch_metrics(
                writer=writer,
                loss_dict=loss_dict,
                lr_scheduler=lr_scheduler,
                pairing_data=pairing_data,
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

            for batch in tqdm.tqdm(unobserved_dataloader, desc="Unobserved sat batches", disable=quiet):
                with torch.no_grad():
                    miner_satellite_embeddings, _ = satellite_model(
                        satellite_model.model_input_from_batch(batch).to("cuda"))
                miner.consume(None, miner_satellite_embeddings, batch)

        # compute validation set metrics
        debug_log(f"Computing validation metrics for epoch {epoch_idx}")
        validation_metrics = compute_validation_metrics(
            sat_model=satellite_model,
            pano_model=panorama_model,
            validation_datasets=validation_datasets,
            distance_model=distance_model,
            quiet=quiet)
        log_validation_metrics(
            writer=writer,
            validation_metrics=validation_metrics,
            epoch_idx=epoch_idx,
            quiet=quiet)

        # Check if this is the best model
        is_best = False
        if use_mrr_for_best:
            # Compute average MRR across non-training validation datasets
            mrr_values = []
            for name in non_training_val_datasets:
                key = f"{name}/positive_mean_recip_rank"
                if key not in validation_metrics:
                    raise KeyError(f"Expected validation metric '{key}' not found in validation_metrics")
                mrr_values.append(validation_metrics[key])
            current_metric = sum(mrr_values) / len(mrr_values)
            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch_idx
                is_best = True
                debug_log(f"New best model at epoch {epoch_idx} with avg validation MRR: {current_metric:.4f}")
        else:
            # Fall back to using loss (lower is better)
            # We need to track average loss over the epoch - use the last batch's loss as proxy
            current_metric = loss_dict["loss"].item()
            if best_metric is None or current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch_idx
                is_best = True
                debug_log(f"New best model at epoch {epoch_idx} with loss: {current_metric:.4f}")

        # Save best model if this is the best so far
        if is_best:
            debug_log(f"Saving best model (epoch {epoch_idx})")
            save_checkpoint(
                output_dir=output_dir,
                checkpoint_name="best",
                panorama_model=panorama_model,
                satellite_model=satellite_model,
                distance_model=distance_model,
                dataset=dataset,
                remove_existing=True,
            )

        # Periodic checkpoint every 50 epochs
        if epoch_idx > 0 and epoch_idx % 50 == 0:
            debug_log(f"Saving periodic checkpoint for epoch {epoch_idx}")
            save_checkpoint(
                output_dir=output_dir,
                checkpoint_name=f"{epoch_idx:04d}",
                panorama_model=panorama_model,
                satellite_model=satellite_model,
                distance_model=distance_model,
                dataset=dataset,
            )

    # Always save the last model
    debug_log(f"Saving last model (epoch {opt_config.num_epochs - 1})")
    save_checkpoint(
        output_dir=output_dir,
        checkpoint_name="last",
        panorama_model=panorama_model,
        satellite_model=satellite_model,
        distance_model=distance_model,
        dataset=dataset,
    )


    # Signal training completion
    debug_log("🎉 TRAINING COMPLETED SUCCESSFULLY 🎉")
    debug_log("Training process exiting normally")



def main(
        dataset_base_path: Path,
        output_base_path: Path,
        train_config_path: Path,
        quiet: bool,
        no_ipdb: bool,
        lr_sweep: bool = False,
        capture_model_data: bool = False,
        num_batches_to_capture: int = 10,
        seed: int | None = None,
):

    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(file_in.read(), type=TrainConfig, dec_hook=msgspec_dec_hook)
    pprint(train_config)
    if train_config.seed is not None and seed is not None:
        raise RuntimeError(f"Got seed in both train config ({train_config.seed}) and through args ({seed}). Only set it in one place")
    
    train_config.seed = seed
    # Set up reproducibility first, before any model/dataset initialization
    generator = setup_reproducibility(train_config.seed)

    if isinstance(train_config.sat_model_config, patch_embedding.WagPatchEmbeddingConfig):
        satellite_model = patch_embedding.WagPatchEmbedding(train_config.sat_model_config)
    elif isinstance(train_config.sat_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        satellite_model = swag_patch_embedding.SwagPatchEmbedding(train_config.sat_model_config)
    else:
        raise TypeError("Unsupported satellite model config type")

    if isinstance(train_config.pano_model_config, patch_embedding.WagPatchEmbeddingConfig):
        panorama_model = patch_embedding.WagPatchEmbedding(train_config.pano_model_config)
    elif isinstance(train_config.pano_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        panorama_model = swag_patch_embedding.SwagPatchEmbedding(train_config.pano_model_config)
    else:
        raise TypeError("Unsupported panorama model config type")

    # Derive data requirements from both models
    sat_requirements = derive_data_requirements_from_model(
        satellite_model,
        use_cached_extractors=getattr(train_config.sat_model_config, 'use_cached_extractors', []))
    pano_requirements = derive_data_requirements_from_model(
        panorama_model,
        use_cached_extractors=getattr(train_config.pano_model_config, 'use_cached_extractors', []))
    all_requirements = sat_requirements | pano_requirements

    # Determine what to load based on requirements
    should_load_images = ExtractorDataRequirement.IMAGES in all_requirements
    should_load_landmarks = ExtractorDataRequirement.LANDMARKS in all_requirements
    # Force landmarks to load if any dataset uses proper noun filtering
    if train_config.dataset_config.require_proper_noun_match:
        should_load_landmarks = True
    for val_cfg in train_config.validation_dataset_configs:
        if val_cfg.require_proper_noun_match:
            should_load_landmarks = True

    dataset_config = vigor_dataset.VigorDatasetConfig(
        satellite_patch_size=train_config.sat_model_config.patch_dims,
        panorama_size=train_config.pano_model_config.patch_dims,
        satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=train_config.dataset_config.paths,
            model_type="satellite",
            landmark_version=train_config.dataset_config.landmark_version,
            panorama_landmark_radius_px=train_config.dataset_config.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=train_config.dataset_config.landmark_correspondence_inflation_factor,
            extractor_info=satellite_model.cache_info()),
        panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=train_config.dataset_config.paths,
            model_type="panorama",
            landmark_version=train_config.dataset_config.landmark_version,
            panorama_landmark_radius_px=train_config.dataset_config.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=train_config.dataset_config.landmark_correspondence_inflation_factor,
            extractor_info=panorama_model.cache_info()),
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
        factor=train_config.dataset_config.factor,
        should_load_images=should_load_images,
        should_load_landmarks=should_load_landmarks,
        landmark_version=train_config.dataset_config.landmark_version,
        load_cache_debug=capture_model_data,
        panorama_landmark_radius_px=train_config.dataset_config.panorama_landmark_radius_px,
        landmark_correspondence_inflation_factor=train_config.dataset_config.landmark_correspondence_inflation_factor,
        require_proper_noun_match=train_config.dataset_config.require_proper_noun_match,
        pano_gemini_base_path=train_config.dataset_config.pano_gemini_base_path)

    dataset_paths = [dataset_base_path / p for p in train_config.dataset_config.paths]
    dataset = vigor_dataset.VigorDataset(dataset_paths, dataset_config)

    validation_datasets = {}
    for validation_dataset_config in train_config.validation_dataset_configs:
        assert len(validation_dataset_config.paths) == 1
        validation_dataset_paths = [dataset_base_path / p for p in validation_dataset_config.paths]

        # Use same data requirements for validation datasets
        validation_datasets[validation_dataset_paths[0].name] = vigor_dataset.VigorDataset(
            validation_dataset_paths,
            vigor_dataset.VigorDatasetConfig(
                satellite_patch_size=satellite_model.patch_dims,
                panorama_size=panorama_model.patch_dims,
                satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_keys=validation_dataset_config.paths,
                    model_type="satellite",
                    landmark_version=validation_dataset_config.landmark_version,
                    panorama_landmark_radius_px=validation_dataset_config.panorama_landmark_radius_px,
                    landmark_correspondence_inflation_factor=validation_dataset_config.landmark_correspondence_inflation_factor,
                    extractor_info=satellite_model.cache_info()),
                panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_keys=validation_dataset_config.paths,
                    model_type="panorama",
                    landmark_version=validation_dataset_config.landmark_version,
                    panorama_landmark_radius_px=validation_dataset_config.panorama_landmark_radius_px,
                    landmark_correspondence_inflation_factor=validation_dataset_config.landmark_correspondence_inflation_factor,
                    extractor_info=panorama_model.cache_info()),
                sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
                factor=validation_dataset_config.factor,
                should_load_images=should_load_images,
                should_load_landmarks=should_load_landmarks,
                landmark_version=validation_dataset_config.landmark_version,
                panorama_landmark_radius_px=validation_dataset_config.panorama_landmark_radius_px,
                landmark_correspondence_inflation_factor=validation_dataset_config.landmark_correspondence_inflation_factor,
                require_proper_noun_match=validation_dataset_config.require_proper_noun_match,
                pano_gemini_base_path=validation_dataset_config.pano_gemini_base_path))

    # Add datetime prefix to output directory
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = output_base_path / f"{timestamp}_{train_config.output_dir}"
    tensorboard_output = train_config.tensorboard_output
    tensorboard_output = tensorboard_output if tensorboard_output is not None else output_dir

    train_config.output_dir = output_dir
    train_config.tensorboard_output = tensorboard_output

    # Run learning rate sweep if requested
    if lr_sweep:
        # Create default LR sweep config if not in train config
        if train_config.opt_config.lr_sweep_config is None:
            train_config.opt_config.lr_sweep_config = LearningRateSweepConfig()

        distance_model = create_distance_from_config(train_config.distance_model_config)
        optimal_lr = run_lr_sweep(
            lr_sweep_config=train_config.opt_config.lr_sweep_config,
            dataset=dataset,
            panorama_model=panorama_model,
            satellite_model=satellite_model,
            distance_model=distance_model,
            opt_config=train_config.opt_config,
            output_dir=output_dir,
            compute_forward_pass_and_loss_fn=compute_forward_pass_and_loss,
            create_training_components_fn=create_training_components,
            setup_models_for_training_fn=setup_models_for_training,
            quiet=quiet,
            generator=generator
        )

        # Update the training config to use the optimal learning rate
        train_config.opt_config.lr_schedule.initial_lr = optimal_lr
        if not quiet:
            print(f"Updated initial learning rate to {optimal_lr:.2e}")

    with ipdb.launch_ipdb_on_exception() if not no_ipdb else nullcontext():
        train(
            train_config,
            output_dir=output_dir,
            dataset=dataset,
            validation_datasets=validation_datasets,
            panorama_model=panorama_model,
            satellite_model=satellite_model,
            quiet=quiet,
            capture_model_data=capture_model_data,
            num_batches_to_capture=num_batches_to_capture,
            generator=generator)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", help="path to dataset", default="/data/overhead_matching/datasets/VIGOR")
    parser.add_argument("--output_base", help="path to output", required=True)
    parser.add_argument("--train_config", help="path to train_config", required=True)
    parser.add_argument("--no_ipdb", action="store_true",
                        help="Don't run IPDB around the training job")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--capture_model_data", action="store_true",
                        help="Capture model inputs and outputs for debugging")
    parser.add_argument("--num_batches_to_capture", type=int, default=10,
                        help="Number of batches to capture (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible training. If seed set in CLI and train_config, throws")
    args = parser.parse_args()

    main(
        Path(args.dataset_base),
        Path(args.output_base),
        Path(args.train_config),
        no_ipdb=args.no_ipdb,
        quiet=args.quiet,
        capture_model_data=args.capture_model_data,
        num_batches_to_capture=args.num_batches_to_capture,
        seed=args.seed)
