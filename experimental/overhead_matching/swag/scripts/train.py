import argparse
import json
import os
import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
from torch.utils.tensorboard import SummaryWriter
import itertools
from pathlib import Path
from common.python.serialization import flatten_dict
from experimental.overhead_matching.swag.scripts.losses import LossConfig, compute_loss
from experimental.overhead_matching.swag.scripts.distances import DistanceTypes, distance_from_type
from experimental.overhead_matching.swag.scripts.pairing import PairingType, create_pairs, create_anchors, Pairs, PairingDataType
from experimental.overhead_matching.swag.data import (
        vigor_dataset, satellite_embedding_database as sed)
from experimental.overhead_matching.swag.model import (
        patch_embedding, swag_patch_embedding)
from experimental.overhead_matching.swag.scripts.logging_utils import (
        log_batch_metrics, log_embedding_stats, log_gradient_stats, log_validation_metrics)
from typing import Union
from dataclasses import dataclass
import tqdm
import msgspec
from pprint import pprint
import ipdb
from contextlib import nullcontext
from experimental.overhead_matching.swag.scripts.lr_sweep import LearningRateSweepConfig, run_lr_sweep




@dataclass
class LearningRateSchedule:
    initial_lr: float
    lr_step_factor: float
    num_epochs_at_lr: int

    warmup_factor: float
    num_warmup_epochs: int


@dataclass
class WeightMatrixModelConfig:
    input_types: list[str]  # Options incldue "pano", "sat" or empty, in which case a single weight matrix is learned
    hidden_dim: int
    output_dim: int


class WeightMatrixModel(torch.nn.Module):
    """
    Produce a weight matrix to be used with mahalanobis distance:
    distance = (x-x').T M(x-x')

    If no input is specified, learns a non-input-conditioned matrix

    If inputs (pano/sat) are provided, creates a matrix per pano/sat embedding. 
    If both are provided, creates a matrix per pano/sat embedding pair

    Output: 
        num_pano_embeds x num_sat_embeds x num_pano_class_tokens x num_sat_class_tokens x d_emb x d_emb
        If an embedding is not part of the input (e.g., num_sat_embeds if pano is input)
        the dimension is set to 1
    """
    def __init__(self, 
                 config: WeightMatrixModelConfig):
        super().__init__()
        self.config = config
        if len(self.config.input_types) == 0:
            weight_matrix = 0.01 * (torch.rand(1, 1, 1, 1, config.output_dim, config.output_dim) - 0.5)
            weight_matrix[0, 0, 0, 0].fill_diagonal_(1.0)
            weight_matrix_semipos = torch.matmul(weight_matrix.transpose(-2, -1), weight_matrix)
            self.weight_matrix = torch.nn.Parameter(weight_matrix_semipos)
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(config.output_dim * len(self.config.input_types), self.config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_dim, config.output_dim**2)
            )
    def forward(self, 
                sat_embedding: torch.Tensor | None,
                pano_embedding: torch.Tensor | None) -> torch.Tensor:
        # if not conditional, return weight matrix
        if len(self.config.input_types) == 0:
            return self.weight_matrix
        for item in self.config.input_types:
            if item not in ["pano", "sat"]:
                raise RuntimeError(f"Invalid item in config: {item}")
        
        # otherwise, run MLP to get matrix
        embedding_dim = sat_embedding.shape[-1] if sat_embedding is not None else pano_embedding.shape[-1]
        input = []
        if "pano" in self.config.input_types and "sat" in self.config.input_types:
            assert pano_embedding.ndim == 3  #(num_pano, num_class_tokens, D_emb)
            assert sat_embedding.ndim == 3  #(num_sat, num_class_token, D_emb)
            assert sat_embedding.shape[1] == pano_embedding.shape[1]
            target_size = (pano_embedding.shape[0], 
                           sat_embedding.shape[0],
                           sat_embedding.shape[1],
                           pano_embedding.shape[-1])
            input = torch.cat([
                pano_embedding.unsqueeze(1).expand(target_size),
                sat_embedding.unsqueeze(0).expand(target_size)
            ], dim=-1)
            out_matrix = self.mlp(input)
        elif "sat" in self.config.input_types:
            assert sat_embedding.ndim == 3  #(num_sat, num_embeddings, D_emb)
            out_matrix = self.mlp(sat_embedding).unsqueeze(0).unsqueeze(2) # 1, num_sat, 1, num_sat_emb, D_emb**2
        elif "pano" in self.config.input_types:
            assert pano_embedding.ndim == 3  #(num_pano, num_embeddings, D_emb)
            out_matrix = self.mlp(pano_embedding).unsqueeze(1).unsqueeze(3)
        else:
            raise RuntimeError(f"Invalid input config {self.config.input_types}")

        out_matrix = out_matrix.unflatten(-1, (embedding_dim, embedding_dim))
        # Make matrix positive semi-definite by computing A^T @ A
        out_matrix = torch.matmul(out_matrix.transpose(-2, -1), out_matrix)
        return out_matrix

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
    loss_configs: list[LossConfig]
    distance_type: DistanceTypes
    pairing_type: PairingType = PairingType.PAIRS
    num_vector_embeddings: int = 1
    weight_matrix_model_config: WeightMatrixModelConfig | None = None

def enc_hook(obj):
    if isinstance(obj, Path):
        return str(obj)
    else:
        raise ValueError(f"Unhandled Value: {obj}")


def dec_hook(type, obj):
    if type is Path:
        return Path(obj)
    raise ValueError(f"Unhandled type: {type=} {obj=}")


def compute_validation_metrics(
        sat_model,
        pano_model,
        validation_datasets,
        distance_type: DistanceTypes,
        weight_model: WeightMatrixModel | None):
    out = {}
    for name, dataset in validation_datasets.items():
        sat_embeddings = sed.build_satellite_db(
            sat_model,
            vigor_dataset.get_dataloader(dataset.get_sat_patch_view(), batch_size=64, num_workers=8))
        pano_embeddings = sed.build_panorama_db(
            pano_model,
            vigor_dataset.get_dataloader(dataset.get_pano_view(), batch_size=64, num_workers=8))
        weight_matrix = None
        if weight_model is not None:
            weight_matrix = weight_model(
                sat_embedding=sat_embeddings,
                pano_embedding=pano_embeddings
            )
        similarity = distance_from_type(
            distance_type=distance_type,
            pano_embeddings=pano_embeddings,
            sat_embeddings=sat_embeddings,
            weight_matrix=weight_matrix
        )

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

def setup_models_for_training(panorama_model, satellite_model, weight_matrix_model):
    """Move models to GPU and set to training mode."""
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()
    panorama_model.train()
    satellite_model.train()
    if weight_matrix_model is not None:
        weight_matrix_model = weight_matrix_model.cuda()
        weight_matrix_model.train()
    return panorama_model, satellite_model, weight_matrix_model



def create_training_components(dataset, panorama_model, satellite_model, opt_config):
    """Create miner, dataloader, and optimizer for training."""
    # Create miner and dataloader  
    miner = vigor_dataset.HardNegativeMiner(
            batch_size=opt_config.batch_size,
            embedding_dimension=panorama_model.output_dim,
            random_sample_type=opt_config.random_sample_type,
            hard_negative_pool_size=opt_config.hard_negative_pool_size,
            dataset=dataset)
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_sampler=miner, num_workers=24, persistent_workers=True)
    
    # Create optimizer
    opt = torch.optim.AdamW(
        list(panorama_model.parameters()) + list(satellite_model.parameters()),
        lr=opt_config.lr_schedule.initial_lr
    )
    
    return miner, dataloader, opt


def compute_forward_pass_and_loss(batch, 
                                  panorama_model, 
                                  satellite_model, 
                                  weight_matrix_model,
                                  pairing_data: PairingDataType,
                                  train_config: TrainConfig):
    """Compute forward pass and loss for a batch."""
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        panorama_embeddings = panorama_model(
                panorama_model.model_input_from_batch(batch).to("cuda"))
        sat_embeddings = satellite_model(
                satellite_model.model_input_from_batch(batch).to("cuda"))
        
        weight_matrix = None
        if weight_matrix_model is not None:
            weight_matrix = weight_matrix_model(
                sat_embedding=sat_embeddings,
                pano_embedding=panorama_embeddings,
            )
        loss_dict = compute_loss(
                pano_embeddings=panorama_embeddings,
                sat_embeddings=sat_embeddings,
                pairing_data=pairing_data,
                distance_type=train_config.distance_type,
                weight_matrix=weight_matrix,
                loss_configs=train_config.loss_configs)
    
    return loss_dict, panorama_embeddings, sat_embeddings


def train(config: TrainConfig, 
          *, 
          dataset, 
          validation_datasets, 
          panorama_model, 
          satellite_model, 
          weight_matrix_model, 
          quiet):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # save config:
    config_json = msgspec.json.encode(config, enc_hook=enc_hook)
    config_dict = json.loads(config_json)
    with open(config.output_dir / "config.json", 'wb') as f:
        f.write(config_json)

    with open(config.output_dir / "train_config.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config, enc_hook=enc_hook))

    with open(config.output_dir / "satellite_model.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config.sat_model_config, enc_hook=enc_hook))

    with open(config.output_dir / "panorama_model.yaml", 'wb') as f:
        f.write(msgspec.yaml.encode(config.pano_model_config, enc_hook=enc_hook))

    writer = SummaryWriter(
        log_dir=config.tensorboard_output
    )
    # write hyperparameters
    writer.add_hparams(
        flatten_dict(config_dict['opt_config']), {},
        run_name="."
    )
    
    # Setup models using extracted function
    panorama_model, satellite_model, weight_matrix_model = setup_models_for_training(panorama_model, satellite_model, weight_matrix_model)

    print(f"working with train dataset {len(dataset._satellite_metadata)=}" +
          f" {len(dataset._panorama_metadata)=} {len(dataset._landmark_metadata)=}")
    for name, val_dataset in validation_datasets.items():
        print(f"working with validation dataset {name} {len(val_dataset._satellite_metadata)=}" +
              f" {len(val_dataset._panorama_metadata)=} {len(val_dataset._landmark_metadata)=}")

    opt_config = config.opt_config

    # Create training components using extracted function
    miner, dataloader, opt = create_training_components(dataset, panorama_model, satellite_model, opt_config)

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
            match config.pairing_type:
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
                    raise RuntimeError(f"Pairing type not recongnized, {config.pairing_type}")
            opt.zero_grad()

            # Use extracted function for forward pass and loss
            loss_dict, panorama_embeddings, sat_embeddings = compute_forward_pass_and_loss(
                batch, panorama_model, satellite_model, weight_matrix_model, pairing_data, config)

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
                            raise RuntimeError(f"Parameter {name} for model {model_name} requires grad, but had no update.")
                        if param.grad is not None and torch.any(torch.isinf(param.grad)):
                            print(f"Warining: INF was found in parameter gradient: {name} in model {model_name}")

            log_gradient_stats(writer, panorama_model, "panorama", total_batches)
            log_gradient_stats(writer, satellite_model, "satellite", total_batches)
            log_embedding_stats(writer, "pano", panorama_embeddings.detach(), total_batches)
            log_embedding_stats(writer, "sat", sat_embeddings.detach(), total_batches)

            # Hard Negative Mining
            # miner.consume(
            #         panorama_embeddings=pano_embedding.detach(),
            #         satellite_embeddings=sat_embedding.detach(),
            #         batch=batch)

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

            for batch in tqdm.tqdm(unobserved_dataloader, desc="Unobserved sat batches"):
                with torch.no_grad():
                    sat_embeddings = satellite_model(
                        satellite_model.model_input_from_batch(batch).to("cuda"))
                miner.consume(None, sat_embeddings, batch)

        # compute validation set metrics
        validation_metrics = compute_validation_metrics(
                sat_model=satellite_model,
                pano_model=panorama_model,
                validation_datasets=validation_datasets,
                distance_type=config.distance_type,
                weight_model=weight_matrix_model)
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
            weight_matrix_model_path = config.output_dir / f"{epoch_idx:04d}_weight_matrix"

            save_dataloader = vigor_dataset.get_dataloader(dataset, batch_size=16)
            batch = next(iter(save_dataloader))
            pano_model_input = panorama_model.model_input_from_batch(batch).to("cuda")
            sat_model_input = satellite_model.model_input_from_batch(batch).to("cuda")
            save_model(panorama_model, panorama_model_path,
                       (pano_model_input,))

            save_model(satellite_model, satellite_model_path,
                       (sat_model_input,))
            if weight_matrix_model is not None:
                with torch.no_grad():
                    save_model(weight_matrix_model, weight_matrix_model_path,
                            (satellite_model(sat_model_input), panorama_model(pano_model_input)))


def main(
        dataset_base_path: Path,
        output_base_path: Path,
        train_config_path: Path,
        quiet: bool,
        no_ipdb: bool,
        lr_sweep: bool = False,
):
    with open(train_config_path, 'r') as file_in:
        train_config = msgspec.yaml.decode(file_in.read(), type=TrainConfig, dec_hook=dec_hook)
    pprint(train_config)

    if isinstance(train_config.sat_model_config, patch_embedding.WagPatchEmbeddingConfig):
        satellite_model = patch_embedding.WagPatchEmbedding(train_config.sat_model_config)
    elif isinstance(train_config.sat_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        satellite_model = swag_patch_embedding.SwagPatchEmbedding(train_config.num_vector_embeddings, train_config.sat_model_config)
    else:
        raise TypeError("Unsupported satellite model config type")

    if isinstance(train_config.pano_model_config, patch_embedding.WagPatchEmbeddingConfig):
        panorama_model = patch_embedding.WagPatchEmbedding(train_config.pano_model_config)
    elif isinstance(train_config.pano_model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        panorama_model = swag_patch_embedding.SwagPatchEmbedding(train_config.num_vector_embeddings, train_config.pano_model_config)
    else:
        raise TypeError("Unsupported panorama model config type")
    
    weight_matrix_model = None 
    if train_config.weight_matrix_model_config is not None:
        weight_matrix_model = WeightMatrixModel(
            train_config.weight_matrix_model_config
        )

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
                should_load_images=validation_dataset_config.should_load_images))

    output_dir = output_base_path / train_config.output_dir
    tensorboard_output = train_config.tensorboard_output
    tensorboard_output = tensorboard_output if tensorboard_output is not None else output_dir

    train_config.output_dir = output_dir
    train_config.tensorboard_output = tensorboard_output

    # Run learning rate sweep if requested
    if lr_sweep:
        # Create default LR sweep config if not in train config
        if train_config.opt_config.lr_sweep_config is None:
            train_config.opt_config.lr_sweep_config = LearningRateSweepConfig()
        
        optimal_lr = run_lr_sweep(
            lr_sweep_config=train_config.opt_config.lr_sweep_config,
            dataset=dataset,
            panorama_model=panorama_model,
            satellite_model=satellite_model,
            opt_config=train_config.opt_config,
            output_dir=output_dir,
            compute_forward_pass_and_loss_fn=compute_forward_pass_and_loss,
            create_training_components_fn=create_training_components,
            setup_models_for_training_fn=setup_models_for_training,
            quiet=quiet
        )
        
        # Update the training config to use the optimal learning rate
        train_config.opt_config.lr_schedule.initial_lr = optimal_lr
        if not quiet:
            print(f"Updated initial learning rate to {optimal_lr:.2e}")

    with ipdb.launch_ipdb_on_exception() if not no_ipdb else nullcontext():
        train(
            train_config,
            dataset=dataset,
            validation_datasets=validation_datasets,
            panorama_model=panorama_model,
            satellite_model=satellite_model,
            weight_matrix_model=weight_matrix_model,
            quiet=quiet)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", help="path to dataset", required=True)
    parser.add_argument("--output_base", help="path to output", required=True)
    parser.add_argument("--train_config", help="path to train_config", required=True)
    parser.add_argument("--no_ipdb", action="store_true", help="Don't run IPDB around the training job")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    main(
        Path(args.dataset_base),
        Path(args.output_base),
        Path(args.train_config),
        no_ipdb=args.no_ipdb,
        quiet=args.quiet)
