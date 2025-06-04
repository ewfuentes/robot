import argparse
import json
import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import itertools
from pathlib import Path
from experimental.overhead_matching.swag.data import vigor_dataset
from experimental.overhead_matching.swag.model import patch_embedding
from dataclasses import dataclass
import tqdm


@dataclass
class OptimizationConfig:
    num_epochs: int

    # The number of batches that are used to create the embedding pool
    num_embedding_pool_batches: int

    # This is the number of examples from which we build triplets
    # This is computed with torch.no_grad, so more examples can
    # be processed in a batch than is possible when we are taking a
    # gradient step.
    embedding_pool_batch_size: int

    # This is the number of triplets that are processed in a batch.
    opt_batch_size: int


@dataclass
class TrainConfig:
    opt_config: OptimizationConfig
    output_dir: Path


@dataclass
class Pairs:
    positive_pairs: list[tuple[int, int]]
    negative_pairs: list[tuple[int, int]]
    semipositive_pairs: list[tuple[int, int]]

def dataclass_to_dict(input_object)->dict:
    """Convert a dataclass instance (and any nested dataclasses) to a dictionary.
    
    Args:
        input_object: A dataclass instance or a structure containing dataclasses
        
    Returns:
        A dictionary representation of the dataclass with nested dataclasses also converted
    """
    from dataclasses import is_dataclass, fields
    from pathlib import Path
    
    # Base case: None
    if input_object is None:
        return None
        
    # Recursive case: dataclass instance
    if is_dataclass(input_object):
        result = {}
        for field in fields(input_object):
            field_value = getattr(input_object, field.name)
            result[field.name] = dataclass_to_dict(field_value)
        return result
        
    # Recursive case: list/tuple containing possibly nested dataclasses
    if isinstance(input_object, (list, tuple)):
        return type(input_object)(dataclass_to_dict(item) for item in input_object)
        
    # Recursive case: dictionary containing possibly nested dataclasses
    if isinstance(input_object, dict):
        return {key: dataclass_to_dict(value) for key, value in input_object.items()}
        
    # Special case: Path objects
    if isinstance(input_object, Path):
        return str(input_object)
        
    # Base case: return the object itself (int, float, str, etc.)
    return input_object


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


def train(config: TrainConfig, *, dataset, panorama_model, satellite_model):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # save config:
    config_dict = dataclass_to_dict(config)
    with open(config.output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f)
    writer = SummaryWriter(
        log_dir=config.output_dir / "logs"
    )
    # write hyperparameters
    writer.add_hparams(
        config_dict['opt_config'], {}
    )
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()
    panorama_model.train()
    satellite_model.train()

    print(dataset._satellite_metadata)
    print(dataset._panorama_metadata)

    opt_config = config.opt_config

    overall_batch_size = (
        opt_config.embedding_pool_batch_size * opt_config.num_embedding_pool_batches
    )
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_size=overall_batch_size, num_workers=2, shuffle=True, persistent_workers=True
    )

    opt = torch.optim.Adam(
        list(panorama_model.parameters()) + list(satellite_model.parameters()),
        lr=1e-4
    )

    torch.set_printoptions(linewidth=200)

    total_batches = 0

    for epoch_idx in tqdm.tqdm(range(config.opt_config.num_epochs),  desc="Epoch"):
        for batch_idx, batch in enumerate(dataloader):
            pairs = create_pairs(
                batch.panorama_metadata,
                batch.satellite_metadata
            )

            opt.zero_grad()

            panorama_embeddings = panorama_model(batch.panorama.cuda())
            sat_embeddings = satellite_model(batch.satellite.cuda())

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
            writer.add_scalar("train/num_positive_pairs", len(pairs.positive_pairs), global_step=total_batches)
            writer.add_scalar("train/num_sempipos_pairs", len(pairs.semipositive_pairs), global_step=total_batches)
            writer.add_scalar("train/num_neg_pairs", len(pairs.negative_pairs), global_step=total_batches)
            writer.add_scalar("train/loss_pos", pos_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss_semipos", semipos_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss_neg", neg_loss.item(), global_step=total_batches)
            writer.add_scalar("train/loss", loss.item(), global_step=total_batches)
            print(f"{epoch_idx=:4d} {batch_idx=:4d} num_pos_pairs: {len(pairs.positive_pairs):3d}" +
                  f" num_semipos_pairs: {len(pairs.semipositive_pairs):3d}" +
                  f"  num_neg_pairs: {len(pairs.negative_pairs):3d} {pos_loss.item()=:0.6f}" +
                  f" {semipos_loss.item()=:0.6f} {neg_loss.item()=:0.6f} {loss.item()=:0.6f}", end='\r')
            if batch_idx % 50 == 0:
                print()
            opt.step()

            total_batches += 1
        print()

        if epoch_idx % 10 == 0:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            panorama_model_path = config.output_dir / f"{epoch_idx:04d}_panorama"
            satellite_model_path = config.output_dir / f"{epoch_idx:04d}_satellite"

            batch = next(iter(dataloader))
            save_model(panorama_model, panorama_model_path,
                       (batch.panorama[:opt_config.opt_batch_size].cuda(),))
            save_model(satellite_model, satellite_model_path,
                       (batch.satellite[:opt_config.opt_batch_size].cuda(),))


def main(dataset_path: Path, output_dir: Path):
    PANORAMA_NEIGHBOR_RADIUS_DEG = 1e-6
    NUM_SAFA_HEADS = 4
    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=PANORAMA_NEIGHBOR_RADIUS_DEG,
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640),
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
    )
    dataset = vigor_dataset.VigorDataset(dataset_path, dataset_config)

    satellite_model = patch_embedding.WagPatchEmbedding(
        patch_embedding.WagPatchEmbeddingConfig(
            patch_dims=dataset_config.satellite_patch_size,
            num_aggregation_heads=NUM_SAFA_HEADS,
        )
    )

    panorama_model = patch_embedding.WagPatchEmbedding(
        patch_embedding.WagPatchEmbeddingConfig(
            patch_dims=dataset_config.panorama_size,
            num_aggregation_heads=NUM_SAFA_HEADS,
        )
    )

    config = TrainConfig(
        opt_config=OptimizationConfig(
            num_epochs=1000,
            num_embedding_pool_batches=1,
            embedding_pool_batch_size=40,
            opt_batch_size=40,
        ),
        output_dir=output_dir,
    )

    train(
        config,
        dataset=dataset,
        panorama_model=panorama_model,
        satellite_model=satellite_model,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to dataset", required=True)
    parser.add_argument("--output_dir", help="path to output", required=True)
    args = parser.parse_args()

    main(Path(args.dataset), Path(args.output_dir))
