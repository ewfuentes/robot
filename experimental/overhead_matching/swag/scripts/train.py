import argparse

import common.torch.load_torch_deps
import torch
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
class Triplet:
    anchor_panorama_idx: int
    positive_satellite_idx: int
    negative_satellite_idx: int


def create_triplets(
    panorama_metadata, satellite_metadata, panorama_embeddings, satellite_embeddings
):
    # Generate an exhaustive set of triplets where the anchor is a panorama
    # TODO consider creating triplets where a satellite patch is the anchor
    out = []
    for batch_pano_idx in range(panorama_embeddings.shape[0]):
        anchor_sat_idx = panorama_metadata[batch_pano_idx]["satellite_idx"]
        # For now, we assume that the panorama and satellite metadata are paired since
        # the satellite metadata does not currently contain the satellite index
        for batch_sat_idx in range(panorama_embeddings.shape[0]):
            neg_sat_idx = panorama_metadata[batch_sat_idx]["satellite_idx"]
            if anchor_sat_idx == neg_sat_idx:
                continue
            out.append(
                Triplet(
                    anchor_panorama_idx=batch_pano_idx,
                    positive_satellite_idx=batch_pano_idx,
                    negative_satellite_idx=batch_sat_idx,
                )
            )
    return out


def compute_loss(anchor_embedding, positive_embeddings, negative_embeddings):
    return torch.nn.functional.triplet_margin_loss(
        anchor_embedding, positive_embeddings, negative_embeddings
    )


def train(config: TrainConfig, *, dataset, panorama_model, satellite_model):
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()

    opt_config = config.opt_config

    overall_batch_size = (
        opt_config.embedding_pool_batch_size * opt_config.num_embedding_pool_batches
    )
    dataloader = vigor_dataset.get_dataloader(
        dataset, batch_size=overall_batch_size, num_workers=1
    )

    opt = torch.optim.Adam(
        list(panorama_model.parameters()) + list(satellite_model.parameters())
    )

    for epoch_idx in tqdm.tqdm(range(config.opt_config.num_epochs),  desc="Epoch"):
        for batch_idx, batch in enumerate(dataloader):
            panorama_embeddings = []
            satellite_embeddings = []
            for i in range(opt_config.num_embedding_pool_batches):
                start_idx = i * opt_config.embedding_pool_batch_size
                end_idx = start_idx + opt_config.embedding_pool_batch_size
                with torch.no_grad():
                    panos = batch.panorama[start_idx:end_idx].cuda()
                    satellite_patches = batch.satellite[start_idx:end_idx].cuda()
                    panorama_embeddings.append(panorama_model(panos).cpu())
                    satellite_embeddings.append(
                        satellite_model(satellite_patches).cpu()
                    )

            panorama_embeddings = torch.cat(panorama_embeddings)
            satellite_embeddings = torch.cat(satellite_embeddings)

            triplets = create_triplets(
                batch.panorama_metadata,
                batch.satellite_metadata,
                panorama_embeddings,
                satellite_embeddings,
            )

            print(f"{epoch_idx=} {batch_idx=} {len(triplets)=}")

            # Iterate through the triplets
            batch_loss = 0
            for start_idx in tqdm.tqdm(range(0, len(triplets), opt_config.opt_batch_size), desc="triplets"):
                end_idx = start_idx + opt_config.opt_batch_size
                opt.zero_grad()
                batch_triplets = triplets[start_idx:end_idx]
                anchor_panorama_idxs = [x.anchor_panorama_idx for x in batch_triplets]
                pos_satellite_idxs = [x.positive_satellite_idx for x in batch_triplets]
                neg_satellite_idxs = [x.negative_satellite_idx for x in batch_triplets]

                panos = batch.panorama[anchor_panorama_idxs].cuda()
                pos_satellite = batch.satellite[pos_satellite_idxs].cuda()
                neg_satellite = batch.satellite[neg_satellite_idxs].cuda()

                anchor_embeddings = panorama_model(panos)
                pos_satellite_embeddings = satellite_model(pos_satellite)
                neg_satellite_embeddings = satellite_model(neg_satellite)

                loss = compute_loss(
                    anchor_embeddings,
                    pos_satellite_embeddings,
                    neg_satellite_embeddings,
                )
                batch_loss += loss.item()
                loss.backward()
                opt.step()
            print(f"total loss: {batch_loss}")


def main(dataset_path: Path, output_dir: Path):
    PANORAMA_NEIGHBOR_RADIUS_DEG = 1e-6
    NUM_SAFA_HEADS = 4
    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=PANORAMA_NEIGHBOR_RADIUS_DEG,
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640),
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
            num_epochs=100,
            num_embedding_pool_batches=2,
            embedding_pool_batch_size=8,
            opt_batch_size=4,
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
