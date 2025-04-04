
import argparse

import common.torch.load_torch_deps
import torch
from pathlib import Path
from experimental.overhead_matching.swag.data import vigor_dataset
from experimental.overhead_matching.swag.model import patch_embedding
from dataclasses import dataclass
import tqdm


@dataclass
class OptimizationConfig:
    num_epochs: int


@dataclass
class TrainConfig:
    opt_config: OptimizationConfig
    output_dir: Path


def train(config: TrainConfig, *, dataset, panorama_model, satellite_model):
    panorama_model = panorama_model.cuda()
    satellite_model = satellite_model.cuda()

    dataloader = vigor_dataset.get_dataloader(dataset, batch_size=16, num_workers=0)

    for epoch_idx in tqdm.tqdm(range(config.opt_config.num_epochs)):
        print(epoch_idx)
        for batch_idx, batch in enumerate(iter(dataloader)):
            # Compute the embeddings for all panoramas and satellite patches
            with torch.no_grad():
                panos = batch.panorama.cuda()
                satellite_patches = batch.satellite.cuda()

                pano_embeddings = panorama_model(panos)
                satellite_embeddings = satellite_model(satellite_patches)

            print(epoch_idx, batch_idx, pano_embeddings.shape, satellite_embeddings.shape, flush=True)
        break


def main(dataset_path: Path, output_dir: Path):
    PANORAMA_NEIGHBOR_RADIUS_DEG = 1e-6
    NUM_SAFA_HEADS = 4
    dataset_config = vigor_dataset.VigorDatasetConfig(
        panorama_neighbor_radius=PANORAMA_NEIGHBOR_RADIUS_DEG,
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640)
    )
    dataset = vigor_dataset.VigorDataset(dataset_path, dataset_config)

    satellite_model = patch_embedding.WagPatchEmbedding(
            patch_embedding.WagPatchEmbeddingConfig(patch_dims=dataset_config.satellite_patch_size,
                                                    num_aggregation_heads=NUM_SAFA_HEADS))

    panorama_model = patch_embedding.WagPatchEmbedding(
            patch_embedding.WagPatchEmbeddingConfig(patch_dims=dataset_config.panorama_size,
                                                    num_aggregation_heads=NUM_SAFA_HEADS))

    config = TrainConfig(
        opt_config=OptimizationConfig(
            num_epochs=100
        ),
        output_dir=output_dir
    )

    train(config, dataset=dataset, panorama_model=panorama_model, satellite_model=satellite_model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to dataset', required=True)
    parser.add_argument('--output_dir', help='path to output', required=True)
    args = parser.parse_args()

    main(Path(args.dataset), Path(args.output_dir))
