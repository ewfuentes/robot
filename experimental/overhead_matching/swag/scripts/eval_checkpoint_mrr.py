"""Evaluate a saved checkpoint's Seattle MRR by both methods.

Loads the saved {best_panorama,best_satellite}/{model.pt,model_weights.pt}
two ways and runs validation on Seattle:
  1. model.pt — full pickled module
  2. model_weights.pt — state dict loaded into a fresh model from the
     train_config.yaml in the run directory

Both should give the same MRR. If not, that explains why warm-start for Stage 2
gives different MRR than the same checkpoint reported in training.
"""
import argparse
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
import msgspec

from common.python.serialization import msgspec_dec_hook
from experimental.overhead_matching.swag.data import (
    satellite_embedding_database as sed,
    vigor_dataset,
)
from experimental.overhead_matching.swag.evaluation.retrieval_metrics import (
    validation_metrics_from_similarity,
)
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
from experimental.overhead_matching.swag.scripts.distances import (
    CosineDistanceConfig, create_distance_from_config,
)


def _load_full_model(path: Path):
    return torch.load(path / "model.pt", map_location="cpu", weights_only=False)


def _build_model_from_yaml_and_load_weights(yaml_path: Path, model_path: Path, kind: str):
    """Build a fresh sat or pano model from the YAML config and load weights."""
    raw = yaml_path.read_text()
    train_config_dict = msgspec.yaml.decode(raw)
    cfg_struct = train_config_dict[f"{kind}_model_config"]
    cfg = msgspec.convert(
        cfg_struct,
        type=swag_patch_embedding.SwagPatchEmbeddingConfig,
        dec_hook=msgspec_dec_hook,
    )
    model = swag_patch_embedding.SwagPatchEmbedding(cfg)
    state = torch.load(model_path / "model_weights.pt", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    out = model.load_state_dict(state, strict=False)
    print(f"  load: missing={len(out.missing_keys)} unexpected={len(out.unexpected_keys)}")
    return model


def evaluate(sat_model, pano_model, city: str = "Seattle"):
    sat_model = sat_model.cuda()
    pano_model = pano_model.cuda()
    distance_model = create_distance_from_config(CosineDistanceConfig()).cuda()

    dataset_path = Path(f"/data/overhead_matching/datasets/VIGOR/{city}")
    cfg = vigor_dataset.VigorDatasetConfig(
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640),
        satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=[Path(city)], model_type="satellite",
            landmark_version="v1", panorama_landmark_radius_px=640.0,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=sat_model.cache_info()),
        panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=[Path(city)], model_type="panorama",
            landmark_version="v1", panorama_landmark_radius_px=640.0,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=pano_model.cache_info()),
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
        factor=1.0,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="v1",
        panorama_landmark_radius_px=640.0,
        landmark_correspondence_inflation_factor=1.0,
    )
    ds = vigor_dataset.VigorDataset([dataset_path], cfg)

    with torch.no_grad():
        sat_emb = sed.build_satellite_db(
            sat_model,
            vigor_dataset.get_dataloader(ds.get_sat_patch_view(), batch_size=64, num_workers=4))
        pano_emb = sed.build_panorama_db(
            pano_model,
            vigor_dataset.get_dataloader(ds.get_pano_view(), batch_size=64, num_workers=4))

    distance_model = distance_model.cpu()
    sat_emb = sat_emb.cpu()
    pano_emb = pano_emb.cpu()
    with torch.no_grad():
        sim = distance_model(
            pano_embeddings_unnormalized=pano_emb,
            sat_embeddings_unnormalized=sat_emb).cpu()
    metrics = validation_metrics_from_similarity(city, sim, panorama_metadata=ds._panorama_metadata)
    print(f"  {city}/positive_mean_recip_rank: {metrics[city + '/positive_mean_recip_rank']:.4f}")
    print(f"  {city}/pos_recall@1: {metrics[city + '/pos_recall@1']:.4f}")
    print(f"  {city}/pos_recall@5: {metrics[city + '/pos_recall@5']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True,
                        help="Path to a Stage 1' run dir (contains best_panorama, best_satellite, train_config.yaml)")
    parser.add_argument("--mode", choices=["model_pt", "weights_pt", "both"], default="both")
    args = parser.parse_args()

    yaml_path = args.checkpoint_dir / "train_config.yaml"
    sat_dir = args.checkpoint_dir / "best_satellite"
    pano_dir = args.checkpoint_dir / "best_panorama"

    if args.mode in ("model_pt", "both"):
        print("=== Loading via model.pt (full pickle) ===")
        sat = _load_full_model(sat_dir)
        pano = _load_full_model(pano_dir)
        if hasattr(sat, "_orig_mod"):
            sat = sat._orig_mod
        if hasattr(pano, "_orig_mod"):
            pano = pano._orig_mod
        sat.eval()
        pano.eval()
        evaluate(sat, pano)

    if args.mode in ("weights_pt", "both"):
        print("\n=== Loading via state_dict (model_weights.pt), .eval() applied ===")
        print("  satellite:")
        sat = _build_model_from_yaml_and_load_weights(yaml_path, sat_dir, "sat")
        print("  panorama:")
        pano = _build_model_from_yaml_and_load_weights(yaml_path, pano_dir, "pano")
        sat.eval()
        pano.eval()
        evaluate(sat, pano)

        print("\n=== Loading via state_dict (model_weights.pt), .train() applied (mirror train.py) ===")
        print("  satellite:")
        sat = _build_model_from_yaml_and_load_weights(yaml_path, sat_dir, "sat")
        print("  panorama:")
        pano = _build_model_from_yaml_and_load_weights(yaml_path, pano_dir, "pano")
        sat.train()
        pano.train()
        evaluate(sat, pano)


if __name__ == "__main__":
    main()
