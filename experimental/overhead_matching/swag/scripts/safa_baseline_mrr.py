"""Compute SAFA's intrinsic MRR on Seattle.

Builds a SwagPatchEmbedding wrapper around SAFA whose projection is forced to
identity and marker to zero, so the model's output is just the cached SAFA
tensor (post-normalize). Runs validation and prints MRR. This is the upper
bound on what Stage 1 distillation can achieve — distilled models can only
match this number, not exceed it.
"""
import sys
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.data import (
    satellite_embedding_database as sed,
    vigor_dataset,
)
from experimental.overhead_matching.swag.evaluation.retrieval_metrics import (
    validation_metrics_from_similarity,
)
from experimental.overhead_matching.swag.model import swag_patch_embedding as spe
from experimental.overhead_matching.swag.model.swag_config_types import (
    NullPositionEmbeddingConfig,
    SafaExtractorConfig,
    TransformerAggregatorConfig,
)
from experimental.overhead_matching.swag.scripts.distances import (
    CosineDistanceConfig,
    create_distance_from_config,
)


SAFA_BASE = (
    "/data/overhead_matching/training_outputs/260215_baseline_retraining/"
    "260218_093851_all_chicago_dinov3_wag_bs18_v2")


def make_passthrough_model(safa_path: str, patch_dims):
    """SwagPatchEmbedding configured so the forward output IS the SAFA token.

    skip_aggregation=True returns the input tokens directly (no aggregator). With
    only SAFA registered, the lone input token is the (identity-projected,
    zero-marker) SAFA feature — i.e. SAFA itself, normalized.
    """
    cfg = spe.SwagPatchEmbeddingConfig(
        position_embedding_config=NullPositionEmbeddingConfig(),
        aggregation_config=TransformerAggregatorConfig(
            num_transformer_layers=1, num_attention_heads=1,
            hidden_dim=2048, dropout_frac=0.0),
        patch_dims=patch_dims,
        output_dim=2048,
        num_embeddings=1,
        extractor_config_by_name={
            "safa_extractor": SafaExtractorConfig(model_path=safa_path, freeze=True),
        },
        use_cached_extractors=["safa_extractor"],
        normalize_embeddings=True,
        normalize_input_tokens=False,
        skip_aggregation=True,
    )
    model = spe.SwagPatchEmbedding(cfg)
    with torch.no_grad():
        proj = model._projection_by_name["safa_extractor"]
        proj.weight.zero_()
        n = min(proj.weight.shape[0], proj.weight.shape[1])
        proj.weight[:n, :n].copy_(torch.eye(n))
        proj.bias.zero_()
        model._token_marker_by_name["safa_extractor"].zero_()
    return model


def main(city: str = "Seattle"):
    sat_model = make_passthrough_model(f"{SAFA_BASE}/best_satellite", (320, 320)).cuda().eval()
    pano_model = make_passthrough_model(f"{SAFA_BASE}/best_panorama", (320, 640)).cuda().eval()
    distance_model = create_distance_from_config(CosineDistanceConfig()).cuda().eval()

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
        should_load_images=True,
        should_load_landmarks=False,
        landmark_version="v1",
        panorama_landmark_radius_px=640.0,
        landmark_correspondence_inflation_factor=1.0,
    )
    ds = vigor_dataset.VigorDataset([dataset_path], cfg)

    print(f"Building {city} sat embedding db...")
    sat_emb = sed.build_satellite_db(
        sat_model,
        vigor_dataset.get_dataloader(ds.get_sat_patch_view(), batch_size=64, num_workers=4))
    print(f"Building {city} pano embedding db...")
    pano_emb = sed.build_panorama_db(
        pano_model,
        vigor_dataset.get_dataloader(ds.get_pano_view(), batch_size=64, num_workers=4))

    distance_model = distance_model.cpu()
    pano_emb = pano_emb.cpu()
    sat_emb = sat_emb.cpu()
    with torch.no_grad():
        sim = distance_model(
            pano_embeddings_unnormalized=pano_emb,
            sat_embeddings_unnormalized=sat_emb).cpu()
    metrics = validation_metrics_from_similarity(city, sim, panorama_metadata=ds._panorama_metadata)
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "Seattle"
    main(city)
