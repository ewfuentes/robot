import itertools
import msgspec
from pathlib import Path
from common.python.serialization import msgspec_enc_hook
import experimental.overhead_matching.swag.scripts.train as T
from experimental.overhead_matching.swag.scripts import losses, distances
import experimental.overhead_matching.swag.model.patch_embedding as pe
from experimental.overhead_matching.swag.data.vigor_dataset import HardNegativeMiner
import experimental.overhead_matching.swag.model.swag_patch_embedding as spe


def generate_config(panorama_landmark_radius_px: float):
    """Generate a training config based on semantic_landmark_config.yaml with varying radius."""

    opt_config = T.OptimizationConfig(
        num_epochs=100,
        batch_size=512,
        lr_schedule=T.LearningRateSchedule(
            initial_lr=0.0001,
            lr_step_factor=1.0,
            num_epochs_at_lr=20,
            warmup_factor=1.0,
            num_warmup_epochs=5),
        enable_hard_negative_sampling_after_epoch_idx=20,
        hard_negative_pool_size=25,
        random_sample_type=HardNegativeMiner.RandomSampleType.NEAREST)

    loss_configs = [
        losses.PairwiseContrastiveLossConfig(
            positive_weight=5.0,
            avg_positive_similarity=0.0,
            semipositive_weight=6.0,
            avg_semipositive_similarity=0.3,
            negative_weight=20.0,
            avg_negative_similarity=0.7),
        losses.BatchUniformityLossConfig(
            batch_uniformity_weight=0.01,
            batch_uniformity_hinge_location=0.7)
    ]

    # Shared extractor configuration
    extractor_config_by_name = {
        "point_semantic_landmark_extractor": spe.SemanticLandmarkExtractorConfig(
            landmark_type="point",
            openai_embedding_size=1536,
            embedding_version="v3_no_addresses",
            auxiliary_info_key="semantic_embedding_base_path"),
        "linestring_semantic_landmark_extractor": spe.SemanticLandmarkExtractorConfig(
            landmark_type="linestring",
            openai_embedding_size=1536,
            embedding_version="v3_no_addresses",
            auxiliary_info_key="semantic_embedding_base_path"),
        "polygon_semantic_landmark_extractor": spe.SemanticLandmarkExtractorConfig(
            landmark_type="polygon",
            openai_embedding_size=1536,
            embedding_version="v3_no_addresses",
            auxiliary_info_key="semantic_embedding_base_path"),
        "multipolygon_semantic_landmark_extractor": spe.SemanticLandmarkExtractorConfig(
            landmark_type="multipolygon",
            openai_embedding_size=1536,
            embedding_version="v3_no_addresses",
            auxiliary_info_key="semantic_embedding_base_path"),
    }

    use_cached_extractors = [
        "point_semantic_landmark_extractor",
        "linestring_semantic_landmark_extractor",
        "polygon_semantic_landmark_extractor",
        "multipolygon_semantic_landmark_extractor",
    ]

    auxiliary_info = {
        "semantic_embedding_base_path": "/data/overhead_matching/datasets/semantic_landmark_embeddings"
    }

    sat_model_config = spe.SwagPatchEmbeddingConfig(
        num_embeddings=1,
        position_embedding_config=spe.NullPositionEmbeddingConfig(),
        aggregation_config=spe.TransformerAggregatorConfig(
            num_transformer_layers=2,
            num_attention_heads=8,
            hidden_dim=1024,
            dropout_frac=0.1),
        patch_dims=(320, 320),
        output_dim=256,
        extractor_config_by_name=extractor_config_by_name,
        auxiliary_info=auxiliary_info,
        normalize_embeddings=True,
        feature_map_extractor_config=None,
        semantic_token_extractor_config=None,
        use_cached_feature_maps=False,
        use_cached_semantic_tokens=False,
        use_cached_extractors=use_cached_extractors)

    pano_model_config = spe.SwagPatchEmbeddingConfig(
        num_embeddings=1,
        position_embedding_config=spe.NullPositionEmbeddingConfig(),
        aggregation_config=spe.TransformerAggregatorConfig(
            num_transformer_layers=2,
            num_attention_heads=8,
            hidden_dim=1024,
            dropout_frac=0.1),
        patch_dims=(320, 640),
        output_dim=256,
        extractor_config_by_name=extractor_config_by_name,
        auxiliary_info=auxiliary_info,
        normalize_embeddings=True,
        feature_map_extractor_config=None,
        semantic_token_extractor_config=None,
        use_cached_feature_maps=False,
        use_cached_semantic_tokens=False,
        use_cached_extractors=use_cached_extractors)

    dataset_config = T.DatasetConfig(
        paths=["Chicago"],
        factor=1.0,
        landmark_version="v3",
        panorama_landmark_radius_px=panorama_landmark_radius_px)

    validation_dataset_configs = [
        T.DatasetConfig(
            paths=["Seattle"],
            factor=0.3,
            landmark_version="v3",
            panorama_landmark_radius_px=panorama_landmark_radius_px),
        T.DatasetConfig(
            paths=["Chicago"],
            factor=0.3,
            landmark_version="v3",
            panorama_landmark_radius_px=panorama_landmark_radius_px),
    ]

    distance_model_config = distances.CosineDistanceConfig()

    output_dir = f"semantic_landmark_pano_radius_{int(panorama_landmark_radius_px)}"

    return T.TrainConfig(
        opt_config=opt_config,
        sat_model_config=sat_model_config,
        pano_model_config=pano_model_config,
        distance_model_config=distance_model_config,
        loss_configs=loss_configs,
        output_dir=output_dir,
        tensorboard_output=None,
        dataset_config=dataset_config,
        validation_dataset_configs=validation_dataset_configs
    )


if __name__ == "__main__":
    # Generate configs with varying panorama landmark radius values
    radius_values = [50, 100, 120, 200, 320, 400, 480, 560, 640, 720, 800]

    for radius in radius_values:
        cfg = generate_config(panorama_landmark_radius_px=radius)

        file_name = Path(f"/tmp/{cfg.output_dir}.yaml")
        yaml_config = msgspec.yaml.encode(cfg, enc_hook=msgspec_enc_hook)
        file_name.write_bytes(yaml_config)
        print(f"Generated config: {file_name}")
