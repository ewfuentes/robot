import itertools
import msgspec
from pathlib import Path
from common.python.serialization import msgspec_enc_hook
import experimental.overhead_matching.swag.scripts.train as T
from experimental.overhead_matching.swag.scripts import losses, distances
import experimental.overhead_matching.swag.model.patch_embedding as pe
from experimental.overhead_matching.swag.data.vigor_dataset import HardNegativeMiner
import experimental.overhead_matching.swag.model.swag_patch_embedding as spe

def generate_safa():
    """Generate a training config based on semantic_landmark_config.yaml with varying radius."""
    opt_config = T.OptimizationConfig(
        num_epochs=100,
        batch_size=18,
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

    backbone_config = pe.DinoConfig(
        project=512,
        model="dinov3_vitb16")

    # Shared extractor configuration
    sat_model_config = pe.WagPatchEmbeddingConfig(
        patch_dims = (320, 320),
        num_aggregation_heads=4,
        backbone_config=backbone_config
        )

    pano_model_config = pe.WagPatchEmbeddingConfig(
        patch_dims = (320, 640),
        num_aggregation_heads=4,
        backbone_config=backbone_config)

    dataset_config = T.DatasetConfig(
        paths=["Chicago"],
        factor=1.0,
        landmark_version="v4",
        panorama_landmark_radius_px=640)

    validation_dataset_configs = [
        T.DatasetConfig(
            paths=["Seattle"],
            factor=1.0,
            landmark_version="v4",
            panorama_landmark_radius_px=640),
        T.DatasetConfig(
            paths=["Chicago"],
            factor=1.0,
            landmark_version="v4",
            panorama_landmark_radius_px=640),
    ]

    distance_model_config = distances.CosineDistanceConfig()

    output_dir = f"safa_project_512"

    return T.TrainConfig(
        opt_config=opt_config,
        sat_model_config=sat_model_config,
        pano_model_config=pano_model_config,
        distance_model_config=distance_model_config,
        loss_configs=loss_configs,
        output_dir=output_dir,
        tensorboard_output=None,
        dataset_config=dataset_config,
        validation_dataset_configs=validation_dataset_configs,
        pano_landmark_dropout_schedules=[],
        sat_landmark_dropout_schedules=[],
    )

def generate_config(num_layers: int):
    """Generate a training config based on semantic_landmark_config.yaml with varying radius."""

    opt_config = T.OptimizationConfig(
        num_epochs=100,
        batch_size=18,
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
        "dinov3_feature_extractor": spe.DinoFeatureMapExtractorConfig(
            model_str="dinov3_vitb16")
    }

    sat_model_config = spe.SwagPatchEmbeddingConfig(
        num_embeddings=1,
        position_embedding_config=spe.PlanarPositionEmbeddingConfig(
            min_scale=0.01,
            scale_step=2.0,
            embedding_dim=64),
        aggregation_config=spe.TransformerAggregatorConfig(
            num_transformer_layers=num_layers,
            num_attention_heads=8,
            hidden_dim=1024,
            dropout_frac=0.1),
        patch_dims=(320, 320),
        output_dim=256,
        extractor_config_by_name=extractor_config_by_name,
        auxiliary_info={},
        normalize_embeddings=True,
        feature_map_extractor_config=None,
        semantic_token_extractor_config=None,
        use_cached_feature_maps=False,
        use_cached_semantic_tokens=False,
        use_cached_extractors=[])

    pano_model_config = spe.SwagPatchEmbeddingConfig(
        num_embeddings=1,
        position_embedding_config=spe.SphericalPositionEmbeddingConfig(
            scale_step=2.0,
            embedding_dim=64),
        aggregation_config=spe.TransformerAggregatorConfig(
            num_transformer_layers=num_layers,
            num_attention_heads=8,
            hidden_dim=1024,
            dropout_frac=0.1),
        patch_dims=(320, 640),
        output_dim=256,
        extractor_config_by_name=extractor_config_by_name,
        auxiliary_info={},
        normalize_embeddings=True,
        feature_map_extractor_config=None,
        semantic_token_extractor_config=None,
        use_cached_feature_maps=False,
        use_cached_semantic_tokens=False,
        use_cached_extractors=[])

    dataset_config = T.DatasetConfig(
        paths=["Chicago"],
        factor=1.0,
        landmark_version="v4",
        panorama_landmark_radius_px=640)

    validation_dataset_configs = [
        T.DatasetConfig(
            paths=["Seattle"],
            factor=1.0,
            landmark_version="v4",
            panorama_landmark_radius_px=640),
        T.DatasetConfig(
            paths=["Chicago"],
            factor=1.0,
            landmark_version="v4",
            panorama_landmark_radius_px=640),
    ]

    distance_model_config = distances.CosineDistanceConfig()

    output_dir = f"transformer_layer{num_layers:0d}_attn8"

    return T.TrainConfig(
        opt_config=opt_config,
        sat_model_config=sat_model_config,
        pano_model_config=pano_model_config,
        distance_model_config=distance_model_config,
        loss_configs=loss_configs,
        output_dir=output_dir,
        tensorboard_output=None,
        dataset_config=dataset_config,
        validation_dataset_configs=validation_dataset_configs,
        pano_landmark_dropout_schedules=[],
        sat_landmark_dropout_schedules=[],
    )


if __name__ == "__main__":
    # Generate configs with varying panorama landmark radius values
    num_layers = [2, 4, 8, 16]

    for num_layer in num_layers:
        cfg = generate_config(num_layer)

        file_name = Path(f"/tmp/{cfg.output_dir}.yaml")
        yaml_config = msgspec.yaml.encode(cfg, enc_hook=msgspec_enc_hook)
        file_name.write_bytes(yaml_config)
        print(f"Generated config: {file_name}")

    safa_cfg = generate_safa()

    file_name = Path(f"/tmp/{safa_cfg.output_dir}.yaml")
    yaml_config = msgspec.yaml.encode(safa_cfg, enc_hook=msgspec_enc_hook)
    file_name.write_bytes(yaml_config)
    print(f"Generated config: {file_name}")
