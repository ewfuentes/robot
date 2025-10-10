import itertools
import msgspec
from pathlib import Path
from common.python.serialization import msgspec_enc_hook
import experimental.overhead_matching.swag.scripts.train as T
from experimental.overhead_matching.swag.scripts import losses, distances
import experimental.overhead_matching.swag.model.patch_embedding as pe
from experimental.overhead_matching.swag.data.vigor_dataset import HardNegativeMiner
import experimental.overhead_matching.swag.model.swag_patch_embedding as spe


def generate_config(model_config):
    opt_config = T.OptimizationConfig(
        num_epochs=50,
        batch_size=512,
        lr_schedule=T.LearningRateSchedule(
            num_warmup_epochs=5,
            initial_lr=1e-4,
            lr_step_factor=1.0,
            num_epochs_at_lr=20,
            warmup_factor=1.0),
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
                batch_uniformity_hinge_location=0.7 if model_config["cos_mean_hinge"] else 2.0)]

    aggregation_config = spe.TransformerAggregatorConfig(
            num_transformer_layers=model_config["num_transformer_layers"],
            num_attention_heads=8,
            hidden_dim=4 * model_config["output_dim"],
            dropout_frac=0.1)

    spherical_embedding = spe.PlanarPositionEmbeddingConfig(
            min_scale=0.1, scale_step=2.0, embedding_dim=64)
    planar_embedding = spe.SphericalPositionEmbeddingConfig(
            scale_step=2.0, embedding_dim=64)

    sat_model_config = spe.SwagPatchEmbeddingConfig(
        position_embedding_config=planar_embedding,
        aggregation_config=aggregation_config,
        patch_dims=(320, 320),
        output_dim=model_config["output_dim"],
        num_embeddings=1,
        extractor_config_by_name={
            "synthetic_landmarks": spe.SyntheticLandmarkExtractorConfig(
                log_grid_spacing=model_config["log_grid_spacing"],
                grid_bounds_px=640,
                should_produce_bearing_position_for_pano=False,
                embedding_dim=model_config["syn_embedding_dim"]
                )},
        use_cached_extractors=[])
    pano_model_config = spe.SwagPatchEmbeddingConfig(
        position_embedding_config=(planar_embedding if
                                   model_config["is_pano_pos_planar"] else spherical_embedding),
        aggregation_config=aggregation_config,
        patch_dims=(320, 640),
        output_dim=model_config["output_dim"],
        num_embeddings=1,
        extractor_config_by_name={
            "synthetic_landmarks": spe.SyntheticLandmarkExtractorConfig(
                log_grid_spacing=model_config["log_grid_spacing"],
                grid_bounds_px=640,
                should_produce_bearing_position_for_pano=False,
                embedding_dim=model_config["syn_embedding_dim"]
                )},
        use_cached_extractors=[])

    output_dir = f'all_chicago_logGridSpacing_{model_config["log_grid_spacing"]}_outputDim_{model_config["output_dim"]}_hinge_{model_config["cos_mean_hinge"]}_isPanoPosPlanar_{model_config["is_pano_pos_planar"]}_synEmbeddingDim_{model_config["syn_embedding_dim"]:02d}'
    dataset_config = T.DatasetConfig(
        paths=["Chicago"],
        should_load_images=False,
        landmark_version='v1',
        factor=1.0)
    validation_dataset_configs = [
        T.DatasetConfig(
            paths=["Seattle"],
            should_load_images=False,
            landmark_version='v1',
            factor=0.3),
        T.DatasetConfig(
            paths=["Chicago"],
            should_load_images=False,
            landmark_version='v1',
            factor=0.3),
    ]

    distance_model_config = distances.CosineDistanceConfig()

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
    is_pano_pos_planar = [False, True]
    embedding_dim = [64, 32, 16, 8, 4, 2, 0]

    for emb_dim, is_planar in itertools.product(embedding_dim, is_pano_pos_planar):
        cfg = generate_config({
                "output_dim": 256,
                "num_transformer_layers": 2,
                "log_grid_spacing": 7,
                "cos_mean_hinge": True,
                "is_pano_pos_planar": is_planar,
                "syn_embedding_dim": emb_dim,
                })

        file_name = Path(f"/tmp/{cfg.output_dir}.yaml")
        yaml_config = msgspec.yaml.encode(cfg, enc_hook=msgspec_enc_hook)
        file_name.write_bytes(yaml_config)
