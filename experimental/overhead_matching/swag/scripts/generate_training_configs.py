import itertools
import msgspec
from pathlib import Path

import experimental.overhead_matching.swag.scripts.train as T
import experimental.overhead_matching.swag.model.patch_embedding as pe
from experimental.overhead_matching.swag.data.vigor_dataset import HardNegativeMiner
import experimental.overhead_matching.swag.model.patch_embedding as pe


def enc_hook(obj):
    if isinstance(obj, Path):
        return str(obj)
    else:
        raise ValueError(f"Unhandled Value: {obj}")


if __name__ == "__main__":
    for (has_lr_schedule, has_hard_negative_mining, has_pos_semipos) in itertools.product(*[[False, True]]*3):
        print(f"{has_lr_schedule=} {has_hard_negative_mining=} {has_pos_semipos=}")
        NUM_EPOCHS = 60
        random_sample_type = (
                HardNegativeMiner.RandomSampleType.POS_SEMIPOS
                if has_pos_semipos else
                HardNegativeMiner.RandomSampleType.NEAREST)

        hard_mining_after_epoch = 20 if has_hard_negative_mining else NUM_EPOCHS

        exp_name = f"all_chicago_lr_schedule_{has_lr_schedule}_negative_mining_{has_hard_negative_mining}_pos_semipos_{has_pos_semipos}"

        out = T.TrainConfig(
            opt_config=T.OptimizationConfig(
                num_epochs=NUM_EPOCHS,
                batch_size=18,
                lr_schedule=T.LearningRateSchedule(
                        initial_lr=1e-4,
                        lr_step_factor=0.25 if has_lr_schedule else 1.0,
                        num_epochs_at_lr=20
                    ),
                enable_hard_negative_sampling_after_epoch_idx=hard_mining_after_epoch,
                random_sample_type=random_sample_type),
            sat_model_config=pe.WagPatchEmbeddingConfig(
                patch_dims=(320, 320),
                num_aggregation_heads=4,
                backbone_config=pe.VGGConfig()),
            pano_model_config=pe.WagPatchEmbeddingConfig(
                patch_dims=(320, 640),
                num_aggregation_heads=4,
                backbone_config=pe.VGGConfig()),
            output_dir=Path(exp_name),
            tensorboard_output=None,
            dataset_path=[Path("Chicago")])

        file_name = Path(f"/tmp/{exp_name}.yaml")
        yaml_config = msgspec.yaml.encode(out, enc_hook=enc_hook)
        file_name.write_bytes(yaml_config)

    for project_dim in [None, 512, 768, 1024]:
        print(f"{project_dim=}")
        NUM_EPOCHS = 60
        random_sample_type = (HardNegativeMiner.RandomSampleType.NEAREST)

        hard_mining_after_epoch = NUM_EPOCHS

        exp_name = f"all_chicago_dino_project_{project_dim}"

        out = T.TrainConfig(
                opt_config=T.OptimizationConfig(
                    num_epochs=NUM_EPOCHS,
                    batch_size=18,
                    lr_schedule=T.LearningRateSchedule(
                            initial_lr=1e-4,
                            lr_step_factor=1.0,
                            num_epochs_at_lr=20),
                    enable_hard_negative_sampling_after_epoch_idx=hard_mining_after_epoch,
                    random_sample_type=random_sample_type),
                sat_model_config=pe.WagPatchEmbeddingConfig(
                    patch_dims=(322, 322),
                    num_aggregation_heads=4,
                    backbone_config=pe.DinoConfig(project=project_dim)),
                pano_model_config=pe.WagPatchEmbeddingConfig(
                    patch_dims=(322, 644),
                    num_aggregation_heads=4,
                    backbone_config=pe.DinoConfig(project=project_dim)),
                output_dir=Path(exp_name),
                tensorboard_output=None,
                dataset_path=[Path("Chicago")])

        file_name = Path(f"/tmp/{exp_name}.yaml")
        yaml_config = msgspec.yaml.encode(out, enc_hook=enc_hook)
        file_name.write_bytes(yaml_config)

