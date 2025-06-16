import itertools
import msgspec
from pathlib import Path

from experimental.overhead_matching.swag.scripts.train import OptimizationConfig, LearningRateSchedule
from experimental.overhead_matching.swag.data.vigor_dataset import HardNegativeMiner

if __name__ == "__main__":
    for (has_lr_schedule, has_hard_negative_mining, has_pos_semipos) in itertools.product(*[[False, True]]*3):
        print(f"{has_lr_schedule=} {has_hard_negative_mining=} {has_pos_semipos=}")

        NUM_EPOCHS = 60
        random_sample_type = (
                HardNegativeMiner.RandomSampleType.POS_SEMIPOS
                if has_pos_semipos else
                HardNegativeMiner.RandomSampleType.NEAREST)

        hard_mining_after_epoch = 20 if has_hard_negative_mining else NUM_EPOCHS

        out = OptimizationConfig(
            num_epochs=NUM_EPOCHS,
            batch_size=18,
            lr_schedule=LearningRateSchedule(
                    initial_lr=1e-4,
                    lr_step_factor=0.25 if has_lr_schedule else 1.0,
                    num_epochs_at_lr=20
                ),
            enable_hard_negative_sampling_after_epoch_idx=hard_mining_after_epoch,
            random_sample_type=random_sample_type
        )

        file_name = Path(f"/tmp/all_chicago_lr_schedule_{has_lr_schedule}_negative_mining_{has_hard_negative_mining}_pos_semipos_{has_pos_semipos}.yaml")
        yaml_config = msgspec.yaml.encode(out)
        file_name.write_bytes(yaml_config)

