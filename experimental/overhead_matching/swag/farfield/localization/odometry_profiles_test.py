import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.localization import (
    odometry_profiles,
    structs,
)


class OdometryProfilesTest(unittest.TestCase):
    def test_planar_v1_rotates_fixed_body_bias_through_a_right_turn(self):
        nominal = [
            structs.OdometryDelta(
                keyframe_idx=1, forward_m=10.0, left_m=0.0,
                delta_yaw_cw_rad=0.0, sigma_m=0.0,
                sigma_yaw_rad=0.0),
            structs.OdometryDelta(
                keyframe_idx=2, forward_m=10.0, left_m=0.0,
                delta_yaw_cw_rad=math.pi / 2.0, sigma_m=0.0,
                sigma_yaw_rad=0.0),
        ]
        with mock.patch.object(
                odometry_profiles, "EPSON_ARW_DEG_SQRT_HR", 0.0), \
                mock.patch.object(
                    odometry_profiles,
                    "EPSON_ACCEL_VRW_MPS_SQRT_HR", 0.0):
            actual, realization = odometry_profiles._derive_planar_imu_v1(
                nominal, [0.0, 1.0, 2.0], 0, "right_turn")

        bias_forward, bias_left = realization["accel_bias_body_mps2"]
        self.assertAlmostEqual(
            actual[0].forward_m, 10.0 + 0.5 * bias_forward)
        self.assertAlmostEqual(actual[0].left_m, 0.5 * bias_left)
        self.assertAlmostEqual(
            actual[1].forward_m,
            10.0 - bias_left + 0.5 * bias_forward)
        self.assertAlmostEqual(
            actual[1].left_m, bias_forward + 0.5 * bias_left)

    def test_planar_v1_carries_velocity_error_and_uses_timestamps(self):
        truth = [structs.TruthPose(
            keyframe_idx=index, east_m=0.0, north_m=10.0 * index,
            course_world_cw_deg=0.0) for index in range(4)]
        selected = {
            "odometry_sigma_pair_m": 1.0,
            "displacement_gate_m": 2.0,
            "stationary_sigma_m": 3.0,
            "slow_yaw_sigma_deg": 30.0,
            "course_yaw_drift_sigma_deg": 2.0,
            "imu_translation_noise_frac": 0.02,
            "imu_yaw_noise_frac": 0.01,
            "reverse_keyframe_ranges": [],
            "reverse_annotation_source": "reviewed_no_reverse",
        }
        data = types.SimpleNamespace(
            truth=truth, odometry=[],
            artifact_ref=types.SimpleNamespace(dataset="test_dataset"),
            manifest=types.SimpleNamespace(
                config={"localization_inputs": selected}),
            meta=types.SimpleNamespace(
                motion={"content_sha256": "motion-digest"}))

        with tempfile.TemporaryDirectory() as root:
            input_dir = Path(root)
            (input_dir / "motion_source.csv").write_text(
                "idx,video_t_s\n0,0\n1,1\n2,3\n3,6\n")
            with mock.patch.object(
                    odometry_profiles, "EPSON_ARW_DEG_SQRT_HR", 0.0), \
                    mock.patch.object(
                        odometry_profiles,
                        "EPSON_ACCEL_VRW_MPS_SQRT_HR", 0.0):
                actual, metadata = odometry_profiles.derive(
                    input_dir, data, odometry_profiles.PLANAR_IMU_PROFILE)
                repeated, _ = odometry_profiles.derive(
                    input_dir, data, odometry_profiles.PLANAR_IMU_PROFILE)
                different, different_metadata = odometry_profiles.derive(
                    input_dir, data, odometry_profiles.PLANAR_IMU_PROFILE,
                    noise_seed=7)
                data.artifact_ref = types.SimpleNamespace(
                    dataset="other_dataset")
                other_dataset, _ = odometry_profiles.derive(
                    input_dir, data, odometry_profiles.PLANAR_IMU_PROFILE)

        self.assertEqual(actual, repeated)
        self.assertNotEqual(actual, different)
        self.assertNotEqual(actual, other_dataset)
        self.assertEqual(different_metadata["noise"]["base_seed"], 7)
        bias_forward, bias_left = metadata["noise"]["realization"][
            "accel_bias_body_mps2"]
        gyro_bias = metadata["noise"]["realization"][
            "gyro_bias_deg_per_hr"]
        coefficients = (0.5, 4.0, 13.5)
        durations = (1.0, 2.0, 3.0)
        cumulative_times = (1.0, 3.0, 6.0)
        previous_time = 0.0
        accel_sigma = metadata["translation_model"]["bias_sigma_mps2"]
        for delta, coefficient, duration, cumulative_time in zip(
                actual, coefficients, durations, cumulative_times):
            self.assertAlmostEqual(
                delta.forward_m, 10.0 + coefficient * bias_forward)
            self.assertAlmostEqual(
                delta.left_m, coefficient * bias_left)
            expected_sigma_coefficient = 0.5 * math.sqrt(
                cumulative_time ** 4 - previous_time ** 4)
            self.assertAlmostEqual(
                delta.sigma_m,
                expected_sigma_coefficient * accel_sigma)
            self.assertAlmostEqual(
                delta.delta_yaw_cw_rad,
                math.radians(gyro_bias * duration / 3600.0))
            self.assertAlmostEqual(
                delta.sigma_yaw_rad,
                math.radians(
                    odometry_profiles.EPSON_BIAS_SIGMA_DEG_PER_HR
                    * math.sqrt(
                        cumulative_time ** 2 - previous_time ** 2)
                    / 3600.0))
            previous_time = cumulative_time
        self.assertEqual(
            metadata["schema"], "epson_mg570_calibrated_planar/v1")
        self.assertEqual(
            metadata["translation_model"]["velocity_position_error_reset"],
            "trajectory_start_only")
        self.assertTrue(metadata["calibration_assumptions"][
            "perfect_roll_pitch_and_gravity_compensation"])
        self.assertEqual(
            metadata["noise"]["dataset_stream_id"], "test_dataset")


if __name__ == "__main__":
    unittest.main()
