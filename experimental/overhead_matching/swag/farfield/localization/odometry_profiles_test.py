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
    def test_turn_can_produce_zero_variance_increments(self):
        nominal = [structs.OdometryDelta(
            keyframe_idx=k, forward_m=10.0, left_m=0.0,
            delta_yaw_cw_rad=math.pi if k == 5 else 0.0,
            sigma_m=0.0, sigma_yaw_rad=0.0) for k in range(1, 13)]
        actual, _ = odometry_profiles._derive_planar_imu_v1(
            nominal, [100.0 * k for k in range(13)], 0, "out_and_back")
        self.assertTrue(any(delta.sigma_m == 0.0 for delta in actual))
        self.assertTrue(any(delta.sigma_m > 0.0 for delta in actual))
        for delta in actual:
            self.assertTrue(math.isfinite(delta.sigma_m))
            self.assertGreaterEqual(delta.sigma_m, 0.0)
            self.assertGreater(delta.sigma_yaw_rad, 0.0)

    def test_episode_slices_clean_motion_then_resets_and_pairs_noise(self):
        configured = {
            'odometry_sigma_pair_m': 1.0, 'displacement_gate_m': 2.0,
            'stationary_sigma_m': 3.0, 'slow_yaw_sigma_deg': 30.0,
            'course_yaw_drift_sigma_deg': 2.0,
            'imu_translation_noise_frac': 0.02, 'imu_yaw_noise_frac': 0.01,
            'reverse_keyframe_ranges': [[1, 4]], 'reverse_annotation_source': 'test',
        }
        data = types.SimpleNamespace(
            truth=[structs.TruthPose(i, 0.0, 10.0 * i, 0.0) for i in range(5)],
            artifact_ref=types.SimpleNamespace(dataset='same_parent'),
            manifest=types.SimpleNamespace(config={'localization_inputs': configured}),
            meta=types.SimpleNamespace(motion={'content_sha256': 'motion'}))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / 'motion_source.csv').write_text('idx,video_t_s\n0,0\n1,100\n2,200\n3,201\n4,202\n')
            full, _ = odometry_profiles.derive(path, data, odometry_profiles.PLANAR_IMU_PROFILE)
            with mock.patch.object(odometry_profiles, 'derive_from_motion',
                                   wraps=odometry_profiles.derive_from_motion) as derive:
                episode, meta = odometry_profiles.derive(
                    path, data, odometry_profiles.PLANAR_IMU_PROFILE, keyframe_range=(2, 4))
                args = derive.call_args.args
                self.assertEqual(args[:3], ([0.0, 0.0, 0.0], [20.0, 30.0, 40.0], [200.0, 201.0, 202.0]))
                self.assertEqual(args[3]['reverse_keyframe_ranges'], [[1, 2]])
            repeated, paired = odometry_profiles.derive(
                path, data, odometry_profiles.PLANAR_IMU_PROFILE, keyframe_range=(2, 4))
            different, _ = odometry_profiles.derive(
                path, data, odometry_profiles.PLANAR_IMU_PROFILE, noise_seed=1, keyframe_range=(2, 4))
            _, other_window = odometry_profiles.derive(
                path, data, odometry_profiles.PLANAR_IMU_PROFILE, noise_seed=0, keyframe_range=(1, 3))
            self.assertEqual(episode, repeated)
            self.assertEqual(meta, paired)
            self.assertNotEqual(episode, different)
            self.assertNotEqual(meta['noise']['realization'], other_window['noise']['realization'])
            self.assertEqual([d.keyframe_idx for d in episode], [1, 2])
            self.assertLess(episode[0].sigma_m, full[2].sigma_m / 100)
            self.assertEqual(meta['noise']['dataset_stream_id'], 'same_parent/episode_kf_2_4')
            self.assertEqual(configured['reverse_keyframe_ranges'], [[1, 4]])
            with self.assertRaises(ValueError):
                odometry_profiles.derive(path, data, 'recorded', keyframe_range=(2, 4))

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
