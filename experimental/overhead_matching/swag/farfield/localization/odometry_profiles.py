"""Experiment-only odometry profiles over a validated localization export."""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    gps_to_odometry,
    structs,
)


PLANAR_IMU_PROFILE = "epson_mg570_calibrated_planar_v1"
PROFILE_CHOICES = ("recorded", PLANAR_IMU_PROFILE)
EPSON_ARW_DEG_SQRT_HR = 0.04
EPSON_BIAS_SIGMA_DEG_PER_HR = 0.5
EPSON_ACCEL_BIAS_SIGMA_MICRO_G = 14.0
EPSON_ACCEL_VRW_MPS_SQRT_HR = 0.012
STANDARD_GRAVITY_MPS2 = 9.80665
DEFAULT_NOISE_SEED = 0
EPSON_DATASHEET = (
    "https://www.epsondevice.com/sensing/en/pdf/"
    "m-g570pr20_datasheet_e_rev1_4.pdf")
_CONFIG_KEYS = (
    "odometry_sigma_pair_m",
    "displacement_gate_m",
    "stationary_sigma_m",
    "slow_yaw_sigma_deg",
    "course_yaw_drift_sigma_deg",
    "imu_translation_noise_frac",
    "imu_yaw_noise_frac",
    "reverse_keyframe_ranges",
    "reverse_annotation_source",
)


def _selected_config(data):
    selected = data.manifest.config.get("localization_inputs")
    if not isinstance(selected, dict):
        raise ValueError("manifest does not record localization_inputs")
    missing = [key for key in _CONFIG_KEYS if key not in selected]
    if missing:
        raise ValueError(f"manifest odometry config is missing {missing}")
    return {key: selected[key] for key in _CONFIG_KEYS}


def load_timestamps(input_dir: Path, n_keyframes: int):
    path = Path(input_dir) / "motion_source.csv"
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        indices = [int(row["idx"]) for row in rows]
        timestamps = [float(row["video_t_s"]) for row in rows]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read timestamps from {path}: {exc}") from exc
    if indices != list(range(n_keyframes)):
        raise ValueError("motion_source.csv indices do not match keyframes")
    return timestamps


def _derive_planar_imu_v1(nominal, timestamps, noise_seed, stream_id):
    """Apply the locked calibrated planar IMU error model to one trajectory."""
    if len(timestamps) != len(nominal) + 1 \
            or any(not math.isfinite(value) for value in timestamps) \
            or any(end <= start for start, end in zip(
                timestamps[:-1], timestamps[1:])):
        raise ValueError("planar IMU timestamps must be finite and increasing")
    if (isinstance(noise_seed, bool) or not isinstance(noise_seed, int)
            or noise_seed < 0):
        raise ValueError("noise_seed must be a nonnegative integer")
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream_id must be a non-empty string")
    stream_sha256 = hashlib.sha256(stream_id.encode("utf-8")).hexdigest()
    stream_words = [
        int(stream_sha256[offset:offset + 8], 16)
        for offset in range(0, len(stream_sha256), 8)]
    streams = np.random.SeedSequence(
        [noise_seed, *stream_words]).spawn(4)
    gyro_bias_rng, gyro_white_rng, accel_bias_rng, accel_white_rng = (
        np.random.default_rng(stream) for stream in streams)
    gyro_bias_deg_per_hr = float(gyro_bias_rng.normal(
        0.0, EPSON_BIAS_SIGMA_DEG_PER_HR))
    accel_bias_sigma_mps2 = (
        EPSON_ACCEL_BIAS_SIGMA_MICRO_G * 1e-6 * STANDARD_GRAVITY_MPS2)
    accel_bias_body_mps2 = accel_bias_rng.normal(
        0.0, accel_bias_sigma_mps2, size=2)
    accel_white_mps_sqrt_s = EPSON_ACCEL_VRW_MPS_SQRT_HR / 60.0

    nominal_yaw = 0.0
    nominal_position = np.zeros(2, dtype=np.float64)
    position_error = np.zeros(2, dtype=np.float64)
    velocity_error = np.zeros(2, dtype=np.float64)
    bias_velocity_map = np.zeros((2, 2), dtype=np.float64)
    bias_position_map = np.zeros((2, 2), dtype=np.float64)
    position_variance_envelope = 0.0
    yaw_variance_deg2 = 0.0
    elapsed_s = 0.0
    noisy_positions = [nominal_position.copy()]
    nominal_yaws = []
    yaw_deltas = []
    yaw_sigmas = []
    translation_sigmas = []

    for delta, start_s, end_s in zip(
            nominal, timestamps[:-1], timestamps[1:]):
        dt_s = float(end_s - start_s)
        dt_hr = dt_s / 3600.0
        nominal_yaw += delta.delta_yaw_cw_rad
        sin_yaw, cos_yaw = math.sin(nominal_yaw), math.cos(nominal_yaw)
        body_to_world = np.asarray([
            [sin_yaw, -cos_yaw],
            [cos_yaw, sin_yaw],
        ])

        nominal_position += body_to_world @ np.asarray([
            delta.forward_m, delta.left_m])
        accel_bias_world = body_to_world @ accel_bias_body_mps2
        z_velocity = accel_white_rng.normal(size=2)
        z_position = accel_white_rng.normal(size=2)
        white_velocity = (
            accel_white_mps_sqrt_s * math.sqrt(dt_s) * z_velocity)
        white_position = accel_white_mps_sqrt_s * dt_s ** 1.5 * (
            0.5 * z_velocity + z_position / math.sqrt(12.0))
        position_error += (
            velocity_error * dt_s
            + 0.5 * accel_bias_world * dt_s ** 2
            + white_position)
        velocity_error += accel_bias_world * dt_s + white_velocity
        noisy_positions.append((nominal_position + position_error).copy())
        nominal_yaws.append(nominal_yaw)

        bias_position_map += (
            bias_velocity_map * dt_s
            + 0.5 * body_to_world * dt_s ** 2)
        bias_velocity_map += body_to_world * dt_s
        elapsed_s += dt_s
        covariance = (
            accel_bias_sigma_mps2 ** 2
            * bias_position_map @ bias_position_map.T
            + accel_white_mps_sqrt_s ** 2
            * elapsed_s ** 3 / 3.0
            * np.eye(2))
        cumulative_variance = float(np.linalg.eigvalsh(covariance)[-1])
        next_envelope = max(
            position_variance_envelope, cumulative_variance)
        translation_sigmas.append(math.sqrt(max(
            0.0, next_envelope - position_variance_envelope)))
        position_variance_envelope = next_envelope

        white_yaw_sigma_deg = (
            EPSON_ARW_DEG_SQRT_HR * math.sqrt(dt_hr))
        white_yaw_deg = float(gyro_white_rng.normal(
            0.0, white_yaw_sigma_deg))
        yaw_deltas.append((
            delta.delta_yaw_cw_rad
            + math.radians(gyro_bias_deg_per_hr * dt_hr + white_yaw_deg)
            + math.pi) % (2.0 * math.pi) - math.pi)
        elapsed_hr = elapsed_s / 3600.0
        next_yaw_variance_deg2 = (
            EPSON_ARW_DEG_SQRT_HR ** 2 * elapsed_hr
            + EPSON_BIAS_SIGMA_DEG_PER_HR ** 2 * elapsed_hr ** 2)
        yaw_sigmas.append(math.radians(math.sqrt(max(
            0.0, next_yaw_variance_deg2 - yaw_variance_deg2))))
        yaw_variance_deg2 = next_yaw_variance_deg2

    odometry = []
    for index, delta in enumerate(nominal):
        world_delta = noisy_positions[index + 1] - noisy_positions[index]
        sin_yaw, cos_yaw = (
            math.sin(nominal_yaws[index]), math.cos(nominal_yaws[index]))
        odometry.append(structs.OdometryDelta(
            keyframe_idx=delta.keyframe_idx,
            forward_m=float(
                world_delta[0] * sin_yaw + world_delta[1] * cos_yaw),
            left_m=float(
                -world_delta[0] * cos_yaw + world_delta[1] * sin_yaw),
            delta_yaw_cw_rad=yaw_deltas[index],
            sigma_m=translation_sigmas[index],
            sigma_yaw_rad=yaw_sigmas[index]))
    return odometry, {
        "gyro_bias_deg_per_hr": gyro_bias_deg_per_hr,
        "accel_bias_body_mps2": [
            float(value) for value in accel_bias_body_mps2],
        "stream_id": stream_id,
        "stream_sha256": stream_sha256,
    }


def derive(input_dir: Path, data, profile: str,
           noise_seed: int = DEFAULT_NOISE_SEED):
    """Return ``(odometry, metadata)`` without mutating the export data."""
    if profile not in PROFILE_CHOICES:
        raise ValueError(f"unknown odometry profile {profile!r}")
    configured = _selected_config(data)
    if profile == "recorded":
        return data.odometry, {
            "name": profile,
            "source": "localization_inputs/tier1_odometry.jsonl",
            "parameters": configured,
        }

    truth_indices = [pose.keyframe_idx for pose in data.truth]
    if truth_indices != list(range(len(data.truth))):
        raise ValueError("Epson odometry profile requires contiguous truth")
    timestamps = load_timestamps(input_dir, len(data.truth))
    if profile == PLANAR_IMU_PROFILE:
        nominal = gps_to_odometry.derive_increments(
            [pose.east_m for pose in data.truth],
            [pose.north_m for pose in data.truth],
            sigma_pair_m=configured["odometry_sigma_pair_m"],
            displacement_gate_m=configured["displacement_gate_m"],
            stationary_sigma_m=configured["stationary_sigma_m"],
            slow_yaw_sigma_deg=configured["slow_yaw_sigma_deg"],
            course_yaw_drift_sigma_deg=(
                configured["course_yaw_drift_sigma_deg"]),
            reverse_keyframe_ranges=configured["reverse_keyframe_ranges"],
            imu_translation_noise_frac=0.0,
            imu_yaw_noise_frac=0.0)
        odometry, realization = _derive_planar_imu_v1(
            nominal, timestamps, noise_seed, data.artifact_ref.dataset)
        timestamp_meta = data.meta.motion
        accel_bias_sigma_mps2 = (
            EPSON_ACCEL_BIAS_SIGMA_MICRO_G * 1e-6
            * STANDARD_GRAVITY_MPS2)
        return odometry, {
            "name": profile,
            "schema": "epson_mg570_calibrated_planar/v1",
            "source": "synthetic_from_truth_planar_inertial_error_model",
            "position_source": "localization_inputs/truth.jsonl",
            "timestamp_source": {
                "file": "motion_source.csv",
                "column": "video_t_s",
                "content_sha256": timestamp_meta["content_sha256"],
                "n_keyframes": len(timestamps),
                "start_s": timestamps[0],
                "end_s": timestamps[-1],
            },
            "yaw_model": {
                "type": "white_arw_plus_constant_trajectory_rate_bias",
                "arw_deg_sqrt_hr": EPSON_ARW_DEG_SQRT_HR,
                "bias_sigma_deg_per_hr": EPSON_BIAS_SIGMA_DEG_PER_HR,
                "sigma_yaw_rad": (
                    "increments of cumulative ARW_plus_bias_variance"),
                "temporal_correlations_serialized": False,
            },
            "translation_model": {
                "type": "stateful_planar_accelerometer_preintegration",
                "velocity_random_walk_mps_sqrt_hr":
                    EPSON_ACCEL_VRW_MPS_SQRT_HR,
                "velocity_random_walk_mps_sqrt_s":
                    EPSON_ACCEL_VRW_MPS_SQRT_HR / 60.0,
                "bias_sigma_micro_g": EPSON_ACCEL_BIAS_SIGMA_MICRO_G,
                "bias_sigma_mps2": accel_bias_sigma_mps2,
                "bias_frame": "fixed_sensor_forward_left",
                "velocity_position_error_reset": "trajectory_start_only",
                "white_noise_discretization": (
                    "continuous_white_acceleration_exact_interval_covariance"),
                "sigma_m": (
                    "increments of the nondecreasing cumulative largest_axis_"
                    "covariance_envelope"),
                "temporal_correlations_serialized": False,
                "pose_filter_interpretation": (
                    "conservative_marginal_moment_match_for_pose_only_grid_"
                    "or_particle_filter"),
            },
            "noise": {
                "base_seed": noise_seed,
                "dataset_stream_id": data.artifact_ref.dataset,
                "dataset_stream_sha256": realization["stream_sha256"],
                "seed_composition": "base_seed_plus_dataset_sha256_words",
                "generation_scope": "full_trajectory_before_episode_slicing",
                "independent_streams": [
                    "gyro_bias", "gyro_white", "accel_bias", "accel_white"],
                "realization": realization,
            },
            "calibration_assumptions": {
                "bias_instability_as_constant_residual_1sigma_proxy": True,
                "initial_gyro_bias_error_deg_per_hr_excluded": 360.0,
                "initial_accel_bias_error_milli_g_excluded": 2.0,
                "initial_velocity_error_mps": 0.0,
                "perfect_roll_pitch_and_gravity_compensation": True,
                "nominal_temperature_c": 25.0,
                "omitted_errors": [
                    "scale_factor", "misalignment", "temperature_change",
                    "vibration", "clock", "bias_time_variation",
                    "gyro_attitude_error_coupling_into_specific_force_and_"
                    "gravity",
                ],
                "interpretation": (
                    "optimistic_calibrated_synthetic_model_not_hardware_"
                    "validation"),
            },
            "datasheet": EPSON_DATASHEET,
            "reverse_keyframe_ranges": configured["reverse_keyframe_ranges"],
            "reverse_annotation_source":
                configured["reverse_annotation_source"],
        }

    raise AssertionError("unreachable")
