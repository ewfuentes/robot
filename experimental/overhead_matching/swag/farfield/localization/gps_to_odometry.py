"""Derive nominal-forward dead-reckoning odometry from GPS fixes (§5.2).

The deployed system has no GPS; GPS exists only in data collection. This
producer turns a sequence of ENU fixes into the OdometryDelta increments the
filter consumes, the way §5.2 specifies:

  GPS course is a surrogate for nominal-forward orientation during ordinary
  forward travel. A usable chord becomes forward=length, left=0. A
  human-reviewed reverse chord becomes forward=-length and its course is
  rotated 180 degrees before yaw differencing, so reversing does not invent a
  platform turn. Crab/current remain declared motion-model uncertainty.

  delta_yaw_cw = differenced usable course proxy. Course is the direction of
  each step; its noise
  is geometric — sigma_course ~ atan(sigma_pair / step) — so it is computed
  per step from the step length rather than declared as a constant. A step
  below the displacement gate emits zero translation and zero yaw with
  explicitly inflated uncertainties. This prevents stationary GPS jitter
  from accumulating as false travel. When usable courses are separated by a
  gap, the catch-up yaw
  spans the whole gap (its measurement noise still telescopes to the two
  endpoint course sigmas); the gapped steps in between already carried the
  inflated sigma.

  sigma_m is the honest per-fix-pair constant (~1 m: correlated absolute GPS
  error differences out). No IMU-style step scaling is cosplayed onto real
  data — emulating worse odometry is an explicit, labelled experiment via
  the extra-noise parameters, never a default.

Differenced-course yaw is substantially noisier than a real gyro, which is
the safe direction for the paper's claims: convergence demonstrated on
course-grade yaw lower-bounds what an IMU would deliver.

The serialized motion convention is rotate-then-move and clockwise-positive:
``yaw_k = yaw_{k-1} + delta_yaw_cw_rad`` and then translation is rotated by
``yaw_k``. It is deliberately not called generic SE(2).

Publication belongs to ``build_export`` and its transactional
``localization_inputs`` artifact.  This module intentionally exposes only the
pure derivation boundary so there is no second, unmanifested export writer.
"""

import math

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import structs


def _finite_nonnegative(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value


def _reverse_keyframes(reverse_keyframe_ranges, n_steps: int) -> set[int]:
    if not isinstance(reverse_keyframe_ranges, (list, tuple)):
        raise ValueError("reverse_keyframe_ranges must be a list or tuple")
    result = set()
    previous_end = 0
    for index, interval in enumerate(reverse_keyframe_ranges):
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in interval)):
            raise ValueError(
                f"reverse_keyframe_ranges[{index}] must be [start, end] ints")
        start, end = interval
        if start < 1 or end < start or end > n_steps:
            raise ValueError(
                f"reverse range [{start}, {end}] is outside increments "
                f"1..{n_steps}")
        if start <= previous_end:
            raise ValueError("reverse ranges must be sorted and non-overlapping")
        result.update(range(start, end + 1))
        previous_end = end
    return result


def derive_increments(east_m, north_m, *,
                      sigma_pair_m: float,
                      displacement_gate_m: float,
                      stationary_sigma_m: float,
                      slow_yaw_sigma_deg: float,
                      reverse_keyframe_ranges,
                      extra_sigma_m: float = 0.0,
                      extra_yaw_sigma_deg: float = 0.0,
                      noise_seed: int = 0) -> list:
    """ENU fixes (keyframes 0..N) -> OdometryDelta increments (1..N).

    Baseline values and reverse annotations are required keywords — callers
    pass immutable build-config values, so the recorded recipe shaped the
    odometry. The
    extra-noise parameters inject additional noise AND declare it (an honest
    producer emulating a worse sensor, not a lying one); they exist for
    drift-injection experiments and default to off.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    if (east_m.shape != north_m.shape or east_m.ndim != 1
            or east_m.size < 2 or not np.all(np.isfinite(east_m))
            or not np.all(np.isfinite(north_m))):
        raise ValueError("need matching 1-D east/north arrays of >= 2 fixes")
    sigma_pair_m = _finite_nonnegative(
        sigma_pair_m, "sigma_pair_m", positive=True)
    displacement_gate_m = _finite_nonnegative(
        displacement_gate_m, "displacement_gate_m", positive=True)
    stationary_sigma_m = _finite_nonnegative(
        stationary_sigma_m, "stationary_sigma_m", positive=True)
    if stationary_sigma_m < sigma_pair_m:
        raise ValueError("stationary_sigma_m must be >= sigma_pair_m")
    slow_yaw_sigma_deg = _finite_nonnegative(
        slow_yaw_sigma_deg, "slow_yaw_sigma_deg", positive=True)
    extra_sigma_m = _finite_nonnegative(extra_sigma_m, "extra_sigma_m")
    extra_yaw_sigma_deg = _finite_nonnegative(
        extra_yaw_sigma_deg, "extra_yaw_sigma_deg")
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        raise ValueError("noise_seed must be an integer")
    reverse = _reverse_keyframes(reverse_keyframe_ranges, east_m.size - 1)

    rng = np.random.default_rng(noise_seed)
    slow_sigma_rad = math.radians(slow_yaw_sigma_deg)
    extra_yaw_rad = math.radians(extra_yaw_sigma_deg)
    inject = extra_sigma_m > 0.0 or extra_yaw_rad > 0.0

    prev_course_rad = None  # last USABLE course
    prev_course_sigma_rad = None
    increments = []
    for kf in range(1, east_m.size):
        d_east = float(east_m[kf] - east_m[kf - 1])
        d_north = float(north_m[kf] - north_m[kf - 1])
        step_m = math.hypot(d_east, d_north)

        delta_yaw_cw_rad = 0.0
        sigma_yaw_rad = slow_sigma_rad
        if step_m >= displacement_gate_m:
            course_rad = math.atan2(d_east, d_north)
            if kf in reverse:
                # A reverse chord points aft; rotate it to the platform's
                # nominal-forward proxy before differencing yaw.
                course_rad = float(geo.wrap_rad(course_rad + math.pi))
            course_sigma_rad = math.atan(sigma_pair_m / step_m)
            if prev_course_rad is not None:
                delta_yaw_cw_rad = float(geo.wrap_rad(course_rad - prev_course_rad))
                sigma_yaw_rad = math.hypot(course_sigma_rad,
                                           prev_course_sigma_rad)
            prev_course_rad = course_rad
            prev_course_sigma_rad = course_sigma_rad

            forward_m = -step_m if kf in reverse else step_m
            sigma_m = sigma_pair_m
        else:
            forward_m = 0.0
            sigma_m = stationary_sigma_m
        left_m = 0.0
        if inject:
            forward_m += float(rng.normal(0.0, extra_sigma_m))
            left_m += float(rng.normal(0.0, extra_sigma_m))
            delta_yaw_cw_rad = float(geo.wrap_rad(
                delta_yaw_cw_rad + rng.normal(0.0, extra_yaw_rad)))
            sigma_m = math.hypot(sigma_m, extra_sigma_m)
            sigma_yaw_rad = math.hypot(sigma_yaw_rad, extra_yaw_rad)

        increments.append(structs.OdometryDelta(
            keyframe_idx=kf,
            forward_m=forward_m,
            left_m=left_m,
            delta_yaw_cw_rad=delta_yaw_cw_rad,
            sigma_m=sigma_m,
            sigma_yaw_rad=sigma_yaw_rad))
    return increments
