"""Derive nominal-forward dead-reckoning odometry from GPS fixes (§5.2).

The deployed system has no GPS; GPS exists only in data collection. This
producer turns a sequence of ENU fixes into the OdometryDelta increments the
filter consumes, the way §5.2 specifies:

  GPS course is a surrogate for nominal-forward orientation during ordinary
  forward travel. A usable chord becomes forward=length, left=0. A
  human-reviewed reverse chord becomes forward=-length and its course is
  rotated 180 degrees before yaw differencing, so reversing does not invent a
  platform turn. Crab/current remain declared motion-model uncertainty.

  delta_yaw_cw = differenced usable course proxy. Its measurement noise is
  ANTI-CORRELATED across steps: consecutive deltas share the middle course,
  so the increments telescope and the integrated yaw error stays bounded by
  the two endpoint course sigmas (~atan(sigma_pair / step)) no matter how
  many steps compose. The filter composes sigma_yaw_rad as independent
  per-step noise, so emitting the per-chord course noise here overstates
  heading drift by sqrt(n): on mount_washington_20260815_leg3 that meant a
  ~26 deg/keyframe modeled random walk against a truly bounded ~26 deg
  total, which caused whole-map mode death and 20x+ seed variance
  (tn-mass@500m 0.008-0.061 across seeds; 0.50-0.53 with this fixed —
  see PR #695's A/B). sigma_yaw_rad on a differenced step is therefore
  `course_yaw_drift_sigma_deg`, a small per-step budget for genuinely
  accumulating error (course smoothing, crab, timing) — not the endpoint
  measurement noise. A step below the displacement gate emits zero
  translation and zero yaw with explicitly inflated uncertainties,
  preventing stationary GPS jitter from accumulating as false travel; a
  catch-up yaw after such a gap spans the whole gap and carries the same
  drift budget (its measurement noise still telescopes to the endpoints).

  sigma_m is the honest per-fix-pair chord noise (~1 m: correlated
  absolute GPS error differences out). Unlike yaw, chord noise is NOT
  emitted as a small drift budget: the filter re-integrates increments in
  the heading frame, so consecutive chord errors cancel only where heading
  is unchanged and the telescoping argument fails on turns. Measured
  (mtw leg3 whole-map): sigma_m 0.1 m collapses capture (tn-mass@500m
  0.52 -> 0.05); 0.25/0.5/1.0 m give 0.44/0.47/0.52 — the per-pair value
  is load-bearing, keep it.

  The retained `gps_course_distance_wiener_v1` profile adds independent
  per-step Gaussian noise to make GPS-differenced deltas less friendly than
  their naturally telescoping errors. Its noise is a Wiener process driven
  by distance travelled: per-step sigma is coefficient x sqrt(|forward|).
  The configured coefficients keep ReWAG's 2% odometry / 1% heading values.
  The default Epson profile uses this function only for clean nominal
  geometry, then applies its calibrated time-domain sensor model elsewhere.

  When a per-keyframe nominal-forward heading is available, it replaces GPS
  chord course as the orientation source. GPS still supplies displacement,
  which is projected into the arrival keyframe's forward/left frame. This
  naturally retains sideways and reverse motion, while heading changes remain
  observable even when the displacement gate suppresses GPS jitter.

The serialized noise realization is deterministic (fixed noise_seed), so a
rebuilt export reproduces byte-identical increments.

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
                      course_yaw_drift_sigma_deg: float,
                      reverse_keyframe_ranges,
                      forward_world_cw_deg=None,
                      imu_translation_noise_frac: float = 0.0,
                      imu_yaw_noise_frac: float = 0.0,
                      noise_seed: int = 0) -> list:
    """ENU fixes (keyframes 0..N) -> OdometryDelta increments (1..N).

    Baseline values and reverse annotations are required keywords — callers
    pass immutable build-config values, so the recorded recipe shaped the
    odometry. The imu_* parameters inject independent per-step noise into
    the delta values AND declare it in the emitted sigmas (an honest
    producer emulating the deployed IMU, not a lying one). The retained
    legacy profile config requires them positive; zero remains valid here for
    exact geometry and for the calibrated Epson profile's nominal increments.
    Reverse annotations apply only to the GPS-course fallback; explicit
    headings make their sign/180-degree correction unnecessary.
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
    course_yaw_drift_sigma_deg = _finite_nonnegative(
        course_yaw_drift_sigma_deg, "course_yaw_drift_sigma_deg",
        positive=True)
    imu_translation_noise_frac = _finite_nonnegative(
        imu_translation_noise_frac, "imu_translation_noise_frac")
    imu_yaw_noise_frac = _finite_nonnegative(
        imu_yaw_noise_frac, "imu_yaw_noise_frac")
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        raise ValueError("noise_seed must be an integer")
    reverse = _reverse_keyframes(reverse_keyframe_ranges, east_m.size - 1)
    if forward_world_cw_deg is not None:
        forward_world_cw_deg = np.asarray(
            forward_world_cw_deg, dtype=np.float64)
        if (forward_world_cw_deg.shape != east_m.shape
                or not np.all(np.isfinite(forward_world_cw_deg))
                or np.any((forward_world_cw_deg < 0.0)
                          | (forward_world_cw_deg >= 360.0))):
            raise ValueError(
                "forward_world_cw_deg must be canonical and match GPS fixes")
        forward_world_cw_rad = np.radians(forward_world_cw_deg)
    else:
        forward_world_cw_rad = None

    rng = np.random.default_rng(noise_seed)
    slow_sigma_rad = math.radians(slow_yaw_sigma_deg)
    drift_sigma_rad = math.radians(course_yaw_drift_sigma_deg)
    inject = imu_translation_noise_frac > 0.0 or imu_yaw_noise_frac > 0.0

    prev_course_rad = None  # last USABLE course
    increments = []
    for kf in range(1, east_m.size):
        d_east = float(east_m[kf] - east_m[kf - 1])
        d_north = float(north_m[kf] - north_m[kf - 1])
        step_m = math.hypot(d_east, d_north)

        delta_yaw_cw_rad = 0.0
        sigma_yaw_rad = slow_sigma_rad
        if forward_world_cw_rad is not None:
            heading_rad = forward_world_cw_rad[kf]
            delta_yaw_cw_rad = float(geo.wrap_rad(
                heading_rad - forward_world_cw_rad[kf - 1]))
            if step_m >= displacement_gate_m:
                sin_heading = math.sin(heading_rad)
                cos_heading = math.cos(heading_rad)
                forward_m = d_east * sin_heading + d_north * cos_heading
                left_m = -d_east * cos_heading + d_north * sin_heading
                sigma_m = sigma_pair_m
                sigma_yaw_rad = drift_sigma_rad
            else:
                forward_m = left_m = 0.0
                sigma_m = stationary_sigma_m
        elif step_m >= displacement_gate_m:
            course_rad = math.atan2(d_east, d_north)
            if kf in reverse:
                # A reverse chord points aft; rotate it to the platform's
                # nominal-forward proxy before differencing yaw.
                course_rad = float(geo.wrap_rad(course_rad + math.pi))
            if prev_course_rad is not None:
                delta_yaw_cw_rad = float(geo.wrap_rad(course_rad - prev_course_rad))
                sigma_yaw_rad = drift_sigma_rad
            prev_course_rad = course_rad

            forward_m = -step_m if kf in reverse else step_m
            left_m = 0.0
            sigma_m = sigma_pair_m
        else:
            forward_m = 0.0
            left_m = 0.0
            sigma_m = stationary_sigma_m
        if inject:
            travel_m = (math.hypot(forward_m, left_m)
                        if forward_world_cw_rad is not None
                        else abs(forward_m))
            sqrt_travel = math.sqrt(travel_m)
            translation_noise = imu_translation_noise_frac * sqrt_travel
            yaw_noise = imu_yaw_noise_frac * sqrt_travel
            forward_m += float(rng.normal(0.0, translation_noise)) \
                if translation_noise else 0.0
            if translation_noise and forward_world_cw_rad is not None:
                left_m += float(rng.normal(0.0, translation_noise))
            delta_yaw_cw_rad = float(geo.wrap_rad(
                delta_yaw_cw_rad + rng.normal(0.0, yaw_noise))) \
                if yaw_noise else delta_yaw_cw_rad
            sigma_m = math.hypot(sigma_m, translation_noise)
            sigma_yaw_rad = math.hypot(sigma_yaw_rad, yaw_noise)

        increments.append(structs.OdometryDelta(
            keyframe_idx=kf,
            forward_m=forward_m,
            left_m=left_m,
            delta_yaw_cw_rad=delta_yaw_cw_rad,
            sigma_m=sigma_m,
            sigma_yaw_rad=sigma_yaw_rad))
    return increments
