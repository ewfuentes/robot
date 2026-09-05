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

  THE OBJECTIVE OF DIFFERENCING IS TO APPROXIMATE AN IMU: the deployed
  platform integrates inertial increments whose noise is genuinely
  independent per step, while raw GPS-differenced deltas carry the
  anti-correlated (telescoping, bounded) noise above — strictly friendlier
  data than an IMU would produce. To make the emitted delta stream an
  honest IMU surrogate, every production build INJECTS independent
  per-step Gaussian noise into the delta values and inflates the emitted
  sigmas to match, so the declared uncertainty describes noise that is
  actually in the data. The noise is a WIENER PROCESS driven by distance
  travelled — per step, sigma = coefficient x sqrt(|forward|) — so the
  injected drift is self-consistent under any keyframe spacing and
  diffuses on straightaways the way a real gyro does (per-delta
  proportional noise, as in ReWAG, injects nothing on straight travel and
  composes inconsistently under step refinement). The coefficients keep
  ReWAG's values (Downes et al., arXiv:2308.07432: "We add 2% noise to
  the ground-truth odometry and 1% noise to the ground-truth heading at
  each time step"): `imu_translation_noise_frac` = 0.02 m/sqrt(m) and
  `imu_yaw_noise_frac` = 0.01 rad/sqrt(m) — at our ~3 m keyframes that is
  ~1.0 deg/keyframe of yaw diffusion, the same grade ReWAG's 1% implies on
  typical turns. The m/sqrt(m) convention matches the histogram filter's
  OdometryNoiseConfig.sigma_noise_frac. Stationary (gated) steps travel no
  distance and receive no injected diffusion; their heading hold is
  covered by slow_yaw_sigma_deg. The injected grade is a recorded
  build-config decision.

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
                      imu_translation_noise_frac: float = 0.0,
                      imu_yaw_noise_frac: float = 0.0,
                      noise_seed: int = 0) -> list:
    """ENU fixes (keyframes 0..N) -> OdometryDelta increments (1..N).

    Baseline values and reverse annotations are required keywords — callers
    pass immutable build-config values, so the recorded recipe shaped the
    odometry. The imu_* parameters inject independent per-step noise into
    the delta values AND declare it in the emitted sigmas (an honest
    producer emulating the deployed IMU, not a lying one); the pipeline
    config requires them positive so production exports are always IMU
    surrogates, while zero remains valid here for exact-geometry tests of
    the pure derivation.
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
        if step_m >= displacement_gate_m:
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
            sigma_m = sigma_pair_m
        else:
            forward_m = 0.0
            sigma_m = stationary_sigma_m
        left_m = 0.0
        if inject:
            sqrt_travel = math.sqrt(abs(forward_m))
            translation_noise = imu_translation_noise_frac * sqrt_travel
            yaw_noise = imu_yaw_noise_frac * sqrt_travel
            forward_m += float(rng.normal(0.0, translation_noise)) \
                if translation_noise else 0.0
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
