"""Derive body-frame dead-reckoning odometry from GPS fixes (§5.2).

The deployed system has no GPS; GPS exists only in data collection. This
producer turns a sequence of ENU fixes into the OdometryDelta increments the
filter consumes, the way §5.2 specifies:

  forward = fix-to-fix step length, left = 0. Rotating a displacement by its
  own direction is pure forward, so real leeway/crab is MISASSIGNED into
  forward+heading — the documented §5.2 derivation artifact, priced by the
  scenario crab knob, not hidden here.

  dyaw = differenced course. Course is the direction of each step; its noise
  is geometric — sigma_course ~ atan(sigma_pair / step) — so it is computed
  per step from the step length rather than declared as a constant. A step
  too short to carry a usable course (speed gate) emits dyaw = 0 with
  sigma_yaw_rad inflated to `slow_yaw_sigma_deg`, the §5.2 "no yaw signal"
  posture. When usable courses are separated by a gap, the catch-up dyaw
  spans the whole gap (its measurement noise still telescopes to the two
  endpoint course sigmas); the gapped steps in between already carried the
  inflated sigma.

  sigma_m is the honest per-fix-pair constant (~1 m: correlated absolute GPS
  error differences out). No IMU-style step scaling is cosplayed onto real
  data — emulating worse odometry is an explicit, labelled experiment via
  the --extra_* knobs, never a default.

Differenced-course dyaw is substantially noisier than a real gyro, which is
the safe direction for the paper's claims: convergence demonstrated on
course-grade yaw lower-bounds what an IMU would deliver.

As a CLI, rewrites a localization export (export_ingest.py layout) into a
sibling directory with regenerated tier1_odometry.jsonl and a bumped
schema_version, leaving the source export untouched:

  bazel run //experimental/overhead_matching/swag/bearing_only_localization:gps_to_odometry -- \
    --export_dir /path/to/localization_export_temp \
    --output_dir /path/to/localization_export_v02
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    structs,
)


def derive_increments(east_m, north_m, *,
                      sigma_pair_m: float = 1.0,
                      min_step_m: float = 2.0,
                      slow_yaw_sigma_deg: float = 30.0,
                      extra_sigma_m: float = 0.0,
                      extra_yaw_sigma_deg: float = 0.0,
                      noise_seed: int = 0) -> list:
    """ENU fixes (keyframes 0..N) -> OdometryDelta increments (1..N).

    The --extra_* knobs inject additional noise AND declare it (an honest
    producer emulating a worse sensor, not a lying one); they exist for
    drift-injection experiments and default to off.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    north_m = np.asarray(north_m, dtype=np.float64)
    if east_m.shape != north_m.shape or east_m.ndim != 1 or east_m.size < 2:
        raise ValueError("need matching 1-D east/north arrays of >= 2 fixes")
    if sigma_pair_m <= 0.0 or min_step_m <= 0.0:
        raise ValueError("sigma_pair_m and min_step_m must be positive")

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

        dyaw_rad = 0.0
        sigma_yaw_rad = slow_sigma_rad
        if step_m >= min_step_m:
            course_rad = math.atan2(d_east, d_north)
            course_sigma_rad = math.atan(sigma_pair_m / step_m)
            if prev_course_rad is not None:
                dyaw_rad = float(geodesy.wrap_rad(
                    course_rad - prev_course_rad))
                sigma_yaw_rad = math.hypot(course_sigma_rad,
                                           prev_course_sigma_rad)
            prev_course_rad = course_rad
            prev_course_sigma_rad = course_sigma_rad

        forward_m, left_m = step_m, 0.0
        sigma_m = sigma_pair_m
        if inject:
            forward_m += float(rng.normal(0.0, extra_sigma_m))
            left_m += float(rng.normal(0.0, extra_sigma_m))
            dyaw_rad = float(geodesy.wrap_rad(
                dyaw_rad + rng.normal(0.0, extra_yaw_rad)))
            sigma_m = math.hypot(sigma_m, extra_sigma_m)
            sigma_yaw_rad = math.hypot(sigma_yaw_rad, extra_yaw_rad)

        increments.append(structs.OdometryDelta(
            keyframe_idx=kf,
            forward_m=forward_m,
            left_m=left_m,
            dyaw_rad=dyaw_rad,
            sigma_m=sigma_m,
            sigma_yaw_rad=sigma_yaw_rad))
    return increments


def rewrite_export(export_dir: Path, output_dir: Path, **derive_kwargs) -> list:
    """Copy an export, regenerating its odometry from truth.jsonl fixes.

    The source export is left untouched (its odometry may be in an older
    schema this build no longer reads). truth.jsonl carries the GPS fixes in
    the export's ENU frame; headings in it are diagnostics and are not used.
    """
    export_dir, output_dir = Path(export_dir), Path(output_dir)
    if output_dir.resolve() == export_dir.resolve():
        raise ValueError("refusing to rewrite an export in place; give a "
                         "separate --output_dir")
    truth = [msgspec.json.decode(line, type=structs.TruthPose,
                                 dec_hook=msgspec_dec_hook)
             for line in (export_dir / "truth.jsonl").read_bytes().splitlines()
             if line.strip()]
    if len(truth) < 2:
        raise ValueError(f"{export_dir}/truth.jsonl has {len(truth)} poses; "
                         "cannot derive odometry")
    increments = derive_increments([t.east_m for t in truth],
                                   [t.north_m for t in truth],
                                   **derive_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("landmarks.json", "tier1_tables.json",
                 "tier1_measurements.jsonl", "truth.jsonl"):
        shutil.copy2(export_dir / name, output_dir / name)
    with open(output_dir / "tier1_odometry.jsonl", "wb") as f:
        for increment in increments:
            f.write(msgspec.json.encode(increment,
                                        enc_hook=msgspec_enc_hook))
            f.write(b"\n")
    # Raw-dict round trip so meta fields this build does not model survive.
    meta = json.loads((export_dir / "export_meta.json").read_text())
    meta["schema_version"] = structs.SCHEMA_VERSION
    (output_dir / "export_meta.json").write_text(json.dumps(meta, indent=1))
    return increments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sigma_pair_m", type=float, default=1.0)
    parser.add_argument("--min_step_m", type=float, default=2.0)
    parser.add_argument("--slow_yaw_sigma_deg", type=float, default=30.0)
    parser.add_argument("--extra_sigma_m", type=float, default=0.0,
                        help="drift-injection experiment: extra translation "
                             "noise, injected AND declared")
    parser.add_argument("--extra_yaw_sigma_deg", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    args = parser.parse_args()

    increments = rewrite_export(
        args.export_dir, args.output_dir,
        sigma_pair_m=args.sigma_pair_m, min_step_m=args.min_step_m,
        slow_yaw_sigma_deg=args.slow_yaw_sigma_deg,
        extra_sigma_m=args.extra_sigma_m,
        extra_yaw_sigma_deg=args.extra_yaw_sigma_deg,
        noise_seed=args.noise_seed)

    steps = np.array([i.forward_m for i in increments])
    gated = sum(1 for i in increments
                if i.dyaw_rad == 0.0
                and i.sigma_yaw_rad >= math.radians(args.slow_yaw_sigma_deg))
    yaw_sigmas = np.degrees([i.sigma_yaw_rad for i in increments])
    print(f"{len(increments)} increments -> {args.output_dir}")
    print(f"step: median {np.median(steps):.1f} m, max {steps.max():.1f} m")
    print(f"dyaw sigma: median {np.median(yaw_sigmas):.2f} deg; "
          f"{gated} slow/gapped steps at {args.slow_yaw_sigma_deg:.0f} deg")


if __name__ == "__main__":
    main()
