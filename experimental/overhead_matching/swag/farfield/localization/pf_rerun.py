"""Rerun the particle filter on a localization_inputs export with the
odometry yaw sigma corrected (scratch A/B against the grid filter).

The export's per-step sigma_yaw_rad (~26 deg/kf) is an honest PER-STEP
course-difference sigma, but consecutive differenced-course increments are
anti-correlated, so the integrated heading error is bounded (~one chord
sigma) rather than a 26 deg * sqrt(n) random walk. The filter models the
steps as independent, which overstates heading drift enormously. This
driver scales sigma_yaw_rad (delta_yaw itself untouched) and runs the PF
otherwise byte-identically to a recorded run's filter_config.

  bazel run //experimental/overhead_matching/swag/farfield/localization:pf_rerun -- \
    --input_dir <localization_inputs dir> --like_run <run dir> \
    --yaw_sigma_scale 0.0 --seed 0
"""

import argparse
import json
from pathlib import Path

import msgspec

from common.python.serialization import msgspec_dec_hook
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter as pf,
    metrics,
    runner,
    structs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--like_run", required=True,
                        help="run dir whose run_manifest.json filter_config "
                             "is reused verbatim (seed aside)")
    parser.add_argument("--yaw_sigma_scale", type=float, default=0.0)
    parser.add_argument("--inject_yaw_sigma_deg", type=float, default=0.0,
                        help="IMU emulation: add independent per-step noise "
                             "of this sigma to delta_yaw values and fold it "
                             "into the emitted sigma (deterministic seed)")
    parser.add_argument("--sigma_m_override", type=float, default=None,
                        help="replace each delta's sigma_m (emulates the "
                             "translation drift-budget export scheme)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--box_from_export", action="store_true",
                        help="replace the template's init box with this "
                             "export's whole-catalog region box (for "
                             "cross-dataset use of one config template)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = export_ingest.load(Path(args.input_dir))
    manifest = json.loads(
        (Path(args.like_run) / "run_manifest.json").read_text())
    config = msgspec.json.decode(
        json.dumps(manifest["filter_config"]).encode(),
        type=structs.FilterConfig, dec_hook=msgspec_dec_hook)
    if args.seed is not None:
        config = msgspec.structs.replace(config, seed=args.seed)
    if args.box_from_export:
        config = msgspec.structs.replace(
            config, init=export_ingest.region_box(data, 0.0))

    import math

    import numpy as np
    inject_rad = math.radians(args.inject_yaw_sigma_deg)
    noise_rng = np.random.default_rng(0)
    odometry = [
        structs.OdometryDelta(
            keyframe_idx=item.keyframe_idx,
            forward_m=item.forward_m,
            left_m=item.left_m,
            delta_yaw_cw_rad=item.delta_yaw_cw_rad
            + (float(noise_rng.normal(0.0, inject_rad)) if inject_rad else 0.0),
            sigma_m=(args.sigma_m_override if args.sigma_m_override is not None
                     else item.sigma_m),
            sigma_yaw_rad=math.hypot(
                max(item.sigma_yaw_rad * args.yaw_sigma_scale, 1e-6),
                inject_rad))
        for item in data.odometry
    ]

    metric_config = metrics.position_mass_metric_config(
        (50.0, 100.0, 250.0, 500.0, 1000.0))
    recorder = runner.PositionMassRecorder(data.truth, metric_config)
    history = pf.run_filter(
        config, data.catalog, odometry, data.measurements, data.tables,
        observer=recorder)
    for record in history.health:
        record.position_probability_mass = recorder.by_keyframe[
            record.keyframe_idx]
    summary = metrics.position_mass_summary(history.health, metric_config)
    print(f"seed={config.seed} yaw_sigma_scale={args.yaw_sigma_scale}")
    print(metrics.describe_position_mass_summary(summary, "diagnostic"))
    if args.out:
        series = {
            f"{radius:g}": [
                recorder.by_keyframe[kf][
                    metrics.position_mass_metric_key(metric_config, radius)]
                for kf in sorted(recorder.by_keyframe)
            ]
            for radius in metric_config.radii_m
        }
        Path(args.out).write_text(json.dumps({
            "seed": config.seed,
            "yaw_sigma_scale": args.yaw_sigma_scale,
            "like_run": args.like_run,
            "summary": summary,
            "mass_by_keyframe": series,
        }, indent=1))


if __name__ == "__main__":
    main()
