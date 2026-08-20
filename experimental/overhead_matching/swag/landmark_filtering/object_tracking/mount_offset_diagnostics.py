"""Why is a mount-offset sweep flat? Per-tracklet geometry behind the curve.

`mount_offset_sweep` reports the shape of the residual curve and refuses to
publish an offset when that shape is not trustworthy. It does not say *why*, and
the three failure verdicts have completely different repairs:

  FLAT            the tracklets cannot resolve the offset -- a geometry problem
  MULTIMODAL      the bearings or the poses disagree -- a data problem
  UNDER-SUPPORTED there are not enough tracklets -- a tracking problem

This module measures the quantity that separates them: **how much does one
tracklet's triangulation residual care about the offset?** Rotating every bearing
by `delta` moves the rays; whether that breaks their intersection depends on the
observation geometry, not on the offset. Two rays always intersect, and rays that
are nearly parallel -- a distant object seen from a short stretch of track --
still nearly intersect after rotation. So the residual's response to `delta` is
governed by the **observation arc**: how far the bearing to the object swept
while it was tracked.

The arc is set by the baseline and the range, arc ~ baseline_across / range. A
harbour buoy 400 m off, passed over a 600 m run, sweeps tens of degrees and
pins the offset hard. A tower 4 km down a river reach, seen over 300 m of
straight track, sweeps 4 deg and says almost nothing -- and *that* is a flat
curve, no matter how many tracklets there are or how clean the bearings are.

`sensitivity_deg_per_deg` is the measured version: the mean absolute change in
that tracklet's residual per degree of offset error, over +-20 deg. A tracklet
below ~0.05 is nearly blind to the offset; the median over tracklets predicts
the sweep's contrast better than any count does.

    bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:mount_offset_diagnostics -- \
        --run_dir <runs>/r001_v4
"""

import argparse
import json
import math
import statistics
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    bearing_matcher as bm,
    mount_offset_sweep as mos,
)

# Below this, a tracklet's residual barely moves when the offset is rotated, so
# it contributes support without contributing information.
BLIND_SENSITIVITY = 0.05

# Offsets either side of the candidate over which sensitivity is measured. Wide
# enough to leave the numerical floor, narrow enough to stay in one basin.
PROBE_DEG = 20.0


def span_deg(angles):
    """Angular spread of a set of bearings, smallest arc containing them."""
    if len(angles) < 2:
        return 0.0
    ordered = sorted(a % 360.0 for a in angles)
    gaps = [(b - a) for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 360.0 - ordered[-1])
    return 360.0 - max(gaps)


def baseline_m(observations):
    """Longest distance between any two observation positions."""
    best = 0.0
    for i, (e1, n1, *_) in enumerate(observations):
        for e2, n2, *_ in observations[i + 1:]:
            best = max(best, math.hypot(e2 - e1, n2 - n1))
    return best


def rays_at(observations, offset_deg):
    return [
        bm.Observation(anchor_keyframe_idx=kf, east_m=east, north_m=north,
                       bearing_world_deg=(course + camera - offset_deg) % 360.0,
                       bearing_camera_deg=camera, course_deg=course)
        for east, north, camera, course, kf in observations
    ]


def describe(tracklet_id, observations, offset_deg, probe=PROBE_DEG):
    """Geometry and offset-sensitivity for one tracklet. None if it cannot
    triangulate at the candidate offset."""
    result = bm.triangulate(rays_at(observations, offset_deg))
    if result is None:
        return None
    east, north, residual, condition = result

    ranges = [math.hypot(east - e, north - n) for e, n, *_ in observations]
    arc = span_deg([r.bearing_world_deg
                    for r in rays_at(observations, offset_deg)])

    # Mean |d residual| per degree, probed either side so an asymmetric basin
    # does not read as flat.
    slopes = []
    for delta in (-probe, probe):
        probed = bm.triangulate(rays_at(observations, offset_deg + delta))
        if probed is not None:
            slopes.append(abs(probed[2] - residual) / probe)
    return {
        "tracklet_id": tracklet_id,
        "n_observations": len(observations),
        "baseline_m": round(baseline_m(observations), 1),
        "median_range_m": round(statistics.median(ranges), 1),
        "arc_deg": round(arc, 2),
        "residual_deg": round(residual, 3),
        "condition": round(condition, 1),
        "sensitivity_deg_per_deg": round(
            statistics.mean(slopes), 4) if slopes else None,
    }


def summarise(rows, max_condition):
    """Aggregate the per-tracklet table into the numbers that explain a curve."""
    passing = [r for r in rows if r["condition"] <= max_condition]
    sens = [r["sensitivity_deg_per_deg"] for r in passing
            if r["sensitivity_deg_per_deg"] is not None]
    blind = [s for s in sens if s < BLIND_SENSITIVITY]
    return {
        "n_tracklets": len(rows),
        "n_well_conditioned": len(passing),
        "median_arc_deg": round(statistics.median(
            [r["arc_deg"] for r in passing]), 2) if passing else None,
        "median_baseline_m": round(statistics.median(
            [r["baseline_m"] for r in passing]), 1) if passing else None,
        "median_range_m": round(statistics.median(
            [r["median_range_m"] for r in passing]), 1) if passing else None,
        "median_sensitivity": round(statistics.median(sens), 4) if sens else None,
        "total_sensitivity": round(sum(sens), 3) if sens else None,
        "n_blind": len(blind),
        "frac_blind": round(len(blind) / len(sens), 3) if sens else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--offset_deg", type=float, default=None,
                        help="Candidate to probe around; defaults to the "
                             "offset in mount_offset_sweep.json")
    parser.add_argument("--min_observations", type=int, default=4)
    parser.add_argument("--max_condition", type=float, default=500.0)
    parser.add_argument("--top", type=int, default=15,
                        help="Print this many tracklets, most informative first")
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()
    paths = farfield_paths.resolve(parser, args, infer_from=args.run_dir,
                                   require=("dataset_base",))

    offset = args.offset_deg
    if offset is None:
        sweep_path = args.run_dir / "mount_offset_sweep.json"
        if not sweep_path.exists():
            raise SystemExit(f"no {sweep_path}; pass --offset_deg")
        offset = json.loads(sweep_path.read_text())["mount_offset_deg"]
        print(f"probing around the sweep's {offset} deg")

    by_tracklet = mos.load_tracklets(args.run_dir, paths, args.min_observations)
    rows = [r for r in (describe(t, obs, offset)
                        for t, obs in sorted(by_tracklet.items())) if r]
    summary = summarise(rows, args.max_condition)

    print(f"\n{'tracklet':16s} {'n':>3s} {'base_m':>8s} {'range_m':>9s} "
          f"{'arc':>7s} {'resid':>7s} {'cond':>8s} {'d|r|/ddeg':>10s}")
    ranked = sorted(rows, key=lambda r: -(r["sensitivity_deg_per_deg"] or 0))
    for r in ranked[:args.top]:
        flag = "" if r["condition"] <= args.max_condition else "  (ill-cond)"
        print(f"{r['tracklet_id']:16s} {r['n_observations']:3d} "
              f"{r['baseline_m']:8.1f} {r['median_range_m']:9.1f} "
              f"{r['arc_deg']:7.2f} {r['residual_deg']:7.3f} "
              f"{r['condition']:8.1f} "
              f"{r['sensitivity_deg_per_deg'] or 0:10.4f}{flag}")
    if len(ranked) > args.top:
        print(f"  ... {len(ranked) - args.top} more")

    print("\nsummary")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(f"\n  {summary['n_blind']} of {summary['n_well_conditioned']} "
          f"well-conditioned tracklets are effectively blind to the offset "
          f"(sensitivity < {BLIND_SENSITIVITY}).")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"dataset": paths.dataset, "run": args.run_dir.name,
             "offset_deg": offset, "summary": summary, "tracklets": rows},
            indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
