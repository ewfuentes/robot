"""Trajectory and evidence statistics for a localization export.

**This module reports numbers. It does not predict whether a leg will localize,
because nothing measured here does.** Two candidate predictors were fitted to
these seven datasets and both were refuted by the next dataset to finish; the
record is kept below so a third is not proposed from the same evidence.

Measured, single seed, uniform prior over the catalog's own extent, which is the
hardest setting there is:

    dataset               meas  landmarks  density  course span  net/path   final
    charles_river          561      30370   0.0185         313      0.10     25 m
    boston leg3            764      13210   0.0578         240      0.85     34 m
    mount_washington l3    347       4237   0.0819         329      0.49    190 m
    mount_washington l2    232       4237   0.0548         116      0.91    304 m
    boston leg1            437      13210   0.0331         100      0.93   6504 m
    mount_washington l1     39       4237   0.0092         131      0.86   2074 m
    boston leg2            214      13210   0.0162         179      0.83  11651 m

**Refuted #1: course span.** The argument was that a straight track lets a family
of poses slide along it, and the first successes turned through 240 and 329 deg
against failures at 100-179. mount_washington leg2 killed it: 116 deg, net/path
0.91, the straightest mountain leg, and it localizes to 304 m.

**Refuted #2: bearing density** (measurements / catalog landmarks). The argument
was better -- every catalog landmark is another way for bearings to be explained
by the wrong pose, so the bearings needed scale with candidate count -- and it
ordered all six datasets then measured, with a clean gap at 0.033|0.055, and
*correctly predicted mount_washington leg2 out of sample*. charles_river killed
it: density 0.0185, the second lowest of all seven, and it returns **25 m**, the
best result in the table.

Density was then refuted a second time, causally rather than by correlation.
Thinning boston leg3's measurements at random, leaving its trajectory and catalog
untouched:

    kept   measurements   density   leg3 result   the leg at that density
    100%            764    0.0578         34 m    --
     57%            437    0.0331         70 m    leg1: 6504 m
     28%            214    0.0162         36 m    leg2: 11651 m

leg3 at leg1's density is 93x better than leg1; at leg2's density, 320x better
than leg2. The numerator is not what separates them, and a leg that localizes
does so on a quarter of its bearings.

A disjunction of the two ("density >= 0.045 OR course span >= 240") does separate
all seven. It is also two thresholds fitted to seven points after seeing every
outcome, which is what overfitting looks like from the inside. Not implemented.

What the seven runs *do* say, and it is not a summary statistic: the failures are
distinguished by the **quality** of their evidence rather than its quantity.
charles pairs its 25 m with a median bearing residual of 0.25 deg, a null share of
0.01, and a >80% single-landmark claim on 387 of 561 measurements; boston leg1
fails at 6504 m with 4.31 deg, 0.31, and 127 of 437.

Those three cannot become a predictor even in principle, and it is worth being
clear why: they are **posteriors of the run itself**. The residual is measured
against the filter's own pose, the null share and the single-landmark claims come
out of its association posterior. A diverged filter reports small residuals
against the wrong landmarks quite happily. Nor do they order the outcomes anyway
-- mount_washington leg3 works on 5% strong claims and 19.3 deg of residual,
boston leg1 fails on 29% and 4.3 deg.

The quantities available *before* a run are the measurement count, the catalog
size, the trajectory shape, and the compatibility tables' strength. None of them
separate these seven.

Use this module to see a leg's shape and evidence before and after a change, and
to compare legs. To find out whether one localizes, run the filter.

    bazel run //experimental/overhead_matching/swag/bearing_only_localization:observability -- \\
        --export_dir <run>/localization_export_llm
"""
import argparse
import json
import math
from pathlib import Path

def angular_span_deg(angles_deg) -> float:
    """Smallest arc containing every angle. Wraps at north."""
    ordered = sorted(a % 360.0 for a in angles_deg)
    if len(ordered) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 360.0 - ordered[-1])
    return 360.0 - max(gaps)


def describe_trajectory(east, north, heading_deg) -> dict:
    """Descriptive only -- see the module docstring on course span."""
    path_m = sum(math.hypot(east[i + 1] - east[i], north[i + 1] - north[i])
                 for i in range(len(east) - 1))
    net_m = math.hypot(east[-1] - east[0], north[-1] - north[0])
    return {
        "n_poses": len(east),
        "path_m": round(path_m, 1),
        "net_over_path": round(net_m / path_m, 3) if path_m else None,
        "extent_m": round(math.hypot(max(east) - min(east),
                                     max(north) - min(north)), 1),
        "course_span_deg": round(angular_span_deg(heading_deg), 1),
    }


def bearing_density(n_measurements: int, n_landmarks: int) -> float:
    """Measurements per catalog candidate. The predictor."""
    return n_measurements / n_landmarks if n_landmarks else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse
                                     .RawDescriptionHelpFormatter)
    parser.add_argument("--export_dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=40,
                        help="keyframes per sliding window for local spans")
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    measurements_path = args.export_dir / "tier1_measurements.jsonl"
    landmarks_path = args.export_dir / "landmarks.json"
    truth_path = args.export_dir / "truth.jsonl"
    for path in (measurements_path, landmarks_path):
        if not path.exists():
            raise SystemExit(f"no {path}; this needs an export directory")

    n_measurements = sum(1 for line in
                         measurements_path.read_text().splitlines()
                         if line.strip())
    landmarks = json.loads(landmarks_path.read_text())
    n_landmarks = len(landmarks)
    density = bearing_density(n_measurements, n_landmarks)

    print(f"export: {args.export_dir}")
    print(f"  measurements       {n_measurements}")
    print(f"  catalog landmarks  {n_landmarks}")
    print(f"  bearing_density    {density:.4f}")

    record = {"n_measurements": n_measurements, "n_landmarks": n_landmarks,
              "bearing_density": round(density, 5)}

    if truth_path.exists():
        poses = [json.loads(line)
                 for line in truth_path.read_text().splitlines() if line.strip()]
        if len(poses) >= 2:
            summary = describe_trajectory([p["east_m"] for p in poses],
                                          [p["north_m"] for p in poses],
                                          [p["heading_deg"] for p in poses])
            print("  --- trajectory (descriptive; does NOT predict success) ---")
            for key, value in summary.items():
                print(f"  {key:18s} {value}")
            # How much of the searched area the vehicle actually visited.
            # Reported because it is the first thing anyone reaches for -- and it
            # does NOT predict: mount_washington leg2 has the smallest ratio of
            # all seven (0.018, 0.8 km of travel in a 29 km catalog) and
            # localizes to 304 m, while charles_river at 0.037 is the one
            # predicted to fail. Only bearing_density separates them.
            lat = [lm["lat_deg"] for lm in landmarks]
            lon = [lm["lon_deg"] for lm in landmarks]
            mid = math.radians(sum(lat) / len(lat))
            catalog_extent_m = math.hypot(
                (max(lon) - min(lon)) * 111_320.0 * math.cos(mid),
                (max(lat) - min(lat)) * 110_540.0)
            visited = (summary["extent_m"] / catalog_extent_m
                       if catalog_extent_m else None)
            print(f"  {'catalog_extent_m':18s} {catalog_extent_m:.0f}")
            print(f"  {'extent_ratio':18s} {visited:.3f}"
                  f"  (trajectory extent / catalog extent)")
            record.update(summary)
            record["catalog_extent_m"] = round(catalog_extent_m, 1)
            record["extent_ratio"] = round(visited, 4) if visited else None

    print("\n  No verdict: see the module docstring. Two candidate predictors "
          "were\n  fitted to these datasets and both were refuted by the next "
          "one to finish.\n  Run the filter to find out.")
    if args.json_out:
        args.json_out.write_text(json.dumps(record, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
