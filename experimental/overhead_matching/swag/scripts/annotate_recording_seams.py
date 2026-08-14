"""Record where a stitched dataset's odometry must not be trusted.

These trajectories are assembled from several Mapillary sequences, so the step
between two adjacent frames is sometimes not a step at all: it spans a recording
restart, during which the vessel kept moving and turning while nothing was
logged. Design doc §5.2 requires `dyaw_rad` on every OdometryDelta and allows a
producer to emit 0 with an inflated sigma when it cannot measure one -- but the
producer can only do that if it knows which steps those are, and nothing in the
dataset currently says.

Two kinds of break get marked, because they need the same treatment and arrive
by different routes:

  sequence  the frames either side come from different Mapillary sequences
  gap       the frames are far apart in time relative to the dataset's own
            cadence, whatever their sequence

A seam is *not* automatically bad. A ferry sitting at its berth for three
minutes between recordings has a break in time but continuous pose, and the
position gap shows that: metres, not hundreds of metres. Both numbers are
written so the consumer can decide, rather than a boolean that has already
decided. `implied_speed_mps` is the tell -- a value far above what the vessel can
do means the two frames are not a motion step at all.

    bazel run //experimental/overhead_matching/swag/scripts:annotate_recording_seams -- \
        --dataset_path /data/farfield_matching/mapillary_datasets/nyc_east_river
"""

import argparse
import csv
import json
import math
from pathlib import Path

R_EARTH_M = 6371000.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * math.asin(math.sqrt(a))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def find_seams(gps_rows, log_rows, gap_multiple, min_gap_s):
    lat = [float(r["latitude"]) for r in gps_rows]
    lon = [float(r["longitude"]) for r in gps_rows]
    t = [float(r["sensor_elapsed_s"]) for r in gps_rows]
    seq = [r["sequence_id"] for r in log_rows]
    dts = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    typical_dt = max(median(dts), 1e-3)

    seams = []
    for i in range(len(t) - 1):
        dt = dts[i]
        step = haversine_m(lat[i], lon[i], lat[i + 1], lon[i + 1])
        is_seq = seq[i] != seq[i + 1]
        is_gap = dt > max(gap_multiple * typical_dt, min_gap_s)
        if not (is_seq or is_gap):
            continue
        seams.append({
            "after_idx": i,
            "kind": "sequence" if is_seq else "gap",
            "dt_s": round(dt, 1),
            "step_m": round(step, 1),
            "implied_speed_mps": round(step / dt, 2) if dt > 0 else None,
            "from_sequence": seq[i],
            "to_sequence": seq[i + 1],
        })
    return seams, typical_dt


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", nargs="+", required=True, type=Path)
    p.add_argument("--gap_multiple", type=float, default=10.0,
                   help="a step this many times the median dt counts as a gap")
    p.add_argument("--min_gap_s", type=float, default=20.0)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    for ds in args.dataset_path:
        if not (ds / "frames_gps.csv").exists():
            continue
        gps_rows = list(csv.DictReader(open(ds / "frames_gps.csv")))
        log_rows = list(csv.DictReader(open(ds / "extraction_log.csv")))
        if len(gps_rows) != len(log_rows):
            print(f"{ds.name}: row-count mismatch, skipping")
            continue
        seams, typical_dt = find_seams(gps_rows, log_rows, args.gap_multiple,
                                       args.min_gap_s)
        worst = max((s["implied_speed_mps"] or 0.0) for s in seams) if seams else 0.0
        print(f"{ds.name:<24} {len(seams):>3} seam(s)  median dt {typical_dt:.1f}s  "
              f"worst implied speed {worst:.1f} m/s")
        for s in seams[:6]:
            print(f"    after idx {s['after_idx']:>5} {s['kind']:<8} "
                  f"dt={s['dt_s']:>7.1f}s step={s['step_m']:>7.1f}m "
                  f"implied={s['implied_speed_mps']} m/s")
        if len(seams) > 6:
            print(f"    ... {len(seams) - 6} more")
        if args.dry_run:
            continue
        meta_path = ds / "pipeline_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["recording_seams"] = {
            "median_dt_s": round(typical_dt, 3),
            "gap_multiple": args.gap_multiple,
            "min_gap_s": args.min_gap_s,
            "n_seams": len(seams),
            "seams": seams,
            "note": ("Steps listed here span a recording restart or a logging "
                     "gap. Treat each as a break in odometry continuity: emit "
                     "dyaw_rad=0 with an inflated sigma rather than a measured "
                     "increment (design doc 5.2). A small step_m means the "
                     "platform did not move during the break and pose is "
                     "probably continuous anyway."),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
