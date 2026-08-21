#!/usr/bin/env python3
"""GPS-consistency and image-density QC for candidate trajectories.

    bazel run //experimental/overhead_matching/swag/farfield/collection:qc_candidates -- --seeds 957814683411464,548801286115301
    bazel run //experimental/overhead_matching/swag/farfield/collection:qc_candidates -- --local <farfield_root>/datasets/seattle
    bazel run //experimental/overhead_matching/swag/farfield/collection:qc_candidates -- --seeds ... --output qc.json

Sits between discovery (`discover_tracks.py`, tile geometry only) and
collection (`seed_to_trajectory.py`, downloads everything). One metadata fetch
per seed answers the two questions the tiles cannot:

  * **Is the motion usable?** A track whose direction of travel oscillates —
    GPS marching forward/backward, heading flapping — cannot be worked with:
    mount-offset calibration pairs frames by metres travelled, and the odometry
    producer derives course from position differences. Big *occasional* jumps
    are fine (recording gaps; the stitcher's seam logic handles them), so jumps
    are detected first and excluded from every direction statistic rather than
    letting them poison the answer.
  * **Is it dense enough to track?** Mapillary datasets have no source video,
    so SAM2 propagates between the frames themselves; the collected sets were
    built at --min_spacing_m 5. A raw capture with 40 m between frames cannot
    be densified after the fact.

Direction statistics use only "moving" steps (>= JITTER_FLOOR_M): below GPS
noise a bearing is meaningless, and counting jitter as backtracking would fail
every red light. The same reasoning as seed_to_trajectory's 0.5 m `moved`
floor, but stricter because we take a *direction*, not just a length.

`--local` computes the identical metrics from a collected dataset's
frames_gps.csv, so thresholds are calibrated against tracks with known
verdicts (seattle good, harima_b_pano rejected for noise) instead of invented.
Density is NOT comparable in local mode — collected sets are already
subsampled to 5 m spacing — so only consistency verdicts apply there.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry

# --------------------------------------------------------------------------
# thresholds on bare (lat, lng) tuples so the same code serves API images and
# local CSV rows; the local-metres conversion is geometry.enu_from_latlon (the
# one ENU, REORG.md rule 1)
# --------------------------------------------------------------------------

JITTER_FLOOR_M = 3.0      # below this a step has no meaningful direction
JUMP_FACTOR = 5.0         # step > max(150 m, factor x median) = discontinuity
JUMP_FLOOR_M = 150.0
WINDOW_STEPS = 12         # progress-ratio window, in moving steps


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[k]


def motion_consistency(points: list[tuple], times_ms: list = None) -> dict:
    """Direction and density statistics over an ordered (lat, lng) track.

    Returns a dict of metrics; see VERDICT_GATES for how they are judged.
    """
    lat0, lng0 = points[0]
    xy = [geometry.enu_from_latlon(lat, lng, lat0, lng0) for lat, lng in points]

    # raw steps
    steps = []
    for i in range(len(xy) - 1):
        dx, dy = xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]
        steps.append((math.hypot(dx, dy), dx, dy))
    lens = sorted(s[0] for s in steps)
    med_step = _percentile(lens, 0.5)

    # classify: jump (recording gap - allowed, excluded), jitter (no
    # direction), moving (usable for direction statistics)
    jump_cut = max(JUMP_FLOOR_M, JUMP_FACTOR * max(med_step, 1.0))
    n_jumps = sum(1 for s in steps if s[0] > jump_cut)

    # runs of consecutive non-jump steps; direction stats never cross a jump
    runs, run = [], []
    for s in steps:
        if s[0] > jump_cut:
            if run:
                runs.append(run)
            run = []
        else:
            run.append(s)
    if run:
        runs.append(run)

    turns = []            # |turn angle| between consecutive moving steps
    progress = []         # windowed net displacement / path length
    for run in runs:
        moving = [s for s in run if s[0] >= JITTER_FLOOR_M]
        for a, b in zip(moving, moving[1:]):
            dot = a[1] * b[1] + a[2] * b[2]
            cross = a[1] * b[2] - a[2] * b[1]
            turns.append(abs(math.degrees(math.atan2(cross, dot))))
        for i in range(0, max(1, len(moving) - WINDOW_STEPS + 1),
                       max(1, WINDOW_STEPS // 2)):
            w = moving[i:i + WINDOW_STEPS]
            if len(w) < WINDOW_STEPS // 2:
                continue
            path = sum(s[0] for s in w)
            net = math.hypot(sum(s[1] for s in w), sum(s[2] for s in w))
            if path > 0:
                progress.append(net / path)

    turns_sorted = sorted(turns)
    progress_sorted = sorted(progress)
    length_km = sum(s[0] for s in steps) / 1000.0

    out = {
        "n_points": len(points),
        "length_km": round(length_km, 2),
        "median_spacing_m": round(med_step, 1),
        "p90_spacing_m": round(_percentile(lens, 0.9), 1),
        "img_per_km": round((len(points) - 1) / length_km, 1) if length_km else 0.0,
        "n_jumps": n_jumps,
        "jump_cut_m": round(jump_cut, 0),
        "moving_frac": round(sum(1 for s in steps if
                                 JITTER_FLOOR_M <= s[0] <= jump_cut)
                             / max(1, len(steps)), 3),
        "backtrack_frac": round(sum(1 for t in turns if t > 120.0)
                                / max(1, len(turns)), 4),
        "turn_median_deg": round(_percentile(turns_sorted, 0.5), 1),
        "turn_p90_deg": round(_percentile(turns_sorted, 0.9), 1),
        "progress_median": round(_percentile(progress_sorted, 0.5), 3),
        "progress_p10": round(_percentile(progress_sorted, 0.1), 3),
        "n_turn_pairs": len(turns),
    }

    if times_ms and len([t for t in times_ms if t]) > 1:
        ts = [t for t in times_ms if t]
        dur = (max(ts) - min(ts)) / 1000.0
        out["duration_s"] = round(dur, 0)
        out["mean_speed_mps"] = round(length_km * 1000 / dur, 1) if dur else None
    return out


# --------------------------------------------------------------------------
# verdicts — thresholds calibrated on collected datasets with known outcomes
# (see the table in the module run log / Sightline Scout artifact):
# seattle / mississippi_rural / nyc_east_river pass cleanly; harima_b_pano
# (rejected 2026-08-13 for GPS noise) and kurashiki (worst kept) must land in
# WARN/FAIL territory on consistency.
# --------------------------------------------------------------------------

def judge(m: dict, local: bool = False) -> tuple[str, list]:
    reasons = []
    # direction consistency: oscillation, not occasional jumps
    if m["backtrack_frac"] > 0.15:
        reasons.append(f"backtrack {m['backtrack_frac']:.0%}")
    if m["progress_p10"] < 0.55 and m["n_turn_pairs"] > 20:
        reasons.append(f"progress_p10 {m['progress_p10']:.2f}")
    if m["turn_p90_deg"] > 90:
        reasons.append(f"turn_p90 {m['turn_p90_deg']:.0f} deg")
    hard = len(reasons)

    warn = []
    if 0.05 < m["backtrack_frac"] <= 0.15:
        warn.append(f"backtrack {m['backtrack_frac']:.0%}")
    if 0.55 <= m["progress_p10"] < 0.75:
        warn.append(f"progress_p10 {m['progress_p10']:.2f}")
    # density gates only mean something on raw API sequences
    if not local:
        if m["median_spacing_m"] > 20:
            reasons.append(f"sparse {m['median_spacing_m']:.0f} m/frame")
            hard += 1
        elif m["median_spacing_m"] > 10:
            warn.append(f"spacing {m['median_spacing_m']:.0f} m")

    if hard:
        return "FAIL", reasons + warn
    if warn:
        return "WARN", warn
    return "PASS", []


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def qc_seed(client, pkey: str) -> dict:
    detail = client.get_image_detail(pkey)
    seq_id = detail.get("sequence")
    if not seq_id:
        return {"seed": pkey, "error": "no sequence on image"}
    seq = client.get_full_sequence(seq_id)
    if not seq.images:
        return {"seed": pkey, "sequence": seq_id, "error": "empty sequence"}
    pts = [(i.lat, i.lng) for i in seq.images]
    ts = [i.captured_at for i in seq.images]
    m = motion_consistency(pts, ts)
    m["seed"] = pkey
    m["sequence"] = seq_id
    m["is_pano"] = seq.images[0].is_equirectangular
    m["geometry_source"] = seq.images[0].geometry_source
    verdict, why = judge(m)
    m["verdict"], m["reasons"] = verdict, why
    return m


def qc_local(dataset_dir: Path) -> dict:
    rows = list(csv.DictReader(open(dataset_dir / "frames_gps.csv")))
    pts = [(float(r["latitude"]), float(r["longitude"])) for r in rows]
    ts = [int(float(r["sensor_elapsed_s"]) * 1000) for r in rows
          if r.get("sensor_elapsed_s")]
    m = motion_consistency(pts, ts if len(ts) == len(pts) else None)
    m["seed"] = dataset_dir.name
    verdict, why = judge(m, local=True)
    m["verdict"], m["reasons"] = verdict, why
    return m


HEADER = (f"{'verdict':7s} {'km':>6} {'img/km':>7} {'med_m':>6} "
          f"{'backtr':>7} {'t_p90':>6} {'prog10':>7} {'jumps':>5}  name/seed")


def print_row(m: dict):
    if "error" in m:
        print(f"{'ERROR':7s} {'':>6} {'':>7} {'':>6} {'':>7} {'':>6} {'':>7} "
              f"{'':>5}  {m['seed']}  ({m['error']})")
        return
    print(f"{m['verdict']:7s} {m['length_km']:6.1f} {m['img_per_km']:7.1f} "
          f"{m['median_spacing_m']:6.1f} {m['backtrack_frac']:7.1%} "
          f"{m['turn_p90_deg']:6.0f} {m['progress_p10']:7.2f} "
          f"{m['n_jumps']:5d}  {m['seed']}"
          + (f"  [{', '.join(m['reasons'])}]" if m["reasons"] else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seeds", help="comma-separated seed pKeys")
    src.add_argument("--local", nargs="+", type=Path,
                     help="collected dataset dirs (frames_gps.csv) for calibration")
    p.add_argument("--output", type=Path, help="write full metrics JSON here")
    args = p.parse_args()

    results = []
    print(HEADER)
    if args.local:
        for d in args.local:
            m = qc_local(d)
            results.append(m)
            print_row(m)
    else:
        from experimental.overhead_matching.swag.farfield.collection.api import MapillaryClient
        client = MapillaryClient()
        for pkey in args.seeds.split(","):
            pkey = pkey.strip()
            if not pkey:
                continue
            try:
                m = qc_seed(client, pkey)
            except Exception as e:  # one bad seed must not kill the batch
                m = {"seed": pkey, "error": str(e)[:120]}
            results.append(m)
            print_row(m)

    if args.output:
        from experimental.overhead_matching.swag.farfield.collection.provenance_util import (
            provenance_record,
        )
        payload = {
            "provenance": provenance_record(
                generator="//experimental/overhead_matching/swag/farfield/"
                          "collection:qc_candidates",
                inputs={"seeds": args.seeds or "",
                        "local": ",".join(str(d) for d in (args.local or []))},
                config={"jitter_floor_m": JITTER_FLOOR_M,
                        "jump_factor": JUMP_FACTOR,
                        "jump_floor_m": JUMP_FLOOR_M,
                        "window_steps": WINDOW_STEPS}),
            "results": results,
        }
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
