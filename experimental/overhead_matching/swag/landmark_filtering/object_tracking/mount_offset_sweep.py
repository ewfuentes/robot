"""Camera-to-body yaw offset by minimising triangulation residual.

**This is the reference method** for `mount_offset_deg` -- the angle that turns a
camera-frame bearing into a body-frame one
(`bearing_body = bearing_camera - offset`). It is what produced
boston_harbor_leg1's accepted 214 deg, and the runbook's Stage 6 described it as
a procedure to carry out by hand; this module is that procedure, so the number is
reproducible rather than remembered.

Why this and not the alternatives:

- `swag/scripts/calibrate_mount_offset.py` recovers the same angle from the focus
  of expansion, needing no tracks at all. On the one dataset with an external
  reference it lands 14 deg away *and* inverted, at an axis MAD of 2 deg -- a
  confidently wrong answer. It is a corroborator.
- `bow_calibration.py` measures the **bow**, which differs from the direction of
  travel by the crab angle. On a boat that is not a constant.

The idea: a wrong offset rotates every bearing by the same constant, and rotated
rays to a static object stop intersecting. So sweep the offset, triangulate each
tracklet's bearings at each candidate, and take the angle that minimises the
median residual. No map, and no assumed tracklet-to-landmark match, enter the
estimate -- which matters, because the map match is what the offset is needed for
downstream.

Three gates keep noise out of the minimum:

- **condition number** (`--max_condition`, default 500). Bearings taken over a
  short arc intersect at a glancing angle, so a tiny residual can coexist with a
  position uncertain by kilometres. Residual alone is not enough.
- **observation count** (`--min_observations`, default 4). Three is the bare
  minimum for a residual to exist at all; four makes it mean something.
- **support** (`--min_support_frac`, default 0.5). *This one is not optional*,
  and the reason is a trap in the obvious version of this procedure: the
  condition gate is applied per candidate offset, so how many tracklets survive
  it is itself a function of the offset. A badly wrong offset drops almost every
  tracklet, and the handful that remain can have an excellent median residual
  purely by selection. Run on leg1 without this gate, the curve's argmin is
  85 deg at 1.04 deg residual over **5** tracklets, while the true basin at
  210 deg sits at 1.05 deg over **37** -- a numerically better minimum that
  explains an eighth as much data. Candidates are therefore only eligible if
  they keep at least this fraction of the best-supported candidate's tracklets,
  which encodes the obvious principle: at equal residual, the offset that
  explains more tracklets wins.

Read the printed curve, not just the minimum. A genuine calibration is smooth and
unimodal with a clear contrast between the floor and the rest; a flat or
multimodal curve means the bearings or the poses are wrong, and its argmin is
noise. The summary says which of those it looks like, and `--write_metadata`
refuses to record a value when the curve does not qualify.

    bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:mount_offset_sweep -- \
        --run_dir /data/farfield_matching/artifacts/object_tracks/<ds>/v1/m3_tracks/runs/<run>
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    bearing_matcher as bm,
    heading as heading_mod,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

CONVENTION = (
    "mount_offset_deg is the azimuth, IN THE CAMERA FRAME, of the vehicle's "
    "DIRECTION OF TRAVEL - not the bow. Applied as bearing_body_deg = "
    "(bearing_camera_deg - mount_offset_deg) mod 360.")

# A minimum this shallow relative to the rest of the curve is not a minimum.
MIN_CONTRAST = 1.5


def hc_delta(a_deg, b_deg):
    """Signed smallest angular difference a - b, in (-180, 180]."""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def load_tracklets(run_dir: Path, paths, min_observations: int):
    """{tracklet_id: [(east_m, north_m, bearing_camera_deg, course_deg)]}.

    Bearings come from m6's fused `merged/measurements.json`, which stores the
    raw camera azimuth precisely because the offset has not been applied yet.
    Poses and course come from the same `ingest` + `heading` path the tracking
    stages use, so the sweep cannot disagree with the pipeline about where the
    boat was pointing.
    """
    measurements_path = run_dir / "merged" / "measurements.json"
    if not measurements_path.exists():
        raise SystemExit(
            f"{measurements_path} not found - run m6_merge_tracks first "
            f"(Stage 5); the sweep needs fused per-tracklet bearings.")
    measurements = json.loads(measurements_path.read_text())

    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                              IngestConfig())
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in result.frames], [f.y_m for f in result.frames],
        [f.time_s for f in result.frames])

    by_tracklet = defaultdict(list)
    for m in measurements:
        frame = frames_by_idx.get(m["anchor_keyframe_idx"])
        if frame is None:
            continue
        by_tracklet[m["tracklet_id"]].append(
            (frame.x_m, frame.y_m, m["bearing_camera_deg"],
             model.at(frame.time_s), m["anchor_keyframe_idx"]))

    kept = {t: obs for t, obs in by_tracklet.items()
            if len(obs) >= min_observations}
    print(f"{len(measurements)} measurements over {len(by_tracklet)} tracklets; "
          f"{len(kept)} have >= {min_observations} observations")
    return kept


def residual_at(offset_deg: float, by_tracklet, max_condition: float):
    """Median triangulation residual over well-conditioned tracklets."""
    residuals = []
    for observations in by_tracklet.values():
        rays = [
            bm.Observation(
                anchor_keyframe_idx=kf, east_m=east, north_m=north,
                bearing_world_deg=(course + camera - offset_deg) % 360.0,
                bearing_camera_deg=camera, course_deg=course)
            for east, north, camera, course, kf in observations
        ]
        result = bm.triangulate(rays)
        if result is None:
            continue
        _, _, residual, condition = result
        if condition > max_condition:
            continue
        residuals.append(residual)
    if not residuals:
        return None, 0
    residuals.sort()
    return residuals[len(residuals) // 2], len(residuals)


def sweep(by_tracklet, max_condition, start, stop, step):
    """[(offset_deg, median_residual_deg, n_tracklets)] over a grid."""
    curve = []
    offset = start
    while offset < stop - 1e-9:
        residual, n = residual_at(offset % 360.0, by_tracklet, max_condition)
        if residual is not None:
            curve.append((offset % 360.0, residual, n))
        offset += step
    return curve


def eligible(curve, min_support_frac):
    """Candidates that keep enough tracklets to be compared on residual.

    See the module docstring: the condition gate's survivor count varies with
    the offset, so an unrestricted argmin can be won by a candidate that
    discarded almost everything. Returns (eligible_curve, support_floor).
    """
    if not curve:
        return [], 0
    best_support = max(n for _, _, n in curve)
    floor = max(1, int(math.ceil(best_support * min_support_frac)))
    return [c for c in curve if c[2] >= floor], floor


def local_minima(curve, tolerance=1e-9):
    """Indices that are no worse than both neighbours (cyclic)."""
    found = []
    for i, (_, residual, _) in enumerate(curve):
        prev = curve[i - 1][1]
        nxt = curve[(i + 1) % len(curve)][1]
        if residual <= prev + tolerance and residual <= nxt + tolerance:
            found.append(i)
    return found


def assess(curve, best_residual):
    """Is this curve's minimum trustworthy? Returns (verdict, detail, ok)."""
    residuals = sorted(r for _, r, _ in curve)
    median = residuals[len(residuals) // 2]
    contrast = median / best_residual if best_residual > 0 else float("inf")

    # Count only minima that are actually competitive with the best; a coarse
    # grid on a smooth curve still produces tiny numerical wiggles.
    threshold = best_residual * 1.25
    basins = [i for i in local_minima(curve) if curve[i][1] <= threshold]
    # Adjacent indices are one basin sampled twice, not two basins.
    distinct = [i for j, i in enumerate(basins)
                if j == 0 or i - basins[j - 1] > 1]

    if contrast < MIN_CONTRAST:
        return ("FLAT",
                f"median/min residual is only {contrast:.2f}x (want "
                f">={MIN_CONTRAST}); the sweep barely prefers any offset, so "
                f"its argmin is noise", False)
    if len(distinct) > 1:
        angles = ", ".join(f"{curve[i][0]:.0f}" for i in distinct)
        return ("MULTIMODAL",
                f"{len(distinct)} competitive minima at {angles} deg; the "
                f"bearings or the poses are inconsistent", False)
    return ("SMOOTH UNIMODAL",
            f"single minimum, median/min residual {contrast:.1f}x", True)


def print_curve(curve, best_offset, support_floor, width=44):
    """Sparkline of the sweep, so the shape is visible without a plot.

    Log-scaled: residuals span two orders of magnitude between the basin and a
    ninety-degrees-wrong offset, so a linear bar shows one spike and nothing
    else. Under-supported candidates are marked rather than hidden, since
    "almost nothing triangulated here" is itself the evidence that the offset is
    wrong.
    """
    residuals = [r for _, r, _ in curve]
    lo, hi = min(residuals), max(residuals)
    log_lo, log_span = math.log(lo), max(math.log(hi) - math.log(lo), 1e-9)
    print(f"\n  offset  residual  tracklets            "
          f"[{lo:.2f} .. {hi:.2f} deg, log scale]")
    for offset, residual, n in curve:
        bar = "#" * max(1, int(round(
            (math.log(residual) - log_lo) / log_span * width)))
        flag = "  " if n >= support_floor else " ~"   # ~ = under-supported
        mark = " <-- min" if abs(offset - best_offset) < 1e-9 else ""
        print(f"  {offset:6.1f}  {residual:7.2f}  n={n:3d}{flag}{bar}{mark}")
    print(f"  (~ = fewer than {support_floor} tracklets survived the condition "
          f"gate; not eligible)")


def write_metadata(paths, record, ok, supersede_validated=False):
    """Record the calibration where consumers look for it, gates applied.

    A failed curve still gets written, because the alternative is the next
    person re-deriving it -- but `status` says so and no angle is published, so
    it cannot be misread as an answer.

    **Refuses to overwrite an externally validated offset.** boston_harbor_leg1
    carries `accuracy_validated: true` and 214.0 deg, earned by checking against
    a surveyed building over 72 keyframes -- evidence this sweep does not have
    and cannot reproduce, since it is map-free by construction. This sweep
    returns 212.0 there, which agrees with 214 inside that check's own 2.42 deg
    std; writing it anyway would trade a validated number for an unvalidated one
    two degrees away, flip `accuracy_validated` to false, and desynchronise the
    metadata from every artifact already built at 214. `--supersede_validated`
    forces it, and even then the old block is kept under `superseded`.
    """
    meta_path = paths.metadata_path
    meta = json.loads(meta_path.read_text())
    previous = meta.get("mount_offset") or {}

    if previous.get("accuracy_validated") and not supersede_validated:
        old = previous.get("mount_offset_deg")
        print(f"\nREFUSING to overwrite {meta_path}")
        print(f"  existing: {old} deg, accuracy_validated=true")
        print(f"    source: {previous.get('source', 'unrecorded')}")
        print(f"  this sweep: {record['mount_offset_deg']} deg "
              f"(map-free, accuracy_validated=false)")
        if old is not None:
            delta = abs(hc_delta(record["mount_offset_deg"], old))
            print(f"  they differ by {delta:.1f} deg")
        print("  The existing value has external evidence this sweep does not. "
              "Pass --supersede_validated only if you mean to replace it; the "
              "sweep result is in the JSON either way.")
        return

    block = {
        "status": "triangulation_verified" if ok else record["verdict"].lower(),
        "curve_verdict": record["verdict"],
        "curve_detail": record["detail"],
        "median_residual_deg": record["best_residual_deg"],
        "tracklets_used": record["tracklets_used"],
        "max_condition": record["max_condition"],
        "min_observations": record["min_observations"],
        "support_floor": record["support_floor"],
        "source": f"triangulation-residual sweep on {record['run']}",
        "method": ("median triangulation residual over well-conditioned "
                   "tracklets (mount_offset_sweep.py)"),
        # The sweep is self-consistent and map-free, which is not the same as
        # externally validated. leg1's 214 deg earned that label only after a
        # separate check against a surveyed building over 72 keyframes.
        "accuracy_validated": False,
        "accuracy_note": ("Map-free and self-consistent. For an independent "
                          "check, hypothesise a confidently named landmark and "
                          "compare bearing_matcher.estimate_mount_offset, whose "
                          "per-tracklet spread should be small if the "
                          "hypothesis and this angle agree."),
        "convention": CONVENTION,
    }
    if ok:
        block["mount_offset_deg"] = record["mount_offset_deg"]
    else:
        block["rejected_argmin_deg"] = record["mount_offset_deg"]
    # Keep whatever was there. A calibration's history is part of its evidence,
    # and artifacts already built under the old value need it to stay legible.
    if previous:
        block["superseded"] = previous
    meta["mount_offset"] = block
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote mount_offset block to {meta_path}")
    if previous:
        print(f"  previous value ({previous.get('mount_offset_deg')} deg) kept "
              f"under mount_offset.superseded")
    print("NOTE: this changes pipeline_metadata.json, which is listed in the "
          "dataset's checksums.sha256 - regenerate it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="m3 run dir containing merged/measurements.json")
    parser.add_argument("--coarse_step", type=float, default=5.0)
    parser.add_argument("--fine_step", type=float, default=1.0)
    parser.add_argument("--fine_halfwidth", type=float, default=6.0,
                        help="Refine +/- this far around the coarse minimum")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Coarse sweep start (default: full circle)")
    parser.add_argument("--stop", type=float, default=360.0)
    parser.add_argument("--min_observations", type=int, default=4)
    parser.add_argument("--max_condition", type=float, default=500.0)
    parser.add_argument("--min_support_frac", type=float, default=0.5,
                        help="A candidate offset must keep at least this "
                             "fraction of the best-supported candidate's "
                             "tracklets to be eligible (see module docstring: "
                             "without it, an offset that discards almost every "
                             "tracklet can win on residual)")
    parser.add_argument("--out_json", type=Path, default=None,
                        help="default: <run_dir>/mount_offset_sweep.json")
    parser.add_argument("--write_metadata", action="store_true",
                        help="record the result in the dataset's "
                             "pipeline_metadata.json")
    parser.add_argument("--supersede_validated", action="store_true",
                        help="allow --write_metadata to replace an offset "
                             "marked accuracy_validated (e.g. leg1's 214 deg, "
                             "checked against a surveyed building). The old "
                             "block is kept under mount_offset.superseded.")
    args = parser.parse_args()

    paths = farfield_paths.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks"))
    print(f"dataset: {paths.dataset}\nrun:     {args.run_dir}")

    by_tracklet = load_tracklets(args.run_dir, paths, args.min_observations)
    if not by_tracklet:
        raise SystemExit(
            f"no tracklet has {args.min_observations} observations; lower "
            f"--min_observations or check that m6 produced measurements")

    coarse = sweep(by_tracklet, args.max_condition, args.start, args.stop,
                   args.coarse_step)
    if not coarse:
        raise SystemExit("every candidate offset failed to triangulate; the "
                         "poses or bearings are unusable")

    coarse_eligible, support_floor = eligible(coarse, args.min_support_frac)
    if not coarse_eligible:
        raise SystemExit("no candidate offset kept enough tracklets to compare")
    coarse_best = min(coarse_eligible, key=lambda c: c[1])
    print(f"\ncoarse ({args.coarse_step} deg): {len(coarse_eligible)} of "
          f"{len(coarse)} candidates kept >= {support_floor} tracklets; "
          f"minimum near {coarse_best[0]:.1f} deg")

    fine = sweep(by_tracklet, args.max_condition,
                 coarse_best[0] - args.fine_halfwidth,
                 coarse_best[0] + args.fine_halfwidth + args.fine_step,
                 args.fine_step)
    # Hold the support floor from the coarse pass: the refinement is a local
    # polish, not a fresh competition with its own bar.
    fine_eligible = [c for c in fine if c[2] >= support_floor]
    best = min(fine_eligible or [coarse_best], key=lambda c: c[1])
    best_offset, best_residual, n_used = best

    print_curve(coarse, coarse_best[0], support_floor)
    verdict, detail, ok = assess(coarse_eligible,
                                 min(r for _, r, _ in coarse_eligible))

    print(f"\n  mount_offset_deg  {best_offset:.1f}")
    print(f"  median residual   {best_residual:.2f} deg over {n_used} "
          f"well-conditioned tracklets (of {len(by_tracklet)})")
    print(f"  curve             {verdict} - {detail}")
    if not ok:
        print("\n  DO NOT USE this angle. The curve does not support it.")

    record = {
        "dataset": paths.dataset,
        "run": args.run_dir.name,
        "mount_offset_deg": round(best_offset, 2),
        "best_residual_deg": round(best_residual, 3),
        "tracklets_used": n_used,
        "tracklets_available": len(by_tracklet),
        "min_observations": args.min_observations,
        "max_condition": args.max_condition,
        "min_support_frac": args.min_support_frac,
        "support_floor": support_floor,
        "verdict": verdict,
        "detail": detail,
        "usable": ok,
        "convention": CONVENTION,
        "coarse_curve": [{"offset_deg": o, "residual_deg": r, "n": n}
                         for o, r, n in coarse],
        "fine_curve": [{"offset_deg": o, "residual_deg": r, "n": n}
                       for o, r, n in fine],
    }
    out_json = args.out_json or (args.run_dir / "mount_offset_sweep.json")
    out_json.write_text(json.dumps(record, indent=1) + "\n")
    print(f"\nwrote {out_json}")

    if args.write_metadata:
        write_metadata(paths, record, ok, args.supersede_validated)


if __name__ == "__main__":
    main()
