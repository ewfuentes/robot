"""Camera-to-body yaw offset by minimising triangulation residual.

**This is the reference relative method** for `mount_offset_deg` -- the angle
that turns a camera-frame bearing into a body-frame one
(see geometry.MOUNT_OFFSET_CONVENTION). It exists as a tool so the number is
reproducible rather than remembered.

Why this and not the alternatives:

- Focus-of-expansion calibration recovers the same angle with no tracks at
  all, but on the one dataset with an external reference it was confidently
  wrong -- tiny axis spread, inverted answer. It is a corroborator.
- Bow calibration measures the **bow**, which differs from the direction of
  travel by the crab angle. On a boat that is not a constant.
- `sun_offset_check` is the one *absolute* method; prefer it when the sky
  cooperates, and use this sweep to corroborate (or when it does not).

The idea: a wrong offset rotates every bearing by the same constant, and
rotated rays to a static object stop intersecting. So sweep the offset,
triangulate each tracklet's bearings at each candidate, and take the angle
that minimises the median residual. No map, and no assumed
tracklet-to-landmark match, enter the estimate -- which matters, because the
map match is what the offset is needed for downstream.

The gate that matters most is on the **observation arc** -- how far the
bearing to the object swept while it was tracked (`--min_arc_deg`). Two
properties make it the right gate:

- It is **offset-invariant**. `bearing_world = course + camera - offset`, so
  a candidate offset subtracts the same constant from every bearing in a
  tracklet and leaves their spread exactly unchanged. The tracklet set is
  therefore fixed across the whole sweep, and residuals at different offsets
  are comparable without any correction. The condition number is *not*
  offset-invariant, which is the entire reason the support gate below had to
  exist.
- It removes the tracklets that **cannot answer the question**. Rays spread
  over a wide arc can never be rotated into parallelism; rays over a 2 deg
  arc are nearly parallel at *every* offset, so they always show a small
  angular residual and always vote for whichever offset makes them most
  parallel. That is a spurious minimum with real support behind it: run
  without the arc gate, the argmin can land >100 deg from the truth with
  excellent-looking residuals. It also rescues narrow-waterway datasets whose
  diluted median the old gates could only call FLAT.

Three further gates keep noise out of the minimum:

- **condition number** (`--max_condition`). Bearings taken over a short arc
  intersect at a glancing angle, so a tiny residual can coexist with a
  position uncertain by kilometres. Residual alone is not enough.
- **observation count** (`--min_observations`). Three is the bare minimum for
  a residual to exist at all; four makes it mean something.
- **support** (`--min_support_frac`). *This one is not optional*, and the
  reason is a trap in the obvious version of this procedure: the condition
  gate is applied per candidate offset, so how many tracklets survive it is
  itself a function of the offset. A badly wrong offset drops almost every
  tracklet, and the handful that remain can have an excellent median residual
  purely by selection -- a numerically better minimum that explains an eighth
  as much data has been observed winning for real. Candidates are therefore
  only eligible if they keep at least this fraction of the best-supported
  candidate's tracklets, which encodes the obvious principle: at equal
  residual, the offset that explains more tracklets wins.

Tuning history: see docs/farfield (previously inline here).

Read the printed curve, not just the minimum. A genuine calibration is smooth
and unimodal with a clear contrast between the floor and the rest; a flat or
multimodal curve means the bearings or the poses are wrong, and its argmin is
noise. The summary says which of those it looks like.

Writes `<run_dir>/mount_offset_sweep.json` -- a sidecar in the run directory,
always. It NEVER touches the dataset's `pipeline_metadata.json`: datasets are
frozen (REORG.md rule 7). A separate, explicit "publish to dataset metadata"
tool (with the accuracy-validated guard and checksum regeneration) will land
later; until then, consumers read the sidecar.

    bazel run //experimental/overhead_matching/swag/farfield/calibration:mount_offset_sweep -- \\
        --run_dir <object_tracks>/<ds>/vN/runs/<run> \\
        --coarse_step ... --fine_step ... --fine_halfwidth ... \\
        --min_observations ... --min_arc_deg ... --max_condition ... \\
        --min_tracklets ... --min_support_frac ... \\
        --epoch_keyframes ... --bearing_sigma_deg ...
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.calibration import (
    audit_io,
    heading as heading_mod,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets

# One shared definition. Restating it is what produced the pohang 180 deg
# slip; see docs/conventions.md.
CONVENTION = geo.MOUNT_OFFSET_CONVENTION

SIDECAR_NAME = "mount_offset_sweep.json"

# A minimum this shallow relative to the rest of the curve is not a minimum.
MIN_CONTRAST = 1.5

# An offset is a claim about the mount, and one tracklet cannot support it: a
# single tracklet's residual is a smooth function of the offset by
# construction, so the curve looks textbook-unimodal while saying nothing.
# Observed for real: a 1-tracklet sweep returned "SMOOTH UNIMODAL, 0.95 deg"
# while the same leg with less bearing fusion (7 tracklets) disagreed by
# nearly 180 deg. The other gates are all relative, so none of them can catch
# this. This is `assess`'s default floor; the CLI value is required.
MIN_TRACKLETS = 5


def arc_deg(observations):
    """Angular spread of a tracklet's bearings -- the smallest arc holding them.

    Computed on `course + camera`, i.e. the world bearing at offset 0. The
    candidate offset is subtracted from every bearing alike, so it cannot
    change a spread: this number is the same at every offset, which is what
    lets it select a fixed tracklet set for the whole sweep.
    """
    angles = sorted((course + camera) % 360.0
                    for _, _, camera, course, _ in observations)
    if len(angles) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(angles, angles[1:])]
    gaps.append(angles[0] + 360.0 - angles[-1])
    return 360.0 - max(gaps)


def triangulate(rays):
    """Least-squares intersection of bearing rays, in ENU metres.

    rays: [(east_m, north_m, bearing_world_deg)]. Returns
    (east_m, north_m, median_abs_residual_deg, condition_number), or None if
    under-determined.

    This is the honest consistency check for a tracklet, and it replaces
    "circular std of the world bearing", which was wrong: a static object at
    1 km sweeps tens of degrees of true bearing as the vessel passes it, so
    spread measures parallax at least as much as error. What we actually want
    to know is whether the bearings are consistent with *some* single static
    point -- which is exactly what the residual about the triangulated
    intersection reports.

    The condition number matters as much as the residual: bearings taken over
    a short arc intersect at a glancing angle, so a tiny residual can hide a
    position uncertain by kilometres along the line of sight. Callers must
    gate on both.

    Ported from bearing_matcher.triangulate on the checkpoint branch; when
    the matching package lands (REORG.md PR 09) its copy and this one should
    merge into a single owner.
    """
    if len(rays) < 2:
        return None
    a_mat = np.zeros((2, 2))
    b_vec = np.zeros(2)
    units, points = [], []
    for east_m, north_m, bearing_world_deg in rays:
        theta = math.radians(bearing_world_deg)
        unit = np.array([math.sin(theta), math.cos(theta)])   # (east, north)
        proj = np.eye(2) - np.outer(unit, unit)
        a_mat += proj
        b_vec += proj @ np.array([east_m, north_m])
        units.append(unit)
        points.append(np.array([east_m, north_m]))
    eigenvalues = np.linalg.eigvalsh(a_mat)
    if eigenvalues.min() <= 1e-9:
        return None
    condition = float(eigenvalues.max() / eigenvalues.min())
    point = np.linalg.solve(a_mat, b_vec)

    # The least-squares solve intersects LINES, not rays, so it can place the
    # object behind the observer -- which then reports a ~180 deg residual and
    # is meaningless. Seen for real: one tracklet came back at 179.7 deg.
    # Require the solution to lie ahead of most observations.
    ahead = sum(1 for unit, origin in zip(units, points)
                if float(np.dot(point - origin, unit)) > 0.0)
    if ahead * 2 < len(rays):
        return None

    residuals = []
    for (east_m, north_m, bearing_world_deg), origin in zip(rays, points):
        predicted = geo.compass_bearing_deg(point[0] - east_m,
                                            point[1] - north_m)
        residuals.append(abs(float(geo.circular_diff_deg(
            bearing_world_deg, predicted))))
    residuals.sort()
    return (float(point[0]), float(point[1]),
            float(residuals[len(residuals) // 2]), condition)


def load_tracks(run_dir: Path) -> dict:
    """{track_id: track} merged across every tracks_*.json in the run.

    A run may be split across several range files. The old merge stage read
    only the first (`next(glob)`), silently dropping every other range's
    tracks; load them all, and refuse a duplicate id rather than let one
    range's track silently shadow another's.
    """
    track_paths = sorted(Path(run_dir).glob("tracks_*.json"))
    if not track_paths:
        raise SystemExit(f"no tracks_*.json under {run_dir} -- run the "
                         f"tracking stage first")
    tracks = {}
    for path in track_paths:
        artifact = json.loads(path.read_text())
        for track in artifact["tracks"]:
            tid = track["track_id"]
            if tid in tracks:
                raise SystemExit(
                    f"track_id {tid!r} appears in more than one "
                    f"tracks_*.json under {run_dir} (again in {path.name}); "
                    f"range files must not share ids")
            tracks[tid] = track
    return tracks


def load_tracklets(run_dir: Path, paths, fusion: tracklets.TrackletParams,
                   min_observations: int, min_arc_deg: float):
    """{tracklet_id: [(east_m, north_m, bearing_camera_deg, course_deg, kf)]}.

    Bearings are fused per-epoch camera azimuths straight from
    tracklets.build_measurements -- the merge stage is gone, and each audited
    track is its own tracklet. Poses and course come from the same dataset
    frames + heading path the tracking stages use, so the sweep cannot
    disagree with the pipeline about where the vehicle was pointing.
    """
    tracks = load_tracks(run_dir)
    audits = audit_io.load_audits(run_dir)
    if not audits:
        raise SystemExit(
            f"{run_dir}/semantic_audit has no results -- run the audit stage "
            f"first. Tracklets exist only for audited tracks (audit "
            f"membership is the support gate), so the sweep has nothing to "
            f"triangulate.")

    frames = dataset.load_frames(paths.dataset_base)
    if not frames:
        raise SystemExit(f"no panoramas under {paths.dataset_base}/panorama")
    dataset.fill_enu(frames)
    with Image.open(
            paths.panorama_dir / f"{frames[0].pano_stem}.jpg") as probe:
        pano_w = probe.size[0]

    measurements = tracklets.build_measurements(tracks, audits, pano_w,
                                                fusion)

    frames_by_idx = {f.frame_idx: f for f in frames}
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in frames], [f.y_m for f in frames],
        [f.time_s for f in frames])

    by_tracklet = defaultdict(list)
    for m in measurements:
        frame = frames_by_idx.get(m.anchor_keyframe_idx)
        if frame is None:
            continue
        by_tracklet[m.tracklet_id].append(
            (frame.x_m, frame.y_m, m.bearing_camera_deg,
             float(model.at(frame.time_s)), m.anchor_keyframe_idx))

    enough = {t: obs for t, obs in by_tracklet.items()
              if len(obs) >= min_observations}
    kept = {t: obs for t, obs in enough.items()
            if arc_deg(obs) >= min_arc_deg}
    print(f"{len(measurements)} measurements over {len(by_tracklet)} "
          f"tracklets; {len(enough)} have >= {min_observations} "
          f"observations, {len(kept)} of those sweep >= {min_arc_deg:.0f} "
          f"deg of bearing")
    if enough and not kept:
        arcs = sorted((arc_deg(o) for o in enough.values()), reverse=True)
        print(f"  widest arc available is {arcs[0]:.1f} deg - this run cannot "
              f"support a calibration at --min_arc_deg {min_arc_deg:.0f}")
    return kept


def residual_at(offset_deg: float, by_tracklet, max_condition: float):
    """Median triangulation residual over well-conditioned tracklets."""
    residuals = []
    for observations in by_tracklet.values():
        rays = [
            (east, north,
             float(geo.body_to_world_bearing_deg(
                 course, geo.apply_mount_offset(camera, offset_deg))))
            for east, north, camera, course, _ in observations
        ]
        result = triangulate(rays)
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


def assess(curve, best_residual, n_used, min_tracklets=MIN_TRACKLETS):
    """Is this curve's minimum trustworthy? Returns (verdict, detail, ok)."""
    if n_used < min_tracklets:
        return ("UNDER-SUPPORTED",
                f"the winning offset triangulates only {n_used} "
                f"well-conditioned tracklet(s) (want >={min_tracklets}); with "
                f"so few, the curve's shape is a property of those tracklets "
                f"rather than of the mount", False)
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

    Log-scaled: residuals span two orders of magnitude between the basin and
    a ninety-degrees-wrong offset, so a linear bar shows one spike and nothing
    else. Under-supported candidates are marked rather than hidden, since
    "almost nothing triangulated here" is itself the evidence that the offset
    is wrong.
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
    print(f"  (~ = fewer than {support_floor} tracklets survived the "
          f"condition gate; not eligible)")


def write_sidecar(run_dir: Path, record: dict) -> Path:
    """Write the sweep's result JSON into the run dir -- always.

    The sidecar is the ONLY thing this tool writes. It never touches the
    dataset's pipeline_metadata.json: datasets are frozen (REORG.md rule 7),
    and publishing a calibration into dataset metadata is a separate,
    explicit tool (with the accuracy-validated guard and checksum
    regeneration). The convention/frame/provenance fields are stamped here so
    every sidecar carries them by construction.
    """
    record = dict(record)
    record["convention"] = geo.MOUNT_OFFSET_CONVENTION
    record["frame"] = geo.MOUNT_OFFSET_FRAME
    record["git_commit"] = provenance.git_commit()
    record["argv"] = list(sys.argv)
    out = Path(run_dir) / SIDECAR_NAME
    out.write_text(json.dumps(record, indent=1) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True,
                        help="tracking run dir containing tracks_*.json and "
                             "semantic_audit/")
    # Every tuned value is required (REORG.md rule 2: no stale defaults on
    # assumption-carrying args). The previous defaults are quoted for
    # reference, not authority.
    # TODO(REORG.md PR 12): these move into the run's recorded config
    # (run_config.json); CLI-required until then.
    parser.add_argument("--coarse_step", type=float, required=True,
                        help="Coarse sweep grid, degrees (previously 5.0 on "
                             "the harbor datasets)")
    parser.add_argument("--fine_step", type=float, required=True,
                        help="Fine sweep grid, degrees (previously 1.0 on "
                             "the harbor datasets)")
    parser.add_argument("--fine_halfwidth", type=float, required=True,
                        help="Refine +/- this far around the coarse minimum "
                             "(previously 6.0 on the harbor datasets)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Coarse sweep start (default: full circle)")
    parser.add_argument("--stop", type=float, default=360.0,
                        help="Coarse sweep stop (default: full circle)")
    parser.add_argument("--min_observations", type=int, required=True,
                        help="A tracklet needs at least this many fused "
                             "bearings to vote (previously 4 on the harbor "
                             "datasets)")
    parser.add_argument("--min_arc_deg", type=float, required=True,
                        help="A tracklet's bearings must sweep at least this "
                             "far to vote. Offset-invariant, so it fixes the "
                             "tracklet set for the whole sweep; below "
                             "roughly 20 deg the argmin is decided by which "
                             "rays are most parallel, not by the mount "
                             "(previously 20 on the harbor datasets)")
    parser.add_argument("--max_condition", type=float, required=True,
                        help="Triangulation condition-number gate "
                             "(previously 500 on the harbor datasets)")
    parser.add_argument("--min_tracklets", type=int, required=True,
                        help="refuse an offset supported by fewer than this "
                             "many well-conditioned tracklets (previously "
                             f"{MIN_TRACKLETS} on the harbor datasets)")
    parser.add_argument("--min_support_frac", type=float, required=True,
                        help="A candidate offset must keep at least this "
                             "fraction of the best-supported candidate's "
                             "tracklets to be eligible (see module "
                             "docstring: without it, an offset that discards "
                             "almost every tracklet can win on residual) "
                             "(previously 0.5 on the harbor datasets)")
    parser.add_argument("--epoch_keyframes", type=int, required=True,
                        help="Keyframes fused into one bearing measurement "
                             "(tracklets.TrackletParams; previously 5 on the "
                             "harbor datasets)")
    parser.add_argument("--bearing_sigma_deg", type=float, required=True,
                        help="Per-observation bearing noise floor, degrees "
                             "(tracklets.TrackletParams; previously 1.0 on "
                             "the harbor datasets)")
    args = parser.parse_args()

    paths = paths_lib.resolve(parser, args, infer_from=args.run_dir,
                              require=("dataset_base", "panorama_dir"))
    # The sweep's camera azimuths are pano-column azimuths; refuse a dataset
    # whose panoramas are not stored in the camera frame.
    metadata = dataset.load_metadata(paths.dataset_base)
    dataset.require_camera_frame_panoramas(metadata, paths.dataset_base)
    print(f"dataset: {paths.dataset}\nrun:     {args.run_dir}")

    fusion = tracklets.TrackletParams(
        epoch_keyframes=args.epoch_keyframes,
        bearing_sigma_deg=args.bearing_sigma_deg)
    by_tracklet = load_tracklets(args.run_dir, paths, fusion,
                                 args.min_observations, args.min_arc_deg)
    if not by_tracklet:
        raise SystemExit(
            f"no tracklet has {args.min_observations} observations and an arc "
            f"of {args.min_arc_deg:.0f} deg. Lowering --min_arc_deg will "
            f"produce a number, but see the module docstring: below ~20 deg "
            f"the argmin is decided by which rays are most parallel, not by "
            f"the mount. Prefer sun_offset_check, or an offset from a "
            f"sibling leg of the same rig.")

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
                                 min(r for _, r, _ in coarse_eligible),
                                 n_used, args.min_tracklets)

    print(f"\n  mount_offset_deg  {best_offset:.1f}")
    print(f"  median residual   {best_residual:.2f} deg over {n_used} "
          f"well-conditioned tracklets (of {len(by_tracklet)})")
    print(f"  curve             {verdict} - {detail}")
    if not ok:
        print("\n  DO NOT USE this angle. The curve does not support it.")

    record = {
        "dataset": paths.dataset,
        "run": args.run_dir.name,
        "generator": "farfield/calibration/mount_offset_sweep.py",
        "mount_offset_deg": round(best_offset, 2),
        "best_residual_deg": round(best_residual, 3),
        "tracklets_used": n_used,
        "tracklets_available": len(by_tracklet),
        "min_observations": args.min_observations,
        "min_arc_deg": args.min_arc_deg,
        "median_arc_deg": round(sorted(
            arc_deg(o) for o in by_tracklet.values())[len(by_tracklet) // 2],
            2),
        "max_condition": args.max_condition,
        "min_support_frac": args.min_support_frac,
        "min_tracklets": args.min_tracklets,
        "epoch_keyframes": args.epoch_keyframes,
        "bearing_sigma_deg": args.bearing_sigma_deg,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "fine_halfwidth": args.fine_halfwidth,
        "support_floor": support_floor,
        "verdict": verdict,
        "detail": detail,
        "usable": ok,
        "coarse_curve": [{"offset_deg": o, "residual_deg": r, "n": n}
                         for o, r, n in coarse],
        "fine_curve": [{"offset_deg": o, "residual_deg": r, "n": n}
                       for o, r, n in fine],
    }
    out_json = write_sidecar(args.run_dir, record)
    print(f"\nwrote {out_json}")
    print("publishing to dataset metadata is a separate explicit tool (not "
          "yet ported); datasets stay frozen -- consume the sidecar")


if __name__ == "__main__":
    main()
