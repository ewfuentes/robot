"""Cut frames out of a collected dataset, keeping every file consistent.

Two things condemn frames, and this handles both.

*Range* trims come out of visual review of `gps_timelapse.mp4`, which usually
objects to a *part* of a trajectory: the operator swings the camera after the
first minute, a recording restarts pointing the other way, the last few frames
come from a different day.

*Density* trims (`--spacing_m`) thin a trajectory that was sampled far denser
than the work needs. Nothing is wrong with the dropped frames; there are
simply too many of them. A collect on a 3 m grid costs ~10x what the same
trajectory costs on a 30 m grid at every per-frame stage, and for far-field
landmark work consecutive frames 3 m apart are near-identical: a landmark
several kilometres away does not move measurably between them.

Either way the result still satisfies the contract
`farfield/audit_dataset.py` enforces.

Trimming is not just dropping CSV rows. The audit requires `frames_gps.idx` to
be 0..N-1 contiguous *and* `pano_id[1:] == idx`, because that equality is the
ingest join key. Cutting from the middle therefore forces a renumber, and the
renumber forces an image rename, and the rename has to reach
`frames_gps.csv`, `extraction_log.csv`, `intrinsics.csv` and
`pano_id_mapping.csv` together or the dataset silently comes apart. It also
has to reach every record keyed on frame indices: `recording_seams` entries
name the frame each odometry break sits after, and an earlier version left
them pointing at pre-trim indices — well-formed, silently wrong. This trim
rebases them (see `rebase_seams`).

This is one of the two explicit dataset-mutating tools (REORG.md rule 7); it
finishes by regenerating `checksums.sha256` through the shared implementation
in `checksums.py`.

Indices in --keep/--drop are ORIGINAL frame indices (half-open, Python slice
semantics), read off the timelapse via `--video_fps` if that is easier:

    # keep only the first 11 seconds of a 15 fps timelapse
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:trim_dataset -- \\
        --dataset_path /data/farfield_matching/datasets/fukuoka_yumechan_a \\
        --keep 0:165 --reason "operator swings the camera after t=11 s" --dry_run

    # thin a 3 m collect to a 10 m grid, into its own trim directory
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:trim_dataset -- \\
        --dataset_path /data/farfield_matching/datasets/charles_river_20260727 \\
        --spacing_m 10 --trim_dir trimmed_frames_for_density \\
        --reason "3.1 m/frame is ~10x denser than far-field extraction needs"

Dropped images and the original CSVs move to `--trim_dir` inside the dataset,
so a trim is reversible and costs no extra disk. Give each *kind* of trim its
own directory: the dropped frames then say why they were dropped by where they
sit, and `dropped_frames.csv` in each directory records the same thing per
file for anyone who merges them back together.
"""

import argparse
import bisect
import csv
import datetime
import json
import shutil
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    checksums,
)

CSV_NAMES = ["frames_gps.csv", "extraction_log.csv", "intrinsics.csv",
             "pano_id_mapping.csv"]
SEAMS_SIDECAR = Path("_manifests") / "recording_seams.json"
# Above this many kept runs, inlining them in pipeline_metadata.json hides
# everything else in the file instead of documenting the trim.
MAX_RECORDED_RANGES = 12


def read_csv(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path: Path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_ranges(specs, n):
    """['0:165', '300:'] -> [(0, 165), (300, n)]. Half-open, Python semantics."""
    out = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"range {spec!r} must be A:B")
        lo, _, hi = spec.partition(":")
        lo = int(lo) if lo.strip() else 0
        hi = int(hi) if hi.strip() else n
        if not (0 <= lo < hi <= n):
            raise ValueError(f"range {spec!r} outside 0:{n}")
        out.append((lo, hi))
    return sorted(out)


def keep_by_spacing(gps_rows, spacing_m):
    """Indices on a `spacing_m` along-track grid, greedy from frame 0.

    Walks the recorded cumulative `dist_m` rather than re-integrating
    positions, so the grid follows the same track the dataset was built on and
    a stationary stretch collapses to a single frame instead of surviving as a
    cluster.

    Each step takes the frame whose distance is *closest* to the target, not
    the first one at or past it. Taking the first one past overshoots by up to
    a source interval every step and never undershoots, which on a 3 m collect
    turns a 10 m request into a 12-13 m result; picking the nearest frame
    centres the realized spacing on what was asked for.

    The last frame is kept when it sits at least half a spacing past the
    previous keeper: the endpoint of a trajectory is worth more than the
    uniformity of the final interval, but not worth a near-duplicate.
    """
    dists = [float(r["dist_m"]) for r in gps_rows]
    if any(b < a for a, b in zip(dists, dists[1:])):
        raise ValueError("dist_m is not monotonic; fix the dataset before "
                         "thinning by distance")
    last = len(dists) - 1
    keep = [0]
    while dists[last] >= dists[keep[-1]] + spacing_m:
        target = dists[keep[-1]] + spacing_m
        j = keep[-1] + 1
        while dists[j] < target:      # terminates: dists[last] >= target
            j += 1
        # j is the first frame at or past the target, j-1 the last one before
        # it. Prefer whichever is nearer, but never step backwards.
        if (j - 1 > keep[-1]
                and abs(dists[j - 1] - target) < abs(dists[j] - target)):
            j -= 1
        keep.append(j)
    if keep[-1] != last and dists[last] - dists[keep[-1]] >= spacing_m / 2:
        keep.append(last)
    if len(keep) < 2:
        raise ValueError(
            f"--spacing_m {spacing_m} keeps {len(keep)} frame(s) of "
            f"{len(dists)}; the track is only "
            f"{dists[-1] - dists[0]:.0f} m long")
    return keep


def keep_indices(keep_specs, drop_specs, n):
    if keep_specs and drop_specs:
        raise ValueError("pass --keep or --drop, not both")
    if keep_specs:
        ranges = parse_ranges(keep_specs, n)
        keep = sorted({i for lo, hi in ranges for i in range(lo, hi)})
    elif drop_specs:
        ranges = parse_ranges(drop_specs, n)
        dropped = {i for lo, hi in ranges for i in range(lo, hi)}
        keep = [i for i in range(n) if i not in dropped]
    else:
        raise ValueError("pass --keep or --drop")
    if not keep:
        raise ValueError("the trim keeps no frames")
    return keep


def new_frame_file(old_name: str, new_idx: int) -> str:
    """`f0319,40.70,-74.01,.jpg` -> `f0007,40.70,-74.01,.jpg`.

    Only the pano_id field changes. The audit cross-checks the lat/lon fields
    of the name against frames_gps, so they must survive the rename untouched.
    """
    parts = old_name.split(",")
    parts[0] = f"f{new_idx:04d}"
    return ",".join(parts)


def summarize(gps_rows, log_rows, keep):
    """Distance, duration and sequence make-up of a kept subset."""
    lat = [float(gps_rows[i]["latitude"]) for i in keep]
    lon = [float(gps_rows[i]["longitude"]) for i in keep]
    ts = [int(log_rows[i]["captured_at"]) / 1000.0 for i in keep]
    dist = sum(geo.haversine_m(lat[k - 1], lon[k - 1], lat[k], lon[k])
               for k in range(1, len(keep)))
    seqs = [log_rows[i]["sequence_id"] for i in keep]
    return {
        "n": len(keep),
        "dist_km": dist / 1000.0,
        "dur_min": (ts[-1] - ts[0]) / 60.0,
        "n_sequences": len(set(seqs)),
        "start_utc": datetime.datetime.fromtimestamp(
            ts[0], datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "end_utc": datetime.datetime.fromtimestamp(
            ts[-1], datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def rebuild_gps_rows(gps_rows, log_rows, keep, kind="range"):
    """frames_gps for the kept subset: renumbered, with dist_m rebased to zero.

    `dist_m` rebases because it describes the trimmed trajectory. `video_t_s`
    and `sensor_elapsed_s` do NOT: they are addresses into the source video
    and the sensor log, neither of which the trim touches. An earlier version
    rebased them to zero at the new first frame, reasoning that the result
    should be "indistinguishable from one collected over that range" -- but
    the mp4 still starts where it always started, so every kept frame then
    addressed content earlier than itself by the first kept frame's original
    `video_t_s`.

    Note that a head cut is not required to trigger this: on
    charles_river_20260727 the trim was a *density* thin that kept frame 0,
    and the damage was still 510 s, because the dataset's frames were exported
    starting at video t=510 (`video.export_start_video_t_s`) and zeroing the
    column threw that offset away. Tracking then cropped every window from a
    different part of the sail (verified 2026-08-18 by cross-correlating
    frames against their panoramas: 0.49-0.74 as stored, 0.999-1.000 once
    restored). Carry both columns through verbatim.

    Where dist_m comes from depends on what was cut. A *range* trim removes
    stretches of track, so the distance must be re-derived from the surviving
    positions -- carrying the original column forward would keep charging for
    the removed stretch. Across a cut that adds the straight-line jump, which
    is what the converter does at a stitch seam.

    A *density* trim removes no track at all, so the original column is still
    the right answer and is re-used rebased. It matters: `dist_m` is normally
    accumulated along the dataset's smoothed/bridged track, while re-deriving
    it walks the raw reported positions and picks up their noise instead. On
    the charles collect the two differ by 6.7%, so re-deriving would make a
    trim that removes nothing report a *longer* trajectory than before it ran.
    """
    out = []
    cumulative_m = 0.0
    for new_idx, old in enumerate(keep):
        row = dict(gps_rows[old])
        if new_idx > 0:
            prev = gps_rows[keep[new_idx - 1]]
            if kind == "density":
                step = float(row["dist_m"]) - float(prev["dist_m"])
            else:
                step = geo.haversine_m(
                    float(prev["latitude"]), float(prev["longitude"]),
                    float(row["latitude"]), float(row["longitude"]))
            cumulative_m += step
            dt = (int(log_rows[old]["captured_at"])
                  - int(log_rows[keep[new_idx - 1]]["captured_at"])) / 1000.0
            speed = round(step / dt, 3) if dt > 0 else -1.0
        else:
            speed = -1.0
        row["idx"] = new_idx
        row["dist_m"] = round(cumulative_m, 1)
        row["speed_mps"] = speed
        row["frame_file"] = new_frame_file(row["frame_file"], new_idx)
        out.append(row)
    return out


def rebase_seams(record: dict, keep: list, kind: str) -> dict:
    """Re-key a recording_seams record onto the post-trim frame numbering.

    A seam sits *between* original frames `after_idx` and `after_idx + 1`.
    After a trim that boundary still exists between the last kept frame at or
    before `after_idx` and the first kept frame after it, so the seam maps to
    `new_after = (# kept indices <= after_idx) - 1`. Seams that fall before
    the first kept frame or after the last one no longer separate anything and
    are dropped; two seams collapsing onto one surviving boundary keep the
    first (the boundary is still a break either way).

    Left un-rebased, every seam silently points at the wrong frame after a
    renumbering trim — the same well-formed-but-wrong failure mode as the
    video_t_s rebase, in the other direction (this record is *indices*, which
    the trim changes, not *addresses*, which it must not).

    The recorded dt/step metrics describe the original flanking pair; when
    either flank was itself dropped, the surviving boundary spans farther than
    the seam measured, so those entries are marked `metrics_stale`. A range
    trim also *creates* discontinuities this record never saw, so the rebase
    note says to re-run annotate_recording_seams for a fresh record.
    """
    kept_sorted = list(keep)
    new_seams, seen_after = [], set()
    n_dropped = 0
    for seam in record.get("seams", []):
        old_after = int(seam["after_idx"])
        new_after = bisect.bisect_right(kept_sorted, old_after) - 1
        if new_after < 0 or new_after >= len(kept_sorted) - 1:
            n_dropped += 1
            continue
        if new_after in seen_after:
            n_dropped += 1
            continue
        seen_after.add(new_after)
        rebased = dict(seam)
        rebased["after_idx"] = new_after
        rebased["original_after_idx"] = old_after
        if (kept_sorted[new_after] != old_after
                or kept_sorted[new_after + 1] != old_after + 1):
            rebased["metrics_stale"] = True
        new_seams.append(rebased)
    out = dict(record)
    out["seams"] = new_seams
    out["n_seams"] = len(new_seams)
    out["rebased_by_trim"] = {
        "trim_kind": kind,
        "n_seams_before": len(record.get("seams", [])),
        "n_seams_dropped": n_dropped,
        "rebased_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": ("after_idx values were re-keyed to the post-trim frame "
                 "numbering (original_after_idx keeps the pre-trim index). "
                 + ("A range trim also creates new odometry breaks this "
                    "record has never seen; re-run annotate_recording_seams "
                    "for a fresh record." if kind != "density" else
                    "A density trim widens every step; dt/step metrics "
                    "describe the original flanking pair where marked "
                    "metrics_stale.")),
    }
    return out


def write_dropped_csv(backup_dir: Path, gps_rows, log_rows, dropped, kind,
                      reason: str, stamp: str):
    """Per-file record of what left and why, alongside the files themselves.

    The directory a dropped frame sits in already says which trim took it, but
    only while the directories stay separate. This survives them being merged,
    and carries the original index so a frame can be put back.
    """
    quality_keys = [k for k in ("gps_quality", "gps_valid", "course_deg")
                    if log_rows and k in log_rows[0]]
    fields = (["original_idx", "frame_file", "trim_kind", "reason",
               "trimmed_at", "video_t_s", "dist_m", "latitude", "longitude"]
              + quality_keys)
    rows = []
    for i in dropped:
        row = {"original_idx": i,
               "frame_file": gps_rows[i]["frame_file"],
               "trim_kind": kind,
               "reason": reason,
               "trimmed_at": stamp}
        for key in ("video_t_s", "dist_m", "latitude", "longitude"):
            row[key] = gps_rows[i].get(key, "")
        for key in quality_keys:
            row[key] = log_rows[i].get(key, "")
        rows.append(row)
    path = backup_dir / "dropped_frames.csv"
    existing, _ = read_csv(path) if path.exists() else ([], None)
    write_csv(path, existing + rows, fields)
    return path


def apply_trim(ds: Path, keep, reason: str, video_fps: float | None, *,
               trim_dir: str = "trimmed_frames", kind: str = "range"):
    tables = {name: read_csv(ds / name) for name in CSV_NAMES}
    gps_rows, gps_fields = tables["frames_gps.csv"]
    log_rows, log_fields = tables["extraction_log.csv"]
    n = len(gps_rows)
    keep_set = set(keep)

    backup_dir = ds / trim_dir
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in CSV_NAMES:
        shutil.copy2(ds / name, backup_dir / f"{stamp}.{name}")
    dropped = [i for i in range(n) if i not in keep_set]
    write_dropped_csv(backup_dir, gps_rows, log_rows, dropped, kind, reason,
                      stamp)

    # Images first: move every dropped file out, THEN renumber. A kept frame's
    # new index is always <= its old one, so an ascending rename can only ever
    # land on a name that has already been vacated. Renaming before the moves
    # would clobber a dropped file that still had to be preserved.
    frames_dir = (ds / "panorama").resolve()
    moved = 0
    for old in range(n):
        if old in keep_set:
            continue
        src = frames_dir / gps_rows[old]["frame_file"]
        if src.exists():
            shutil.move(str(src), str(backup_dir / src.name))
            moved += 1
    renamed = 0
    for new_idx, old in enumerate(keep):
        old_name = gps_rows[old]["frame_file"]
        new_name = new_frame_file(old_name, new_idx)
        if old_name == new_name:
            continue
        src, dst = frames_dir / old_name, frames_dir / new_name
        if dst.exists():
            raise FileExistsError(f"rename target already present: {dst}")
        src.rename(dst)
        renamed += 1

    new_gps = rebuild_gps_rows(gps_rows, log_rows, keep, kind)
    write_csv(ds / "frames_gps.csv", new_gps, gps_fields)

    new_log = []
    for new_idx, old in enumerate(keep):
        row = dict(log_rows[old])
        row["frame_idx"] = new_idx
        row["pano_id"] = f"f{new_idx:04d}"
        row["output_filename"] = new_frame_file(row["output_filename"],
                                                new_idx)
        new_log.append(row)
    write_csv(ds / "extraction_log.csv", new_log, log_fields)

    intr_rows, intr_fields = tables["intrinsics.csv"]
    new_intr = []
    for new_idx, old in enumerate(keep):
        row = dict(intr_rows[old])
        row["idx"] = new_idx
        row["pano_id"] = f"f{new_idx:04d}"
        new_intr.append(row)
    write_csv(ds / "intrinsics.csv", new_intr, intr_fields)

    map_rows, map_fields = tables["pano_id_mapping.csv"]
    new_map = []
    for new_idx, old in enumerate(keep):
        row = dict(map_rows[old])
        row["pano_id"] = f"f{new_idx:04d}"
        row["filename"] = new_frame_file(row["filename"], new_idx)
        new_map.append(row)
    write_csv(ds / "pano_id_mapping.csv", new_map, map_fields)

    meta_path = ds / "pipeline_metadata.json"
    meta = json.load(open(meta_path))
    kept_log = [log_rows[i] for i in keep]
    meta["num_images"] = len(keep)
    meta["captured_at_ms"] = int(kept_log[0]["captured_at"])
    meta["capture_date"] = datetime.datetime.fromtimestamp(
        int(kept_log[0]["captured_at"]) / 1000.0,
        datetime.timezone.utc).strftime("%Y-%m-%d")
    seq_ids = list(dict.fromkeys(r["sequence_id"] for r in kept_log))
    meta["component_sequence_ids"] = seq_ids
    meta["stitched_from_n_sequences"] = len(seq_ids)
    counts = {}
    for r in kept_log:
        counts[r["geometry_source"]] = counts.get(r["geometry_source"], 0) + 1
    meta["geometry_source_counts"] = counts

    # The trimmed track is a coarser polyline through the same points, so its
    # length is slightly under the original. Report what the frames now say
    # and keep the pre-trim figure beside it rather than leaving a number that
    # no table in the dataset agrees with.
    # trajectory_km is the dist_m *span*, not its final value: the source
    # dataset can start mid-track with a non-zero dist_m, and rebuild_gps_rows
    # rebases to zero. Taking the span holds either way.
    new_dists = [float(r["dist_m"]) for r in new_gps]
    if "trajectory_km" in meta:
        meta.setdefault("trajectory_km_before_trim", meta["trajectory_km"])
    meta["trajectory_km"] = round((new_dists[-1] - new_dists[0]) / 1000.0, 3)

    ranges = []
    start = keep[0]
    for a, b in zip(keep, keep[1:]):
        if b != a + 1:
            ranges.append([start, a + 1])
            start = b
    ranges.append([start, keep[-1] + 1])
    record = {
        "trim_kind": kind,
        "n_before": n,
        "n_after": len(keep),
        "reason": reason,
        "trimmed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "trim_dir": trim_dir,
        "csv_backup_prefix": f"{trim_dir}/{stamp}.",
        "dropped_frames_csv": f"{trim_dir}/dropped_frames.csv",
        "generator": "farfield/dataset_tools/trim_dataset.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
    }
    # A density trim keeps a frame every few dropped ones, so the run-length
    # encoding that describes a range trim in a line or two would run to
    # hundreds of one-element ranges here and bury the rest of the metadata.
    # dropped_frames.csv already holds every original index; point at it.
    if len(ranges) > MAX_RECORDED_RANGES:
        record["kept_original_ranges"] = (
            f"{len(ranges)} runs — too many to inline; every dropped index is "
            f"in {trim_dir}/dropped_frames.csv, and extraction_log's "
            f"sequence_position still carries each kept frame's original "
            f"index")
    else:
        record["kept_original_ranges"] = ranges
    if video_fps:
        record["video_fps_used_for_review"] = video_fps
    meta.setdefault("trims", []).append(record)

    # A mount offset measured on the untrimmed track no longer describes this
    # one -- a range trim exists precisely because part of the track behaved
    # differently. Keep the number for reference but make it unusable until it
    # is re-measured, rather than silently handing a stale angle downstream.
    #
    # A density trim is the exception: it drops no stretch of track and
    # changes no geometry, and the offset is an angle between the camera and
    # the direction of travel that does not depend on how often that travel
    # was sampled. Invalidating it here would be a false alarm, and one that
    # costs a re-calibration.
    if isinstance(meta.get("mount_offset"), dict) and kind != "density":
        meta["mount_offset"]["self_consistent"] = False
        meta["mount_offset"]["stale_after_trim"] = True

    # recording_seams are keyed on frame indices, which this trim just
    # renumbered — for EVERY trim kind, density included (a density trim keeps
    # the geometry but still renumbers). Both homes of the record are
    # rebased: the legacy in-metadata block and the _manifests/ sidecar the
    # ported annotate_recording_seams writes.
    n_rebased_homes = 0
    if isinstance(meta.get("recording_seams"), dict):
        meta["recording_seams"] = rebase_seams(meta["recording_seams"], keep,
                                               kind)
        n_rebased_homes += 1
    seams_sidecar = ds / SEAMS_SIDECAR
    if seams_sidecar.exists():
        seams_sidecar.write_text(json.dumps(
            rebase_seams(json.loads(seams_sidecar.read_text()), keep, kind),
            indent=1) + "\n")
        n_rebased_homes += 1

    json.dump(meta, open(meta_path, "w"), indent=2)
    (backup_dir / "trim_note.json").write_text(json.dumps({
        "dropped": len(dropped),
        "trim_kind": kind,
        "reason": reason,
        "generator": "farfield/dataset_tools/trim_dataset.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "trimmed_at": record["trimmed_at"],
        "restore": "dropped_frames.csv maps every file here back to its "
                   "original index; the pre-trim CSVs are the "
                   f"{stamp}.*.csv copies in this directory",
    }, indent=1))
    return {"moved": moved, "renamed": renamed, "backup": str(backup_dir),
            "kept_ranges": ranges, "n_dropped": len(dropped),
            "n_seam_records_rebased": n_rebased_homes}


def regenerate_views(ds: Path, fps: int, max_frames: int | None):
    """Re-render trajectory.png and gps_timelapse.mp4 so review sees the trim.

    Stale views are worse than absent ones here: the whole point of the trim
    is that someone watched the video and objected to part of it, so leaving
    the old video in place invites a second reviewer to re-report a fault that
    has already been cut out.
    """
    from experimental.overhead_matching.swag.farfield.dataset_tools import (
        make_dataset_timelapse as tl,
    )

    paths, lats, lons, times, dists = tl.load_frames(ds)
    if len(paths) < 2:
        print(f"  WARNING: {len(paths)} frames left; skipping view "
              f"regeneration")
        return False
    out_dir = tl.view_output_dir(ds)
    tl.stage_plot(ds, lats, lons, times, dists, out_dir / tl.TRAJECTORY_NAME)
    tl.stage_video(paths, lats, lons, out_dir / tl.TIMELAPSE_NAME,
                   1280, fps, max_frames or None)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--keep", nargs="*", default=[],
                   help="original index ranges to KEEP, e.g. 0:165 400:")
    p.add_argument("--drop", nargs="*", default=[],
                   help="original index ranges to DROP (complement of --keep)")
    p.add_argument("--spacing_m", type=float, default=None,
                   help="thin the trajectory to this along-track spacing "
                        "instead of cutting ranges")
    p.add_argument("--trim_dir", default=None,
                   help="directory inside the dataset for dropped frames and "
                        "pre-trim CSVs (default: trimmed_frames, or "
                        "trimmed_frames_for_density with --spacing_m). Give "
                        "each kind of trim its own so the reason survives")
    p.add_argument("--reason", default="",
                   help="recorded in pipeline_metadata.json")
    p.add_argument("--video_fps", type=float, default=None,
                   help="fps of the reviewed timelapse, recorded for "
                        "provenance")
    p.add_argument("--timelapse_fps", type=int, default=15)
    p.add_argument("--timelapse_max_frames", type=int, default=1500)
    p.add_argument("--no_regenerate", action="store_true",
                   help="skip re-rendering trajectory.png / gps_timelapse.mp4")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)

    ds = args.dataset_path
    gps_rows, _ = read_csv(ds / "frames_gps.csv")
    log_rows, _ = read_csv(ds / "extraction_log.csv")
    n = len(gps_rows)
    if len(log_rows) != n:
        print(f"ERROR: frames_gps has {n} rows, extraction_log has "
              f"{len(log_rows)}")
        return 1

    if args.spacing_m is not None:
        if args.keep or args.drop:
            print("ERROR: --spacing_m selects frames by distance; it cannot "
                  "be combined with --keep/--drop. Run them as separate "
                  "trims.")
            return 1
        kind = "density"
        keep = keep_by_spacing(gps_rows, args.spacing_m)
    else:
        kind = "range"
        keep = keep_indices(args.keep, args.drop, n)
    trim_dir = args.trim_dir or (
        "trimmed_frames_for_density" if kind == "density"
        else "trimmed_frames")

    before = summarize(gps_rows, log_rows, list(range(n)))
    after = summarize(gps_rows, log_rows, keep)

    print(f"{ds.name}: {before['n']} -> {after['n']} frames "
          f"({before['n'] - after['n']} dropped, {kind} trim -> {trim_dir}/)")
    print(f"  distance  {before['dist_km']:.2f} km -> "
          f"{after['dist_km']:.2f} km")
    print(f"  spacing   {before['dist_km'] * 1000 / before['n']:.1f} -> "
          f"{after['dist_km'] * 1000 / after['n']:.1f} m/frame")
    print(f"  duration  {before['dur_min']:.1f} min -> "
          f"{after['dur_min']:.1f} min")
    print(f"  sequences {before['n_sequences']} -> {after['n_sequences']}")
    print(f"  window    {after['start_utc']} .. {after['end_utc']} UTC")
    if kind == "density":
        steps = [float(gps_rows[b]["dist_m"]) - float(gps_rows[a]["dist_m"])
                 for a, b in zip(keep, keep[1:])]
        steps.sort()
        print(f"  gaps      min {steps[0]:.1f} m, median "
              f"{steps[len(steps) // 2]:.1f} m, max {steps[-1]:.1f} m")
    if args.video_fps:
        print(f"  kept video window: "
              f"{keep[0] / args.video_fps:.1f} s .. "
              f"{keep[-1] / args.video_fps:.1f} s")

    if args.dry_run:
        print("DRY RUN — nothing written")
        return 0

    info = apply_trim(ds, keep, args.reason, args.video_fps,
                      trim_dir=trim_dir, kind=kind)
    ranges = info["kept_ranges"]
    if len(ranges) > MAX_RECORDED_RANGES:
        print(f"  kept {len(ranges)} original runs (listed per frame in "
              f"{trim_dir}/dropped_frames.csv)")
    else:
        print(f"  kept original ranges {ranges}")
    print(f"  moved {info['moved']} images and renamed {info['renamed']} "
          f"(backups in {info['backup']})")
    if info["n_seam_records_rebased"]:
        print(f"  rebased {info['n_seam_records_rebased']} recording_seams "
              f"record(s) onto the new frame numbering")
    if not args.no_regenerate:
        regenerate_views(ds, args.timelapse_fps, args.timelapse_max_frames)
    n_sums = checksums.regenerate(ds)
    if n_sums:
        print(f"  regenerated {checksums.CHECKSUM_FILE} over {n_sums} files")
    if kind == "density":
        print("  NOTE: mount_offset left as-is — a density trim does not "
              "change the geometry it was measured from.")
    else:
        print("  NOTE: mount_offset is now marked stale — re-run the offset "
              "sweep (farfield/calibration) and publish_mount_offset.")
    meta = json.load(open(ds / "pipeline_metadata.json"))
    if meta.get("projection") == "equirectangular":
        print("  NOTE: equirectangular dataset — pinhole faces reference the "
              "old pano_ids and must be regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
