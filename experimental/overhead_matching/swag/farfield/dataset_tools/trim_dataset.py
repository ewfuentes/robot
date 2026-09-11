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
name the frame each odometry break sits after, so this trim rebases them (see
`rebase_seams`).

It finishes by regenerating `checksums.sha256` through the shared
implementation in `checksums.py`.

Indices in --keep/--drop are ORIGINAL frame indices (half-open, Python slice
semantics), read off the timelapse via `--video_fps` if that is easier:

    # keep only the first 11 seconds of a 15 fps timelapse
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:trim_dataset -- \\
        --dataset_path /path/to/dataset \\
        --keep 0:165 --reason "operator swings the camera after t=11 s" --dry_run

    # thin a 3 m collect to a 10 m grid, into its own trim directory
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:trim_dataset -- \\
        --dataset_path /path/to/dataset \\
        --spacing_m 10 --trim_dir trimmed_frames_for_density \\
        --reason "3.1 m/frame is ~10x denser than far-field extraction needs"

Dropped images and the original CSVs move to `--trim_dir` inside the dataset,
so a trim is reversible and costs no extra disk. Give each trim invocation its
own directory: publication is no-clobber, so a later trim cannot replace or
append to the first trim's restore material.
"""

import argparse
import bisect
import csv
import datetime
import json
import os
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
    and `sensor_elapsed_s` remain unchanged because they address the source
    video and sensor log, neither of which the trim rewrites.

    Where dist_m comes from depends on what was cut. A *range* trim removes
    stretches of track, so the distance must be re-derived from the surviving
    positions -- carrying the original column forward would keep charging for
    the removed stretch. Across a cut that adds the straight-line jump, which
    is what the converter does at a stitch seam.

    A *density* trim removes no track, so the original smoothed/bridged distance
    column remains authoritative and is reused after rebasing. Re-deriving it
    from raw reported positions would introduce position noise.
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


TRIM_REQUIRED_COLUMNS = {
    "frames_gps.csv": {"idx", "video_t_s", "dist_m", "latitude",
                       "longitude", "frame_file"},
    "extraction_log.csv": {"frame_idx", "pano_id", "sequence_id",
                           "geometry_source", "captured_at",
                           "output_filename"},
    "intrinsics.csv": {"idx", "pano_id"},
    "pano_id_mapping.csv": {"pano_id", "filename"},
}


def preflight_trim(ds: Path, keep, trim_dir: str,
                   kind: str = "range") -> dict:
    """Resolve and validate every trim input and destination without writes."""
    ds = Path(ds)
    if ds.is_symlink() or not ds.is_dir():
        raise ValueError(f"dataset must be a regular directory: {ds}")
    trim_path = Path(trim_dir)
    if (not trim_dir or trim_path.is_absolute() or len(trim_path.parts) != 1
            or trim_path.name in (".", "..")):
        raise ValueError("trim_dir must be one path-free directory name")
    backup_dir = ds / trim_dir
    if backup_dir.exists() or backup_dir.is_symlink():
        raise FileExistsError(
            f"trim output already exists; choose a new --trim_dir: {backup_dir}")
    staging = ds.parent / f".{ds.name}.trim_dataset.incomplete"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(
            f"incomplete trim transaction exists; inspect or remove: {staging}")

    tables = {}
    for name in CSV_NAMES:
        rows, fields = read_csv(ds / name)
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"{name} has missing or duplicate column names")
        missing = sorted(TRIM_REQUIRED_COLUMNS[name] - set(fields))
        if missing:
            raise ValueError(f"{name} is missing required columns {missing}")
        tables[name] = (rows, fields)
    gps_rows, _ = tables["frames_gps.csv"]
    n = len(gps_rows)
    if n == 0:
        raise ValueError("cannot trim an empty dataset")
    for name, (rows, _) in tables.items():
        if len(rows) != n:
            raise ValueError(
                f"{name} has {len(rows)} rows; expected {n} row-aligned rows")
    keep = list(keep)
    if (not keep or any(type(index) is not int for index in keep)
            or keep != sorted(set(keep))
            or keep[0] < 0 or keep[-1] >= n):
        raise ValueError(
            f"keep indices must be a non-empty sorted unique subset of 0:{n}")

    log_rows, _ = tables["extraction_log.csv"]
    intr_rows, _ = tables["intrinsics.csv"]
    map_rows, _ = tables["pano_id_mapping.csv"]
    frame_names = []
    for index in range(n):
        pano_id = f"f{index:04d}"
        try:
            indices = (int(gps_rows[index]["idx"]),
                       int(log_rows[index]["frame_idx"]),
                       int(intr_rows[index]["idx"]))
            float(gps_rows[index]["video_t_s"])
            float(gps_rows[index]["dist_m"])
            float(gps_rows[index]["latitude"])
            float(gps_rows[index]["longitude"])
            int(log_rows[index]["captured_at"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {index} has an invalid typed value: {exc}") from exc
        if indices != (index, index, index):
            raise ValueError(
                f"row {index} index join mismatch: gps/log/intrinsics={indices}")
        panos = (log_rows[index]["pano_id"], intr_rows[index]["pano_id"],
                 map_rows[index]["pano_id"])
        if panos != (pano_id, pano_id, pano_id):
            raise ValueError(
                f"row {index} pano_id join mismatch: expected {pano_id}, got {panos}")
        names = (gps_rows[index]["frame_file"],
                 log_rows[index]["output_filename"],
                 map_rows[index]["filename"])
        if len(set(names)) != 1:
            raise ValueError(f"row {index} frame filename join mismatch: {names}")
        name = names[0]
        if not name or Path(name).name != name:
            raise ValueError(f"row {index} frame_file must be a safe basename")
        frame_names.append(name)
    if len(frame_names) != len(set(frame_names)):
        raise ValueError("frame_file join keys must be unique")

    panorama = ds / "panorama"
    frames_dir = panorama.resolve()
    if (not frames_dir.is_dir()
            or frames_dir.parent != ds.resolve()
            or frames_dir == ds.resolve()):
        raise ValueError(
            "panorama must resolve to a direct child directory of the dataset")
    for name in frame_names:
        path = frames_dir / name
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"recorded frame is not a regular file: {path}")

    meta_path = ds / "pipeline_metadata.json"
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError("pipeline_metadata.json must contain a JSON object")
    if "trims" in meta and not isinstance(meta["trims"], list):
        raise ValueError("pipeline_metadata.json trims must be a JSON list")
    if ("recording_seams" in meta
            and not isinstance(meta["recording_seams"], dict)):
        raise ValueError(
            "pipeline_metadata.json recording_seams must be a JSON object")
    if isinstance(meta.get("recording_seams"), dict):
        rebase_seams(meta["recording_seams"], keep, kind)

    from experimental.overhead_matching.swag.farfield.dataset_tools import (
        make_dataset_timelapse as tl,
    )
    tl.reject_legacy_views(ds)
    timelapse = tl.view_output_dir(ds)
    timelapse_incomplete = timelapse.with_name(
        timelapse.name + ".incomplete")
    if timelapse_incomplete.exists() or timelapse_incomplete.is_symlink():
        raise FileExistsError(
            "incomplete timelapse blocks a trim transaction: "
            f"{timelapse_incomplete}")
    has_timelapse = timelapse.exists() or timelapse.is_symlink()
    if has_timelapse:
        tl.validate_completed(ds)

    seams_path = ds / SEAMS_SIDECAR
    seams = None
    if seams_path.exists():
        if seams_path.is_symlink() or not seams_path.is_file():
            raise ValueError(f"recording seams sidecar is not regular: {seams_path}")
        seams = json.loads(seams_path.read_text())
        if not isinstance(seams, dict):
            raise ValueError("recording_seams.json must contain a JSON object")
        rebase_seams(seams, keep, kind)

    generated_names = [
        new_frame_file(gps_rows[old]["frame_file"], new_index)
        for new_index, old in enumerate(keep)
    ]
    if (len(generated_names) != len(set(generated_names))
            or any(Path(name).name != name for name in generated_names)):
        raise ValueError("trim would produce duplicate or unsafe frame names")

    # Exercise every result-shaping calculation now. Any malformed retained
    # row fails before a frame, table, checksum, or archive path is touched.
    rebuild_gps_rows(gps_rows, log_rows, keep, kind)
    return {
        "dataset": ds,
        "backup_dir": backup_dir,
        "staging": staging,
        "frames_dir": frames_dir,
        "tables": tables,
        "metadata": meta,
        "seams": seams,
        "timelapse": timelapse,
        "has_timelapse": has_timelapse,
        "keep": keep,
    }


def apply_trim(ds: Path, keep, reason: str, video_fps: float | None, *,
               trim_dir: str = "trimmed_frames", kind: str = "range",
               regenerate=None):
    plan = preflight_trim(ds, keep, trim_dir, kind)
    ds = plan["dataset"]
    keep = plan["keep"]
    tables = plan["tables"]
    gps_rows, gps_fields = tables["frames_gps.csv"]
    log_rows, log_fields = tables["extraction_log.csv"]
    n = len(gps_rows)
    keep_set = set(keep)
    backup_dir = plan["backup_dir"]
    staging = plan["staging"]
    frames_dir = plan["frames_dir"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dropped = [i for i in range(n) if i not in keep_set]
    staging.mkdir()
    archive = staging / "archive"
    new_files = staging / "new_files"
    new_frames = staging / "new_frames"
    rollback = staging / "rollback"
    for path in (archive, new_files, new_frames, rollback):
        path.mkdir()

    for name in CSV_NAMES:
        shutil.copy2(ds / name, archive / f"{stamp}.{name}")
    write_dropped_csv(archive, gps_rows, log_rows, dropped, kind, reason,
                      stamp)
    for old in dropped:
        source = frames_dir / gps_rows[old]["frame_file"]
        os.link(source, archive / source.name, follow_symlinks=False)
    renamed = 0
    for new_idx, old in enumerate(keep):
        old_name = gps_rows[old]["frame_file"]
        new_name = new_frame_file(old_name, new_idx)
        os.link(frames_dir / old_name, new_frames / new_name,
                follow_symlinks=False)
        renamed += old_name != new_name

    new_gps = rebuild_gps_rows(gps_rows, log_rows, keep, kind)
    write_csv(new_files / "frames_gps.csv", new_gps, gps_fields)

    new_log = []
    for new_idx, old in enumerate(keep):
        row = dict(log_rows[old])
        row["frame_idx"] = new_idx
        row["pano_id"] = f"f{new_idx:04d}"
        row["output_filename"] = new_frame_file(row["output_filename"],
                                                new_idx)
        new_log.append(row)
    write_csv(new_files / "extraction_log.csv", new_log, log_fields)

    intr_rows, intr_fields = tables["intrinsics.csv"]
    new_intr = []
    for new_idx, old in enumerate(keep):
        row = dict(intr_rows[old])
        row["idx"] = new_idx
        row["pano_id"] = f"f{new_idx:04d}"
        new_intr.append(row)
    write_csv(new_files / "intrinsics.csv", new_intr, intr_fields)

    map_rows, map_fields = tables["pano_id_mapping.csv"]
    new_map = []
    for new_idx, old in enumerate(keep):
        row = dict(map_rows[old])
        row["pano_id"] = f"f{new_idx:04d}"
        row["filename"] = new_frame_file(row["filename"], new_idx)
        new_map.append(row)
    write_csv(new_files / "pano_id_mapping.csv", new_map, map_fields)

    meta = plan["metadata"]
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

    # recording_seams are keyed on frame indices, which this trim just
    # renumbered — for EVERY trim kind, density included (a density trim keeps
    # the geometry but still renumbers). Both homes of the record are
    # rebased in both metadata and the derived sidecar.
    n_rebased_homes = 0
    if isinstance(meta.get("recording_seams"), dict):
        meta["recording_seams"] = rebase_seams(meta["recording_seams"], keep,
                                               kind)
        n_rebased_homes += 1
    if plan["seams"] is not None:
        staged_seams = new_files / SEAMS_SIDECAR
        staged_seams.parent.mkdir(parents=True)
        staged_seams.write_text(json.dumps(
            rebase_seams(plan["seams"], keep, kind), indent=1) + "\n")
        n_rebased_homes += 1

    (new_files / "pipeline_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    (archive / "trim_note.json").write_text(json.dumps({
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
    }, indent=1) + "\n")

    changed = [Path(name) for name in CSV_NAMES]
    changed.append(Path("pipeline_metadata.json"))
    if plan["seams"] is not None:
        changed.append(SEAMS_SIDECAR)
    checksum = ds / checksums.CHECKSUM_FILE
    checksum_backup = rollback / checksums.CHECKSUM_FILE
    had_checksum = checksum.is_file() and not checksum.is_symlink()
    if checksum.exists() and not had_checksum:
        raise ValueError(f"checksum manifest is not a regular file: {checksum}")
    if had_checksum:
        shutil.copy2(checksum, checksum_backup)

    moved_originals = []
    archived_timelapse = archive / "pre_trim_timelapse"
    timelapse_moved = False
    frames_original_moved = frames_swapped = archive_published = False
    try:
        if plan["has_timelapse"]:
            os.rename(plan["timelapse"], archived_timelapse)
            timelapse_moved = True
        old_frames = rollback / "frames"
        os.rename(frames_dir, old_frames)
        frames_original_moved = True
        os.rename(new_frames, frames_dir)
        frames_swapped = True
        for relative in changed:
            current = ds / relative
            saved = rollback / "files" / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.rename(current, saved)
            moved_originals.append(relative)
            os.rename(new_files / relative, current)
        os.rename(archive, backup_dir)
        archive_published = True
        if regenerate is not None:
            regenerate()
        checksum_count = checksums.regenerate(ds)
    except BaseException:
        if archive_published and backup_dir.exists():
            os.rename(backup_dir, archive)
        if timelapse_moved or not plan["has_timelapse"]:
            failed_review = staging / "failed_timelapse"
            for label, candidate in (
                    ("complete", plan["timelapse"]),
                    ("incomplete", plan["timelapse"].with_name(
                        plan["timelapse"].name + ".incomplete"))):
                if candidate.exists() or candidate.is_symlink():
                    failed_review.mkdir(exist_ok=True)
                    os.rename(candidate, failed_review / label)
        failed_files = staging / "failed_files"
        for relative in reversed(moved_originals):
            current = ds / relative
            if current.exists() or current.is_symlink():
                failed = failed_files / relative
                failed.parent.mkdir(parents=True, exist_ok=True)
                os.rename(current, failed)
            saved = rollback / "files" / relative
            if saved.exists() or saved.is_symlink():
                current.parent.mkdir(parents=True, exist_ok=True)
                os.rename(saved, current)
        if frames_swapped and frames_dir.exists():
            os.rename(frames_dir, staging / "failed_frames")
        if frames_original_moved and (rollback / "frames").exists():
            os.rename(rollback / "frames", frames_dir)
        if had_checksum:
            shutil.copy2(checksum_backup, checksum)
        elif checksum.exists() or checksum.is_symlink():
            checksum.unlink()
        if timelapse_moved:
            plan["timelapse"].parent.mkdir(parents=True, exist_ok=True)
            os.rename(archived_timelapse, plan["timelapse"])
        raise

    # The replaced frame directory can now be removed: every kept inode is
    # linked from the new directory and every dropped inode from the archive.
    shutil.rmtree(rollback / "frames")
    shutil.rmtree(staging)
    return {"moved": len(dropped), "renamed": renamed,
            "backup": str(backup_dir),
            "kept_ranges": ranges, "n_dropped": len(dropped),
            "n_seam_records_rebased": n_rebased_homes,
            "checksum_count": checksum_count}


def regenerate_views(ds: Path, fps: int, max_frames: int | None):
    """Re-render trajectory.png and gps_timelapse.mp4 so review sees the trim.

    Views must represent the current frame set so review cannot report an
    already-trimmed interval.
    """
    from experimental.overhead_matching.swag.farfield.dataset_tools import (
        make_dataset_timelapse as tl,
    )

    tl.render(ds, 1280, fps, max_frames or None, False)
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
                        "each trim invocation its own; publication never "
                        "overwrites an existing restore directory")
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
    # Dry runs validate exactly the same complete contract as real runs.
    preflight_trim(ds, keep, trim_dir, kind)

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

    regenerate = (None if args.no_regenerate else
                  lambda: regenerate_views(
                      ds, args.timelapse_fps, args.timelapse_max_frames))
    info = apply_trim(ds, keep, args.reason, args.video_fps,
                      trim_dir=trim_dir, kind=kind,
                      regenerate=regenerate)
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
    n_sums = info["checksum_count"]
    if n_sums:
        print(f"  regenerated {checksums.CHECKSUM_FILE} over {n_sums} files")
    print("  NOTE: the dataset identity and frame set changed; regenerate "
          "dependent artifacts and use a dataset-bound approved "
          "nominal-forward record for the next build.")
    meta = json.load(open(ds / "pipeline_metadata.json"))
    if meta.get("projection") == "equirectangular":
        print("  NOTE: equirectangular dataset — pinhole faces reference the "
              "pre-trim pano_ids and must be regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
