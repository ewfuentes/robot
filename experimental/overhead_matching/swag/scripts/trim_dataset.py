"""Cut frame ranges out of a collected dataset, keeping every file consistent.

Visual review of `gps_timelapse.mp4` is what actually decides whether a
trajectory is usable, and it usually condemns a *part* of one: the operator
swings the camera after the first minute, a recording restarts pointing the
other way, the last few frames come from a different day. This trims those
ranges out and leaves a dataset that still satisfies the contract
`audit_dataset.py` enforces.

Trimming is not just dropping CSV rows. The audit requires `frames_gps.idx` to
be 0..N-1 contiguous *and* `pano_id[1:] == idx`, because that equality is the
join key the landmark_filtering ingest uses. Cutting from the middle therefore
forces a renumber, and the renumber forces an image rename, and the rename has
to reach `frames_gps.csv`, `extraction_log.csv`, `intrinsics.csv` and
`pano_id_mapping.csv` together or the dataset silently comes apart. Dropping
rows by hand gets this wrong; that is why this script exists.

Indices in --keep/--drop are ORIGINAL frame indices (half-open, Python slice
semantics), read off the timelapse via `--video_fps` if that is easier:

    # keep only the first 11 seconds of a 15 fps timelapse
    bazel run //experimental/overhead_matching/swag/scripts:trim_dataset -- \
        --dataset_path /data/farfield_matching/mapillary_datasets/fukuoka_yumechan_a \
        --keep 0:165 --reason "operator swings the camera after t=11 s" --dry_run

Dropped images and the original CSVs move to `trimmed_frames/` inside the
dataset, so a trim is reversible and costs no extra disk.
"""

import argparse
import csv
import datetime
import json
import math
import shutil
import sys
from pathlib import Path

R_EARTH_M = 6371000.0
CSV_NAMES = ["frames_gps.csv", "extraction_log.csv", "intrinsics.csv",
             "pano_id_mapping.csv"]


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH_M * math.asin(math.sqrt(a))


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

    Only the pano_id field changes. The audit cross-checks the lat/lon fields of
    the name against frames_gps, so they must survive the rename untouched.
    """
    parts = old_name.split(",")
    parts[0] = f"f{new_idx:04d}"
    return ",".join(parts)


def summarize(gps_rows, log_rows, keep):
    """Distance, duration and sequence make-up of a kept subset."""
    lat = [float(gps_rows[i]["latitude"]) for i in keep]
    lon = [float(gps_rows[i]["longitude"]) for i in keep]
    ts = [int(log_rows[i]["captured_at"]) / 1000.0 for i in keep]
    dist = sum(haversine_m(lat[k - 1], lon[k - 1], lat[k], lon[k])
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


def rebuild_gps_rows(gps_rows, log_rows, keep):
    """frames_gps for the kept subset, rebased exactly as the converter builds it.

    video_t_s restarts at the new first frame and dist_m re-accumulates, so the
    trimmed dataset is indistinguishable from one collected over that range.
    Across a cut the cumulative distance still adds the straight-line jump,
    which is the same thing the converter does at a stitch seam.
    """
    out = []
    t0 = int(log_rows[keep[0]]["captured_at"])
    cumulative_m = 0.0
    for new_idx, old in enumerate(keep):
        row = dict(gps_rows[old])
        if new_idx > 0:
            prev = gps_rows[keep[new_idx - 1]]
            step = haversine_m(float(prev["latitude"]), float(prev["longitude"]),
                               float(row["latitude"]), float(row["longitude"]))
            cumulative_m += step
            dt = (int(log_rows[old]["captured_at"])
                  - int(log_rows[keep[new_idx - 1]]["captured_at"])) / 1000.0
            speed = round(step / dt, 3) if dt > 0 else -1.0
        else:
            speed = -1.0
        t_s = round((int(log_rows[old]["captured_at"]) - t0) / 1000.0, 3)
        row["idx"] = new_idx
        row["video_t_s"] = t_s
        row["sensor_elapsed_s"] = t_s
        row["dist_m"] = round(cumulative_m, 1)
        row["speed_mps"] = speed
        row["frame_file"] = new_frame_file(row["frame_file"], new_idx)
        out.append(row)
    return out


def apply_trim(ds: Path, keep, reason: str, video_fps: float | None):
    tables = {name: read_csv(ds / name) for name in CSV_NAMES}
    gps_rows, gps_fields = tables["frames_gps.csv"]
    log_rows, log_fields = tables["extraction_log.csv"]
    n = len(gps_rows)
    keep_set = set(keep)

    backup_dir = ds / "trimmed_frames"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in CSV_NAMES:
        shutil.copy2(ds / name, backup_dir / f"{stamp}.{name}")

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

    new_gps = rebuild_gps_rows(gps_rows, log_rows, keep)
    write_csv(ds / "frames_gps.csv", new_gps, gps_fields)

    new_log = []
    for new_idx, old in enumerate(keep):
        row = dict(log_rows[old])
        row["frame_idx"] = new_idx
        row["pano_id"] = f"f{new_idx:04d}"
        row["output_filename"] = new_frame_file(row["output_filename"], new_idx)
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

    ranges = []
    start = keep[0]
    for a, b in zip(keep, keep[1:]):
        if b != a + 1:
            ranges.append([start, a + 1])
            start = b
    ranges.append([start, keep[-1] + 1])
    record = {
        "kept_original_ranges": ranges,
        "n_before": n,
        "n_after": len(keep),
        "reason": reason,
        "trimmed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "csv_backup_prefix": f"trimmed_frames/{stamp}.",
    }
    if video_fps:
        record["video_fps_used_for_review"] = video_fps
    meta.setdefault("trims", []).append(record)

    # A mount offset measured on the untrimmed track no longer describes this
    # one -- the trim exists precisely because part of the track behaved
    # differently. Keep the number for reference but make it unusable until it
    # is re-measured, rather than silently handing a stale angle downstream.
    if isinstance(meta.get("mount_offset"), dict):
        meta["mount_offset"]["usable"] = False
        meta["mount_offset"]["stale_after_trim"] = True

    json.dump(meta, open(meta_path, "w"), indent=2)
    return {"moved": moved, "renamed": renamed, "backup": str(backup_dir),
            "kept_ranges": ranges}


def regenerate_views(ds: Path, fps: int, max_frames: int):
    """Re-render trajectory.png and gps_timelapse.mp4 so review sees the trim.

    Stale views are worse than absent ones here: the whole point of the trim is
    that someone watched the video and objected to part of it, so leaving the
    old video in place invites a second reviewer to re-report a fault that has
    already been cut out.
    """
    # Under bazel the sibling script is a runfile, not an importable top-level
    # module, so put its directory on the path rather than relying on cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_dataset_timelapse as tl

    paths, lats, lons, times, dists = tl.load_frames(ds)
    if len(paths) < 2:
        print(f"  WARNING: {len(paths)} frames left; skipping view regeneration")
        return False
    tl.stage_plot(ds, lats, lons, times, dists, ds / "trajectory.png")
    tl.stage_video(paths, lats, lons, ds / "gps_timelapse.mp4",
                   1280, fps, max_frames or None)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--keep", nargs="*", default=[],
                   help="original index ranges to KEEP, e.g. 0:165 400:")
    p.add_argument("--drop", nargs="*", default=[],
                   help="original index ranges to DROP (complement of --keep)")
    p.add_argument("--reason", default="", help="recorded in pipeline_metadata.json")
    p.add_argument("--video_fps", type=float, default=None,
                   help="fps of the reviewed timelapse, recorded for provenance")
    p.add_argument("--timelapse_fps", type=int, default=15)
    p.add_argument("--timelapse_max_frames", type=int, default=1500)
    p.add_argument("--no_regenerate", action="store_true",
                   help="skip re-rendering trajectory.png / gps_timelapse.mp4")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ds = args.dataset_path
    gps_rows, _ = read_csv(ds / "frames_gps.csv")
    log_rows, _ = read_csv(ds / "extraction_log.csv")
    n = len(gps_rows)
    if len(log_rows) != n:
        print(f"ERROR: frames_gps has {n} rows, extraction_log has {len(log_rows)}")
        return 1

    keep = keep_indices(args.keep, args.drop, n)
    before = summarize(gps_rows, log_rows, list(range(n)))
    after = summarize(gps_rows, log_rows, keep)

    print(f"{ds.name}: {before['n']} -> {after['n']} frames "
          f"({before['n'] - after['n']} dropped)")
    print(f"  distance  {before['dist_km']:.2f} km -> {after['dist_km']:.2f} km")
    print(f"  duration  {before['dur_min']:.1f} min -> {after['dur_min']:.1f} min")
    print(f"  sequences {before['n_sequences']} -> {after['n_sequences']}")
    print(f"  window    {after['start_utc']} .. {after['end_utc']} UTC")
    if args.video_fps:
        print(f"  kept video window: "
              f"{keep[0] / args.video_fps:.1f} s .. {keep[-1] / args.video_fps:.1f} s")

    if args.dry_run:
        print("DRY RUN — nothing written")
        return 0

    info = apply_trim(ds, keep, args.reason, args.video_fps)
    print(f"  kept original ranges {info['kept_ranges']}")
    print(f"  moved {info['moved']} images and renamed {info['renamed']} "
          f"(backups in {info['backup']})")
    if not args.no_regenerate:
        regenerate_views(ds, args.timelapse_fps, args.timelapse_max_frames)
    print("  NOTE: mount_offset is now marked stale — rerun calibrate_mount_offset.")
    meta = json.load(open(ds / "pipeline_metadata.json"))
    if meta.get("projection") == "equirectangular":
        print("  NOTE: equirectangular dataset — pinhole faces reference the old "
              "pano_ids and must be regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
