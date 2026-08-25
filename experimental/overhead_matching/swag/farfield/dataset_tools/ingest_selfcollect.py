"""Convert a self-collected 360 export into the dataset contract.

Self-collected recordings arrive with globally unique frame IDs and a rich
frames_gps.csv. Farfield ingest and `farfield/audit_dataset.py` require the
dataset contract: `f{idx:04d},{lat:.6f},{lon:.6f},.jpg` filenames whose index is
the row index (the ingest join key), five row-aligned tables, and a relative
`panorama/ -> frames` symlink. This script performs that conversion once,
preserving the original IDs and per-frame quality columns in
extraction_log.csv.

Rows with no usable position (empty latitude/longitude) cannot even be named
under the filename convention, so they are dropped here — parked with their
source CSV rows in `trimmed_frames/`, the same reversible-audit-trail
convention `trim_dataset.py` uses.

GPS course over ground is retained explicitly as diagnostic data in
``frames_gps.csv:gps_course_deg``. It is not camera heading and this ingest
never rotates it into the camera frame. The intrinsics heading columns remain
present for the common table shape but are deliberately empty; only a
separate, human-approved nominal-forward record may rotate localization
bearings.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:ingest_selfcollect -- \\
        --source_dir /path/to/source_recording \\
        --frames_subdir frames_2s \\
        --output /path/to/dataset \\
        --dataset_id charles_river_20260727 \\
        --width 7680 --height 3840 \\
        --raw_material raw_material/charles_river_20260727 \\
        --dry_run
"""

import argparse
import csv
import datetime
import fcntl
import json
import math
import os
import shutil
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import provenance

# frames_gps.csv contract columns (audit_dataset checks idx/video_t_s/dist_m;
# ingest joins on idx). Everything else from the source CSV goes to
# extraction_log.csv.
GPS_CONTRACT = ["idx", "video_t_s", "sensor_elapsed_s", "dist_m",
                "latitude", "longitude", "altitude_m", "speed_mps",
                "gps_course_deg", "frame_file"]

INTRINSICS_COLS = ["idx", "pano_id", "projection", "width", "height",
                   "focal_norm", "k1", "k2", "hfov_deg", "vfov_deg",
                   "computed_compass_angle_true_deg",
                   "compass_angle_true_deg",
                   "heading_optical_axis_true_deg",
                   "heading_column0_true_deg", "selected_heading_source",
                   "focal_source"]

EXTLOG_FIXED = ["frame_idx", "pano_id", "sequence_id", "sequence_position",
                "camera_type", "geometry_source", "lat", "lng", "heading_used",
                "captured_at", "original_path", "output_filename"]


def read_rows(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_rows(path: Path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def publish_dataset(staging: Path, output: Path) -> None:
    """Publish a fully staged dataset with a locked, no-clobber rename."""
    if staging.parent.resolve() != output.parent.resolve():
        raise ValueError("staging and output must be sibling directories")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    parent_fd = os.open(output.parent, flags)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace existing output: {output}")
        os.rename(staging, output)
        os.fsync(parent_fd)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        finally:
            os.close(parent_fd)


def fill_gps_course_from_track(kept: list) -> None:
    """Fill diagnostic ``gps_course_deg`` from consecutive GPS positions.

    Self-collected logs without a per-fix course (boston legs) get the bearing
    of the surviving track instead. This remains course over ground: it makes
    no claim about camera heading or the platform's nominal-forward axis.
    """
    for i, r in enumerate(kept):
        a = kept[max(0, i - 1) if i == len(kept) - 1 else i]
        b = kept[min(len(kept) - 1, i + 1)]
        la1 = math.radians(float(a["latitude"]))
        lo1 = math.radians(float(a["longitude"]))
        la2 = math.radians(float(b["latitude"]))
        lo2 = math.radians(float(b["longitude"]))
        dlon = lo2 - lo1
        brg = math.degrees(math.atan2(
            math.sin(dlon) * math.cos(la2),
            math.cos(la1) * math.sin(la2)
            - math.sin(la1) * math.cos(la2) * math.cos(dlon))) % 360.0
        r["gps_course_deg"] = f"{brg:.4f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_dir", type=Path, required=True,
                   help="unpacked collect export containing frames + "
                        "frames_gps.csv")
    p.add_argument("--frames_subdir", default="frames",
                   help="image directory inside source_dir")
    p.add_argument("--gps_csv", default="frames_gps.csv",
                   help="per-frame CSV inside source_dir")
    p.add_argument("--output", type=Path, required=True,
                   help="dataset directory to create (datasets/<dataset_id>)")
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--raw_material", default=None,
                   help="raw_material/<id> this dataset was produced from")
    p.add_argument("--log_start_utc", default=None,
                   help="ISO UTC start of the sensor log, e.g. "
                        "2026-07-27T21:22:04Z; used with sensor_elapsed_s to "
                        "fill captured_at (epoch ms)")
    p.add_argument("--coord_decimals", type=int, default=6,
                   help="lat/lon decimals in frame filenames. Mapillary "
                        "datasets use 6; boston_harbor_leg1's pre-existing "
                        "stems (which pinhole dirs and embeddings.pkl key on) "
                        "use 7")
    p.add_argument("--extra_metadata", type=Path, default=None,
                   help="JSON merged into pipeline_metadata.json (extras win)")
    p.add_argument("--dry_run", action="store_true")
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    out = args.output
    staging = out.with_name(out.name + ".incomplete")
    if out.exists() or out.is_symlink():
        p.error(f"--output already exists; choose a new dataset path: {out}")
    if staging.exists() or staging.is_symlink():
        p.error(f"incomplete output already exists; inspect or remove it: "
                f"{staging}")
    prospective_output = out.parent.resolve() / out.name
    source = args.source_dir.resolve()
    source_frames = (args.source_dir / args.frames_subdir).resolve()
    if (prospective_output == source
            or prospective_output == source_frames
            or prospective_output.is_relative_to(source_frames)):
        p.error("--output may not replace --source_dir or its frame directory")

    extra_metadata = None
    if args.extra_metadata:
        extra_metadata = json.loads(args.extra_metadata.read_text())
        if not isinstance(extra_metadata, dict):
            p.error("--extra_metadata must contain a JSON object")
        forbidden = sorted(set(extra_metadata) & {
            "azimuth_convention",
            "heading_note",
            "heading_reliable",
            "heading_source",
            "mount_offset",
        })
        if forbidden:
            p.error("--extra_metadata may not replace orientation authority "
                    f"fields: {forbidden}")

    frames_dir = args.source_dir / args.frames_subdir
    rows, src_cols = read_rows(args.source_dir / args.gps_csv)
    if not rows:
        sys.exit("empty frames_gps.csv")
    required_cols = {"frame_file", "latitude", "longitude", "video_t_s"}
    missing_cols = sorted(required_cols - set(src_cols or []))
    if missing_cols:
        sys.exit(f"frames_gps.csv is missing required columns: {missing_cols}")
    names = [r["frame_file"] for r in rows]
    if any(not name or Path(name).name != name for name in names):
        sys.exit("frame_file values must be non-empty basenames")
    if len(names) != len(set(names)):
        sys.exit("frame_file values must be unique")

    kept = [r for r in rows
            if r["latitude"].strip() and r["longitude"].strip()]
    dropped = [r for r in rows
               if not (r["latitude"].strip() and r["longitude"].strip())]
    if not kept:
        sys.exit("no rows have a usable latitude/longitude")
    for row in kept:
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            video_t_s = float(row["video_t_s"])
        except (TypeError, ValueError) as exc:
            sys.exit(f"invalid numeric GPS row for {row['frame_file']}: {exc}")
        if not (math.isfinite(lat) and -90 <= lat <= 90
                and math.isfinite(lon) and -180 <= lon <= 180
                and math.isfinite(video_t_s)):
            sys.exit(f"non-finite or out-of-range GPS row for "
                     f"{row['frame_file']}")

    course_from_track = "course_deg" not in src_cols
    if course_from_track:
        fill_gps_course_from_track(kept)
    else:
        for row in kept:
            raw_course = (row.get("course_deg") or "").strip()
            row["gps_course_deg"] = (
                "" if not raw_course else f"{float(raw_course) % 360.0:.4f}")

    missing = [r["frame_file"] for r in rows
               if not (frames_dir / r["frame_file"]).is_file()
               or (frames_dir / r["frame_file"]).is_symlink()]
    if missing:
        sys.exit(f"{len(missing)} frame files in CSV not found on disk: "
                 f"{missing[:3]}")

    log_start = None
    if args.log_start_utc:
        try:
            log_start = datetime.datetime.fromisoformat(
                args.log_start_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            sys.exit(f"invalid --log_start_utc: {exc}")
        if log_start.tzinfo is None:
            sys.exit("--log_start_utc must include a timezone")

    id_width = max(4, len(str(len(kept) - 1)))
    print(f"{len(rows)} source rows -> keeping {len(kept)}, "
          f"dropping {len(dropped)} with no usable position")
    dec = args.coord_decimals
    if args.dry_run:
        r = kept[0]
        name = (f"f{0:0{id_width}d},{float(r['latitude']):.{dec}f},"
                f"{float(r['longitude']):.{dec}f},.jpg")
        print(f"dry run: first kept frame {r['frame_file']} -> frames/{name}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    work = staging
    (work / "frames").mkdir()
    trimmed = work / "trimmed_frames"

    gps_course_source = (
        "derived_from_positions" if course_from_track
        else "source_course_over_ground")

    gps_rows, mapping_rows, log_rows, intr_rows = [], [], [], []
    for idx, r in enumerate(kept):
        pano_id = f"f{idx:0{id_width}d}"
        lat, lon = float(r["latitude"]), float(r["longitude"])
        name = f"{pano_id},{lat:.{dec}f},{lon:.{dec}f},.jpg"
        shutil.copy2(frames_dir / r["frame_file"], work / "frames" / name)

        gps_rows.append({
            "idx": idx, "video_t_s": r["video_t_s"],
            "sensor_elapsed_s": r.get("sensor_elapsed_s", ""),
            "dist_m": r.get("dist_m", ""), "latitude": r["latitude"],
            "longitude": r["longitude"],
            "altitude_m": r.get("altitude_m", ""),
            "speed_mps": r.get("speed_mps", ""),
            "gps_course_deg": r.get("gps_course_deg", ""),
            "frame_file": name,
        })
        mapping_rows.append({"pano_id": pano_id, "lat": r["latitude"],
                             "lon": r["longitude"], "filename": name})

        captured_at = ""
        if log_start is not None and r.get("sensor_elapsed_s", "").strip():
            captured_at = int((log_start + datetime.timedelta(
                seconds=float(r["sensor_elapsed_s"]))).timestamp() * 1000)
        log_row = {
            "frame_idx": idx, "pano_id": pano_id,
            "sequence_id": args.dataset_id, "sequence_position": idx,
            "camera_type": "equirectangular",
            "geometry_source": r.get("gps_quality", "sensor_logger"),
            "lat": r["latitude"], "lng": r["longitude"],
            "heading_used": "",
            "captured_at": captured_at,
            "original_path": str(frames_dir / r["frame_file"]),
            "output_filename": name,
        }
        # Preserve every source column the contract tables don't carry.
        for col in src_cols:
            if col not in GPS_CONTRACT and col not in log_row:
                log_row[col] = r.get(col, "")
        log_rows.append(log_row)

        intr_rows.append({
            "idx": idx, "pano_id": pano_id, "projection": "equirectangular",
            "width": args.width, "height": args.height,
            "focal_norm": "", "k1": "", "k2": "",
            "hfov_deg": 360.0, "vfov_deg": 180.0,
            "computed_compass_angle_true_deg": "",
            "compass_angle_true_deg": "",
            "heading_optical_axis_true_deg": "",
            "heading_column0_true_deg": "",
            "selected_heading_source": "",
            "focal_source": "n/a",
        })

    write_rows(work / "frames_gps.csv", gps_rows, GPS_CONTRACT)
    write_rows(work / "pano_id_mapping.csv", mapping_rows,
               ["pano_id", "lat", "lon", "filename"])
    extlog_cols = EXTLOG_FIXED + [
        c for c in src_cols
        if c not in GPS_CONTRACT and c not in EXTLOG_FIXED]
    write_rows(work / "extraction_log.csv", log_rows, extlog_cols)
    write_rows(work / "intrinsics.csv", intr_rows, INTRINSICS_COLS)

    if dropped:
        trimmed.mkdir(exist_ok=True)
        for r in dropped:
            shutil.copy2(frames_dir / r["frame_file"],
                         trimmed / r["frame_file"])
        write_rows(trimmed / "dropped_frames_gps.csv", dropped, src_cols)
        (trimmed / "trim_note.json").write_text(json.dumps({
            "dropped": len(dropped),
            "reason": "no usable GPS fix at ingest (empty latitude/longitude; "
                      "gps_quality=unusable in dropped_frames_gps.csv)",
            "generator": "farfield/dataset_tools/ingest_selfcollect.py",
            "git_commit": provenance.git_commit(),
            "argv": list(sys.argv),
        }, indent=1))

    (work / "panorama").symlink_to("frames")

    lats = [float(r["latitude"]) for r in kept]
    lons = [float(r["longitude"]) for r in kept]
    dists = [float(r["dist_m"]) for r in kept if r.get("dist_m", "").strip()]
    meta = {
        "dataset_name": args.dataset_id,
        "source": "self_collect",
        "raw_material": args.raw_material,
        "sequence_id": args.dataset_id,
        "component_sequence_ids": [args.dataset_id],
        "stitched_from_n_sequences": 1,
        "projection": "equirectangular",
        "is_equirectangular": True,
        "north_aligned": False,
        "azimuth_convention": {
            "images_rotated": False,
            "frame": "camera (as captured)",
            "bearing_increases": "left_to_right",
            "camera_frame": geo.CAMERA_FRAME,
        },
        "camera_type": "equirectangular",
        "image_dir": "frames",
        "num_images": len(kept),
        "resolution": f"{args.width}x{args.height}",
        "gps_course_diagnostic": {
            "field": "frames_gps.csv:gps_course_deg",
            "source": gps_course_source,
            "use": "diagnostic_only",
            "note": "course over ground; not camera heading and not "
                    "nominal-forward calibration",
        },
        "bbox": {"south": min(lats), "north": max(lats),
                 "west": min(lons), "east": max(lons)},
        "trajectory_km": (round((dists[-1] - dists[0]) / 1000, 3)
                          if dists else None),
        "ingest": {
            "generator": "farfield/dataset_tools/ingest_selfcollect.py",
            "git_commit": provenance.git_commit(),
            "argv": list(sys.argv),
            "created": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"),
            "source_rows": len(rows),
            "kept": len(kept),
            "dropped_no_gps": len(dropped),
        },
    }
    if extra_metadata:
        meta.update(extra_metadata)
    (work / "pipeline_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    publish_dataset(staging, out)

    if dists:
        print(f"wrote {out}: {len(kept)} frames, {len(dropped)} trimmed, "
              f"{(dists[-1] - dists[0]) / 1000:.2f} km")
    else:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
