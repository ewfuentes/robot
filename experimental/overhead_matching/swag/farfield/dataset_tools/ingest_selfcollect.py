"""Convert a self-collected 360 export into the dataset contract.

Self-collected recordings (boston harbor legs, charles river sail) come out of
their collect-local pipelines with globally-unique frame ids like
`charles_river_20260727_p00255_t0510.00s.jpg` and a rich frames_gps.csv. The
farfield ingest and `farfield/audit_dataset.py` instead require the dataset
contract: `f{idx:04d},{lat:.6f},{lon:.6f},.jpg` filenames whose numeric part IS
the row index (the ingest join key), five row-aligned tables, and a relative
`panorama/ -> frames` symlink. This script performs that conversion once,
preserving the original ids and per-frame quality columns in
extraction_log.csv.

Rows with no usable position (empty latitude/longitude) cannot even be named
under the filename convention, so they are dropped here — parked with their
source CSV rows in `trimmed_frames/`, the same reversible-audit-trail
convention `trim_dataset.py` uses.

heading_deg (the absolute bearing of pano column 0) is derived from the GPS
course over ground. With `--mount_offset_deg` (quoted in
`geometry.MOUNT_OFFSET_FRAME` — see `geometry.MOUNT_OFFSET_CONVENTION` for
what that means) the offset is folded into heading_deg through the geometry
helpers, and the metadata's mount_offset block records
`applied_to_heading_deg: true` so no consumer applies it twice. Without the
flag, heading_deg is the raw course — a PLACEHOLDER that assumes column 0
faces the direction of travel — and no mount_offset block is written at all:
per `dataset.mount_offset_record`, an absent block is the honest statement
that the mount is uncalibrated. Either way `heading_reliable` is false until
a calibration validates the chain.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:ingest_selfcollect -- \\
        --source_dir /data/farfield_matching/inbox/charles_river_unpack/charles_river_sail_07_27_26 \\
        --frames_subdir frames_2s \\
        --output /data/farfield_matching/datasets/charles_river_20260727 \\
        --dataset_id charles_river_20260727 \\
        --width 7680 --height 3840 \\
        --raw_material raw_material/charles_river_20260727 \\
        --dry_run
"""

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import provenance

# frames_gps.csv contract columns (audit_dataset checks idx/video_t_s/dist_m;
# ingest joins on idx). Everything else from the source CSV goes to
# extraction_log.csv.
GPS_CONTRACT = ["idx", "video_t_s", "sensor_elapsed_s", "dist_m",
                "latitude", "longitude", "altitude_m", "speed_mps",
                "frame_file"]

INTRINSICS_COLS = ["idx", "pano_id", "projection", "width", "height",
                   "focal_norm", "k1", "k2", "hfov_deg", "vfov_deg",
                   "heading_deg", "heading_reference", "heading_source"]

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


def heading_deg_text(course_text: str | None,
                     mount_offset_deg: float | None, pano_w: int) -> str:
    """heading_deg (bearing of pano column 0) for one row, "" without a course.

    THE one heading computation. The original ingest carried two copies — a
    guarded one for extraction_log and an unguarded duplicate for
    intrinsics.csv — and the unguarded one crashed on a blank `course_deg`
    that the guarded copy had already survived. Both tables now go through
    this function.

    With a mount offset (in geometry.MOUNT_OFFSET_FRAME), the bearing of
    column 0 is computed through the geometry helpers: column 0's camera-frame
    azimuth -> body frame via the offset -> world frame via the course. The
    column-0-vs-centre-column half turn falls out of
    `geometry.azimuth_of_pano_column(0)` instead of being hand-typed, so the
    conversion cannot be restated wrongly here. Without an offset, heading is
    the raw course — the uncalibrated placeholder the metadata flags.
    """
    if not (course_text or "").strip():
        return ""
    course_deg = float(course_text)
    if mount_offset_deg is None:
        heading = course_deg % 360.0
    else:
        heading = float(geo.body_to_world_bearing_deg(
            course_deg,
            geo.apply_mount_offset(geo.azimuth_of_pano_column(0.0, pano_w),
                                   mount_offset_deg)))
    return f"{heading:.4f}"


def fill_course_from_track(kept: list) -> None:
    """Fill `course_deg` from the GPS track between consecutive kept fixes.

    Self-collected logs without a per-fix course (boston legs) get the bearing
    of the surviving track instead. Noisy when near-stationary, but heading
    here is a prior flagged unreliable anyway.
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
        r["course_deg"] = f"{brg:.4f}"


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
    p.add_argument("--mount_offset_deg", type=float, default=None,
                   help="mount-offset prior, quoted in "
                        "geometry.MOUNT_OFFSET_FRAME (the camera-frame "
                        "azimuth, centre column zero, of the direction of "
                        "travel). It is folded into heading_deg and the "
                        "metadata records applied_to_heading_deg=true. Omit "
                        "for an uncalibrated mount: heading_deg is then the "
                        "raw course and NO mount_offset block is written")
    p.add_argument("--mount_offset_source", default=None,
                   help="one-line provenance for --mount_offset_deg, recorded "
                        "in pipeline_metadata.json (required with it)")
    p.add_argument("--coord_decimals", type=int, default=6,
                   help="lat/lon decimals in frame filenames. Mapillary "
                        "datasets use 6; boston_harbor_leg1's pre-existing "
                        "stems (which pinhole dirs and embeddings.pkl key on) "
                        "use 7")
    p.add_argument("--extra_metadata", type=Path, default=None,
                   help="JSON merged into pipeline_metadata.json (extras win)")
    p.add_argument("--copy", action="store_true",
                   help="copy images instead of moving them out of source_dir")
    p.add_argument("--dry_run", action="store_true")
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.mount_offset_deg is not None and not args.mount_offset_source:
        p.error("--mount_offset_deg needs --mount_offset_source: an offset "
                "without provenance cannot be judged later")

    frames_dir = args.source_dir / args.frames_subdir
    rows, src_cols = read_rows(args.source_dir / args.gps_csv)
    if not rows:
        sys.exit("empty frames_gps.csv")

    kept = [r for r in rows
            if r["latitude"].strip() and r["longitude"].strip()]
    dropped = [r for r in rows
               if not (r["latitude"].strip() and r["longitude"].strip())]

    course_from_track = "course_deg" not in src_cols
    if course_from_track:
        fill_course_from_track(kept)

    missing = [r["frame_file"] for r in rows
               if not (frames_dir / r["frame_file"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} frame files in CSV not found on disk: "
                 f"{missing[:3]}")

    log_start = None
    if args.log_start_utc:
        log_start = datetime.datetime.fromisoformat(
            args.log_start_utc.replace("Z", "+00:00"))

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

    out = args.output
    (out / "frames").mkdir(parents=True, exist_ok=True)
    trimmed = out / "trimmed_frames"

    transfer = (lambda s, d: d.write_bytes(s.read_bytes())) if args.copy \
        else (lambda s, d: s.rename(d))

    heading_source = (
        ("track_bearing" if course_from_track else "gps_course")
        + ("_minus_mount_prior" if args.mount_offset_deg is not None
           else "_placeholder"))

    gps_rows, mapping_rows, log_rows, intr_rows = [], [], [], []
    for idx, r in enumerate(kept):
        pano_id = f"f{idx:0{id_width}d}"
        lat, lon = float(r["latitude"]), float(r["longitude"])
        name = f"{pano_id},{lat:.{dec}f},{lon:.{dec}f},.jpg"
        transfer(frames_dir / r["frame_file"], out / "frames" / name)

        gps_rows.append({
            "idx": idx, "video_t_s": r["video_t_s"],
            "sensor_elapsed_s": r.get("sensor_elapsed_s", ""),
            "dist_m": r.get("dist_m", ""), "latitude": r["latitude"],
            "longitude": r["longitude"],
            "altitude_m": r.get("altitude_m", ""),
            "speed_mps": r.get("speed_mps", ""), "frame_file": name,
        })
        mapping_rows.append({"pano_id": pano_id, "lat": r["latitude"],
                             "lon": r["longitude"], "filename": name})

        captured_at = ""
        if log_start is not None and r.get("sensor_elapsed_s", "").strip():
            captured_at = int((log_start + datetime.timedelta(
                seconds=float(r["sensor_elapsed_s"]))).timestamp() * 1000)
        # The ONE guarded heading computation, for both tables (see
        # heading_deg_text: the unguarded intrinsics duplicate is the bug this
        # port fixes).
        heading = heading_deg_text(r.get("course_deg"), args.mount_offset_deg,
                                   args.width)
        log_row = {
            "frame_idx": idx, "pano_id": pano_id,
            "sequence_id": args.dataset_id, "sequence_position": idx,
            "camera_type": "equirectangular",
            "geometry_source": r.get("gps_quality", "sensor_logger"),
            "lat": r["latitude"], "lng": r["longitude"],
            "heading_used": heading,
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
            "heading_deg": heading,
            "heading_reference": "column_0",
            "heading_source": heading_source,
        })

    write_rows(out / "frames_gps.csv", gps_rows, GPS_CONTRACT)
    write_rows(out / "pano_id_mapping.csv", mapping_rows,
               ["pano_id", "lat", "lon", "filename"])
    extlog_cols = EXTLOG_FIXED + [
        c for c in src_cols
        if c not in GPS_CONTRACT and c not in EXTLOG_FIXED]
    write_rows(out / "extraction_log.csv", log_rows, extlog_cols)
    write_rows(out / "intrinsics.csv", intr_rows, INTRINSICS_COLS)

    if dropped:
        trimmed.mkdir(exist_ok=True)
        for r in dropped:
            transfer(frames_dir / r["frame_file"], trimmed / r["frame_file"])
        write_rows(trimmed / "dropped_frames_gps.csv", dropped, src_cols)
        (trimmed / "trim_note.json").write_text(json.dumps({
            "dropped": len(dropped),
            "reason": "no usable GPS fix at ingest (empty latitude/longitude; "
                      "gps_quality=unusable in dropped_frames_gps.csv)",
            "generator": "farfield/dataset_tools/ingest_selfcollect.py",
            "git_commit": provenance.git_commit(),
            "argv": list(sys.argv),
        }, indent=1))

    pano = out / "panorama"
    if not pano.exists():
        pano.symlink_to("frames")

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
            "heading_deg_is_bearing_of": "column_0",
            "formula": "azimuth_deg = (heading_deg + (col / width) * 360) "
                       "mod 360",
            "heading_per_frame": "intrinsics.csv:heading_deg",
            # The machine tag for the frame mount offsets are quoted in
            # (which is NOT this formula's column-0 frame); the convention
            # paragraph itself lives in the mount_offset block, imported from
            # geometry.MOUNT_OFFSET_CONVENTION, never retyped.
            "mount_offset_frame": geo.MOUNT_OFFSET_FRAME,
        },
        "camera_type": "equirectangular",
        "image_dir": "frames",
        "num_images": len(kept),
        "resolution": f"{args.width}x{args.height}",
        "heading_source": heading_source,
        "heading_reliable": False,
        "heading_note": "heading_deg is the absolute bearing of pano column "
                        "0, derived from the GPS course over ground (see "
                        "heading_source; on a boat, leeway/crab also makes "
                        "course differ from heading). Validate before "
                        "trusting bearings.",
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
    if args.mount_offset_deg is not None:
        # The block satisfies dataset.mount_offset_record by construction:
        # frame is the machine tag, applied_to_heading_deg records what this
        # script ACTUALLY did (heading_deg_text folded the offset in), and the
        # convention paragraph is the geometry constant verbatim.
        meta["mount_offset"] = {
            "mount_offset_deg": args.mount_offset_deg,
            "frame": geo.MOUNT_OFFSET_FRAME,
            "applied_to_heading_deg": True,
            "status": "prior",
            "accuracy_validated": False,
            "source": args.mount_offset_source,
            "convention": geo.MOUNT_OFFSET_CONVENTION,
        }
    # else: NO mount_offset block. Absent is the contract's word for
    # "uncalibrated" (dataset.mount_offset_record); a block with a null angle
    # is unconsumable and a block with an invented one is a 180-degree
    # incident waiting to happen.
    if args.extra_metadata:
        meta.update(json.loads(args.extra_metadata.read_text()))
    (out / "pipeline_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    if dists:
        print(f"wrote {out}: {len(kept)} frames, {len(dropped)} trimmed, "
              f"{(dists[-1] - dists[0]) / 1000:.2f} km")
    else:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
