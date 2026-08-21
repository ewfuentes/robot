"""Build a localization export from a tracking run's primary artifacts.

One tool replaces the old two-step (m11_base_export + m11_localization_export
and their duplicated table plumbing): everything matcher-independent is
derived from primary artifacts, and `--tables` selects where the
compatibility tables come from —

  --tables uninformative   one flat table per tracklet (no matcher has
                           spoken): the association-ambiguity floor, what
                           bearings + dead reckoning can do with no matcher.
  --tables <path.json>     the matcher's CompatibilityTable list (matching
                           stage output), keyed by the same per-track
                           tracklet ids the tracklets library emits.

What gets written (the export_ingest.py contract):

  tier1_measurements.jsonl  tracklets.build_measurements over tracks + audit,
                            rotated into the body frame:
                            geometry.apply_mount_offset. THE OFFSET IS BAKED
                            IN HERE — this tool refuses to guess it and
                            records which source it used.
  tier1_odometry.jsonl      gps_to_odometry.derive_increments over the
                            keyframe fixes (§5.2 body-frame derivation).
  truth.jsonl               the same fixes; heading is GPS COURSE, not a
                            measured heading — diagnostics only.
  landmarks.json            every catalog row: the honest candidate universe
                            is the whole map, no spatial shortlist.
  tier1_tables.json         per --tables.
  export_meta.json          anchor, matcher, mount-offset provenance
                            (offset + source + frame), catalog path,
                            git commit / argv / created.

Tracklets the semantic audit dropped (verdict: drop) are excluded: the
pipeline has already decided they are not landmarks, the matcher never
queries them, and exporting them feeds the filter bearings classified as
clutter. --keep_dropped_tracklets restores them for a control run.

Keyframe indices stay the dataset's own. Since tracklets are per-track (no
merge stage), one tracklet cannot produce two epochs on one keyframe; a
duplicate (tracklet, keyframe) now indicates a bug and is an error.

The export is read straight back through export_ingest.load at the end, so
a broken export fails here rather than deep inside a filter run.
"""

import argparse
import json
import sys
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import (
    dataset as dataset_lib,
    geometry as geo,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    audit_io,
    heading as heading_mod,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    gps_to_odometry,
    run_io,
    structs,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets

UNINFORMATIVE_MATCHER = "uninformative_v1"

# Coarse display type for the viewer's glyphs; the filter itself only uses
# ids and positions. THE single definition (previously mirrored in two m11
# tools that had to be kept in sync by hand).
TYPE_TAGS = ("seamark:type", "man_made", "leisure", "amenity", "natural",
             "building", "place", "highway")


def type_key(tags: dict) -> str:
    for key in TYPE_TAGS:
        if key in tags:
            return f"{key}={tags[key]}"
    return "landmark"


def load_tracks(run_dir: Path) -> dict:
    """All tracks across every range file of the run, keyed by track_id.

    Reads every tracks_*.json (the old readers took next(glob(...)), which
    silently picked one arbitrary range on multi-range runs and crashed with
    a bare StopIteration when tracking had not run).
    """
    paths = sorted(Path(run_dir).glob("tracks_*.json"))
    if not paths:
        raise SystemExit(
            f"no tracks_*.json under {run_dir} — run the tracking stage "
            f"first.")
    tracks = {}
    for path in paths:
        for track in json.loads(path.read_text())["tracks"]:
            tid = track["track_id"]
            if tid in tracks:
                raise SystemExit(
                    f"track_id {tid} appears in more than one range file "
                    f"under {run_dir}; range files must partition tracks.")
            tracks[tid] = track
    return tracks


def body_frame_measurements(camera_measurements: list,
                            mount_offset_deg: float) -> list:
    """Camera-frame tracklet measurements -> body-frame TrackletMeasurement.

    Kappa is carried verbatim; the offset is a pure rotation.
    """
    seen = set()
    out = []
    for m in camera_measurements:
        key = (m.tracklet_id, m.anchor_keyframe_idx)
        if key in seen:
            raise SystemExit(
                f"duplicate information epoch {key}: with per-track "
                f"tracklets this cannot happen legitimately — the tracks "
                f"files are corrupt or overlapping.")
        seen.add(key)
        out.append(structs.TrackletMeasurement(
            tracklet_id=m.tracklet_id,
            anchor_keyframe_idx=m.anchor_keyframe_idx,
            bearing_body_deg=float(geo.apply_mount_offset(
                m.bearing_camera_deg, mount_offset_deg)),
            kappa=m.kappa))
    return out


def uninformative_tables(measurements: list, default_log_lr: float,
                         clip: float) -> list:
    """One flat table per measured tracklet: no matcher has spoken yet."""
    return [structs.CompatibilityTable(
        tracklet_id=tracklet_id, matcher_version=UNINFORMATIVE_MATCHER,
        entries=[], default_log_lr=default_log_lr,
        clip_lo=-clip, clip_hi=clip, status="fast")
        for tracklet_id in sorted({m.tracklet_id for m in measurements})]


def landmark_entries(feather_path: Path, anchor_lat: float,
                     anchor_lon: float) -> list:
    """Whole catalog as LandmarkEntry, positions round-tripped through ENU.

    Goes through catalog.load_catalog so the ids, tag pruning and hull
    handling are the same ones the matcher saw.
    """
    entries = catalog_lib.load_catalog(feather_path, anchor_lat, anchor_lon,
                                       keep_hulls=False)
    frame = geo.RegionFrame(anchor_lat, anchor_lon)
    landmarks = []
    for entry in entries:
        lat, lon = frame.latlon_from_enu(entry.east_m, entry.north_m)
        landmarks.append(structs.LandmarkEntry(
            landmark_id=entry.landmark_id, lat_deg=float(lat),
            lon_deg=float(lon), type_key=type_key(entry.tags)))
    return landmarks


def resolve_mount_offset(run_dir: Path, metadata: dict, dataset_base: Path,
                         override: float | None) -> tuple[float, str]:
    """The offset to bake in, and where it came from.

    Order, best evidence first:
      1. an explicit --mount_offset_deg;
      2. a validated dataset record (dataset.mount_offset_record enforces
         the frame + applied qualifiers) marked accuracy_validated — an
         ABSOLUTE check: surveyed landmark, or the sun;
      3. this run's sun_offset_check.json, when usable (also absolute);
      4. this run's mount_offset_sweep.json, when usable. The sweep is
         RELATIVE: it makes rays to unknown objects agree with each other,
         so it reproduces any error the poses and heading model share —
         including a 180 deg slip, which it fits perfectly;
      5. an unvalidated dataset record, used but announced loudly.
    """
    if override is not None:
        return float(override), "--mount_offset_deg"

    record = dataset_lib.mount_offset_record(metadata, dataset_base)
    if record is not None and record.accuracy_validated:
        return record.offset_deg, (
            f"pipeline_metadata.mount_offset ({record.status}, "
            f"accuracy_validated)")

    sun_path = run_dir / "sun_offset_check.json"
    if sun_path.exists():
        sun = json.loads(sun_path.read_text())
        if sun.get("usable") and sun.get("frame") == geo.MOUNT_OFFSET_FRAME:
            return (float(sun["mount_offset_deg"]),
                    f"sun_offset_check.json ({sun.get('verdict')})")

    sweep_path = run_dir / "mount_offset_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        if sweep.get("usable") and sweep.get("frame") == geo.MOUNT_OFFSET_FRAME:
            return (float(sweep["mount_offset_deg"]),
                    f"mount_offset_sweep.json ({sweep.get('verdict')}, "
                    f"{sweep.get('tracklets_used')} tracklets)")
        print(f"  sweep present but not usable ({sweep.get('verdict')}); "
              f"falling back")

    if record is None:
        raise SystemExit(
            f"no mount offset available: no usable sun_offset_check.json or "
            f"mount_offset_sweep.json under {run_dir}, and the dataset "
            f"records none. Run the calibration stage first, or pass "
            f"--mount_offset_deg.")
    print(f"  WARNING: dataset offset ({record.status or 'unvalidated'}) was "
          f"never accuracy-validated; every bearing in this export "
          f"inherits it")
    return record.offset_deg, (
        f"pipeline_metadata.mount_offset ({record.status or 'unvalidated'})")


def write_export(out_dir: Path, meta: dict, landmarks: list, tables: list,
                 measurements: list, odometry: list, truth: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=1))
    for name, payload in (("landmarks.json", landmarks),
                          ("tier1_tables.json", tables)):
        (out_dir / name).write_bytes(
            msgspec.json.encode(payload, enc_hook=msgspec_enc_hook))
    for name, records in (("tier1_measurements.jsonl", measurements),
                          ("tier1_odometry.jsonl", odometry),
                          ("truth.jsonl", truth)):
        run_io.write_jsonl(out_dir / name, records)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    paths_lib.add_arguments(parser, feather=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="e.g. <run_dir>/localization_export")
    parser.add_argument("--tables", required=True,
                        help="'uninformative', or the path to a matching "
                             "stage's compatibility.json")
    parser.add_argument("--mount_offset_deg", type=float, default=None,
                        help="override every recorded offset source")
    parser.add_argument("--scenario_name", default=None,
                        help="default: <dataset>_<run dir name>")
    parser.add_argument("--keep_dropped_tracklets", action="store_true",
                        help="export bearings the audit returned "
                             "verdict=drop for (controls only)")
    # Modeling parameters: required, recorded into the meta (REORG.md rule 2).
    parser.add_argument("--epoch_keyframes", type=int, required=True,
                        help="keyframes fused per bearing (previously 5)")
    parser.add_argument("--bearing_sigma_deg", type=float, required=True,
                        help="per-observation bearing noise floor "
                             "(previously 1.0)")
    parser.add_argument("--default_log_lr", type=float, required=True,
                        help="flat score in uninformative tables "
                             "(previously 0.0)")
    parser.add_argument("--clip", type=float, required=True,
                        help="log-LR clip the filter applies (previously 4)")
    parser.add_argument("--min_step_m", type=float, required=True,
                        help="odometry speed gate (previously 2.0)")
    parser.add_argument("--sigma_pair_m", type=float, required=True,
                        help="per-fix-pair GPS sigma (previously 1.0)")
    parser.add_argument("--max_visible_range_m", type=float, required=True,
                        help="catalog visibility radius, recorded for "
                             "replay (previously 15000 in one place and "
                             "10000 in another — which is why it is now "
                             "required)")
    args = parser.parse_args()
    paths = paths_lib.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "feather"))

    metadata = dataset_lib.load_metadata(paths.dataset_base)
    dataset_lib.require_camera_frame_panoramas(metadata, paths.dataset_base)
    offset_deg, offset_source = resolve_mount_offset(
        args.run_dir, metadata, paths.dataset_base, args.mount_offset_deg)
    print(f"mount offset: {offset_deg} deg from {offset_source}")

    frames = dataset_lib.load_frames(paths.dataset_base)
    anchor_lat, anchor_lon = dataset_lib.fill_enu(frames)
    frames = sorted(frames, key=lambda f: f.frame_idx)
    if any(f.time_s is None for f in frames):
        raise SystemExit("frames without time_s: the heading model and the "
                         "odometry speed gate both need timestamps")

    tracks = load_tracks(args.run_dir)
    audits = audit_io.load_audits(args.run_dir)
    if not audits:
        raise SystemExit(
            f"no semantic audit under {args.run_dir}: audit membership is "
            f"the tracklet gate, so an unaudited run has nothing to export. "
            f"Run the audit stage first.")
    if not args.keep_dropped_tracklets:
        dropped = sorted(tid for tid, a in audits.items()
                         if (a or {}).get("verdict") == "drop")
        audits = {tid: a for tid, a in audits.items()
                  if (a or {}).get("verdict") != "drop"}
        if dropped:
            print(f"audit verdict=drop: excluded {len(dropped)} tracks "
                  f"({dropped[:5]}{'...' if len(dropped) > 5 else ''})")
    else:
        dropped = []

    probe = sorted((paths.dataset_base / "panorama").glob("*.jpg"))[0]
    from PIL import Image
    pano_w = Image.open(probe).size[0]

    camera_measurements = tracklets.build_measurements(
        tracks, audits, pano_w,
        tracklets.TrackletParams(epoch_keyframes=args.epoch_keyframes,
                                 bearing_sigma_deg=args.bearing_sigma_deg))
    measurements = body_frame_measurements(camera_measurements, offset_deg)

    east = np.array([f.x_m for f in frames], dtype=np.float64)
    north = np.array([f.y_m for f in frames], dtype=np.float64)
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in frames], [f.y_m for f in frames],
        [f.time_s for f in frames])
    course = [model.at(f.time_s) for f in frames]
    odometry = gps_to_odometry.derive_increments(
        east, north, sigma_pair_m=args.sigma_pair_m,
        min_step_m=args.min_step_m)
    truth = [structs.TruthPose(keyframe_idx=i, east_m=float(east[i]),
                               north_m=float(north[i]),
                               heading_deg=float(course[i]) % 360.0)
             for i in range(len(frames))]

    if args.tables == "uninformative":
        tables = uninformative_tables(measurements, args.default_log_lr,
                                      args.clip)
        matcher_version = UNINFORMATIVE_MATCHER
        log_lr_scheme = {"source": "no matcher; flat tables",
                         "default_log_lr": args.default_log_lr,
                         "clip": args.clip}
    else:
        tables_path = Path(args.tables)
        tables = msgspec.json.decode(
            tables_path.read_bytes(), type=list[structs.CompatibilityTable],
            dec_hook=msgspec_dec_hook)
        versions = {t.matcher_version for t in tables}
        if len(versions) != 1:
            raise SystemExit(f"{tables_path} mixes matcher versions "
                             f"{sorted(versions)}")
        matcher_version = versions.pop()
        log_lr_scheme = {"source": str(tables_path)}
        have = {t.tracklet_id for t in tables}
        need = {m.tracklet_id for m in measurements}
        if need - have:
            raise SystemExit(
                f"{len(need - have)} measured tracklets have no table in "
                f"{tables_path} (e.g. {sorted(need - have)[:3]}): the "
                f"matching run and this export disagree about which tracks "
                f"exist. Re-run matching on this run's tracks.")

    landmarks = landmark_entries(paths.feather, anchor_lat, anchor_lon)

    scenario = args.scenario_name or f"{paths.dataset}_{args.run_dir.name}"
    meta = {
        "schema_version": structs.SCHEMA_VERSION,
        "scenario_name": scenario,
        "anchor_lat_deg": anchor_lat,
        "anchor_lon_deg": anchor_lon,
        "n_keyframes": len(frames),
        "matcher_version": matcher_version,
        "mount_offset_deg": offset_deg,
        "mount_offset_source": offset_source,
        "mount_offset_frame": geo.MOUNT_OFFSET_FRAME,
        "log_lr_scheme": log_lr_scheme,
        # Provenance extras (ExportMeta ignores unknown fields; readers of
        # the raw JSON get the full record).
        "audit_dropped_tracklets": dropped,
        "epoch_keyframes": args.epoch_keyframes,
        "bearing_sigma_deg": args.bearing_sigma_deg,
        "min_step_m": args.min_step_m,
        "sigma_pair_m": args.sigma_pair_m,
        "max_visible_range_m": args.max_visible_range_m,
        "truth_heading_note": "GPS course, not a measured heading",
        "catalog": str(paths.feather),
        "run_dir": str(args.run_dir),
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
    }
    write_export(args.output_dir, meta, landmarks, tables, measurements,
                 odometry, truth)
    print(f"{len(measurements)} measurements over "
          f"{len({m.tracklet_id for m in measurements})} tracklets, "
          f"{len(odometry)} odometry steps, {len(landmarks)} landmarks")
    print(f"export written to {args.output_dir}\n")
    # Read it straight back through the filter's own loader: validate() is
    # the boundary that would otherwise fail deep inside a run.
    print(export_ingest.describe(export_ingest.load(
        args.output_dir, max_visible_range_m=args.max_visible_range_m)))


if __name__ == "__main__":
    main()
