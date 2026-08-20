"""Build the Tier-1 localization export a tracking run implies.

`m11_localization_export` swaps M9's compatibility tables into an export it is
*handed*: it copies bearings, truth and odometry from `--base_export`
unchanged, because those are matcher-independent. Nothing built that base.
boston_harbor_leg1's came out of the earlier wedge/GPS campaign
(`matcher_version: vlm_pairing_gemini3flash_wedge_gps_TEMP`), so every dataset
processed since had tracks, merges and matches but no way to reach the filter
at all. This is that missing half, derived from primary artifacts:

  tier1_measurements.jsonl  m6's `merged/measurements.json`, rotated into the
                            body frame: `bearing_body = (bearing_camera -
                            mount_offset) mod 360`, kappa carried verbatim.
                            THE OFFSET IS BAKED IN HERE -- which is why this
                            runs after `mount_offset_sweep`, refuses to guess,
                            and records in the meta which source it used.
  tier1_odometry.jsonl      `gps_to_odometry.derive_increments` over the
                            keyframe fixes: the §5.2 body-frame derivation, so
                            leeway lands in forward+heading rather than being
                            quietly dropped.
  truth.jsonl              the same fixes, diagnostics only -- the filter never
                            reads them. `heading_deg` is GPS **course**, not a
                            measured heading: they differ by the crab angle and
                            nothing in these datasets measures the difference.
  landmarks.json            every row of the catalog. The honest candidate
                            universe is the whole map; no spatial shortlist,
                            for the reason m9's docstring gives.
  tier1_tables.json         one UNINFORMATIVE table per measured tracklet --
                            empty entries, `default_log_lr` 0 -- because
                            `export_ingest.validate` requires a table for every
                            measurement. Run the export as it stands for the
                            association-ambiguity floor (what bearings plus
                            dead reckoning can do with no matcher at all), then
                            let m11 overwrite them with the matcher's.

Two epochs of one tracklet can land on the same keyframe when a merged tracklet
fuses source tracks that were both alive there. `export_ingest` rejects that as
a duplicate information epoch, and it is one physically: the two bearings are
the same object seen twice. They are fused as a von Mises product (resultant of
kappa-weighted unit vectors), which is what "fused body-frame bearing" in
`structs.TrackletMeasurement` already promises.

**Tracklets the semantic audit dropped are excluded**, because the pipeline has
already decided they are not landmarks. m5's audit looks at every detection of an
object across all its frames and returns `verdict: drop` for the ones that are
not a usable distinct object; `m9_match_landmarks` honours that and never queries
them, so they have no compatibility table. Exporting their bearings anyway
produced two failures at once: `m11_localization_export` died on measurements
with no table -- blaming a stale matching run, which was the wrong diagnosis --
and any run that got past it fed the filter bearings the pipeline had already
classified as clutter, inflating the clutter rate above the `pi0` the filter
assumes. Five of leg1's 107 tracklets and three of mount_washington leg3's 199
were in this state. `--keep_dropped_tracklets` restores the old behaviour for a
control run.

Keyframe indices stay the dataset's own, so a run log points straight back at
`keyframes/f####.html`. A dataset whose frames are not contiguous 0..N-1 is
refused rather than silently renumbered -- the filter needs contiguous
odometry, and renumbering would misalign every measurement against the viewer.

Run:
  bazel run //...object_tracking:m11_base_export -- --run_dir <runs>/r001_v4
then, once matching exists:
  bazel run //...object_tracking:m11_localization_export -- --run_dir <run> \\
      --base_export <run>/localization_export_base \\
      --output_dir <run>/localization_export_llm_chunked
"""

import argparse
import json
import math
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.bearing_only_localization import (
    export_ingest,
    geodesy,
    gps_to_odometry,
    structs,
)
from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    semantic_audit,
    harbor_catalog,
    heading as heading_mod,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

UNINFORMATIVE_MATCHER = "uninformative_v1"

# Coarse display type for the viewer's glyphs; the filter itself only uses ids
# and positions. Mirrors m11_localization_export._TYPE_TAGS.
TYPE_TAGS = ("seamark:type", "man_made", "leisure", "amenity", "natural",
             "building", "place", "highway")


def type_key(tags: dict) -> str:
    for key in TYPE_TAGS:
        if key in tags:
            return f"{key}={tags[key]}"
    return "landmark"


def fuse_bearings(rows: list) -> tuple[float, float]:
    """(bearing_deg, kappa) of the von Mises product of `(bearing, kappa)`.

    Exact for a product of von Mises densities: the resultant of the
    kappa-weighted unit vectors carries both the fused mean and its
    concentration, so two tight agreeing bearings sharpen and two opposed ones
    cancel rather than averaging into a confident wrong answer.
    """
    east = sum(kappa * math.sin(math.radians(bearing)) for bearing, kappa in rows)
    north = sum(kappa * math.cos(math.radians(bearing)) for bearing, kappa in rows)
    return math.degrees(math.atan2(east, north)) % 360.0, math.hypot(east, north)


def body_frame_measurements(rows: list,
                            mount_offset_deg: float) -> tuple[list, int]:
    """`merged/measurements.json` rows -> TrackletMeasurement list.

    `rows` are dicts with tracklet_id, anchor_keyframe_idx, bearing_camera_deg
    and kappa, exactly as m6 writes them. Returns the measurements and how many
    epochs had to be fused.
    """
    by_epoch = {}
    for row in rows:
        kappa = float(row["kappa"])
        if not math.isfinite(kappa) or kappa <= 0.0:
            raise ValueError(
                f"non-positive kappa {kappa} on {row['tracklet_id']} at "
                f"keyframe {row['anchor_keyframe_idx']}; the filter would "
                f"divide by it")
        key = (row["tracklet_id"], int(row["anchor_keyframe_idx"]))
        body = (float(row["bearing_camera_deg"]) - mount_offset_deg) % 360.0
        by_epoch.setdefault(key, []).append((body, kappa))
    out = []
    for (tracklet_id, keyframe), observations in sorted(by_epoch.items()):
        bearing, kappa = (observations[0] if len(observations) == 1
                          else fuse_bearings(observations))
        out.append(structs.TrackletMeasurement(
            tracklet_id=tracklet_id, anchor_keyframe_idx=keyframe,
            bearing_body_deg=bearing, kappa=kappa))
    fused = sum(1 for obs in by_epoch.values() if len(obs) > 1)
    return out, fused


def uninformative_tables(measurements: list, default_log_lr: float,
                         clip: float) -> list:
    """One flat table per measured tracklet: no matcher has spoken yet.

    Every landmark scores the same, so a measurement still says "a bearing to
    *something* in the catalog" -- the association-ambiguity floor -- without
    asserting which.
    """
    return [structs.CompatibilityTable(
        tracklet_id=tracklet_id, matcher_version=UNINFORMATIVE_MATCHER,
        entries=[], default_log_lr=default_log_lr,
        clip_lo=-clip, clip_hi=clip, status="fast")
        for tracklet_id in sorted({m.tracklet_id for m in measurements})]


def truth_poses(east_m, north_m, heading_deg) -> list:
    return [structs.TruthPose(keyframe_idx=i, east_m=float(east_m[i]),
                              north_m=float(north_m[i]),
                              heading_deg=float(heading_deg[i]))
            for i in range(len(east_m))]


def landmark_entries(feather_path: Path, anchor_lat: float,
                     anchor_lon: float) -> list:
    """Whole catalog as LandmarkEntry, positions round-tripped through ENU.

    Goes through `harbor_catalog.load_catalog` rather than reading the feather
    directly so the ids, the tag pruning and the hull handling are the same
    ones the matcher saw.
    """
    entries = harbor_catalog.load_catalog(feather_path, anchor_lat, anchor_lon,
                                          keep_hulls=False)
    frame = geodesy.RegionFrame(anchor_lat, anchor_lon)
    landmarks = []
    for entry in entries:
        lat, lon = frame.latlon_from_enu(entry.east_m, entry.north_m)
        landmarks.append(structs.LandmarkEntry(
            landmark_id=entry.landmark_id, lat_deg=float(lat),
            lon_deg=float(lon), type_key=type_key(entry.tags)))
    return landmarks


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
        with open(out_dir / name, "wb") as handle:
            for record in records:
                handle.write(msgspec.json.encode(
                    record, enc_hook=msgspec_enc_hook) + b"\n")


def audit_dropped_tracklets(run_dir: Path) -> set:
    """Merged tracklet ids whose semantic audit returned `verdict: drop`.

    Reads the same two files `m9_match_landmarks.query_bundles` reads, and
    applies the same rule, so the export's measurement set and the matcher's
    query set cannot drift apart. An absent audit returns the empty set with a
    warning rather than failing: a run without m5 has no verdicts to honour,
    and the caller may legitimately be building a pre-audit baseline.
    """
    meta_path = run_dir / "semantic_audit" / "audit_meta.json"
    results_path = run_dir / "semantic_audit" / "results.jsonl"
    landmarks_path = run_dir / "merged" / "landmarks.json"
    if not (meta_path.exists() and results_path.exists()
            and landmarks_path.exists()):
        print("  WARNING: no semantic audit under this run, so no `drop` "
              "verdicts can be honoured; every merged tracklet is exported")
        return set()

    key_for_track = {v["track_id"]: k
                     for k, v in json.loads(meta_path.read_text()).items()}
    verdicts = {}
    with open(results_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            key, audit, _ = semantic_audit.parse_result_line(json.loads(line))
            if audit:
                verdicts[key] = audit.get("verdict")

    dropped = set()
    for landmark in json.loads(landmarks_path.read_text()):
        # First audited constituent decides, matching query_bundles' loop.
        for track_id in landmark["track_ids"]:
            verdict = verdicts.get(key_for_track.get(track_id))
            if verdict is not None:
                if verdict == "drop":
                    dropped.add(landmark["landmark_id"])
                break
    return dropped


def resolve_mount_offset(run_dir: Path, metadata_path: Path,
                         override: float | None) -> tuple[float, str]:
    """The offset to bake in, and where it came from.

    Order, best evidence first:

      1. an explicit `--mount_offset_deg`;
      2. a recorded offset marked `accuracy_validated` -- an **absolute** check,
         meaning one that compared against something outside the run's own
         bearings: a surveyed landmark, or the sun via `sun_offset_check`;
      3. this run's own `mount_offset_sweep.json`, if the curve was usable;
      4. a recorded offset that was never checked, used but announced loudly.

    The validated value outranks the sweep, and that ordering is the correction
    to an earlier version which had it the other way round. The sweep is
    **relative**: it finds the angle that makes rays to unknown objects agree
    with each other, so it silently reproduces any error the poses and the
    heading model share -- including a 180 deg convention slip, which it fits
    perfectly. It was right about the harbour legs to within 1-3 deg, but being
    right is not the same as being checkable, and the export bakes this number
    into every bearing it writes.
    """
    if override is not None:
        return float(override), "--mount_offset_deg"

    block = json.loads(metadata_path.read_text()).get("mount_offset") or {}
    recorded = block.get("mount_offset_deg")
    if recorded is not None and block.get("accuracy_validated", False):
        return float(recorded), (f"pipeline_metadata.mount_offset "
                                 f"({block.get('status', '?')}, "
                                 f"accuracy_validated)")

    sweep_path = run_dir / "mount_offset_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        if sweep.get("usable"):
            return (float(sweep["mount_offset_deg"]),
                    f"mount_offset_sweep.json ({sweep['verdict']}, "
                    f"{sweep['tracklets_used']} tracklets)")
        print(f"  sweep present but not usable ({sweep.get('verdict')}); "
              f"falling back to the dataset's recorded offset")

    if recorded is None:
        raise SystemExit(
            f"no mount offset available: {run_dir}/mount_offset_sweep.json is "
            f"absent or unusable and {metadata_path} records none. Run "
            f"sun_offset_check or mount_offset_sweep first, or pass "
            f"--mount_offset_deg.")
    source = f"pipeline_metadata.mount_offset ({block.get('status', '?')})"
    print(f"  WARNING: {source} was never accuracy-validated; every "
          f"bearing in this export inherits it")
    return float(recorded), source


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    farfield_paths.add_arguments(parser, feather=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="default: <run_dir>/localization_export_base")
    parser.add_argument("--mount_offset_deg", type=float, default=None,
                        help="override the sweep and the dataset metadata")
    parser.add_argument("--scenario_name", default=None)
    parser.add_argument("--keep_dropped_tracklets", action="store_true",
                        help="export bearings for tracklets the semantic audit "
                             "returned verdict=drop for. They have no "
                             "compatibility table, so m11_localization_export "
                             "cannot consume the result; for controls only")
    parser.add_argument("--default_log_lr", type=float, default=0.0,
                        help="flat score every landmark gets in the "
                             "uninformative tables (default: 0, no evidence)")
    parser.add_argument("--clip", type=float, default=4.0,
                        help="log-LR clip the filter applies (default: 4)")
    parser.add_argument("--min_step_m", type=float, default=2.0,
                        help="steps shorter than this carry no usable course "
                             "(gps_to_odometry speed gate)")
    parser.add_argument("--sigma_pair_m", type=float, default=1.0,
                        help="per-fix-pair GPS sigma for odometry noise")
    args = parser.parse_args()
    paths = farfield_paths.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks", "feather"))
    out_dir = args.output_dir or (args.run_dir / "localization_export_base")

    measurements_path = args.run_dir / "merged" / "measurements.json"
    if not measurements_path.exists():
        raise SystemExit(
            f"{measurements_path} not found - run m6_merge_tracks first; the "
            f"export is built out of its fused bearings.")

    offset_deg, offset_source = resolve_mount_offset(
        args.run_dir, paths.metadata_path, args.mount_offset_deg)
    print(f"mount offset: {offset_deg} deg from {offset_source}")

    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                              IngestConfig())
    frames = sorted(result.frames, key=lambda f: f.frame_idx)
    indices = [f.frame_idx for f in frames]
    if indices != list(range(len(frames))):
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise SystemExit(
            f"keyframe indices are not contiguous 0..{len(frames) - 1} "
            f"({len(missing)} gaps, e.g. {missing[:5]}). The filter needs "
            f"contiguous odometry and renumbering here would silently "
            f"misalign every measurement against the tracking viewer.")
    if any(f.time_s is None for f in frames):
        raise SystemExit("frames without time_s: the heading model and the "
                         "odometry speed gate both need timestamps")

    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in frames], [f.y_m for f in frames],
        [f.time_s for f in frames])
    east = np.array([f.x_m for f in frames], dtype=np.float64)
    north = np.array([f.y_m for f in frames], dtype=np.float64)
    course = [model.at(f.time_s) for f in frames]

    rows = json.loads(measurements_path.read_text())
    dropped = set() if args.keep_dropped_tracklets else \
        audit_dropped_tracklets(args.run_dir)
    if dropped:
        before = len({r["tracklet_id"] for r in rows})
        rows = [r for r in rows if r["tracklet_id"] not in dropped]
        kept = len({r["tracklet_id"] for r in rows})
        print(f"audit verdict=drop: excluded {before - kept} of {before} "
              f"tracklets ({sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''})")
    measurements, fused = body_frame_measurements(rows, offset_deg)
    odometry = gps_to_odometry.derive_increments(
        east, north, sigma_pair_m=args.sigma_pair_m,
        min_step_m=args.min_step_m)
    truth = truth_poses(east, north, course)
    tables = uninformative_tables(measurements, args.default_log_lr, args.clip)
    landmarks = landmark_entries(paths.feather, result.anchor_lat,
                                 result.anchor_lon)

    scenario = args.scenario_name or f"{paths.dataset}_{args.run_dir.name}"
    meta = {
        "schema_version": structs.SCHEMA_VERSION,
        "scenario_name": scenario,
        "anchor_lat_deg": result.anchor_lat,
        "anchor_lon_deg": result.anchor_lon,
        "n_keyframes": len(frames),
        "matcher_version": UNINFORMATIVE_MATCHER,
        "mount_offset_deg": offset_deg,
        "mount_offset_source": offset_source,
        "audit_dropped_tracklets": sorted(dropped),
        "log_lr_scheme": {
            "source": "no matcher; flat tables from m11_base_export",
            "default_log_lr": args.default_log_lr,
            "clip": args.clip,
        },
        "truth_heading_note": "GPS course, not a measured heading",
        "catalog": str(paths.feather),
        "run_dir": str(args.run_dir),
    }
    write_export(out_dir, meta, landmarks, tables, measurements, odometry,
                 truth)
    print(f"{len(measurements)} measurements over {len(tables)} tracklets "
          f"({fused} epochs fused from 2+ source tracks), "
          f"{len(odometry)} odometry steps, {len(landmarks)} landmarks")
    print(f"export written to {out_dir}\n")
    # Read it straight back through the filter's own loader: validate() is the
    # boundary that would otherwise fail deep inside a run.
    print(export_ingest.describe(export_ingest.load(out_dir)))


if __name__ == "__main__":
    main()
