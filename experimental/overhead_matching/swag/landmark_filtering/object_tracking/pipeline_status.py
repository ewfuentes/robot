"""What exists for a dataset, what is missing, and the exact next command.

Rerunning this pipeline used to mean reading the runbook, deriving six or seven
paths by hand, and hoping none of them belonged to a different leg. Paths now
resolve from `--dataset` alone (`swag/data/farfield_paths.py`), and this tool
closes the remaining gap: it reports which stages are done, checks the things
that are wrong *silently* rather than loudly, and prints the command for the
next incomplete stage.

Not itself an orchestrator -- `run_pipeline` runs the stages, and this reports on
them. The split is deliberate: `run_pipeline` resumes from wherever the artifacts
stop, so a read-only view of *why* it would start there is worth having
separately, especially for the checks below that pass silently in a normal run.

The consistency checks are the point. Each is a failure that produces plausible
output rather than an error:

- **stem mismatch** between panoramas and pinhole faces. Stems key
  `embeddings.pkl` and the detection ids, so a partial render silently drops
  landmarks for the missing frames.
- **resolution drift** between the rendered faces and what a manifest claims.
- **video identity**: whether `video.source_video` resolves and exists. This is
  the one that used to build tracks from the wrong leg's footage.
- **catalog coverage**: the trajectory bbox against the catalog's extent, since
  m9 does no spatial gating and so the catalog bounds what can ever match.
- **run provenance**: whether each run recorded the inputs it used.

    bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:pipeline_status -- \
        --dataset boston_harbor_leg2
"""

import argparse
import json
import math
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths

OK, MISSING, WARN = "ok", "missing", "warn"
MARK = {OK: "  ok  ", MISSING: "MISSING", WARN: " WARN "}

# Sightlines in a harbour routinely exceed this; a catalog margin smaller than
# it means detections in that direction can have no correct answer in set 2.
SIGHTLINE_M = 5000.0


def image_size(path: Path):
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.width, img.height
    except (OSError, ImportError):
        return None


def check_dataset(paths):
    rows = []
    stems = []
    if paths.panorama_dir.is_dir():
        stems = sorted(p.stem for p in paths.panorama_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rows.append((OK, "panoramas", f"{len(stems)} frames"))
    else:
        rows.append((MISSING, "panoramas", str(paths.panorama_dir)))

    for name, path in (("frames_gps.csv", paths.frames_gps),
                       ("intrinsics.csv", paths.intrinsics),
                       ("metadata", paths.metadata_path)):
        rows.append((OK if path.exists() else MISSING, name,
                     "" if path.exists() else str(path)))

    # The check that used to be a silent wrong answer.
    try:
        video = paths.video
        if video.exists():
            gb = video.stat().st_size / 1e9
            rows.append((OK, "source video", f"{video.name} ({gb:.1f} GB)"))
        else:
            rows.append((MISSING, "source video",
                         f"{video} (declared in metadata, absent on disk)"))
    except farfield_paths.MissingInput as exc:
        rows.append((MISSING, "source video", str(exc).split(";")[0]))

    rows.append((OK if paths.feather.exists() else MISSING, "map catalog",
                 paths.feather.name if paths.feather.exists()
                 else str(paths.feather)))
    rows.append((OK if paths.sam2_checkpoint.exists() else MISSING,
                 "SAM2 checkpoint",
                 "" if paths.sam2_checkpoint.exists()
                 else str(paths.sam2_checkpoint)))
    return rows, stems


def check_mount_offset(paths):
    try:
        block = paths.metadata().get("mount_offset") or {}
    except farfield_paths.MissingInput:
        return [(MISSING, "mount offset", "no metadata")]
    if not block:
        return [(MISSING, "mount offset", "absent from metadata")]
    offset = block.get("mount_offset_deg")
    status = block.get("status", "unrecorded")
    if block.get("accuracy_validated"):
        return [(OK, "mount offset", f"{offset} deg, {status}, "
                                     f"accuracy_validated")]
    return [(WARN, "mount offset",
             f"{offset} deg, status={status!r}, NOT accuracy_validated - "
             f"run mount_offset_sweep before trusting m9 matching")]


def check_pinholes(paths, stems):
    path = paths.pinhole_images
    if not path.is_dir():
        return [(MISSING, "pinhole_images", str(path))]
    dirs = {p.name for p in path.iterdir() if p.is_dir()}
    rows = []
    missing = [s for s in stems if s not in dirs]
    if missing:
        rows.append((MISSING, "pinhole_images",
                     f"{len(dirs)}/{len(stems)} stems rendered; first missing "
                     f"{missing[0]}"))
        return rows
    faces = ("yaw_000", "yaw_090", "yaw_180", "yaw_270")
    incomplete = [s for s in stems
                  if not all((path / s / f"{f}.jpg").exists() for f in faces)]
    if incomplete:
        rows.append((MISSING, "pinhole_images",
                     f"{len(incomplete)} stems missing faces "
                     f"(e.g. {incomplete[0]})"))
        return rows
    detail = f"{len(stems)} stems x 4 faces"
    if stems:
        size = image_size(path / stems[0] / "yaw_000.jpg")
        if size:
            detail += f" @ {size[0]}px"
            manifest = path / "manifest.json"
            if manifest.exists():
                claimed = (json.loads(manifest.read_text())
                           .get("config", {}).get("res_x"))
                if claimed and claimed != size[0]:
                    rows.append((WARN, "pinhole_images",
                                 f"manifest claims res_x={claimed} but faces "
                                 f"are {size[0]}px"))
    rows.append((OK, "pinhole_images", detail))
    rows.append(manifest_row(path, "pinhole manifest"))
    return rows


def manifest_row(artifact_dir: Path, label: str):
    path = artifact_dir / "manifest.json"
    if not path.exists():
        return (WARN, label, "no manifest.json - settings/sources unrecorded")
    manifest = json.loads(path.read_text())
    config = manifest.get("config")
    n = len(config) if isinstance(config, dict) else 1
    return (OK, label, f"{n} config field(s), commit "
                       f"{str(manifest.get('git_commit', '?'))[:8]}")


def check_frame_landmarks(paths, stems):
    path = paths.frame_landmarks
    if not path.is_dir():
        return [(MISSING, "frame_landmarks", str(path))]
    rows = []
    predictions = sorted(path.rglob("predictions.jsonl"))
    if not predictions:
        rows.append((MISSING, "frame_landmarks",
                     "no sentences/results/**/predictions.jsonl - extraction "
                     "did not reach the download stage"))
    else:
        lines = sum(sum(1 for _ in open(p, "rb")) for p in predictions)
        rows.append((OK, "VLM predictions",
                     f"{lines} lines in {len(predictions)} file(s)"))
    # embeddings.pkl is opt-in (--with_embeddings): nothing in the tracking
    # pipeline reads it, so its absence is not a gap.
    embeddings = path / "embeddings" / "embeddings.pkl"
    rows.append((OK, "embeddings",
                 f"{embeddings.stat().st_size / 1e6:.0f} MB"
                 if embeddings.exists()
                 else "absent (opt-in; only the cosine matcher reads it)"))
    rows.append(manifest_row(path, "frame_landmarks manifest"))
    return rows


def check_tracks(paths):
    path = paths.object_tracks
    runs_root = paths.tracks_runs_root
    if not runs_root.is_dir():
        return [(MISSING, "object_tracks", f"no runs under {runs_root}")], []
    runs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    if not runs:
        return [(MISSING, "object_tracks", f"no runs under {runs_root}")], []
    rows = [(OK, "object_tracks", f"{len(runs)} run(s): "
                                  f"{', '.join(p.name for p in runs)}")]
    rows.append(manifest_row(path, "object_tracks manifest"))
    for run in runs:
        meta = run / "run_meta.json"
        if not meta.exists():
            rows.append((WARN, f"  {run.name}", "no run_meta.json"))
            continue
        recorded = json.loads(meta.read_text())
        stages = []
        for label, probe in (("tracks", "tracks_*.json"),
                             ("audit", "semantic_audit/results.jsonl"),
                             ("merged", "merged/measurements.json"),
                             ("offset", "mount_offset_sweep.json"),
                             ("matching", "matching/matches.json")):
            hit = list(run.glob(probe)) if "*" in probe else (
                [run / probe] if (run / probe).exists() else [])
            if hit:
                stages.append(label)
        has_inputs = "inputs" in recorded
        rows.append((OK if has_inputs else WARN, f"  {run.name}",
                     f"stages: {', '.join(stages) or 'none'}"
                     + ("" if has_inputs else
                        "  (run_meta records no inputs - predates provenance "
                        "recording, so which video it used is unknown)")))
    return rows, runs


def check_catalog_coverage(paths):
    """Catalog extent vs trajectory bbox. m9 has no spatial gating."""
    if not paths.feather.exists():
        return []
    try:
        meta = paths.metadata()
    except farfield_paths.MissingInput:
        return []
    bbox = meta.get("bbox")
    if not bbox:
        return []
    try:
        import shapely
        from experimental.overhead_matching.swag.data import landmark_schema
        frame = landmark_schema.read_frame(paths.feather)
        geometry = frame["geometry"].values
        # read_frame hands back shapely geometries via geopandas, but a plain
        # pandas read of the same file yields WKB bytes; accept either.
        if len(geometry) and isinstance(geometry[0], (bytes, bytearray)):
            geometry = shapely.from_wkb(geometry)
        centroids = shapely.centroid(geometry)
        lat, lon = shapely.get_y(centroids), shapely.get_x(centroids)
    except Exception as exc:  # optional check; never block on it
        return [(WARN, "catalog coverage", f"could not read catalog: {exc}")]

    km_lat = 111.32
    km_lon = km_lat * math.cos(math.radians(bbox["south"]))
    margins = {
        "south": (bbox["south"] - float(lat.min())) * km_lat,
        "north": (float(lat.max()) - bbox["north"]) * km_lat,
        "west": (bbox["west"] - float(lon.min())) * km_lon,
        "east": (float(lon.max()) - bbox["east"]) * km_lon,
    }
    tight = {k: v for k, v in margins.items() if v * 1000.0 < SIGHTLINE_M}
    detail = ", ".join(f"{k} {v:.1f}km" for k, v in margins.items())
    if any(v < 0 for v in margins.values()):
        return [(MISSING, "catalog coverage",
                 f"catalog does not cover the trajectory: {detail}")]
    if tight:
        return [(WARN, "catalog coverage",
                 f"{detail} - {'/'.join(tight)} margin(s) below a "
                 f"{SIGHTLINE_M / 1000:.0f} km sightline, so detections that "
                 f"way may have no correct answer in set 2")]
    return [(OK, "catalog coverage", detail)]


def next_command(paths, rows, runs):
    """The one command to run next, given what is missing.

    Almost always `run_pipeline`, which resumes from wherever the artifacts
    stop; the per-stage commands are for working on one stage deliberately.
    """
    missing = {label.strip() for status, label, _ in rows if status == MISSING}
    target = "//experimental/overhead_matching/swag/landmark_filtering/object_tracking"
    ds = paths.dataset

    if {"panoramas", "source video", "map catalog"} & missing:
        return ("Fix the dataset first - a frozen dataset is not something this "
                "pipeline creates. See ingest_selfcollect_dataset.py.")

    if missing:
        latest = runs[-1].name if runs else f"r001_{ds}"
        return (f"bazel run {target}:run_pipeline -- \\\n"
                f"    --dataset {ds} --run_name {latest}\n"
                f"# resumes at the first incomplete stage; --dry_run to see the "
                f"plan, --online for on-demand instead of batch")
    if "pinhole_images" in missing or "frame_landmarks" in missing:
        return ("bazel run //experimental/overhead_matching/swag/scripts:"
                f"extract_gemini_landmarks_from_panoramas -- \\\n"
                f"    --dataset {ds} \\\n"
                f"    --prompt_type osm_tags_farfield \\\n"
                f"    --pinhole_resolution 2048 \\\n"
                f"    --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \\\n"
                f"    --model gemini-3.1-pro-preview")
    if "object_tracks" in missing:
        n = 0
        if paths.panorama_dir.is_dir():
            n = len([p for p in paths.panorama_dir.iterdir()
                     if p.suffix.lower() == ".jpg"])
        return (f"bazel run {target}:m0_render_boxes -- --dataset {ds}\n"
                f"# then, after checking the boxes land on the objects:\n"
                f"bazel run {target}:m3_track_viewer -- \\\n"
                f"    --dataset {ds} --run_name r001_full \\\n"
                f"    --range full 0 {max(n - 1, 0)} \\\n"
                f"    --notes 'first full pass'")

    latest = runs[-1] if runs else None
    if latest is None:
        return "Nothing further resolved; see the runbook."
    if not (latest / "merged" / "measurements.json").exists():
        return (f"bazel run {target}:m5_build_audit_requests -- "
                f"--run_dir {latest} --submit\n"
                f"# read semantic_audit/review/index.html, then:\n"
                f"bazel run {target}:m6_merge_tracks -- --run_dir {latest}")
    if not (latest / "mount_offset_sweep.json").exists():
        return (f"bazel run {target}:mount_offset_sweep -- "
                f"--run_dir {latest}")
    if not (latest / "matching" / "matches.json").exists():
        return (f"bazel run {target}:m9_match_landmarks -- "
                f"--run_dir {latest} --submit\n"
                f"bazel run {target}:m10_match_viewer -- --run_dir {latest}")
    return (f"All artifact stages present for {ds}. Index with:\n"
            f"bazel run {target}:run_index -- --run_dir {latest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--skip_coverage", action="store_true",
                        help="skip the catalog-extent check (reads the feather)")
    args = parser.parse_args()
    paths = farfield_paths.resolve(parser, args)

    print(f"\ndataset: {paths.dataset}   root: {paths.root}\n")
    rows, stems = check_dataset(paths)
    rows += check_mount_offset(paths)
    if not args.skip_coverage:
        rows += check_catalog_coverage(paths)
    rows += check_pinholes(paths, stems)
    rows += check_frame_landmarks(paths, stems)
    track_rows, runs = check_tracks(paths)
    rows += track_rows

    width = max(len(label) for _, label, _ in rows)
    for status, label, detail in rows:
        print(f"  [{MARK[status]}] {label:<{width}}  {detail}")

    counts = {k: sum(1 for s, _, _ in rows if s == k) for k in (OK, WARN,
                                                                MISSING)}
    print(f"\n  {counts[OK]} ok, {counts[WARN]} warning(s), "
          f"{counts[MISSING]} missing")
    print(f"\nnext:\n{next_command(paths, rows, runs)}\n")


if __name__ == "__main__":
    main()
