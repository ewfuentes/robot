#!/usr/bin/env python3
"""Collect the far-field Mapillary trajectories end to end.

Stages, per trajectory:
  1 RESOLVE   seed pKey -> stitched manifest             (seed_to_trajectory)
  2 DOWNLOAD  manifest -> ordered jpg+json staging       (extract_stitch)
  3 CONVERT   staging -> dataset dir                     (mapillary_to_vigor)
  4 TIMELAPSE trajectory.png + gps_timelapse.mp4         (make_dataset_timelapse)
  5 OSM       landmark feather (+ENC where NOAA covers)  (extract_landmarks_*)
  6 PINHOLE   4 yaw faces, equirectangular only          (panorama_to_pinhole)
  7 TRIM      observable-from-water catalog subset       (trim_landmark_feather)
  8 PLOT      landmark coverage figure + gap checks      (plot_landmarks)
  9 AUDIT     dataset contract audit                     (audit_dataset)

TIMELAPSE sits directly after CONVERT (moved 2026-08-17, was stage 8) because
it is the input to the human triage pass, and triage is the cheapest gate:
7 of the first 21 collected trajectories were rejected on the timelapse for
faults no audit sees (unfixed mounts, panning cameras). Rendering it before
the landmark stages means a rejected trajectory costs one mp4, not an
Overpass catalog + pinhole render.

Stage 6 is deliberately skipped for perspective captures: a limited-FOV frame
is already a single view, and its azimuth mapping comes from intrinsics.csv
rather than from four fixed 90-degree faces.

Layout (lifecycle lanes, docs/farfield-data-organization.md):
    <output_base>/<name>/       dataset (panorama/, frames_gps.csv, landmarks/, ...)
    <manifest_dir>/<name>.json  stage-1 stitched manifest
    <raw_dir>/<name>/           staged originals, prunable after stage 3
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.farfield_trajectories import PILOT, TRAJECTORIES, collectable

# Every stage is a bazel target now, invoked as a subprocess from the repo
# root (nested `bazel run` is fine: the outer client releases the workspace
# lock once its binary is executing).
CROSSVIEW_REPO = Path.home() / "code" / "robot-farfield-crossview"
MT = "//experimental/overhead_matching/swag/mapillary_tools"
OSM_DIR = Path.home() / "scratch" / "osm_downloads"
# Lifecycle lanes per docs/farfield-data-organization.md: frozen datasets flat
# in datasets/ (the old mapillary_datasets path is a compat symlink), staging
# and manifests in raw_material/, pinhole faces as a versioned artifact.
DEFAULT_OUTPUT_BASE = Path("/data/farfield_matching/datasets")
DEFAULT_MANIFEST_DIR = Path("/data/farfield_matching/raw_material/mapillary_manifests")
DEFAULT_RAW_DIR = Path("/data/farfield_matching/raw_material/mapillary_raw")
DEFAULT_PINHOLE_BASE = Path("/data/farfield_matching/artifacts/pinhole_images")

# Landmark bbox buffer. Deliberately much wider than a typical VIGOR
# environment: these are far-field datasets, and on water the useful landmarks
# are far outside any buffer sized for a street-level dataset. From a ferry deck
# the sea horizon is ~11 km away, but a 100 m cliff or a harbour crane stays
# visible for 30-40 km -- the Dover cliffs are in shot from mid-Channel, 20+ km
# off the track. A buffer that only covers what is *near* the trajectory would
# omit exactly the landmarks these datasets exist to test.
LANDMARK_BUFFER_KM = 25.0

# extract_landmarks_historical builds a node-location index for the WHOLE pbf in
# RAM before it can filter by bbox, so peak memory tracks file size and not the
# area requested. Whole-France (4.7 GB) reached 28 GB RSS growing 1.2 GB/min and
# took the machine down. Two independent guards, because one is advisory and the
# other is enforced:
#   * refuse files over MAX_PBF_MB (generous now: with the dict schema and the
#     bounded node index a 4.7 GB national extract peaks at 3.0 GB);
#   * run the extraction inside a cgroup with a hard memory ceiling, so a
#     surprise gets killed in its own scope instead of exhausting the host.
MAX_PBF_MB = 6000
EXTRACT_MEM_CAP_GB = 24


def mem_capped(cmd, cap_gb):
    """Wrap a command in a cgroup scope with a hard memory ceiling.

    The heavy worker is a child of the `bazel run` client, so capping the client
    caps the worker. MemorySwapMax=0 matters as much as MemoryMax: swap was
    already exhausted when this went wrong, and letting a runaway swap just moves
    the stall rather than stopping it.
    """
    if not cap_gb:
        return cmd
    if shutil.which("systemd-run") is None:
        print(f"  WARNING: systemd-run unavailable; running without a memory cap")
        return cmd
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={cap_gb}G", "-p", "MemorySwapMax=0", "--"] + cmd


def run(cmd, desc, dry_run=False, cwd=None, check=True):
    print(f"\n{'[DRY RUN] ' if dry_run else ''}=== {desc}")
    print("  $ " + " ".join(str(c) for c in cmd))
    if cwd:
        print(f"  (cwd: {cwd})")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {desc}")
        if check:
            return False
    return True


def bbox_from_dataset(dataset_dir: Path, buffer_km: float):
    """(west, south, east, north) around a converted dataset's trajectory."""
    import csv
    import math
    lats, lngs = [], []
    with open(dataset_dir / "pano_id_mapping.csv") as f:
        for row in csv.DictReader(f):
            lats.append(float(row["lat"]))
            lngs.append(float(row["lon"]))
    mid = (min(lats) + max(lats)) / 2
    dlat = buffer_km / 111.0
    dlng = buffer_km / max(1e-6, 111.0 * math.cos(math.radians(mid)))
    return (min(lngs) - dlng, min(lats) - dlat, max(lngs) + dlng, max(lats) + dlat)


def stage_resolve(name, cfg, args):
    manifest = args.manifest_dir / f"{name}.json"
    if manifest.exists() and not args.force:
        print(f"\n=== [1 RESOLVE] {name}: manifest exists, skipping "
              f"(--force to redo)")
        return True
    cmd = ["bazel", "run", f"{MT}:seed_to_trajectory", "--",
           "--seed_pkey", cfg["seed_pkey"], "--name", name,
           "-o", str(manifest),
           "--stitch_time", str(args.stitch_time),
           "--stitch_dist", str(args.stitch_dist),
           "--window_hours", str(args.window_hours),
           "--workers", str(args.workers)]
    return run(cmd, f"[1 RESOLVE] {name}", args.dry_run, cwd=CROSSVIEW_REPO)


def stage_download(name, cfg, args):
    manifest = args.manifest_dir / f"{name}.json"
    if not manifest.exists() and not args.dry_run:
        print(f"  ERROR: {manifest} missing — run stage 1 first")
        return False
    cmd = ["bazel", "run", f"{MT}:extract_stitch", "--",
           "--manifest", str(manifest), "--sequence", name,
           "--output", str(args.raw_dir / name),
           "--workers", str(args.workers),
           "--max_width", str(args.max_width),
           "--min_spacing_m", str(args.min_spacing_m)]
    return run(cmd, f"[2 DOWNLOAD] {name}", args.dry_run, cwd=CROSSVIEW_REPO)


def stage_convert(name, cfg, args):
    cmd = ["bazel", "run", f"{MT}:mapillary_to_vigor", "--",
           "--sequence_dir", str(args.raw_dir / name),
           "--vigor_dir", str(args.output_base / name),
           "--dataset_name", name,
           "--num_workers", str(args.convert_workers),
           "--jpeg_quality", str(args.jpeg_quality)]
    if args.max_width:
        cmd += ["--resize", str(args.max_width)]
    if args.visualize:
        cmd.append("--visualize")
    ok = run(cmd, f"[3 CONVERT] {name}", args.dry_run, cwd=CROSSVIEW_REPO)
    if ok and args.prune_raw and not args.dry_run:
        # Staging the originals costs several GB per trajectory (Mapillary has no
        # 4096 thumbnail, so a 4096 cap means fetching the full-size image and
        # downscaling). Drop them once the dataset exists; extraction_log.csv
        # keeps the Mapillary ids needed to re-fetch.
        raw = args.raw_dir / name
        converted = list((args.output_base / name / "panorama").glob("*.jpg"))
        if not converted:
            print(f"  NOT pruning {raw}: no converted images found")
        else:
            import shutil
            size_gb = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(raw)
            print(f"  pruned {raw} ({size_gb:.2f} GB) — {len(converted)} converted "
                  f"images retained")
    return ok


def stage_landmarks(name, cfg, args):
    """OSM landmarks, plus ENC and a merge where NOAA has coverage."""
    dataset_dir = args.output_base / name
    if not (dataset_dir / "pano_id_mapping.csv").exists() and not args.dry_run:
        print(f"  ERROR: {dataset_dir}/pano_id_mapping.csv missing — run stage 3 first")
        return False

    buffer_km = cfg.get("landmark_buffer_km", args.landmark_buffer_km)
    if args.dry_run:
        west = south = east = north = 0.0
    else:
        west, south, east, north = bbox_from_dataset(dataset_dir, buffer_km)
    print(f"  landmark bbox (buffer {buffer_km} km): "
          f"{west:.6f} {south:.6f} {east:.6f} {north:.6f}")

    # A trajectory may need several national extracts (see the registry note on
    # cross-border far-field), so normalize to a list.
    osm_specs = cfg["osm"] if isinstance(cfg["osm"], (list, tuple)) else [cfg["osm"]]
    pbf_paths = []
    for spec in osm_specs:
        wanted = spec.split("/")[-1]
        matches = sorted(OSM_DIR.glob(wanted.replace("-latest", "-*")))
        if not matches:
            if args.dry_run:
                pbf_paths.append(OSM_DIR / wanted)
                continue
            print(f"  ERROR: no OSM extract for {spec} in {OSM_DIR} "
                  f"(looked for {wanted.replace('-latest', '-*')})")
            return False
        pbf = matches[-1]
        size_mb = pbf.stat().st_size / 1e6
        if size_mb > args.max_pbf_mb and not args.allow_large_pbf:
            print(f"  ERROR: {pbf.name} is {size_mb:.0f} MB, over the "
                  f"{args.max_pbf_mb} MB limit. The extractor holds a whole-file "
                  f"node index in RAM, so this risks an out-of-memory kill "
                  f"regardless of how small the bbox is.")
            print(f"  Use a smaller Geofabrik sub-extract covering the bbox "
                  f"(e.g. europe/united-kingdom/england/kent at 53 MB instead of "
                  f"europe/united-kingdom at 1925 MB), or pass --allow_large_pbf "
                  f"if you have verified there is headroom.")
            return False
        pbf_paths.append(pbf)

    # Pre-flight: a sub-extract that does not reach the whole bbox yields a
    # partial catalog with no error from the extractor itself. Verified against
    # Geofabrik .poly clip boundaries, before spending extraction time.
    if not args.skip_coverage_check:
        from pbf_coverage import check_coverage
        ok, msg, cov_details = check_coverage(
            osm_specs, (west, south, east, north),
            reference_specs=cfg.get("osm_reference"))
        print(f"  coverage: {'OK' if ok else 'FAIL'} — {msg}")
        for d in cov_details:
            if "spec" in d:
                print(f"    {d['spec'].rsplit('/', 1)[-1]}: "
                      f"{100*d['covers_frac_of_request']:.1f}% of the request")
        if not ok:
            print(f"  ERROR: refusing to build a partial landmark catalog for {name}. "
                  f"Fix the registry's osm list, or pass --skip_coverage_check to "
                  f"accept it.")
            return False

    sources = dataset_dir / "landmarks" / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    osm_feathers = []
    for pbf_path in pbf_paths:
        # Full region name, not the first hyphen-separated token: "new-york" and
        # "new-jersey" both start with "new", so splitting on "-" gave both source
        # feathers the same filename and the second silently overwrote the first
        # (the merge then saw New Jersey twice).
        region = re.sub(r"-\d{6}$", "", pbf_path.name.replace(".osm.pbf", ""))
        out = sources / (f"osm_{name}_{region}_v1" if len(pbf_paths) > 1
                         else f"osm_{name}_v1")
        # --bbox order is WEST SOUTH EAST NORTH for both extractors (matching
        # the canonical boston_harbor invocation).
        extract_cmd = ["bazel", "run",
                       "//experimental/overhead_matching/swag/scripts:extract_landmarks_historical",
                       "--", "--pbf_file", str(pbf_path),
                       "--bbox", str(west), str(south), str(east), str(north),
                       "--output_path", str(out)]
        if args.node_margin_deg >= 0:
            extract_cmd += ["--node_margin_deg", str(args.node_margin_deg)]
        ok = run(mem_capped(extract_cmd, args.extract_mem_cap_gb),
                 f"[5 OSM] {name} ({pbf_path.name}, "
                 f"{pbf_path.stat().st_size/1e6:.0f} MB, cap "
                 f"{args.extract_mem_cap_gb} GB)", args.dry_run, cwd=CROSSVIEW_REPO)
        if not ok:
            return False
        osm_feathers.append(out.with_suffix(".feather"))

    if len(osm_feathers) > 1:
        combined = sources / f"osm_{name}_v1.feather"
        ok = run(["bazel", "run",
                  "//experimental/overhead_matching/swag/scripts:merge_landmark_feathers",
                  "--", "--inputs", *[str(f) for f in osm_feathers],
                  "--output", str(combined),
                  "--dedupe_tolerance_m", str(args.dedupe_tolerance_m)],
                 f"[5 OSM merge] {name} ({len(osm_feathers)} extracts)",
                 args.dry_run, cwd=CROSSVIEW_REPO)
        if not ok:
            return False
        osm_out = combined.with_suffix("")
    else:
        osm_out = osm_feathers[0].with_suffix("")

    if not cfg.get("enc_state"):
        print(f"  no NOAA ENC coverage for {name} (non-US waters) — OSM-only catalog. "
              f"Fixed navaids will be sparser here; recorded in landmarks/PROVENANCE.json")
        if not args.dry_run:
            _write_provenance(dataset_dir, name, pbf_paths, (west, south, east, north),
                              osm_specs=osm_specs,
                              enc_state=None, merged=False)
            _link_pipeline_feather(dataset_dir, osm_out.with_suffix(".feather"), args)
        return True

    enc_out = sources / f"enc_{name}_v1"
    ok = run(["bazel", "run",
              "//experimental/overhead_matching/swag/scripts:download_enc_cells",
              "--", "--catalog_state", cfg["enc_state"],
              "--bbox", str(west), str(south), str(east), str(north)],
             f"[5 ENC dl] {name} ({cfg['enc_state']})", args.dry_run, cwd=CROSSVIEW_REPO)
    if not ok:
        return False

    # extract_landmarks_from_enc needs explicit cell names; discover what the
    # download left in the ENC root for this bbox.
    cells = _cells_for_bbox(args.enc_root, (west, south, east, north), args.dry_run)
    if not cells:
        print(f"  WARNING: no ENC cells found covering {name}'s bbox; "
              f"continuing with an OSM-only catalog")
        if not args.dry_run:
            _write_provenance(dataset_dir, name, pbf_paths, (west, south, east, north),
                              osm_specs=osm_specs,
                              enc_state=cfg["enc_state"], merged=False)
            _link_pipeline_feather(dataset_dir, osm_out.with_suffix(".feather"), args)
        return True
    print(f"  ENC cells: {' '.join(cells)}")

    ok = run(["bazel", "run",
              "//experimental/overhead_matching/swag/scripts:extract_landmarks_from_enc",
              "--", "--enc_root", str(args.enc_root), "--cells", *cells,
              "--bbox", str(west), str(south), str(east), str(north),
              "--output_path", str(enc_out)],
             f"[5 ENC extract] {name}", args.dry_run, cwd=CROSSVIEW_REPO)
    if not ok:
        return False

    merged = dataset_dir / "landmarks" / f"{name}_osm_enc_v1.feather"
    ok = run(["bazel", "run",
              "//experimental/overhead_matching/swag/scripts:merge_landmark_feathers",
              "--", "--inputs", str(osm_out.with_suffix(".feather")),
              str(enc_out.with_suffix(".feather")),
              "--output", str(merged),
              "--dedupe_tolerance_m", str(args.dedupe_tolerance_m)],
             f"[5 MERGE] {name}", args.dry_run, cwd=CROSSVIEW_REPO)
    if not ok:
        return False
    if not args.dry_run:
        _write_provenance(dataset_dir, name, pbf_paths, (west, south, east, north),
                              osm_specs=osm_specs,
                          enc_state=cfg["enc_state"], merged=True, cells=cells)
        _link_pipeline_feather(dataset_dir, merged, args)
    return True


def _cells_for_bbox(enc_root: Path, bbox, dry_run: bool):
    """ENC cell names under enc_root whose data covers the bbox.

    download_enc_cells --catalog_state fetches by bbox but does not report which
    cells it chose, so read them back off disk. Band 5 (harbour) cells are the
    ones with the landmark detail we want.
    """
    if dry_run or not enc_root.exists():
        return []
    root = enc_root / "ENC_ROOT"
    search = root if root.exists() else enc_root
    return sorted(d.name for d in search.iterdir()
                  if d.is_dir() and (d / f"{d.name}.000").exists()
                  and d.name.startswith("US5"))


def _link_pipeline_feather(dataset_dir: Path, feather: Path, args):
    """Expose the chosen catalog as landmarks/<landmark_version>.feather.

    vigor_dataset.py opens landmarks/<version>.feather by exact name, so give it
    a stable one instead of requiring a per-dataset --landmark_version.
    """
    if not feather.exists():
        print(f"  WARNING: expected {feather} to exist; not linking")
        return
    link = dataset_dir / "landmarks" / f"{args.landmark_version}.feather"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(Path(feather).relative_to(link.parent))
    print(f"  linked landmarks/{link.name} -> {link.readlink()}")


def _write_provenance(dataset_dir: Path, name: str, pbf_paths, bbox,
                      enc_state, merged: bool, cells=None, osm_specs=None):
    out = dataset_dir / "landmarks" / "PROVENANCE.json"
    out.write_text(json.dumps({
        "dataset": name,
        "bbox_wsen": list(bbox),
        "osm_pbf": [Path(p).name for p in (pbf_paths if isinstance(pbf_paths, (list, tuple)) else [pbf_paths])],
        # Geofabrik-relative specs, so plot_landmarks can draw the clip boundaries.
        "osm_specs": list(osm_specs or []),
        "enc_state": enc_state,
        "enc_cells": cells or [],
        "enc_available": bool(merged),
        "note": ("OSM+ENC merged catalog" if merged else
                 "OSM-only catalog: NOAA ENC covers US waters only, so fixed "
                 "navaids (beacons, buoys, lights) are substantially less "
                 "complete here than in the US datasets"),
    }, indent=2))
    print(f"  wrote {out}")


def stage_pinhole(name, cfg, args):
    dataset_dir = args.output_base / name
    meta_path = dataset_dir / "pipeline_metadata.json"
    is_equirect = cfg["pano"]
    if meta_path.exists():
        # Trust the converted dataset over the registry: the registry's flag is
        # a convenience, the metadata records what was actually written.
        is_equirect = json.loads(meta_path.read_text()).get("is_equirectangular", is_equirect)
    if not is_equirect:
        print(f"\n=== [6 PINHOLE] {name}: skipped (perspective capture — a "
              f"limited-FOV frame is already a single view; bearings come from "
              f"intrinsics.csv)")
        return True
    # Pinhole faces are a versioned artifact (artifacts/pinhole_images/<name>/
    # v1/ + manifest.json), not part of the frozen dataset contract — see
    # docs/farfield-data-organization.md.
    out_dir = args.pinhole_base / name / "v1"
    ok = run(["bazel", "run",
              "//experimental/overhead_matching/swag/scripts:panorama_to_pinhole",
              "--", str(dataset_dir / "panorama"),
              str(out_dir),
              "--num_workers", str(args.convert_workers),
              "--res_x", str(args.pinhole_res)],
             f"[6 PINHOLE] {name}", args.dry_run, cwd=CROSSVIEW_REPO)
    if ok and not args.dry_run:
        (out_dir / "manifest.json").write_text(json.dumps({
            "kind": "pinhole_images",
            "dataset": name,
            "version": "v1",
            "generator": "//experimental/overhead_matching/swag/scripts:"
                         "panorama_to_pinhole (stage 6)",
            "config": {
                "faces": ["yaw_000", "yaw_090", "yaw_180", "yaw_270"],
                "res_x": args.pinhole_res,
            },
        }, indent=2))
    return ok


def stage_trim(name, cfg, args):
    """Trim the catalog to landmarks plausibly observable from the water.

    Runs on the merged feather and writes a sibling `<version>_trimmed.feather`,
    leaving the untrimmed catalog in place: the trim encodes judgement about what
    is visible at range, and that is worth being able to revisit without
    re-extracting.
    """
    ds = args.output_base / name
    link = ds / "landmarks" / f"{args.landmark_version}.feather"
    if not link.exists() and not args.dry_run:
        print(f"  ERROR: {link} missing — run stage 5 (OSM) first")
        return False
    target = link.resolve() if link.exists() else link
    out = target.with_name(target.stem + "_trimmed.feather")
    ok = run(["bazel", "run",
              "//experimental/overhead_matching/swag/scripts:trim_landmark_feather",
              "--", "--input", str(target), "--output", str(out)],
             f"[7 TRIM] {name}", args.dry_run, cwd=CROSSVIEW_REPO)
    if ok and not args.dry_run and out.exists():
        trimmed_link = ds / "landmarks" / f"{args.landmark_version}_trimmed.feather"
        if trimmed_link.exists() or trimmed_link.is_symlink():
            trimmed_link.unlink()
        trimmed_link.symlink_to(out.relative_to(trimmed_link.parent))
        print(f"  linked landmarks/{trimmed_link.name} -> {trimmed_link.readlink()}")
    return ok


def stage_plot(name, cfg, args):
    """Render the landmark coverage figure and run the emptiness checks."""
    ds = args.output_base / name
    out = ds / "landmarks" / "landmark_coverage.png"
    return run(["bazel", "run",
                "//experimental/overhead_matching/swag/scripts:plot_landmarks",
                "--", str(ds), "-o", str(out)],
               f"[8 PLOT] {name}", args.dry_run, cwd=CROSSVIEW_REPO)


def stage_timelapse(name, cfg, args):
    """trajectory.png + gps_timelapse.mp4, as the self-collected datasets carry.

    The video is the cheapest check that frames and positions have not come
    apart: a mis-stitched trajectory, a non-temporal ordering or a reversed run
    is obvious on sight and invisible in a summary table.
    """
    return run(["bazel", "run",
                "//experimental/overhead_matching/swag/scripts:make_dataset_timelapse",
                "--", "--dataset_path", str(args.output_base / name),
                "--fps", str(args.timelapse_fps),
                "--max_frames", str(args.timelapse_max_frames)],
               f"[4 TIMELAPSE] {name}", args.dry_run, cwd=CROSSVIEW_REPO)


def stage_audit(name, cfg, args):
    """Contract audit; non-zero exit if anything fails."""
    return run(["bazel", "run",
                "//experimental/overhead_matching/swag/scripts:audit_dataset",
                "--", str(args.output_base / name)],
               f"[9 AUDIT] {name}", args.dry_run, cwd=CROSSVIEW_REPO)


STAGES = {1: stage_resolve, 2: stage_download, 3: stage_convert,
          4: stage_timelapse, 5: stage_landmarks, 6: stage_pinhole,
          7: stage_trim, 8: stage_plot, 9: stage_audit}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", default="pilot",
                   help="'all', 'pilot', 'pano', 'perspective', or comma-separated names")
    p.add_argument("--stages", default="1,2,3,4,5,6,7,8,9",
                   help="Comma-separated stage numbers (default: all)")
    p.add_argument("--output_base", type=Path, default=DEFAULT_OUTPUT_BASE)
    p.add_argument("--manifest_dir", dest="manifest_dir_arg", type=Path,
                   default=DEFAULT_MANIFEST_DIR,
                   help="Stage-1 stitched manifests (raw_material lane)")
    p.add_argument("--raw_dir", dest="raw_dir_arg", type=Path,
                   default=DEFAULT_RAW_DIR,
                   help="Stage-2 staged originals (raw_material lane; "
                        "--prune_raw removes them after conversion)")
    p.add_argument("--pinhole_base", type=Path, default=DEFAULT_PINHOLE_BASE)
    p.add_argument("--enc_root", type=Path,
                   default=Path("/data/overhead_matching/datasets/enc_cells"))
    p.add_argument("--max_width", type=int, default=4096,
                   help="Cap on stored image width (default: 4096)")
    p.add_argument("--min_spacing_m", type=float, default=5.0,
                   help="Drop frames closer than this along the track. These are "
                        "mostly video extractions at 1-30 fps, so consecutive "
                        "frames are metres apart and often share a GPS fix "
                        "outright — a frame whose recorded position belongs to a "
                        "neighbour adds pixels but no usable geometry. On the "
                        "Folkestone crossing this takes 10711 frames to 399 with "
                        "no loss of distinct positions (default: 5)")
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--pinhole_res", type=int, default=2048,
                   help="Pinhole face resolution (2048 matches boston_harbor_leg1)")
    p.add_argument("--landmark_buffer_km", type=float, default=LANDMARK_BUFFER_KM,
                   help="Buffer around the trajectory for the landmark catalog. "
                        "Large by default because far-field landmarks on water "
                        f"are visible well beyond the track (default: {LANDMARK_BUFFER_KM} km)")
    p.add_argument("--landmark_version", default="v1",
                   help="Name of the landmarks/<version>.feather link")
    p.add_argument("--dedupe_tolerance_m", type=float, default=10.0,
                   help="Merge identical-tag features whose geometries touch "
                        "within this distance (default: 10, as used for boston_harbor)")
    p.add_argument("--stitch_time", type=float, default=300.0)
    p.add_argument("--stitch_dist", type=float, default=100.0)
    p.add_argument("--window_hours", type=float, default=36.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--convert_workers", type=int, default=8)
    p.add_argument("--timelapse_fps", type=int, default=15)
    p.add_argument("--timelapse_max_frames", type=int, default=1500,
                   help="Subsample the timelapse above this many frames (0 = all)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--prune_raw", action="store_true",
                   help="Delete _raw/<name> after a successful conversion. "
                        "Recommended when running many trajectories: the "
                        "staged originals are ~15x larger than the stored "
                        "4096-wide result")
    p.add_argument("--force", action="store_true",
                   help="Redo stage 1 even if a manifest already exists")
    p.add_argument("--node_margin_deg", type=float, default=0.1,
                   help="Bound the way-geometry node index to bbox + this margin "
                        "(~11 km). This is what makes a national extract cheap: "
                        "whole France goes from 28 GB and climbing to 3.0 GB. Costs "
                        "a couple of multipolygons whose rings cross the margin. "
                        "Negative disables")
    p.add_argument("--max_pbf_mb", type=int, default=MAX_PBF_MB,
                   help=f"Refuse OSM extracts larger than this; the extractor "
                        f"indexes the whole file in RAM (default: {MAX_PBF_MB})")
    p.add_argument("--allow_large_pbf", action="store_true",
                   help="Override the PBF size limit (verify memory headroom first)")
    p.add_argument("--extract_mem_cap_gb", type=int, default=EXTRACT_MEM_CAP_GB,
                   help=f"Hard cgroup memory ceiling for landmark extraction; 0 "
                        f"disables (default: {EXTRACT_MEM_CAP_GB})")
    p.add_argument("--skip_coverage_check", action="store_true",
                   help="Proceed even if the OSM extracts do not cover the bbox "
                        "(builds a knowingly partial landmark catalog)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    sel = args.trajectories
    if sel == "all":
        names = list(collectable())
    elif sel == "pilot":
        names = list(PILOT)
    elif sel == "pano":
        names = [k for k, v in collectable().items() if v["pano"]]
    elif sel == "perspective":
        names = [k for k, v in collectable().items() if not v["pano"]]
    else:
        names = [n.strip() for n in sel.split(",") if n.strip()]
    unknown = [n for n in names if n not in TRAJECTORIES]
    if unknown:
        print(f"ERROR: unknown trajectory name(s): {unknown}")
        print(f"Known: {', '.join(TRAJECTORIES)}")
        return 1

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"ERROR: unknown stage(s) {bad}; valid are {sorted(STAGES)}")
        return 1

    args.manifest_dir = args.manifest_dir_arg
    args.raw_dir = args.raw_dir_arg
    if not args.dry_run:
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
        args.raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Trajectories: {len(names)} ({', '.join(names)})")
    print(f"Stages: {stages}")
    print(f"Output: {args.output_base}")

    failures = []
    for name in names:
        cfg = TRAJECTORIES[name]
        print("\n" + "=" * 74)
        print(f"{name}  [{'360' if cfg['pano'] else 'perspective'}]  {cfg['note']}")
        print("=" * 74)
        for s in stages:
            if not STAGES[s](name, cfg, args):
                failures.append((name, s))
                print(f"  stopping {name} at stage {s}")
                break

    print("\n" + "=" * 74)
    if failures:
        print(f"{len(failures)} failure(s):")
        for name, s in failures:
            print(f"  {name} failed at stage {s}")
        return 1
    print(f"All {len(names)} trajectory/trajectories completed stages {stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
