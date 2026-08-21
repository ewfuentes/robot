#!/usr/bin/env python3
"""Collect the far-field Mapillary trajectories end to end.

Stages, per trajectory:
  1 RESOLVE   seed pKey -> stitched manifest             (seed_to_trajectory, in-process)
  2 DOWNLOAD  manifest -> ordered jpg+json staging       (extract_stitch, in-process)
  3 CONVERT   staging -> dataset dir                     (mapillary_to_vigor, in-process)
  4 TIMELAPSE trajectory.png + gps_timelapse.mp4         (dataset_tools.make_dataset_timelapse)
  5 OSM       landmark feather (+ENC where NOAA covers)  (extract_landmarks_* via bazel)
  6 PINHOLE   4 yaw faces, equirectangular only          (panorama_to_pinhole via bazel)
  7 TRIM      observable-from-water catalog subset       (dataset_tools.trim_catalog)
  8 PLOT      landmark coverage figure + gap checks      (dataset_tools.plot_landmarks)
  9 AUDIT     dataset contract audit                     (farfield.audit_dataset, in-process)

Stages that are collection code run IN-PROCESS by importing the ported modules
(the old orchestrator shelled out to `bazel run` inside a hardcoded
~/code/robot-farfield-crossview checkout, which does not exist). A subprocess
survives only where it is genuinely needed — the landmark extractors run under
a cgroup memory cap, and panorama_to_pinhole lives in another package — and
those resolve the repo root from $BUILD_WORKSPACE_DIRECTORY (set by `bazel
run`) and error when it is absent. The dataset_tools stages (4, 7, 8) import
`farfield.dataset_tools.*`; until that PR lands in this stack they fail with a
pointed message rather than running some stale copy.

TIMELAPSE sits directly after CONVERT (moved 2026-08-17, was stage 8) because
it is the input to the human triage pass, and triage is the cheapest gate:
7 of the first 21 collected trajectories were rejected on the timelapse for
faults no audit sees (unfixed mounts, panning cameras). Rendering it before
the landmark stages means a rejected trajectory costs one mp4, not an
Overpass catalog + pinhole render.

Stage 6 is deliberately skipped for perspective captures: a limited-FOV frame
is already a single view, and its azimuth mapping comes from intrinsics.csv
rather than from four fixed 90-degree faces.

Layout (lifecycle lanes; defaults derive from farfield.paths.default_root()):
    <output_base>/<name>/       dataset (panorama/, frames_gps.csv, landmarks/, ...)
    <manifest_dir>/<name>.json  stage-1 stitched manifest
    <raw_dir>/<name>/           staged originals, prunable after stage 3
"""

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    audit_dataset,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.collection import (
    extract_stitch,
    mapillary_to_vigor,
    seed_to_trajectory,
)
from experimental.overhead_matching.swag.farfield.collection.farfield_trajectories import (
    PILOT, REJECTED_TRAJECTORIES, TRAJECTORIES, collectable,
)
from experimental.overhead_matching.swag.farfield.collection.geometry_helpers import (
    bbox_from_dataset,
)
from experimental.overhead_matching.swag.farfield.collection.pbf_coverage import (
    check_coverage,
)

SCRIPTS = "//experimental/overhead_matching/swag/scripts"

# `bazel run` must be invoked from the source workspace, not the runfiles
# tree. Only the stages that genuinely need a subprocess consult this.
WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

# extract_landmarks_historical builds a node-location index for the WHOLE pbf in
# RAM before it can filter by bbox, so peak memory tracks file size and not the
# area requested. Whole-France (4.7 GB) reached 28 GB RSS growing 1.2 GB/min and
# took the machine down. Two independent guards, because one is advisory and the
# other is enforced:
#   * refuse files over --max_pbf_mb (generous now: with the dict schema and the
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
        print("  WARNING: systemd-run unavailable; running without a memory cap")
        return cmd
    return ["systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={cap_gb}G", "-p", "MemorySwapMax=0", "--"] + cmd


def run_bazel(target, target_args, desc, args, mem_cap_gb=0):
    """Run a bazel target as a subprocess from the source workspace.

    Only for stages a subprocess genuinely serves (memory-capped extraction,
    tools in other packages). Requires $BUILD_WORKSPACE_DIRECTORY.
    """
    cmd = mem_capped(["bazel", "run", target, "--"] + [str(a) for a in target_args],
                     mem_cap_gb)
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== {desc}")
    print("  $ " + " ".join(str(c) for c in cmd))
    if args.dry_run:
        return True
    if WORKSPACE is None:
        print("  ERROR: BUILD_WORKSPACE_DIRECTORY is unset — this stage runs "
              "bazel targets as subprocesses and must itself be started via "
              "`bazel run`, from the source workspace.")
        return False
    result = subprocess.run(cmd, cwd=WORKSPACE)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {desc}")
        return False
    return True


def run_module(main_fn, argv, desc, args):
    """Run a ported module's main(argv) in-process, subprocess-style logging."""
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== {desc}")
    print("  $ " + main_fn.__module__ + ".main " + " ".join(str(a) for a in argv))
    if args.dry_run:
        return True
    try:
        rc = main_fn([str(a) for a in argv])
    except SystemExit as exc:  # argparse errors and explicit exits
        rc = exc.code
    except Exception as exc:  # a failed stage must not kill the batch loop
        print(f"  FAILED ({type(exc).__name__}: {exc}): {desc}")
        return False
    if rc not in (None, 0):
        print(f"  FAILED (exit {rc}): {desc}")
        return False
    return True


def dataset_tools_module(module_name: str):
    """Import farfield.dataset_tools.<module>, or explain why it is absent.

    Stages 4/7/8 belong to the dataset-tools PR, which lands in the same
    stack. Until it does, this errors loudly instead of silently running a
    stale script from the checkpoint branch.
    """
    qualified = ("experimental.overhead_matching.swag.farfield.dataset_tools."
                 + module_name)
    try:
        return importlib.import_module(qualified)
    except ImportError as exc:
        print(f"  ERROR: cannot import {qualified}: dataset-tools PR not yet "
              f"merged ({exc}). This stage runs once reorg/14-dataset-tools "
              f"lands in the stack.")
        return None


def stage_resolve(name, cfg, args):
    manifest = args.manifest_dir / f"{name}.json"
    if manifest.exists() and not args.force:
        print(f"\n=== [1 RESOLVE] {name}: manifest exists, skipping "
              f"(--force to redo)")
        return True
    return run_module(
        seed_to_trajectory.main,
        ["--seed_pkey", cfg["seed_pkey"], "--name", name,
         "--output", manifest,
         "--stitch_time", args.stitch_time,
         "--stitch_dist", args.stitch_dist,
         "--window_hours", args.window_hours,
         "--workers", args.workers],
        f"[1 RESOLVE] {name}", args)


def stage_download(name, cfg, args):
    manifest = args.manifest_dir / f"{name}.json"
    if not manifest.exists() and not args.dry_run:
        print(f"  ERROR: {manifest} missing — run stage 1 first")
        return False
    return run_module(
        extract_stitch.main,
        ["--manifest", manifest, "--sequence", name,
         "--output", args.raw_dir / name,
         "--workers", args.workers,
         "--max_width", args.max_width,
         "--min_spacing_m", args.min_spacing_m],
        f"[2 DOWNLOAD] {name}", args)


def stage_convert(name, cfg, args):
    argv = ["--sequence_dir", args.raw_dir / name,
            "--vigor_dir", args.output_base / name,
            "--dataset_name", name,
            "--num_workers", args.convert_workers,
            "--jpeg_quality", args.jpeg_quality,
            "--resize", args.max_width,
            # Frames were already decimated at download; converting must not
            # thin them again.
            "--min_spacing", 0,
            "--heading_source", args.heading_source,
            "--max_heading_error_deg", args.max_heading_error_deg,
            "--max_perspective_offset_std_deg",
            args.max_perspective_offset_std_deg,
            "--max_heading_source_disagreement_deg",
            args.max_heading_source_disagreement_deg]
    if args.visualize:
        argv.append("--visualize")
    ok = run_module(mapillary_to_vigor.main, argv, f"[3 CONVERT] {name}", args)
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
            size_gb = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file()) / 1e9
            shutil.rmtree(raw)
            print(f"  pruned {raw} ({size_gb:.2f} GB) — {len(converted)} converted "
                  f"images retained")
    return ok


def stage_timelapse(name, cfg, args):
    """trajectory.png + gps_timelapse.mp4, as the self-collected datasets carry.

    The video is the cheapest check that frames and positions have not come
    apart: a mis-stitched trajectory, a non-temporal ordering or a reversed run
    is obvious on sight and invisible in a summary table.
    """
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== [4 TIMELAPSE] {name}")
    if args.dry_run:
        return True
    mod = dataset_tools_module("make_dataset_timelapse")
    if mod is None:
        return False
    return run_module(
        mod.main,
        ["--dataset_path", args.output_base / name,
         "--fps", args.timelapse_fps,
         "--max_frames", args.timelapse_max_frames],
        f"[4 TIMELAPSE] {name}", args)


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
        matches = sorted(args.osm_cache_dir.glob(wanted.replace("-latest", "-*")))
        if not matches:
            if args.dry_run:
                pbf_paths.append(args.osm_cache_dir / wanted)
                continue
            print(f"  ERROR: no OSM extract for {spec} in {args.osm_cache_dir} "
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
    # (check_coverage is a package import with a BUILD dep now; the old bare
    # `from pbf_coverage import check_coverage` never resolved under bazel, so
    # this gate — the one that catches the 99.6%-catalog-loss case — was
    # unreachable in exactly the environment the orchestrator runs in.)
    if not args.skip_coverage_check:
        ok, msg, cov_details = check_coverage(
            osm_specs, (west, south, east, north),
            cache_dir=args.osm_cache_dir / "poly",
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
    if not args.dry_run:
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
        size_mb = pbf_path.stat().st_size / 1e6 if pbf_path.exists() else 0
        ok = run_bazel(
            f"{SCRIPTS}:extract_landmarks_historical",
            ["--pbf_file", pbf_path,
             "--bbox", west, south, east, north,
             "--node_margin_deg", args.node_margin_deg,
             "--output_path", out],
            f"[5 OSM] {name} ({pbf_path.name}, {size_mb:.0f} MB, cap "
            f"{args.extract_mem_cap_gb} GB)", args,
            mem_cap_gb=args.extract_mem_cap_gb)
        if not ok:
            return False
        osm_feathers.append(out.with_suffix(".feather"))

    if len(osm_feathers) > 1:
        combined = sources / f"osm_{name}_v1.feather"
        ok = run_bazel(
            f"{SCRIPTS}:merge_landmark_feathers",
            ["--inputs", *osm_feathers, "--output", combined,
             "--dedupe_tolerance_m", args.dedupe_tolerance_m],
            f"[5 OSM merge] {name} ({len(osm_feathers)} extracts)", args)
        if not ok:
            return False
        osm_out = combined.with_suffix("")
    else:
        osm_out = osm_feathers[0].with_suffix("")

    if not cfg.get("enc_state"):
        print(f"  no NOAA ENC coverage for {name} (non-US waters) — OSM-only catalog. "
              f"Fixed navaids will be sparser here; recorded in the landmarks manifest")
        if not args.dry_run:
            _write_provenance(dataset_dir, name, pbf_paths, (west, south, east, north),
                              osm_specs=osm_specs,
                              enc_state=None, merged=False, args=args)
            _link_pipeline_feather(dataset_dir, osm_out.with_suffix(".feather"), args)
        return True

    enc_out = sources / f"enc_{name}_v1"
    ok = run_bazel(
        f"{SCRIPTS}:download_enc_cells",
        ["--catalog_state", cfg["enc_state"],
         "--bbox", west, south, east, north],
        f"[5 ENC dl] {name} ({cfg['enc_state']})", args)
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
                              enc_state=cfg["enc_state"], merged=False, args=args)
            _link_pipeline_feather(dataset_dir, osm_out.with_suffix(".feather"), args)
        return True
    print(f"  ENC cells: {' '.join(cells)}")

    ok = run_bazel(
        f"{SCRIPTS}:extract_landmarks_from_enc",
        ["--enc_root", args.enc_root, "--cells", *cells,
         "--bbox", west, south, east, north,
         "--output_path", enc_out],
        f"[5 ENC extract] {name}", args)
    if not ok:
        return False

    merged = dataset_dir / "landmarks" / f"{name}_osm_enc_v1.feather"
    ok = run_bazel(
        f"{SCRIPTS}:merge_landmark_feathers",
        ["--inputs", osm_out.with_suffix(".feather"), enc_out.with_suffix(".feather"),
         "--output", merged,
         "--dedupe_tolerance_m", args.dedupe_tolerance_m],
        f"[5 MERGE] {name}", args)
    if not ok:
        return False
    if not args.dry_run:
        _write_provenance(dataset_dir, name, pbf_paths, (west, south, east, north),
                          osm_specs=osm_specs,
                          enc_state=cfg["enc_state"], merged=True, cells=cells,
                          args=args)
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

    Dataset loaders open landmarks/<version>.feather by exact name, so give it
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
                      enc_state, merged: bool, args, cells=None, osm_specs=None):
    """Record how landmarks/ was produced, via the one manifest writer."""
    provenance.write(
        dataset_dir / "landmarks",
        generator="//experimental/overhead_matching/swag/farfield/"
                  "collection:run_farfield_collection (stage 5)",
        inputs={f"pbf_{i}": Path(p).name for i, p in enumerate(
            pbf_paths if isinstance(pbf_paths, (list, tuple)) else [pbf_paths])},
        config={
            "dataset": name,
            "bbox_wsen": list(bbox),
            # Geofabrik-relative specs, so the coverage plot can draw the
            # clip boundaries.
            "osm_specs": list(osm_specs or []),
            "enc_state": enc_state,
            "enc_cells": cells or [],
            "enc_available": bool(merged),
            "dedupe_tolerance_m": args.dedupe_tolerance_m,
            "node_margin_deg": args.node_margin_deg,
        },
        notes=("OSM+ENC merged catalog" if merged else
               "OSM-only catalog: NOAA ENC covers US waters only, so fixed "
               "navaids (beacons, buoys, lights) are substantially less "
               "complete here than in the US datasets"),
    )
    print(f"  wrote {dataset_dir / 'landmarks' / provenance.MANIFEST_NAME}")


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
    # <version>/ + manifest.json), not part of the frozen dataset contract.
    out_dir = args.pinhole_base / name / args.pinhole_version
    ok = run_bazel(
        f"{SCRIPTS}:panorama_to_pinhole",
        [dataset_dir / "panorama", out_dir,
         "--num_workers", args.convert_workers,
         "--res_x", args.pinhole_res],
        f"[6 PINHOLE] {name}", args)
    if ok and not args.dry_run:
        provenance.write(
            out_dir,
            generator=f"{SCRIPTS}:panorama_to_pinhole "
                      "(collection stage 6)",
            inputs={"panorama": (dataset_dir / "panorama").resolve()},
            config={"faces": ["yaw_000", "yaw_090", "yaw_180", "yaw_270"],
                    "res_x": args.pinhole_res},
            extra={"kind": "pinhole_images", "dataset": name,
                   "version": args.pinhole_version},
        )
    return ok


def stage_trim(name, cfg, args):
    """Trim the catalog to landmarks plausibly observable from the water.

    Runs on the merged feather and writes a sibling `<version>_trimmed.feather`,
    leaving the untrimmed catalog in place: the trim encodes judgement about what
    is visible at range, and that is worth being able to revisit without
    re-extracting.

    The recall guard is mandatory: the trim tool measures every rule against a
    matched set (--matched_from) and/or a pairing positive set (--positive_set)
    and refuses rules that drop either. Running with neither reference silently
    skips the one check that has killed two "obviously right" rules, so this
    stage refuses outright rather than trimming unguarded. (The old
    orchestrator passed neither, so the guard never ran on a collected
    dataset.)
    """
    if not args.matched_from and not args.positive_set:
        print(f"\n=== [7 TRIM] {name}: REFUSING to trim without a recall "
              f"reference. Pass --matched_from <m9 run dir> (repeatable) "
              f"and/or --positive_set <positive_set.json>; the recall guard "
              f"is load-bearing and does not run without them.")
        return False
    ds = args.output_base / name
    link = ds / "landmarks" / f"{args.landmark_version}.feather"
    if not link.exists() and not args.dry_run:
        print(f"  ERROR: {link} missing — run stage 5 (OSM) first")
        return False
    target = link.resolve() if link.exists() else link
    out = target.with_name(target.stem + "_trimmed.feather")
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== [7 TRIM] {name}")
    if args.dry_run:
        return True
    mod = dataset_tools_module("trim_catalog")
    if mod is None:
        return False
    argv = ["--input", target, "--output", out,
            "--min_building_area_m2", args.trim_min_building_area_m2,
            "--min_building_levels", args.trim_min_building_levels,
            "--confidence_floor", args.trim_confidence_floor]
    for matched in args.matched_from or []:
        argv += ["--matched_from", matched]
    if args.positive_set:
        argv += ["--positive_set", args.positive_set]
    ok = run_module(mod.main, argv, f"[7 TRIM] {name}", args)
    if ok and out.exists():
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
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}=== [8 PLOT] {name}")
    if args.dry_run:
        return True
    mod = dataset_tools_module("plot_landmarks")
    if mod is None:
        return False
    return run_module(mod.main, [ds, "-o", out], f"[8 PLOT] {name}", args)


def stage_audit(name, cfg, args):
    """Contract audit (farfield.audit_dataset, in-process); fails the stage on
    any violation."""
    return run_module(audit_dataset.main, [args.output_base / name],
                      f"[9 AUDIT] {name}", args)


STAGES = {1: stage_resolve, 2: stage_download, 3: stage_convert,
          4: stage_timelapse, 5: stage_landmarks, 6: stage_pinhole,
          7: stage_trim, 8: stage_plot, 9: stage_audit}


def main(argv=None) -> int:
    root = paths_lib.default_root()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", required=True,
                   help="'all', 'pilot', 'pano', 'perspective', or "
                        "comma-separated registry names (the old default was "
                        "'pilot')")
    p.add_argument("--stages", default="1,2,3,4,5,6,7,8,9",
                   help="Comma-separated stage numbers (default: all)")

    # Lifecycle lanes: defaults derive from the one disk-layout owner
    # (farfield.paths), overridable per flag; no hardcoded /data literals.
    p.add_argument("--output_base", type=Path, default=root / "datasets",
                   help="Dataset lane (default: <farfield_root>/datasets)")
    p.add_argument("--manifest_dir", type=Path,
                   default=root / "raw_material" / "mapillary_manifests",
                   help="Stage-1 stitched manifests (default: "
                        "<farfield_root>/raw_material/mapillary_manifests)")
    p.add_argument("--raw_dir", type=Path,
                   default=root / "raw_material" / "mapillary_raw",
                   help="Stage-2 staged originals (default: "
                        "<farfield_root>/raw_material/mapillary_raw; "
                        "--prune_raw removes them after conversion)")
    p.add_argument("--pinhole_base", type=Path,
                   default=root / "artifacts" / paths_lib.PINHOLE_IMAGES,
                   help="Pinhole artifact lane (default: "
                        "<farfield_root>/artifacts/pinhole_images)")
    p.add_argument("--osm_cache_dir", type=Path, required=True,
                   help="Directory of downloaded Geofabrik .osm.pbf extracts; "
                        ".poly boundaries cache under <osm_cache_dir>/poly "
                        "(the old hardcoded location was ~/scratch/"
                        "osm_downloads)")
    p.add_argument("--enc_root", type=Path, required=True,
                   help="NOAA ENC cell root used by download_enc_cells / "
                        "extract_landmarks_from_enc (the old hardcoded "
                        "location was /data/overhead_matching/datasets/"
                        "enc_cells)")

    # Stage 1-3 parameters (assumption-carrying: required, old values quoted).
    p.add_argument("--stitch_time", type=float, required=True,
                   help="Max seam time gap in seconds (the old default was 300)")
    p.add_argument("--stitch_dist", type=float, required=True,
                   help="Max seam spatial gap floor in meters (the old default "
                        "was 100)")
    p.add_argument("--window_hours", type=float, required=True,
                   help="Sibling-sequence capture-time window (the old default "
                        "was 36)")
    p.add_argument("--max_width", type=int, required=True,
                   help="Cap on stored image width, also the convert resize; "
                        "0 disables (the old default was 4096)")
    p.add_argument("--min_spacing_m", type=float, required=True,
                   help="Drop frames closer than this along the track at "
                        "download time. These are mostly video extractions at "
                        "1-30 fps, so consecutive frames are metres apart and "
                        "often share a GPS fix outright. On the Folkestone "
                        "crossing 5 m takes 10711 frames to 399 with no loss "
                        "of distinct positions (the old default was 5)")
    p.add_argument("--jpeg_quality", type=int, required=True,
                   help="JPEG quality for stored frames (the old default was 95)")
    p.add_argument("--heading_source", choices=("auto", "computed", "compass"),
                   required=True,
                   help="Heading source for the converter (the old default "
                        "was 'auto')")
    p.add_argument("--max_heading_error_deg", type=float, required=True,
                   help="Converter heading-reliability gate (the old default "
                        "was 10)")
    p.add_argument("--max_perspective_offset_std_deg", type=float, required=True,
                   help="Converter hand-held-vs-fixed-mount report threshold "
                        "(the old default was 45)")
    p.add_argument("--max_heading_source_disagreement_deg", type=float,
                   required=True,
                   help="Converter SfM-vs-magnetometer warning threshold (the "
                        "old default was 25)")

    # Stage 5-8 parameters.
    p.add_argument("--pinhole_res", type=int, required=True,
                   help="Pinhole face resolution (the old default was 2048, "
                        "matching boston_harbor_leg1)")
    p.add_argument("--pinhole_version", required=True,
                   help="Version directory name under the pinhole artifact "
                        "lane (the old code hardcoded 'v1')")
    p.add_argument("--landmark_buffer_km", type=float, required=True,
                   help="Buffer around the trajectory for the landmark "
                        "catalog; per-trajectory registry overrides win. Large "
                        "by design: far-field landmarks on water are visible "
                        "well beyond the track — from a ferry deck the sea "
                        "horizon is ~11 km but a 100 m cliff or harbour crane "
                        "stays visible 30-40 km out (the old default was 25)")
    p.add_argument("--landmark_version", required=True,
                   help="Name of the landmarks/<version>.feather link (the "
                        "old default was 'v1')")
    p.add_argument("--dedupe_tolerance_m", type=float, required=True,
                   help="Merge identical-tag features whose geometries touch "
                        "within this distance (the old default was 10, as "
                        "used for boston_harbor)")
    p.add_argument("--node_margin_deg", type=float, required=True,
                   help="Bound the extractor's way-geometry node index to "
                        "bbox + this margin. This is what makes a national "
                        "extract cheap: whole France goes from 28 GB and "
                        "climbing to 3.0 GB, at the cost of a couple of "
                        "multipolygons whose rings cross the margin. One "
                        "default story: required here, forwarded verbatim to "
                        "the extractor — no -1 sentinel (the old default was "
                        "0.1, ~11 km)")
    p.add_argument("--trim_min_building_area_m2", type=float, required=True,
                   help="Stage-7 trim: drop untagged buildings smaller than "
                        "this footprint (the old, Boston-tuned default was "
                        "2000)")
    p.add_argument("--trim_min_building_levels", type=float, required=True,
                   help="Stage-7 trim: keep small buildings only at/above "
                        "this many levels (the old, Boston-tuned default was 6)")
    p.add_argument("--trim_confidence_floor", type=float, required=True,
                   help="Stage-7 trim: positive-set confidence floor (the "
                        "old default was 0.5)")
    p.add_argument("--matched_from", type=Path, action="append",
                   help="Matching run dir(s) whose chosen signatures the trim "
                        "must not drop (repeatable). Stage 7 REFUSES to run "
                        "with neither this nor --positive_set")
    p.add_argument("--positive_set", type=Path,
                   help="landmark_positive_set.py output the trim must retain. "
                        "Stage 7 REFUSES to run with neither this nor "
                        "--matched_from")

    # Mechanical knobs and guards.
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--convert_workers", type=int, default=8)
    p.add_argument("--timelapse_fps", type=int, default=15)
    p.add_argument("--timelapse_max_frames", type=int, default=1500,
                   help="Subsample the timelapse above this many frames (0 = all)")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--prune_raw", action="store_true",
                   help="Delete <raw_dir>/<name> after a successful conversion. "
                        "Recommended when running many trajectories: the "
                        "staged originals are ~15x larger than the stored "
                        "result")
    p.add_argument("--force", action="store_true",
                   help="Redo stage 1 even if a manifest already exists")
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
    args = p.parse_args(argv)

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
    rejected = [n for n in unknown if n in REJECTED_TRAJECTORIES]
    if rejected:
        for n in rejected:
            print(f"ERROR: {n} was screened and rejected: "
                  f"{REJECTED_TRAJECTORIES[n]['reason']}")
        return 1
    if unknown:
        print(f"ERROR: unknown trajectory name(s): {unknown}")
        print(f"Known: {', '.join(TRAJECTORIES)}")
        return 1

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"ERROR: unknown stage(s) {bad}; valid are {sorted(STAGES)}")
        return 1

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
