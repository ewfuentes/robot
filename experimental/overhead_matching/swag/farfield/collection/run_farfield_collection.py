#!/usr/bin/env python3
"""Collect the far-field Mapillary trajectories end to end.

Stages, per trajectory:
  1 RESOLVE   seed pKey -> stitched manifest             (seed_to_trajectory, in-process)
  2 DOWNLOAD  manifest -> ordered jpg+json staging       (extract_stitch, in-process)
  3 CONVERT   staging -> dataset dir                     (mapillary_to_vigor, in-process)
  4 TIMELAPSE trajectory.png + gps_timelapse.mp4         (dataset_tools.make_dataset_timelapse)
  5 OSM       publish full typed catalog (+ENC where covered)
  6 PINHOLE   4 yaw faces, equirectangular only          (panorama_to_pinhole via bazel)
  7 TRIM      full catalog -> matchable typed catalog     (trim_catalog)
  8 PLOT      full-catalog coverage diagnostic artifact  (plot_landmarks)
  9 AUDIT     dataset contract audit                     (farfield.audit_dataset, in-process)

Collection stages run in-process when they expose a Python entry point. The
landmark extractors run as subprocesses under a cgroup memory cap, and
panorama_to_pinhole is invoked through its Bazel target. Those subprocesses
resolve the repository root from $BUILD_WORKSPACE_DIRECTORY and fail when it is
absent. Stage 4 imports its dataset tool directly.
Stage 5 publishes the full catalog in the typed artifact lane, stage 7
publishes its trimmed descendant before matching, and stage 8 evaluates the
full source catalog independently of that semantic trim. Matching-derived
evidence may guard a later revision, but is not required to construct the
first trimmed catalog or its coverage diagnostic.

TIMELAPSE sits directly after CONVERT because it is the input to the human
triage pass. Rendering it before the landmark stages catches capture defects
such as moving or panning cameras before catalog extraction and pinhole render.

Stage 6 is deliberately skipped for perspective captures: a limited-FOV frame
is already a single view. The converter records its optical-axis convention in
intrinsics.csv, but the current downstream extraction pipeline supports only
equirectangular inputs; perspective collection is not an end-to-end far-field
localization path yet.

Layout (lifecycle lanes; defaults derive from farfield.paths.default_root()):
    <output_base>/<name>/       dataset (panorama/, frames_gps.csv, ...)
    <manifest_dir>/<name>.json  stage-1 stitched manifest
    <raw_dir>/<name>/           staged originals, prunable after stage 3
    <catalog_sources_base>/<name>/<version>/  loose OSM/ENC build material
    <catalog_base>/<name>/<version>/          typed full/trimmed catalogs
    <catalog_coverage_base>/<name>/<version>/ typed coverage diagnostics
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    audit_dataset,
    paths as paths_lib,
    publication,
    provenance,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    schema as catalog_schema,
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
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    download_enc_cells,
    make_dataset_timelapse,
    plot_landmarks,
    trim_catalog,
)

SCRIPTS = "//experimental/overhead_matching/swag/scripts"
DATASET_TOOLS = "//experimental/overhead_matching/swag/farfield/dataset_tools"

# `bazel run` must be invoked from the source workspace, not the runfiles
# tree. Only the stages that genuinely need a subprocess consult this.
WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

# The OSM writer reads the caller-selected full PBF directly with the common
# extractor's complete geometry index. Keep a cgroup ceiling as a final
# host-safety guard, independent of source PBF size.
# This ceiling is now the ONLY guard on extractor memory; the bounded node
# index and the source-size refusal that used to sit in front of it are
# gone. What they were protecting against, and what to restore if a
# country-sized PBF is needed again: docs/farfield/decisions.md, 2026-08
# 'OSM extraction memory'.
EXTRACT_MEM_CAP_GB = 24

PINHOLE_FACES = paths_lib.PINHOLE_FACES


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


def stage_resolve(name, cfg, args):
    manifest = args.manifest_dir / f"{name}.json"
    expected_config = {
        "name": name,
        "window_hours": args.window_hours,
        "stitch_time_s": args.stitch_time,
        "stitch_dist_m": args.stitch_dist,
    }
    if manifest.exists() or manifest.is_symlink():
        try:
            document = seed_to_trajectory.validate_sequence_manifest(
                manifest,
                expected_sequence_id=name,
                expected_seed_pkey=cfg["seed_pkey"],
            )
            if document["provenance"]["config"] != expected_config:
                raise ValueError(
                    "recorded stitch recipe does not match the requested recipe")
        except (OSError, ValueError) as exc:
            print(f"  ERROR: invalid completed stage-1 manifest {manifest}: {exc}")
            return False
        print(f"\n=== [1 RESOLVE] {name}: validated completed manifest")
        return True
    incomplete = manifest.with_name(
        manifest.name + seed_to_trajectory.MANIFEST_INCOMPLETE_SUFFIX)
    if incomplete.exists() or incomplete.is_symlink():
        print(f"  ERROR: incomplete stage-1 manifest exists: {incomplete}")
        return False
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
    return run_module(
        make_dataset_timelapse.main,
        ["--dataset_path", args.output_base / name,
         "--fps", args.timelapse_fps,
         "--max_frames", args.timelapse_max_frames],
        f"[4 TIMELAPSE] {name}", args)


def _catalog_artifact_dir(args, name: str, version: str) -> Path:
    return args.catalog_base / name / version


def _publish_full_catalog(name: str, source: Path, args, config: dict) -> bool:
    """Publish stage 5's selected compact Feather as the full typed catalog."""
    output_dir = _catalog_artifact_dir(args, name, args.catalog_version)
    try:
        frame = catalog_schema.read_frame(source)
        source_digest = artifact.sha256_file(source)
        generator = ("//experimental/overhead_matching/swag/farfield/"
                     "collection:run_farfield_collection (stage 5)")
        manifest_config = {
            **config,
            "schema": catalog_schema.FULL_ARTIFACT_SCHEMA,
            "selected_source_feather": str(source.resolve()),
            "selected_source_sha256": source_digest,
            "rows": int(len(frame)),
        }
        if output_dir.exists() or output_dir.is_symlink():
            artifact.open_artifact(
                output_dir,
                expected_kind=paths_lib.CATALOGS,
                expected_dataset=name,
                expected_version=args.catalog_version)
            manifest = artifact.load_manifest(output_dir)
            payload = output_dir / "catalog.feather"
            if (manifest.generator != generator
                    or manifest.git_commit != provenance.git_commit()
                    or manifest.upstreams
                    or dict(manifest.config) != manifest_config
                    or manifest.declared_outputs != ("catalog.feather",)
                    or artifact.sha256_file(payload) != source_digest):
                raise artifact.ArtifactValidationError(
                    "completed full catalog has a different source, recipe, "
                    "or payload")
            catalog_schema.read_frame(payload)
            print(f"  reusing exact full catalog: {output_dir}")
            return True
        with publication.published_artifact(
                output_dir,
                kind=paths_lib.CATALOGS,
                dataset=name,
                version=args.catalog_version,
                generator=generator,
                git_commit=provenance.git_commit(),
                config=manifest_config,
                declared_outputs=("catalog.feather",)) as builder:
            shutil.copyfile(source, builder.output_path("catalog.feather"))
    except (artifact.ArtifactError, artifact.ArtifactExistsError,
            catalog_schema.CatalogSchemaError,
            publication.PublicationValidationError, OSError) as exc:
        print(f"  ERROR: failed to publish full catalog {output_dir}: {exc}")
        return False
    print(f"  published full catalog: {output_dir}")
    return True


def _finish_landmark_stage(name: str, selected: Path, pbf_paths, bbox,
                           osm_specs, enc_state, merged: bool, args,
                           source_coverage: dict, cells=None,
                           enc_selection: Path | None = None) -> bool:
    source_dir = args.catalog_sources_base / name / args.catalog_version
    _write_provenance(source_dir, name, pbf_paths, bbox,
                      osm_specs=osm_specs, enc_state=enc_state,
                      merged=merged, cells=cells, args=args,
                      enc_selection=enc_selection)
    return _publish_full_catalog(
        name,
        selected,
        args,
        config={
            "bbox_wsen": list(bbox),
            "osm_specs": list(osm_specs),
            "enc_state": enc_state,
            "enc_cells": list(cells or []),
            "enc_available": merged,
            "enc_selection": (
                {
                    "path": str(enc_selection.resolve()),
                    "sha256": artifact.sha256_file(enc_selection),
                }
                if enc_selection is not None else None
            ),
            "dedupe_tolerance_m": args.dedupe_tolerance_m,
            "osm_geometry_index_mode": "full_pbf_complete_geometry_index",
            "source_coverage": source_coverage,
        },
    )


def stage_landmarks(name, cfg, args):
    """OSM landmarks, plus ENC and a merge where NOAA has coverage."""
    dataset_dir = args.output_base / name
    if args.dry_run:
        osm_specs = (cfg["osm"] if isinstance(cfg["osm"], (list, tuple))
                     else [cfg["osm"]])
        print(f"  [DRY RUN] would validate and extract {len(osm_specs)} "
              f"OSM source(s) for {name}; no cache, network, or dataset I/O")
        return True
    if not (dataset_dir / "pano_id_mapping.csv").exists():
        print(f"  ERROR: {dataset_dir}/pano_id_mapping.csv missing — run stage 3 first")
        return False

    buffer_km = cfg.get("landmark_buffer_km", args.landmark_buffer_km)
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
            print(f"  ERROR: no OSM extract for {spec} in {args.osm_cache_dir} "
                  f"(looked for {wanted.replace('-latest', '-*')})")
            return False
        pbf = matches[-1]
        pbf_paths.append(pbf)

    # Verify the requested bbox against the Geofabrik clip boundaries before
    # extraction; the extractor itself cannot detect a partial source set.
    ok, msg, cov_details = check_coverage(
        osm_specs, (west, south, east, north),
        cache_dir=args.osm_cache_dir / "poly", pbf_paths=pbf_paths)
    print(f"  coverage: {'OK' if ok else 'FAIL'} — {msg}")
    for d in cov_details:
        if "spec" in d:
            print(f"    {d['spec'].rsplit('/', 1)[-1]}: "
                  f"{100*d['covers_frac_of_request']:.1f}% of the request")
    if not ok:
        print(f"  ERROR: refusing to build a partial landmark catalog for {name}. "
              f"Fix the registry's osm list.")
        return False
    source_coverage = {
        "schema": "farfield_catalog_source_coverage/v2",
        "status": "passed",
        "message": msg,
        "details": cov_details,
    }

    sources = args.catalog_sources_base / name / args.catalog_version
    sources.mkdir(parents=True, exist_ok=True)

    osm_feathers = []
    for pbf_path in pbf_paths:
        # Preserve the full region name so distinct hyphenated regions remain
        # distinct source identities.
        region = re.sub(r"-\d{6}$", "", pbf_path.name.replace(".osm.pbf", ""))
        out = sources / (f"osm_{name}_{region}_v1" if len(pbf_paths) > 1
                         else f"osm_{name}_v1")
        # Both extractors require --bbox in WEST SOUTH EAST NORTH order.
        size_mb = pbf_path.stat().st_size / 1e6 if pbf_path.exists() else 0
        ok = run_bazel(
            f"{DATASET_TOOLS}:extract_landmarks_from_osm",
            ["--pbf_file", pbf_path,
             "--bbox", west, south, east, north,
             "--output_path", out],
            f"[5 OSM] {name} ({pbf_path.name}, {size_mb:.0f} MB, cap "
            f"{args.extract_mem_cap_gb} GB)", args,
            mem_cap_gb=args.extract_mem_cap_gb)
        if not ok:
            return False
        osm_feathers.append(out.with_suffix(".feather"))

    if len(osm_feathers) > 1:
        combined = sources / f"osm_{name}.feather"
        ok = run_bazel(
            f"{DATASET_TOOLS}:merge_landmark_feathers",
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
              f"Fixed navaids will be sparser here; recorded in source provenance")
        if args.dry_run:
            return True
        return _finish_landmark_stage(
            name, osm_out.with_suffix(".feather"), pbf_paths,
            (west, south, east, north), osm_specs, None, False, args,
            source_coverage)

    enc_out = sources / f"enc_{name}_v1"
    selection_output = sources / f"enc_{name}_selection.json"
    ok = run_bazel(
        f"{DATASET_TOOLS}:download_enc_cells",
        ["--catalog_state", cfg["enc_state"],
         "--bbox", west, south, east, north,
         "--band", 5,
         "--output_dir", args.enc_root,
         "--selection_output", selection_output],
        f"[5 ENC dl] {name} ({cfg['enc_state']})", args)
    if not ok:
        return False
    if args.dry_run:
        return True
    try:
        selection = download_enc_cells.validate_selection(
            selection_output, args.enc_root)
    except (OSError, ValueError) as error:
        print(f"  ERROR: invalid ENC selection record: {error}")
        return False
    expected_selection = {
        "catalog_state": cfg["enc_state"],
        "bbox": [west, south, east, north],
        "band": 5,
        "explicit_cells": False,
    }
    disagreements = {
        key: (selection.get(key), expected)
        for key, expected in expected_selection.items()
        if selection.get(key) != expected
    }
    if disagreements:
        print(f"  ERROR: ENC selection disagrees with this invocation: "
              f"{disagreements}")
        return False
    cells = selection["cells"]
    if not cells:
        print(f"  ERROR: ENC selection returned no band-5 cells for {name}")
        return False
    print(f"  ENC cells: {' '.join(cells)}")

    ok = run_bazel(
        f"{DATASET_TOOLS}:extract_landmarks_from_enc",
        ["--enc_root", args.enc_root, "--selection", selection_output,
         "--bbox", west, south, east, north,
         "--dedupe_tolerance_m", args.dedupe_tolerance_m,
         "--output_path", enc_out],
        f"[5 ENC extract] {name}", args)
    if not ok:
        return False

    merged = sources / f"{name}_osm_enc.feather"
    ok = run_bazel(
        f"{DATASET_TOOLS}:merge_landmark_feathers",
        ["--inputs", osm_out.with_suffix(".feather"), enc_out.with_suffix(".feather"),
         "--output", merged,
         "--dedupe_tolerance_m", args.dedupe_tolerance_m],
        f"[5 MERGE] {name}", args)
    if not ok:
        return False
    if args.dry_run:
        return True
    return _finish_landmark_stage(
        name, merged, pbf_paths, (west, south, east, north), osm_specs,
        cfg["enc_state"], True, args, source_coverage, cells=cells,
        enc_selection=selection_output)
def _write_provenance(source_dir: Path, name: str, pbf_paths, bbox,
                      enc_state, merged: bool, args, cells=None, osm_specs=None,
                      enc_selection: Path | None = None):
    """Record how the loose raw-material catalog sources were produced."""
    provenance.write(
        source_dir,
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
            "enc_selection": (
                {
                    "path": str(enc_selection.resolve()),
                    "sha256": artifact.sha256_file(enc_selection),
                }
                if enc_selection is not None else None
            ),
            "dedupe_tolerance_m": args.dedupe_tolerance_m,
            "osm_geometry_index_mode": "full_pbf_complete_geometry_index",
        },
        notes=("OSM+ENC merged catalog" if merged else
               "OSM-only catalog: NOAA ENC covers US waters only, so fixed "
               "navaids (beacons, buoys, lights) are substantially less "
               "complete here than in the US datasets"),
    )
    print(f"  wrote {source_dir / provenance.MANIFEST_NAME}")


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
              f"limited-FOV frame is already a single view; intrinsics.csv "
              f"records its optical-axis convention, but downstream far-field "
              f"extraction does not support perspective inputs yet)")
        return True
    # Pinhole faces are a versioned artifact (artifacts/pinhole_images/<name>/
    # <version>/ + manifest.json), not part of the frozen dataset contract.
    out_dir = args.pinhole_base / name / args.pinhole_version
    staging_dir = out_dir.with_name(out_dir.name + artifact.INCOMPLETE_SUFFIX)
    if args.dry_run:
        return run_bazel(
            f"{SCRIPTS}:panorama_to_pinhole",
            [dataset_dir / "panorama", staging_dir,
             "--num_workers", args.convert_workers,
             "--res_x", args.pinhole_res],
            f"[6 PINHOLE] {name}", args)

    try:
        source_digests = paths_lib.dataset_source_digests(dataset_dir)
        panorama_files = sorted((dataset_dir / "panorama").glob("*.jpg"))
        if not panorama_files:
            raise ValueError(
                f"{dataset_dir / 'panorama'} contains no panorama JPEGs")
        panorama_keys = [path.stem for path in panorama_files]
        declared_outputs = paths_lib.pinhole_declared_outputs(panorama_keys)
        config = paths_lib.pinhole_manifest_config(
            source_digests, resolution=args.pinhole_res,
            panorama_keys=panorama_keys)
        with publication.published_artifact(
                out_dir,
                kind=paths_lib.PINHOLE_IMAGES,
                dataset=name,
                version=args.pinhole_version,
                generator=f"{SCRIPTS}:panorama_to_pinhole "
                          "(collection stage 6)",
                git_commit=provenance.git_commit(),
                config=config,
                declared_outputs=declared_outputs) as builder:
            if not run_bazel(
                    f"{SCRIPTS}:panorama_to_pinhole",
                    [dataset_dir / "panorama", builder.path,
                     "--num_workers", args.convert_workers,
                     "--res_x", args.pinhole_res],
                    f"[6 PINHOLE] {name}", args):
                raise RuntimeError("panorama-to-pinhole subprocess failed")
            for relative in declared_outputs:
                output = builder.path / relative
                if output.is_symlink() or not output.is_file():
                    raise ValueError(
                        f"pinhole renderer omitted regular output {relative}")
                try:
                    with Image.open(output) as image:
                        image.load()
                        if image.format != "JPEG":
                            raise ValueError(
                                f"pinhole output is not JPEG: {relative}")
                        if image.size != (args.pinhole_res,
                                          args.pinhole_res):
                            raise ValueError(
                                f"pinhole output {relative} has size "
                                f"{image.size}, expected "
                                f"{(args.pinhole_res, args.pinhole_res)}")
                except OSError as exc:
                    raise ValueError(
                        f"pinhole output is not decodable: {relative}") from exc
            if paths_lib.dataset_source_digests(dataset_dir) != source_digests:
                raise ValueError(
                    "dataset source bytes changed during pinhole rendering")
    except (artifact.ArtifactError, OSError, RuntimeError, ValueError,
            paths_lib.MissingInput,
            publication.PublicationValidationError) as exc:
        print(f"  ERROR: failed to publish pinhole artifact {out_dir}: {exc}")
        return False
    print(f"  published pinhole artifact: {out_dir}")
    return True


def stage_trim(name, cfg, args):
    """Publish the matchable catalog from stage 5's full typed catalog."""
    del cfg
    input_dir = _catalog_artifact_dir(args, name, args.catalog_version)
    output_dir = _catalog_artifact_dir(
        args, name, args.trimmed_catalog_version)
    if not input_dir.exists() and not args.dry_run:
        print(f"  ERROR: {input_dir} missing — run stage 5 (OSM) first")
        return False
    argv = [
        "--input_catalog_dir", input_dir,
        "--output_dir", output_dir,
        "--min_building_area_m2", args.trim_min_building_area_m2,
        "--min_building_levels", args.trim_min_building_levels,
    ]
    for matched in args.matched_from or []:
        argv += ["--matched_from", matched]
    if args.matched_from:
        argv += ["--confidence_floor", args.trim_confidence_floor]
    if args.positive_set:
        argv += ["--positive_set", args.positive_set]
    return run_module(trim_catalog.cli, argv, f"[7 TRIM] {name}", args)


def stage_plot(name, cfg, args):
    """Publish a review artifact for the full pre-trim catalog."""
    del cfg
    catalog_dir = _catalog_artifact_dir(args, name, args.catalog_version)
    if not catalog_dir.exists() and not args.dry_run:
        print(f"  ERROR: {catalog_dir} missing — run stage 5 (OSM) first")
        return False
    output_dir = (
        args.catalog_coverage_base / name / args.catalog_coverage_version)
    argv = [
        "--dataset", name,
        "--dataset_dir", args.output_base / name,
        "--catalog_dir", catalog_dir,
        "--poly_cache_dir", args.osm_cache_dir / "poly",
        "--output_dir", output_dir,
        "--grid_cells", args.coverage_grid_cells,
        "--max_empty_run", args.coverage_max_empty_run,
        "--empty_fraction_warning", args.coverage_empty_fraction_warning,
        "--far_range_km", args.coverage_far_range_km,
        "--min_far_fraction", args.coverage_min_far_fraction,
        "--max_track_samples", args.coverage_max_track_samples,
    ]
    return run_module(
        plot_landmarks.cli, argv, f"[8 PLOT] {name}", args)


def stage_audit(name, cfg, args):
    """Contract audit (farfield.audit_dataset, in-process); fails the stage on
    any violation."""
    return run_module(audit_dataset.main, [args.output_base / name],
                      f"[9 AUDIT] {name}", args)


STAGES = {1: stage_resolve, 2: stage_download, 3: stage_convert,
          4: stage_timelapse, 5: stage_landmarks, 6: stage_pinhole,
          7: stage_trim, 8: stage_plot, 9: stage_audit}
DEFAULT_STAGES = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def main(argv=None) -> int:
    root = paths_lib.default_root()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectories", required=True,
                   help="'all', 'pilot', 'pano', 'perspective', or "
                        "comma-separated registry names")
    p.add_argument("--stages", default=",".join(map(str, DEFAULT_STAGES)),
                   help="Comma-separated stage numbers (default: all stages)")

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
    p.add_argument("--catalog_sources_base", type=Path,
                   default=root / "raw_material" / "catalog_sources",
                   help="Loose OSM/ENC source-build lane (default: "
                        "<farfield_root>/raw_material/catalog_sources)")
    p.add_argument("--catalog_base", type=Path,
                   default=root / "artifacts" / paths_lib.CATALOGS,
                   help="Typed catalog artifact lane (default: "
                        "<farfield_root>/artifacts/catalogs)")
    p.add_argument("--catalog_coverage_base", type=Path,
                   default=root / "artifacts" / paths_lib.CATALOG_COVERAGE,
                   help="Typed catalog-coverage diagnostic lane (default: "
                        "<farfield_root>/artifacts/catalog_coverage)")
    p.add_argument("--pinhole_base", type=Path,
                   default=root / "artifacts" / paths_lib.PINHOLE_IMAGES,
                   help="Pinhole artifact lane (default: "
                        "<farfield_root>/artifacts/pinhole_images)")
    p.add_argument("--osm_cache_dir", type=Path, required=True,
                   help="Directory of downloaded Geofabrik .osm.pbf extracts; "
                        ".poly boundaries cache under <osm_cache_dir>/poly")
    p.add_argument("--enc_root", type=Path, required=True,
                   help="NOAA ENC cell root used by download_enc_cells / "
                        "extract_landmarks_from_enc")

    # Stage 1-3 result-shaping parameters are explicit and required.
    p.add_argument("--stitch_time", type=float, required=True,
                   help="Maximum seam time gap in seconds")
    p.add_argument("--stitch_dist", type=float, required=True,
                   help="Maximum seam spatial-gap floor in metres")
    p.add_argument("--window_hours", type=float, required=True,
                   help="Sibling-sequence capture-time window in hours")
    p.add_argument("--max_width", type=int, required=True,
                   help="Cap on stored image width, also the convert resize; "
                        "0 disables")
    p.add_argument("--min_spacing_m", type=float, required=True,
                   help="Drop frames closer than this along the track at "
                        "download time. These are mostly video extractions at "
                        "1-30 fps, so consecutive frames are metres apart and "
                        "often share a GPS fix outright")
    p.add_argument("--jpeg_quality", type=int, required=True,
                   help="JPEG quality for stored frames")
    p.add_argument("--heading_source", choices=("auto", "computed", "compass"),
                   required=True,
                   help="Heading source for the converter")
    p.add_argument("--max_heading_error_deg", type=float, required=True,
                   help="Converter heading-reliability gate in degrees")
    p.add_argument("--max_perspective_offset_std_deg", type=float, required=True,
                   help="Converter hand-held-vs-fixed-mount report threshold")
    p.add_argument("--max_heading_source_disagreement_deg", type=float,
                   required=True,
                   help="Converter SfM-vs-magnetometer warning threshold")

    # Stage 5-8 parameters.
    p.add_argument("--pinhole_res", type=int, required=True,
                   help="Pinhole face resolution")
    p.add_argument("--pinhole_version", required=True,
                   help="Version directory under the pinhole artifact lane")
    p.add_argument("--landmark_buffer_km", type=float, required=True,
                   help="Buffer around the trajectory for the landmark "
                        "catalog; per-trajectory registry overrides win. Large "
                        "by design because elevated far-field landmarks can be "
                        "visible well beyond the track")
    p.add_argument("--catalog_version", required=True,
                   help="Full typed CATALOGS version published by stage 5")
    p.add_argument("--trimmed_catalog_version", required=True,
                   help="Trimmed typed CATALOGS version published by stage 7")
    p.add_argument("--catalog_coverage_version", required=True,
                   help="Typed catalog_coverage version published by stage 8")
    p.add_argument("--dedupe_tolerance_m", type=float, required=True,
                   help="Merge identical-tag features whose geometries touch "
                        "within this distance")
    p.add_argument("--trim_min_building_area_m2", type=float, required=True,
                   help="Stage-7 trim: drop untagged buildings smaller than "
                        "this footprint")
    p.add_argument("--trim_min_building_levels", type=float, required=True,
                   help="Stage-7 trim: keep small buildings only at/above "
                        "this many levels")
    p.add_argument("--trim_confidence_floor", type=float, default=0.5,
                   help="Stage-7 optional matched-result guard: ignore matches "
                        "below this confidence (default 0.5)")
    p.add_argument("--matched_from", type=Path, action="append",
                   help="Complete typed LANDMARK_MATCHES artifact(s) whose "
                        "chosen signatures trim_catalog must not drop "
                        "(optional, repeatable)")
    p.add_argument("--positive_set", type=Path,
                   help="Schema-v2 landmark_positive_set.py output that "
                        "trim_catalog must retain (optional)")
    p.add_argument("--coverage_grid_cells", type=int, required=True,
                   help="Stage-8 square density-grid resolution")
    p.add_argument("--coverage_max_empty_run", type=int, required=True,
                   help="Stage-8 failing interior empty-band run length")
    p.add_argument("--coverage_empty_fraction_warning", type=float,
                   required=True,
                   help="Stage-8 warning threshold for the empty grid "
                        "fraction")
    p.add_argument("--coverage_far_range_km", type=float, required=True,
                   help="Stage-8 distance defining the far-field tail")
    p.add_argument("--coverage_min_far_fraction", type=float, required=True,
                   help="Stage-8 warning threshold for the fraction beyond "
                        "the far-range distance")
    p.add_argument("--coverage_max_track_samples", type=int, required=True,
                   help="Stage-8 deterministic trajectory sample cap used by "
                        "distance calculations")

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
    p.add_argument("--extract_mem_cap_gb", type=int, default=EXTRACT_MEM_CAP_GB,
                   help=f"Hard cgroup memory ceiling for landmark extraction; 0 "
                        f"disables (default: {EXTRACT_MEM_CAP_GB})")
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
            try:
                completed = STAGES[s](name, cfg, args)
            except Exception as exc:
                print(f"  FAILED ({type(exc).__name__}: {exc}): "
                      f"stage {s} for {name}")
                completed = False
            if not completed:
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
