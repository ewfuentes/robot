#!/usr/bin/env python3
"""Plan or materialize strict full catalogs for the eight active datasets.

The default mode is report-only: it validates the exact frozen trajectory
tables, caller-pinned full PBF files, and Geofabrik coverage, then prints a
content-addressed plan. Coverage validation may fetch missing Geofabrik
``.poly``/index metadata into ``--poly_cache_dir``; it never downloads a PBF,
an ENC catalog/cell, or builds an artifact. Materialization requires both
``--materialize`` and the reported digest, and delegates extraction, ENC
handling, and source merges to their existing dataset-tools owners.

ENC catalog and cell identities cannot be in the reviewed plan because the
report phase deliberately does not download them. The plan binds the exact ENC
selection policy. Materialization then validates the immutable selection
record and publishes its SHA-256; that record contains the selected catalog
identity and every cell's manifest/content SHA-256, so the artifact binds all
ENC bytes transitively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    dataset as dataset_lib,
    geometry,
    paths as paths_lib,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    lineage as catalog_lineage,
    schema as catalog_schema,
)
from experimental.overhead_matching.swag.farfield.collection import (
    geometry_helpers,
    pbf_coverage,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    download_enc_cells,
    extract_landmarks_from_enc,
    extract_landmarks_from_osm,
    merge_landmark_feathers,
)


PLAN_SCHEMA = "farfield.active_catalog_plan/v1"
GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "collection:active_catalogs (stage 5)")
OSM_GEOMETRY_INDEX_MODE = "full_pbf_complete_geometry_index"
BBOX_BUFFER_KM = 25.0
ENC_BAND = 5
COLLISION_RADIUS_M = 150.0
MAPPING_COLUMNS = ("pano_id", "lat", "lon", "filename")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ActiveCatalogError(ValueError):
    """An active catalog plan or materialization is unsafe or inconsistent."""


@dataclass(frozen=True)
class ActiveCatalogScope:
    name: str
    output_datasets: tuple[str, ...]
    bbox_datasets: tuple[str, ...]
    osm_specs: tuple[str, ...]
    enc_state: str | None


ACTIVE_SCOPES = (
    ActiveCatalogScope(
        name="boston_harbor_20260712",
        output_datasets=(
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ),
        bbox_datasets=(
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ),
        osm_specs=(
            "north-america/us/massachusetts-latest.osm.pbf",
        ),
        enc_state="MA",
    ),
    ActiveCatalogScope(
        name="charles_river_20260727",
        output_datasets=("charles_river_20260727",),
        bbox_datasets=("charles_river_20260727",),
        osm_specs=(
            "north-america/us/massachusetts-latest.osm.pbf",
        ),
        enc_state="MA",
    ),
    ActiveCatalogScope(
        name="mount_washington_20260815",
        output_datasets=(
            "mount_washington_20260815_leg1",
            "mount_washington_20260815_leg2",
            "mount_washington_20260815_leg3",
        ),
        bbox_datasets=(
            "mount_washington_20260815_leg1",
            "mount_washington_20260815_leg2",
            "mount_washington_20260815_leg3",
        ),
        osm_specs=(
            "north-america/us/new-hampshire-latest.osm.pbf",
            "north-america/us/maine-latest.osm.pbf",
            "north-america/us/vermont-latest.osm.pbf",
        ),
        enc_state=None,
    ),
    ActiveCatalogScope(
        name="pohang_canal_04",
        output_datasets=("pohang_canal_04",),
        bbox_datasets=("pohang_canal_04",),
        osm_specs=("asia/south-korea-latest.osm.pbf",),
        enc_state=None,
    ),
)
SCOPE_BY_NAME = {scope.name: scope for scope in ACTIVE_SCOPES}


def select_scopes(names: list[str] | tuple[str, ...]) \
        -> tuple[ActiveCatalogScope, ...]:
    if not names:
        raise ActiveCatalogError("at least one --scope is required")
    if "all" in names:
        if len(names) != 1:
            raise ActiveCatalogError("--scope all cannot be combined with names")
        return ACTIVE_SCOPES
    if len(names) != len(set(names)):
        raise ActiveCatalogError("scope names must be unique")
    unknown = sorted(set(names) - set(SCOPE_BY_NAME))
    if unknown:
        raise ActiveCatalogError(f"unknown active scopes: {unknown}")
    requested = set(names)
    return tuple(scope for scope in ACTIVE_SCOPES if scope.name in requested)


def parse_pbf_map(values: list[str] | tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ActiveCatalogError(
                "each --pbf must be GEOFABRIK_SPEC=/absolute/file.osm.pbf")
        spec, raw_path = value.split("=", 1)
        if not spec or not raw_path:
            raise ActiveCatalogError(
                "each --pbf must have a non-empty spec and path")
        if not Path(raw_path).is_absolute():
            raise ActiveCatalogError("each --pbf path must be absolute")
        if spec in result:
            raise ActiveCatalogError(f"duplicate --pbf spec: {spec}")
        result[spec] = Path(raw_path)
    return result


def _stat_tuple(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _file_identity(path: Path, *, what: str) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ActiveCatalogError(f"{what} is not a regular non-symlink file: {path}")
    before = _stat_tuple(path)
    digest = artifact.sha256_file(path)
    after = _stat_tuple(path)
    if before != after:
        raise ActiveCatalogError(f"{what} changed while it was hashed: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": before[2],
        "sha256": digest,
    }


def _read_dataset_tables(
        dataset: str, root: Path) -> tuple[dict, list[float], list[float]]:
    """Bind and cross-check bbox mapping against canonical GPS/panoramas."""
    dataset_dir = Path(root) / "datasets" / dataset
    mapping_path = dataset_dir / "pano_id_mapping.csv"
    gps_path = dataset_dir / "frames_gps.csv"
    mapping_identity = _file_identity(
        mapping_path, what=f"{dataset} mapping table")
    gps_identity = _file_identity(gps_path, what=f"{dataset} GPS table")

    mapping_rows = []
    pano_ids: set[str] = set()
    try:
        with mapping_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MAPPING_COLUMNS:
                raise ActiveCatalogError(
                    f"{dataset} mapping header must be exactly "
                    f"{list(MAPPING_COLUMNS)}, got {reader.fieldnames}")
            for line_number, row in enumerate(reader, start=2):
                pano_id = row["pano_id"]
                if not pano_id or pano_id in pano_ids:
                    raise ActiveCatalogError(
                        f"{dataset} mapping has empty or duplicate pano_id "
                        f"at line {line_number}")
                if not row["filename"]:
                    raise ActiveCatalogError(
                        f"{dataset} mapping has an empty filename at line "
                        f"{line_number}")
                filename = row["filename"]
                if (Path(filename).name != filename
                        or not filename.lower().endswith(".jpg")):
                    raise ActiveCatalogError(
                        f"{dataset} mapping filename is not a JPEG basename at "
                        f"line {line_number}")
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                except ValueError as error:
                    raise ActiveCatalogError(
                        f"{dataset} mapping has non-numeric coordinates at "
                        f"line {line_number}") from error
                if (not math.isfinite(lat) or not math.isfinite(lon)
                        or not -90.0 <= lat <= 90.0
                        or not -180.0 <= lon <= 180.0):
                    raise ActiveCatalogError(
                        f"{dataset} mapping has invalid WGS84 coordinates at "
                        f"line {line_number}")
                pano_ids.add(pano_id)
                mapping_rows.append((pano_id, filename, lat, lon))
    except OSError as error:
        raise ActiveCatalogError(
            f"cannot read {mapping_path}: {error}") from error
    if not mapping_rows:
        raise ActiveCatalogError(
            f"{dataset} mapping table is empty: {mapping_path}")
    if (_file_identity(mapping_path, what=f"{dataset} mapping table")
            != mapping_identity):
        raise ActiveCatalogError(f"{dataset} mapping changed while it was read")

    panorama_dir = dataset_dir / "panorama"
    if not panorama_dir.is_symlink():
        raise ActiveCatalogError(
            f"{dataset} panorama must be the symlink 'panorama -> frames'")
    if panorama_dir.readlink() != Path("frames"):
        raise ActiveCatalogError(
            f"{dataset} panorama symlink text must be exactly 'frames'")
    frames_dir = dataset_dir / "frames"
    try:
        panorama_resolved = panorama_dir.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ActiveCatalogError(
            f"{dataset} panorama -> frames symlink is dangling or cyclic") from error
    if frames_dir.is_symlink() or not frames_dir.is_dir():
        raise ActiveCatalogError(
            f"{dataset} frames must be a real non-symlink directory")
    try:
        frames_resolved = frames_dir.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ActiveCatalogError(
            f"{dataset} frames directory cannot be resolved") from error
    if panorama_resolved != frames_resolved:
        raise ActiveCatalogError(
            f"{dataset} panorama must resolve to its real frames directory")
    panorama_entries = sorted(
        panorama_dir.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file()
           for path in panorama_entries):
        raise ActiveCatalogError(
            f"{dataset} panorama set contains a per-JPEG symlink or "
            "non-regular file")
    try:
        frames = dataset_lib.load_frames(dataset_dir)
    except dataset_lib.ContractViolation as error:
        raise ActiveCatalogError(
            f"{dataset} violates the canonical frame contract: {error}") from error
    if not frames:
        raise ActiveCatalogError(f"{dataset} canonical frame table is empty")
    if (_file_identity(gps_path, what=f"{dataset} GPS table")
            != gps_identity):
        raise ActiveCatalogError(
            f"{dataset} GPS table changed while canonical frames were loaded")

    panorama_names = [path.name for path in panorama_entries]
    frame_names = [frame.pano_stem + ".jpg" for frame in frames]
    if (set(panorama_names) != set(frame_names)
            or len(panorama_names) != len(frame_names)):
        raise ActiveCatalogError(
            f"{dataset} panorama JPEG set does not exactly match canonical "
            "frames")
    if len(mapping_rows) != len(frames):
        raise ActiveCatalogError(
            f"{dataset} mapping/canonical frame counts disagree: "
            f"{len(mapping_rows)} != {len(frames)}")
    for position, (mapping_row, frame) in enumerate(
            zip(mapping_rows, frames, strict=True)):
        pano_id, mapping_name, mapping_lat, mapping_lon = mapping_row
        if (pano_id != frame.pano_id
                or mapping_name != frame.pano_stem + ".jpg"):
            raise ActiveCatalogError(
                f"{dataset} stale mapping identity disagrees with canonical "
                f"frame at row {position}")
        if geometry.haversine_m(
                mapping_lat, mapping_lon, frame.lat, frame.lon) > 1.0:
            raise ActiveCatalogError(
                f"{dataset} stale mapping coordinates disagree with canonical "
                f"frame at row {position}")

    lats = [frame.lat for frame in frames]
    lons = [frame.lon for frame in frames]
    return ({
        "dataset": dataset,
        "pano_id_mapping": mapping_identity,
        "frames_gps": gps_identity,
        "panorama": {
            "path": str(panorama_dir.resolve()),
            "relative_link": "frames",
            "jpeg_count": len(panorama_names),
            "filenames_sha256": artifact.sha256_json(panorama_names),
        },
        "rows": len(lats),
        "canonical_frame_bbox_wsen": [
            min(lons), min(lats), max(lons), max(lats)],
    }, lats, lons)


def _required_specs(scopes: tuple[ActiveCatalogScope, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        spec for scope in scopes for spec in scope.osm_specs))


def _validate_pbf_map(scopes: tuple[ActiveCatalogScope, ...],
                      pbf_map: dict[str, Path]) -> dict[str, dict]:
    required = _required_specs(scopes)
    missing = sorted(set(required) - set(pbf_map))
    extra = sorted(set(pbf_map) - set(required))
    if missing or extra:
        raise ActiveCatalogError(
            f"--pbf map must exactly match selected scopes; "
            f"missing={missing}, extra={extra}")
    identities = {
        spec: _file_identity(pbf_map[spec], what=f"PBF for {spec}")
        for spec in required
    }
    paths = [record["path"] for record in identities.values()]
    if len(paths) != len(set(paths)):
        raise ActiveCatalogError("distinct PBF specs must name distinct files")
    return identities


def _scope_paths(root: Path, scope: ActiveCatalogScope,
                 catalog_version: str) -> dict:
    source_dir = (root / "raw_material" / "catalog_sources" /
                  scope.name / catalog_version)
    artifact_dirs = [
        root / "artifacts" / paths_lib.CATALOGS / dataset / catalog_version
        for dataset in scope.output_datasets
    ]
    osm_bases = []
    for position, spec in enumerate(scope.osm_specs):
        region = spec.rsplit("/", 1)[-1].removesuffix("-latest.osm.pbf")
        region = re.sub(r"[^A-Za-z0-9._-]+", "_", region)
        osm_bases.append(source_dir / f"osm_{position:02d}_{region}")
    osm_selected = (source_dir / "osm_merged" if len(osm_bases) > 1
                    else osm_bases[0])
    selected = source_dir / ("catalog_osm_enc" if scope.enc_state
                             else osm_selected.name)
    return {
        "source_dir": str(source_dir.resolve()),
        "osm_output_bases": [str(path.resolve()) for path in osm_bases],
        "osm_selected_base": str(osm_selected.resolve()),
        "enc_selection": (str((source_dir / "enc_selection.json").resolve())
                          if scope.enc_state else None),
        "enc_output_base": (str((source_dir / "enc").resolve())
                            if scope.enc_state else None),
        "selected_base": str(selected.resolve()),
        "artifact_dirs": [str(path.resolve()) for path in artifact_dirs],
    }


def build_plan(*, farfield_root: Path, scope_names: list[str] | tuple[str, ...],
               pbf_map: dict[str, Path], poly_cache_dir: Path,
               enc_root: Path | None, catalog_version: str,
               dedupe_tolerance_m: float) -> dict:
    """Validate every plan input and return a deterministic addressed plan."""
    root = Path(farfield_root)
    scopes = select_scopes(scope_names)
    try:
        artifact.require_identifier(catalog_version, "catalog version")
    except artifact.ArtifactValidationError as error:
        raise ActiveCatalogError(str(error)) from error
    if (isinstance(dedupe_tolerance_m, bool)
            or not isinstance(dedupe_tolerance_m, (int, float))
            or not math.isfinite(dedupe_tolerance_m)
            or dedupe_tolerance_m < 0):
        raise ActiveCatalogError(
            "dedupe_tolerance_m must be a finite non-negative number")
    if any(scope.enc_state for scope in scopes) and enc_root is None:
        raise ActiveCatalogError("--enc_root is required for MA ENC scopes")
    resolved_root = root.resolve()
    resolved_poly_cache = Path(poly_cache_dir).resolve()
    try:
        resolved_poly_cache.relative_to(resolved_root)
    except ValueError:
        pass
    else:
        raise ActiveCatalogError(
            "poly_cache_dir must be outside farfield_root so report planning "
            "cannot mutate the live data root")

    pbf_identities = _validate_pbf_map(scopes, pbf_map)
    scope_plans = []
    for scope in scopes:
        dataset_tables = []
        lats: list[float] = []
        lons: list[float] = []
        for dataset in scope.bbox_datasets:
            record, table_lats, table_lons = _read_dataset_tables(dataset, root)
            dataset_tables.append(record)
            lats.extend(table_lats)
            lons.extend(table_lons)
        bbox = geometry_helpers.padded_bbox_wsen(
            lats, lons, BBOX_BUFFER_KM)
        pbf_paths = [Path(pbf_identities[spec]["path"])
                     for spec in scope.osm_specs]
        ok, message, details = pbf_coverage.check_coverage(
            list(scope.osm_specs), bbox, Path(poly_cache_dir),
            pbf_paths=pbf_paths)
        if not ok:
            raise ActiveCatalogError(
                f"{scope.name} PBF coverage did not pass: {message}")
        source_coverage = {
            "schema": catalog_lineage.SOURCE_COVERAGE_SCHEMA,
            "status": "passed",
            "message": message,
            "details": details,
        }
        # Validate JSON shape now, rather than after expensive extraction.
        artifact.canonical_json_bytes(source_coverage)
        scope_plans.append({
            "name": scope.name,
            "output_datasets": list(scope.output_datasets),
            "bbox_datasets": list(scope.bbox_datasets),
            "bbox_buffer_km": BBOX_BUFFER_KM,
            "bbox_wsen": list(bbox),
            "dataset_tables": dataset_tables,
            "osm_specs": list(scope.osm_specs),
            "pbf_inputs": [pbf_identities[spec] for spec in scope.osm_specs],
            "enc_policy": ({
                "catalog_state": scope.enc_state,
                "band": ENC_BAND,
                "explicit_cells": False,
                "include_buoys": True,
                "identity_phase": "materialize",
                "published_identity_binding": "selection_sha256",
            } if scope.enc_state else None),
            "source_coverage": source_coverage,
            "paths": _scope_paths(root, scope, catalog_version),
        })

    body = {
        "schema": PLAN_SCHEMA,
        "generator": GENERATOR,
        "generator_git_commit": provenance.git_commit(),
        "farfield_root": str(root.resolve()),
        "poly_cache_dir": str(Path(poly_cache_dir).resolve()),
        "enc_root": (str(Path(enc_root).resolve()) if enc_root is not None
                     else None),
        "catalog_version": catalog_version,
        "dedupe_tolerance_m": float(dedupe_tolerance_m),
        "collision_radius_m": COLLISION_RADIUS_M,
        "osm_geometry_index_mode": OSM_GEOMETRY_INDEX_MODE,
        "report_io": {
            "geofabrik_metadata_cache": "may_fetch_missing_poly_and_index",
            "downloads_pbf": False,
            "downloads_enc": False,
            "builds_catalog": False,
        },
        "scopes": scope_plans,
    }
    return {**body, "plan_digest": artifact.sha256_json(body)}


def _verify_plan_digest(plan: dict, expected_digest: str | None = None) -> str:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ActiveCatalogError("unsupported active catalog plan")
    digest = plan.get("plan_digest")
    body = {key: value for key, value in plan.items() if key != "plan_digest"}
    actual = artifact.sha256_json(body)
    if digest != actual:
        raise ActiveCatalogError("active catalog plan digest is invalid")
    if expected_digest is not None and expected_digest != digest:
        raise ActiveCatalogError(
            f"expected plan digest {expected_digest}, computed {digest}")
    return digest


def _assert_plan_inputs(plan: dict) -> None:
    for scope in plan["scopes"]:
        for record in scope["dataset_tables"]:
            current, _, _ = _read_dataset_tables(
                record["dataset"], Path(plan["farfield_root"]))
            if current != record:
                raise ActiveCatalogError(
                    f"{record['dataset']} canonical dataset tables no longer "
                    "match plan")
        for spec, record in zip(
                scope["osm_specs"], scope["pbf_inputs"], strict=True):
            current = _file_identity(
                Path(record["path"]), what=f"PBF for {spec}")
            if current != record:
                raise ActiveCatalogError(f"PBF no longer matches plan: {spec}")


def _enc_selection_reference(scope_plan: dict, selection: dict,
                             plan: dict) -> dict:
    """Bind the validated selection that commits to catalog and cell bytes."""
    path = Path(scope_plan["paths"]["enc_selection"])
    before = artifact.sha256_file(path)
    reopened = download_enc_cells.validate_selection(
        path, Path(plan["enc_root"]))
    after = artifact.sha256_file(path)
    if before != after or reopened != selection:
        raise ActiveCatalogError(
            "ENC selection changed between validation and publication")
    cells = reopened.get("cells")
    refs = reopened.get("cell_refs")
    if (not isinstance(cells, list) or not cells
            or not isinstance(refs, list) or len(refs) != len(cells)
            or [ref.get("cell") if isinstance(ref, dict) else None
                for ref in refs] != cells):
        raise ActiveCatalogError(
            "ENC selection does not bind one ordered cell ref per cell")
    for ref in refs:
        if (set(ref) != {"cell", "path", "manifest_sha256", "content_sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or not isinstance(ref["manifest_sha256"], str)
                or not SHA256_RE.fullmatch(ref["manifest_sha256"])
                or not isinstance(ref["content_sha256"], str)
                or not SHA256_RE.fullmatch(ref["content_sha256"])):
            raise ActiveCatalogError(
                "ENC selection has an invalid cell manifest/content identity")
    catalog = reopened.get("catalog")
    if (not isinstance(catalog, dict)
            or set(catalog) != {"url", "path", "sha256"}
            or not isinstance(catalog["sha256"], str)
            or not SHA256_RE.fullmatch(catalog["sha256"])):
        raise ActiveCatalogError(
            "catalog-selected ENC plan lacks an exact catalog identity")
    # `before` is a digest of the complete validated JSON document, including
    # `catalog` and every `cell_refs` manifest/content digest above.
    return {"path": str(path.resolve()), "sha256": before}


def _full_catalog_config(scope_plan: dict, selected: Path,
                         selection: dict | None, plan: dict) -> dict:
    return {
        "schema": catalog_schema.FULL_ARTIFACT_SCHEMA,
        "bbox_wsen": list(scope_plan["bbox_wsen"]),
        "osm_specs": list(scope_plan["osm_specs"]),
        "enc_state": (scope_plan["enc_policy"]["catalog_state"]
                      if scope_plan["enc_policy"] else None),
        "enc_cells": list(selection["cells"] if selection else []),
        "enc_available": selection is not None,
        "enc_selection": (_enc_selection_reference(
            scope_plan, selection, plan) if selection else None),
        "dedupe_tolerance_m": plan["dedupe_tolerance_m"],
        "osm_geometry_index_mode": OSM_GEOMETRY_INDEX_MODE,
        "selected_source_feather": str(selected.resolve()),
        "selected_source_sha256": artifact.sha256_file(selected),
        "rows": int(len(catalog_schema.read_frame(selected))),
        "source_coverage": scope_plan["source_coverage"],
    }


def _validate_catalog(directory: Path, dataset: str, version: str,
                      expected_config: dict) -> artifact.ArtifactRef:
    reference = artifact.open_artifact(
        directory, expected_kind=paths_lib.CATALOGS,
        expected_dataset=dataset, expected_version=version)
    manifest = artifact.load_manifest(directory)
    payload = directory / "catalog.feather"
    if (manifest.generator != GENERATOR
            or manifest.git_commit != provenance.git_commit()
            or manifest.upstreams
            or dict(manifest.config) != expected_config
            or manifest.declared_outputs != ("catalog.feather",)
            or payload.is_symlink() or not payload.is_file()
            or artifact.sha256_file(payload)
            != expected_config["selected_source_sha256"]):
        raise ActiveCatalogError(
            f"completed full catalog differs from exact plan: {directory}")
    catalog_schema.read_frame(payload)
    terminal = catalog_lineage.require_passed_source_coverage(reference)
    if terminal != reference:
        raise ActiveCatalogError(f"full catalog was not terminal: {directory}")
    return reference


def _publish_catalog(directory: Path, dataset: str, version: str,
                     selected: Path, config: dict) -> artifact.ArtifactRef:
    if directory.exists() or directory.is_symlink():
        return _validate_catalog(directory, dataset, version, config)
    with publication.published_artifact(
            directory,
            kind=paths_lib.CATALOGS,
            dataset=dataset,
            version=version,
            generator=GENERATOR,
            git_commit=provenance.git_commit(),
            upstreams=(),
            config=config,
            declared_outputs=("catalog.feather",)) as builder:
        shutil.copyfile(selected, builder.output_path("catalog.feather"))
    return _validate_catalog(directory, dataset, version, config)


def materialize(plan: dict, *, expected_plan_digest: str) \
        -> list[artifact.ArtifactRef]:
    """Execute an exact validated plan with no overwrite or implicit inputs."""
    _verify_plan_digest(plan, expected_plan_digest)
    current_commit = provenance.git_commit()
    if plan.get("generator_git_commit") != current_commit:
        raise ActiveCatalogError(
            "reviewed plan generator commit does not match this checkout: "
            f"planned={plan.get('generator_git_commit')!r}, "
            f"current={current_commit!r}")
    _assert_plan_inputs(plan)
    enc_root = Path(plan["enc_root"]) if plan["enc_root"] else None
    outputs = []
    for scope in plan["scopes"]:
        bbox = tuple(scope["bbox_wsen"])
        paths = scope["paths"]
        osm_feathers = []
        for record, output_base in zip(
                scope["pbf_inputs"], paths["osm_output_bases"], strict=True):
            extract_landmarks_from_osm.main(
                pbf_file=Path(record["path"]), bbox=bbox,
                output_path=Path(output_base))
            osm_feathers.append(Path(output_base).with_suffix(".feather"))
        if len(osm_feathers) > 1:
            merge_landmark_feathers.main(
                inputs=osm_feathers,
                output=Path(paths["osm_selected_base"]),
                dedupe_tolerance_m=plan["dedupe_tolerance_m"],
                collision_radius_m=plan["collision_radius_m"])
        osm_selected = Path(paths["osm_selected_base"]).with_suffix(".feather")

        selection = None
        selected = osm_selected
        if scope["enc_policy"]:
            if enc_root is None:
                raise ActiveCatalogError("ENC plan has no enc_root")
            policy = scope["enc_policy"]
            selection_path = Path(paths["enc_selection"])
            download_enc_cells.main(
                cells=None,
                catalog_state=policy["catalog_state"],
                bbox=bbox,
                band=policy["band"],
                output_dir=enc_root,
                selection_output=selection_path,
                force=False)
            selection = download_enc_cells.validate_selection(
                selection_path, enc_root)
            expected_selection = {
                "catalog_state": policy["catalog_state"],
                "bbox": list(bbox),
                "band": policy["band"],
                "explicit_cells": policy["explicit_cells"],
            }
            disagreements = {
                key: (selection.get(key), expected)
                for key, expected in expected_selection.items()
                if selection.get(key) != expected
            }
            if disagreements or not selection["cells"]:
                raise ActiveCatalogError(
                    f"ENC selection disagrees with plan: {disagreements}")
            extract_landmarks_from_enc.main(
                enc_root=enc_root,
                selection_path=selection_path,
                output_path=Path(paths["enc_output_base"]),
                bbox=bbox,
                include_buoys=policy["include_buoys"],
                landmark_type="enc",
                dedupe_tolerance_m=plan["dedupe_tolerance_m"])
            selected = Path(paths["selected_base"]).with_suffix(".feather")
            merge_landmark_feathers.main(
                inputs=[osm_selected,
                        Path(paths["enc_output_base"]).with_suffix(".feather")],
                output=Path(paths["selected_base"]),
                dedupe_tolerance_m=plan["dedupe_tolerance_m"],
                collision_radius_m=plan["collision_radius_m"])

        catalog_schema.read_frame(selected)
        config = _full_catalog_config(scope, selected, selection, plan)
        for dataset, directory in zip(
                scope["output_datasets"], paths["artifact_dirs"], strict=True):
            outputs.append(_publish_catalog(
                Path(directory), dataset, plan["catalog_version"],
                selected, config))
        _assert_plan_inputs(plan)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", required=True, type=Path)
    parser.add_argument(
        "--scope", action="append", required=True,
        help="fixed active scope name, repeated as needed, or exactly 'all'")
    parser.add_argument(
        "--pbf", action="append", required=True,
        help="exact GEOFABRIK_SPEC=/path/to/dated.osm.pbf mapping")
    parser.add_argument(
        "--poly_cache_dir", required=True, type=Path,
        help="Geofabrik .poly/index cache; planning may fetch missing metadata")
    parser.add_argument("--enc_root", type=Path)
    parser.add_argument("--catalog_version", required=True)
    parser.add_argument("--dedupe_tolerance_m", required=True, type=float)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--expected_plan_digest",
        help="required with --materialize; copy it from a reviewed plan")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.materialize != bool(args.expected_plan_digest):
        parser.error(
            "--materialize and --expected_plan_digest must be provided together")
    try:
        plan = build_plan(
            farfield_root=args.farfield_root,
            scope_names=args.scope,
            pbf_map=parse_pbf_map(args.pbf),
            poly_cache_dir=args.poly_cache_dir,
            enc_root=args.enc_root,
            catalog_version=args.catalog_version,
            dedupe_tolerance_m=args.dedupe_tolerance_m)
        if args.materialize:
            references = materialize(
                plan, expected_plan_digest=args.expected_plan_digest)
            print(json.dumps({
                "plan_digest": plan["plan_digest"],
                "published": [reference.to_dict() for reference in references],
            }, indent=2, sort_keys=True))
        else:
            print(json.dumps(plan, indent=2, sort_keys=True))
    except (ActiveCatalogError, artifact.ArtifactError,
            catalog_schema.CatalogSchemaError,
            catalog_lineage.CatalogLineageError,
            publication.PublicationValidationError,
            FileExistsError, FileNotFoundError, OSError, RuntimeError,
            ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
