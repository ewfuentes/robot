#!/usr/bin/env python3
"""Publish the OSM landmarks needed by one LOCI region.

The source is the immutable far-field catalog, complete within its recorded
bounding box. LOCI deliberately uses a different semantic vocabulary from
the far-field bearing matcher, so this producer derives a separate typed
artifact rather than reusing a far-field semantic trim. Geometry is selected
with ``intersects`` against the union envelope of the exact satellite patch
footprints. In particular,
lines and polygons crossing the region boundary are retained whole; a
representative-point clip would incorrectly discard them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import shapely

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    artifact_recipe,
    paths as paths_lib,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.loci import region
from experimental.overhead_matching.swag.model import semantic_landmark_utils


SCHEMA = "loci_osm_landmarks/v1"
ARTIFACT_KIND = "loci_osm_landmarks"
GENERATOR = "//experimental/overhead_matching/swag/farfield/loci:osm"
LANDMARK_OUTPUT = "landmarks.feather"
STATS_OUTPUT = "stats.json"


class LociOsmError(ValueError):
    """The requested LOCI OSM artifact or one of its inputs is invalid."""


def _canonical_tags(tags: dict[str, str]) -> str:
    return json.dumps(
        tags, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _vocabulary_digest() -> str:
    # Ordering is load-bearing for the correspondence checkpoint's tag-key
    # embedding indices, so hash the ordered vocabulary rather than a set.
    return artifact.sha256_json(
        list(semantic_landmark_utils._TAGS_TO_KEEP))


def _validate_source_catalog(
        catalog_dir: Path, *, catalog_dataset: str) \
        -> tuple[artifact.ArtifactRef, artifact.ArtifactManifest, object]:
    catalog_dir = Path(catalog_dir).resolve()
    reference = artifact.open_artifact(
        catalog_dir, expected_kind=paths_lib.CATALOGS,
        expected_dataset=catalog_dataset)
    manifest = artifact.load_manifest(catalog_dir)
    if manifest.config.get("schema") != schema.FULL_ARTIFACT_SCHEMA:
        raise LociOsmError(
            "LOCI OSM input must be a full catalog artifact, not a "
            f"semantic trim: {catalog_dir}")
    payload = catalog_dir / "catalog.feather"
    if manifest.declared_outputs != ("catalog.feather",):
        raise LociOsmError(
            f"full catalog declares unexpected outputs: {catalog_dir}")
    frame = schema.read_frame(payload)
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise LociOsmError(
            f"full catalog must use EPSG:4326, found {frame.crs}")
    return reference, manifest, frame


def _validate_region_lineage(
        region_dir: Path, catalog_ref: artifact.ArtifactRef, *, dataset: str) \
        -> tuple[artifact.ArtifactRef, dict]:
    region_ref, plan = region.load_region(region_dir)
    if region_ref.dataset != dataset:
        raise LociOsmError(
            f"region dataset mismatch: expected {dataset!r}, found "
            f"{region_ref.dataset!r}")
    manifest = artifact.load_manifest(region_dir)
    if catalog_ref not in manifest.upstreams:
        raise LociOsmError(
            "region was not derived from the selected full catalog: "
            f"{region_dir}")
    return region_ref, plan


def select_landmarks(frame, footprint_bbox_wsen) -> tuple[object, dict]:
    """Return a compact LOCI-pruned frame and deterministic selection stats."""
    west, south, east, north = region._validate_bbox(
        footprint_bbox_wsen, "satellite footprint bbox")
    footprint = shapely.box(west, south, east, north)

    osm_mask = frame["landmark_type"].eq("osm")
    spatial_mask = osm_mask & frame.geometry.intersects(footprint)
    spatial = frame.loc[spatial_mask]
    decoded = schema.tag_dicts(spatial)

    keep_positions: list[int] = []
    pruned_records: list[dict[str, str]] = []
    for position, props in enumerate(decoded):
        pruned = dict(semantic_landmark_utils.prune_landmark(props))
        if not pruned:
            continue
        keep_positions.append(position)
        pruned_records.append(pruned)

    selected = spatial.iloc[keep_positions]
    output = schema.build_frame(
        ids=selected["id"].tolist(),
        geometries=selected.geometry.tolist(),
        landmark_types=["osm"] * len(selected),
        tags=pruned_records,
        crs="EPSG:4326",
    )
    geometry_types = dict(sorted(Counter(
        output.geometry.geom_type).items()))
    tag_occurrences = sum(len(record) for record in pruned_records)
    unique_pairs = {
        (key, value)
        for record in pruned_records
        for key, value in record.items()
    }
    stats = {
        "schema": SCHEMA,
        "footprint_bbox_wsen": [west, south, east, north],
        "source_rows": int(len(frame)),
        "source_osm_rows": int(osm_mask.sum()),
        "spatially_intersecting_osm_rows": int(len(spatial)),
        "empty_loci_tag_rows_dropped": int(
            len(spatial) - len(output)),
        "output_rows": int(len(output)),
        "output_geometry_types": geometry_types,
        "output_tag_occurrences": int(tag_occurrences),
        "output_unique_tag_keys": int(len({
            key for key, _ in unique_pairs})),
        "output_unique_tag_values": int(len({
            value for _, value in unique_pairs})),
        "output_unique_key_value_pairs": int(len(unique_pairs)),
    }
    return output, stats


def _config(*, region_ref: artifact.ArtifactRef,
            catalog_ref: artifact.ArtifactRef, plan: dict,
            artifact_dataset: str) -> dict:
    grid = plan["grid"]
    config = {
        "schema": SCHEMA,
        "region_manifest_digest": region_ref.manifest_digest,
        "catalog_manifest_digest": catalog_ref.manifest_digest,
        "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
        "spatial_predicate": "geometry.intersects(footprint_bbox_wsen)",
        "geometry_clipped": False,
        "landmark_type": "osm",
        "tag_pruner": (
            "experimental.overhead_matching.swag.model."
            "semantic_landmark_utils.prune_landmark"),
        "ordered_tag_vocabulary_sha256": _vocabulary_digest(),
        "output_schema": {
            "columns": list(schema.META_COLUMNS),
            "crs": "EPSG:4326",
            "tags": "canonical JSON containing only LOCI-kept key/value pairs",
        },
    }
    if catalog_ref.dataset != artifact_dataset:
        config["catalog_dataset"] = catalog_ref.dataset
    return config


def load_loci_osm_artifact(path: Path) \
        -> tuple[artifact.ArtifactRef, object, dict]:
    """Strictly open a completed LOCI OSM artifact."""
    path = Path(path).resolve()
    reference = artifact.open_artifact(path, expected_kind=ARTIFACT_KIND)
    manifest = artifact.load_manifest(path)
    if manifest.config.get("schema") != SCHEMA:
        raise LociOsmError(
            f"unsupported LOCI OSM schema in {path}: "
            f"{manifest.config.get('schema')!r}")
    if manifest.declared_outputs != (LANDMARK_OUTPUT, STATS_OUTPUT):
        raise LociOsmError(
            f"unexpected LOCI OSM outputs in {path}: "
            f"{manifest.declared_outputs}")
    frame = schema.read_frame(path / LANDMARK_OUTPUT)
    if tuple(frame.columns) != schema.META_COLUMNS:
        raise LociOsmError(
            f"LOCI OSM Feather must contain exactly {schema.META_COLUMNS}")
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise LociOsmError(
            f"LOCI OSM Feather must use EPSG:4326, found {frame.crs}")
    if not frame["landmark_type"].eq("osm").all():
        raise LociOsmError("LOCI OSM Feather contains non-OSM rows")
    records = schema.tag_dicts(frame)
    for index, (raw, props) in enumerate(zip(frame["tags"], records)):
        if not props:
            raise LociOsmError(
                f"LOCI OSM Feather has empty tags at row {index}")
        if raw != _canonical_tags(props):
            raise LociOsmError(
                f"LOCI OSM Feather has non-canonical tags at row {index}")
        if semantic_landmark_utils.prune_landmark(props) \
                != frozenset(props.items()):
            raise LociOsmError(
                f"LOCI OSM Feather has non-LOCI tags at row {index}")
    try:
        stats = json.loads(
            (path / STATS_OUTPUT).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LociOsmError(f"cannot read {path / STATS_OUTPUT}: {error}") \
            from error
    if stats.get("schema") != SCHEMA:
        raise LociOsmError(f"invalid LOCI OSM stats schema in {path}")
    if stats.get("output_rows") != len(frame):
        raise LociOsmError(
            f"LOCI OSM stats row count disagrees with Feather in {path}")
    return reference, frame, stats


def materialize(*, farfield_root: Path, dataset: str, region_dir: Path,
                catalog_dir: Path, version: str,
                catalog_dataset: str | None = None) -> artifact.ArtifactRef:
    farfield_root = Path(farfield_root).resolve()
    dataset = artifact.require_identifier(dataset, "artifact dataset")
    version = artifact.require_identifier(version, "artifact version")
    expected_catalog_dataset = artifact.require_identifier(
        catalog_dataset if catalog_dataset is not None else dataset,
        "catalog dataset")
    catalog_ref, _, source = _validate_source_catalog(
        catalog_dir, catalog_dataset=expected_catalog_dataset)
    region_ref, plan = _validate_region_lineage(
        Path(region_dir).resolve(), catalog_ref, dataset=dataset)
    config = _config(
        region_ref=region_ref, catalog_ref=catalog_ref, plan=plan,
        artifact_dataset=dataset)
    upstreams = (region_ref, catalog_ref)
    stage_config_digest = artifact.sha256_json(config)
    identity = artifact_identity.compute(
        kind=ARTIFACT_KIND, dataset=dataset,
        stage_config_digest=stage_config_digest,
        upstreams=upstreams, build_inputs={})
    recipe = artifact_recipe.build(
        stage="loci_osm_landmarks", stage_config=config, build_inputs={},
        identity_upstreams=upstreams)
    destination = (farfield_root / "artifacts" / ARTIFACT_KIND
                   / dataset / version)

    if destination.exists() or destination.is_symlink():
        reference, _, _ = load_loci_osm_artifact(destination)
        manifest = artifact.load_manifest(destination)
        if (reference.dataset != dataset or reference.version != version
                or dict(manifest.config) != config
                or set(manifest.upstreams) != set(upstreams)):
            raise LociOsmError(
                f"existing LOCI OSM artifact differs from request: "
                f"{destination}")
        return reference

    output, stats = select_landmarks(
        source, plan["grid"]["footprint_bbox_wsen"])
    with publication.published_artifact(
            destination, kind=ARTIFACT_KIND, dataset=dataset,
            version=version, generator=GENERATOR,
            git_commit=provenance.git_commit(), upstreams=upstreams,
            config=config, artifact_identity=identity, recipe=recipe,
            declared_outputs=(LANDMARK_OUTPUT, STATS_OUTPUT)) as builder:
        output.to_feather(builder.output_path(LANDMARK_OUTPUT))
        artifact.atomic_write_json(
            builder.output_path(STATS_OUTPUT), stats)
    reference, _, _ = load_loci_osm_artifact(destination)
    return reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path,
                        default=paths_lib.DEFAULT_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--region_dir", required=True, type=Path)
    parser.add_argument("--catalog_dir", required=True, type=Path)
    parser.add_argument(
        "--catalog_dataset",
        help="dataset identity recorded by --catalog_dir (default: "
             "--dataset); set this for a shared artifact scope")
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = materialize(
        farfield_root=args.farfield_root, dataset=args.dataset,
        region_dir=args.region_dir, catalog_dir=args.catalog_dir,
        version=args.version, catalog_dataset=args.catalog_dataset)
    print(reference.path)


if __name__ == "__main__":
    main()
