#!/usr/bin/env python3
"""Publish the map landmarks needed by one LOCI region.

The selected paper catalog identifies its direct untrimmed source composite,
which carries the same OSM, ENC, FAA, or other typed sources as the paper
pipeline. LOCI deliberately uses a different semantic vocabulary from the
far-field bearing matcher, so this producer derives a separate typed artifact
rather than reusing the spatially clipped paper catalog. Geometry is selected
with ``intersects`` against the union envelope of the exact satellite patch
footprints. Lines and polygons crossing the region boundary are retained
whole; a representative-point clip would incorrectly discard them.
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


SCHEMA = "loci_osm_landmarks/v2"
LEGACY_SCHEMA = "loci_osm_landmarks/v1"
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


def _load_source_catalog(reference: artifact.ArtifactRef) -> object:
    source_dir = Path(reference.path)
    manifest = artifact.load_manifest(source_dir)
    payload = source_dir / "catalog.feather"
    if manifest.declared_outputs != ("catalog.feather",):
        raise LociOsmError(
            f"untrimmed catalog declares unexpected outputs: {source_dir}")
    frame = schema.read_frame(payload)
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise LociOsmError(
            f"untrimmed catalog must use EPSG:4326, found {frame.crs}")
    return frame


def _resolve_paper_catalog(farfield_root: Path, paper_group: str) \
        -> tuple[object, region.PaperCatalogInputs]:
    try:
        group, catalog_dir = region.resolve_paper_group(
            farfield_root, paper_group)
        catalog = region.load_paper_catalog(catalog_dir, group=group)
    except region.RegionError as error:
        raise LociOsmError(
            f"cannot resolve current paper group {paper_group!r}: {error}") \
            from error
    return group, catalog


def _validate_region_lineage(
        region_dir: Path, *, farfield_root: Path, dataset: str,
        paper_group: str,
        expected_trajectory_datasets: tuple[str, ...],
        catalog: region.PaperCatalogInputs) \
        -> tuple[artifact.ArtifactRef, dict]:
    region_ref, plan = region.load_region(region_dir)
    if region_ref.dataset != dataset:
        raise LociOsmError(
            f"region dataset mismatch: expected {dataset!r}, found "
            f"{region_ref.dataset!r}")
    manifest = artifact.load_manifest(region_dir)
    plan_datasets = tuple(
        plan.get("trajectory", {}).get("datasets", ()))
    configured_datasets = tuple(
        manifest.config.get("trajectory_datasets", ()))
    if (plan_datasets != expected_trajectory_datasets
            or configured_datasets != expected_trajectory_datasets):
        raise LociOsmError(
            "region trajectories disagree with the current table_common "
            f"paper group {paper_group!r}")
    try:
        current_trajectory = region.load_trajectory_extent(
            farfield_root, expected_trajectory_datasets)
        region.require_trajectory_coverage(
            plan["bbox_wsen"], current_trajectory,
            minimum_margin_m=float(plan.get(
                "minimum_trajectory_margin_m",
                region.DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M)),
            metric_reference_lat_deg=plan.get("metric_reference_lat_deg"))
    except (KeyError, TypeError, ValueError, region.RegionError) as error:
        raise LociOsmError(
            "region no longer covers the current canonical trajectories") \
            from error
    footprint_bbox = plan["grid"]["footprint_bbox_wsen"]
    try:
        region._require_footprint_coverage(
            catalog.selected_region_bbox_wsen, footprint_bbox)
        region._require_footprint_coverage(
            catalog.source_bbox_wsen, footprint_bbox)
    except region.RegionError as error:
        raise LociOsmError(
            "region footprint is not covered by the current "
            f"table_common paper group {paper_group!r}") from error

    catalogs = tuple(
        reference for reference in manifest.upstreams
        if reference.kind == paths_lib.CATALOGS)
    if len(catalogs) != 1:
        raise LociOsmError(
            "region must have exactly one catalog upstream")

    authority_keys = (
        "paper_group",
        "paper_region_bbox_wsen",
        "untrimmed_catalog_manifest_digest",
    )
    if any(key in manifest.config for key in authority_keys):
        if catalogs != (catalog.selected_ref,):
            raise LociOsmError(
                "region was not derived from the current table_common-"
                "selected paper catalog")
        expected_config = {
            "paper_group": paper_group,
            "paper_region_bbox_wsen": list(
                catalog.selected_region_bbox_wsen),
            "catalog_manifest_digest": catalog.selected_ref.manifest_digest,
            "untrimmed_catalog_manifest_digest": (
                catalog.untrimmed_ref.manifest_digest),
        }
        if any(manifest.config.get(key) != value
               for key, value in expected_config.items()):
            raise LociOsmError(
                "region config disagrees with its selected paper catalog")
    elif manifest.config.get("catalog_manifest_digest") \
            != catalogs[0].manifest_digest:
        raise LociOsmError(
            "legacy region config disagrees with its catalog upstream")
    return region_ref, plan


def select_landmarks(frame, footprint_bbox_wsen) -> tuple[object, dict]:
    """Return a compact LOCI-pruned frame and deterministic selection stats."""
    west, south, east, north = region._validate_bbox(
        footprint_bbox_wsen, "satellite footprint bbox")
    footprint = shapely.box(west, south, east, north)

    spatial = frame.loc[frame.geometry.intersects(footprint)]
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
        landmark_types=selected["landmark_type"].tolist(),
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
        "source_rows_by_landmark_type": dict(sorted(Counter(
            frame["landmark_type"]).items())),
        "spatially_intersecting_rows": int(len(spatial)),
        "spatially_intersecting_rows_by_landmark_type": dict(sorted(Counter(
            spatial["landmark_type"]).items())),
        "empty_loci_tag_rows_dropped": int(
            len(spatial) - len(output)),
        "output_rows": int(len(output)),
        "output_rows_by_landmark_type": dict(sorted(Counter(
            output["landmark_type"]).items())),
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
            catalog: region.PaperCatalogInputs, plan: dict,
            paper_group: str, artifact_dataset: str,
            source_landmark_types: list[str],
            required_landmark_types: list[str]) -> dict:
    grid = plan["grid"]
    config = {
        "schema": SCHEMA,
        "paper_group": paper_group,
        "region_manifest_digest": region_ref.manifest_digest,
        "catalog_manifest_digest": catalog.selected_ref.manifest_digest,
        "untrimmed_catalog_manifest_digest": (
            catalog.untrimmed_ref.manifest_digest),
        "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
        "spatial_predicate": "geometry.intersects(footprint_bbox_wsen)",
        "geometry_clipped": False,
        "source_landmark_types": source_landmark_types,
        "required_landmark_types": required_landmark_types,
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
    if catalog.selected_ref.dataset != artifact_dataset:
        config["catalog_dataset"] = catalog.selected_ref.dataset
    return config


def load_loci_osm_artifact(path: Path) \
        -> tuple[artifact.ArtifactRef, object, dict]:
    """Strictly open a completed LOCI OSM artifact."""
    path = Path(path).resolve()
    reference = artifact.open_artifact(path, expected_kind=ARTIFACT_KIND)
    manifest = artifact.load_manifest(path)
    artifact_schema = manifest.config.get("schema")
    if artifact_schema not in (LEGACY_SCHEMA, SCHEMA):
        raise LociOsmError(
            f"unsupported LOCI OSM schema in {path}: "
            f"{artifact_schema!r}")
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
    if artifact_schema == LEGACY_SCHEMA \
            and not frame["landmark_type"].eq("osm").all():
        raise LociOsmError("LOCI OSM Feather contains non-OSM rows")
    if artifact_schema == SCHEMA:
        configured_types = manifest.config.get("source_landmark_types")
        if (not isinstance(configured_types, list)
                or configured_types != sorted(set(configured_types))
                or not set(frame["landmark_type"]).issubset(configured_types)):
            raise LociOsmError(
                "LOCI map Feather source types disagree with its manifest")
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
    if stats.get("schema") != artifact_schema:
        raise LociOsmError(f"invalid LOCI OSM stats schema in {path}")
    if stats.get("output_rows") != len(frame):
        raise LociOsmError(
            f"LOCI OSM stats row count disagrees with Feather in {path}")
    return reference, frame, stats


def materialize(*, farfield_root: Path, dataset: str, paper_group: str,
                region_dir: Path, version: str) -> artifact.ArtifactRef:
    farfield_root = Path(farfield_root).resolve()
    dataset = artifact.require_identifier(dataset, "artifact dataset")
    version = artifact.require_identifier(version, "artifact version")
    group, catalog = _resolve_paper_catalog(
        farfield_root, paper_group)
    region_ref, plan = _validate_region_lineage(
        Path(region_dir).resolve(), farfield_root=farfield_root,
        dataset=dataset,
        paper_group=paper_group,
        expected_trajectory_datasets=group.sequences,
        catalog=catalog)
    source = _load_source_catalog(catalog.untrimmed_ref)
    source_landmark_types = sorted(
        source["landmark_type"].unique().tolist())
    required_landmark_types = sorted(group.landmark_types)
    missing_sources = sorted(
        set(required_landmark_types) - set(source_landmark_types))
    if missing_sources:
        raise LociOsmError(
            f"untrimmed catalog is missing required landmark sources: "
            f"{missing_sources}")
    config = _config(
        region_ref=region_ref, catalog=catalog, plan=plan,
        paper_group=paper_group, artifact_dataset=dataset,
        source_landmark_types=source_landmark_types,
        required_landmark_types=required_landmark_types)
    upstreams = (
        region_ref, catalog.selected_ref, catalog.untrimmed_ref)
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
    parser.add_argument("--paper_group", required=True)
    parser.add_argument("--region_dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = materialize(
        farfield_root=args.farfield_root, dataset=args.dataset,
        paper_group=args.paper_group, region_dir=args.region_dir,
        version=args.version)
    print(reference.path)


if __name__ == "__main__":
    main()
