#!/usr/bin/env python3
"""Plan and publish one reusable LOCI search-region artifact.

The paper roster selects the authoritative catalog and its 625 km2 or fetch
region. LOCI deliberately uses a smaller region: it keeps the untrimmed source
bbox's shape and centre while insetting every side by the same metric amount
until it reaches the requested area. Publication requires the complete LOCI
patch footprint to remain inside the selected paper region. If the requested
inset would violate trajectory containment, the inset is capped and the
larger containment-limited area is recorded explicitly.

The artifact also owns the exact Web-Mercator patch grid.  Satellite imagery
and OSM landmarks consume the same grid contract, so their boundaries cannot
quietly diverge.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from experimental.overhead_matching.swag.farfield import (
    artifact,
    artifact_identity,
    artifact_recipe,
    paths as paths_lib,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.collection import (
    active_catalogs,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    lineage as catalog_lineage,
)
from experimental.overhead_matching.swag.farfield.paper import table_common


SCHEMA = "loci_region/v1"
ARTIFACT_KIND = "loci_regions"
GENERATOR = "//experimental/overhead_matching/swag/farfield/loci:region"
REGION_OUTPUT = "region.json"

DEFAULT_ZOOM = 20
DEFAULT_TILE_PX = 256
DEFAULT_PATCH_PX = 640
DEFAULT_SOURCE_PX = 640
DEFAULT_OVERLAP_FRACTION = 0.5
DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M = 500.0


class RegionError(ValueError):
    """A region request or persisted region is invalid."""


@dataclass(frozen=True)
class TrajectoryExtent:
    datasets: tuple[str, ...]
    n_points: int
    bbox_wsen: tuple[float, float, float, float]
    dataset_tables: dict[str, dict]


@dataclass(frozen=True)
class PaperCatalogInputs:
    selected_ref: artifact.ArtifactRef
    selected_region_bbox_wsen: tuple[float, float, float, float]
    untrimmed_ref: artifact.ArtifactRef
    source_bbox_wsen: tuple[float, float, float, float]


def _validate_bbox(value: Iterable[float], what: str) \
        -> tuple[float, float, float, float]:
    try:
        west, south, east, north = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise RegionError(f"{what} must contain W,S,E,N numbers") from error
    values = (west, south, east, north)
    if not all(math.isfinite(item) for item in values):
        raise RegionError(f"{what} must contain finite numbers")
    if not (-180.0 <= west < east <= 180.0
            and -90.0 <= south < north <= 90.0):
        raise RegionError(f"{what} is not an ordered WGS84 W,S,E,N box")
    return values


def resolve_paper_group(farfield_root: Path, paper_group: str):
    """Return the roster group and its selected catalog path."""
    try:
        group = table_common.DATASET_GROUP_BY_KEY[paper_group]
    except KeyError as error:
        raise RegionError(f"unknown paper group {paper_group!r}") from error
    if group.catalog_version is None:
        raise RegionError(f"paper group {paper_group!r} has no catalog pin")
    scope = active_catalogs.SCOPE_BY_NAME.get(group.catalog_scope)
    if scope is None or scope.output_datasets != group.sequences:
        raise RegionError(
            f"paper group {paper_group!r} disagrees with active catalog "
            f"scope {group.catalog_scope!r}")
    owner = group.sequences[0]
    catalog_dir = (
        Path(farfield_root) / "artifacts" / paths_lib.CATALOGS
        / owner / group.catalog_version)
    return group, catalog_dir


def load_paper_catalog(catalog_dir: Path, *,
                       group: table_common.DatasetGroup) \
        -> PaperCatalogInputs:
    """Resolve the selected paper catalog and its untrimmed source payload."""
    catalog_dir = Path(catalog_dir).resolve()
    catalog_dataset = group.sequences[0]
    try:
        selected_ref = artifact.open_artifact(
            catalog_dir, expected_kind=paths_lib.CATALOGS,
            expected_dataset=catalog_dataset,
            expected_version=group.catalog_version)
        selected_manifest = artifact.load_manifest(catalog_dir)
    except artifact.ArtifactError as error:
        raise RegionError(
            f"cannot open selected paper catalog {catalog_dir}: {error}") \
            from error
    if selected_manifest.declared_outputs != ("catalog.feather",):
        raise RegionError(
            f"selected paper catalog declares unexpected outputs: "
            f"{catalog_dir}")
    selected_bbox = _validate_bbox(
        selected_manifest.config.get("region_bbox_wsen", ()),
        "selected paper region bbox")
    if group.region_policy == "area625":
        clip_plan = selected_manifest.config.get("clip_plan")
        if (selected_manifest.config.get("region_source") != "clip_bbox_wsen"
                or not isinstance(clip_plan, dict)
                or clip_plan.get("scope") != group.catalog_scope
                or tuple(clip_plan.get("bbox_datasets", ()))
                != group.sequences
                or _validate_bbox(
                    clip_plan.get("bbox_wsen", ()), "catalog clip-plan bbox")
                != selected_bbox):
            raise RegionError(
                "selected paper catalog disagrees with its table_common "
                "area625 policy")
    elif (selected_manifest.config.get("region_source")
          != "full_catalog_bbox_wsen"
          or selected_manifest.config.get("clip_plan") is not None):
        raise RegionError(
            "selected paper catalog disagrees with its table_common "
            "fetch_bbox policy")
    parents = tuple(
        reference for reference in selected_manifest.upstreams
        if reference.kind == paths_lib.CATALOGS)
    if len(parents) != 1:
        raise RegionError(
            "selected paper catalog must have exactly one CATALOGS parent")
    recorded_source = parents[0]
    try:
        untrimmed_ref = artifact.open_artifact(
            recorded_source.path, expected_kind=paths_lib.CATALOGS,
            expected_dataset=catalog_dataset,
            expected_version=recorded_source.version)
        if untrimmed_ref != recorded_source:
            raise RegionError(
                "selected paper catalog's parent identity changed")
        untrimmed_manifest = artifact.load_manifest(untrimmed_ref.path)
        if untrimmed_manifest.declared_outputs != ("catalog.feather",):
            raise RegionError(
                "untrimmed catalog declares unexpected outputs")
        if untrimmed_manifest.config.get("region_bbox_wsen") is not None:
            raise RegionError(
                "selected paper catalog's direct parent is spatially trimmed")
        full_ref = catalog_lineage.require_passed_source_coverage(
            untrimmed_ref)
        full_manifest = artifact.load_manifest(full_ref.path)
    except artifact.ArtifactError as error:
        raise RegionError(
            f"invalid untrimmed catalog lineage for {catalog_dir}: {error}") \
            from error
    source_bbox = _validate_bbox(
        full_manifest.config.get("bbox_wsen", ()),
        "full catalog source bbox")
    if group.region_policy == "fetch_bbox" and selected_bbox != source_bbox:
        raise RegionError(
            "fetch_bbox paper region differs from the full catalog bbox")
    return PaperCatalogInputs(
        selected_ref=selected_ref,
        selected_region_bbox_wsen=selected_bbox,
        untrimmed_ref=untrimmed_ref,
        source_bbox_wsen=source_bbox,
    )


def _metric_scales(mid_lat_deg: float) -> tuple[float, float]:
    metres_per_degree_lat = geometry.METERS_PER_DEG_LAT
    metres_per_degree_lon = (
        metres_per_degree_lat * math.cos(math.radians(mid_lat_deg)))
    if metres_per_degree_lon <= 0.0:
        raise RegionError("region midpoint is too close to a pole")
    return metres_per_degree_lon, metres_per_degree_lat


def metric_dimensions(bbox_wsen: Iterable[float]) -> tuple[float, float]:
    west, south, east, north = _validate_bbox(bbox_wsen, "bbox")
    metres_lon, metres_lat = _metric_scales((south + north) / 2.0)
    return ((east - west) * metres_lon,
            (north - south) * metres_lat)


def _require_footprint_coverage(source_bbox_wsen: Iterable[float],
                                footprint_bbox_wsen: Iterable[float]) -> None:
    source = _validate_bbox(source_bbox_wsen, "source bbox")
    footprint = _validate_bbox(
        footprint_bbox_wsen, "satellite footprint bbox")
    if not (source[0] <= footprint[0] <= footprint[2] <= source[2]
            and source[1] <= footprint[1] <= footprint[3] <= source[3]):
        raise RegionError(
            "satellite patch footprint extends outside the source catalog bbox")


def lat_lon_to_pixel(lat: float, lon: float, zoom: int) \
        -> tuple[float, float]:
    scale = DEFAULT_TILE_PX * (2 ** zoom)
    x = ((lon + 180.0) / 360.0) * scale
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


def pixel_to_lat_lon(x: float, y: float, zoom: int) \
        -> tuple[float, float]:
    scale = DEFAULT_TILE_PX * (2 ** zoom)
    lon = x / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1.0 - 2.0 * y / scale))))
    return lat, lon


def _count_axis(start: float, stop: float, stride: float) -> int:
    if not stride > 0.0 or stop < start:
        raise RegionError("invalid patch-grid axis")
    # This is equivalent to the release downloader's repeated `value +=
    # stride; value <= stop` enumeration without accumulating float error.
    return int(math.floor((stop - start) / stride)) + 1


def nearest_pixel_origin(center_px: float, source_px: int) -> int:
    """Quantize one patch's crop origin to its nearest source pixel."""
    return math.floor(center_px - source_px / 2.0 + 0.5)


def build_grid(bbox_wsen: Iterable[float], *, zoom: int = DEFAULT_ZOOM,
               patch_px: int = DEFAULT_PATCH_PX,
               source_px: int = DEFAULT_SOURCE_PX,
               overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
               contain_footprints: bool = False) -> dict:
    west, south, east, north = _validate_bbox(bbox_wsen, "region bbox")
    if type(zoom) is not int or zoom <= 0:
        raise RegionError("zoom must be a positive integer")
    if type(patch_px) is not int or patch_px <= 0:
        raise RegionError("patch_px must be a positive integer")
    if type(source_px) is not int or source_px <= 0:
        raise RegionError("source_px must be a positive integer")
    if not 0.0 <= overlap_fraction < 1.0:
        raise RegionError("overlap_fraction must lie in [0, 1)")
    stride_px = source_px * (1.0 - overlap_fraction)
    min_x, min_y = lat_lon_to_pixel(north, west, zoom)
    max_x, max_y = lat_lon_to_pixel(south, east, zoom)
    if contain_footprints:
        # Keep inverse Web-Mercator roundoff from placing an edge a few
        # ulps outside the authoritative box.
        half = source_px / 2.0 + 1e-6
        min_x += half
        min_y += half
        max_x -= half
        max_y -= half
    n_x = _count_axis(min_x, max_x, stride_px)
    n_y = _count_axis(min_y, max_y, stride_px)
    last_x = min_x + (n_x - 1) * stride_px
    last_y = min_y + (n_y - 1) * stride_px

    center_north, center_west = pixel_to_lat_lon(min_x, min_y, zoom)
    center_south, center_east = pixel_to_lat_lon(last_x, last_y, zoom)
    half = source_px / 2.0
    footprint_north, footprint_west = pixel_to_lat_lon(
        min_x - half, min_y - half, zoom)
    footprint_south, footprint_east = pixel_to_lat_lon(
        last_x + half, last_y + half, zoom)

    first_origin_x = nearest_pixel_origin(min_x, source_px)
    first_origin_y = nearest_pixel_origin(min_y, source_px)
    last_origin_x = nearest_pixel_origin(last_x, source_px)
    last_origin_y = nearest_pixel_origin(last_y, source_px)
    tile_x_min = first_origin_x // DEFAULT_TILE_PX
    tile_x_max = (last_origin_x + source_px - 1) // DEFAULT_TILE_PX
    tile_y_min = first_origin_y // DEFAULT_TILE_PX
    tile_y_max = (last_origin_y + source_px - 1) // DEFAULT_TILE_PX

    metres_per_pixel = (
        2.0 * math.pi * geometry.EARTH_RADIUS_M
        * math.cos(math.radians((south + north) / 2.0))
        / (DEFAULT_TILE_PX * (2 ** zoom)))
    return {
        "schema": "loci_web_mercator_grid/v1",
        "zoom": zoom,
        "tile_px": DEFAULT_TILE_PX,
        "patch_px": patch_px,
        "source_px": source_px,
        "overlap_fraction": overlap_fraction,
        "contain_footprints": contain_footprints,
        "stride_px": stride_px,
        "min_pixel_xy": [min_x, min_y],
        "max_requested_pixel_xy": [max_x, max_y],
        "last_center_pixel_xy": [last_x, last_y],
        "shape_xy": [n_x, n_y],
        "n_patches": n_x * n_y,
        "center_bbox_wsen": [
            center_west, center_south, center_east, center_north],
        "footprint_bbox_wsen": [
            footprint_west, footprint_south,
            footprint_east, footprint_north],
        "source_tile_range_xyxy": [
            tile_x_min, tile_y_min, tile_x_max, tile_y_max],
        "n_source_tiles": (
            (tile_x_max - tile_x_min + 1)
            * (tile_y_max - tile_y_min + 1)),
        "metres_per_pixel_at_mid_lat": metres_per_pixel,
        "patch_ground_m_at_mid_lat": source_px * metres_per_pixel,
        "stride_ground_m_at_mid_lat": stride_px * metres_per_pixel,
    }


def iter_grid_centres(grid: dict):
    """Yield deterministic row-major ``(x, y)`` Web-Mercator centres."""
    min_x, min_y = grid["min_pixel_xy"]
    n_x, n_y = grid["shape_xy"]
    stride = grid["stride_px"]
    for y_index in range(n_y):
        y = min_y + y_index * stride
        for x_index in range(n_x):
            yield min_x + x_index * stride, y


def load_trajectory_extent(root: Path, datasets: Iterable[str]) \
        -> TrajectoryExtent:
    dataset_names = tuple(datasets)
    if not dataset_names or len(dataset_names) != len(set(dataset_names)):
        raise RegionError("trajectory datasets must be non-empty and unique")
    lats: list[float] = []
    lons: list[float] = []
    table_records: dict[str, dict] = {}
    for dataset in dataset_names:
        artifact.require_identifier(dataset, "trajectory dataset")
        try:
            record, dataset_lats, dataset_lons = \
                active_catalogs.read_dataset_tables(dataset, Path(root))
        except (active_catalogs.ActiveCatalogError, OSError) as error:
            raise RegionError(
                f"cannot bind canonical trajectory tables for {dataset}: "
                f"{error}") \
                from error
        lats.extend(dataset_lats)
        lons.extend(dataset_lons)
        table_records[dataset] = record
    if not all(math.isfinite(value) for value in (*lats, *lons)):
        raise RegionError("trajectory contains non-finite coordinates")
    return TrajectoryExtent(
        datasets=dataset_names,
        n_points=len(lats),
        bbox_wsen=(min(lons), min(lats), max(lons), max(lats)),
        dataset_tables=dict(sorted(table_records.items())),
    )


def require_trajectory_coverage(
        bbox_wsen: Iterable[float], trajectory: TrajectoryExtent, *,
        minimum_margin_m: float,
        metric_reference_lat_deg: float | None = None) -> None:
    """Require a region to contain the current trajectory and its margin."""
    bbox = _validate_bbox(bbox_wsen, "region bbox")
    track = _validate_bbox(trajectory.bbox_wsen, "trajectory bbox")
    if not math.isfinite(minimum_margin_m) or minimum_margin_m < 0.0:
        raise RegionError("minimum trajectory margin must be non-negative")
    mid_lat = ((bbox[1] + bbox[3]) / 2.0
               if metric_reference_lat_deg is None
               else float(metric_reference_lat_deg))
    if not math.isfinite(mid_lat):
        raise RegionError("metric reference latitude must be finite")
    metres_lon, metres_lat = _metric_scales(mid_lat)
    clearances = (
        (track[0] - bbox[0]) * metres_lon,
        (bbox[2] - track[2]) * metres_lon,
        (track[1] - bbox[1]) * metres_lat,
        (bbox[3] - track[3]) * metres_lat,
    )
    if min(clearances) < minimum_margin_m - 1e-5:
        raise RegionError(
            "region does not contain the current canonical trajectories "
            "with their promised clearance")


def derive_region(source_bbox_wsen: Iterable[float],
                  trajectory: TrajectoryExtent, *,
                  target_area_km2: float,
                  minimum_trajectory_margin_m: float =
                  DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M,
                  zoom: int = DEFAULT_ZOOM,
                  patch_px: int = DEFAULT_PATCH_PX,
                  source_px: int = DEFAULT_SOURCE_PX,
                  overlap_fraction: float = DEFAULT_OVERLAP_FRACTION) -> dict:
    source = _validate_bbox(source_bbox_wsen, "source bbox")
    track = _validate_bbox(trajectory.bbox_wsen, "trajectory bbox")
    if not math.isfinite(target_area_km2) or target_area_km2 <= 0.0:
        raise RegionError("target_area_km2 must be positive and finite")
    if (not math.isfinite(minimum_trajectory_margin_m)
            or minimum_trajectory_margin_m < 0.0):
        raise RegionError(
            "minimum_trajectory_margin_m must be finite and non-negative")
    if not (source[0] <= track[0] <= track[2] <= source[2]
            and source[1] <= track[1] <= track[3] <= source[3]):
        raise RegionError("source bbox does not contain every trajectory")

    west, south, east, north = source
    mid_lat = (south + north) / 2.0
    metres_lon, metres_lat = _metric_scales(mid_lat)
    width_m = (east - west) * metres_lon
    height_m = (north - south) * metres_lat
    source_area_m2 = width_m * height_m
    target_area_m2 = target_area_km2 * 1_000_000.0
    if target_area_m2 > source_area_m2:
        raise RegionError(
            f"target area {target_area_km2:g} km^2 exceeds source area "
            f"{source_area_m2 / 1e6:.3f} km^2; this producer trims only")

    discriminant = (width_m - height_m) ** 2 + 4.0 * target_area_m2
    requested_inset_m = (
        (width_m + height_m) - math.sqrt(discriminant)) / 4.0

    track_west, track_south, track_east, track_north = track
    available = {
        "west": (track_west - west) * metres_lon,
        "east": (east - track_east) * metres_lon,
        "south": (track_south - south) * metres_lat,
        "north": (north - track_north) * metres_lat,
    }
    max_containing_inset_m = min(available.values()) \
        - minimum_trajectory_margin_m
    if max_containing_inset_m < 0.0:
        raise RegionError(
            "source bbox lacks the requested minimum trajectory margin")
    inset_m = min(requested_inset_m, max_containing_inset_m)
    containment_limited = inset_m < requested_inset_m - 1e-6

    bbox = (
        west + inset_m / metres_lon,
        south + inset_m / metres_lat,
        east - inset_m / metres_lon,
        north - inset_m / metres_lat,
    )
    bbox = _validate_bbox(bbox, "derived region bbox")
    result_width_m, result_height_m = metric_dimensions(bbox)
    actual_area_km2 = result_width_m * result_height_m / 1_000_000.0
    clearances = {
        "west": (track_west - bbox[0]) * metres_lon,
        "east": (bbox[2] - track_east) * metres_lon,
        "south": (track_south - bbox[1]) * metres_lat,
        "north": (bbox[3] - track_north) * metres_lat,
    }
    if min(clearances.values()) < minimum_trajectory_margin_m - 1e-5:
        raise RegionError("derived region violated trajectory containment")

    grid = build_grid(
        bbox, zoom=zoom, patch_px=patch_px, source_px=source_px,
        overlap_fraction=overlap_fraction)
    _require_footprint_coverage(
        source, grid["footprint_bbox_wsen"])
    return {
        "schema": SCHEMA,
        "source_bbox_wsen": list(source),
        "bbox_wsen": list(bbox),
        "metric_reference_lat_deg": mid_lat,
        "source_size_m": [width_m, height_m],
        "source_area_km2": source_area_m2 / 1_000_000.0,
        "requested_target_area_km2": target_area_km2,
        "actual_area_km2": actual_area_km2,
        "requested_uniform_inset_m": requested_inset_m,
        "uniform_inset_m": inset_m,
        "containment_limited": containment_limited,
        "minimum_trajectory_margin_m": minimum_trajectory_margin_m,
        "trajectory": {
            **asdict(trajectory),
            "datasets": list(trajectory.datasets),
            "bbox_wsen": list(trajectory.bbox_wsen),
            "clearance_m": clearances,
        },
        "grid": grid,
    }


def derive_paper_region(paper_bbox_wsen: Iterable[float],
                        trajectory: TrajectoryExtent, *,
                        minimum_trajectory_margin_m: float =
                        DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M,
                        zoom: int = DEFAULT_ZOOM,
                        patch_px: int = DEFAULT_PATCH_PX,
                        source_px: int = DEFAULT_SOURCE_PX,
                        overlap_fraction: float = DEFAULT_OVERLAP_FRACTION) \
        -> dict:
    """Use the complete authoritative paper box without imagery overfetch."""
    bbox = _validate_bbox(paper_bbox_wsen, "paper region bbox")
    require_trajectory_coverage(
        bbox, trajectory,
        minimum_margin_m=minimum_trajectory_margin_m)
    width_m, height_m = metric_dimensions(bbox)
    metres_lon, metres_lat = _metric_scales((bbox[1] + bbox[3]) / 2.0)
    track_west, track_south, track_east, track_north = trajectory.bbox_wsen
    grid = build_grid(
        bbox, zoom=zoom, patch_px=patch_px, source_px=source_px,
        overlap_fraction=overlap_fraction, contain_footprints=True)
    _require_footprint_coverage(bbox, grid["footprint_bbox_wsen"])
    return {
        "schema": SCHEMA,
        "source_bbox_wsen": list(bbox),
        "bbox_wsen": list(bbox),
        "metric_reference_lat_deg": (bbox[1] + bbox[3]) / 2.0,
        "source_size_m": [width_m, height_m],
        "source_area_km2": width_m * height_m / 1_000_000.0,
        "requested_target_area_km2": width_m * height_m / 1_000_000.0,
        "actual_area_km2": width_m * height_m / 1_000_000.0,
        "requested_uniform_inset_m": 0.0,
        "uniform_inset_m": 0.0,
        "containment_limited": False,
        "minimum_trajectory_margin_m": minimum_trajectory_margin_m,
        "trajectory": {
            **asdict(trajectory),
            "datasets": list(trajectory.datasets),
            "bbox_wsen": list(trajectory.bbox_wsen),
            "clearance_m": {
                "west": (track_west - bbox[0]) * metres_lon,
                "east": (bbox[2] - track_east) * metres_lon,
                "south": (track_south - bbox[1]) * metres_lat,
                "north": (bbox[3] - track_north) * metres_lat,
            },
        },
        "grid": grid,
    }


def _region_config(plan: dict, catalog: PaperCatalogInputs,
                   artifact_dataset: str, paper_group: str) -> dict:
    config = {
        "schema": SCHEMA,
        "paper_group": paper_group,
        "catalog_manifest_digest": catalog.selected_ref.manifest_digest,
        "paper_region_bbox_wsen": list(
            catalog.selected_region_bbox_wsen),
        "untrimmed_catalog_manifest_digest": (
            catalog.untrimmed_ref.manifest_digest),
        "target_area_km2": plan["requested_target_area_km2"],
        "actual_area_km2": plan["actual_area_km2"],
        "bbox_wsen": plan["bbox_wsen"],
        "uniform_inset_m": plan["uniform_inset_m"],
        "trajectory_datasets": plan["trajectory"]["datasets"],
        "trajectory_dataset_tables": plan["trajectory"]["dataset_tables"],
        "grid": plan["grid"],
    }
    if catalog.selected_ref.dataset != artifact_dataset:
        config["catalog_dataset"] = catalog.selected_ref.dataset
    return config


def materialize(*, farfield_root: Path, dataset: str,
                paper_group: str, version: str, target_area_km2: float,
                full_paper_region: bool = False,
                zoom: int = DEFAULT_ZOOM,
                minimum_trajectory_margin_m: float =
                DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M) \
        -> artifact.ArtifactRef:
    farfield_root = Path(farfield_root).resolve()
    dataset = artifact.require_identifier(dataset, "artifact dataset")
    version = artifact.require_identifier(version, "artifact version")
    group, catalog_dir = resolve_paper_group(farfield_root, paper_group)
    trajectory_datasets = group.sequences
    catalog = load_paper_catalog(catalog_dir, group=group)
    trajectory = load_trajectory_extent(farfield_root, trajectory_datasets)
    plan = (derive_paper_region(
        catalog.selected_region_bbox_wsen, trajectory, zoom=zoom,
        minimum_trajectory_margin_m=minimum_trajectory_margin_m)
        if full_paper_region else derive_region(
            catalog.source_bbox_wsen, trajectory,
            target_area_km2=target_area_km2,
            minimum_trajectory_margin_m=minimum_trajectory_margin_m,
            zoom=zoom))
    try:
        _require_footprint_coverage(
            catalog.selected_region_bbox_wsen,
            plan["grid"]["footprint_bbox_wsen"])
    except RegionError as error:
        raise RegionError(
            "LOCI patch footprint is outside the table_common-selected "
            f"paper region for {paper_group!r}") from error
    config = _region_config(plan, catalog, dataset, paper_group)
    build_inputs = {}
    for name, record in trajectory.dataset_tables.items():
        build_inputs[
            f"trajectory_{name}_pano_id_mapping_sha256"] = record[
                "pano_id_mapping"]["sha256"]
        build_inputs[f"trajectory_{name}_frames_gps_sha256"] = record[
            "frames_gps"]["sha256"]
        build_inputs[f"trajectory_{name}_filenames_sha256"] = record[
            "panorama"]["filenames_sha256"]
    stage_config_digest = artifact.sha256_json(config)
    identity = artifact_identity.compute(
        kind=ARTIFACT_KIND, dataset=dataset,
        stage_config_digest=stage_config_digest,
        upstreams=(catalog.selected_ref,), build_inputs=build_inputs)
    recipe = artifact_recipe.build(
        stage="loci_region", stage_config=config,
        build_inputs=build_inputs,
        identity_upstreams=(catalog.selected_ref,))
    destination = (farfield_root / "artifacts" / ARTIFACT_KIND
                   / dataset / version)

    if destination.exists() or destination.is_symlink():
        reference = artifact.open_artifact(
            destination, expected_kind=ARTIFACT_KIND,
            expected_dataset=dataset, expected_version=version)
        existing = json.loads(
            (destination / REGION_OUTPUT).read_text(encoding="utf-8"))
        manifest = artifact.load_manifest(destination)
        if existing != plan or dict(manifest.config) != config:
            raise RegionError(
                f"existing region artifact differs from request: {destination}")
        return reference

    with publication.published_artifact(
            destination, kind=ARTIFACT_KIND, dataset=dataset,
            version=version, generator=GENERATOR,
            git_commit=provenance.git_commit(),
            upstreams=(catalog.selected_ref,),
            config=config, artifact_identity=identity, recipe=recipe,
            declared_outputs=(REGION_OUTPUT,)) as builder:
        artifact.atomic_write_json(
            builder.output_path(REGION_OUTPUT), plan)
    return artifact.open_artifact(
        destination, expected_kind=ARTIFACT_KIND,
        expected_dataset=dataset, expected_version=version)


def load_region(path: Path) -> tuple[artifact.ArtifactRef, dict]:
    path = Path(path).resolve()
    reference = artifact.open_artifact(path, expected_kind=ARTIFACT_KIND)
    try:
        plan = json.loads((path / REGION_OUTPUT).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegionError(f"cannot read region artifact {path}: {error}") \
            from error
    if plan.get("schema") != SCHEMA:
        raise RegionError(
            f"unsupported region schema {plan.get('schema')!r}: {path}")
    _validate_bbox(plan.get("bbox_wsen", ()), "persisted region bbox")
    grid = plan.get("grid")
    if not isinstance(grid, dict) or grid.get("schema") \
            != "loci_web_mercator_grid/v1":
        raise RegionError(f"invalid grid in region artifact: {path}")
    expected_grid = build_grid(
        plan["bbox_wsen"], zoom=grid["zoom"],
        patch_px=grid["patch_px"], source_px=grid["source_px"],
        overlap_fraction=grid["overlap_fraction"],
        contain_footprints=grid.get("contain_footprints", False))
    if grid != expected_grid:
        raise RegionError(f"persisted region grid is not reproducible: {path}")
    _require_footprint_coverage(
        plan.get("source_bbox_wsen", ()), grid["footprint_bbox_wsen"])
    return reference, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path,
                        default=paths_lib.DEFAULT_ROOT)
    parser.add_argument("--dataset", required=True,
                        help="artifact scope name")
    parser.add_argument(
        "--paper_group", required=True,
        choices=tuple(table_common.DATASET_GROUP_BY_KEY),
        help="table_common dataset-group key that owns trajectories and the "
             "selected paper catalog")
    parser.add_argument("--version", required=True)
    parser.add_argument("--target_area_km2", type=float, default=150.0)
    parser.add_argument("--full_paper_region", action="store_true")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--minimum_trajectory_margin_m", type=float,
                        default=DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = materialize(
        farfield_root=args.farfield_root, dataset=args.dataset,
        paper_group=args.paper_group, version=args.version,
        target_area_km2=args.target_area_km2, zoom=args.zoom,
        full_paper_region=args.full_paper_region,
        minimum_trajectory_margin_m=args.minimum_trajectory_margin_m)
    print(reference.path)


if __name__ == "__main__":
    main()
