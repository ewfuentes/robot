#!/usr/bin/env python3
"""Plan and publish one reusable LOCI search-region artifact.

The far-field full catalogs use trajectory bounding boxes padded by a metric
distance on every side.  A LOCI region keeps that box's shape and centre by
insetting every side by the same metric amount until it reaches the requested
area.  If the requested inset would violate trajectory containment, the inset
is capped and the larger containment-limited area is recorded explicitly.

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
    schema as catalog_schema,
)


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
               overlap_fraction: float = DEFAULT_OVERLAP_FRACTION) -> dict:
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


def _region_config(plan: dict, catalog_ref: artifact.ArtifactRef,
                   artifact_dataset: str) -> dict:
    config = {
        "schema": SCHEMA,
        "catalog_manifest_digest": catalog_ref.manifest_digest,
        "target_area_km2": plan["requested_target_area_km2"],
        "actual_area_km2": plan["actual_area_km2"],
        "bbox_wsen": plan["bbox_wsen"],
        "uniform_inset_m": plan["uniform_inset_m"],
        "trajectory_datasets": plan["trajectory"]["datasets"],
        "trajectory_dataset_tables": plan["trajectory"]["dataset_tables"],
        "grid": plan["grid"],
    }
    if catalog_ref.dataset != artifact_dataset:
        config["catalog_dataset"] = catalog_ref.dataset
    return config


def materialize(*, farfield_root: Path, dataset: str,
                trajectory_datasets: Iterable[str], catalog_dir: Path,
                version: str, target_area_km2: float,
                catalog_dataset: str | None = None,
                zoom: int = DEFAULT_ZOOM,
                minimum_trajectory_margin_m: float =
                DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M) \
        -> artifact.ArtifactRef:
    farfield_root = Path(farfield_root).resolve()
    dataset = artifact.require_identifier(dataset, "artifact dataset")
    version = artifact.require_identifier(version, "artifact version")
    trajectory_datasets = tuple(trajectory_datasets)
    expected_catalog_dataset = artifact.require_identifier(
        catalog_dataset if catalog_dataset is not None else dataset,
        "catalog dataset")
    if expected_catalog_dataset not in trajectory_datasets:
        raise RegionError(
            "catalog dataset must be one of the trajectory datasets: "
            f"{expected_catalog_dataset!r} not in {trajectory_datasets!r}")
    catalog_dir = Path(catalog_dir).resolve()
    catalog_ref = artifact.open_artifact(
        catalog_dir, expected_kind=paths_lib.CATALOGS,
        expected_dataset=expected_catalog_dataset)
    catalog_manifest = artifact.load_manifest(catalog_dir)
    if catalog_manifest.config.get("schema") \
            != catalog_schema.FULL_ARTIFACT_SCHEMA:
        raise RegionError(
            "LOCI region input must be a full catalog artifact, not a "
            f"semantic trim: {catalog_dir}")
    source_bbox = catalog_manifest.config.get("bbox_wsen")
    if source_bbox is None:
        raise RegionError(
            f"catalog manifest records no config.bbox_wsen: {catalog_dir}")
    trajectory = load_trajectory_extent(farfield_root, trajectory_datasets)
    plan = derive_region(
        source_bbox, trajectory,
        target_area_km2=target_area_km2,
        minimum_trajectory_margin_m=minimum_trajectory_margin_m,
        zoom=zoom)
    config = _region_config(plan, catalog_ref, dataset)
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
        upstreams=(catalog_ref,), build_inputs=build_inputs)
    recipe = artifact_recipe.build(
        stage="loci_region", stage_config=config,
        build_inputs=build_inputs, identity_upstreams=(catalog_ref,))
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
            git_commit=provenance.git_commit(), upstreams=(catalog_ref,),
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
        overlap_fraction=grid["overlap_fraction"])
    if grid != expected_grid:
        raise RegionError(f"persisted region grid is not reproducible: {path}")
    return reference, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path,
                        default=paths_lib.DEFAULT_ROOT)
    parser.add_argument("--dataset", required=True,
                        help="artifact scope name")
    parser.add_argument(
        "--trajectory_dataset", action="append", default=[],
        help="dataset whose trajectory must be contained; repeat for a "
             "multi-leg shared region (default: --dataset)")
    parser.add_argument("--catalog_dir", required=True, type=Path)
    parser.add_argument(
        "--catalog_dataset",
        help="dataset identity recorded by --catalog_dir (default: "
             "--dataset); set this when publishing a shared artifact scope")
    parser.add_argument("--version", required=True)
    parser.add_argument("--target_area_km2", type=float, default=150.0)
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--minimum_trajectory_margin_m", type=float,
                        default=DEFAULT_MINIMUM_TRAJECTORY_MARGIN_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = args.trajectory_dataset or [args.dataset]
    reference = materialize(
        farfield_root=args.farfield_root, dataset=args.dataset,
        trajectory_datasets=datasets, catalog_dir=args.catalog_dir,
        version=args.version, target_area_km2=args.target_area_km2,
        catalog_dataset=args.catalog_dataset, zoom=args.zoom,
        minimum_trajectory_margin_m=args.minimum_trajectory_margin_m)
    print(reference.path)


if __name__ == "__main__":
    main()
