"""Extract one compact far-field landmark catalog from an OSM PBF.

This is the far-field owner of OSM-to-Feather conversion.  The historical
cross-view writer deliberately keeps its own schema and behavior; collection
stage 5 calls this target with an explicit WGS84 bounding box instead.

The common libosmium extractor selects an element when it carries any key in
``TAG_FILTER_KEYS``.  Once selected, all of the element's raw OSM tags are
preserved in the compact JSON object.  Pruning is a separate catalog-trimming
decision: doing it here would make the source Feather impossible to revisit
when the trim rules change.

Each invocation writes:

* ``<output>.feather`` with exactly ``catalog.schema.META_COLUMNS``;
* ``<output>.provenance.json`` with the exact bbox, source PBF identity,
  extraction mode, filter keys, output digest, and row diagnostics.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:extract_landmarks_from_osm -- \\
        --pbf_file /path/to/region.osm.pbf \\
        --bbox -71.2 42.2 -70.8 42.5 \\
        --node_margin_deg 0.5 \\
        --output_path /path/to/landmarks/sources/osm_region_v1
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from common.openstreetmap import extract_landmarks_python as elm
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.catalog import schema

PROVENANCE_SCHEMA = "farfield_osm_extraction/v1"
REGION_ID = "requested_bbox"

# These are element-selection keys, not the persisted tag vocabulary.  The
# extractor returns every tag on a selected element and schema.build_frame
# stores that complete mapping as canonical JSON.  Keep the long-standing OSM
# selection families while explicitly adding the far-field structural families
# that the old cross-view target omitted (notably place and seamark:type).
TAG_FILTER_KEYS = (
    "aeroway",
    "amenity",
    "barrier",
    "beacon",
    "boat",
    "bridge",
    "building",
    "craft",
    "dock",
    "emergency",
    "ferry",
    "geological",
    "harbour",
    "highway",
    "historic",
    "industrial",
    "landuse",
    "leisure",
    "lighthouse",
    "lock",
    "man_made",
    "maritime",
    "military",
    "mooring",
    "natural",
    "office",
    "place",
    "power",
    "public_transport",
    "railway",
    "seamark:type",
    "ship",
    "shop",
    "tourism",
    "water",
    "waterway",
    "wreck",
)

_OSM_TYPE_NAMES = {
    elm.OsmType.NODE: "node",
    elm.OsmType.WAY: "way",
    elm.OsmType.RELATION: "relation",
}
_OSM_TYPE_ORDER = {"node": 0, "way": 1, "relation": 2}


def validate_bbox(bbox) -> tuple[float, float, float, float]:
    """Validate and return ``(west, south, east, north)`` in WGS84 degrees."""
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must contain four finite WGS84 degrees") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("bbox must contain four finite WGS84 degrees")
    west, south, east, north = values
    if not -180.0 <= west <= 180.0 or not -180.0 <= east <= 180.0:
        raise ValueError("bbox west/east must lie in [-180, 180]")
    if not -90.0 <= south <= 90.0 or not -90.0 <= north <= 90.0:
        raise ValueError("bbox south/north must lie in [-90, 90]")
    if west >= east:
        raise ValueError(
            "bbox west must be less than east; antimeridian-spanning boxes "
            "must be split into two explicit extractions")
    if south >= north:
        raise ValueError("bbox south must be less than north")
    return values


def validate_node_margin_deg(value: float) -> float:
    """Accept exactly -1 (full index) or a finite nonnegative margin."""
    if isinstance(value, bool):
        raise ValueError("node_margin_deg must be -1 or a nonnegative number")
    try:
        margin = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "node_margin_deg must be -1 or a nonnegative number") from exc
    if not math.isfinite(margin) or (margin < 0.0 and margin != -1.0):
        raise ValueError("node_margin_deg must be -1 or a nonnegative number")
    return margin


def create_shapely_geometry(geometry):
    """Convert one geometry variant from the common extractor to Shapely."""
    if isinstance(geometry, elm.PointGeometry):
        return Point(geometry.coord.lon, geometry.coord.lat)
    if isinstance(geometry, elm.LineStringGeometry):
        return LineString([(coord.lon, coord.lat) for coord in geometry.coords])
    if isinstance(geometry, elm.PolygonGeometry):
        exterior = [(coord.lon, coord.lat) for coord in geometry.exterior]
        holes = [
            [(coord.lon, coord.lat) for coord in hole]
            for hole in geometry.holes
        ]
        return Polygon(exterior, holes or None)
    if isinstance(geometry, elm.MultiPolygonGeometry):
        polygons = []
        for polygon in geometry.polygons:
            exterior = [
                (coord.lon, coord.lat) for coord in polygon.exterior
            ]
            holes = [
                [(coord.lon, coord.lat) for coord in hole]
                for hole in polygon.holes
            ]
            polygons.append(Polygon(exterior, holes or None))
        return MultiPolygon(polygons)
    raise ValueError(f"unknown OSM geometry type {type(geometry).__name__}")


def _feature_record(feature) -> dict:
    try:
        osm_type = _OSM_TYPE_NAMES[feature.osm_type]
    except KeyError as exc:
        raise ValueError(f"unknown OSM element type {feature.osm_type!r}") from exc

    if isinstance(feature.osm_id, bool):
        raise ValueError("OSM element id must be a positive integer")
    osm_id = int(feature.osm_id)
    if osm_id <= 0 or osm_id != feature.osm_id:
        raise ValueError(
            f"OSM {osm_type} id must be a positive integer, got "
            f"{feature.osm_id!r}")

    tags = dict(feature.tags)
    for key, value in tags.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"OSM {osm_type} {osm_id} has a non-string or empty tag key")
        if not isinstance(value, str):
            raise ValueError(
                f"OSM {osm_type} {osm_id} tag {key!r} has non-string "
                f"value {value!r}")

    geometry = create_shapely_geometry(feature.geometry)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(
            f"OSM {osm_type} {osm_id} produced "
            f"{'empty' if geometry.is_empty else 'invalid'} geometry")

    return {
        "id": f"osm:{osm_type}:{osm_id}",
        "osm_id": osm_id,
        "osm_type": osm_type,
        "geometry": geometry,
        "tags": tags,
    }


def features_to_geodataframe(features: list) -> gpd.GeoDataFrame:
    """Build the exact four-column compact far-field catalog frame."""
    records = [_feature_record(feature) for feature in features]
    records.sort(
        key=lambda record: (
            _OSM_TYPE_ORDER[record["osm_type"]], record["osm_id"]
        ))
    return schema.build_frame(
        ids=[record["id"] for record in records],
        geometries=[record["geometry"] for record in records],
        landmark_types=["osm"] * len(records),
        tags=[record["tags"] for record in records],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "resolved_path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _same_stat(left: dict, right: dict) -> bool:
    return (left["resolved_path"] == right["resolved_path"]
            and left["size_bytes"] == right["size_bytes"]
            and left["mtime_ns"] == right["mtime_ns"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_frame_atomically(frame: gpd.GeoDataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".partial.feather",
        dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        frame.to_feather(temporary)
        # Validate the persisted representation, not only the in-memory frame.
        persisted = schema.read_frame(temporary)
        if len(persisted) != len(frame):
            raise RuntimeError(
                f"persisted row count changed from {len(frame)} to "
                f"{len(persisted)}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomically(record: dict, destination: Path) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".partial.json",
        dir=destination.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(record, handle, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _diagnostics(frame: gpd.GeoDataFrame,
                 bbox: tuple[float, float, float, float]) -> dict:
    tags = schema.tag_dicts(frame)
    osm_types = [landmark_id.split(":", 2)[1] for landmark_id in frame["id"]]
    geometry_types = [geometry.geom_type for geometry in frame.geometry]
    tag_key_counts = Counter(key for record in tags for key in record)
    filter_hit_counts = {
        key: tag_key_counts[key] for key in TAG_FILTER_KEYS if tag_key_counts[key]
    }

    bounds = None
    extends_bbox = False
    if len(frame):
        min_x, min_y, max_x, max_y = map(float, frame.total_bounds)
        bounds = [min_x, min_y, max_x, max_y]
        west, south, east, north = bbox
        extends_bbox = (
            min_x < west or min_y < south or max_x > east or max_y > north
        )

    return {
        "rows_out": int(len(frame)),
        "by_osm_type": dict(sorted(Counter(osm_types).items())),
        "by_geometry_type": dict(sorted(Counter(geometry_types).items())),
        "named_rows": sum(bool(record.get("name")) for record in tags),
        "place_rows": sum("place" in record for record in tags),
        "seamark_type_rows": sum("seamark:type" in record for record in tags),
        "tag_filter_hit_counts": filter_hit_counts,
        "geometry_bounds_wgs84": bounds,
        # Ways and relations are selected by an in-bbox vertex; their complete
        # geometry may legitimately extend beyond the requested selection box.
        "geometry_extends_requested_bbox": extends_bbox,
    }


def main(pbf_file: Path, bbox, output_path: Path,
         node_margin_deg: float) -> gpd.GeoDataFrame:
    """Extract, validate, atomically write, and describe one OSM source table."""
    pbf_file = Path(pbf_file)
    if not pbf_file.is_file():
        raise FileNotFoundError(f"OSM PBF does not exist or is not a file: {pbf_file}")
    bbox = validate_bbox(bbox)
    node_margin_deg = validate_node_margin_deg(node_margin_deg)

    source_before = _stat_identity(pbf_file)
    source_sha256 = _sha256(pbf_file)
    source_after_hash = _stat_identity(pbf_file)
    if not _same_stat(source_before, source_after_hash):
        raise RuntimeError(f"source PBF changed while it was hashed: {pbf_file}")

    bbox_object = elm.BoundingBox(*bbox)
    tag_filters = {key: True for key in TAG_FILTER_KEYS}
    print(f"Extracting OSM landmarks from {pbf_file}")
    print(f"  bbox (west south east north): {bbox}")
    if node_margin_deg >= 0.0:
        print(
            "  WARNING: bounded node index is an explicit degraded geometry "
            f"mode (bbox + {node_margin_deg:g} deg); selected ways may lose "
            "vertices outside that margin")
    results = elm.extract_landmarks(
        str(pbf_file), {REGION_ID: bbox_object}, tag_filters, node_margin_deg)
    unexpected_regions = sorted({region for region, _ in results} - {REGION_ID})
    if unexpected_regions:
        raise RuntimeError(
            f"common OSM extractor returned unexpected regions "
            f"{unexpected_regions}")
    source_after_extraction = _stat_identity(pbf_file)
    if not _same_stat(source_after_hash, source_after_extraction):
        raise RuntimeError(f"source PBF changed during extraction: {pbf_file}")

    features = [feature for region, feature in results if region == REGION_ID]
    frame = features_to_geodataframe(features)
    diagnostics = _diagnostics(frame, bbox)
    print(f"  {schema.summarize(frame)}")
    print(f"  OSM types: {diagnostics['by_osm_type']}")
    print(
        f"  named={diagnostics['named_rows']}, "
        f"place={diagnostics['place_rows']}, "
        f"seamark:type={diagnostics['seamark_type_rows']}")

    feather_path = Path(output_path).with_suffix(".feather")
    _write_frame_atomically(frame, feather_path)
    output_identity = _stat_identity(feather_path)
    output_identity["sha256"] = _sha256(feather_path)

    sidecar = feather_path.with_suffix(".provenance.json")
    record = {
        "schema": PROVENANCE_SCHEMA,
        "tool": "farfield/dataset_tools/extract_landmarks_from_osm.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "inputs": {
            "pbf": {
                **source_after_extraction,
                "sha256": source_sha256,
            },
        },
        "arguments": {
            "bbox_wgs84": list(bbox),
            "output_path": str(feather_path),
            "node_margin_deg": node_margin_deg,
            "geometry_index_mode": (
                "full" if node_margin_deg < 0.0 else "bounded_degraded"
            ),
            "tag_filter_keys": list(TAG_FILTER_KEYS),
        },
        "output": {
            **output_identity,
            "catalog_schema_version": schema.SCHEMA_VERSION,
            "columns": list(frame.columns),
        },
        "diagnostics": diagnostics,
    }
    _write_json_atomically(record, sidecar)
    print(f"Wrote {feather_path}")
    print(f"      {sidecar}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pbf_file", required=True, type=Path)
    parser.add_argument(
        "--bbox",
        nargs=4,
        required=True,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="explicit WGS84 selection bounds",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=Path,
        help="output path; the .feather suffix is applied",
    )
    parser.add_argument(
        "--node_margin_deg",
        required=True,
        type=float,
        help="-1 retains the full node-location index; a nonnegative margin "
             "explicitly enables the common extractor's bounded/degraded "
             "geometry mode",
    )
    arguments = parser.parse_args()
    try:
        main(
            pbf_file=arguments.pbf_file,
            bbox=arguments.bbox,
            output_path=arguments.output_path,
            node_margin_deg=arguments.node_margin_deg,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
