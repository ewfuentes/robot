"""
Extract OSM landmarks from historical PBF files for the VIGOR dataset.

This script uses the C++ libosmium library (via pybind11) to extract landmarks
from historical OSM PBF files, then converts them to geopandas GeoDataFrame
and saves to Feather format for compatibility with the existing pipeline.
"""

import argparse
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from common.openstreetmap import extract_landmarks_python as elm
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from common.gps import web_mercator
from experimental.overhead_matching.swag.data import landmark_schema as ls


def create_shapely_geometry(geom):
    """Convert C++ geometry variant to Shapely geometry."""
    if isinstance(geom, elm.PointGeometry):
        return Point(geom.coord.lon, geom.coord.lat)

    elif isinstance(geom, elm.LineStringGeometry):
        return LineString([(c.lon, c.lat) for c in geom.coords])

    elif isinstance(geom, elm.PolygonGeometry):
        exterior = [(c.lon, c.lat) for c in geom.exterior]
        holes = [[(c.lon, c.lat) for c in hole] for hole in geom.holes]
        return Polygon(exterior, holes if holes else None)

    elif isinstance(geom, elm.MultiPolygonGeometry):
        polygons = []
        for poly in geom.polygons:
            exterior = [(c.lon, c.lat) for c in poly.exterior]
            holes = [[(c.lon, c.lat) for c in hole] for hole in poly.holes]
            polygons.append(Polygon(exterior, holes if holes else None))
        return MultiPolygon(polygons)

    else:
        raise ValueError(f"Unknown geometry type: {type(geom)}")


def compute_bbox_from_dataset(dataset_path: Path, zoom_level: int = 20):
    """Compute bounding box from VIGOR satellite metadata."""
    sat_metadata = vd.load_satellite_metadata(dataset_path / "satellite", zoom_level)

    min_yx_pixel = sat_metadata[["web_mercator_y", "web_mercator_x"]].min().to_numpy()
    max_yx_pixel = sat_metadata[["web_mercator_y", "web_mercator_x"]].max().to_numpy()

    top, left = web_mercator.pixel_coords_to_latlon(*min_yx_pixel, zoom_level)
    bottom, right = web_mercator.pixel_coords_to_latlon(*max_yx_pixel, zoom_level)

    # Add 10% buffer
    height_delta = top - bottom
    width_delta = right - left

    return elm.BoundingBox(
        left - 0.1 * width_delta,
        bottom - 0.1 * height_delta,
        right + 0.1 * width_delta,
        top + 0.1 * height_delta,
    )


def bbox_from_dataset_path(dataset_path: Path, zoom_level: int = 20):
    """Read bounding box from satellite_bbox.json, falling back to computing from dataset."""
    bbox_path = dataset_path / "satellite_bbox.json"
    if not bbox_path.exists():
        print(f"satellite_bbox.json not found at {bbox_path}, computing from dataset metadata (zoom_level={zoom_level})...")
        return compute_bbox_from_dataset(dataset_path, zoom_level)

    with open(bbox_path) as f:
        meta = json.load(f)

    west, south = meta["west"], meta["south"]
    east, north = meta["east"], meta["north"]

    # Add 10% buffer
    height_delta = north - south
    width_delta = east - west

    return elm.BoundingBox(
        west - 0.1 * width_delta,
        south - 0.1 * height_delta,
        east + 0.1 * width_delta,
        north + 0.1 * height_delta,
    )


def main(
    pbf_path: Path,
    dataset_path: Path | None,
    bbox: tuple[float, float, float, float] | None,
    zoom_level: int,
    output_path: Path,
    node_margin_deg: float = -1.0,
):
    # Determine bounding box
    if bbox is not None:
        bbox_obj = elm.BoundingBox(*bbox)
        print(f"Using provided bounding box: {bbox}")
    elif dataset_path is not None:
        bbox_obj = bbox_from_dataset_path(dataset_path, zoom_level)
        print(
            f"Loaded bounding box from satellite_bbox.json: "
            f"[{bbox_obj.left_deg}, {bbox_obj.bottom_deg}, {bbox_obj.right_deg}, {bbox_obj.top_deg}]"
        )
    else:
        raise ValueError("Must provide either --bbox or --dataset_path")

    # Tag filters (matching original script + waterway)
    tag_filters = {
        "amenity": True,
        "building": True,
        "tourism": True,
        "shop": True,
        "craft": True,
        "emergency": True,
        "geological": True,
        "highway": True,
        "historic": True,
        "landuse": True,
        "leisure": True,
        "man_made": True,
        "military": True,
        "natural": True,
        "office": True,
        "power": True,
        "public_transport": True,
        "railway": True,
        "waterway": True,
        # Far-field / maritime (2026-08): islands (place=island is a top
        # panorama-extracted category) and raw-OSM navigational aids.
        "place": True,
        "seamark:type": True,
    }

    print(f"Extracting landmarks from {pbf_path}...")
    if node_margin_deg >= 0:
        print(f"  Bounding the way-geometry index to bbox + {node_margin_deg} deg "
              f"(~{node_margin_deg * 111:.0f} km): peak memory scales with the "
              f"request area rather than the whole file")
    results = elm.extract_landmarks(str(pbf_path), {"region": bbox_obj}, tag_filters,
                                    node_margin_deg)
    features = [feature for region_id, feature in results]
    print(f"Extracted {len(features)} features")

    if len(features) == 0:
        print("WARNING: No features extracted. Check bounding box and PBF file.")
        return

    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")

    # Convert OsmType enum to string for DataFrame
    osm_type_map = {
        elm.OsmType.NODE: "node",
        elm.OsmType.WAY: "way",
        elm.OsmType.RELATION: "relation"
    }

    # Tags go into a single dict column. See data/landmark_schema.py: the old
    # one-column-per-tag-key layout cost columns x rows for a table that is
    # ~99.8% empty, and every reader immediately converted it back to per-row
    # dicts anyway.
    gdf = ls.build_frame(
        ids=[f"('{osm_type_map[f.osm_type]}', {f.osm_id})" for f in features],
        geometries=[create_shapely_geometry(f.geometry) for f in features],
        landmark_types=["historical" for _ in features],
        tags=[dict(f.tags) for f in features],
    )

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Feather (binary, fast)
    feather_path = output_path.with_suffix(".feather")
    print(f"Saving to {feather_path}...")
    gdf.to_feather(feather_path)

    print(f"Done! Extracted {len(features)} landmarks")
    print(f"  - Nodes: {sum(1 for f in features if f.osm_type == elm.OsmType.NODE)}")
    print(f"  - Ways: {sum(1 for f in features if f.osm_type == elm.OsmType.WAY)}")
    print(f"  - Relations: {sum(1 for f in features if f.osm_type == elm.OsmType.RELATION)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract OSM landmarks from historical PBF files"
    )
    parser.add_argument(
        "--pbf_file", required=True, type=Path, help="Path to OSM PBF file (e.g., ~/Downloads/illinois-latest.osm.pbf)"
    )

    bbox_group = parser.add_mutually_exclusive_group(required=True)
    bbox_group.add_argument(
        "--dataset_path",
        type=Path,
        help="Path to VIGOR dataset (reads bbox from satellite_bbox.json)",
    )
    bbox_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Bounding box as: left bottom right top (e.g., -87.7 41.8 -87.6 41.9)",
    )

    parser.add_argument("--zoom_level", type=int, default=20,
                        help="Zoom level for satellite metadata (used when falling back from satellite_bbox.json, default: 20)")
    parser.add_argument("--output_path", required=True, type=Path, help="Output path for landmarks (will create .feather)")
    parser.add_argument("--node_margin_deg", type=float, default=-1.0,
                        help="If >= 0, hold node locations only within bbox + this "
                             "margin while building way geometry. Peak memory then "
                             "scales with the requested area instead of the whole "
                             "file, which is what lets a country-sized PBF run "
                             "(whole-France otherwise reached 28 GB and climbing). "
                             "Selected ways are unchanged, but their geometry is "
                             "clipped to the retained vertices, so use a margin "
                             "larger than the longest segment you care about. "
                             "Default -1 keeps every node (original behaviour).")

    args = parser.parse_args()

    main(args.pbf_file, args.dataset_path, args.bbox, args.zoom_level, args.output_path,
         args.node_margin_deg)
