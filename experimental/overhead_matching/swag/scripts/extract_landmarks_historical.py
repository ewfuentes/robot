"""
Extract OSM landmarks from historical PBF files for the VIGOR dataset.

This script uses the C++ libosmium library (via pybind11) to extract landmarks
from historical OSM PBF files, then converts them to geopandas GeoDataFrame
and saves to Feather format for compatibility with the existing pipeline.
"""

import argparse
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from common.openstreetmap import extract_landmarks_python as elm
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from common.gps import web_mercator


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


def compute_bbox_from_dataset(dataset_path: Path, zoom_level: int):
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


def main(
    pbf_path: Path,
    dataset_path: Path | None,
    bbox: tuple[float, float, float, float] | None,
    zoom_level: int,
    output_path: Path,
):
    # Determine bounding box
    if bbox is not None:
        bbox_obj = elm.BoundingBox(*bbox)
        print(f"Using provided bounding box: {bbox}")
    elif dataset_path is not None:
        bbox_obj = compute_bbox_from_dataset(dataset_path, zoom_level)
        print(
            f"Computed bounding box from dataset: "
            f"[{bbox_obj.left}, {bbox_obj.bottom}, {bbox_obj.right}, {bbox_obj.top}]"
        )
    else:
        raise ValueError("Must provide either --bbox or --dataset_path")

    # Tag filters (matching original script)
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
    }

    print(f"Extracting landmarks from {pbf_path}...")
    features = elm.extract_landmarks(str(pbf_path), bbox_obj, tag_filters)
    print(f"Extracted {len(features)} features")

    if len(features) == 0:
        print("WARNING: No features extracted. Check bounding box and PBF file.")
        return

    # Convert to GeoDataFrame
    print("Converting to GeoDataFrame...")
    data = {
        "osm_type": [f.osm_type for f in features],
        "osm_id": [f.osm_id for f in features],
        "geometry": [create_shapely_geometry(f.geometry) for f in features],
        "landmark_type": [f.landmark_type for f in features],
    }

    # Flatten important tags as columns
    all_tag_keys = set()
    for f in features:
        all_tag_keys.update(f.tags.keys())

    for key in sorted(all_tag_keys):
        data[key] = [f.tags.get(key, None) for f in features]

    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Feather (binary, fast)
    feather_path = output_path.with_suffix(".feather")
    print(f"Saving to {feather_path}...")
    gdf.to_feather(feather_path)

    # Save as JSON (text, compatible)
    json_path = output_path.with_suffix(".json")
    print(f"Saving to {json_path}...")
    json_path.write_text(gdf.to_json(na="drop"))

    print(f"Done! Extracted {len(features)} landmarks")
    print(f"  - Nodes: {sum(1 for f in features if f.osm_type == 'node')}")
    print(f"  - Ways: {sum(1 for f in features if f.osm_type == 'way')}")
    print(f"  - Relations: {sum(1 for f in features if f.osm_type == 'relation')}")


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
        help="Path to VIGOR dataset (will compute bbox from satellite metadata)",
    )
    bbox_group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Bounding box as: left bottom right top (e.g., -87.7 41.8 -87.6 41.9)",
    )

    parser.add_argument("--output_path", required=True, type=Path, help="Output path for landmarks (will create .json and .feather)")
    parser.add_argument("--zoom_level", type=int, default=20, help="Zoom level for dataset (default: 20)")

    args = parser.parse_args()

    main(args.pbf_file, args.dataset_path, args.bbox, args.zoom_level, args.output_path)
