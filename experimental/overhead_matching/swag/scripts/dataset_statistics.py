import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PIL import Image

from common.gps import web_mercator
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    _TAGS_TO_KEEP_PREFIXES,
    _TAGS_TO_KEEP_SET,
)

ZOOM_LEVEL = 20
METERS_PER_DEG_LAT = 111319.49
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Map dataset names -> pano landmark embedding directory names
PANO_LANDMARK_DIR_MAP = {
    "Chicago": "Chicago",
    "Seattle": "Seattle",
    "Boston": "boston_snowy",
    "nightdrive": "nightdrive",
    "mapillary/Framingham": "Framingham",
    "mapillary/Gap": "Gap",
    "mapillary/MiamiBeach": "MiamiBeach",
    "mapillary/Middletown": "Middletown",
    "mapillary/Norway": "Norway",
    "mapillary/SanFrancisco_mapillary": "SanFrancisco_mapillary",
    "mapillary/post_hurricane_ian": "post_hurricane_ian",
}


@dataclass
class DatasetConfig:
    name: str
    path: Path
    landmark_file: str  # relative to path, e.g. "landmarks/v4_202001.feather"


@dataclass
class PanoLandmarkStats:
    num_panoramas: int
    num_with_landmarks: int
    total_landmarks: int
    avg_landmarks_per_pano: float


@dataclass
class DatasetStats:
    name: str
    num_panoramas: int
    num_satellites: int
    panorama_resolution: tuple[int, int]
    satellite_resolution: tuple[int, int]
    satellite_ground_coverage_m: tuple[float, float]
    ground_normalized: bool
    bbox: dict
    geographic_extent_km2: float
    num_landmarks_raw: int
    num_landmarks_pruned: int
    landmark_version: str
    osm_date: str | None  # extracted from landmark filename
    trajectory_km: float | None
    capture_date: str | None
    satellite_source: str | None
    meters_per_pixel: float
    pano_landmarks: PanoLandmarkStats | None


def parse_osm_date(landmark_version: str) -> str | None:
    """Extract OSM snapshot date from landmark filename.

    Examples: v4_202001 -> 2020-01, boston -> None,
    Framingham_v1_260101 -> 2026-01-01, Norway_v1_251201 -> 2025-12-01
    """
    # Match YYMMDD or YYYYMM at end of version string
    m = re.search(r'(\d{6})$', landmark_version)
    if not m:
        return None
    digits = m.group(1)
    # Distinguish YYYYMM (e.g. 202001) from YYMMDD (e.g. 260101)
    # YYYYMM: first 4 digits are a plausible year (2000-2099), last 2 are month (01-12)
    year4 = int(digits[:4])
    month2 = int(digits[4:6])
    if 2000 <= year4 <= 2099 and 1 <= month2 <= 12:
        return f"{digits[:4]}-{digits[4:6]}"
    else:
        # YYMMDD format (e.g. 260101 -> 2026-01-01)
        return f"20{digits[:2]}-{digits[2:4]}-{digits[4:6]}"


def discover_datasets(base: Path) -> list[DatasetConfig]:
    configs = []

    for city in ["Chicago", "NewYork", "SanFrancisco", "Seattle"]:
        p = base / city
        if p.is_dir() and (p / "panorama").is_dir():
            configs.append(DatasetConfig(city, p, "landmarks/v4_202001.feather"))

    if (base / "Boston" / "panorama").is_dir():
        configs.append(DatasetConfig("Boston", base / "Boston", "landmarks/boston.feather"))

    if (base / "nightdrive" / "panorama").is_dir():
        configs.append(DatasetConfig("nightdrive", base / "nightdrive", "landmarks/boston.feather"))

    mapillary_dir = base / "mapillary"
    if mapillary_dir.is_dir():
        for loc_dir in sorted(mapillary_dir.iterdir()):
            if not loc_dir.is_dir() or not (loc_dir / "panorama").is_dir():
                continue
            landmarks_dir = loc_dir / "landmarks"
            if not landmarks_dir.is_dir():
                continue
            feather_files = list(landmarks_dir.glob("*.feather"))
            if not feather_files:
                continue
            landmark_file = f"landmarks/{feather_files[0].name}"
            configs.append(DatasetConfig(f"mapillary/{loc_dir.name}", loc_dir, landmark_file))

    return configs


def get_most_common_resolution(directory: Path, sample_size: int = 100) -> tuple[int, int]:
    from collections import Counter
    resolutions = Counter()
    count = 0
    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(f) as img:
                resolutions[img.size] += 1
            count += 1
            if count >= sample_size:
                break
    if not resolutions:
        raise FileNotFoundError(f"No images found in {directory}")
    return resolutions.most_common(1)[0][0]


def parse_satellite_coords(sat_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sat_dir.iterdir():
        if f.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = f.stem.split("_")
        if len(parts) == 3 and parts[0] == "satellite":
            lat, lon = float(parts[1]), float(parts[2])
            rows.append((lat, lon))
    return pd.DataFrame(rows, columns=["lat", "lon"])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6378.137 * math.asin(math.sqrt(a))


def compute_trajectory_km(mapping_path: Path) -> float | None:
    import csv
    with open(mapping_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None
    total = 0.0
    for i in range(1, len(rows)):
        total += haversine_km(
            float(rows[i - 1]["lat"]), float(rows[i - 1]["lon"]),
            float(rows[i]["lat"]), float(rows[i]["lon"]),
        )
    return total


def compute_bbox_area_km2(north: float, south: float, east: float, west: float) -> float:
    height_km = (north - south) * METERS_PER_DEG_LAT / 1000.0
    mid_lat = (north + south) / 2.0
    width_km = (east - west) * METERS_PER_DEG_LAT * math.cos(math.radians(mid_lat)) / 1000.0
    return width_km * height_km


def load_pano_landmark_stats(pano_embed_base: Path, dataset_name: str) -> PanoLandmarkStats | None:
    dir_name = PANO_LANDMARK_DIR_MAP.get(dataset_name)
    if dir_name is None:
        return None
    pkl_path = pano_embed_base / dir_name / "embeddings" / "embeddings.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    panos = data.get("panoramas", {})
    if not panos:
        return None
    total_landmarks = sum(len(p.get("landmarks", [])) for p in panos.values())
    num_with = sum(1 for p in panos.values() if p.get("landmarks"))
    return PanoLandmarkStats(
        num_panoramas=len(panos),
        num_with_landmarks=num_with,
        total_landmarks=total_landmarks,
        avg_landmarks_per_pano=total_landmarks / len(panos),
    )


def compute_stats(config: DatasetConfig, pano_embed_base: Path | None) -> DatasetStats:
    pano_dir = config.path / "panorama"
    sat_dir = config.path / "satellite"

    num_panos = len([f for f in pano_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES])
    num_sats = len([f for f in sat_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES])

    pano_res = get_most_common_resolution(pano_dir)
    sat_res = get_most_common_resolution(sat_dir)

    sat_coords = parse_satellite_coords(sat_dir)
    north = sat_coords["lat"].max()
    south = sat_coords["lat"].min()
    east = sat_coords["lon"].max()
    west = sat_coords["lon"].min()
    bbox = {"north": north, "south": south, "east": east, "west": west}

    mid_lat = (north + south) / 2.0
    mpp = web_mercator.get_meters_per_pixel(mid_lat, ZOOM_LEVEL)

    ground_normalized = False
    tile_meta_path = config.path / "satellite_tile_metadata.csv"
    sb_path = config.path / "satellite_bbox.json"
    sb_data = {}
    if sb_path.exists():
        with open(sb_path) as f:
            sb_data = json.load(f)

    if tile_meta_path.exists():
        tile_meta = pd.read_csv(tile_meta_path)
        sat_ground_w = tile_meta["width_meters"].median()
        sat_ground_h = tile_meta["height_meters"].median()
    elif sb_data.get("target_ground_m") is not None:
        target_ground_m = sb_data["target_ground_m"]
        ground_normalized = True
        sat_ground_w = target_ground_m
        sat_ground_h = target_ground_m
    else:
        sat_ground_w = sat_res[0] * mpp
        sat_ground_h = sat_res[1] * mpp

    half_tile_lat = (sat_ground_h / 2.0) / METERS_PER_DEG_LAT
    half_tile_lon = (sat_ground_w / 2.0) / (METERS_PER_DEG_LAT * math.cos(math.radians(mid_lat)))
    extent_km2 = compute_bbox_area_km2(
        north + half_tile_lat, south - half_tile_lat,
        east + half_tile_lon, west - half_tile_lon,
    )

    # OSM landmarks
    landmark_path = config.path / config.landmark_file
    landmark_version = Path(config.landmark_file).stem
    osm_date = parse_osm_date(landmark_version)

    df = gpd.read_feather(landmark_path) if landmark_path.suffix == ".feather" else gpd.read_file(landmark_path)
    num_raw = len(df)

    meta_cols = {"id", "geometry", "landmark_type", "geometry_px"}
    kept_cols = [
        c for c in df.columns
        if c not in meta_cols
        and (c in _TAGS_TO_KEEP_SET or any(c.startswith(p) for p in _TAGS_TO_KEEP_PREFIXES))
    ]

    if kept_cols:
        kept_df = df[kept_cols]
        non_ts_cols = [c for c in kept_cols if not pd.api.types.is_datetime64_any_dtype(kept_df[c])]
        num_pruned = int(kept_df[non_ts_cols].notna().any(axis=1).sum())
    else:
        num_pruned = 0

    # Capture date and trajectory length from pipeline_metadata.json
    capture_date = None
    trajectory_km = None
    pm_path = config.path / "pipeline_metadata.json"
    if pm_path.exists():
        with open(pm_path) as f:
            pm = json.load(f)
        capture_date = pm.get("capture_date")
        trajectory_km = pm.get("trajectory_km")

    # Fall back to computing trajectory from pano_id_mapping.csv
    if trajectory_km is None:
        mapping_path = config.path / "pano_id_mapping.csv"
        if mapping_path.exists():
            trajectory_km = compute_trajectory_km(mapping_path)

    # Satellite source
    sat_source = None
    if sb_data:
        source = sb_data.get("source", "unknown")
        esri_date = sb_data.get("esri_release_date")
        sat_source = f"{source}" + (f" ({esri_date})" if esri_date else "")

    # Panorama landmarks
    pano_landmarks = None
    if pano_embed_base:
        pano_landmarks = load_pano_landmark_stats(pano_embed_base, config.name)

    return DatasetStats(
        name=config.name,
        num_panoramas=num_panos,
        num_satellites=num_sats,
        panorama_resolution=pano_res,
        satellite_resolution=sat_res,
        satellite_ground_coverage_m=(sat_ground_w, sat_ground_h),
        ground_normalized=ground_normalized,
        bbox=bbox,
        geographic_extent_km2=extent_km2,
        trajectory_km=trajectory_km,
        num_landmarks_raw=num_raw,
        num_landmarks_pruned=num_pruned,
        landmark_version=landmark_version,
        osm_date=osm_date,
        capture_date=capture_date,
        satellite_source=sat_source,
        meters_per_pixel=mpp,
        pano_landmarks=pano_landmarks,
    )


CORE_VIGOR_CITIES = {"Chicago", "NewYork", "SanFrancisco", "Seattle"}


def format_value(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if val > 100:
            return f"{val:,.1f}"
        return f"{val:.1f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def print_transposed_table(title: str, stats_list: list[DatasetStats]):
    """Print a transposed table: rows are metrics, columns are datasets."""
    if not stats_list:
        return

    # Build rows: (label, [values...])
    rows = []
    for s in stats_list:
        sat_cover = f"{s.satellite_ground_coverage_m[0]:.1f}x{s.satellite_ground_coverage_m[1]:.1f}"
        if s.ground_normalized:
            sat_cover += "*"
        pano_res = f"{s.panorama_resolution[0]}x{s.panorama_resolution[1]}"
        vals = {
            "Panoramas": format_value(s.num_panoramas),
            "Satellite patches": format_value(s.num_satellites),
            "Pano resolution": pano_res,
            "Sat ground cover (m)": sat_cover,
            "Area (km²)": format_value(s.geographic_extent_km2),
            "Trajectory (km)": format_value(s.trajectory_km),
            "OSM landmarks": format_value(s.num_landmarks_raw),
            "OSM date": s.osm_date or "—",
            "Pano landmarks": format_value(s.pano_landmarks.total_landmarks) if s.pano_landmarks else "—",
            "Avg pano lmks/pano": format_value(s.pano_landmarks.avg_landmarks_per_pano) if s.pano_landmarks else "—",
            "Capture date": s.capture_date or "—",
            "Satellite source": s.satellite_source or "—",
        }
        rows.append(vals)

    # Use short display names for columns
    short_names = []
    for s in stats_list:
        name = s.name.replace("mapillary/", "")
        short_names.append(name)

    metric_keys = list(rows[0].keys())

    # Compute column widths
    label_w = max(len(k) for k in metric_keys)
    col_widths = [max(len(short_names[i]), max(len(rows[i][k]) for k in metric_keys)) for i in range(len(stats_list))]

    # Print
    print(f"\n{title}")
    print("-" * 40)

    # Header row
    header = f"{'':>{label_w}}"
    for i, name in enumerate(short_names):
        header += f"  {name:>{col_widths[i]}}"
    print(header)

    # Data rows
    for key in metric_keys:
        row = f"{key:>{label_w}}"
        for i in range(len(stats_list)):
            row += f"  {rows[i][key]:>{col_widths[i]}}"
        print(row)


def print_stats(all_stats: list[DatasetStats]):
    print("VIGOR DATASET STATISTICS")
    print("-" * 40)

    for s in all_stats:
        print(f"\n  {s.name}")
        print(f"  ----")
        print(f"  Panoramas:               {s.num_panoramas:,}")
        print(f"  Satellite patches:       {s.num_satellites:,}")
        print(f"  Panorama resolution:     {s.panorama_resolution[0]} x {s.panorama_resolution[1]}")
        print(f"  Satellite resolution:    {s.satellite_resolution[0]} x {s.satellite_resolution[1]}")
        print(f"  Meters/pixel (zoom 20):  {s.meters_per_pixel:.4f}")
        coverage_note = " (normalized)" if s.ground_normalized else " (native z20)"
        print(f"  Sat ground coverage:     {s.satellite_ground_coverage_m[0]:.1f} m x {s.satellite_ground_coverage_m[1]:.1f} m{coverage_note}")
        print(f"  Bounding box:            N={s.bbox['north']:.4f}  S={s.bbox['south']:.4f}  E={s.bbox['east']:.4f}  W={s.bbox['west']:.4f}")
        print(f"  Geographic extent:       {s.geographic_extent_km2:.2f} km²")
        if s.trajectory_km is not None:
            print(f"  Trajectory length:       {s.trajectory_km:.1f} km")
        print(f"  OSM landmarks (raw):     {s.num_landmarks_raw:,}")
        print(f"  OSM landmarks (pruned):  {s.num_landmarks_pruned:,}")
        print(f"  Landmark version:        {s.landmark_version}")
        if s.osm_date:
            print(f"  OSM snapshot date:       {s.osm_date}")
        if s.capture_date:
            print(f"  Capture date:            {s.capture_date}")
        if s.satellite_source:
            print(f"  Satellite source:        {s.satellite_source}")
        if s.pano_landmarks:
            pl = s.pano_landmarks
            print(f"  Pano landmarks:          {pl.total_landmarks:,} total across {pl.num_panoramas:,} panos "
                  f"({pl.num_with_landmarks:,} with, {pl.num_panoramas - pl.num_with_landmarks:,} without, "
                  f"avg {pl.avg_landmarks_per_pano:.1f}/pano)")

    # Split into core VIGOR and extended
    core = [s for s in all_stats if s.name in CORE_VIGOR_CITIES]
    extended = [s for s in all_stats if s.name not in CORE_VIGOR_CITIES]

    print_transposed_table("CORE VIGOR CITIES", core)
    print_transposed_table("EXTENDED DATASETS (* = ground-normalized satellite patches)", extended)


def main():
    parser = argparse.ArgumentParser(description="Gather statistics for VIGOR datasets")
    parser.add_argument("--dataset_base", type=str, required=True,
                        help="Base path to VIGOR datasets")
    parser.add_argument("--pano_embed_base", type=str, default=None,
                        help="Base path to panorama landmark embeddings")
    args = parser.parse_args()

    base = Path(args.dataset_base)
    pano_embed_base = Path(args.pano_embed_base) if args.pano_embed_base else None
    configs = discover_datasets(base)

    print(f"Discovered {len(configs)} datasets:")
    for c in configs:
        print(f"  {c.name}: {c.path}")
    print()

    all_stats = []
    for config in configs:
        print(f"Processing {config.name}...", flush=True)
        stats = compute_stats(config, pano_embed_base)
        all_stats.append(stats)

    print_stats(all_stats)


if __name__ == "__main__":
    main()
