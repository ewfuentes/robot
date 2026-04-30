from pathlib import Path
from typing import NamedTuple

import pandas as pd

from common.gps import web_mercator


class TileBBox(NamedTuple):
    west_lon: float
    south_lat: float
    east_lon: float
    north_lat: float


def satellite_filename_to_center(name: str) -> tuple[float, float]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "satellite":
        raise ValueError(f"unexpected satellite filename: {name!r}")
    return float(parts[1]), float(parts[2])


def center_zoom_to_bbox(lat: float, lon: float, zoom: int, tile_px: int = 640) -> TileBBox:
    """Mirror VigorDataset's pixel-bbox math (vigor_dataset.py:1098-1106).

    web_mercator pixel y increases southward, x increases eastward; the tile
    occupies the square (cy ± tile_px/2, cx ± tile_px/2). The corners are
    converted back to lat/lon, which gives the geographic bbox covering exactly
    the same pixel rectangle the VIGOR loader will consume.
    """
    cy, cx = web_mercator.latlon_to_pixel_coords(lat, lon, zoom)
    half = tile_px / 2.0
    north_lat, west_lon = web_mercator.pixel_coords_to_latlon(cy - half, cx - half, zoom)
    south_lat, east_lon = web_mercator.pixel_coords_to_latlon(cy + half, cx + half, zoom)
    return TileBBox(
        west_lon=float(west_lon),
        south_lat=float(south_lat),
        east_lon=float(east_lon),
        north_lat=float(north_lat),
    )


def boston_csv_to_bboxes(csv_path: Path) -> dict[str, TileBBox]:
    """Boston ships an explicit per-tile bbox; prefer it over center-derived bbox."""
    df = pd.read_csv(csv_path)
    return {
        row.file_name: TileBBox(
            west_lon=float(row.west_lon),
            south_lat=float(row.south_lat),
            east_lon=float(row.east_lon),
            north_lat=float(row.north_lat),
        )
        for row in df.itertuples(index=False)
    }
