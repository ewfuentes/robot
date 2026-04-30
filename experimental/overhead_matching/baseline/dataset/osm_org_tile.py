"""Fetch openstreetmap.org standard tiles and stitch them to match a bbox.

Used by compare_tiles to add an OSM.org column to the visual QA composite, so
we can sanity-check our planetiler+pymgl+OSM-Bright render against the
canonical osm.org rendering of the same geography.

Caches downloaded tiles to /tmp/osm_org_tile_cache/{z}_{x}_{y}.png so repeated
runs don't hammer the OSM tile servers. openstreetmap.org tile servers cap at
zoom 19; we fetch z19 and downscale to fit our zoom-20 bbox.
"""
import io
import time
from pathlib import Path

import requests
from PIL import Image

from common.gps import web_mercator
from experimental.overhead_matching.baseline.dataset.tile_geometry import TileBBox

CACHE_DIR = Path("/tmp/osm_org_tile_cache")
USER_AGENT = "robot2-baseline-compare (https://github.com/anthropics/claude-code)"
OSM_ORG_MAX_Z = 19


def _fetch_tile(z: int, x: int, y: int) -> Image.Image:
    cache = CACHE_DIR / f"{z}_{x}_{y}.png"
    if cache.exists():
        return Image.open(cache).convert("RGB")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    cache.write_bytes(r.content)
    # Polite gap between requests to respect tile-server usage policy.
    time.sleep(0.1)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def fetch_for_bbox(bbox: TileBBox, target_px: int = 640, zoom: int = OSM_ORG_MAX_Z) -> Image.Image:
    """Returns a target_px × target_px PIL image cropped to bbox from osm.org tiles."""
    py_n, px_w = web_mercator.latlon_to_pixel_coords(bbox.north_lat, bbox.west_lon, zoom)
    py_s, px_e = web_mercator.latlon_to_pixel_coords(bbox.south_lat, bbox.east_lon, zoom)

    tx_min = int(px_w // 256)
    tx_max = int(px_e // 256)
    ty_min = int(py_n // 256)
    ty_max = int(py_s // 256)

    stitched = Image.new("RGB", ((tx_max - tx_min + 1) * 256, (ty_max - ty_min + 1) * 256))
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            tile = _fetch_tile(zoom, tx, ty)
            stitched.paste(tile, ((tx - tx_min) * 256, (ty - ty_min) * 256))

    left = float(px_w) - tx_min * 256
    right = float(px_e) - tx_min * 256
    top = float(py_n) - ty_min * 256
    bottom = float(py_s) - ty_min * 256
    cropped = stitched.crop((int(round(left)), int(round(top)),
                             int(round(right)), int(round(bottom))))
    return cropped.resize((target_px, target_px), Image.LANCZOS)
