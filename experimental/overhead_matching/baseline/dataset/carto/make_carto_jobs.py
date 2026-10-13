"""Write the OSM Carto render job list for one VIGOR environment.

One job per satellite tile without a rendered counterpart in <env>/satellite_osm_carto/:
{"filename": "<satellite stem>.png", "bbox": [xmin, ymin, xmax, ymax] (EPSG:3857 m), "px": int}.
The bbox is the satellite tile's exact footprint (tile_geometry.center_zoom_to_bbox, zoom 20,
640 px or satellite_bbox.json:source_px for resolution-normalised envs); px is HALF that
footprint (320 for a 640 grid) so the render is a native z19 Carto raster, upscaled x2 by
consumers to CVG-Text's look. render_carto.py consumes the list inside the docker container.
"""
import argparse
import json
import math
import sys
from pathlib import Path

from common.gps import web_mercator
from experimental.overhead_matching.baseline.dataset import city_pbf_map, tile_geometry

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
OUTPUT_SUBDIR = "satellite_osm_carto"


def region_for(env: str) -> str:
    """Docker volume/container label: PBF stem with '-' -> '_' (new-york-200101 -> new_york_200101)."""
    return city_pbf_map.CITY_TO_PBF[env].pbf_filename.removesuffix(".osm.pbf").replace("-", "_")


def merc(lat: float, lon: float) -> tuple[float, float]:
    r = web_mercator.EARTH_RADIUS_M
    return r * math.radians(lon), r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=city_pbf_map.cities())
    ap.add_argument("--vigor-root", type=Path, default=Path("/data/overhead_matching/datasets/VIGOR"))
    ap.add_argument("--out", type=Path, required=True, help="jobs JSON to write")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true", help="include tiles already rendered")
    a = ap.parse_args()

    env_dir = a.vigor_root / a.env
    bbox_px = 640
    bbox_json = env_dir / "satellite_bbox.json"
    if bbox_json.exists():
        bbox_px = json.loads(bbox_json.read_text()).get("source_px", 640)
    px = round(bbox_px / 2)
    out_dir = env_dir / OUTPUT_SUBDIR

    sats = sorted(p for p in (env_dir / "satellite").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    jobs = []
    for p in sats:
        if not a.force and (out_dir / (p.stem + ".png")).exists():
            continue
        lat, lon = tile_geometry.satellite_filename_to_center(p.name)
        b = tile_geometry.center_zoom_to_bbox(lat, lon, zoom=20, tile_px=bbox_px)
        xmin, ymin = merc(b.south_lat, b.west_lon)
        xmax, ymax = merc(b.north_lat, b.east_lon)
        jobs.append({"filename": p.stem + ".png", "bbox": [xmin, ymin, xmax, ymax], "px": px})
        if a.limit and len(jobs) >= a.limit:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(jobs))
    side = jobs[0]["bbox"][2] - jobs[0]["bbox"][0] if jobs else float("nan")
    print(f"{a.env}: {len(jobs)} jobs of {len(sats)} satellite tiles, px={px} over {side:.1f} merc m -> {a.out}")
    print(f"run: experimental/overhead_matching/baseline/dataset/carto/run_carto_render.sh {region_for(a.env)} {a.out} {out_dir} 16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
