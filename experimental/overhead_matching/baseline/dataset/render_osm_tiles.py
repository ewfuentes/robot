"""Render OSM tiles into <vigor>/<City>/satellite_osm/, matching the bbox of
each existing satellite tile so vigor_dataset.py can load them by toggling
its `satellite_subdir` config field.
"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from experimental.overhead_matching.baseline.dataset import (
    city_pbf_map,
    style_template,
    tile_geometry,
)

logger = logging.getLogger(__name__)


@dataclass
class RenderJob:
    sat_filename: str
    bbox: tile_geometry.TileBBox
    output_path: Path


def discover_jobs(
    city_dir: Path,
    output_subdir: str = "satellite_osm",
    zoom: int = 20,
    tile_px: int = 640,
    force: bool = False,
) -> list[RenderJob]:
    sat_dir = city_dir / "satellite"
    out_dir = city_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[RenderJob] = []
    for p in sorted(sat_dir.iterdir()):
        if p.suffix != ".png":
            continue
        out = out_dir / p.name
        if out.exists() and not force:
            continue
        try:
            lat, lon = tile_geometry.satellite_filename_to_center(p.name)
        except ValueError:
            logger.warning("skipping %s: bad filename", p.name)
            continue
        bbox = tile_geometry.center_zoom_to_bbox(lat, lon, zoom=zoom, tile_px=tile_px)
        jobs.append(RenderJob(sat_filename=p.name, bbox=bbox, output_path=out))
    return jobs


def render_jobs(
    style_str: str,
    jobs: list[RenderJob],
    width: int = 640,
    height: int = 640,
) -> None:
    from pymgl import Map  # local import; require xvfb available

    m = Map(style_str, width=width, height=height)
    for job in tqdm(jobs, desc="render"):
        m.setBounds(
            job.bbox.west_lon, job.bbox.south_lat,
            job.bbox.east_lon, job.bbox.north_lat,
        )
        png = m.renderPNG()
        job.output_path.write_bytes(png)


def render_city(
    city: str,
    vigor_root: Path,
    mbtiles_path: Path,
    output_subdir: str = "satellite_osm",
    zoom: int = 20,
    tile_px: int = 640,
    limit: int | None = None,
    force: bool = False,
) -> None:
    city_dir = vigor_root / city
    jobs = discover_jobs(
        city_dir,
        output_subdir=output_subdir,
        zoom=zoom,
        tile_px=tile_px,
        force=force,
    )
    logger.info("discovered %d unrendered tiles for %s", len(jobs), city)
    if limit is not None:
        jobs = jobs[:limit]
        logger.info("limited to first %d tiles", len(jobs))
    if not jobs:
        return

    logger.info("rendering %d tiles -> %s", len(jobs), city_dir / output_subdir)
    style_str = style_template.render_style(mbtiles_path)
    render_jobs(style_str, jobs, width=tile_px, height=tile_px)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, choices=city_pbf_map.cities())
    p.add_argument("--vigor-root", type=Path, default=Path("/data/overhead_matching/datasets/VIGOR"))
    p.add_argument("--mbtiles-dir", type=Path, default=Path("/data/overhead_matching/baseline/mbtiles"))
    p.add_argument("--output-subdir", default="satellite_osm")
    p.add_argument("--zoom", type=int, default=20)
    p.add_argument("--tile-px", type=int, default=640)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="re-render even if output PNG already exists")
    args = p.parse_args()

    if not os.environ.get("DISPLAY") and not os.environ.get("XVFB_RUN"):
        logger.warning(
            "no DISPLAY set; pymgl needs an X server. Wrap with `xvfb-run -a` if headless."
        )

    mbtiles = args.mbtiles_dir / f"{city_pbf_map.region_label(args.city)}.mbtiles"
    if not mbtiles.exists():
        logger.error("mbtiles not found: %s. Run bake_mbtiles first.", mbtiles)
        return 1

    render_city(
        city=args.city,
        vigor_root=args.vigor_root,
        mbtiles_path=mbtiles,
        output_subdir=args.output_subdir,
        zoom=args.zoom,
        tile_px=args.tile_px,
        limit=args.limit,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
