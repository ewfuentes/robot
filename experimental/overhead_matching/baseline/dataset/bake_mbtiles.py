"""Run planetiler to convert a state-level .osm.pbf into a vector .mbtiles.

Cropped to the VIGOR city bbox + a buffer so we don't waste minutes on data we
will never render. The output mbtiles file feeds pymgl at render time.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from experimental.overhead_matching.baseline.dataset import (
    city_pbf_map,
    tile_geometry,
)

logger = logging.getLogger(__name__)

DEFAULT_DUMPS_DIR = Path("/data/overhead_matching/datasets/osm_dumps")


def _runfiles_dir() -> Path:
    runfiles = os.environ.get("RUNFILES_DIR")
    if runfiles:
        return Path(runfiles)
    # __file__ is a symlink into the runfiles tree; do NOT resolve symlinks
    # or we'll end up in the source repo, which has no .runfiles ancestor.
    here = Path(os.path.abspath(__file__))
    for parent in here.parents:
        if parent.name.endswith(".runfiles"):
            return parent
    raise RuntimeError("could not locate runfiles tree; invoke via `bazel run`")


def planetiler_jar_path() -> Path:
    candidates = [
        _runfiles_dir() / "planetiler_jar" / "file" / "planetiler.jar",
        _runfiles_dir() / "_main" / "external" / "planetiler_jar" / "file" / "planetiler.jar",
    ]
    for c in candidates:
        if c.exists():
            return c
    env = os.environ.get("PLANETILER_JAR")
    if env and Path(env).exists():
        return Path(env)
    raise FileNotFoundError(
        f"planetiler.jar not found; tried {[str(c) for c in candidates]}"
    )


def compute_vigor_bbox(satellite_dir: Path, buffer_km: float = 1.0) -> tile_geometry.TileBBox:
    lats: list[float] = []
    lons: list[float] = []
    for p in satellite_dir.iterdir():
        if p.suffix != ".png":
            continue
        try:
            lat, lon = tile_geometry.satellite_filename_to_center(p.name)
        except ValueError:
            continue
        lats.append(lat)
        lons.append(lon)
    if not lats:
        raise RuntimeError(f"no satellite_*.png files found under {satellite_dir}")
    deg_per_km_lat = 1.0 / 110.574
    # Conservative cosine using the southernmost latitude (largest deg/km).
    import math
    deg_per_km_lon = 1.0 / (111.320 * math.cos(math.radians(min(abs(min(lats)), abs(max(lats))))))
    pad_lat = buffer_km * deg_per_km_lat
    pad_lon = buffer_km * deg_per_km_lon
    return tile_geometry.TileBBox(
        west_lon=min(lons) - pad_lon,
        south_lat=min(lats) - pad_lat,
        east_lon=max(lons) + pad_lon,
        north_lat=max(lats) + pad_lat,
    )


def run_planetiler(
    pbf_path: Path,
    output_mbtiles: Path,
    bbox: tile_geometry.TileBBox | None = None,
    java_xmx: str = "8g",
    download_dir: Path | None = None,
    tmp_dir: Path | None = None,
    force: bool = False,
) -> None:
    if output_mbtiles.exists() and not force:
        logger.info("mbtiles exists; skipping bake (use --force to rebuild): %s", output_mbtiles)
        return

    output_mbtiles.parent.mkdir(parents=True, exist_ok=True)
    if download_dir is not None:
        download_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java",
        f"-Xmx{java_xmx}",
        "-jar",
        str(planetiler_jar_path()),
        f"--osm-path={pbf_path}",
        f"--output={output_mbtiles}",
        # OpenMapTiles profile needs lake_centerline / water-polygons / natural_earth
        # auxiliary datasets. --download fetches them on first run; subsequent
        # runs reuse the cached copies in download_dir.
        "--download",
    ]
    if download_dir is not None:
        cmd.append(f"--download_dir={download_dir}")
    if tmp_dir is not None:
        cmd.append(f"--tmpdir={tmp_dir}")
    if bbox is not None:
        cmd.append(
            f"--bounds={bbox.west_lon},{bbox.south_lat},{bbox.east_lon},{bbox.north_lat}"
        )

    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True, choices=city_pbf_map.cities())
    p.add_argument("--vigor-root", type=Path, default=Path("/data/overhead_matching/datasets/VIGOR"))
    p.add_argument("--dumps-dir", type=Path, default=DEFAULT_DUMPS_DIR)
    p.add_argument("--output-dir", type=Path, default=Path("/data/overhead_matching/baseline/mbtiles"))
    p.add_argument("--download-dir", type=Path,
                   default=Path("/data/overhead_matching/baseline/planetiler_sources"),
                   help="cache for lake_centerline / water-polygons / natural_earth")
    p.add_argument("--tmp-dir", type=Path,
                   default=Path("/data/overhead_matching/baseline/planetiler_tmp"))
    p.add_argument("--buffer-km", type=float, default=1.0)
    p.add_argument("--java-xmx", default="8g")
    p.add_argument("--no-crop", action="store_true",
                   help="bake the full PBF instead of cropping to the city bbox")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    pbf = city_pbf_map.resolve_pbf(args.city, args.dumps_dir)
    if not pbf.exists():
        logger.error("PBF not found: %s", pbf)
        return 1

    output = args.output_dir / f"{city_pbf_map.region_label(args.city)}.mbtiles"

    bbox = None
    if not args.no_crop:
        sat_dir = args.vigor_root / args.city / "satellite"
        if not sat_dir.is_dir():
            logger.error("satellite dir not found: %s", sat_dir)
            return 1
        bbox = compute_vigor_bbox(sat_dir, buffer_km=args.buffer_km)
        logger.info("city bbox (with %.1f km buffer): %s", args.buffer_km, bbox)

    run_planetiler(
        pbf,
        output,
        bbox=bbox,
        java_xmx=args.java_xmx,
        download_dir=args.download_dir,
        tmp_dir=args.tmp_dir,
        force=args.force,
    )
    logger.info("baked: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
