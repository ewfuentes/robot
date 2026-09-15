"""Build a region DSM (or same-source bare-earth DEM ablation) from
classified LiDAR tiles: the deterministic CLD-4 recipe.

Keeps ground/building/bridge classes, rasterizes the per-cell maximum
(frozen after a validation-tile comparison against a high quantile; run
--compare_tile to produce that evidence), fills pinholes with a fixed
neighborhood-median rule, and falls back to the provider bare-earth DEM
elsewhere -- which, being hydro-flattened, is also the declared
water-surface treatment.

Example (Boston Harbor):

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:build_dsm -- \
        --laz_dir /data/farfield_matching/raw_material/usgs_3dep_ma_centraleastern_2021/laz \
        --dem_surface /data/farfield_matching/artifacts/dem_surfaces/boston_harbor/v1_dem/surface \
        --dst_crs EPSG:6348 --resolution_m 1.0 \
        --bounds_latlon 42.18410 42.43373 -71.14662 -70.82276 \
        --output /data/farfield_matching/artifacts/dem_surfaces/boston_harbor/v1_dsm/surface
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    lidar_dsm,
    terrain,
)


def _bounds_from_latlon(south: float, north: float, west: float, east: float,
                        dst_crs: str) -> tuple[float, float, float, float]:
    """Outer integer-aligned bounds of the lat/lon box in the dst CRS."""
    xs, ys = [], []
    for lat in (south, north):
        for lon in (west, east):
            x, y = terrain.utm_from_latlon(lat, lon, dst_crs)
            xs.append(x)
            ys.append(y)
    return (math.floor(min(xs)), math.floor(min(ys)),
            math.ceil(max(xs)), math.ceil(max(ys)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laz_dir", type=Path, default=None)
    parser.add_argument("--laz", type=Path, nargs="*", default=None)
    parser.add_argument("--dem_surface", type=Path, default=None,
                        help="HeightField base path used as bare-earth "
                             "fallback (and thus water surface)")
    parser.add_argument("--dst_crs", default="EPSG:6348",
                        help="NAD83(2011) UTM 19N per the plan's preferred "
                             "project coordinates")
    parser.add_argument("--resolution_m", type=float, default=1.0)
    parser.add_argument("--bounds_latlon", type=float, nargs=4, default=None,
                        metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    parser.add_argument("--bounds_xy", type=float, nargs=4, default=None,
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--keep_classes", type=int, nargs="+",
                        default=list(lidar_dsm.DSM_KEEP_CLASSES))
    parser.add_argument("--fill_passes", type=int, default=2)
    parser.add_argument("--surface_kind", default="static_dsm")
    parser.add_argument("--checksums", type=Path, default=None,
                        help="sha256sum file from the raw download; embeds "
                             "source hashes without re-hashing 80 GB")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--compare_tile", type=Path, default=None,
                        help="Run the max-vs-quantile statistic comparison "
                             "on this one tile and exit (validation-tile "
                             "evidence for freezing the rule)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Base path (no suffix); writes surface .npz + "
                             ".json and provenance.npz beside it")
    args = parser.parse_args()

    if args.compare_tile is not None:
        report = lidar_dsm.compare_statistics(
            args.compare_tile, resolution_m=args.resolution_m,
            keep_classes=tuple(args.keep_classes))
        print(json.dumps(report, indent=1))
        return

    if args.output is None:
        parser.error("--output is required (unless --compare_tile)")
    tiles = sorted(args.laz or [])
    if args.laz_dir is not None:
        tiles += sorted(p for p in args.laz_dir.iterdir()
                        if p.suffix.lower() in (".laz", ".las"))
    if not tiles:
        parser.error("no LAZ tiles given")

    if args.bounds_xy is not None:
        bounds = tuple(args.bounds_xy)
    elif args.bounds_latlon is not None:
        bounds = _bounds_from_latlon(*args.bounds_latlon, args.dst_crs)
    else:
        parser.error("pass --bounds_latlon or --bounds_xy")
    print(f"grid bounds {bounds} @ {args.resolution_m} m in {args.dst_crs}")

    accumulator = lidar_dsm.RasterAccumulator.create(
        bounds_xy=bounds, resolution_m=args.resolution_m, crs=args.dst_crs)
    started = time.monotonic()
    tile_stats = []
    for i, tile in enumerate(tiles):
        stats = lidar_dsm.stream_tile(
            tile, accumulator, keep_classes=tuple(args.keep_classes))
        tile_stats.append(stats)
        if (i + 1) % 20 == 0 or i + 1 == len(tiles):
            print(f"{i + 1}/{len(tiles)} tiles, "
                  f"{sum(s['points_in_bounds'] for s in tile_stats) / 1e9:.2f}"
                  f"G points used, {time.monotonic() - started:.0f}s",
                  flush=True)

    dem = (terrain.HeightField.load(args.dem_surface)
           if args.dem_surface is not None else None)
    field, provenance = lidar_dsm.compose_surface(
        accumulator, fill_passes=args.fill_passes, dem_fallback=dem)

    hashes = {}
    if args.checksums is not None:
        for line in args.checksums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            hashes[Path(name.strip()).name] = digest
    sources = [{"path": str(p),
                "sha256": hashes.get(p.name, "unhashed")} for p in tiles]

    hist = {int(code): int((provenance == code).sum())
            for code in np.unique(provenance)}
    extra = {
        "schema": "dem_baseline_surface/v1",
        "surface_kind": args.surface_kind,
        "recipe": ("classified LiDAR per-cell maximum over classes "
                   f"{sorted(args.keep_classes)}; "
                   f"{args.fill_passes}-pass >=5-of-8-neighbor median hole "
                   "fill; provider bare-earth DEM fallback elsewhere "
                   "(hydro-flattened water = declared water surface); "
                   "residual no-data filled with min elevation and recorded "
                   "in nodata_mask"),
        "keep_classes": sorted(args.keep_classes),
        "fill_passes": args.fill_passes,
        "statistic": "max_per_cell",
        "provenance_histogram": {
            "empty": hist.get(lidar_dsm.PROV_EMPTY, 0),
            "lidar": hist.get(lidar_dsm.PROV_LIDAR, 0),
            "hole_filled": hist.get(lidar_dsm.PROV_HOLE_FILLED, 0),
            "dem_fallback": hist.get(lidar_dsm.PROV_DEM_FALLBACK, 0),
        },
        "points_total": sum(s["points_total"] for s in tile_stats),
        "points_kept_class": sum(s["points_kept_class"] for s in tile_stats),
        "points_in_bounds": sum(s["points_in_bounds"] for s in tile_stats),
        "dem_fallback_surface": (str(args.dem_surface)
                                 if args.dem_surface else None),
        "sources": sources,
        "bounds_request": list(bounds),
    }
    if args.note:
        extra["notes"] = args.note
    field.save(args.output, extra_manifest=extra)
    np.savez_compressed(args.output.parent / "provenance.npz",
                        provenance=provenance,
                        counts=accumulator.counts)
    print(f"wrote {args.output}.npz / .json and provenance.npz; "
          f"provenance {extra['provenance_histogram']}")


if __name__ == "__main__":
    main()
