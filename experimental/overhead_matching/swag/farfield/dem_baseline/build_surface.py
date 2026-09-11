"""Build a region evaluation surface (HeightField) from raw elevation tiles.

The DEM recipe from the plan (section 7.2): take the provider's bare-earth
rasters, reproject ONCE into the experiment CRS, fill provider no-data by a
fixed logged rule (here: recorded mask + minimum-elevation fill), and never
add structures from test imagery. The output manifest records sources,
checksums, CRS, and bounds so renders are reproducible from it.

Example (Mt. Washington 1/3 arc-second wide surface):

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:build_surface -- \
        --tiles /data/farfield_matching/raw_material/usgs_3dep_mount_washington/13_arcsec/USGS_13_*.tif \
        --dst_crs EPSG:26919 --resolution_m 10 \
        --center_latlon 44.2705 -71.3032 --halfwidth_m 40000 \
        --output /data/farfield_matching/artifacts/dem_surfaces/mount_washington/v1/surface
"""

import argparse
import hashlib
import time
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dem_baseline import terrain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", type=Path, nargs="+", required=True)
    parser.add_argument("--dst_crs", required=True,
                        help="Experiment CRS, e.g. EPSG:26919 (NAD83 UTM 19N)")
    parser.add_argument("--resolution_m", type=float, required=True)
    parser.add_argument("--center_latlon", type=float, nargs=2, default=None,
                        metavar=("LAT", "LON"))
    parser.add_argument("--halfwidth_m", type=float, default=None)
    parser.add_argument("--bounds_xy", type=float, nargs=4, default=None,
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--resampling", default="bilinear")
    parser.add_argument("--output", type=Path, required=True,
                        help="Base path (no suffix); writes .npz and .json")
    args = parser.parse_args()

    if args.bounds_xy is not None:
        bounds = tuple(args.bounds_xy)
    elif args.center_latlon is not None and args.halfwidth_m is not None:
        cx, cy = terrain.utm_from_latlon(args.center_latlon[0],
                                         args.center_latlon[1], args.dst_crs)
        bounds = (cx - args.halfwidth_m, cy - args.halfwidth_m,
                  cx + args.halfwidth_m, cy + args.halfwidth_m)
    else:
        parser.error("pass --bounds_xy, or --center_latlon with --halfwidth_m")

    started = time.monotonic()
    hf = terrain.build_height_field(
        sorted(args.tiles), dst_crs=args.dst_crs,
        resolution_m=args.resolution_m, bounds_xy=bounds,
        resampling=args.resampling)
    print(f"built {hf.elevation.shape} field in "
          f"{time.monotonic() - started:.0f}s; "
          f"elevation [{hf.elevation.min():.0f}, {hf.elevation.max():.0f}] m; "
          f"{hf.nodata_mask.sum()} no-data cells")

    sources = [{
        "path": str(p),
        "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
    } for p in sorted(args.tiles)]
    hf.save(args.output, extra_manifest={
        "schema": "dem_baseline_surface/v1",
        "surface_kind": "bare_earth_dem",
        "recipe": "mosaic first-valid-wins, single reprojection "
                  f"({args.resampling}), no-data filled with min elevation "
                  "and recorded in nodata_mask",
        "sources": sources,
        "bounds_request": list(bounds),
    })
    print(f"wrote {args.output}.npz / .json")


if __name__ == "__main__":
    main()
