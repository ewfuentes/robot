"""Fetch an ESRI World Imagery underlay for a localization run, in two levels.

The map view's vector basemap is offline and self-contained by design, which is
right for a forensic record but tells you very little about what a place actually
looks like. This fetches raster imagery for one run and writes it in the layout
`viewer.py --satellite` consumes.

**Two levels, because one cannot serve both jobs.** A catalog extent is 23-31 km
and the trajectory inside it is 0.4-18 km, so:

  wide   the whole catalog extent at a coarse zoom. Context: which side of the
         harbour, where the built-up area stops. Blurry if you zoom in, and that
         is fine, because
  fine   a high-zoom mosaic over the trajectory plus a margin, drawn on top. This
         is the one you are looking at when you zoom to a few hundred metres.

**Licensing.** ESRI World Imagery is licensed, not redistributable. The imagery
is embedded in the viewer page, so a page built with it is for internal use and
must not be shipped with a data release. `satellite.json` records the source and
release so that provenance travels with the file rather than living in someone's
memory. The result is transactionally published beside the immutable run as
`<run>.satellite`; nothing is written into the run or the repository.

Wayback rather than the current World Imagery endpoint: releases are dated, so
imagery can be pinned near a dataset's own capture date. `--date` is REQUIRED
for exactly that reason: construction moves in a harbour, and an underlay from
three years later is a misleading backdrop for a matcher argument — the date is
an assumption-carrying value, so it has no default.

**One known distortion.** Tiles are web mercator; the viewer's frame is
equirectangular with a fixed scale at the anchor. Placing a mosaic by its corner
coordinates therefore stretches it by up to ~0.3% over 25 km of latitude — tens
of metres at the edge of a wide layer, under a metre on a fine one. Acceptable
for a backdrop, not a survey.

    bazel run //experimental/overhead_matching/swag/farfield/localization:satellite_underlay -- \\
        --run_dir <run> --date 2026-07 --dry_run
"""

import argparse
import concurrent.futures
import io
import math
import time
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
    provenance,
)
from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    side_outputs,
)

TILE_PX = 256
WAYBACK_TILE_URL = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
                    "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
                    "{release}/{z}/{y}/{x}")
WAYBACK_CONFIG_URL = ("https://s3-us-west-2.amazonaws.com/"
                      "config.maptiles.arcgis.com/waybackconfig.json")
# A guard, not a preference: a 25 km box at zoom 18 is ~40,000 tiles, which is
# both a long wait and an unreasonable thing to do to someone's tile service.
DEFAULT_MAX_TILES = 1200


def tile_of(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    """Web-mercator tile indices containing a point."""
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    lat = math.radians(max(-85.05, min(85.05, lat_deg)))
    y = int((1.0 - math.asinh(math.tan(lat)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_nw_corner(x: int, y: int, zoom: int) -> tuple[float, float]:
    """(lat, lon) of a tile's north-west corner."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def tile_span(lat_min, lat_max, lon_min, lon_max, zoom):
    """(x0, y0, x1, y1) inclusive tile range covering a lat/lon box."""
    x0, y0 = tile_of(lat_max, lon_min, zoom)      # NW
    x1, y1 = tile_of(lat_min, lon_max, zoom)      # SE
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def find_release(date: str, session) -> tuple[int, str]:
    """(release number, label) for the Wayback release nearest `date`.

    `date` is YYYY-MM or YYYY-MM-DD. Matched by the DATE in each release's
    title, never by release number: Wayback numbers are not chronological
    (picking the largest number once returned a 2023-08-31 release as if it
    were current — a silently three-year-old backdrop).
    """
    config = session.get(WAYBACK_CONFIG_URL, timeout=30).json()
    releases = []
    for entry in config.values():
        title = entry.get("itemTitle", "")
        number = None
        for part in str(entry.get("itemURL", "")).split("/"):
            if part.isdigit():
                number = int(part)
        if number is None:
            continue
        releases.append((number, title))
    if not releases:
        raise SystemExit("could not parse any Wayback releases from the config")

    def stamp(title):
        """YYYYMM out of a title like "World Imagery (Wayback 2026-02-13)"."""
        digits = "".join(c for c in title if c.isdigit())
        tail = digits[-8:] if len(digits) >= 8 else digits
        return int(tail[:6]) if len(tail) >= 6 else 0

    want = int(date.replace("-", "")[:6])
    best = min(releases, key=lambda r: abs(stamp(r[1]) - want))
    return best[0], best[1]


def fetch_mosaic(x0, y0, x1, y1, zoom, release, session, workers=8):
    """Stitched RGB image for a tile rectangle, plus how many tiles failed."""
    from PIL import Image
    width, height = (x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX
    canvas = Image.new("RGB", (width, height), (24, 28, 34))
    failures = 0

    def one(tx, ty):
        url = WAYBACK_TILE_URL.format(release=release, z=zoom, y=ty, x=tx)
        for attempt in range(3):
            try:
                response = session.get(url, timeout=30)
                if response.status_code == 200 and response.content:
                    return tx, ty, response.content
            except Exception:  # noqa: BLE001 - a lost tile is a grey square
                pass
            time.sleep(0.4 * (attempt + 1))
        return tx, ty, None

    jobs = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for tx, ty, blob in pool.map(lambda a: one(*a), jobs):
            if blob is None:
                failures += 1
                continue
            try:
                tile = Image.open(io.BytesIO(blob)).convert("RGB")
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            canvas.paste(tile, ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX))
    return canvas, failures


def enu_bounds_of_tiles(x0, y0, x1, y1, zoom, frame):
    """ENU box covered by a tile rectangle, in the run's own frame."""
    lat_n, lon_w = tile_nw_corner(x0, y0, zoom)
    lat_s, lon_e = tile_nw_corner(x1 + 1, y1 + 1, zoom)
    east, north = frame.enu_from_latlon([lat_n, lat_s], [lon_w, lon_e])
    return (float(min(east)), float(max(east)),
            float(min(north)), float(max(north)))


def layer_plan(name, lat_min, lat_max, lon_min, lon_max, zoom):
    x0, y0, x1, y1 = tile_span(lat_min, lat_max, lon_min, lon_max, zoom)
    n = (x1 - x0 + 1) * (y1 - y0 + 1)
    return {"name": name, "zoom": zoom, "tiles": (x0, y0, x1, y1),
            "n_tiles": n,
            "px": ((x1 - x0 + 1) * TILE_PX, (y1 - y0 + 1) * TILE_PX)}


def fit_zoom(name, lat_min, lat_max, lon_min, lon_max, max_zoom, budget):
    """The highest zoom <= max_zoom whose tile count fits `budget`.

    A fixed zoom cannot serve this corpus: the same z18 that costs 99 tiles over
    mount_washington leg1's 0.4 km track costs 11,730 over boston leg3's 18 km
    one. Asking for the sharpest imagery that fits a tile budget is the request
    someone actually has, so the tool answers that instead of making them bisect
    a zoom by hand.
    """
    for zoom in range(max_zoom, 8, -1):
        plan = layer_plan(name, lat_min, lat_max, lon_min, lon_max, zoom)
        if plan["n_tiles"] <= budget:
            plan["capped"] = zoom < max_zoom
            return plan
    plan = layer_plan(name, lat_min, lat_max, lon_min, lon_max, 9)
    plan["capped"] = True
    return plan


def trajectory_enu(data) -> tuple[np.ndarray, np.ndarray]:
    """Trajectory extent, preferring truth and otherwise the estimated path."""
    if data.truth:
        east = np.array([point.east_m for point in data.truth], dtype=np.float64)
        north = np.array([point.north_m for point in data.truth], dtype=np.float64)
    elif data.health:
        east = np.array(
            [record.mean_east_m for record in data.health], dtype=np.float64)
        north = np.array(
            [record.mean_north_m for record in data.health], dtype=np.float64)
    else:
        raise ValueError(
            "satellite underlay needs truth or estimated health positions")
    return east, north


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse
                                     .RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="default: sibling <run_dir>.satellite")
    parser.add_argument("--date", required=True,
                        help="YYYY-MM: pin imagery near the dataset's capture "
                             "date. Required — imagery from years later is a "
                             "misleading backdrop, so there is no default.")
    parser.add_argument("--wide_zoom", type=int, default=14,
                        help="upper bound; lowered to fit the budget")
    parser.add_argument("--fine_zoom", type=int, default=18,
                        help="upper bound; lowered to fit the budget")
    parser.add_argument("--fine_margin_m", type=float, default=400.0)
    parser.add_argument("--max_tiles", type=int, default=DEFAULT_MAX_TILES)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--dry_run", action="store_true",
                        help="report tile counts and fetch nothing")
    args = parser.parse_args()

    data = run_io.read_run(args.run_dir)
    manifest = data.manifest
    frame = geo.RegionFrame(manifest.anchor_lat_deg, manifest.anchor_lon_deg)

    # wide: the catalog's own extent. fine: the trajectory plus a margin.
    lat = np.array([lm.lat_deg for lm in manifest.landmarks], dtype=np.float64)
    lon = np.array([lm.lon_deg for lm in manifest.landmarks], dtype=np.float64)
    east, north = trajectory_enu(data)
    t_lat, t_lon = frame.latlon_from_enu(
        np.array([east.min() - args.fine_margin_m,
                  east.max() + args.fine_margin_m]),
        np.array([north.min() - args.fine_margin_m,
                  north.max() + args.fine_margin_m]))

    # Wide gets a third of the budget, fine the rest: context is cheap and the
    # sharp layer is the one worth spending tiles on.
    wide = fit_zoom("wide", lat.min(), lat.max(), lon.min(), lon.max(),
                    args.wide_zoom, max(16, args.max_tiles // 3))
    fine = fit_zoom("fine", float(np.min(t_lat)), float(np.max(t_lat)),
                    float(np.min(t_lon)), float(np.max(t_lon)),
                    args.fine_zoom, max(16, args.max_tiles - wide["n_tiles"]))
    plans = [wide, fine]
    total = sum(p["n_tiles"] for p in plans)
    for plan in plans:
        metres = 156543.03392 * math.cos(math.radians(
            manifest.anchor_lat_deg)) / (2 ** plan["zoom"])
        print(f"  {plan['name']:5s} z{plan['zoom']:<3d} {plan['n_tiles']:5d} tiles "
              f"-> {plan['px'][0]}x{plan['px'][1]} px at {metres:.2f} m/px"
              + ("  (zoom capped to fit the budget)" if plan.get("capped")
                 else ""))
    print(f"  total {total} tiles of a {args.max_tiles} budget")
    if total > args.max_tiles:
        raise SystemExit(
            f"{total} tiles still exceeds --max_tiles {args.max_tiles} even at "
            f"the lowest zoom tried; raise the cap deliberately.")
    if args.dry_run:
        print("  dry run: nothing fetched")
        return

    import requests
    session = requests.Session()
    session.headers["User-Agent"] = "farfield-crossview/viewer underlay"
    release, label = find_release(args.date, session)
    print(f"  ESRI Wayback release {release} ({label})")

    with side_outputs.publish_directory(
            args.run_dir, output_dir=args.output_dir,
            suffix=".satellite") as output:
        out_dir = output.staging_dir
        layers = []
        for plan in plans:
            x0, y0, x1, y1 = plan["tiles"]
            image, failures = fetch_mosaic(
                x0, y0, x1, y1, plan["zoom"], release, session)
            name = f"{plan['name']}.jpg"
            image.save(
                out_dir / name, quality=args.jpeg_quality, optimize=True)
            e0, e1, n0, n1 = enu_bounds_of_tiles(
                x0, y0, x1, y1, plan["zoom"], frame)
            layers.append({"image": name, "zoom": plan["zoom"],
                           "east_min": e0, "east_max": e1,
                           "north_min": n0, "north_max": n1,
                           "n_tiles": plan["n_tiles"], "n_failed": failures,
                           "bytes": (out_dir / name).stat().st_size})
            print(
                f"  {plan['name']}: "
                f"{(out_dir / name).stat().st_size / 1e6:.1f} MB"
                + (f", {failures} tile(s) missing" if failures else ""))

        artifact.atomic_write_json(out_dir / "satellite.json", {
            "source": f"ESRI World Imagery Wayback release {release} ({label})",
            "licence": "ESRI World Imagery is licensed and NOT redistributable; "
                       "a viewer page built with it is for internal use only",
            "anchor_lat_deg": manifest.anchor_lat_deg,
            "anchor_lon_deg": manifest.anchor_lon_deg,
            "projection_note":
                "tiles are web mercator, bounds are equirectangular in the "
                "run's frame; up to ~0.3% stretch over 25 km",
            "layers": layers,
        })
        provenance.write(
            out_dir,
            generator="//experimental/overhead_matching/swag/farfield/"
                      "localization:satellite_underlay",
            inputs={"run_dir": Path(args.run_dir).resolve()},
            config={"date": args.date, "release": release,
                    "release_label": label,
                    "wide_zoom": wide["zoom"], "fine_zoom": fine["zoom"],
                    "fine_margin_m": args.fine_margin_m,
                    "max_tiles": args.max_tiles,
                    "jpeg_quality": args.jpeg_quality})
    print(f"  wrote {output.destination}/satellite.json")


if __name__ == "__main__":
    main()
