"""Fetch an ESRI World Imagery underlay for a localization run, in two levels.

The map view's vector basemap is offline and self-contained by design, which is
right for a forensic record but tells you very little about what a place actually
looks like. This fetches raster imagery for one run and writes it in the layout
`viewer.py --satellite` consumes.

**Two levels, because one cannot serve both jobs.** Global context and detailed
trajectory inspection require different raster scales:

  wide   exactly the bounded localization prior at a coarse zoom. Context: which
         hypotheses the filter could initially occupy. Blurry if you zoom in,
         and that is fine, because
  fine   a high-zoom mosaic over the trajectory plus a margin, clipped to the
         prior and drawn on top. This is the one you are looking at when you
         zoom to a few hundred metres.

Whole web-mercator tiles are fetched, but each published raster is resampled to
its exact requested ENU box. Tile alignment therefore affects download cost,
not the imagery extent exposed in the viewer.

**Licensing.** ESRI World Imagery is licensed, not redistributable. The imagery
is embedded in the viewer page, so a page built with it is for internal use and
must not be shipped with a data release. `satellite.json` records the source and
release so that provenance travels with the file rather than living in someone's
memory. The result is transactionally published beside the immutable run as
`<run>.satellite`; nothing is written into the run or the repository.
Viewer creation wraps this producer with dataset-aware discovery: compact new
underlays use that sibling default, while larger mosaics are published once
under `<farfield_root>/reviews/satellite/<dataset>/` and reused by compatible
runs.

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
    structs,
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
GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "localization:satellite_underlay")
LICENCE = ("ESRI World Imagery is licensed and NOT redistributable; a viewer "
           "page built with it is for internal use only")


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


def world_pixel(lat_deg: float, lon_deg: float, zoom: int) -> tuple[float, float]:
    """Fractional global web-mercator pixel coordinates at ``zoom``."""
    size = TILE_PX * (2 ** zoom)
    x = (lon_deg + 180.0) / 360.0 * size
    lat = math.radians(max(-85.05, min(85.05, lat_deg)))
    y = (1.0 - math.asinh(math.tan(lat)) / math.pi) / 2.0 * size
    return x, y


def tile_span(lat_min, lat_max, lon_min, lon_max, zoom):
    """(x0, y0, x1, y1) inclusive tile range covering a lat/lon box."""
    x0, y0 = tile_of(lat_max, lon_min, zoom)      # NW
    x1, y1 = tile_of(lat_min, lon_max, zoom)      # SE
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def find_release(date: str, session) -> tuple[int, str]:
    """(release number, label) for the Wayback release nearest `date`.

    `date` is YYYY-MM or YYYY-MM-DD. Match the date in each release title;
    Wayback release numbers are not chronological.
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


def _pixel_crop_box(plan: dict) -> tuple[float, float, float, float]:
    """Fractional mosaic pixel box for a plan's requested lat/lon bounds."""
    lat_min, lat_max, lon_min, lon_max = plan["latlon_bounds"]
    x0, y0, _, _ = plan["tiles"]
    left, top = world_pixel(lat_max, lon_min, plan["zoom"])
    right, bottom = world_pixel(lat_min, lon_max, plan["zoom"])
    left -= x0 * TILE_PX
    right -= x0 * TILE_PX
    top -= y0 * TILE_PX
    bottom -= y0 * TILE_PX
    width, height = plan["px"]
    tolerance = 1e-5
    if (left < -tolerance or top < -tolerance
            or right > width + tolerance or bottom > height + tolerance
            or not left < right or not top < bottom):
        raise ValueError("requested imagery crop is outside its fetched tiles")
    return (max(0.0, left), max(0.0, top),
            min(float(width), right), min(float(height), bottom))


def crop_mosaic(image, plan: dict):
    """Resample fetched tiles onto exactly the plan's requested footprint."""
    from PIL import Image
    if tuple(image.size) != tuple(plan["px"]):
        raise ValueError(
            f"fetched mosaic is {image.size}, expected {plan['px']}")
    return image.transform(
        tuple(plan["output_px"]), Image.Transform.EXTENT,
        _pixel_crop_box(plan), resample=Image.Resampling.BICUBIC)


def fit_zoom(name, lat_min, lat_max, lon_min, lon_max, max_zoom, budget):
    """Return the highest zoom no greater than ``max_zoom`` within ``budget``."""
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


def prior_enu(manifest) -> tuple[float, float, float, float]:
    """The exact bounded position prior recorded by a localization run."""
    init = manifest.filter_config.init
    if not isinstance(init, structs.UniformBoxInit):
        raise ValueError(
            "satellite underlay requires a bounded UniformBoxInit prior")
    bounds = (float(init.east_min_m), float(init.east_max_m),
              float(init.north_min_m), float(init.north_max_m))
    if (not all(math.isfinite(value) for value in bounds)
            or not bounds[0] < bounds[1] or not bounds[2] < bounds[3]):
        raise ValueError("satellite underlay prior bounds are invalid")
    return bounds


def _latlon_bounds(bounds: tuple[float, float, float, float], frame
                   ) -> tuple[float, float, float, float]:
    e0, e1, n0, n1 = bounds
    lat, lon = frame.latlon_from_enu(
        np.asarray([e0, e0, e1, e1], dtype=np.float64),
        np.asarray([n0, n1, n0, n1], dtype=np.float64))
    return (float(np.min(lat)), float(np.max(lat)),
            float(np.min(lon)), float(np.max(lon)))


def _bounded_plan(name: str, bounds: tuple[float, float, float, float],
                  frame, max_zoom: int, budget: int) -> dict:
    latlon = _latlon_bounds(bounds, frame)
    plan = fit_zoom(name, *latlon, max_zoom, budget)
    plan["bounds_enu"] = bounds
    plan["latlon_bounds"] = latlon
    left, top, right, bottom = _pixel_crop_box(plan)
    plan["output_px"] = (
        max(1, int(round(right - left))),
        max(1, int(round(bottom - top))))
    return plan


def _intersection(first: tuple[float, float, float, float],
                  second: tuple[float, float, float, float]
                  ) -> tuple[float, float, float, float] | None:
    result = (max(first[0], second[0]), min(first[1], second[1]),
              max(first[2], second[2]), min(first[3], second[3]))
    return result if result[0] < result[1] and result[2] < result[3] else None


def plan_underlay(data, *, wide_zoom: int = 14, fine_zoom: int = 17,
                  fine_margin_m: float = 400.0,
                  max_tiles: int = DEFAULT_MAX_TILES) -> list[dict]:
    """Plan the two mosaics without touching the network.

    Keeping this separate from fetching lets viewer creation decide whether a
    compact plan belongs beside one run or a larger plan belongs in the shared
    per-dataset review cache before any tiles are downloaded.
    """
    manifest = data.manifest
    frame = geo.RegionFrame(manifest.anchor_lat_deg, manifest.anchor_lon_deg)

    # Wide is exactly the finite initialization support. Fine is useful only
    # where the trajectory's detailed context intersects that same support.
    prior = prior_enu(manifest)
    east, north = trajectory_enu(data)
    trajectory = (float(east.min() - fine_margin_m),
                  float(east.max() + fine_margin_m),
                  float(north.min() - fine_margin_m),
                  float(north.max() + fine_margin_m))
    fine_bounds = _intersection(prior, trajectory)

    # Wide gets a third of the budget, fine the rest: context is cheap and the
    # sharp layer is the one worth spending tiles on.
    wide = _bounded_plan(
        "wide", prior, frame, wide_zoom, max(16, max_tiles // 3))
    plans = [wide]
    if fine_bounds is not None:
        plans.append(_bounded_plan(
            "fine", fine_bounds, frame, fine_zoom,
            max(16, max_tiles - wide["n_tiles"])))
    total = sum(plan["n_tiles"] for plan in plans)
    if total > max_tiles:
        raise ValueError(
            f"{total} tiles still exceeds max_tiles={max_tiles} even at the "
            "lowest zoom tried; raise the cap deliberately")
    return plans


def describe_plan(plans: list[dict], *, anchor_lat_deg: float,
                  max_tiles: int) -> None:
    """Print the network and raster cost of a planned underlay."""
    for plan in plans:
        metres = 156543.03392 * math.cos(math.radians(
            anchor_lat_deg)) / (2 ** plan["zoom"])
        output_px = plan.get("output_px", plan["px"])
        x0, y0, x1, y1 = plan["tiles"]
        print(f"  {plan['name']:5s} z{plan['zoom']:<3d} "
              f"{plan['n_tiles']:5d} tiles -> "
              f"{output_px[0]}x{output_px[1]} cropped px at {metres:.2f} m/px"
              f"; x={x0}..{x1}, y={y0}..{y1}"
              + ("  (zoom capped to fit the budget)" if plan.get("capped")
                 else ""))
    print(f"  total {sum(plan['n_tiles'] for plan in plans)} tiles of a "
          f"{max_tiles} budget")


def generate_underlay(run_dir: Path, *, date: str,
                      output_dir: Path | None = None,
                      wide_zoom: int = 14, fine_zoom: int = 17,
                      fine_margin_m: float = 400.0,
                      max_tiles: int = DEFAULT_MAX_TILES,
                      jpeg_quality: int = 80, data=None,
                      plans: list[dict] | None = None,
                      session=None) -> Path:
    """Fetch and transactionally publish one planned underlay.

    ``data`` and ``plans`` are accepted so the automatic viewer path can read
    and plan once while it chooses a destination. Tests may also supply a
    session without changing the global requests client.
    """
    run_dir = Path(run_dir)
    data = run_io.read_run(run_dir) if data is None else data
    manifest = data.manifest
    plans = (plan_underlay(
        data, wide_zoom=wide_zoom, fine_zoom=fine_zoom,
        fine_margin_m=fine_margin_m, max_tiles=max_tiles)
             if plans is None else plans)
    total = sum(plan["n_tiles"] for plan in plans)
    if total > max_tiles:
        raise ValueError(
            f"planned underlay has {total} tiles, above max_tiles={max_tiles}")
    wide_plan = next(
        (plan for plan in plans if plan.get("name") == "wide"), None)
    if wide_plan is None or "bounds_enu" not in wide_plan:
        raise ValueError("satellite plan lacks a prior-bounded wide layer")
    prior_bounds = tuple(wide_plan["bounds_enu"])
    for plan in plans:
        bounds = tuple(plan.get("bounds_enu", ()))
        if (len(bounds) != 4 or bounds[0] < prior_bounds[0]
                or bounds[1] > prior_bounds[1]
                or bounds[2] < prior_bounds[2]
                or bounds[3] > prior_bounds[3]):
            raise ValueError(
                f"satellite layer {plan.get('name')!r} exceeds the prior")

    if session is None:
        import requests
        session = requests.Session()
    session.headers["User-Agent"] = "farfield-crossview/viewer underlay"
    release, label = find_release(date, session)
    print(f"  ESRI Wayback release {release} ({label})")

    with side_outputs.publish_directory(
            run_dir, output_dir=output_dir, suffix=".satellite") as output:
        out_dir = output.staging_dir
        layers = []
        for plan in plans:
            x0, y0, x1, y1 = plan["tiles"]
            image, failures = fetch_mosaic(
                x0, y0, x1, y1, plan["zoom"], release, session)
            if failures >= plan["n_tiles"]:
                raise RuntimeError(
                    f"all {plan['n_tiles']} source tiles failed for "
                    f"{plan['name']} at z{plan['zoom']}")
            image = crop_mosaic(image, plan)
            name = f"{plan['name']}.jpg"
            image.save(out_dir / name, quality=jpeg_quality, optimize=True)
            e0, e1, n0, n1 = plan["bounds_enu"]
            layers.append({"image": name, "zoom": plan["zoom"],
                           "east_min": e0, "east_max": e1,
                           "north_min": n0, "north_max": n1,
                           "width_px": image.width,
                           "height_px": image.height,
                           "tiles": list(plan["tiles"]),
                           "n_tiles": plan["n_tiles"], "n_failed": failures,
                           "bytes": (out_dir / name).stat().st_size})
            print(
                f"  {plan['name']}: "
                f"{(out_dir / name).stat().st_size / 1e6:.1f} MB"
                + (f", {failures} tile(s) missing" if failures else ""))

        artifact.atomic_write_json(out_dir / "satellite.json", {
            "schema": "farfield_satellite_underlay/v2",
            "dataset": manifest.dataset,
            "capture_date": date,
            "wide_extent_kind": "localization_prior",
            "prior_bounds": {
                "east_min": prior_bounds[0], "east_max": prior_bounds[1],
                "north_min": prior_bounds[2], "north_max": prior_bounds[3],
            },
            "source": f"ESRI World Imagery Wayback release {release} ({label})",
            "licence": LICENCE,
            "anchor_lat_deg": manifest.anchor_lat_deg,
            "anchor_lon_deg": manifest.anchor_lon_deg,
            "projection_note":
                "source tiles are web mercator and each raster is cropped to "
                "its exact equirectangular ENU display bounds; residual "
                "interior stretch can reach ~0.3% over 25 km",
            "layers": layers,
        })
        provenance.write(
            out_dir,
            generator=GENERATOR,
            inputs={"run_dir": run_dir.resolve()},
            config={"date": date, "release": release,
                    "release_label": label,
                    "wide_extent_kind": "localization_prior",
                    "prior_bounds_enu": list(prior_bounds),
                    "wide_zoom": wide_plan["zoom"],
                    "fine_zoom": next(
                        (plan["zoom"] for plan in plans
                         if plan.get("name") == "fine"), None),
                    "fine_margin_m": fine_margin_m,
                    "max_tiles": max_tiles,
                    "jpeg_quality": jpeg_quality})
    print(f"  wrote {output.destination}/satellite.json")
    return output.destination


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
    parser.add_argument("--fine_zoom", type=int, default=17,
                        help="upper bound; lowered to fit the budget. Wayback "
                             "does not guarantee z18 coverage (default: 17)")
    parser.add_argument("--fine_margin_m", type=float, default=400.0)
    parser.add_argument("--max_tiles", type=int, default=DEFAULT_MAX_TILES)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--dry_run", action="store_true",
                        help="report tile counts and fetch nothing")
    args = parser.parse_args()

    data = run_io.read_run(args.run_dir)
    try:
        plans = plan_underlay(
            data, wide_zoom=args.wide_zoom, fine_zoom=args.fine_zoom,
            fine_margin_m=args.fine_margin_m, max_tiles=args.max_tiles)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    describe_plan(plans, anchor_lat_deg=data.manifest.anchor_lat_deg,
                  max_tiles=args.max_tiles)
    if args.dry_run:
        print("  dry run: nothing fetched")
        return
    generate_underlay(
        args.run_dir, date=args.date, output_dir=args.output_dir,
        wide_zoom=args.wide_zoom, fine_zoom=args.fine_zoom,
        fine_margin_m=args.fine_margin_m, max_tiles=args.max_tiles,
        jpeg_quality=args.jpeg_quality, data=data, plans=plans)


if __name__ == "__main__":
    main()
