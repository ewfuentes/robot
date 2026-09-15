"""Generate square satellite-overhead panels for the paper datasets."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import requests
from matplotlib.lines import Line2D

from experimental.overhead_matching.swag.farfield.localization import (
    satellite_underlay,
)
from experimental.overhead_matching.swag.farfield.paper.dataset_table import (
    _capture_date,
)
from experimental.overhead_matching.swag.farfield.paper.table_common import (
    DATASET_GROUPS,
    DEFAULT_FARFIELD_ROOT,
    DatasetGroup,
    read_json_object,
)

COLORS = ("#ff365e", "#00d7ff", "#ffd43b")
LINESTYLES = ("-", (0, (5, 2)), (0, (1, 2)))
START_MARKER = "o"
STOP_MARKER = "X"
ZOOMED_GROUPS = {"washington", "pohang", "charles", "franconia"}
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def square_bounds(
    bounds: tuple[float, float, float, float],
    margin: float = 0.02,
) -> tuple[float, float, float, float]:
    """Pad a lon/lat box to a square in approximate ground distance."""
    west, south, east, north = bounds
    mid_lat = (south + north) / 2.0
    width = (east - west) * math.cos(math.radians(mid_lat))
    height = north - south
    if width < height:
        pad = (height / math.cos(math.radians(mid_lat)) - (east - west)) / 2.0
        west, east = west - pad, east + pad
    else:
        pad = (width - height) / 2.0
        south, north = south - pad, north + pad
    lon_margin = (east - west) * margin
    lat_margin = (north - south) * margin
    return (
        west - lon_margin,
        south - lat_margin,
        east + lon_margin,
        north + lat_margin,
    )


def load_group(root: Path, group: DatasetGroup):
    trajectories = []
    dates = set()
    region_bounds = set()
    for sequence in group.sequences:
        dataset_dir = root / "datasets" / sequence
        metadata_path = dataset_dir / "pipeline_metadata.json"
        dates.add(_capture_date(read_json_object(metadata_path), metadata_path))
        with (dataset_dir / "frames_gps.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        try:
            trajectory = tuple(
                (float(row["longitude"]), float(row["latitude"])) for row in rows
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{dataset_dir / 'frames_gps.csv'}: invalid GPS row"
            ) from exc
        if len(trajectory) < 2 or not all(
            math.isfinite(value) for point in trajectory for value in point
        ):
            raise ValueError(
                f"{dataset_dir / 'frames_gps.csv'}: invalid trajectory"
            )
        trajectories.append(trajectory)

        manifest_path = (
            root / "artifacts" / "catalogs" / sequence
            / group.catalog_version / "manifest.json"
        )
        config = read_json_object(manifest_path).get("config")
        bounds = config.get("region_bbox_wsen") if isinstance(config, dict) else None
        if not (isinstance(bounds, list) and len(bounds) == 4):
            raise ValueError(f"{manifest_path}: missing config.region_bbox_wsen")
        region_bounds.add(tuple(float(value) for value in bounds))
    if len(dates) != 1 or len(region_bounds) != 1:
        raise ValueError(f"{group.display_name}: grouped sequences disagree")
    return trajectories, region_bounds.pop(), dates.pop()


def _tile_plan(bounds, max_zoom, max_tiles):
    west, south, east, north = bounds
    plan = satellite_underlay.fit_zoom(
        "paper", south, north, west, east, max_zoom, max_tiles)
    plan["latlon_bounds"] = (south, north, west, east)
    left, top, right, bottom = satellite_underlay._pixel_crop_box(plan)
    plan["output_px"] = (
        max(1, round(right - left)), max(1, round(bottom - top)))
    return plan


def fetch_satellite(bounds, date, *, max_zoom=13, max_tiles=64):
    plan = _tile_plan(bounds, max_zoom, max_tiles)
    session = requests.Session()
    session.headers["User-Agent"] = "farfield-crossview/paper overhead figures"
    release, _ = satellite_underlay.find_release(date, session)
    image, failures = satellite_underlay.fetch_mosaic(
        *plan["tiles"], plan["zoom"], release, session)
    if failures:
        raise RuntimeError(f"{failures}/{plan['n_tiles']} ESRI tiles failed")
    return satellite_underlay.crop_mosaic(image, plan), plan


def fetch_osm(bounds, *, max_zoom=13, max_tiles=64):
    plan = _tile_plan(bounds, max_zoom, max_tiles)
    session = requests.Session()
    session.headers["User-Agent"] = "farfield-crossview-paper/1.0"
    image, failures = satellite_underlay.fetch_mosaic(
        *plan["tiles"], plan["zoom"], None, session,
        url_template=OSM_TILE_URL)
    if failures:
        raise RuntimeError(f"{failures}/{plan['n_tiles']} OSM tiles failed")
    return satellite_underlay.crop_mosaic(image, plan), plan


def trajectory_bounds(trajectories):
    points = [point for trajectory in trajectories for point in trajectory]
    longitude, latitude = zip(*points)
    return min(longitude), min(latitude), max(longitude), max(latitude)


def _draw_trajectories(ax, trajectories, linewidth):
    for index, trajectory in enumerate(trajectories):
        (line,) = ax.plot(
            *zip(*trajectory), color=COLORS[index % len(COLORS)],
            lw=linewidth, ls=LINESTYLES[index % len(LINESTYLES)],
            solid_capstyle="round", dash_capstyle="round",
            label=f"Leg {index + 1}", zorder=4)
        line.set_path_effects(
            [path_effects.Stroke(linewidth=linewidth + 2.0, foreground="white"),
             path_effects.Normal()]
        )
        size = (linewidth * 2.8) ** 2
        ax.scatter(*trajectory[0], marker=START_MARKER, s=size,
                   facecolor="white", edgecolor=COLORS[index % len(COLORS)],
                   linewidth=1.5, zorder=5)
        ax.scatter(*trajectory[-1], marker=STOP_MARKER, s=size,
                   facecolor=COLORS[index % len(COLORS)], edgecolor="white",
                   linewidth=1.5, zorder=5)


def draw_panel(trajectories, region_bounds, image, output_path, source,
               inset=None):
    panel_bounds = square_bounds(region_bounds)
    west, south, east, north = panel_bounds
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, extent=(west, east, south, north), origin="upper")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)))

    box_x = (region_bounds[0], region_bounds[2], region_bounds[2],
             region_bounds[0], region_bounds[0])
    box_y = (region_bounds[1], region_bounds[1], region_bounds[3],
             region_bounds[3], region_bounds[1])
    (boundary,) = ax.plot(
        box_x, box_y, color="#ffe066", lw=1.6, ls="--", zorder=3
    )
    boundary.set_path_effects(
        [path_effects.Stroke(linewidth=3.0, foreground="black"),
         path_effects.Normal()]
    )
    _draw_trajectories(ax, trajectories, 3.0)
    handles = [
        Line2D([], [], color=COLORS[index], ls=LINESTYLES[index], lw=3,
               label=f"Leg {index + 1}")
        for index in range(len(trajectories))
    ] + [
        Line2D([], [], color="none", marker=START_MARKER, markersize=6,
               markerfacecolor="white", markeredgecolor="black", label="Start"),
        Line2D([], [], color="none", marker=STOP_MARKER, markersize=6,
               markerfacecolor="black", markeredgecolor="white", label="Stop"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.78,
              fontsize=7, handlelength=3.0, borderpad=0.35)
    if inset is not None:
        inset_image, inset_bounds = inset
        inset_ax = ax.inset_axes((0.52, 0.06, 0.42, 0.42))
        inset_ax.imshow(
            inset_image,
            extent=(inset_bounds[0], inset_bounds[2],
                    inset_bounds[1], inset_bounds[3]),
            origin="upper",
        )
        _draw_trajectories(inset_ax, trajectories, 2.0)
        inset_ax.set_xlim(inset_bounds[0], inset_bounds[2])
        inset_ax.set_ylim(inset_bounds[1], inset_bounds[3])
        inset_ax.set_aspect(
            1.0 / math.cos(math.radians(
                (inset_bounds[1] + inset_bounds[3]) / 2.0)))
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(2.0)
        ax.indicate_inset_zoom(
            inset_ax, edgecolor="white", linewidth=1.5, alpha=1.0, zorder=3.5)
    ax.text(0.01, 0.008, source, transform=ax.transAxes, color="white",
            fontsize=5, ha="left", va="bottom",
            path_effects=[path_effects.Stroke(linewidth=1.5, foreground="black"),
                          path_effects.Normal()])
    ax.set_axis_off()
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farfield_root", type=Path, default=DEFAULT_FARFIELD_ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_zoom", type=int, default=13)
    parser.add_argument("--max_tiles", type=int, default=64)
    parser.add_argument("--basemap", choices=("satellite", "osm", "both"),
                        default="satellite")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for group in DATASET_GROUPS:
        trajectories, region_bounds, date = load_group(args.farfield_root, group)
        panel_bounds = square_bounds(region_bounds)
        basemaps = (
            ("satellite", "osm") if args.basemap == "both"
            else (args.basemap,)
        )
        for basemap in basemaps:
            fetch = fetch_satellite if basemap == "satellite" else fetch_osm
            fetch_args = (
                (panel_bounds, date) if basemap == "satellite"
                else (panel_bounds,)
            )
            image, plan = fetch(
                *fetch_args, max_zoom=args.max_zoom, max_tiles=args.max_tiles)
            inset = None
            if group.key in ZOOMED_GROUPS:
                inset_bounds = square_bounds(
                    trajectory_bounds(trajectories), margin=0.12)
                inset_args = ((inset_bounds, date) if basemap == "satellite"
                              else (inset_bounds,))
                inset_image, _ = fetch(
                    *inset_args, max_zoom=16, max_tiles=args.max_tiles)
                inset = inset_image, inset_bounds
            suffix = "" if basemap == "satellite" else "_osm"
            output = args.output_dir / f"{group.key}{suffix}.png"
            source = ("Esri World Imagery" if basemap == "satellite"
                      else "© OpenStreetMap contributors")
            draw_panel(
                trajectories, region_bounds, image, output, source, inset)
            print(f"{output}: z{plan['zoom']}, {plan['n_tiles']} tiles")


if __name__ == "__main__":
    main()
