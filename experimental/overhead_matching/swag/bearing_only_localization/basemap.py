"""Offline vector basemap in region ENU, from a landmark feather.

§7.4 asks for the map view to sit on OSM/ENC basemap tiles. Tiles are the wrong
dependency for the artifact half of this viewer: a page that needs a tile host
to render is not a frozen forensic record, and it cannot be shared as a file.
The geometry is already on disk, in the same feather the catalog was built from,
so the offline path draws it directly — land and water bodies, the pier and
breakwater and bridge lines that define a harbour's edge, and building
footprints, all projected into the run's own ENU frame so no reprojection
happens in the browser.

What comes out is a small set of layers of simplified paths. The simplification
is in metres and deliberately coarse: this is a backdrop for reading a particle
cloud against, not a chart to navigate by, and every vertex is bytes in a page
that has more valuable things to carry.

The landmark feather is class-filtered upstream (it exists to hold *landmarks*),
so what is available here depends on which feather is passed. `describe` reports
what was found so a thin basemap is a known quantity rather than a surprise, and
an absent or unreadable feather yields an empty basemap rather than an error —
the map view is legible without it, just less so.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
)

# Layer definitions, drawn in this order (first = furthest back):
# (name, geometry kind, tag predicates, simplify scale, vertex budget).
# Ordering is cartographic — fills before lines, big before small — so the
# harbour reads as water-with-edges.
#
# The per-layer simplify scale multiplies the base tolerance. It is not a
# quality dial but an information one: a coastline's shape is what tells you
# where the vessel can be, so it is worth vertices, while a building footprint
# at a 20 km extent is a smudge either way and only its position matters.
LAYER_SPECS = (
    ("land", "polygon", (("place", {"island", "islet"}),
                         ("natural", {"land", "beach", "cape", "peninsula"}),
                         ("landuse", None)), 1.5, 9000),
    ("water", "polygon", (("natural", {"water", "bay", "strait", "wetland"}),
                          ("waterway", {"riverbank", "dock"})), 1.5, 9000),
    ("coastline", "line", (("natural", {"coastline"}),), 1.0, 14000),
    ("pier", "line", (("man_made", {"pier", "breakwater", "groyne", "quay"}),
                      ("waterway", {"dam", "weir"})), 1.5, 7000),
    ("bridge", "line", (("bridge", None),), 2.0, 4000),
    ("building", "polygon", (("building", None),), 4.0, 7000),
)
# Base vertex tolerance, in metres of ENU error. Sized to the display: a harbour
# run spans ~23 km on a canvas around 760 px wide, so one pixel is ~30 m and a
# finer tolerance buys bytes rather than detail.
DEFAULT_SIMPLIFY_M = 30.0
# Fallback per-layer cap for layers with no budget of their own. Reported when
# hit, never silent.
DEFAULT_MAX_VERTICES_PER_LAYER = 8000


@dataclasses.dataclass
class Layer:
    name: str
    kind: str  # "polygon" | "line"
    # Each path is a flat [e0, n0, e1, n1, ...] list of ENU metres, rounded to
    # whole metres: a metre is far below what a backdrop resolves, and the
    # integers halve the page size versus floats.
    paths: list
    n_features: int
    n_vertices: int
    truncated: bool = False


@dataclasses.dataclass
class Basemap:
    layers: list
    source: str | None
    notes: tuple = ()

    @property
    def n_vertices(self) -> int:
        return sum(layer.n_vertices for layer in self.layers)

    def describe(self) -> str:
        if not self.layers:
            return f"basemap: empty ({'; '.join(self.notes) or 'no source'})"
        parts = [f"{layer.name} {layer.n_features}f/{layer.n_vertices}v"
                 + ("(truncated)" if layer.truncated else "")
                 for layer in self.layers]
        head = (f"basemap from {self.source}: " + ", ".join(parts)
                + f" — {self.n_vertices} vertices total")
        return head + ("".join(f"\n  note: {n}" for n in self.notes))

    def to_payload(self) -> dict:
        return {"layers": [{"name": layer.name, "kind": layer.kind,
                            "paths": layer.paths}
                           for layer in self.layers if layer.paths],
                "source": self.source, "notes": list(self.notes)}


def _matches(tags: dict, predicates) -> bool:
    for key, values in predicates:
        if key not in tags:
            continue
        if values is None or tags[key] in values:
            return True
    return False


def _rings(geometry, frame: geodesy.RegionFrame, simplify_m: float) -> list:
    """A shapely geometry -> ENU paths, simplified.

    Simplification happens in ENU rather than in degrees so the tolerance means
    metres everywhere, instead of shrinking with latitude.
    """
    kind = geometry.geom_type
    if kind in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
        out = []
        for part in geometry.geoms:
            out.extend(_rings(part, frame, simplify_m))
        return out
    if kind == "Polygon":
        coords = [geometry.exterior.coords]
    elif kind in ("LineString", "LinearRing"):
        coords = [geometry.coords]
    else:
        return []  # Points contribute nothing to a backdrop.

    paths = []
    for ring in coords:
        lon = np.array([c[0] for c in ring], dtype=float)
        lat = np.array([c[1] for c in ring], dtype=float)
        if lon.size < 2:
            continue
        east, north = frame.enu_from_latlon(lat, lon)
        keep = _decimate(east, north, simplify_m)
        if keep.sum() < 2:
            continue
        path = np.empty(int(keep.sum()) * 2, dtype=np.int64)
        path[0::2] = np.rint(east[keep])
        path[1::2] = np.rint(north[keep])
        paths.append(path.tolist())
    return paths


def _decimate(east: np.ndarray, north: np.ndarray, tolerance_m: float
              ) -> np.ndarray:
    """Keep endpoints plus every vertex more than `tolerance_m` from the last
    kept one. Cruder than Douglas-Peucker and enough for a backdrop: it is
    linear, needs no recursion over 100k-vertex coastlines, and bounds the
    visible error by the tolerance directly."""
    n = east.size
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    if tolerance_m <= 0.0:
        return np.ones(n, dtype=bool)
    last_e, last_n = east[0], north[0]
    for i in range(1, n - 1):
        if (east[i] - last_e) ** 2 + (north[i] - last_n) ** 2 >= tolerance_m ** 2:
            keep[i] = True
            last_e, last_n = east[i], north[i]
    return keep


def build(feather_path: Path | None, anchor_lat_deg: float,
          anchor_lon_deg: float, bounds_enu=None,
          simplify_m: float = DEFAULT_SIMPLIFY_M,
          max_vertices_per_layer: int = DEFAULT_MAX_VERTICES_PER_LAYER
          ) -> Basemap:
    """Extract basemap layers from a landmark feather.

    `bounds_enu` is (east_min, east_max, north_min, north_max); features whose
    centroid falls outside are dropped, so a region-sized feather does not put
    the whole state into a harbour page.
    """
    if feather_path is None:
        return Basemap([], None, ("no feather given",))
    feather_path = Path(feather_path)
    if not feather_path.exists():
        return Basemap([], str(feather_path),
                       (f"{feather_path} does not exist",))

    try:
        import geopandas as gpd
        from experimental.overhead_matching.swag.data import landmark_schema
    except ImportError as exc:  # pragma: no cover - dependency shape
        return Basemap([], str(feather_path),
                       (f"geometry dependencies unavailable: {exc}",))

    frame = geodesy.RegionFrame(anchor_lat_deg, anchor_lon_deg)
    notes: list[str] = []
    try:
        frame_data = gpd.read_feather(feather_path)
    except Exception as exc:  # noqa: BLE001 - a bad feather must not be fatal
        return Basemap([], str(feather_path), (f"unreadable: {exc}",))

    if "geometry" not in frame_data.columns:
        return Basemap([], str(feather_path),
                       ("feather carries no geometry column",))

    tag_dicts = landmark_schema.tag_dicts(frame_data)
    geometries = frame_data.geometry.to_numpy()

    if bounds_enu is not None:
        # Bounding-box midpoint rather than `.centroid`: the frame is in
        # lat/lon, where a true centroid is both wrong and warned about, and a
        # box midpoint is all an extent filter needs.
        box = frame_data.geometry.bounds
        centroid_lon = ((box["minx"] + box["maxx"]) / 2.0).to_numpy()
        centroid_lat = ((box["miny"] + box["maxy"]) / 2.0).to_numpy()
        east, north = frame.enu_from_latlon(centroid_lat, centroid_lon)
        east_min, east_max, north_min, north_max = bounds_enu
        inside = ((east >= east_min) & (east <= east_max)
                  & (north >= north_min) & (north <= north_max))
        n_dropped = int((~inside).sum())
        if n_dropped:
            notes.append(f"{n_dropped} feature(s) outside the run's extent "
                         f"were dropped")
    else:
        inside = np.ones(len(frame_data), dtype=bool)

    # One pass over the rows, assigning each to the first layer it matches, so
    # a building on a pier is drawn once rather than twice.
    buckets: dict[str, list] = {spec[0]: [] for spec in LAYER_SPECS}
    for index in np.flatnonzero(inside):
        tags = tag_dicts[index]
        for name, _, predicates, _, _ in LAYER_SPECS:
            if _matches(tags, predicates):
                buckets[name].append(geometries[index])
                break

    layers = []
    for name, kind, _, scale, budget in LAYER_SPECS:
        cap = budget or max_vertices_per_layer
        tolerance = simplify_m * scale
        # Largest features first, so a cap costs the least important geometry
        # rather than whatever the feather happened to order last.
        candidates = sorted(
            (g for g in buckets[name] if g is not None and not g.is_empty),
            key=lambda g: -g.length if g.length else -g.area)
        paths, vertices, truncated = [], 0, False
        for geometry in candidates:
            for path in _rings(geometry, frame, tolerance):
                if vertices + len(path) // 2 > cap:
                    truncated = True
                    break
                paths.append(path)
                vertices += len(path) // 2
            if truncated:
                break
        if truncated:
            notes.append(
                f"layer {name!r}: {len(candidates)} features exceed the "
                f"{cap}-vertex budget; the largest {len(paths)} are drawn and "
                f"the rest omitted")
        if paths:
            layers.append(Layer(name=name, kind=kind, paths=paths,
                                n_features=len(buckets[name]),
                                n_vertices=vertices, truncated=truncated))
    if not layers:
        notes.append("no basemap-worthy geometry matched; this feather is "
                     "probably class-filtered to point landmarks")
    return Basemap(layers, str(feather_path), tuple(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather", type=Path, required=True)
    parser.add_argument("--anchor_lat_deg", type=float, required=True)
    parser.add_argument("--anchor_lon_deg", type=float, required=True)
    parser.add_argument("--simplify_m", type=float,
                        default=DEFAULT_SIMPLIFY_M)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the payload as JSON for inspection")
    args = parser.parse_args()

    basemap = build(args.feather, args.anchor_lat_deg, args.anchor_lon_deg,
                    simplify_m=args.simplify_m)
    print(basemap.describe())
    if args.output:
        args.output.write_text(json.dumps(basemap.to_payload(),
                                          separators=(",", ":")))
        print(f"wrote {args.output} "
              f"({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
