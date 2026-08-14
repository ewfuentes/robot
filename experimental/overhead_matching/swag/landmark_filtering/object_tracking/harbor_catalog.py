"""Map-side landmark catalog for bearing matching (OSM + ENC).

Loads `landmarks/harbor_osm_enc_v1.feather` into region-anchored ENU with
the tag bundles the correspondence model expects.

**No class filtering happens here** — but as of 2026-08-12 the project does
filter, in a separate pre-pass. `scripts/trim_landmark_feather.py` writes a
trimmed feather (184,805 -> 13,210 rows, guarded at recall 1.0 against the
pairing labels) and leaves the full table untouched, so which classes this
catalog contains is decided by *which feather you hand it*. Load the full one
for anything needing completeness (occlusion, compound skylines, shoreline).

The argument this docstring used to make against filtering still holds on its
own terms and is why the trim is a separate artifact rather than a change
here: the filter's uniform `log_prior = -log(n)` dilutes every candidate as
the catalog grows, and the right fix for that is *spatial gating* — only
candidates inside a pose's bearing wedge participate in that pose's update
(design doc SS5.3 `cand(x)`), the same trick LOCI uses when it takes Set 2
from the satellite tiles covering a panorama rather than from all of OSM.
Gating is still unimplemented, which is what made an interim trim worth it;
it does not discharge that work. Generic entries that survive are still
handled by uniqueness weighting downstream, not exclusion.

Geometry is kept, not collapsed to a point: OSM ships polygons and lines, our
measurements carry an angular width, and the audit labels objects
point_like / small_extended / large_extended. `bearing_span_from` returns the
angular interval a candidate subtends, so an island or a bridge can be
matched as an extent rather than a centroid — which also removes the
centroid-drift bias that makes an extended object's apparent bearing move
with viewing aspect. It is used by the mount-offset calibration; the
`wedge_candidates` spatial gate that also used it has been removed, because
selecting candidates by the vessel's position and then localizing from them
is circular.
"""

import ast
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from experimental.overhead_matching.swag.data import landmark_schema
import shapely
import shapely.wkb

# ---------------------------------------------------------------------------
# Harbor tag vocabulary
# ---------------------------------------------------------------------------
# Deliberately NOT `semantic_landmark_utils.prune_landmark`. That keep-list was
# built for street-level VIGOR panoramas, where a shopfront's opening hours and
# housenumber are legible; here the nearest landmark is hundreds of metres away
# and most are kilometres off. Two changes follow:
#   - ADD the maritime vocabulary the harbor tables actually carry, above all
#     `seamark:*` (559 rows) and ENC's `object_class` (883 rows). These are the
#     surveyed navigation aids - the single most matchable class we have - and
#     the street-level list drops every one of them.
#   - DROP operational street furniture (addr:*, lanes, surface, opening hours,
#     payment) that cannot be observed at range and only dilutes the bundle.
# The correspondence model is being retrained on this environment, so the
# distribution shift against the released checkpoint is accepted deliberately.
#
# Note for retraining: cross-feature 4 is `housenumber_overlap`, which is dead
# weight in this domain (addr:housenumber is dropped, so it is always 0). That
# slot is free for something that matters far-field - angular-size agreement or
# a seamark-category match.

HARBOR_KEEP_KEYS = frozenset({
    # what the thing IS
    "man_made", "historic", "place", "natural", "building", "landuse",
    "leisure", "amenity", "tourism", "power", "bridge", "aeroway", "railway",
    "waterway", "water", "industrial", "military", "barrier", "highway",
    "object_class",           # ENC feature class
    # identity
    "name", "alt_name", "short_name", "official_name", "loc_name",
    "old_name", "operator", "brand", "ref", "description",
    # appearance at range - what a distant observer can actually judge
    "height", "colour", "color", "material", "architect",
    "ele", "width",
    # maritime specifics that are not seamark:-prefixed
    "ferry", "dock", "mooring", "lock", "harbour", "wreck", "boat",
    "ship", "maritime", "lighthouse", "beacon",
})

# Prefix families kept wholesale: every seamark:*/light:*/beacon:* subtag is
# identity-bearing for a navigation aid (character, colour, period, range).
HARBOR_KEEP_PREFIXES = (
    "seamark:", "light:", "beacon:", "tower:", "building:", "roof:",
    "bridge:", "man_made:", "historic:", "generator:", "wreck:",
)

# Explicitly dropped even if a prefix above would admit them.
HARBOR_DROP_PREFIXES = (
    "addr:", "payment:", "massgis:", "mass", "contact:", "survey:",
    "source:", "ref:", "name:",   # name:xx are language variants, not identity
    "building:levels:", "check_date",
)


def prune_harbor_tags(props: dict) -> dict:
    """key -> str value, keeping only far-field-identifying tags."""
    out = {}
    for key, value in props.items():
        if any(key.startswith(p) for p in HARBOR_DROP_PREFIXES):
            continue
        if (key not in HARBOR_KEEP_KEYS
                and not any(key.startswith(p) for p in HARBOR_KEEP_PREFIXES)):
            continue
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "nat", "none"):
            continue
        out[key] = text
    return out

# Map-accuracy classes, projected into the angular domain by
# LandmarkCatalog.kappa_eff. ENC is hydrographic-survey grade; OSM here is
# largely imported/traced and much looser.
ENC_POSITION_SIGMA_M = 5.0
OSM_POSITION_SIGMA_M = 15.0

EARTH_RADIUS_M = 6378137.0

# Bump whenever CatalogEntry's fields or the parsing/pruning logic change.
# The cache key covers the feather and the anchor, but nothing about this
# module - without a version, a fix to id parsing or the keep-list silently
# keeps serving entries built by the old code.
CACHE_VERSION = 3

# Columns that are structural rather than tags.
NON_TAG_COLUMNS = ("id", "geometry", "landmark_type")


@dataclass
class CatalogEntry:
    landmark_id: str
    source: str            # "osm" | "enc"
    east_m: float
    north_m: float
    position_sigma_m: float
    tags: dict             # pruned key=value, correspondence-model vocabulary
    # Convex-hull vertices in ENU, for angular extent. Empty for point
    # features, where the centroid already is the whole story.
    hull_east_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hull_north_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def is_extended(self) -> bool:
        return self.hull_east_m.size > 2


def enu_from_latlon(lat_deg, lon_deg, anchor_lat_deg, anchor_lon_deg):
    """Equirectangular local ENU about the anchor.

    Region-scale only (design doc: local ENU, NOT UTM - grid convergence is
    ~1.3 deg at Boston - and NOT raw lat/lon, whose cos(lat) bearing errors
    reach ~8.5 deg).
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat0 = math.radians(anchor_lat_deg)
    lon0 = math.radians(anchor_lon_deg)
    east = (lon - lon0) * math.cos(lat0) * EARTH_RADIUS_M
    north = (lat - lat0) * EARTH_RADIUS_M
    return east, north


def _id_text(raw) -> str:
    """Flatten the feather's `id` into a stable string.

    The column holds (kind, id) pairs, but stored as their *repr* -
    "('node', 31419650)" - not as tuples, so a plain str() would carry the
    punctuation into every landmark_id. Parse both forms.
    """
    if isinstance(raw, (tuple, list)):
        parts = list(raw)
    else:
        text = str(raw)
        parts = None
        if text.startswith("(") and text.endswith(")"):
            try:
                value = ast.literal_eval(text)
                if isinstance(value, (tuple, list)):
                    parts = list(value)
            except (ValueError, SyntaxError):
                parts = None
        if parts is None:
            return text
    # Join ALL parts: the leading element is the OSM element kind
    # ("node"/"way"/"relation"), and node 31419650 and way 31419650 are
    # different features. Dropping it collides them.
    return ":".join(str(part) for part in parts)


def load_catalog_cached(feather_path, anchor_lat_deg, anchor_lon_deg,
                        cache_dir=None, **kwargs):
    """load_catalog with an on-disk cache.

    Decoding 156 k WKB geometries and their convex hulls takes ~4 minutes,
    which is far too slow to sit in front of every matcher experiment. The
    cache key covers the feather's identity and the anchor, so moving the
    anchor or regenerating the table invalidates it.
    """
    feather_path = Path(feather_path)
    if cache_dir is None:
        cache_dir = feather_path.parent / "catalog_cache"
    cache_dir = Path(cache_dir)
    stat = feather_path.stat()
    key = hashlib.sha256(
        f"v{CACHE_VERSION}|{feather_path}|{stat.st_size}|{int(stat.st_mtime)}|"
        f"{anchor_lat_deg:.6f}|{anchor_lon_deg:.6f}|{sorted(kwargs.items())}"
        .encode()).hexdigest()[:16]
    cache_path = cache_dir / f"catalog_{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    entries = load_catalog(feather_path, anchor_lat_deg, anchor_lon_deg,
                           **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(entries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    return entries


def load_catalog(feather_path, anchor_lat_deg, anchor_lon_deg,
                 keep_hulls=True, max_hull_points=24):
    """Every row of the feather as a CatalogEntry, in ENU about the anchor."""
    frame = pd.read_feather(feather_path)

    geoms = shapely.wkb.loads(frame["geometry"].values)
    centroids = shapely.centroid(geoms)
    lon = shapely.get_x(centroids)
    lat = shapely.get_y(centroids)
    east, north = enu_from_latlon(lat, lon, anchor_lat_deg, anchor_lon_deg)

    # Both the dict schema and the legacy wide layout.
    tag_records = landmark_schema.tag_dicts(frame)
    sources = frame["landmark_type"].values
    ids = frame["id"].values

    entries = []
    for i in range(len(frame)):
        source = "enc" if sources[i] == "enc" else "osm"
        tags = prune_harbor_tags(tag_records[i])
        hull_e = hull_n = np.zeros(0)
        if keep_hulls:
            geom = geoms[i]
            if geom.geom_type not in ("Point",):
                hull = shapely.convex_hull(geom)
                coords = np.asarray(shapely.get_coordinates(hull))
                if coords.shape[0] > 2:
                    if coords.shape[0] > max_hull_points:
                        step = coords.shape[0] // max_hull_points
                        coords = coords[::step]
                    hull_e, hull_n = enu_from_latlon(
                        coords[:, 1], coords[:, 0],
                        anchor_lat_deg, anchor_lon_deg)
        id_text = _id_text(ids[i])
        entries.append(CatalogEntry(
            landmark_id=(id_text if id_text.startswith(f"{source}:")
                         else f"{source}:{id_text}"),
            source=source,
            east_m=float(east[i]),
            north_m=float(north[i]),
            position_sigma_m=(ENC_POSITION_SIGMA_M if source == "enc"
                              else OSM_POSITION_SIGMA_M),
            tags=tags,
            hull_east_m=np.asarray(hull_e, dtype=np.float64),
            hull_north_m=np.asarray(hull_n, dtype=np.float64)))
    return entries


def world_bearing_deg(from_east, from_north, to_east, to_north):
    """Compass bearing (CW from north) from one ENU point to another."""
    return math.degrees(math.atan2(to_east - from_east,
                                   to_north - from_north)) % 360.0


def angular_delta_deg(a_deg, b_deg):
    """Signed smallest difference a - b, in (-180, 180]."""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def bearing_span_from(entry: CatalogEntry, east_m: float, north_m: float):
    """(centre_bearing_deg, half_width_deg) subtended by a candidate.

    Point features give half_width 0. For extended features the span is the
    hull's angular extent about the observer, unwrapped relative to the
    centroid bearing so a feature straddling north does not report ~180 deg.
    """
    centre = world_bearing_deg(east_m, north_m, entry.east_m, entry.north_m)
    if not entry.is_extended:
        return centre, 0.0
    deltas = [
        angular_delta_deg(
            world_bearing_deg(east_m, north_m, float(e), float(n)), centre)
        for e, n in zip(entry.hull_east_m, entry.hull_north_m)]
    lo, hi = min(deltas), max(deltas)
    return (centre + 0.5 * (lo + hi)) % 360.0, 0.5 * (hi - lo)


def tags_to_text(tags: dict) -> list[str]:
    """Unique text values needing embeddings (matches how
    precompute_value_embeddings keys its pickle: by raw value string)."""
    return sorted({str(v) for v in tags.values() if str(v).strip()})
