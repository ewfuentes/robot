"""Map-side landmark catalog for bearing matching (OSM + ENC).

Loads a landmark feather into region-anchored ENU with the far-field tag
bundles the matcher expects. **No class filtering happens here**: which rows
this catalog contains is decided by which feather you hand it (the trim tool
writes trimmed feathers as separate catalog versions; the full table stays
untouched). Load the full one for anything needing completeness.

Geometry is kept, not collapsed to a point: OSM ships polygons and lines, our
measurements carry an angular width, and the audit labels objects point_like /
small_extended / large_extended. `bearing_span_from` returns the angular
interval a candidate subtends, so an island or a bridge can be matched as an
extent rather than a centroid — which also removes the centroid-drift bias
that makes an extended object's apparent bearing move with viewing aspect.

All geometry/frames delegate to `farfield.geometry`; all feather reading goes
through `catalog.schema.read_frame`.
"""

import ast
import hashlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.catalog import schema

# ---------------------------------------------------------------------------
# Far-field tag vocabulary
# ---------------------------------------------------------------------------
# Named for what it selects -- tags a distant observer could judge -- not for the
# environment it was first written in. It is a superset vocabulary: the maritime
# families below stay because a harbour table carries them, and they cost nothing
# to keep where a table has none.
# Deliberately NOT the street-level VIGOR keep-list: there the nearest landmark
# is metres away and a shopfront's opening hours are legible; here most
# landmarks are kilometres off. Two changes follow:
#   - ADD the maritime vocabulary harbor tables actually carry, above all
#     `seamark:*` and ENC's `object_class` -- the surveyed navigation aids are
#     the single most matchable class we have.
#   - DROP operational street furniture (addr:*, lanes, surface, opening hours,
#     payment) that cannot be observed at range and only dilutes the bundle.

FAR_FIELD_KEEP_KEYS = frozenset({
    # what the thing IS
    "man_made", "historic", "place", "natural", "building", "landuse",
    "leisure", "amenity", "tourism", "power", "bridge", "aeroway", "railway",
    "waterway", "water", "industrial", "military", "barrier", "highway",
    "object_class",           # ENC feature class
    # `amenity=place_of_worship` alone is nearly free of information in an
    # Anglophone harbour (~90% christian) and the opposite elsewhere: a torii,
    # tiered tiled roofs, a minaret, and a steeple are about as
    # far-field-legible as buildings get, and the split is real -- tokyo_bay
    # carries shinto 3,464 vs buddhist 3,328. religion/denomination are the
    # discriminating tags on those rows.
    "religion", "denomination",
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
FAR_FIELD_KEEP_PREFIXES = (
    "seamark:", "light:", "beacon:", "tower:", "building:", "roof:",
    "bridge:", "man_made:", "historic:", "generator:", "wreck:",
)

# Explicitly dropped even if a prefix above would admit them.
FAR_FIELD_DROP_PREFIXES = (
    "addr:", "payment:", "massgis:", "mass", "contact:", "survey:",
    "source:", "ref:", "name:",   # except FAR_FIELD_KEEP_NAME_VARIANTS below
    "building:levels:", "check_date",
)

# Exceptions to the `name:` drop, checked FIRST so the blanket prefix cannot
# swallow them.
#
# "name:xx are language variants, not identity" is true exactly when the bare
# `name` is already in the language the observer speaks. Elsewhere the bare
# `name` is in the local script and these are the only strings that share an
# alphabet with what a VLM reports: on pohang_canal_04, 6,360 of 7,207 named
# rows are Hangul-only, and dropping these left just 387 of 12,766 rows with
# any Latin identity. It is also asymmetric, because the *observer* emits
# English. Kept deliberately narrow: `name:en` is the English exonym, `*-Latn`
# the romanization; every other variant is a duplicate of the bare name or
# unrelated noise (pohang's bbox carries 2,520 name:el).
FAR_FIELD_KEEP_NAME_VARIANTS = frozenset({"name:en"})
FAR_FIELD_KEEP_NAME_SUFFIXES = ("-latn",)


def is_kept_name_variant(key: str) -> bool:
    """True for the `name:xx` variants worth keeping (see the note above)."""
    lowered = key.lower()
    if not lowered.startswith("name:"):
        return False
    return (lowered in FAR_FIELD_KEEP_NAME_VARIANTS
            or lowered.endswith(FAR_FIELD_KEEP_NAME_SUFFIXES))


def keeps_tag_key(key: str) -> bool:
    """Whether a tag key survives far-field pruning, by key name alone.

    The single source of truth for that question: the trim tool and this
    catalog must select by the same vocabulary or the trim selects rows by one
    list and the catalog reads them with another, silently.
    """
    if is_kept_name_variant(key):
        return True
    if any(key.startswith(p) for p in FAR_FIELD_DROP_PREFIXES):
        return False
    return (key in FAR_FIELD_KEEP_KEYS
            or any(key.startswith(p) for p in FAR_FIELD_KEEP_PREFIXES))


def prune_far_field_tags(props: dict) -> dict:
    """key -> str value, keeping only far-field-identifying tags."""
    out = {}
    for key, value in props.items():
        if not keeps_tag_key(key):
            continue
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if not text or text.lower() in ("nan", "nat", "none"):
            continue
        out[key] = text
    return out


# Map-accuracy classes, projected into the angular domain downstream. ENC is
# hydrographic-survey grade; OSM here is largely imported/traced and looser.
ENC_POSITION_SIGMA_M = 5.0
OSM_POSITION_SIGMA_M = 15.0

# Bump whenever CatalogEntry's fields or the parsing/pruning logic change.
# The cache key covers the feather and the anchor, but nothing about this
# module -- without a version, a fix to id parsing or the keep-list silently
# keeps serving entries built by the old code.
CACHE_VERSION = 5   # 5: farfield port (schema.read_frame, geometry delegation)


@dataclass
class CatalogEntry:
    landmark_id: str
    source: str            # "osm" | "enc"
    east_m: float
    north_m: float
    position_sigma_m: float
    tags: dict             # pruned key=value, far-field vocabulary
    # Convex-hull vertices in ENU, for angular extent. Empty for point
    # features, where the centroid already is the whole story.
    hull_east_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hull_north_m: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def is_extended(self) -> bool:
        return self.hull_east_m.size > 2


def _id_text(raw) -> str:
    """Flatten the feather's `id` into a stable string.

    The column holds (kind, id) pairs, but stored as their *repr* --
    "('node', 31419650)" -- not as tuples, so a plain str() would carry the
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

    Decoding 156 k geometries and their convex hulls takes minutes, far too
    slow to sit in front of every matcher experiment. The cache key covers the
    feather's identity, the anchor, and CACHE_VERSION, so moving the anchor or
    regenerating the table invalidates it. Caches are deletable by contract.
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
    """Every row of the feather as a CatalogEntry, in ENU about the anchor.

    keep_hulls/max_hull_points shape extent matching; stages pass them from
    the run's recorded config so the values used are the values recorded.
    """
    frame = schema.read_frame(feather_path)
    region = geo.RegionFrame(anchor_lat_deg, anchor_lon_deg)

    geoms = np.asarray(frame.geometry.values)
    centroids = shapely.centroid(geoms)
    lon = shapely.get_x(centroids)
    lat = shapely.get_y(centroids)
    east, north = region.enu_from_latlon(lat, lon)

    tag_records = schema.tag_dicts(frame)
    sources = frame["landmark_type"].values
    ids = frame["id"].values

    entries = []
    for i in range(len(frame)):
        source = "enc" if sources[i] == "enc" else "osm"
        tags = prune_far_field_tags(tag_records[i])
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
                    hull_e, hull_n = region.enu_from_latlon(
                        coords[:, 1], coords[:, 0])
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


def bearing_span_from(entry: CatalogEntry, east_m: float, north_m: float):
    """(centre_bearing_deg, half_width_deg) subtended by a candidate.

    Point features give half_width 0. For extended features the span is the
    hull's angular extent about the observer, unwrapped relative to the
    centroid bearing so a feature straddling north does not report ~180 deg.
    """
    centre = geo.compass_bearing_deg(entry.east_m - east_m,
                                     entry.north_m - north_m)
    if not entry.is_extended:
        return centre, 0.0
    deltas = [
        float(geo.circular_diff_deg(
            geo.compass_bearing_deg(float(e) - east_m, float(n) - north_m),
            centre))
        for e, n in zip(entry.hull_east_m, entry.hull_north_m)]
    lo, hi = min(deltas), max(deltas)
    return (centre + 0.5 * (lo + hi)) % 360.0, 0.5 * (hi - lo)
