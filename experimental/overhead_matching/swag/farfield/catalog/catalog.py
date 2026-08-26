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
import json
import math
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.catalog import schema

# ---------------------------------------------------------------------------
# Far-field tag vocabulary
# ---------------------------------------------------------------------------
# Why this list is not the street-level keep-list, and the row counts behind
# religion/denomination and the name:en / *-Latn exceptions: see
# docs/farfield/decisions.md, 2026-08 'Far-field tag vocabulary'.
# Keep identity and appearance tags that a distant observer can judge,
# including maritime and ENC navigation-aid vocabulary. Operational
# street-level metadata that is not visible at range is omitted.

FAR_FIELD_KEEP_KEYS = frozenset({
    # what the thing IS
    "man_made", "historic", "place", "natural", "building", "landuse",
    "leisure", "amenity", "tourism", "power", "bridge", "aeroway", "railway",
    "waterway", "water", "industrial", "military", "barrier", "highway",
    "object_class",           # ENC feature class
    # `amenity=place_of_worship` alone does not distinguish visually different
    # structures across regions; religion and denomination carry that identity.
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
# The VLM emits English/Latin-script identity strings while a bare map `name`
# may use a local script. Keep the English exonym and Latin transliterations;
# other language variants duplicate the bare name or add unrelated scripts.
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


# Bump whenever CatalogEntry's fields or the parsing/pruning logic change.
# The key also includes the owned Feather schema version, source-file identity,
# exact anchor, and every loader option.
CACHE_VERSION = 7


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
        return self.hull_east_m.size >= 2


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


def _finite_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


def _loader_options(position_sigma_m, keep_hulls) -> dict:
    sigma = _finite_float(position_sigma_m, "position_sigma_m")
    if sigma <= 0.0:
        raise ValueError("position_sigma_m must be positive")
    if not isinstance(keep_hulls, bool):
        raise TypeError("keep_hulls must be a bool")
    return {
        "position_sigma_m": sigma,
        "keep_hulls": keep_hulls,
    }


def _cache_key(feather_path: Path, feather_sha256: str, anchor_lat_deg: float,
               anchor_lon_deg: float, options: dict) -> str:
    identity = {
        "cache_version": CACHE_VERSION,
        "catalog_schema_version": schema.SCHEMA_VERSION,
        "feather": {
            "resolved_path": str(feather_path),
            "sha256": feather_sha256,
        },
        # float.hex is an exact representation; nearby anchors must never
        # collapse onto one rounded cache entry.
        "anchor_lat_deg": anchor_lat_deg.hex(),
        "anchor_lon_deg": anchor_lon_deg.hex(),
        "loader_options": {
            "position_sigma_m": options["position_sigma_m"].hex(),
            "keep_hulls": options["keep_hulls"],
        },
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _read_cache(cache_path: Path, key: str) -> list[CatalogEntry]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("obsolete catalog cache payload")
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError("obsolete catalog cache version")
    if payload.get("catalog_schema_version") != schema.SCHEMA_VERSION:
        raise ValueError("obsolete catalog schema version")
    if payload.get("key") != key:
        raise ValueError("catalog cache identity mismatch")
    entries = payload.get("entries")
    if (not isinstance(entries, list)
            or any(not isinstance(entry, CatalogEntry) for entry in entries)):
        raise ValueError("invalid catalog cache entries")
    return entries


def _publish_cache(cache_path: Path, key: str,
                   entries: list[CatalogEntry]) -> None:
    payload = {
        "cache_version": CACHE_VERSION,
        "catalog_schema_version": schema.SCHEMA_VERSION,
        "key": key,
        "entries": entries,
    }
    fd, temp_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, cache_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def load_catalog_cached(feather_path, anchor_lat_deg, anchor_lon_deg,
                        cache_dir, *, position_sigma_m,
                        keep_hulls=True):
    """Load through an explicit disposable, identity-bound on-disk cache.

    ``cache_dir`` is required because the Feather commonly lives inside an
    immutable CATALOGS artifact.  Inferring a cache beside that file would
    mutate the published artifact and invalidate its manifest.
    """
    options = _loader_options(position_sigma_m, keep_hulls)
    anchor_lat_deg = _finite_float(anchor_lat_deg, "anchor_lat_deg")
    anchor_lon_deg = _finite_float(anchor_lon_deg, "anchor_lon_deg")
    feather_path = Path(feather_path).resolve(strict=True)
    feather_sha256 = artifact.sha256_file(feather_path)

    cache_dir = Path(cache_dir)
    key = _cache_key(
        feather_path, feather_sha256, anchor_lat_deg, anchor_lon_deg, options)
    cache_path = cache_dir / f"catalog_{key}.pkl"
    if cache_path.exists():
        try:
            return _read_cache(cache_path, key)
        except Exception:
            # A cache is derived, never authoritative. Incompatible classes,
            # interrupted writes, and corrupt bytes all rebuild from Feather.
            cache_path.unlink(missing_ok=True)

    entries = load_catalog(
        feather_path,
        anchor_lat_deg,
        anchor_lon_deg,
        position_sigma_m=options["position_sigma_m"],
        keep_hulls=options["keep_hulls"],
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _publish_cache(cache_path, key, entries)
    return entries


def load_catalog(feather_path, anchor_lat_deg, anchor_lon_deg, *,
                 position_sigma_m, keep_hulls=True):
    """Load every catalog row into anchor-relative ENU.

    position_sigma_m is one explicit, positive uncertainty applied uniformly
    to this catalog. Geometry extents retain every coordinate of each convex
    hull; no sampling is performed.
    """
    options = _loader_options(position_sigma_m, keep_hulls)
    anchor_lat_deg = _finite_float(anchor_lat_deg, "anchor_lat_deg")
    anchor_lon_deg = _finite_float(anchor_lon_deg, "anchor_lon_deg")
    frame = schema.read_frame(feather_path)
    if frame.crs.to_epsg() != 4326:
        frame = frame.to_crs("EPSG:4326")
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
    seen_landmark_ids = set()
    for i in range(len(frame)):
        source = sources[i]
        if source not in schema.ALLOWED_LANDMARK_TYPES:
            raise schema.CatalogSchemaError(
                f"unknown landmark source at row {i}: {source!r}")
        tags = prune_far_field_tags(tag_records[i])
        hull_e = np.zeros(0, dtype=np.float64)
        hull_n = np.zeros(0, dtype=np.float64)
        if options["keep_hulls"]:
            geom = geoms[i]
            if geom.geom_type != "Point":
                hull = shapely.convex_hull(geom)
                coords = np.asarray(
                    shapely.get_coordinates(hull), dtype=np.float64)
                if coords.shape[0]:
                    hull_e, hull_n = region.enu_from_latlon(
                        coords[:, 1], coords[:, 0])

        id_text = _id_text(ids[i])
        landmark_id = (id_text if id_text.startswith(f"{source}:")
                       else f"{source}:{id_text}")
        if landmark_id in seen_landmark_ids:
            raise schema.CatalogSchemaError(
                f"catalog ids collide after source namespacing: "
                f"{landmark_id!r}")
        seen_landmark_ids.add(landmark_id)
        entries.append(CatalogEntry(
            landmark_id=landmark_id,
            source=source,
            east_m=float(east[i]),
            north_m=float(north[i]),
            position_sigma_m=options["position_sigma_m"],
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
