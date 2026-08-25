"""Trim a landmark catalog to entries a far-field observer could match.

**The full table is never modified.** Trimming publishes a new artifact; the
input stays as the fallback whenever a landmark turns out to be missing.

**Input and output are published CATALOGS artifacts**, never loose Feather
files. Each contains exactly `catalog.feather` plus its typed manifest. The
output manifest binds the exact input ArtifactRef and all trim configuration;
the dataset remains immutable.

**Recall evidence is optional.** Catalog construction and trimming precede
matching, so a new dataset must be able to publish its trimmed catalog without
already having matching results. When `--positive_set` or `--matched_from` is
supplied, the tool treats it as a regression guard and refuses a trim that
drops a protected signature unless `--allow_recall_loss` is explicit. These
guards are useful when revising an established trim, but they are not part of
the catalog-construction contract.

**A catalog is immutable and versioned, never overwritten.** Every number
anyone has quoted was computed against the published artifact already there.
The artifact transaction refuses an existing output directory, records the
exact input identity as an upstream, and fingerprints the rule sets in config.

**Spatial clipping bounds the prior's extent**, for regional extracts that
reach far past anything a vehicle could see. It is a prior, not a corridor.
``--clip_km`` publishes a metric square around an explicit centre;
``--clip_bbox_wsen`` publishes an exact WGS84 rectangle. The modes are
mutually exclusive. A reviewed ``--clip_plan`` can bind a rectangle to its
active-scope canonical GPS/mapping bytes and to its buffer/area policy. The
canonical datasets are derived from an explicit ``--farfield_root``; the plan
cannot redirect the reader to alternate paths. Its expected canonical digest
is required so a silently edited plan cannot publish a different catalog.

Note the scope this deliberately does not have: `catalog/catalog.py` argues
against class filtering for the *filter's* catalog, because spatial gating
plus uniqueness weighting already handle catalog size there. This produces a
separate artifact for consumers that want a smaller table; point the run
config's catalog stem at whichever file is appropriate.

Tag vocabulary is `farfield.catalog.catalog` (`keeps_tag_key` /
`prune_far_field_tags`) -- the single source -- so a landmark's surviving tags
here are exactly the ones the matcher would see.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:trim_catalog -- \\
        --input_catalog_dir /data/farfield_matching/artifacts/catalogs/boston_harbor_leg1/v1 \\
        --output_dir /data/farfield_matching/artifacts/catalogs/boston_harbor_leg1/v2_trimmed \\
        --min_building_area_m2 2000 --min_building_levels 6
"""

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import geopandas as gpd
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import publication
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.catalog import catalog
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.collection import (
    active_catalogs,
)
from experimental.overhead_matching.swag.farfield.dataset_tools.landmark_positive_set import (  # noqa: E501
    PositiveSetError,
    load_positive_set,
    open_catalog_artifact,
    open_matching_artifact,
    recall,
    signature_id,
    validate_positive_set,
)


CLIP_PLAN_SCHEMA = "farfield.catalog_clip_plan/v1"
CLIP_PLAN_METRIC = (
    "equirectangular_mid_lat_wgs84_equatorial_radius/v1"
)
CLIP_SCOPE_POLICY_SCHEMA = "farfield.catalog_clip_scope_policy/v1"
CLIP_SCOPE_POLICY_KEYS = frozenset({
    "schema", "minimum_area_floor_km2", "require_reviewed_bbox_plan",
})
TRUSTED_CLIP_SCOPE_POLICY_BY_NAME = MappingProxyType({
    "boston_harbor_20260712": MappingProxyType({
        "schema": CLIP_SCOPE_POLICY_SCHEMA,
        "minimum_area_floor_km2": 625.0,
        "require_reviewed_bbox_plan": True,
    }),
})
CLIP_PLAN_KEYS = frozenset({
    "schema", "scope", "output_dataset", "bbox_datasets",
    "dataset_tables", "bbox_wsen", "scope_policy", "policy",
})
CLIP_POLICY_KEYS = frozenset({
    "metric", "track_bbox_wsen", "nominal_buffer_km",
    "minimum_buffer_km", "minimum_area_km2", "resolved_buffer_km",
    "resolved_area_km2",
})

# Structural keys that say what a thing IS, in the far-field vocabulary. A
# landmark with none of these carries no class a distant observer could name.
STRUCTURAL_KEYS = frozenset({
    "seamark:type", "object_class", "man_made", "historic", "place",
    "natural", "building", "landuse", "leisure", "amenity", "tourism",
    "power", "bridge", "aeroway", "railway", "waterway", "military",
    "industrial",
})

# Unobservable tags come in two tiers, because a name means different things
# depending on what it names.
#
# HARD: the tag is an accurate description of something a vessel can never
# resolve, so a name does not rescue it. Bus stops are the case that forced
# this: "Dorchester Ave @ Dix St" is a real name on a real object that is
# invisible from the water, and admitting every named one put 11,357 roads and
# stops into a catalog built to remove exactly that.
#
# SOFT: the tag is weak or miscoded, and a proper noun is better evidence than
# the tag is. Bunker Hill Monument, a 67 m granite obelisk, carries nothing
# but a name and `tourism=information`.
#
# (key, value) pairs rather than whole keys, because `amenity` also carries
# ferry_terminal and `natural` also carries cliff.
HARD_UNOBSERVABLE_TAGS = frozenset({
    ("natural", "tree"), ("natural", "tree_row"), ("natural", "shrub"),
    ("natural", "scrub"), ("natural", "grassland"), ("natural", "heath"),
    ("amenity", "bench"), ("amenity", "waste_basket"),
    ("amenity", "recycling"),
    ("amenity", "bicycle_parking"), ("amenity", "bicycle_rental"),
    ("amenity", "parking"), ("amenity", "parking_space"),
    ("amenity", "parking_entrance"), ("amenity", "drinking_water"),
    ("amenity", "toilets"), ("amenity", "post_box"), ("amenity", "atm"),
    ("amenity", "telephone"), ("amenity", "vending_machine"),
    ("amenity", "charging_station"), ("amenity", "car_sharing"),
    ("amenity", "bbq"), ("amenity", "shelter"), ("amenity", "hunting_stand"),
    ("man_made", "surveillance"), ("man_made", "street_cabinet"),
    ("man_made", "manhole"), ("man_made", "utility_pole"),
    ("man_made", "petroleum_well"), ("man_made", "monitoring_station"),
    ("power", "pole"), ("power", "minor_line"),
    ("power", "cable_distribution_cabinet"),
    # Track, transit stops and station furniture. Rails sit at ground level
    # and stations here are MBTA subway stops carrying no building tag, so
    # none of it has a silhouette from the water -- and being named
    # ("Dorchester Branch", "Mattapan") does not change that. Two things are
    # deliberately NOT in this list and survive: `bridge` is structural, so
    # all 120 rail bridges keep their bridge tag, and yard/depot/workshop
    # infrastructure is kept below. A large terminal building carries its own
    # `building=train_station` row, which LANDMARK_BUILDING_VALUES admits.
    *[("railway", v) for v in (
        "rail", "switch", "subway", "stop", "light_rail", "platform",
        "platform_edge", "abandoned", "subway_entrance", "station",
        "crossing", "buffer_stop", "disused", "level_crossing", "signal",
        "railway_crossing", "junction", "razed", "halt", "wash", "milestone",
        "crossover", "tram", "tram_stop", "monorail", "narrow_gauge",
        "funicular", "construction", "proposed", "ventilation_shaft",
        "turntable", "traverser", "derail", "phone", "switch_box")],
    ("leisure", "picnic_table"), ("leisure", "firepit"),
    ("leisure", "fitness_station"), ("leisure", "outdoor_seating"),
    ("tourism", "picnic_site"), ("tourism", "guest_house"),
    ("tourism", "apartment"), ("tourism", "hostel"), ("tourism", "motel"),
    ("barrier", "gate"), ("barrier", "fence"), ("barrier", "bollard"),
    ("barrier", "kerb"), ("barrier", "wall"), ("barrier", "lift_gate"),
    ("barrier", "cycle_barrier"), ("barrier", "block"),
    # Tenant businesses and street-level services. These occupy a building
    # rather than being one, so a vessel sees (at most) the host building --
    # which is its own catalog row. "Legal Sea Foods" is a real name on a real
    # thing that is not visible from the water, so a name must not rescue
    # them.
    *[("amenity", v) for v in (
        "restaurant", "fast_food", "cafe", "bar", "pub", "ice_cream",
        "biergarten", "food_court", "nightclub", "bank", "bureau_de_change",
        "pharmacy", "dentist", "doctors", "clinic", "veterinary", "childcare",
        "kindergarten", "post_office", "payment_terminal", "car_rental",
        "car_wash", "driving_school", "hairdresser", "laundry",
        "dry_cleaning", "copyshop", "internet_cafe",
        "bicycle_repair_station", "loading_dock", "fuel", "taxi", "shower",
        "fountain", "waste_disposal", "waste_transfer_station", "studio",
        "money_transfer", "gym", "coworking_space", "language_school",
        "music_school", "dojo")],
    # Small, flat recreational facilities: no silhouette at any range.
    *[("leisure", v) for v in (
        "pitch", "playground", "garden", "swimming_pool", "fitness_centre",
        "dog_park", "bleachers", "track", "ice_rink", "bowling_alley",
        "amusement_arcade", "miniature_golf", "sauna", "dance", "escape_game",
        "sports_hall", "carousel", "common", "recreation_ground",
        "horse_riding", "fishing", "bird_hide")],
    # Ground cover: an area classification, not an object.
    *[("landuse", v) for v in (
        "grass", "flowerbed", "allotments", "meadow", "orchard", "farmland",
        "traffic_island", "greenfield", "village_green", "plant_nursery",
        "recreation_ground", "forest")],
})

# Weak or commonly-miscoded tags: unobservable on their own, but a proper noun
# outranks them. Zoning classes live here rather than in HARD because the
# label often sits on the thing itself -- `landuse=residential; name=Harbor
# Towers` is a labelled positive, and so is `landuse=industrial;
# industrial=port; name=Paul W. Conley Terminal`.
SOFT_UNOBSERVABLE_TAGS = frozenset({
    ("tourism", "information"), ("tourism", "artwork"),
    ("historic", "memorial"), ("man_made", "flagpole"),
    *[("landuse", v) for v in (
        "residential", "commercial", "retail", "construction", "brownfield",
        "cemetery", "religious", "government", "education", "conservation")],
})

# `landuse=railway` is how a railyard is drawn (Southampton Street Yard,
# Codman Yard, MBTA Cabot Yard) -- a large open expanse of track and rolling
# stock that reads from the water, so it stays structural and is kept even
# when unnamed. The yard's individual rails are still dropped by the HARD
# list above.

# `highway` is a key rather than a value list: every highway=* value is a
# road, a path, or street furniture, none of which a vessel can pick out --
# and being named does not change that. A highway feature that is ALSO
# structural survives, because the rule looks at what is left once these are
# removed; `bridge` is the important case and is deliberately NOT listed
# here, so `bridge=yes; highway=footway` keeps its bridge.
HARD_UNOBSERVABLE_KEYS = frozenset({"highway", "barrier", "crossing",
                                    "traffic_calming", "traffic_sign",
                                    "entrance", "footway", "cycleway"})

# A proper noun is the strongest far-field signal there is -- the extractor is
# asked for names, and the correspondence model scores a name-specific cross
# feature -- so a named feature is not dropped merely for lacking a structural
# class. It cannot rescue a HARD-unobservable one.
NAME_KEYS = frozenset({"name", "alt_name", "official_name", "short_name",
                       "loc_name", "old_name"})


def has_name(tags: dict) -> bool:
    """Whether a tag bundle carries a proper noun of any kind.

    Counts the kept `name:xx` variants (`catalog.is_kept_name_variant`), not
    just the bare keys. Without this a feature named only in `name:en` -- 219
    rows in pohang's source table -- reads as nameless, loses the name rescue,
    and is dropped as unobservable despite being the best-identified thing on
    its block.
    """
    return any(k in NAME_KEYS or catalog.is_kept_name_variant(k)
               for k in tags)


# Values that make a building conspicuous from the water on their own.
LANDMARK_BUILDING_VALUES = frozenset({
    "cathedral", "church", "chapel", "tower", "lighthouse", "stadium",
    "hangar", "hospital", "university", "hotel", "civic", "industrial",
    "commercial", "government", "college", "train_station", "transportation",
    "warehouse", "silo", "storage_tank", "digester", "grandstand",
})


def far_field_tag_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    """Per-row pruned far-field tags, matching catalog.load_catalog.

    Compact catalogs have no tag columns to pre-select: the decoded row
    objects are already only as wide as the tags that exist.
    """
    return [catalog.prune_far_field_tags(record)
            for record in schema.tag_dicts(gdf)]


def footprint_area_m2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Approximate footprint area; 0 for points and lines."""
    latitude = gdf.geometry.representative_point().y.to_numpy()
    # One degree of longitude shrinks by cos(lat); one of latitude does not.
    scale = (geo.METERS_PER_DEG_LAT ** 2 * np.cos(np.radians(latitude)))
    with warnings.catch_warnings():
        # Area in a geographic CRS is exactly what we want here: degrees^2,
        # scaled per-row above. Reprojecting 156 k geometries to compare
        # against a coarse threshold would cost far more than it buys.
        warnings.filterwarnings("ignore", message=".*geographic CRS.*")
        degrees_squared = gdf.geometry.area.to_numpy()
    return degrees_squared * scale


def _numeric(value) -> float:
    """First number in a messy OSM value ("12 m", "3;4") or 0."""
    if value is None:
        return 0.0
    text = str(value).split(";")[0].strip().rstrip("m").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def evaluate_rules(tags: list, areas: np.ndarray,
                   min_building_area_m2: float,
                   min_building_levels: float) -> dict:
    """Boolean drop-mask per rule.

    Every rule votes on every row, independently of the others, so the run's
    "only-this" column is a real marginal ablation rather than an artefact of
    evaluation order.
    """
    n = len(tags)
    no_tags = np.zeros(n, dtype=bool)
    unobservable = np.zeros(n, dtype=bool)
    generic_building = np.zeros(n, dtype=bool)

    for i, tag in enumerate(tags):
        no_tags[i] = not tag

        hard_blocked = any(k in HARD_UNOBSERVABLE_KEYS
                           or (k, v) in HARD_UNOBSERVABLE_TAGS
                           for k, v in tag.items())
        observable = {
            k: v for k, v in tag.items()
            if k not in HARD_UNOBSERVABLE_KEYS
            and (k, v) not in HARD_UNOBSERVABLE_TAGS
            and (k, v) not in SOFT_UNOBSERVABLE_TAGS}
        # `bridge` is structural, so a named road bridge keeps its bridge tag
        # even though its highway tag is hard-blocked.
        has_structural = any(k in STRUCTURAL_KEYS for k in observable)
        rescued_by_name = not hard_blocked and has_name(observable)
        unobservable[i] = not (has_structural or rescued_by_name)

        # A building survives on its own merits: another structural tag, a
        # name, a landmark use, real height, or a footprint readable at range.
        building = observable.get("building")
        if building is None:
            continue
        if any(k in STRUCTURAL_KEYS and k != "building" for k in observable):
            continue
        if has_name(tag) or building in LANDMARK_BUILDING_VALUES:
            continue
        height = max(_numeric(tag.get("height")),
                     _numeric(tag.get("building:levels")) * 3.5)
        if height >= min_building_levels * 3.5:
            continue
        generic_building[i] = areas[i] < min_building_area_m2

    return {"no_far_field_tags": no_tags,
            "unobservable_only": unobservable,
            "generic_small_building": generic_building}


def clip_mask(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float,
              box_km: float) -> np.ndarray:
    """True where a row lies inside a `box_km` square centred on the point.

    Deliberately a *prior extent*, not a corridor: the whole-map experiment
    needs a prior far larger than the trajectory (charles's sail spans
    0.8 x 1.1 km inside a 25 x 25 km box, which is itself larger than the
    22.9 x 20.9 km harbour prior the method was validated on). Its only job
    is to stop a regional OSM extract from reaching tens of kilometres past
    anything the vehicle could see. Metric via geometry.RegionFrame, so the
    box is square in metres, not degrees.
    """
    point = gdf.geometry.representative_point()
    region = geo.RegionFrame(center_lat, center_lon)
    east, north = region.enu_from_latlon(np.asarray(point.y),
                                         np.asarray(point.x))
    half = box_km * 1000.0 / 2.0
    return (np.abs(east) <= half) & (np.abs(north) <= half)


def validate_bbox_wsen(value) -> tuple[float, float, float, float]:
    """Return a finite, ordered WGS84 ``(west, south, east, north)``."""
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(
            "clip bbox must contain west south east north") from exc
    if len(raw) != 4:
        raise ValueError("clip bbox must contain west south east north")
    if any(type(item) not in (int, float) for item in raw):
        raise ValueError("clip bbox coordinates must be finite numbers")
    try:
        west, south, east, north = map(float, raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "clip bbox coordinates must be finite numbers") from exc
    if not all(map(math.isfinite, (west, south, east, north))):
        raise ValueError("clip bbox coordinates must be finite numbers")
    if not (-180.0 <= west < east <= 180.0):
        raise ValueError("clip bbox needs -180 <= west < east <= 180")
    if not (-90.0 <= south < north <= 90.0):
        raise ValueError("clip bbox needs -90 <= south < north <= 90")
    return west, south, east, north


def _validate_track_bbox_wsen(value) -> tuple[float, float, float, float]:
    """Validate a possibly-degenerate raw trajectory extent."""
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError(
            "clip plan policy.track_bbox_wsen must contain four numbers") \
            from exc
    if len(raw) != 4 or any(type(item) not in (int, float) for item in raw):
        raise ValueError(
            "clip plan policy.track_bbox_wsen must contain four numbers")
    try:
        west, south, east, north = map(float, raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "clip plan policy.track_bbox_wsen must contain finite numbers") \
            from exc
    if not all(map(math.isfinite, (west, south, east, north))):
        raise ValueError(
            "clip plan policy.track_bbox_wsen must contain finite numbers")
    if not (-180.0 <= west <= east <= 180.0):
        raise ValueError(
            "clip plan track bbox needs -180 <= west <= east <= 180")
    if not (-90.0 <= south <= north <= 90.0):
        raise ValueError(
            "clip plan track bbox needs -90 <= south <= north <= 90")
    return west, south, east, north


def clip_bbox_mask(
        gdf: gpd.GeoDataFrame,
        bbox_wsen: tuple[float, float, float, float],
) -> np.ndarray:
    """True where a row's representative point is inside ``bbox_wsen``.

    Bounds are inclusive. Representative points match :func:`clip_mask` and
    keep the decision stable for concave or multipart geometries whose
    centroid may lie outside the feature.
    """
    west, south, east, north = validate_bbox_wsen(bbox_wsen)
    point = gdf.geometry.representative_point()
    lon = np.asarray(point.x)
    lat = np.asarray(point.y)
    return ((west <= lon) & (lon <= east)
            & (south <= lat) & (lat <= north))


def _finite_plan_number(value, field: str, *, minimum=0.0,
                        strictly_positive=False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"clip plan {field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"clip plan {field} must be a finite number") from exc
    invalid = (not math.isfinite(number) or number < minimum
               or (strictly_positive and number <= 0.0))
    if invalid:
        raise ValueError(
            f"clip plan {field} must be a finite "
            f"{'positive ' if strictly_positive else ''}number"
            + (f" >= {minimum:g}" if minimum else ""))
    return number


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    """Build one JSON object while refusing ambiguous duplicate keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _finite_json_float(text: str) -> float:
    try:
        value = float(text)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid JSON number {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number {text!r}")
    return value


def _reject_json_constant(text: str):
    raise ValueError(f"non-finite JSON constant {text!r}")


def _require_exact_keys(mapping: Mapping, expected: frozenset[str],
                        field: str) -> None:
    actual = set(mapping)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected, key=str)
    details = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unknown {unknown}")
    raise ValueError(f"clip plan {field} has " + "; ".join(details))


def _trusted_scope_policy(scope_name: str) -> dict | None:
    policy = TRUSTED_CLIP_SCOPE_POLICY_BY_NAME.get(scope_name)
    return dict(policy) if policy is not None else None


def _dataset_requires_reviewed_bbox_plan(dataset: str) -> bool:
    for scope_name, policy in TRUSTED_CLIP_SCOPE_POLICY_BY_NAME.items():
        if not policy["require_reviewed_bbox_plan"]:
            continue
        scope = active_catalogs.SCOPE_BY_NAME.get(scope_name)
        if scope is not None and dataset in scope.output_datasets:
            return True
    return False


def _track_metrics(track_bbox) -> tuple[float, float, float, float, float]:
    west, south, east, north = track_bbox
    mid_lat = (south + north) / 2.0
    km_per_deg_lat = geo.METERS_PER_DEG_LAT / 1000.0
    km_per_deg_lon = km_per_deg_lat * math.cos(math.radians(mid_lat))
    if not math.isfinite(km_per_deg_lon) or km_per_deg_lon <= 0.0:
        raise ValueError("clip plan track bbox cannot resolve longitude metres")
    raw_width = (east - west) * km_per_deg_lon
    raw_height = (north - south) * km_per_deg_lat
    return (raw_width, raw_height, km_per_deg_lon, km_per_deg_lat,
            mid_lat)


def _resolve_clip_policy(track_bbox, nominal_buffer: float,
                         minimum_buffer: float,
                         minimum_area: float) -> tuple[float, float, tuple]:
    """Return the exact minimum symmetric buffer, area, and WGS84 bbox."""
    raw_width, raw_height, km_lon, km_lat, _ = _track_metrics(track_bbox)
    raw_area = raw_width * raw_height
    if minimum_area > raw_area:
        discriminant_root = math.hypot(
            raw_width - raw_height, 2.0 * math.sqrt(minimum_area))
        area_root = ((minimum_area - raw_area)
                     / (discriminant_root + raw_width + raw_height))
    else:
        area_root = 0.0
    resolved = max(nominal_buffer, minimum_buffer, area_root)
    area = ((raw_width + 2.0 * resolved)
            * (raw_height + 2.0 * resolved))
    # The algebraic root can round one ULP toward the infeasible side. Use the
    # smallest representable buffer that actually satisfies the declared
    # minimum; this keeps "at least" true in the recorded machine arithmetic.
    while area < minimum_area:
        next_resolved = math.nextafter(resolved, math.inf)
        if next_resolved == resolved or not math.isfinite(next_resolved):
            raise ValueError(
                "clip plan minimum area has no finite representable buffer")
        resolved = next_resolved
        area = ((raw_width + 2.0 * resolved)
                * (raw_height + 2.0 * resolved))
    if not all(map(math.isfinite, (area_root, resolved, area))):
        raise ValueError("clip plan policy overflows finite metric bounds")
    west, south, east, north = track_bbox
    bbox = (west - resolved / km_lon, south - resolved / km_lat,
            east + resolved / km_lon, north + resolved / km_lat)
    # The output must remain a representable, non-wrapping WGS84 rectangle.
    bbox = validate_bbox_wsen(bbox)
    return resolved, area, bbox


def validate_clip_plan(
        document: Mapping,
        bbox_wsen: tuple[float, float, float, float],
        *, output_dataset: str,
) -> dict:
    """Validate and return a reviewed, GPS-bound rectangular clip plan.

    The trim artifact stores this complete document and its canonical digest.
    That binds the otherwise-manual bbox to the canonical GPS files and to the
    policy that resolved its symmetric metric buffer and minimum area.
    """
    if type(document) is not dict:
        raise ValueError("clip plan must be a JSON object")
    plan = dict(document)
    _require_exact_keys(plan, CLIP_PLAN_KEYS, "top-level fields")
    if plan.get("schema") != CLIP_PLAN_SCHEMA:
        raise ValueError(
            f"clip plan schema must be exactly {CLIP_PLAN_SCHEMA!r}")
    if type(plan.get("scope")) is not str or not plan["scope"]:
        raise ValueError("clip plan scope must be a non-empty string")
    scope = active_catalogs.SCOPE_BY_NAME.get(plan["scope"])
    if scope is None:
        raise ValueError(
            "clip plan scope must name an active_catalogs.SCOPE_BY_NAME entry")
    trusted_scope_policy = _trusted_scope_policy(plan["scope"])
    planned_scope_policy = plan.get("scope_policy")
    if trusted_scope_policy is None:
        if planned_scope_policy is not None:
            raise ValueError(
                "clip plan scope_policy must be null for an ungoverned scope")
    else:
        if type(planned_scope_policy) is not dict:
            raise ValueError(
                "clip plan scope_policy must be the trusted policy record")
        _require_exact_keys(
            planned_scope_policy, CLIP_SCOPE_POLICY_KEYS,
            "scope_policy fields")
        if planned_scope_policy != trusted_scope_policy:
            raise ValueError(
                "clip plan scope_policy does not match the trusted versioned "
                "scope policy")
    if type(plan.get("output_dataset")) is not str:
        raise ValueError("clip plan output_dataset must be a string")
    if plan["output_dataset"] != output_dataset:
        raise ValueError(
            "clip plan output_dataset does not exactly match input catalog")
    if output_dataset not in scope.output_datasets:
        raise ValueError(
            "clip plan output_dataset does not belong to the active scope")
    planned_bbox = validate_bbox_wsen(plan.get("bbox_wsen", ()))
    if planned_bbox != tuple(bbox_wsen):
        raise ValueError(
            "clip plan bbox_wsen does not exactly match --clip_bbox_wsen")

    datasets = plan.get("bbox_datasets")
    if (type(datasets) is not list or not datasets
            or any(type(value) is not str or not value
                   for value in datasets)
            or len(datasets) != len(set(datasets))):
        raise ValueError(
            "clip plan bbox_datasets must be a non-empty unique string list")
    if datasets != list(scope.bbox_datasets):
        raise ValueError(
            "clip plan bbox_datasets does not exactly match active scope")
    dataset_tables = plan.get("dataset_tables")
    if (type(dataset_tables) is not list
            or len(dataset_tables) != len(datasets)
            or any(type(record) is not dict for record in dataset_tables)
            or [record.get("dataset") for record in dataset_tables]
            != datasets):
        raise ValueError(
            "clip plan dataset_tables must have one ordered record per "
            "bbox dataset")

    policy = plan.get("policy")
    if type(policy) is not dict:
        raise ValueError("clip plan policy must be a JSON object")
    _require_exact_keys(policy, CLIP_POLICY_KEYS, "policy fields")
    if policy.get("metric") != CLIP_PLAN_METRIC:
        raise ValueError(
            f"clip plan policy.metric must be {CLIP_PLAN_METRIC!r}")
    track_bbox = _validate_track_bbox_wsen(
        policy.get("track_bbox_wsen", ()))
    nominal_buffer = _finite_plan_number(
        policy.get("nominal_buffer_km"), "policy.nominal_buffer_km")
    minimum_buffer = _finite_plan_number(
        policy.get("minimum_buffer_km"), "policy.minimum_buffer_km")
    minimum_area = _finite_plan_number(
        policy.get("minimum_area_km2"), "policy.minimum_area_km2",
        strictly_positive=True)
    if (trusted_scope_policy is not None
            and minimum_area
            < trusted_scope_policy["minimum_area_floor_km2"]):
        raise ValueError(
            "clip plan policy.minimum_area_km2 is below the trusted scope "
            f"floor of {trusted_scope_policy['minimum_area_floor_km2']:g}")
    resolved_buffer = _finite_plan_number(
        policy.get("resolved_buffer_km"), "policy.resolved_buffer_km")
    resolved_area = _finite_plan_number(
        policy.get("resolved_area_km2"), "policy.resolved_area_km2",
        strictly_positive=True)
    expected_buffer, expected_area, expected_bbox = _resolve_clip_policy(
        track_bbox, nominal_buffer, minimum_buffer, minimum_area)
    if not math.isclose(
            resolved_buffer, expected_buffer, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "clip plan resolved_buffer_km is not the exact policy minimum")
    if resolved_area < minimum_area:
        raise ValueError(
            "clip plan resolved_area_km2 is below minimum_area_km2")
    if not math.isclose(
            resolved_area, expected_area, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "clip plan resolved_area_km2 does not match track bbox and "
            "resolved buffer")
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
           for actual, expected in zip(bbox_wsen, expected_bbox)):
        raise ValueError(
            "clip plan bbox_wsen does not match track bbox and resolved "
            "symmetric buffer")
    # Prove the full document is valid canonical-JSON input before it enters
    # an artifact manifest or identity digest.
    try:
        artifact.canonical_json_bytes(plan)
    except (artifact.ArtifactValidationError, UnicodeError,
            OverflowError) as exc:
        raise ValueError(f"clip plan is not canonical JSON: {exc}") from exc
    return plan


def _live_scope_tables(scope, farfield_root: Path):
    records = []
    lats = []
    lons = []
    for dataset in scope.bbox_datasets:
        try:
            record, table_lats, table_lons = \
                active_catalogs.read_dataset_tables(dataset, farfield_root)
        except active_catalogs.ActiveCatalogError as exc:
            raise ValueError(
                f"cannot verify canonical dataset {dataset!r}: {exc}") from exc
        records.append(record)
        lats.extend(table_lats)
        lons.extend(table_lons)
    return records, (min(lons), min(lats), max(lons), max(lats))


def verify_clip_plan_sources(plan: Mapping, farfield_root: Path) -> dict:
    """Re-read root-derived canonical tables and recompute trajectory union."""
    scope = active_catalogs.SCOPE_BY_NAME[plan["scope"]]
    records, track_bbox = _live_scope_tables(scope, Path(farfield_root))
    if records != plan["dataset_tables"]:
        raise ValueError(
            "live canonical dataset tables no longer match the reviewed plan")
    planned_track = tuple(plan["policy"]["track_bbox_wsen"])
    if track_bbox != planned_track:
        raise ValueError(
            "live canonical GPS union does not exactly match "
            "policy.track_bbox_wsen")
    return {
        "farfield_root": str(Path(farfield_root)),
        "dataset_tables": records,
        "track_bbox_wsen": list(track_bbox),
    }


def build_clip_plan(*, scope_name: str, output_dataset: str,
                    farfield_root: Path, nominal_buffer_km,
                    minimum_buffer_km, minimum_area_km2) -> dict:
    """Build a deterministic plan from the current active-scope tables."""
    scope = active_catalogs.SCOPE_BY_NAME.get(scope_name)
    if scope is None:
        raise ValueError(f"unknown active catalog scope: {scope_name!r}")
    if output_dataset not in scope.output_datasets:
        raise ValueError("output dataset does not belong to active scope")
    nominal = _finite_plan_number(
        nominal_buffer_km, "policy.nominal_buffer_km")
    minimum = _finite_plan_number(
        minimum_buffer_km, "policy.minimum_buffer_km")
    area = _finite_plan_number(
        minimum_area_km2, "policy.minimum_area_km2",
        strictly_positive=True)
    trusted_scope_policy = _trusted_scope_policy(scope_name)
    if (trusted_scope_policy is not None
            and area < trusted_scope_policy["minimum_area_floor_km2"]):
        raise ValueError(
            "minimum_area_km2 is below the trusted scope floor of "
            f"{trusted_scope_policy['minimum_area_floor_km2']:g}")
    records, track_bbox = _live_scope_tables(scope, Path(farfield_root))
    resolved, resolved_area, bbox = _resolve_clip_policy(
        track_bbox, nominal, minimum, area)
    plan = {
        "schema": CLIP_PLAN_SCHEMA,
        "scope": scope_name,
        "output_dataset": output_dataset,
        "bbox_datasets": list(scope.bbox_datasets),
        "dataset_tables": records,
        "bbox_wsen": list(bbox),
        "scope_policy": trusted_scope_policy,
        "policy": {
            "metric": CLIP_PLAN_METRIC,
            "track_bbox_wsen": list(track_bbox),
            "nominal_buffer_km": nominal,
            "minimum_buffer_km": minimum,
            "minimum_area_km2": area,
            "resolved_buffer_km": resolved,
            "resolved_area_km2": resolved_area,
        },
    }
    return validate_clip_plan(plan, bbox, output_dataset=output_dataset)


def load_clip_plan(
        path: Path,
        expected_digest: str,
        bbox_wsen: tuple[float, float, float, float],
        *, output_dataset: str,
) -> tuple[dict, str]:
    """Read one reviewed plan and verify its canonical digest and bbox."""
    if (type(expected_digest) is not str or len(expected_digest) != 64
            or any(c not in "0123456789abcdef" for c in expected_digest)):
        raise ValueError(
            "expected clip plan digest must be a lowercase SHA-256 digest")
    try:
        _, encoded = active_catalogs.read_regular_file(
            Path(path), what="clip plan", return_bytes=True)
        assert encoded is not None
        text = encoded.decode("utf-8")
        document = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_float=_finite_json_float,
            parse_constant=_reject_json_constant,
        )
    except (active_catalogs.ActiveCatalogError, UnicodeError,
            json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise ValueError(f"cannot read clip plan {path}: {exc}") from exc
    plan = validate_clip_plan(
        document, bbox_wsen, output_dataset=output_dataset)
    try:
        digest = artifact.sha256_json(plan)
    except (artifact.ArtifactValidationError, UnicodeError,
            OverflowError) as exc:
        raise ValueError(f"clip plan is not canonical JSON: {exc}") from exc
    if digest != expected_digest:
        raise ValueError(
            f"clip plan digest mismatch: expected {expected_digest}, "
            f"found {digest}")
    return plan, digest


def rule_fingerprint(min_building_area_m2: float,
                     min_building_levels: float, *,
                     clip_km=None, clip_center_lat=None,
                     clip_center_lon=None, clip_bbox_wsen=None,
                     clip_plan_digest=None) -> str:
    """Short hash of everything that decides what this trimmer keeps.

    Recorded next to each output so "which rules built this catalog?" is
    answerable from the file. Worth having: `v1_trimmed` files that looked
    stale against the current rules turned out to match them exactly -- the
    analysis had passed thresholds by hand -- and this is what would have
    settled it in one line instead of an afternoon.
    """
    payload = {
        "hard_keys": sorted(HARD_UNOBSERVABLE_KEYS),
        "hard_tags": sorted(map(list, HARD_UNOBSERVABLE_TAGS)),
        "soft_tags": sorted(map(list, SOFT_UNOBSERVABLE_TAGS)),
        "structural_keys": sorted(STRUCTURAL_KEYS),
        "name_keys": sorted(NAME_KEYS),
        "landmark_building_values": sorted(LANDMARK_BUILDING_VALUES),
        "keep_keys": sorted(catalog.FAR_FIELD_KEEP_KEYS),
        "keep_prefixes": sorted(catalog.FAR_FIELD_KEEP_PREFIXES),
        "drop_prefixes": sorted(catalog.FAR_FIELD_DROP_PREFIXES),
        "keep_name_variants": sorted(catalog.FAR_FIELD_KEEP_NAME_VARIANTS),
        "keep_name_suffixes": sorted(catalog.FAR_FIELD_KEEP_NAME_SUFFIXES),
        "min_building_area_m2": min_building_area_m2,
        "min_building_levels": min_building_levels,
    }
    # Preserve the established no-clip fingerprint while ensuring every
    # spatially clipped artifact fingerprints the exact decision boundary.
    if clip_bbox_wsen is not None:
        payload["spatial_clip"] = {
            "mode": "representative_point_bbox_wsen_inclusive/v1",
            "bbox_wsen": list(validate_bbox_wsen(clip_bbox_wsen)),
            "clip_plan_digest": clip_plan_digest,
        }
    elif clip_km is not None:
        payload["spatial_clip"] = {
            "mode": "representative_point_metric_square_inclusive/v1",
            "box_km": clip_km,
            "center_lat": clip_center_lat,
            "center_lon": clip_center_lon,
        }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def catalog_descends_from(
        candidate: artifact.ArtifactRef,
        ancestor: artifact.ArtifactRef,
) -> bool:
    """Whether ``candidate`` is ``ancestor`` or a typed catalog descendant.

    Optional recall evidence is commonly produced against an earlier trimmed
    catalog. A later trim of the same full catalog may use that evidence, but an
    unrelated catalog from the same dataset may not. Following only typed
    CATALOGS upstreams proves that relationship without weakening artifact
    identity to a dataset-name comparison.
    """
    if (candidate.kind != paths_lib.CATALOGS
            or ancestor.kind != paths_lib.CATALOGS
            or candidate.dataset != ancestor.dataset):
        return False
    pending = [candidate]
    visited = set()
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        identity = (current.kind, current.dataset, current.version,
                    current.manifest_digest, current.content_digest)
        if identity in visited:
            continue
        visited.add(identity)
        validated = artifact.open_artifact(
            current.path,
            expected_kind=paths_lib.CATALOGS,
            expected_dataset=current.dataset,
            expected_version=current.version,
        )
        if validated != current:
            raise artifact.ArtifactValidationError(
                "catalog lineage reference no longer identifies its artifact: "
                f"expected {current.to_dict()}, found {validated.to_dict()}")
        manifest = artifact.load_manifest(current.path)
        pending.extend(reference for reference in manifest.upstreams
                       if reference.kind == paths_lib.CATALOGS)
    return False


def matched_signatures(
        sources: list, confidence_floor: float,
        expected_catalog_ref: artifact.ArtifactRef,
) -> tuple[dict, tuple[artifact.ArtifactRef, ...]]:
    """Load signatures only from complete, catalog-bound matching artifacts."""
    found = {}
    references = []
    for source in sources:
        try:
            matching_ref, catalog_ref, matches, _ = open_matching_artifact(
                source)
            if not catalog_descends_from(catalog_ref, expected_catalog_ref):
                raise PositiveSetError(
                    "matching catalog is neither the trim input nor its typed "
                    f"descendant: input={expected_catalog_ref.to_dict()}, "
                    f"matching={catalog_ref.to_dict()}")
        except (artifact.ArtifactError, PositiveSetError) as exc:
            raise SystemExit(
                f"invalid --matched_from LANDMARK_MATCHES artifact "
                f"{source}: {exc}") from exc
        references.append(matching_ref)
        label = matching_ref.version
        for tracklet, records in matches.items():
            for match in records:
                confidence = match["aggregate_confidence"]
                if confidence < confidence_floor:
                    continue
                found.setdefault(match["signature_id"], []).append(
                    (label, tracklet, confidence, match["match_type"],
                     match["signature_display"]))
    return found, tuple(references)


def report_matched_recall(matched: dict, tags: list, kept: np.ndarray,
                          masks: dict) -> tuple[list, list]:
    """Print how rules treat matched signatures; return (lost, absent)."""
    by_signature = {}
    for i, tag in enumerate(tags):
        by_signature.setdefault(signature_id(tag), []).append(i)
    surviving = {signature_id(tags[i]) for i in np.flatnonzero(kept)}
    absent = sorted(s for s in matched if s not in by_signature)
    lost = sorted(set(matched) - surviving - set(absent))
    print(f"\nRECALL on {len(matched)} signatures matched by real runs: "
          f"{len(matched) - len(lost) - len(absent)}/"
          f"{len(matched) - len(absent)} of the ones this table holds survive"
          + (f"; {len(absent)} are absent from this catalog "
             f"({100 * len(absent) / len(matched):.1f}%)" if absent else ""))
    if not lost:
        return lost, absent
    print("LOST signatures a matcher already chose -- fix the rule:")
    for signature in lost[:15]:
        # Which rule is responsible: the rules that fired on the rows
        # carrying this signature.
        blame = sorted({name for name, mask in masks.items()
                        for i in by_signature[signature] if mask[i]})
        run, tracklet, confidence, kind, display = matched[signature][0]
        print(f"  [{','.join(blame)}] {run}/{tracklet} {kind} "
              f"{confidence:.2f}: {display[:70]} ({signature[:19]}...)")
    return lost, absent


def positive_set_identity(positive_set: dict, source: Path,
                          input_ref: artifact.ArtifactRef
                          ) -> tuple[artifact.ArtifactRef,
                                     artifact.ArtifactRef]:
    """Validate schema-v2 matching and catalog identities."""
    try:
        matching_ref, catalog_ref = validate_positive_set(positive_set, source)
    except (artifact.ArtifactError, PositiveSetError, TypeError) as exc:
        raise SystemExit(
            f"invalid positive set {source}: {exc}") from exc
    try:
        related = catalog_descends_from(catalog_ref, input_ref)
    except artifact.ArtifactError as exc:
        raise SystemExit(
            f"cannot validate positive-set catalog lineage for {source}: "
            f"{exc}") from exc
    if not related:
        raise SystemExit(
            f"positive set {source} was built against a catalog that is "
            "neither the trim input nor its typed descendant: "
            f"input={input_ref.to_dict()}, evidence={catalog_ref.to_dict()}")
    return matching_ref, catalog_ref


def main(input_catalog_dir: Path, output_dir: Path, positive_set_path,
         min_building_area_m2: float, min_building_levels: float,
         dry_run: bool, matched_from=None, confidence_floor: float = 0.5,
         allow_recall_loss: bool = False,
         allow_absent_matched_signatures: bool = False,
         clip_km=None, clip_center_lat=None,
         clip_center_lon=None, clip_bbox_wsen=None,
         clip_plan_path=None,
         expected_clip_plan_digest=None,
         farfield_root=None) -> gpd.GeoDataFrame:
    try:
        input_ref, input_path = open_catalog_artifact(input_catalog_dir)
    except artifact.ArtifactError as exc:
        raise SystemExit(f"invalid input catalog artifact: {exc}") from exc
    output_dir = Path(output_dir)
    if output_dir.exists() and not dry_run:
        raise SystemExit(
            f"{output_dir} already exists. A catalog is part of the problem "
            f"definition -- every past number was computed against it -- so "
            f"it is immutable and versioned, never overwritten. Publish a "
            f"new artifact version.")
    if clip_bbox_wsen is not None and any(
            value is not None
            for value in (clip_km, clip_center_lat, clip_center_lon)):
        raise SystemExit(
            "--clip_bbox_wsen is mutually exclusive with the metric-square "
            "--clip_km/--clip_center_lat/--clip_center_lon options")
    if clip_km is None and (clip_center_lat is not None
                            or clip_center_lon is not None):
        raise SystemExit(
            "--clip_center_lat/lon may only be used with --clip_km")
    if ((clip_plan_path is None)
            != (expected_clip_plan_digest is None)):
        raise SystemExit(
            "--clip_plan and --expected_clip_plan_digest must be supplied "
            "together")
    if clip_plan_path is not None and clip_bbox_wsen is None:
        raise SystemExit("--clip_plan may only be used with --clip_bbox_wsen")
    if (clip_plan_path is None) != (farfield_root is None):
        raise SystemExit(
            "--farfield_root is required exactly when --clip_plan is used")

    resolved_bbox = None
    if clip_bbox_wsen is not None:
        try:
            resolved_bbox = validate_bbox_wsen(clip_bbox_wsen)
        except ValueError as exc:
            raise SystemExit(f"invalid --clip_bbox_wsen: {exc}") from exc
    if (resolved_bbox is not None
            and _dataset_requires_reviewed_bbox_plan(input_ref.dataset)
            and clip_plan_path is None):
        raise SystemExit(
            "this governed dataset requires --clip_plan, "
            "--expected_clip_plan_digest, and --farfield_root for exact-bbox "
            "clipping")
    clip_plan = None
    clip_plan_digest = None
    clip_plan_sources = None
    if clip_plan_path is not None:
        try:
            clip_plan, clip_plan_digest = load_clip_plan(
                Path(clip_plan_path), expected_clip_plan_digest,
                resolved_bbox, output_dataset=input_ref.dataset)
            clip_plan_sources = verify_clip_plan_sources(
                clip_plan, Path(farfield_root))
        except ValueError as exc:
            raise SystemExit(f"invalid --clip_plan: {exc}") from exc

    gdf = schema.read_frame(input_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        raise SystemExit(
            "input catalog CRS must be exactly WGS84 (EPSG:4326); refusing "
            "to apply a WGS84 clip bbox to another CRS")
    print(f"{input_path.name}: {schema.summarize(gdf)}")

    tags = far_field_tag_records(gdf)
    areas = footprint_area_m2(gdf)
    masks = evaluate_rules(tags, areas, min_building_area_m2,
                           min_building_levels)
    if clip_km is not None:
        if clip_center_lat is None or clip_center_lon is None:
            raise SystemExit("--clip_km needs --clip_center_lat and "
                             "--clip_center_lon; an implicit centre would "
                             "make the catalog impossible to reproduce")
        masks["outside_clip_box"] = ~clip_mask(gdf, clip_center_lat,
                                               clip_center_lon, clip_km)
    if resolved_bbox is not None:
        masks["outside_clip_bbox"] = ~clip_bbox_mask(gdf, resolved_bbox)

    dropped = np.zeros(len(gdf), dtype=bool)
    for mask in masks.values():
        dropped |= mask
    print(f"\n{'rule':26s} {'drops':>8s} {'only-this':>10s}")
    for name, mask in masks.items():
        others = np.zeros(len(gdf), dtype=bool)
        for other_name, other in masks.items():
            if other_name != name:
                others |= other
        print(f"{name:26s} {int(mask.sum()):8d} "
              f"{int((mask & ~others).sum()):10d}")
    kept = ~dropped
    print(f"{'TOTAL':26s} {int(dropped.sum()):8d}")
    print(f"\nsurviving: {int(kept.sum())} of {len(gdf)} "
          f"({100 * kept.sum() / len(gdf):.1f}%)")

    by_source = Counter(gdf["landmark_type"][kept])
    print(f"by source: {dict(by_source)}")

    guard_refs = []
    positive_set_sha256 = None
    n_positive_signatures = 0
    n_lost_positive_signatures = 0
    n_matched_signatures = 0
    n_lost_matched_signatures = 0
    n_absent_matched_signatures = 0
    lost_positives = []
    if positive_set_path is not None:
        positive_set_path = Path(positive_set_path)
        try:
            positive_set, _, _ = load_positive_set(positive_set_path)
        except (artifact.ArtifactError, PositiveSetError) as exc:
            raise SystemExit(
                f"invalid positive set {positive_set_path}: {exc}") from exc
        positive_matching_ref, _ = positive_set_identity(
            positive_set, positive_set_path, input_ref)
        guard_refs.append(positive_matching_ref)
        positive_set_sha256 = artifact.sha256_file(positive_set_path)
        surviving = {signature_id(tags[i]) for i in np.flatnonzero(kept)}
        score, lost_positives = recall(positive_set, surviving)
        n_positive_signatures = len({p["signature_id"]
                                     for p in positive_set["positives"]})
        n_lost_positive_signatures = len({
            p["signature_id"] for p in lost_positives
        })
        print(f"\nRECALL on final matching positives: {score:.4f} "
              f"({n_positive_signatures - n_lost_positive_signatures}/"
              f"{n_positive_signatures} "
              "signatures survive)")
        if lost_positives:
            print("LOST labelled matches -- fix the rule, do not accept "
                  "this:")
            for record in lost_positives[:15]:
                print(f"  {record['tracklet_id']} [{record['match_type']}] "
                      f"{record['signature_display'][:88]} "
                      f"({record['signature_id'][:19]}...)")
        if lost_positives and not allow_recall_loss:
            raise SystemExit(
                f"\nrefusing to write: {n_lost_positive_signatures} "
                f"labelled positive signature(s) would be dropped. Fix the "
                f"rule, or pass --allow_recall_loss if the loss is intended.")

    lost_matches = []
    if matched_from:
        matched, matching_refs = matched_signatures(
            matched_from, confidence_floor, input_ref)
        guard_refs.extend(matching_refs)
        n_matched_signatures = len(matched)
        lost_matches, absent_matches = report_matched_recall(
            matched, tags, kept, masks)
        n_lost_matched_signatures = len(lost_matches)
        n_absent_matched_signatures = len(absent_matches)
        absent_fraction = (len(absent_matches) / len(matched)
                           if matched else 0.0)
        if absent_fraction > 0.40 and not allow_absent_matched_signatures:
            raise SystemExit(
                f"\nrefusing to write: {len(absent_matches)}/{len(matched)} "
                f"matched signatures ({100 * absent_fraction:.1f}%) are "
                "absent from the input catalog. This evidence does not "
                "provide catalog recall coverage; pass "
                "--allow_absent_matched_signatures only after confirming "
                "the mismatch is intentional.")
        if lost_matches and not allow_recall_loss:
            raise SystemExit(
                f"\nrefusing to write: {len(lost_matches)} signature(s) that "
                f"a completed matching artifact chose would be dropped. "
                "Fix the rule, "
                f"or pass --allow_recall_loss if the loss is intended.")

    if dry_run:
        print("\n(dry run: nothing written)")
        return gdf

    out = gdf[kept].reset_index(drop=True)
    unique_guard_refs = []
    for reference in guard_refs:
        if reference not in unique_guard_refs:
            unique_guard_refs.append(reference)
    config = {
        "min_building_area_m2": min_building_area_m2,
        "min_building_levels": min_building_levels,
        "clip_mode": ("bbox_wsen" if resolved_bbox is not None
                      else "metric_square" if clip_km is not None else None),
        "clip_km": clip_km,
        "clip_center_lat": clip_center_lat,
        "clip_center_lon": clip_center_lon,
        "clip_bbox_wsen": (list(resolved_bbox)
                           if resolved_bbox is not None else None),
        "clip_plan": clip_plan,
        "clip_plan_digest": clip_plan_digest,
        "clip_plan_source_verification": clip_plan_sources,
        "rows_in": int(len(gdf)),
        "rows_out": int(kept.sum()),
        "rule_fingerprint": rule_fingerprint(min_building_area_m2,
                                             min_building_levels,
                                             clip_km=clip_km,
                                             clip_center_lat=clip_center_lat,
                                             clip_center_lon=clip_center_lon,
                                             clip_bbox_wsen=resolved_bbox,
                                             clip_plan_digest=
                                             clip_plan_digest),
        "drops_per_rule": {name: int(mask.sum())
                           for name, mask in masks.items()},
        "recall_guards": {
            "positive_set_sha256": positive_set_sha256,
            "matching_artifacts": [
                reference.to_dict() for reference in unique_guard_refs
            ],
            "confidence_floor": confidence_floor,
            "allow_recall_loss": allow_recall_loss,
            "allow_absent_matched_signatures":
                allow_absent_matched_signatures,
            "n_positive_signatures": n_positive_signatures,
            "n_lost_positive_signatures": n_lost_positive_signatures,
            "n_matched_signatures": n_matched_signatures,
            "n_lost_matched_signatures": n_lost_matched_signatures,
            "n_absent_matched_signatures": n_absent_matched_signatures,
        },
    }
    with publication.published_artifact(
            output_dir,
            kind=paths_lib.CATALOGS,
            dataset=input_ref.dataset,
            version=output_dir.name,
            generator="farfield/dataset_tools/trim_catalog.py",
            git_commit=provenance.git_commit(),
            upstreams=(input_ref, *unique_guard_refs),
            config=config,
            declared_outputs=("catalog.feather",)) as builder:
        out.to_feather(builder.output_path("catalog.feather"))
        if clip_plan is not None:
            try:
                final_sources = verify_clip_plan_sources(
                    clip_plan, Path(farfield_root))
            except ValueError as exc:
                raise SystemExit(
                    "canonical GPS source changed before publication: "
                    f"{exc}") from exc
            if final_sources != clip_plan_sources:
                raise SystemExit(
                    "canonical GPS source identity changed before "
                    "publication")
    print(f"\nWrote {output_dir}")
    print(f"Full catalog left untouched at {input_catalog_dir}")
    return out


def cli(argv=None) -> int:
    """Command-line entry point; :func:`main` remains the typed Python API."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_catalog_dir", required=True, type=Path,
                        help="published CATALOGS artifact to trim")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="new immutable CATALOGS artifact directory")
    parser.add_argument("--positive_set", type=Path, default=None,
                        help="schema-v2 landmark_positive_set.py JSON; lost "
                             "positives refuse the write")
    # Thresholds shape the result and therefore are required.
    parser.add_argument("--min_building_area_m2", type=float, required=True,
                        help="untagged buildings smaller than this are dropped")
    parser.add_argument("--min_building_levels", type=float, required=True,
                        help="untagged buildings shorter than this are dropped")
    parser.add_argument("--matched_from", type=Path, action="append",
                        default=[], metavar="LANDMARK_MATCHES_DIR",
                        help="a completed typed matching artifact to guard "
                             "against (repeatable): no signature it already "
                             "matched may be dropped")
    parser.add_argument("--confidence_floor", type=float, required=True,
                        help="ignore matches below this confidence when "
                             "building the recall guard's expectation")
    parser.add_argument("--allow_recall_loss", action="store_true",
                        help="write anyway when the guard finds losses")
    parser.add_argument(
        "--allow_absent_matched_signatures", action="store_true",
        help="continue when more than 40%% of matched signatures are absent "
             "from the input catalog (separate from --allow_recall_loss)")
    clip_group = parser.add_mutually_exclusive_group()
    clip_group.add_argument(
        "--clip_km", type=float, default=None,
        help="keep only rows inside a square box this many km on a side "
             "(a prior extent, not a corridor); needs --clip_center_lat/lon")
    clip_group.add_argument(
        "--clip_bbox_wsen", type=float, nargs=4, default=None,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="keep rows whose representative point lies inside this exact "
             "inclusive WGS84 rectangle")
    parser.add_argument("--clip_center_lat", type=float, default=None)
    parser.add_argument("--clip_center_lon", type=float, default=None)
    parser.add_argument(
        "--clip_plan", type=Path, default=None,
        help="reviewed farfield.catalog_clip_plan/v1 JSON binding a bbox to "
             "root-derived canonical GPS sources and its buffer/area policy")
    parser.add_argument(
        "--farfield_root", type=Path, default=None,
        help="canonical farfield root; required with --clip_plan so dataset "
             "paths cannot be supplied by the plan")
    parser.add_argument(
        "--expected_clip_plan_digest", default=None,
        help="required canonical JSON SHA-256 when --clip_plan is supplied")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    main(args.input_catalog_dir, args.output_dir, args.positive_set,
         args.min_building_area_m2, args.min_building_levels, args.dry_run,
         matched_from=args.matched_from,
         confidence_floor=args.confidence_floor,
         allow_recall_loss=args.allow_recall_loss,
         allow_absent_matched_signatures=
             args.allow_absent_matched_signatures,
         clip_km=args.clip_km, clip_center_lat=args.clip_center_lat,
         clip_center_lon=args.clip_center_lon,
         clip_bbox_wsen=args.clip_bbox_wsen,
         clip_plan_path=args.clip_plan,
         expected_clip_plan_digest=args.expected_clip_plan_digest,
         farfield_root=args.farfield_root)
    return 0


if __name__ == "__main__":
    cli()
