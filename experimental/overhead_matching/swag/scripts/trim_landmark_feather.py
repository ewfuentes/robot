"""Trim a landmark feather to entries a far-field observer could plausibly match.

The harbor feather is 156 k landmarks, of which ~47 % are untagged
`building=yes` footprints and ~30 % is street furniture -- benches, crossings,
waste baskets, CCTV cameras -- that nothing observed from a boat can ever
correspond to. This drops those rows and writes a smaller feather.

**The full table is never modified.** Trimming writes a new file; the input
stays as the fallback to consult whenever a landmark turns out to be missing.

**Rules are judged by the recall guard, not by taste.** Every rule reports how
many rows only it removes and what it costs against two references: the
pairing-run positive set (`--positive_set`, see landmark_positive_set.py) and,
stronger, the signatures a real matching run already chose
(`--matched_from <m9 run dir>`, read from `matching/matches.json`). A rule that
drops either is a bug in the rule, and `--matched_from` refuses to write rather
than report it. That guard has earned its place: it killed two proposed rules
that looked obviously right -- dropping signatures that span many map rows
(boston_harbor_leg1 matched `man_made=pier` across 375 of them) and dropping by
physical extent (matched features include islands ~1 km across).

**A catalog is versioned, never overwritten.** Every number anyone has quoted was
computed against the file that is already there, so writing over it silently
changes the past; the tool refuses and asks for a new version instead
(`--force` if you truly mean to replace it). Each output gets a
`<name>.provenance.json` beside it recording the input and its sha256, every
argument, a fingerprint of the rule sets themselves (they live in code, so the
arguments alone would not pin them down) and a `reproduce` command line. That
file exists because "is this catalog stale?" cost an afternoon to answer: the
shipped `v1_trimmed` tables looked stale against the current rules and were in
fact an exact match -- the analysis had passed building thresholds by hand.

**`--clip_km` bounds the prior's extent**, for regional extracts that reach far
past anything a vehicle could see. It is a prior, not a corridor: charles_river's
1 km sail sits inside a 25 x 25 km box, which is still larger than the
22.9 x 20.9 km harbour prior the method was validated on. The centre must be
given explicitly, because an implicit one cannot be reproduced.

Note the scope this deliberately does not have: `harbor_catalog.py` argues
against class filtering for the *filter's* catalog, because spatial gating plus
uniqueness weighting already handle catalog size there. This produces a
separate artifact for consumers that want a smaller table; point the catalog at
whichever file is appropriate.

Tag vocabulary is `harbor_catalog.prune_far_field_tags`, so a landmark's surviving
tags here are exactly the ones the matcher would see.

Example:
    bazel run //experimental/overhead_matching/swag/scripts:trim_landmark_feather -- \\
        --input  /data/.../landmarks/harbor_osm_enc_v1.feather \\
        --output /data/.../landmarks/harbor_osm_enc_trimmed_v1.feather \\
        --positive_set /data/.../landmarks/positive_set_r003_leg1.json
"""

import argparse
import hashlib
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import geopandas as gpd

from experimental.overhead_matching.swag.data import landmark_schema
import numpy as np
import pandas as pd

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    harbor_catalog as hc,
)
from experimental.overhead_matching.swag.scripts.landmark_positive_set import (
    format_signature,
    recall,
)

# Structural keys that say what a thing IS, in the harbor vocabulary. A
# landmark with none of these carries no class a distant observer could name.
STRUCTURAL_KEYS = frozenset({
    "seamark:type", "object_class", "man_made", "historic", "place", "natural",
    "building", "landuse", "leisure", "amenity", "tourism", "power", "bridge",
    "aeroway", "railway", "waterway", "military", "industrial",
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
# the tag is. Bunker Hill Monument, a 67 m granite obelisk, carries nothing but
# a name and `tourism=information`.
#
# (key, value) pairs rather than whole keys, because `amenity` also carries
# ferry_terminal and `natural` also carries cliff.
HARD_UNOBSERVABLE_TAGS = frozenset({
    ("natural", "tree"), ("natural", "tree_row"), ("natural", "shrub"),
    ("natural", "scrub"), ("natural", "grassland"), ("natural", "heath"),
    ("amenity", "bench"), ("amenity", "waste_basket"), ("amenity", "recycling"),
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
    ("power", "pole"), ("power", "minor_line"), ("power", "cable_distribution_cabinet"),
    # Track, transit stops and station furniture. Rails sit at ground level and
    # stations here are MBTA subway stops carrying no building tag, so none of
    # it has a silhouette from the water -- and being named ("Dorchester
    # Branch", "Mattapan") does not change that. Two things are deliberately
    # NOT in this list and survive: `bridge` is structural, so all 120 rail
    # bridges keep their bridge tag, and yard/depot/workshop infrastructure is
    # kept below. A large terminal building carries its own
    # `building=train_station` row, which LANDMARK_BUILDING_VALUES admits.
    *[("railway", v) for v in (
        "rail", "switch", "subway", "stop", "light_rail", "platform",
        "platform_edge", "abandoned", "subway_entrance", "station", "crossing",
        "buffer_stop", "disused", "level_crossing", "signal",
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
    # thing that is not visible from the water, so a name must not rescue them.
    *[("amenity", v) for v in (
        "restaurant", "fast_food", "cafe", "bar", "pub", "ice_cream",
        "biergarten", "food_court", "nightclub", "bank", "bureau_de_change",
        "pharmacy", "dentist", "doctors", "clinic", "veterinary", "childcare",
        "kindergarten", "post_office", "payment_terminal", "car_rental",
        "car_wash", "driving_school", "hairdresser", "laundry", "dry_cleaning",
        "copyshop", "internet_cafe", "bicycle_repair_station", "loading_dock",
        "fuel", "taxi", "shower", "fountain", "waste_disposal",
        "waste_transfer_station", "studio", "money_transfer", "gym",
        "coworking_space", "language_school", "music_school", "dojo")],
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
# outranks them. Zoning classes live here rather than in HARD because the label
# often sits on the thing itself -- `landuse=residential; name=Harbor Towers` is
# a labelled positive, and so is `landuse=industrial; industrial=port;
# name=Paul W. Conley Terminal`.
SOFT_UNOBSERVABLE_TAGS = frozenset({
    ("tourism", "information"), ("tourism", "artwork"),
    ("historic", "memorial"), ("man_made", "flagpole"),
    *[("landuse", v) for v in (
        "residential", "commercial", "retail", "construction", "brownfield",
        "cemetery", "religious", "government", "education", "conservation")],
})

# `landuse=railway` is how a railyard is drawn (Southampton Street Yard, Codman
# Yard, MBTA Cabot Yard) -- a large open expanse of track and rolling stock that
# reads from the water, so it stays structural and is kept even when unnamed.
# The yard's individual rails are still dropped by the HARD list above.

# `highway` is a key rather than a value list: every highway=* value is a road,
# a path, or street furniture, none of which a vessel can pick out -- and being
# named does not change that. A highway feature that is ALSO structural
# survives, because the rule looks at what is left once these are removed;
# `bridge` is the important case and is deliberately NOT listed here, so
# `bridge=yes; highway=footway` keeps its bridge.
HARD_UNOBSERVABLE_KEYS = frozenset({"highway", "barrier", "crossing",
                                    "traffic_calming", "traffic_sign",
                                    "entrance", "footway", "cycleway"})

# A proper noun is the strongest far-field signal there is -- the extractor is
# asked for names, and the correspondence model scores a name-specific cross
# feature -- so a named feature is not dropped merely for lacking a structural
# class. It cannot rescue a HARD-unobservable one.
NAME_KEYS = frozenset({"name", "alt_name", "official_name", "short_name",
                       "loc_name", "old_name"})

# Values that make a building conspicuous from the water on their own.
LANDMARK_BUILDING_VALUES = frozenset({
    "cathedral", "church", "chapel", "tower", "lighthouse", "stadium",
    "hangar", "hospital", "university", "hotel", "civic", "industrial",
    "commercial", "government", "college", "train_station", "transportation",
    "warehouse", "silo", "storage_tank", "digester", "grandstand",
})

METERS_PER_DEG_LAT = 110574.0
METERS_PER_DEG_LON_EQUATOR = 111320.0


def far_field_tag_columns(columns) -> list[str]:
    """Columns `prune_far_field_tags` would keep, decided by key name alone.

    Pre-selecting columns keeps the row dicts ~50 wide instead of ~1470, which
    is the difference between a 0.5 GB working set and a 5.5 GB one. Only needed
    for the legacy wide layout; the dict schema is already narrow.
    """
    keep = []
    for column in columns:
        if column in hc.NON_TAG_COLUMNS:
            continue
        if any(column.startswith(p) for p in hc.FAR_FIELD_DROP_PREFIXES):
            continue
        if (column in hc.FAR_FIELD_KEEP_KEYS
                or any(column.startswith(p) for p in hc.FAR_FIELD_KEEP_PREFIXES)):
            keep.append(column)
    return keep


def far_field_tag_records(gdf: gpd.GeoDataFrame) -> list[dict]:
    """Per-row pruned harbor tags, matching harbor_catalog.load_catalog.

    Under the dict schema there are no tag columns to pre-select, and none is
    needed: the row dicts are already only as wide as the tags that exist.
    """
    if landmark_schema.is_dict_schema(gdf):
        return [hc.prune_far_field_tags(record)
                for record in landmark_schema.tag_dicts(gdf)]
    columns = far_field_tag_columns(gdf.columns)
    return [hc.prune_far_field_tags(record)
            for record in gdf[columns].to_dict(orient="records")]


def footprint_area_m2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Approximate footprint area; 0 for points and lines."""
    latitude = gdf.geometry.representative_point().y.to_numpy()
    # One degree of longitude shrinks by cos(lat); one of latitude does not.
    scale = (METERS_PER_DEG_LAT * METERS_PER_DEG_LON_EQUATOR
             * np.cos(np.radians(latitude)))
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


def evaluate_rules(tags: list[dict], areas: np.ndarray,
                   min_building_area_m2: float,
                   min_building_levels: float) -> dict[str, np.ndarray]:
    """Boolean drop-mask per rule.

    Every rule votes on every row, independently of the others -- the same
    convention as landmark_filtering.heuristic_filters -- so the run's
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
        rescued_by_name = (not hard_blocked
                           and any(k in NAME_KEYS for k in observable))
        unobservable[i] = not (has_structural or rescued_by_name)

        # A building survives on its own merits: another structural tag, a
        # name, a landmark use, real height, or a footprint readable at range.
        building = observable.get("building")
        if building is None:
            continue
        if any(k in STRUCTURAL_KEYS and k != "building" for k in observable):
            continue
        if "name" in tag or building in LANDMARK_BUILDING_VALUES:
            continue
        height = max(_numeric(tag.get("height")),
                     _numeric(tag.get("building:levels")) * 3.5)
        if height >= min_building_levels * 3.5:
            continue
        generic_building[i] = areas[i] < min_building_area_m2

    return {"no_far_field_tags": no_tags,
            "unobservable_only": unobservable,
            "generic_small_building": generic_building}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reproduce_command(arguments: dict) -> str:
    """The exact command line that rebuilds this catalog."""
    parts = ["bazel run //experimental/overhead_matching/swag/scripts:"
             "trim_landmark_feather --"]
    for key, value in arguments.items():
        if value is None or value is False or value == []:
            continue
        if value is True:
            parts.append(f"--{key}")
        elif isinstance(value, list):
            parts.extend(f"--{key} {item}" for item in value)
        else:
            parts.append(f"--{key} {value}")
    return " ".join(parts)


def clip_mask(gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float,
              box_km: float) -> np.ndarray:
    """True where a row lies inside a `box_km` square centred on the point.

    Deliberately a *prior extent*, not a corridor: the whole-map experiment
    needs a prior far larger than the trajectory (charles's sail spans 0.8 x 1.1
    km inside a 25 x 25 km box, which is itself larger than the 22.9 x 20.9 km
    harbour prior the method was validated on). Its only job is to stop a
    regional OSM extract from reaching tens of kilometres past anything the
    vehicle could see.
    """
    point = gdf.geometry.representative_point()
    half = box_km * 1000.0 / 2.0
    meters_per_deg_lon = METERS_PER_DEG_LON_EQUATOR * math.cos(
        math.radians(center_lat))
    east = (np.asarray(point.x) - center_lon) * meters_per_deg_lon
    north = (np.asarray(point.y) - center_lat) * METERS_PER_DEG_LAT
    return (np.abs(east) <= half) & (np.abs(north) <= half)


def rule_fingerprint(min_building_area_m2: float,
                     min_building_levels: float) -> str:
    """Short hash of everything that decides what this trimmer keeps.

    Recorded next to each output so "which rules built this catalog?" is
    answerable from the file. Worth having: `v1_trimmed` files that looked stale
    against the current rules turned out to match them exactly -- the analysis
    had passed thresholds by hand -- and this is what would have settled it in
    one line instead of an afternoon.
    """
    payload = json.dumps({
        "hard_keys": sorted(HARD_UNOBSERVABLE_KEYS),
        "hard_tags": sorted(map(list, HARD_UNOBSERVABLE_TAGS)),
        "soft_tags": sorted(map(list, SOFT_UNOBSERVABLE_TAGS)),
        "structural_keys": sorted(STRUCTURAL_KEYS),
        "name_keys": sorted(NAME_KEYS),
        "landmark_building_values": sorted(LANDMARK_BUILDING_VALUES),
        "keep_keys": sorted(hc.FAR_FIELD_KEEP_KEYS),
        "keep_prefixes": sorted(hc.FAR_FIELD_KEEP_PREFIXES),
        "drop_prefixes": sorted(hc.FAR_FIELD_DROP_PREFIXES),
        "min_building_area_m2": min_building_area_m2,
        "min_building_levels": min_building_levels,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def matched_signatures(sources: list[Path], confidence_floor: float) -> dict:
    """signature -> [(run, tracklet, confidence, match_type)] from M9 output.

    The positive-set guard asks whether a rule drops a *labelled* match. This
    asks the stronger question -- whether it drops something a matcher already
    chose on a real run -- which is the check that caught two proposed rules
    that looked obviously right: dropping signatures that span many rows (leg1
    matched `man_made=pier` across 375 of them) and dropping by physical extent
    (matched features include islands ~1 km across).
    """
    found = {}
    for source in sources:
        path = source if source.is_file() else source / "matching" / "matches.json"
        if not path.exists():
            raise SystemExit(f"no matches.json at {path}; pass a run dir that "
                             f"has been through m9, or the file itself")
        label = path.parent.parent.name
        for tracklet, record in json.loads(path.read_text()).items():
            for match in record.get("matches", []):
                if match.get("confidence", 0.0) < confidence_floor:
                    continue
                found.setdefault(match["signature"], []).append(
                    (label, tracklet, match["confidence"],
                     match.get("match_type", "?")))
    return found


def report_matched_recall(matched: dict, tags: list[dict], kept: np.ndarray,
                          masks: dict) -> list[str]:
    """Print how the rules treat already-matched signatures; return the lost.

    Only signatures this table actually contains can be *lost by a rule*. A
    signature the matcher chose on some other region is simply not here, which
    is not this tool's business -- counting it as a loss would make the guard
    unusable with any run but one, so those are reported separately and do not
    block a write.
    """
    by_signature = {}
    for i, tag in enumerate(tags):
        by_signature.setdefault(format_signature(tag), []).append(i)
    surviving = {format_signature(tags[i]) for i in np.flatnonzero(kept)}
    absent = sorted(s for s in matched if s not in by_signature)
    lost = sorted(set(matched) - surviving - set(absent))
    print(f"\nRECALL on {len(matched)} signatures matched by real runs: "
          f"{len(matched) - len(lost) - len(absent)}/"
          f"{len(matched) - len(absent)} of the ones this table holds survive"
          + (f"; {len(absent)} are not in this table at all (different region "
             f"or catalog vintage)" if absent else ""))
    if not lost:
        return lost
    print("LOST signatures a matcher already chose -- fix the rule:")
    for signature in lost[:15]:
        # Which rule is responsible: the rules that fired on the rows carrying
        # this signature.
        blame = sorted({name for name, mask in masks.items()
                        for i in by_signature[signature] if mask[i]})
        run, tracklet, confidence, kind = matched[signature][0]
        print(f"  [{','.join(blame)}] {run}/{tracklet} {kind} "
              f"{confidence:.2f}: {signature[:70]}")
    return lost


def main(input_path: Path, output_path: Path, positive_set_path: Path | None,
         min_building_area_m2: float, min_building_levels: float,
         dry_run: bool, matched_from: list[Path] | None = None,
         confidence_floor: float = 0.5, allow_recall_loss: bool = False,
         force: bool = False, clip_km: float | None = None,
         clip_center_lat: float | None = None,
         clip_center_lon: float | None = None) -> gpd.GeoDataFrame:
    output_path = output_path.with_suffix(".feather")
    if output_path.exists() and not (dry_run or force):
        raise SystemExit(
            f"{output_path} already exists. A catalog is part of the problem "
            f"definition -- every past number was computed against it -- so it "
            f"is versioned, not overwritten. Write a new version (v2_trimmed, "
            f"v3_trimmed) or pass --force if you really mean to replace it.")
    gdf = gpd.read_feather(input_path)
    print(f"{input_path.name}: {len(gdf)} landmarks, {len(gdf.columns)} columns")

    tags = far_field_tag_records(gdf)
    areas = footprint_area_m2(gdf)
    masks = evaluate_rules(tags, areas, min_building_area_m2, min_building_levels)
    if clip_km is not None:
        if clip_center_lat is None or clip_center_lon is None:
            raise SystemExit("--clip_km needs --clip_center_lat and "
                             "--clip_center_lon; an implicit centre would make "
                             "the catalog impossible to reproduce")
        masks["outside_clip_box"] = ~clip_mask(gdf, clip_center_lat,
                                               clip_center_lon, clip_km)

    dropped = np.zeros(len(gdf), dtype=bool)
    for mask in masks.values():
        dropped |= mask
    print(f"\n{'rule':26s} {'drops':>8s} {'only-this':>10s}")
    for name, mask in masks.items():
        others = np.zeros(len(gdf), dtype=bool)
        for other_name, other in masks.items():
            if other_name != name:
                others |= other
        print(f"{name:26s} {int(mask.sum()):8d} {int((mask & ~others).sum()):10d}")
    kept = ~dropped
    print(f"{'TOTAL':26s} {int(dropped.sum()):8d}")
    print(f"\nsurviving: {int(kept.sum())} of {len(gdf)} "
          f"({100 * kept.sum() / len(gdf):.1f}%)")

    by_source = Counter(gdf["landmark_type"][kept])
    print(f"by source: {dict(by_source)}")

    if positive_set_path is not None:
        with open(positive_set_path) as f:
            positive_set = json.load(f)
        surviving = {format_signature(tags[i])
                     for i in np.flatnonzero(kept)}
        score, lost = recall(positive_set, surviving)
        n_signatures = len({p["signature"] for p in positive_set["positives"]})
        print(f"\nRECALL on pairing positives: {score:.4f} "
              f"({n_signatures - len({p['signature'] for p in lost})}"
              f"/{n_signatures} signatures survive)")
        if lost:
            print("LOST labelled matches -- fix the rule, do not accept this:")
            for record in lost[:15]:
                print(f"  {record['tracklet']} [{record['match_type']}] "
                      f"{record['signature'][:88]}")

    lost_matches = []
    if matched_from:
        matched = matched_signatures(matched_from, confidence_floor)
        lost_matches = report_matched_recall(matched, tags, kept, masks)
        if lost_matches and not allow_recall_loss:
            raise SystemExit(
                f"\nrefusing to write: {len(lost_matches)} signature(s) that a "
                f"real matching run chose would be dropped. Fix the rule, or "
                f"pass --allow_recall_loss if the loss is intended.")

    if dry_run:
        print("\n(dry run: nothing written)")
        return gdf

    out = gdf[kept].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_feather(output_path)
    provenance = output_path.with_suffix(".provenance.json")
    arguments = {
        "input": str(input_path),
        "output": str(output_path),
        "positive_set": str(positive_set_path) if positive_set_path else None,
        "min_building_area_m2": min_building_area_m2,
        "min_building_levels": min_building_levels,
        "clip_km": clip_km,
        "clip_center_lat": clip_center_lat,
        "clip_center_lon": clip_center_lon,
        "matched_from": [str(p) for p in (matched_from or [])],
        "confidence_floor": confidence_floor,
        "allow_recall_loss": allow_recall_loss,
    }
    # Everything needed to rebuild this file byte for byte: the arguments, the
    # exact input, and a fingerprint of the rules themselves (which live in
    # code, so the arguments alone would not pin them down).
    provenance.write_text(json.dumps({
        "tool": "swag/scripts/trim_landmark_feather.py",
        "arguments": arguments,
        "reproduce": reproduce_command(arguments),
        "input_sha256": sha256_of(input_path),
        "rows_in": int(len(gdf)),
        "rows_out": int(kept.sum()),
        "rule_fingerprint": rule_fingerprint(min_building_area_m2,
                                             min_building_levels),
        "drops_per_rule": {name: int(mask.sum())
                           for name, mask in masks.items()},
        "recall_guard": {
            "matched_from": [str(p) for p in (matched_from or [])],
            "confidence_floor": confidence_floor,
            "lost_signatures": lost_matches,
        },
    }, indent=1))
    print(f"\nWrote {output_path}")
    print(f"       {provenance}")
    print(f"Full table left untouched at {input_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive_set", type=Path, default=None,
                        help="landmark_positive_set.py JSON; reports recall")
    parser.add_argument("--min_building_area_m2", type=float, default=2000.0,
                        help="untagged buildings smaller than this are dropped")
    parser.add_argument("--min_building_levels", type=float, default=6.0,
                        help="untagged buildings shorter than this are dropped")
    parser.add_argument("--matched_from", type=Path, action="append",
                        default=[], metavar="RUN_DIR_OR_MATCHES_JSON",
                        help="an m9 run to guard against (repeatable): no "
                             "signature it already matched may be dropped")
    parser.add_argument("--confidence_floor", type=float, default=0.5,
                        help="ignore matches below this confidence (0.5)")
    parser.add_argument("--allow_recall_loss", action="store_true",
                        help="write anyway when the guard finds losses")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing catalog instead of "
                             "versioning alongside it")
    parser.add_argument("--clip_km", type=float, default=None,
                        help="keep only rows inside a square box this many km "
                             "on a side (a prior extent, not a corridor); "
                             "needs --clip_center_lat/lon")
    parser.add_argument("--clip_center_lat", type=float, default=None)
    parser.add_argument("--clip_center_lon", type=float, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(args.input, args.output, args.positive_set,
         args.min_building_area_m2, args.min_building_levels, args.dry_run,
         matched_from=args.matched_from,
         confidence_floor=args.confidence_floor,
         allow_recall_loss=args.allow_recall_loss, force=args.force,
         clip_km=args.clip_km, clip_center_lat=args.clip_center_lat,
         clip_center_lon=args.clip_center_lon)
