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

**`--clip_km` bounds the prior's extent**, for regional extracts that reach
far past anything a vehicle could see. It is a prior, not a corridor:
charles_river's 1 km sail sits inside a 25 x 25 km box, which is still larger
than the 22.9 x 20.9 km harbour prior the method was validated on. The centre
must be given explicitly, because an implicit one cannot be reproduced.

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
import warnings
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import publication
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.catalog import catalog
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools.landmark_positive_set import (  # noqa: E501
    PositiveSetError,
    load_positive_set,
    open_catalog_artifact,
    open_matching_artifact,
    recall,
    signature_id,
    validate_positive_set,
)

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


def rule_fingerprint(min_building_area_m2: float,
                     min_building_levels: float) -> str:
    """Short hash of everything that decides what this trimmer keeps.

    Recorded next to each output so "which rules built this catalog?" is
    answerable from the file. Worth having: `v1_trimmed` files that looked
    stale against the current rules turned out to match them exactly -- the
    analysis had passed thresholds by hand -- and this is what would have
    settled it in one line instead of an afternoon.
    """
    payload = json.dumps({
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
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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
         clip_center_lon=None) -> gpd.GeoDataFrame:
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
    gdf = schema.read_frame(input_path)
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
        "clip_km": clip_km,
        "clip_center_lat": clip_center_lat,
        "clip_center_lon": clip_center_lon,
        "rows_in": int(len(gdf)),
        "rows_out": int(kept.sum()),
        "rule_fingerprint": rule_fingerprint(min_building_area_m2,
                                             min_building_levels),
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
        help="continue when more than 40% of matched signatures are absent "
             "from the input catalog (separate from --allow_recall_loss)")
    parser.add_argument("--clip_km", type=float, default=None,
                        help="keep only rows inside a square box this many "
                             "km on a side (a prior extent, not a corridor); "
                             "needs --clip_center_lat/lon")
    parser.add_argument("--clip_center_lat", type=float, default=None)
    parser.add_argument("--clip_center_lon", type=float, default=None)
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
         clip_center_lon=args.clip_center_lon)
    return 0


if __name__ == "__main__":
    cli()
