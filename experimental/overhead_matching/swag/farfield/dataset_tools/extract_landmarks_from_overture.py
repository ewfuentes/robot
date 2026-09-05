"""Extract landmarks from an Overture Maps Places GeoParquet into the catalog
Feather format.

Overture Places (https://docs.overturemaps.org/guides/places/) is a named-POI
layer built from Meta, Microsoft, Foursquare and AllThePlaces. It contains no
OpenStreetMap data, so it adds places OSM lacks; `add_catalog_source` then
drops the ones OSM already names. Every record is mapped onto the OSM-style
far-field tag vocabulary by its taxonomy hierarchy, so the output flows through
`catalog.schema.read_frame` and `trim_catalog` exactly like an OSM Feather and
the same trim rules decide what a vessel can see.

Download (no account or key; pin the release):
    overturemaps releases latest
    overturemaps download --type=place --bbox=W,S,E,N -r <release> \\
        -f geoparquet -o overture_places.parquet

Mapping is by the record's taxonomy hierarchy (root -> leaf). A leaf listed in
LEAF_TAGS wins; otherwise its root's default in ROOT_TAGS applies and the leaf
is counted under `leaf_fallbacks` in the provenance sidecar so the table can be
extended from what the data actually contains. Records with no taxonomy or an
unknown root are dropped and counted, so a new Overture taxonomy surfaces
instead of silently shrinking the catalog.

Tags per record: `name` (names.primary), `name:<lang>` (names.common),
`brand`, the mapped OSM-style tags, and `overture:*` provenance (category,
hierarchy, confidence, version, and the per-record `sources` with their
licences -- Places is licensed per source, so they stay with the row). The
`overture:*` keys are outside the far-field keep vocabulary and never reach the
matcher. Rows are written highest confidence first, which is the order
`add_catalog_source` uses to pick the survivor among same-name duplicates.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:extract_landmarks_from_overture -- \\
        --parquet /data/farfield_matching/raw_material/pohang_canal_20210716/overture_places.parquet \\
        --release 2026-08-19.0 \\
        --bbox 129.0925109374265 35.79889417897012 129.6663520625735 36.27511282102988 \\
        --min_confidence 0.5 \\
        --output_path /data/farfield_matching/raw_material/catalog_sources/pohang_canal_04/overture_2026-08-19.0_v1
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq
import shapely

from experimental.overhead_matching.swag.farfield import artifact, provenance
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)

LANDMARK_TYPE = "overture"
COLUMNS = ("id", "geometry", "names", "categories", "taxonomy", "brand",
           "confidence", "sources", "version")

# Default OSM-style tags per taxonomy root (Overture release 2026-08-19.0).
ROOT_TAGS = {
    "food_and_drink": {"amenity": "restaurant"},
    "shopping": {"shop": "yes"},
    "services_and_business": {"office": "yes"},
    "lodging": {"tourism": "guest_house"},
    "education": {"amenity": "school"},
    "cultural_and_historic": {"tourism": "attraction"},
    "sports_and_recreation": {"leisure": "sports_centre"},
    "lifestyle_services": {"shop": "beauty"},
    "community_and_government": {"office": "government"},
    "travel_and_transportation": {"office": "yes"},
    "health_care": {"amenity": "clinic"},
    "arts_and_entertainment": {"amenity": "arts_centre"},
    "geographic_entities": {"place": "locality"},
}

_WORSHIP = {"amenity": "place_of_worship"}


def _leaves(tags: dict, *leaves: str) -> dict:
    return {leaf: tags for leaf in leaves}


# Leaf overrides. Only leaves whose OSM reading differs from the root default.
LEAF_TAGS = {
    # food_and_drink
    **_leaves({"amenity": "cafe"}, "cafe", "coffee_shop", "tea_room",
              "bubble_tea_shop", "dessert_shop", "ice_cream_shop",
              "donut_shop", "cupcake_shop", "frozen_yogurt_shop",
              "bagel_shop", "pie_shop"),
    "bakery": {"shop": "bakery"},
    **_leaves({"amenity": "bar"}, "bar", "beer_bar", "sake_bar", "wine_bar",
              "cocktail_bar", "lounge"),
    "pub": {"amenity": "pub"},
    **_leaves({"amenity": "fast_food"}, "fast_food_restaurant",
              "burger_restaurant", "sandwich_shop", "food_truck_stand"),
    # shopping
    "shopping_mall": {"shop": "mall"},
    "department_store": {"shop": "department_store"},
    **_leaves({"shop": "supermarket"}, "grocery_store",
              "warehouse_club_store", "discount_store"),
    "convenience_store": {"shop": "convenience"},
    **_leaves({"shop": "car"}, "auto_dealer", "motorcycle_dealer",
              "boat_dealer"),
    "farmers_market": {"amenity": "marketplace"},
    "pharmacy": {"amenity": "pharmacy"},
    # services_and_business
    **_leaves({"amenity": "bank"}, "bank_or_credit_union", "bank",
              "credit_union"),
    "post_office": {"amenity": "post_office"},
    **_leaves({"man_made": "works"}, "manufacturer", "metal_fabricator",
              "industrial_equipment_manufacturer", "auto_manufacturer",
              "commercial_industrial", "jewelry_and_watches_manufacturer"),
    **_leaves({"landuse": "farmyard"}, "farm", "agriculture",
              "agricultural_cooperative"),
    "ferry_boat_company": {"amenity": "ferry_terminal"},
    **_leaves({"amenity": "social_facility"}, "retirement_home",
              "assisted_living_facility"),
    "day_care_preschool": {"amenity": "kindergarten"},
    # lodging
    **_leaves({"tourism": "hotel"}, "hotel", "resort", "beach_resort", "inn",
              "service_apartment"),
    "motel": {"tourism": "motel"},
    "hostel": {"tourism": "hostel"},
    **_leaves({"tourism": "camp_site"}, "campground", "rv_park"),
    # education
    "preschool": {"amenity": "kindergarten"},
    **_leaves({"amenity": "university"}, "college_university",
              "campus_building", "medical_school", "nursing_school"),
    "library": {"amenity": "library"},
    **_leaves({"office": "educational_institution"}, "language_school",
              "music_school", "art_school", "tutoring_service",
              "driving_school", "computer_coaching", "educational_service",
              "dance_school"),
    **_leaves({"amenity": "research_institute"}, "medical_research_institute",
              "educational_research_institute"),
    # cultural_and_historic
    "historic_site": {"historic": "yes", "tourism": "attraction"},
    "monument": {"historic": "monument"},
    **_leaves({"historic": "memorial"}, "memorial_site", "memorial"),
    "palace": {"historic": "castle"},
    "cemetery": {"landuse": "cemetery"},
    "lighthouse": {"man_made": "lighthouse"},
    "cultural_center": {"amenity": "community_centre"},
    **_leaves(_WORSHIP, "place_of_worship", "religious_organization"),
    "christian_place_of_worship": {**_WORSHIP, "religion": "christian"},
    "roman_catholic_place_of_worship": {
        **_WORSHIP, "religion": "christian", "denomination": "roman_catholic"},
    "baptist_place_of_worship": {
        **_WORSHIP, "religion": "christian", "denomination": "baptist"},
    "anglican_or_episcopal_place_of_worship": {
        **_WORSHIP, "religion": "christian", "denomination": "anglican"},
    "buddhist_place_of_worship": {**_WORSHIP, "religion": "buddhist"},
    "muslim_place_of_worship": {**_WORSHIP, "religion": "muslim"},
    "hindu_place_of_worship": {**_WORSHIP, "religion": "hindu"},
    # sports_and_recreation
    **_leaves({"leisure": "park"}, "park", "national_park", "state_park"),
    **_leaves({"leisure": "golf_course"}, "golf_course", "driving_range"),
    **_leaves({"leisure": "fitness_centre"}, "gym", "sport_or_fitness_facility",
              "martial_arts_club", "dance_studio", "pilates_studio",
              "yoga_studio", "fitness_trainer", "boot_camp"),
    "swimming_pool": {"leisure": "swimming_pool"},
    "water_park": {"leisure": "water_park"},
    "bowling_alley": {"leisure": "bowling_alley"},
    **_leaves({"leisure": "pitch"}, "soccer_field", "baseball_field",
              "hockey_field", "squash_court", "batting_cage", "skate_park",
              "shooting_range", "rock_climbing_spot"),
    "hiking_trail": {"highway": "path"},
    "playground": {"leisure": "playground"},
    "ice_skating_rink": {"leisure": "ice_rink"},
    **_leaves({"leisure": "horse_riding"}, "horse_riding",
              "equestrian_facility"),
    "indoor_playcenter": {"leisure": "indoor_play"},
    # lifestyle_services
    "veterinarian": {"amenity": "veterinary"},
    "onsen": {"amenity": "public_bath"},
    # community_and_government
    "town_hall": {"amenity": "townhall"},
    "police_station": {"amenity": "police"},
    "fire_station": {"amenity": "fire_station"},
    "courthouse": {"amenity": "courthouse"},
    "military_site": {"military": "base"},
    **_leaves({"amenity": "community_centre"}, "community_center",
              "senior_citizen_service", "youth_organization"),
    "public_plaza": {"place": "square"},
    "public_fountain": {"amenity": "fountain"},
    "public_restroom": {"amenity": "toilets"},
    # travel_and_transportation
    "gas_station": {"amenity": "fuel"},
    **_leaves({"shop": "car_repair"}, "automotive_repair",
              "automotive_service", "tire_dealer_and_repair",
              "auto_body_shop", "motorcycle_repair", "tire_shop",
              "car_window_tinting", "auto_detailing"),
    "car_wash": {"amenity": "car_wash"},
    "train_station": {"railway": "station"},
    "bus_station": {"amenity": "bus_station"},
    "parking": {"amenity": "parking"},
    "airport": {"aeroway": "aerodrome"},
    "airport_terminal": {"aeroway": "terminal"},
    "taxi_service": {"amenity": "taxi"},
    "boat_service": {"shop": "boat"},
    **_leaves({"office": "travel_agent"}, "travel_service", "tour_operator",
              "sightseeing_tour_agency", "luggage_storage"),
    # health_care
    **_leaves({"amenity": "hospital"}, "hospital", "emergency_department"),
    **_leaves({"amenity": "dentist"}, "dental_clinic", "general_dentistry",
              "cosmetic_dentistry", "oral_and_maxillofacial_surgery"),
    "medical_service_organization": {"office": "yes"},
    # arts_and_entertainment
    **_leaves({"tourism": "museum"}, "museum", "history_museum", "art_museum",
              "science_museum", "contemporary_art_museum",
              "community_museum"),
    "art_gallery": {"tourism": "gallery"},
    **_leaves({"leisure": "stadium"}, "stadium_arena", "baseball_stadium",
              "soccer_stadium"),
    "movie_theater": {"amenity": "cinema"},
    **_leaves({"amenity": "theatre"}, "music_venue", "performing_arts_venue",
              "theatre_venue", "auditorium", "event_venue"),
    "exhibition_and_trade_fair_venue": {"amenity": "exhibition_centre"},
    "amusement_park": {"tourism": "theme_park"},
    "zoo": {"tourism": "zoo"},
    "arcade": {"leisure": "amusement_arcade"},
    **_leaves({"amenity": "nightclub"}, "dance_club", "karaoke_venue",
              "nightlife_venue", "adult_entertainment_venue"),
    "sculpture_statue": {"tourism": "artwork"},
    # geographic_entities
    "beach": {"natural": "beach"},
    "mountain": {"natural": "peak"},
    "hill": {"natural": "hill"},
    "valley": {"natural": "valley"},
    "cape": {"natural": "cape"},
    "bay": {"natural": "bay"},
    "forest": {"natural": "wood"},
    "cave": {"natural": "cave_entrance"},
    "island": {"place": "island"},
    "bridge": {"man_made": "bridge"},
    "pier": {"man_made": "pier"},
    "dam": {"waterway": "dam"},
    **_leaves({"man_made": "tower"}, "tower", "observation_deck"),
    "lake": {"natural": "water", "water": "lake"},
    "river": {"waterway": "river"},
    "waterfall": {"waterway": "waterfall"},
    "hot_springs": {"natural": "hot_spring"},
    "botanical_garden": {"leisure": "garden"},
    "nature_reserve": {"leisure": "nature_reserve"},
    "marina": {"leisure": "marina"},
    **_leaves({"industrial": "port", "landuse": "industrial"}, "harbor",
              "port"),
}


def tags_for_hierarchy(hierarchy) -> tuple[dict | None, bool]:
    """(mapped tags, used the root default) for one taxonomy hierarchy.

    None means the record has no usable taxonomy and is dropped.
    """
    if not hierarchy:
        return None, False
    root, leaf = hierarchy[0], hierarchy[-1]
    if leaf in LEAF_TAGS:
        return dict(LEAF_TAGS[leaf]), False
    if root in ROOT_TAGS:
        return dict(ROOT_TAGS[root]), True
    return None, False


def _pairs(value) -> list[tuple[str, str]]:
    """Normalise a pyarrow map (list of pairs) or dict to (key, value) pairs."""
    if not value:
        return []
    if isinstance(value, dict):
        return [(k, v) for k, v in value.items() if v]
    return [(k, v) for k, v in value if v]


def _source_records(sources) -> list[dict]:
    return [
        {key: record.get(key) for key in
         ("dataset", "provider", "license", "record_id", "update_time")}
        for record in (sources or [])
    ]


def record_to_row(record: dict) -> tuple[dict | None, str]:
    """Return (row, reason). A row is None when dropped; reason names why."""
    names = record.get("names") or {}
    primary = (names.get("primary") or "").strip()
    if not primary:
        return None, "no_name"
    taxonomy = record.get("taxonomy") or {}
    hierarchy = list(taxonomy.get("hierarchy") or [])
    mapped, fell_back = tags_for_hierarchy(hierarchy)
    if mapped is None:
        return None, ("no_taxonomy" if not hierarchy else "unmapped_root")
    tags = {"name": primary}
    for lang, value in _pairs(names.get("common")):
        tags[f"name:{lang}"] = str(value).strip()
    brand = ((record.get("brand") or {}).get("names") or {}).get("primary")
    if brand:
        tags["brand"] = str(brand).strip()
    tags.update(mapped)
    confidence = record.get("confidence")
    tags["overture:category"] = (
        (record.get("categories") or {}).get("primary") or hierarchy[-1])
    tags["overture:hierarchy"] = "/".join(hierarchy)
    tags["overture:confidence"] = (
        f"{float(confidence):.4f}" if confidence is not None else "")
    tags["overture:version"] = str(record.get("version"))
    tags["overture:sources"] = json.dumps(
        _source_records(record.get("sources")), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False)
    row = {
        "id": f"{LANDMARK_TYPE}:{record['id']}",
        "geometry": shapely.from_wkb(record["geometry"]),
        "tags": {key: value for key, value in tags.items() if value != ""},
        "confidence": float(confidence) if confidence is not None else 0.0,
        "leaf": hierarchy[-1] if fell_back else None,
    }
    return row, "kept"


def read_places(parquet_path: Path) -> list[dict]:
    schema_names = set(pq.read_schema(parquet_path).names)
    missing = [column for column in COLUMNS if column not in schema_names]
    if missing:
        raise ValueError(
            f"{parquet_path} lacks Overture Places columns {missing}; this "
            "tool needs a release that carries `taxonomy` (2025-08 or later)")
    return pq.read_table(parquet_path, columns=list(COLUMNS)).to_pylist()


def extract(records: list[dict], bbox: tuple, min_confidence: float,
            ) -> tuple[gpd.GeoDataFrame, dict]:
    west, south, east, north = bbox
    dropped: Counter = Counter()
    by_root: Counter = Counter()
    leaf_fallbacks: Counter = Counter()
    rows = []
    for record in records:
        row, reason = record_to_row(record)
        if row is None:
            dropped[reason] += 1
            continue
        point = row["geometry"]
        if not (west <= point.x <= east and south <= point.y <= north):
            dropped["outside_bbox"] += 1
            continue
        if row["confidence"] < min_confidence:
            dropped["low_confidence"] += 1
            continue
        if row["leaf"] is not None:
            leaf_fallbacks[row["leaf"]] += 1
        by_root[row["tags"]["overture:hierarchy"].split("/")[0]] += 1
        rows.append(row)
    rows.sort(key=lambda row: (-row["confidence"], row["id"]))
    frame = schema.build_frame(
        ids=[row["id"] for row in rows],
        geometries=[row["geometry"] for row in rows],
        landmark_types=[LANDMARK_TYPE] * len(rows),
        tags=[row["tags"] for row in rows])
    report = {
        "rows_in": len(records),
        "rows_out": len(rows),
        "dropped": dict(sorted(dropped.items())),
        "by_root": dict(sorted(by_root.items())),
        "leaf_fallbacks": dict(sorted(leaf_fallbacks.items())),
    }
    return frame, report


def main(parquet_path: Path, release: str, bbox: tuple, min_confidence: float,
         output_path: Path) -> gpd.GeoDataFrame:
    parquet_path = Path(parquet_path)
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"bbox must be W S E N in WGS84 order: {bbox}")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError("min_confidence must be within [0, 1]")
    parquet_sha256 = artifact.sha256_file(parquet_path)
    feather_path = source_publication.output_paths(output_path)[0]
    provenance_base = {
        "tool": "farfield/dataset_tools/extract_landmarks_from_overture.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "arguments": {
            "parquet": str(parquet_path.resolve()),
            "parquet_sha256": parquet_sha256,
            "release": release,
            "bbox": [float(value) for value in bbox],
            "min_confidence": float(min_confidence),
            "landmark_type": LANDMARK_TYPE,
            "output_path": str(feather_path),
        },
    }

    def expected_provenance(_frame, document):
        report = document.get("report")
        if not isinstance(report, dict):
            raise ValueError("completed Overture source lacks its report")
        return {**provenance_base, "report": report}

    completed = source_publication.reuse_completed(
        output_path, expected_provenance)
    if completed is not None:
        print(f"Reusing exact completed source {feather_path}")
        return completed
    source_publication.preflight_output(output_path)

    records = read_places(parquet_path)
    frame, report = extract(records, bbox, min_confidence)
    print(f"{report['rows_in']} places -> {report['rows_out']} landmarks")
    print(f"dropped: {report['dropped']}")
    print(f"by root: {report['by_root']}")
    if report["leaf_fallbacks"]:
        print("leaves on their root default (extend LEAF_TAGS if any matter):")
        for leaf, count in sorted(report["leaf_fallbacks"].items(),
                                  key=lambda item: -item[1]):
            print(f"  {count:5d}  {leaf}")

    if artifact.sha256_file(parquet_path) != parquet_sha256:
        raise RuntimeError(
            "parquet content changed during extraction; refusing to publish")
    feather_path, sidecar = source_publication.publish(
        frame, output_path, {**provenance_base, "report": report})
    print(f"Wrote {feather_path}")
    print(f"      {sidecar}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet", required=True, type=Path,
                        help="Overture Places GeoParquet from "
                             "`overturemaps download --type=place`")
    parser.add_argument("--release", required=True,
                        help="Overture release the parquet was downloaded "
                             "from, e.g. 2026-08-19.0 (recorded only)")
    parser.add_argument("--bbox", required=True, type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        help="keep places inside this WGS84 box")
    parser.add_argument("--min_confidence", required=True, type=float,
                        help="drop places whose Overture confidence is below "
                             "this (0-1)")
    parser.add_argument("--output_path", required=True, type=Path,
                        help="output Feather stem, typically under "
                             "raw_material/catalog_sources/<dataset>/")
    args = parser.parse_args()
    main(args.parquet, args.release, tuple(args.bbox), args.min_confidence,
         args.output_path)
