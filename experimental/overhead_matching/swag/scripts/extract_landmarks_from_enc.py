"""Extract landmarks from NOAA ENC (S-57) chart cells into the swag feather format.

Reads S-57 object-class layers from downloaded ENC cells (see download_enc_cells.py)
with geopandas/pyogrio and maps them onto the existing OSM-style tag vocabulary
(_TAGS_TO_KEEP in swag/model/semantic_landmark_utils.py), so the output feather
flows through vigor_dataset.load_landmark_geojson / prune_landmark and the
correspondence pipeline exactly like an OSM landmark feather.

Class mapping summary (fixed structures are the core far-field targets; floating
buoys are included by default but carry seamark:type + object_class so they can
be filtered downstream — the osm_tags_farfield pano prompt reports them too, so
the two sides meet on man_made=buoy + seamark:type + colour/shape):
    LNDMRK          -> man_made/historic per CATLMK + seamark:type=landmark
    LIGHTS (named)  -> man_made=lighthouse (suppressed when riding on a buoy)
    BCN*/DAYMAR     -> man_made=beacon + seamark:type=beacon_*/daymark
    BOY*            -> man_made=buoy + seamark:type=buoy_* (--exclude_buoys to drop)
    SLCONS          -> man_made=pier/quay/breakwater/groyne per CATSLC
    SILTNK          -> man_made=storage_tank/silo/water_tower per CATSIL
    BUISGL          -> building=<FUNCTN> (kept only if named or visually conspicuous)
    BRIDGE/CRANES   -> man_made=bridge/crane + bridge:type/crane:type
    FORSTC          -> historic=fort/castle/...
    LNDARE/LNDRGN   -> place=island/islet, natural=beach/cape/cliff/... per toponym
                       (named features only; unnamed land is the mainland and bare rocks)
    HRBFAC          -> leisure=marina, amenity=ferry_terminal, ... per CATHAF
Common attributes: OBJNAM->name, COLOUR->colour, HEIGHT->height, CONVIS/BOYSHP/
CATLAM/CATCAM -> description.

Example:
    bazel run //experimental/overhead_matching/swag/scripts:extract_landmarks_from_enc -- \\
        --cells US5BOSCC US5BOSCD US5BOSCE US5BOSDB US5BOSDC US5BOSDD US5BOSBE US5BOSBF \\
        --output_path /data/overhead_matching/datasets/enc_cells/extracts/boston_harbor_enc_v1
"""

import os

# Must be set before pyogrio first opens an S-57 dataset: without it the GDAL
# S-57 driver exposes COLOUR/COLPAT/FUNCTN/... as list-typed fields, which
# pyogrio silently drops.
os.environ.setdefault("OGR_S57_OPTIONS", "LIST_AS_STRING=ON")

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import geopandas as gpd

from experimental.overhead_matching.swag.data import landmark_schema as ls
import pandas as pd
import pyogrio

from experimental.overhead_matching.swag.scripts.landmark_feather_utils import (
    dedupe_exact_duplicates,
)

DEFAULT_ENC_ROOT = Path("/data/overhead_matching/datasets/enc_cells")

BUOY_CLASS_TO_SEAMARK = {
    "BOYLAT": "buoy_lateral",
    "BOYSPP": "buoy_special_purpose",
    "BOYCAR": "buoy_cardinal",
    "BOYISD": "buoy_isolated_danger",
    "BOYSAW": "buoy_safe_water",
    "BOYINB": "buoy_installation",
}

BEACON_CLASS_TO_SEAMARK = {
    "BCNLAT": "beacon_lateral",
    "BCNSPP": "beacon_special_purpose",
    "BCNCAR": "beacon_cardinal",
    "BCNISD": "beacon_isolated_danger",
    "BCNSAW": "beacon_safe_water",
    "DAYMAR": "daymark",
}

STRUCTURE_CLASSES = ["LNDMRK", "LIGHTS", "SLCONS", "SILTNK", "BUISGL", "BRIDGE",
                     "CRANES", "FORSTC"]
# Named land areas / land regions / harbour facilities. LNDARE and LNDRGN supply
# the island and headland targets for panorama `place=island` observations.
PLACE_CLASSES = ["LNDARE", "LNDRGN", "HRBFAC"]
ALL_CLASSES = (STRUCTURE_CLASSES + PLACE_CLASSES
               + list(BEACON_CLASS_TO_SEAMARK) + list(BUOY_CLASS_TO_SEAMARK))

# Decoded CATLMK meaning -> tags. Meanings come from GDAL's s57expectedinput.csv.
CATLMK_TO_TAGS = {
    "cairn": [("man_made", "cairn")],
    "cemetery": [("landuse", "cemetery")],
    "chimney": [("man_made", "chimney")],
    "dish aerial": [("man_made", "antenna")],
    "flagstaff (flagpole)": [("man_made", "flagpole")],
    "flare stack": [("man_made", "flare")],
    "mast": [("man_made", "mast")],
    "windsock": [("man_made", "windsock")],
    "monument": [("historic", "monument")],
    "column (pillar)": [("man_made", "column")],
    "memorial plaque": [("historic", "memorial")],
    "obelisk": [("historic", "monument")],
    "statue": [("historic", "monument")],
    "cross": [("man_made", "cross")],
    "dome": [("man_made", "dome")],
    "radar scanner": [("man_made", "antenna")],
    "tower": [("man_made", "tower")],
    "windmill": [("man_made", "windmill")],
    "windmotor": [("man_made", "windmill")],
    "spire/minaret": [("man_made", "tower"), ("tower:type", "spire")],
}

CATSLC_TO_TAGS = {
    "breakwater": [("man_made", "breakwater")],
    "groyne (groin)": [("man_made", "groyne")],
    "mole": [("man_made", "breakwater")],
    "pier ( jetty)": [("man_made", "pier")],
    "promenadepier": [("man_made", "pier")],
    "wharf (quay)": [("man_made", "quay")],
    "training wall": [("man_made", "breakwater")],
    "solid face wharf": [("man_made", "quay")],
    "open face wharf": [("man_made", "quay")],
    # rip rap / revetment / sea wall / landing steps / ramp / slipway / fender:
    # deliberately unmapped — not useful far-field landmarks.
}

CATSIL_TO_TAGS = {
    "silo in general": [("man_made", "silo")],
    "tank in general": [("man_made", "storage_tank")],
    "grain elevator": [("man_made", "silo")],
    "water tower": [("man_made", "water_tower")],
}

CATHAF_TO_TAGS = {
    "yacht harbour/marina": [("leisure", "marina")],
    "ferry terminal": [("amenity", "ferry_terminal")],
    "passenger terminal": [("amenity", "ferry_terminal")],
    "RoRo-terminal": [("amenity", "ferry_terminal")],
    "shipyard": [("landuse", "industrial"), ("industrial", "shipyard")],
    "naval base": [("landuse", "military")],
    "fishing harbour": [("landuse", "port")],
    "tanker terminal": [("landuse", "port")],
    "container terminal": [("landuse", "port")],
    "bulk terminal": [("landuse", "port")],
}

# Trailing generic term of an ENC toponym -> tags. ENC names land features
# "<proper name> <generic term>" ("Georges Island", "Nantasket Beach"), and it
# splits islands inconsistently across LNDARE and LNDRGN (Georges Island is a
# LNDARE, Deer Island a LNDRGN), so the toponym is the only consistent signal
# for what kind of place a named region actually is.
TOPONYM_GENERIC_TO_TAGS = {
    "island": [("place", "island")],
    "islands": [("place", "island")],
    "isle": [("place", "island")],
    "islet": [("place", "islet")],
    "rock": [("natural", "rock")],
    "rocks": [("natural", "rock")],
    "ledge": [("natural", "rock")],
    "beach": [("natural", "beach")],
    "bluff": [("natural", "cliff")],
    "cliff": [("natural", "cliff")],
    "cliffs": [("natural", "cliff")],
    "head": [("natural", "cape")],
    "point": [("natural", "cape")],
    "neck": [("natural", "cape")],
    "cape": [("natural", "cape")],
    "spit": [("natural", "cape")],
    "hill": [("natural", "peak")],
    "hills": [("natural", "peak")],
    "heights": [("natural", "peak")],
}


def tags_from_toponym(name: str) -> list[tuple[str, str]] | None:
    """Infer place/natural tags from a toponym's trailing generic term."""
    last_word = name.strip().rsplit(" ", 1)[-1].lower()
    return TOPONYM_GENERIC_TO_TAGS.get(last_word)

# Lights within this distance of a buoy are buoy lights, not fixed lighthouses.
LIGHT_ON_BUOY_SUPPRESSION_M = 15.0
# LNDMRK features within this distance of a LIGHTS point are light-supports.
LIGHT_COLOCATION_M = 15.0


def load_s57_enum_tables() -> dict[str, dict[int, str]]:
    """Build {attribute_acronym: {enum_code: meaning}} from GDAL's S-57 tables.

    The tables ship inside the pyogrio wheel (gdal_data/). They are latin-1
    encoded (some meanings contain accented characters).
    """
    gdal_data = Path(pyogrio.__file__).parent / "gdal_data"
    acronym_by_code: dict[str, str] = {}
    with open(gdal_data / "s57attributes.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            acronym_by_code[row["Code"]] = row["Acronym"]
    tables: dict[str, dict[int, str]] = {}
    with open(gdal_data / "s57expectedinput.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            acronym = acronym_by_code.get(row["Code"])
            if acronym is None:
                continue
            tables.setdefault(acronym, {})[int(row["ID"])] = row["Meaning"]
    return tables


def decode_enum(enums: dict[str, dict[int, str]], acronym: str, raw) -> list[str]:
    """Decode a raw S-57 enum value ("3", "3,6", 3, 3.0) to meaning strings."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    if isinstance(raw, str):
        codes = [c.strip() for c in raw.split(",") if c.strip()]
    else:
        codes = [raw]
    table = enums.get(acronym, {})
    out = []
    for code in codes:
        try:
            meaning = table.get(int(float(code)))
        except (TypeError, ValueError):
            meaning = None
        if meaning is not None:
            out.append(meaning)
    return out


def _get(row, key):
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _meters_between(lon1, lat1, lon2, lat2) -> float:
    meters_per_deg_lat = 110_574.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat1))
    return math.hypot((lon2 - lon1) * meters_per_deg_lon,
                      (lat2 - lat1) * meters_per_deg_lat)


def _common_tags(row, enums) -> tuple[dict[str, str], list[str]]:
    """Tags + description parts shared by every mapped class."""
    tags: dict[str, str] = {}
    description_parts: list[str] = []
    name = _get(row, "OBJNAM")
    if name:
        tags["name"] = str(name)
    colours = decode_enum(enums, "COLOUR", _get(row, "COLOUR"))
    if colours:
        tags["colour"] = ";".join(colours)
    height = _get(row, "HEIGHT")
    if height is not None:
        tags["height"] = f"{float(height):g}"
    if decode_enum(enums, "CONVIS", _get(row, "CONVIS")) == ["visual conspicuous"]:
        description_parts.append("visually conspicuous")
    return tags, description_parts


def map_class_row(object_class: str, row: dict, enums) -> dict[str, str] | None:
    """Map one S-57 feature record to OSM-style tags; None means skip."""
    tags, description_parts = _common_tags(row, enums)

    if object_class in BUOY_CLASS_TO_SEAMARK:
        tags["man_made"] = "buoy"
        tags["seamark:type"] = BUOY_CLASS_TO_SEAMARK[object_class]
        description_parts += [f"{s} buoy" for s in decode_enum(enums, "BOYSHP", _get(row, "BOYSHP"))]
        description_parts += decode_enum(enums, "CATLAM", _get(row, "CATLAM"))
        description_parts += decode_enum(enums, "CATCAM", _get(row, "CATCAM"))
    elif object_class in BEACON_CLASS_TO_SEAMARK:
        tags["man_made"] = "beacon"
        tags["seamark:type"] = BEACON_CLASS_TO_SEAMARK[object_class]
        description_parts += decode_enum(enums, "CATLAM", _get(row, "CATLAM"))
        description_parts += decode_enum(enums, "CATCAM", _get(row, "CATCAM"))
    elif object_class == "LNDMRK":
        tags["seamark:type"] = "landmark"
        for meaning in decode_enum(enums, "CATLMK", _get(row, "CATLMK")):
            for key, value in CATLMK_TO_TAGS.get(meaning, [("man_made", meaning)]):
                tags[key] = value
        functions = decode_enum(enums, "FUNCTN", _get(row, "FUNCTN"))
        if any("light support" in f for f in functions):
            tags["man_made"] = "lighthouse"
    elif object_class == "LIGHTS":
        # Only named standalone lights become features; unnamed lights merely
        # promote co-located structures (handled in assemble_features).
        if "name" not in tags:
            return None
        tags["man_made"] = "lighthouse"
        tags["seamark:type"] = "light"
    elif object_class == "SLCONS":
        mapped = None
        for meaning in decode_enum(enums, "CATSLC", _get(row, "CATSLC")):
            mapped = CATSLC_TO_TAGS.get(meaning)
            if mapped:
                break
        if not mapped:
            return None
        for key, value in mapped:
            tags[key] = value
    elif object_class == "SILTNK":
        mapped = [("man_made", "storage_tank")]
        for meaning in decode_enum(enums, "CATSIL", _get(row, "CATSIL")):
            mapped = CATSIL_TO_TAGS.get(meaning, mapped)
        for key, value in mapped:
            tags[key] = value
    elif object_class == "BUISGL":
        # Charted buildings are only worth keeping when the chart singles them
        # out: named or marked visually conspicuous.
        if "name" not in tags and "visually conspicuous" not in description_parts:
            return None
        functions = decode_enum(enums, "FUNCTN", _get(row, "FUNCTN"))
        tags["building"] = functions[0] if functions else "yes"
    elif object_class == "BRIDGE":
        tags["man_made"] = "bridge"
        for meaning in decode_enum(enums, "CATBRG", _get(row, "CATBRG")):
            tags["bridge:type"] = meaning.removesuffix(" bridge")
            break
    elif object_class == "CRANES":
        tags["man_made"] = "crane"
        for meaning in decode_enum(enums, "CATCRN", _get(row, "CATCRN")):
            tags["crane:type"] = meaning
            break
    elif object_class == "FORSTC":
        category = decode_enum(enums, "CATFOR", _get(row, "CATFOR"))
        tags["historic"] = category[0].lower() if category else "fort"
    elif object_class in ("LNDARE", "LNDRGN"):
        # Only named regions are useful match targets: unnamed land is the
        # mainland plus hundreds of bare rocks, none of which a panorama
        # landmark names or distinguishes.
        if "name" not in tags:
            return None
        # A named land area on a marine chart is an island; a named land region
        # may be any topographic feature, so it stays generic unless the
        # toponym says otherwise.
        default = ([("place", "island")] if object_class == "LNDARE"
                   else [("place", "locality")])
        for key, value in tags_from_toponym(tags["name"]) or default:
            tags[key] = value
    elif object_class == "HRBFAC":
        mapped = None
        for meaning in decode_enum(enums, "CATHAF", _get(row, "CATHAF")):
            mapped = CATHAF_TO_TAGS.get(meaning)
            if mapped:
                break
        if not mapped:
            return None
        for key, value in mapped:
            tags[key] = value
    else:
        return None

    if description_parts:
        tags["description"] = ", ".join(description_parts)
    return tags


def assemble_features(
    per_class: dict[str, gpd.GeoDataFrame],
    enums: dict[str, dict[int, str]],
    include_buoys: bool = True,
) -> tuple[list[dict], Counter]:
    """Map per-class S-57 GeoDataFrames into landmark feature dicts.

    Cross-class logic:
      - LIGHTS within LIGHT_ON_BUOY_SUPPRESSION_M of a buoy are dropped (buoy
        lights must not become fixed "lighthouses").
      - LNDMRK features within LIGHT_COLOCATION_M of a surviving light are
        promoted to man_made=lighthouse (e.g. "Deer Island Light" is charted
        as a conspicuous LNDMRK with a co-located LIGHTS point).

    Returns (features, skip_counter); each feature is
    {"lnam", "object_class", "geometry", "tags"}.
    """
    buoy_points = []
    for object_class in BUOY_CLASS_TO_SEAMARK:
        if object_class not in per_class:
            continue
        for geom in per_class[object_class].geometry.tolist():
            if geom is not None:
                point = geom.representative_point()
                buoy_points.append((point.x, point.y))

    def near_buoy(geom) -> bool:
        point = geom.representative_point()
        return any(_meters_between(point.x, point.y, bx, by) <= LIGHT_ON_BUOY_SUPPRESSION_M
                   for bx, by in buoy_points)

    fixed_light_points = []
    if "LIGHTS" in per_class:
        for geom in per_class["LIGHTS"].geometry.tolist():
            if geom is not None and not near_buoy(geom):
                point = geom.representative_point()
                fixed_light_points.append((point.x, point.y))

    def near_fixed_light(geom) -> bool:
        point = geom.representative_point()
        return any(_meters_between(point.x, point.y, lx, ly) <= LIGHT_COLOCATION_M
                   for lx, ly in fixed_light_points)

    features = []
    skipped = Counter()
    for object_class, gdf in per_class.items():
        if object_class in BUOY_CLASS_TO_SEAMARK and not include_buoys:
            skipped[f"{object_class}: buoys excluded"] += len(gdf)
            continue
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                skipped[f"{object_class}: no geometry"] += 1
                continue
            if object_class == "LIGHTS" and near_buoy(geom):
                skipped["LIGHTS: on buoy"] += 1
                continue
            tags = map_class_row(object_class, row, enums)
            if tags is None:
                skipped[f"{object_class}: unmapped"] += 1
                continue
            if object_class == "LNDMRK" and near_fixed_light(geom):
                tags["man_made"] = "lighthouse"
            features.append({
                "lnam": row.get("LNAM"),
                "object_class": object_class,
                "geometry": geom,
                "tags": tags,
            })
    return features, skipped


def read_cells(enc_root: Path, cells: list[str]) -> dict[str, gpd.GeoDataFrame]:
    """Read the in-scope layers of each cell, deduplicating features by LNAM.

    LNAM (agency + feature id) is globally unique and stable, so the same
    physical feature appearing in adjacent cells is kept once, from the first
    cell in --cells order.
    """
    per_class: dict[str, list[gpd.GeoDataFrame]] = {}
    seen_lnam: set[str] = set()
    for cell in cells:
        path = enc_root / "ENC_ROOT" / cell / f"{cell}.000"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run download_enc_cells.py first")
        available = {name for name, _ in pyogrio.list_layers(str(path))}
        for object_class in ALL_CLASSES:
            if object_class not in available:
                continue
            gdf = gpd.read_file(path, layer=object_class)
            if "LNAM" not in gdf.columns:
                raise RuntimeError(f"{cell}/{object_class}: no LNAM column")
            fresh = gdf[~gdf["LNAM"].isin(seen_lnam)]
            seen_lnam.update(fresh["LNAM"].tolist())
            if len(fresh):
                per_class.setdefault(object_class, []).append(fresh)
    return {
        object_class: gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True), crs="EPSG:4326")
        for object_class, frames in per_class.items()
    }


def features_to_geodataframe(features: list[dict], landmark_type: str) -> gpd.GeoDataFrame:
    """Build the pipeline-format feather frame: id/geometry/landmark_type +
    object_class provenance + a `tags` dict column (see data/landmark_schema.py).
    Never writes pruned_props (vigor_dataset.load_landmark_geojson computes it at
    load time).

    object_class stays a real column, because landmark_feather_utils treats it as
    metadata when deduping, and is *also* placed inside the tag dict, because tag
    consumers have always seen it there -- under the old wide layout it was simply
    another column and so landed in every tag record."""
    frame = ls.build_frame(
        ids=[f"('enc', '{feature['lnam']}')" for feature in features],
        geometries=[feature["geometry"] for feature in features],
        landmark_types=[landmark_type] * len(features),
        tags=[{**feature["tags"], "object_class": feature["object_class"]}
              for feature in features],
    )
    frame["object_class"] = [feature["object_class"] for feature in features]
    return frame


def bbox_from_dataset_path(dataset_path: Path) -> tuple[float, float, float, float]:
    """Read (west, south, east, north) from satellite_bbox.json, +10% buffer
    (mirrors extract_landmarks_historical.bbox_from_dataset_path)."""
    with open(dataset_path / "satellite_bbox.json") as f:
        meta = json.load(f)
    west, south, east, north = meta["west"], meta["south"], meta["east"], meta["north"]
    width, height = east - west, north - south
    return (west - 0.1 * width, south - 0.1 * height,
            east + 0.1 * width, north + 0.1 * height)


def filter_features_to_bbox(features: list[dict],
                            bbox: tuple[float, float, float, float]) -> list[dict]:
    west, south, east, north = bbox
    kept = []
    for feature in features:
        point = feature["geometry"].representative_point()
        if west <= point.x <= east and south <= point.y <= north:
            kept.append(feature)
    return kept


def main(enc_root: Path, cells: list[str], output_path: Path,
         bbox: tuple[float, float, float, float] | None,
         include_buoys: bool, landmark_type: str,
         dedupe_tolerance_m: float = 10.0) -> gpd.GeoDataFrame:
    enums = load_s57_enum_tables()
    per_class = read_cells(enc_root, cells)
    print("Layer feature counts (LNAM-deduped): "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(per_class.items())))

    features, skipped = assemble_features(per_class, enums, include_buoys=include_buoys)
    if skipped:
        print("Skipped: " + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())))
    if bbox is not None:
        before = len(features)
        features = filter_features_to_bbox(features, bbox)
        print(f"Bbox filter {bbox}: {before} -> {len(features)} features")

    gdf = features_to_geodataframe(features, landmark_type)
    if dedupe_tolerance_m > 0:
        gdf = dedupe_exact_duplicates(gdf, dedupe_tolerance_m)
    print(f"\n{len(gdf)} landmarks")
    print("object_class counts:")
    print(gdf["object_class"].value_counts().to_string())
    # Read names through the schema helper: under the dict layout there is no
    # `name` column, and a column check silently reports zero.
    print(f"named: {sum(1 for t in ls.tag_dicts(gdf) if t.get('name'))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather_path = output_path.with_suffix(".feather")
    gdf.to_feather(feather_path)
    print(f"Wrote {feather_path}")
    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enc_root", type=Path, default=DEFAULT_ENC_ROOT,
                        help="Directory holding ENC_ROOT/ (see download_enc_cells.py)")
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output path; .feather suffix is applied")
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument("--bbox", nargs=4, type=float,
                            metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    bbox_group.add_argument("--dataset_path", type=Path,
                            help="VIGOR-style dataset dir; bbox from satellite_bbox.json +10%%")
    parser.add_argument("--exclude_buoys", action="store_true",
                        help="Drop floating buoys (BOY* classes)")
    parser.add_argument("--landmark_type", default="enc",
                        help="Provenance value for the landmark_type column")
    parser.add_argument("--dedupe_tolerance_m", type=float, default=10.0,
                        help="Merge identical-tag features whose geometries are "
                             "within this distance (0 disables)")
    args = parser.parse_args()

    resolved_bbox = None
    if args.bbox:
        resolved_bbox = tuple(args.bbox)
    elif args.dataset_path:
        resolved_bbox = bbox_from_dataset_path(args.dataset_path)

    main(enc_root=args.enc_root, cells=args.cells, output_path=args.output_path,
         bbox=resolved_bbox, include_buoys=not args.exclude_buoys,
         landmark_type=args.landmark_type,
         dedupe_tolerance_m=args.dedupe_tolerance_m)
