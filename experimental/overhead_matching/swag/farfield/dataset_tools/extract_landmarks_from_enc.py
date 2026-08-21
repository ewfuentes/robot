"""Extract landmarks from NOAA ENC (S-57) chart cells into the feather format.

Reads S-57 object-class layers from downloaded ENC cells (see
download_enc_cells.py) with geopandas/pyogrio and maps them onto the OSM-style
far-field tag vocabulary owned by `farfield.catalog.catalog`
(`keeps_tag_key`), so the output feather flows through
`catalog.schema.read_frame` / `catalog.load_catalog` exactly like an OSM
landmark feather. Every key this mapper emits survives that vocabulary — the
test enforces it, because a key outside the keep-list is silently pruned at
load time.

Class mapping summary (fixed structures are the core far-field targets;
floating buoys are included by default but carry seamark:type + object_class
so they can be filtered downstream — the osm_tags_farfield pano prompt reports
them too, so the two sides meet on man_made=buoy + seamark:type +
colour/shape):
    LNDMRK          -> man_made/historic per CATLMK + seamark:type=landmark
    LIGHTS (named)  -> man_made=lighthouse (suppressed when riding on a buoy)
    BCN*/DAYMAR     -> man_made=beacon + seamark:type=beacon_*/daymark
    BOY*            -> man_made=buoy + seamark:type=buoy_* (--exclude_buoys)
    SLCONS          -> man_made=pier/quay/breakwater/groyne per CATSLC
    SILTNK          -> man_made=storage_tank/silo/water_tower per CATSIL
    BUISGL          -> building=<FUNCTN> (kept only if named or conspicuous)
    BRIDGE/CRANES   -> man_made=bridge/crane + bridge:type (crane category
                       goes into `description`: `crane:type` is not in the
                       far-field keep vocabulary and would be pruned)
    FORSTC          -> historic=fort/castle/...
    LNDARE/LNDRGN   -> place=island/islet, natural=beach/cape/... per toponym
                       (named features only; unnamed land is the mainland and
                       bare rocks)
    HRBFAC          -> leisure=marina, amenity=ferry_terminal, ... per CATHAF
Common attributes: OBJNAM->name, COLOUR->colour, HEIGHT->height,
CONVIS/BOYSHP/CATLAM/CATCAM -> description.

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:extract_landmarks_from_enc -- \\
        --enc_root /data/farfield_matching/raw_material/enc_cells \\
        --cells US5BOSCC US5BOSCD \\
        --dedupe_tolerance_m 10 \\
        --output_path /data/farfield_matching/artifacts/catalogs/boston_harbor_leg1/sources/enc_v1
"""

import os

# Must be set before pyogrio first opens an S-57 dataset: without it the GDAL
# S-57 driver exposes COLOUR/COLPAT/FUNCTN/... as list-typed fields, which
# pyogrio silently drops.
os.environ.setdefault("OGR_S57_OPTIONS", "LIST_AS_STRING=ON")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402

from experimental.overhead_matching.swag.farfield import (  # noqa: E402
    provenance,
)
from experimental.overhead_matching.swag.farfield.catalog import (  # noqa: E402,E501
    schema,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (  # noqa: E402,E501
    feather_utils,
)

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

STRUCTURE_CLASSES = ["LNDMRK", "LIGHTS", "SLCONS", "SILTNK", "BUISGL",
                     "BRIDGE", "CRANES", "FORSTC"]
# Named land areas / land regions / harbour facilities. LNDARE and LNDRGN
# supply the island and headland targets for panorama `place=island`
# observations.
PLACE_CLASSES = ["LNDARE", "LNDRGN", "HRBFAC"]
ALL_CLASSES = (STRUCTURE_CLASSES + PLACE_CLASSES
               + list(BEACON_CLASS_TO_SEAMARK) + list(BUOY_CLASS_TO_SEAMARK))

# Decoded CATLMK meaning -> tags. Meanings come from GDAL's
# s57expectedinput.csv.
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
    # rip rap / revetment / sea wall / landing steps / ramp / slipway /
    # fender: deliberately unmapped — not useful far-field landmarks.
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


def tags_from_toponym(name: str):
    """Infer place/natural tags from a toponym's trailing generic term."""
    last_word = name.strip().rsplit(" ", 1)[-1].lower()
    return TOPONYM_GENERIC_TO_TAGS.get(last_word)


# Lights within this distance of a buoy are buoy lights, not fixed
# lighthouses.
LIGHT_ON_BUOY_SUPPRESSION_M = 15.0
# LNDMRK features within this distance of a LIGHTS point are light-supports.
LIGHT_COLOCATION_M = 15.0


def load_s57_enum_tables() -> dict:
    """Build {attribute_acronym: {enum_code: meaning}} from GDAL's tables.

    The tables ship inside the pyogrio wheel (gdal_data/). They are latin-1
    encoded (some meanings contain accented characters).
    """
    gdal_data = Path(pyogrio.__file__).parent / "gdal_data"
    acronym_by_code: dict = {}
    with open(gdal_data / "s57attributes.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            acronym_by_code[row["Code"]] = row["Acronym"]
    tables: dict = {}
    with open(gdal_data / "s57expectedinput.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            acronym = acronym_by_code.get(row["Code"])
            if acronym is None:
                continue
            tables.setdefault(acronym, {})[int(row["ID"])] = row["Meaning"]
    return tables


def decode_enum(enums: dict, acronym: str, raw) -> list:
    """Decode a raw S-57 enum value ("3", "3,6", 3, 3.0) to meanings."""
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


# A mark's board designator, as it appears at the end of its ENC name:
# "Hingham Harbor Channel Buoy 16" -> "16", "Squantum Channel Buoy 1SC" ->
# "1SC", "President Roads Anchorage Buoy C" -> "C".
_DESIGNATOR_RE = re.compile(
    r"\b(?:buoy|beacon|mark|light|daybeacon)\s+"
    r"([0-9]{1,3}[A-Z]{0,2}|[A-Z]{1,3})$",
    re.IGNORECASE)


def designator_from_name(name: str):
    """The number or letter painted on a mark, pulled out of its ENC name.

    The extractor reads these off the images and reports them as `ref=`, but
    the catalog only ever carried them buried inside `name`, so the two could
    not meet: a detection tagged `ref=16` matched nothing, while a detection
    *named* `16` let the matcher pick one of the three "Buoy 16" rows in
    Boston Harbor - which sit 19.4 km apart. Publishing the designator as its
    own tag puts every mark carrying that number on an equal footing in the
    evidence (91% of named buoys in this catalog share their designator with
    another, a median 14.7 km apart).

    That makes a disjunction *possible*, not automatic: nothing downstream
    forces same-`ref` rows to be scored equally. The matcher is an LLM and
    may still prefer one; enforcing it would need either an instruction in
    the matcher prompt or a post-hoc pass that levels the likelihood across
    every catalog row sharing the tracklet's `ref`.
    """
    if not name:
        return None
    match = _DESIGNATOR_RE.search(name.strip())
    return match.group(1).upper() if match else None


def _common_tags(row, enums):
    """Tags + description parts shared by every mapped class."""
    tags: dict = {}
    description_parts: list = []
    name = _get(row, "OBJNAM")
    if name:
        tags["name"] = str(name)
        designator = designator_from_name(str(name))
        if designator:
            tags["ref"] = designator
    colours = decode_enum(enums, "COLOUR", _get(row, "COLOUR"))
    if colours:
        tags["colour"] = ";".join(colours)
    height = _get(row, "HEIGHT")
    if height is not None:
        tags["height"] = f"{float(height):g}"
    if decode_enum(enums, "CONVIS",
                   _get(row, "CONVIS")) == ["visual conspicuous"]:
        description_parts.append("visually conspicuous")
    return tags, description_parts


def map_class_row(object_class: str, row: dict, enums):
    """Map one S-57 feature record to OSM-style tags; None means skip."""
    tags, description_parts = _common_tags(row, enums)

    if object_class in BUOY_CLASS_TO_SEAMARK:
        tags["man_made"] = "buoy"
        tags["seamark:type"] = BUOY_CLASS_TO_SEAMARK[object_class]
        description_parts += [
            f"{s} buoy"
            for s in decode_enum(enums, "BOYSHP", _get(row, "BOYSHP"))]
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
            for key, value in CATLMK_TO_TAGS.get(meaning,
                                                 [("man_made", meaning)]):
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
        # Charted buildings are only worth keeping when the chart singles
        # them out: named or marked visually conspicuous.
        if ("name" not in tags
                and "visually conspicuous" not in description_parts):
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
        # The crane category goes into `description`, not a `crane:type`
        # tag: `crane:` is not a kept prefix in the far-field vocabulary
        # (catalog.keeps_tag_key), so a crane:type tag would be silently
        # pruned the moment the catalog loads this row.
        for meaning in decode_enum(enums, "CATCRN", _get(row, "CATCRN")):
            description_parts.append(meaning)
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
        # A named land area on a marine chart is an island; a named land
        # region may be any topographic feature, so it stays generic unless
        # the toponym says otherwise.
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


def assemble_features(per_class: dict, enums: dict,
                      include_buoys: bool = True):
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
        return any(
            _meters_between(point.x, point.y, bx, by)
            <= LIGHT_ON_BUOY_SUPPRESSION_M
            for bx, by in buoy_points)

    fixed_light_points = []
    if "LIGHTS" in per_class:
        for geom in per_class["LIGHTS"].geometry.tolist():
            if geom is not None and not near_buoy(geom):
                point = geom.representative_point()
                fixed_light_points.append((point.x, point.y))

    def near_fixed_light(geom) -> bool:
        point = geom.representative_point()
        return any(
            _meters_between(point.x, point.y, lx, ly) <= LIGHT_COLOCATION_M
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


def read_cells(enc_root: Path, cells: list) -> dict:
    """Read the in-scope layers of each cell, deduplicating features by LNAM.

    LNAM (agency + feature id) is globally unique and stable, so the same
    physical feature appearing in adjacent cells is kept once, from the first
    cell in --cells order.
    """
    per_class: dict = {}
    seen_lnam: set = set()
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


def features_to_geodataframe(features: list,
                             landmark_type: str) -> gpd.GeoDataFrame:
    """Build the feather frame through catalog.schema.build_frame:
    id/geometry/landmark_type + a `tags` dict column, plus `object_class`
    provenance.

    object_class stays a real column, because feather_utils treats it as
    metadata when deduping, and is *also* placed inside the tag dict, because
    tag consumers have always seen it there -- under the old wide layout it
    was simply another column and so landed in every tag record.
    """
    frame = schema.build_frame(
        ids=[f"('enc', '{feature['lnam']}')" for feature in features],
        geometries=[feature["geometry"] for feature in features],
        landmark_types=[landmark_type] * len(features),
        tags=[{**feature["tags"], "object_class": feature["object_class"]}
              for feature in features],
    )
    frame["object_class"] = [feature["object_class"]
                             for feature in features]
    return frame


def filter_features_to_bbox(features: list, bbox: tuple) -> list:
    west, south, east, north = bbox
    kept = []
    for feature in features:
        point = feature["geometry"].representative_point()
        if west <= point.x <= east and south <= point.y <= north:
            kept.append(feature)
    return kept


def main(enc_root: Path, cells: list, output_path: Path,
         bbox: tuple | None, include_buoys: bool, landmark_type: str,
         dedupe_tolerance_m: float) -> gpd.GeoDataFrame:
    enums = load_s57_enum_tables()
    per_class = read_cells(enc_root, cells)
    print("Layer feature counts (LNAM-deduped): "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(per_class.items())))

    features, skipped = assemble_features(per_class, enums,
                                          include_buoys=include_buoys)
    if skipped:
        print("Skipped: "
              + ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items())))
    n_before_bbox = len(features)
    if bbox is not None:
        features = filter_features_to_bbox(features, bbox)
        print(f"Bbox filter {bbox}: {n_before_bbox} -> {len(features)} "
              f"features")

    gdf = features_to_geodataframe(features, landmark_type)
    if dedupe_tolerance_m > 0:
        gdf = feather_utils.dedupe_exact_duplicates(gdf, dedupe_tolerance_m)
    print(f"\n{len(gdf)} landmarks")
    print("object_class counts:")
    print(gdf["object_class"].value_counts().to_string())
    # Read names through the schema helper: under the dict layout there is no
    # `name` column, and a column check silently reports zero.
    print(f"named: {sum(1 for t in schema.tag_dicts(gdf) if t.get('name'))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather_path = output_path.with_suffix(".feather")
    gdf.to_feather(feather_path)
    sidecar = feather_path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "tool": "farfield/dataset_tools/extract_landmarks_from_enc.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "arguments": {
            "enc_root": str(enc_root),
            "cells": list(cells),
            "output_path": str(feather_path),
            "bbox": list(bbox) if bbox else None,
            "include_buoys": include_buoys,
            "landmark_type": landmark_type,
            "dedupe_tolerance_m": dedupe_tolerance_m,
        },
        "rows_out": int(len(gdf)),
        "skipped": dict(skipped),
    }, indent=1))
    print(f"Wrote {feather_path}")
    print(f"      {sidecar}")
    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required: three hardcoded copies of one ENC root existed on the
    # checkpoint branch; zero defaults now (REORG.md rule 2).
    parser.add_argument("--enc_root", type=Path, required=True,
                        help="Directory holding ENC_ROOT/ (see "
                             "download_enc_cells.py)")
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--output_path", type=Path, required=True,
                        help="Output path; .feather suffix is applied")
    bbox_group = parser.add_mutually_exclusive_group()
    bbox_group.add_argument("--bbox", nargs=4, type=float,
                            metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    bbox_group.add_argument("--dataset_path", type=Path,
                            help="farfield dataset dir; bbox from its "
                                 "pipeline_metadata.json (legacy: "
                                 "satellite_bbox.json), buffered by "
                                 "--bbox_buffer_frac")
    parser.add_argument("--bbox_buffer_frac", type=float, default=None,
                        help="fraction of the dataset bbox span added on "
                             "every side (required with --dataset_path; "
                             "previously 0.1)")
    parser.add_argument("--exclude_buoys", action="store_true",
                        help="Drop floating buoys (BOY* classes)")
    parser.add_argument("--landmark_type", default="enc",
                        help="Provenance value for the landmark_type column")
    parser.add_argument("--dedupe_tolerance_m", type=float, required=True,
                        help="Merge identical-tag features whose geometries "
                             "are within this distance; 0 disables "
                             "(previously 10.0)")
    args = parser.parse_args()

    resolved_bbox = None
    if args.bbox:
        resolved_bbox = tuple(args.bbox)
    elif args.dataset_path:
        if args.bbox_buffer_frac is None:
            parser.error("--dataset_path needs --bbox_buffer_frac (the "
                         "buffer shapes which rows the catalog contains; "
                         "previously 0.1)")
        resolved_bbox = feather_utils.buffered_bbox(
            feather_utils.bbox_from_dataset(args.dataset_path),
            args.bbox_buffer_frac)

    main(enc_root=args.enc_root, cells=args.cells,
         output_path=args.output_path, bbox=resolved_bbox,
         include_buoys=not args.exclude_buoys,
         landmark_type=args.landmark_type,
         dedupe_tolerance_m=args.dedupe_tolerance_m)
