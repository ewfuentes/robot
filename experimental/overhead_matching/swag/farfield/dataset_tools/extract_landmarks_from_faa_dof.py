"""Extract obstacles from an FAA Digital Obstacle File (DOF) state file into
the catalog Feather format.

The DOF (https://www.faa.gov/air_traffic/flight_info/aeronav/digital_products/dof/,
public domain, 56-day cycle) lists every man-made structure the FAA charts as
an obstacle to aviation: towers, transmission-line towers, stacks, tanks,
tall buildings, wind turbines, cranes. From the air these are exactly the
things a vessel-calibrated OSM catalog is thinnest on, and the FAA records a
surveyed height for each. The file is fixed-width ASCII per the DOF README
(columns are 1-based in the README; `_FIELDS` below is 0-based).

Every kept record is mapped onto OSM-style far-field tags (`TYPE_TAGS`) so
`trim_catalog` judges an FAA tower by the same rules as an OSM
`man_made=tower`; the FAA facts travel as `faa:*` tags, outside the keep
vocabulary, and `height` (metres, from AGL feet) is the one FAA measurement
the matcher sees. Obstacle types that are airport-survey furniture (poles,
fences, signs, approach-light systems, navaids) are dropped by name
(`DROP_TYPES`); an unknown type is dropped and counted separately so a new
FAA vocabulary surfaces instead of silently shrinking the catalog.

Rows are written tallest first; `add_catalog_source` uses row order to pick a
survivor among same-name duplicates (FAA rows are nameless, so this only
orders the output).

Example:
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:extract_landmarks_from_faa_dof -- \\
        --dat /data/farfield_matching/raw_material/faa_dof/23-ME.Dat \\
        --bbox -70.4447 43.5523 -69.7886 44.2032 \\
        --verified_only \\
        --output_path /data/farfield_matching/raw_material/catalog_sources/portland_flight_20260906/faa_dof_20260802_v1
"""

import argparse
import datetime as dt
import re
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
import shapely

from experimental.overhead_matching.swag.farfield import artifact, provenance
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)

LANDMARK_TYPE = "faa"
FEET_TO_M = 0.3048

# 0-based half-open column slices, from the DOF README record layout.
_FIELDS = {
    "oas": (0, 9), "verification": (10, 11), "country": (12, 14),
    "state": (15, 17), "city": (18, 34), "lat": (35, 47), "lon": (48, 61),
    "type": (62, 80), "quantity": (81, 82), "agl_ft": (83, 88),
    "amsl_ft": (89, 94), "lighting": (95, 96), "accuracy_h": (97, 98),
    "accuracy_v": (99, 100), "marking": (101, 102), "study": (103, 117),
    "action": (118, 119), "julian_date": (120, 127),
}
_RECORD_RE = re.compile(r"[0-9A-Z]{2}-\d{6} ")
_DMS_RE = re.compile(r"(\d+) (\d+) ([\d.]+)([NSEW])")
_CURRENCY_RE = re.compile(r"CURRENCY DATE = (\d{2})/(\d{2})/(\d{2})")

# Horizontal accuracy code -> the README's +- tolerance, in metres.
POSITION_TOLERANCE_M = {
    "1": 20 * FEET_TO_M, "2": 50 * FEET_TO_M, "3": 100 * FEET_TO_M,
    "4": 250 * FEET_TO_M, "5": 500 * FEET_TO_M, "6": 1000 * FEET_TO_M,
    "7": 926.0, "8": 1852.0,
}
LIGHTING = {
    "R": "red", "D": "medium_white_strobe_and_red",
    "H": "high_white_strobe_and_red", "M": "medium_white_strobe",
    "S": "high_white_strobe", "F": "flood", "C": "dual_medium_catenary",
    "W": "synchronized_red", "L": "lit_type_unknown", "N": "none",
    "U": "unknown",
}
MARKING = {
    "P": "orange_white_paint", "W": "white_paint", "M": "marked",
    "F": "flag", "S": "spherical", "N": "none", "U": "unknown",
}

# FAA obstacle type -> OSM-style tags. What a distant observer could name.
TYPE_TAGS = {
    "TOWER": {"man_made": "tower"},
    "BLDG-TWR": {"man_made": "tower", "building": "yes"},
    "VERTICAL STRUCTURE": {"man_made": "tower"},
    "CTRL TWR": {"man_made": "tower", "tower:type": "control"},
    "SPIRE": {"man_made": "tower", "tower:type": "spire"},
    "ANTENNA": {"man_made": "mast"},
    "MET": {"man_made": "mast", "tower:type": "meteorological"},
    "T-L TWR": {"power": "tower"},
    "CATENARY": {"power": "line"},
    "WINDMILL": {"power": "generator", "generator:source": "wind",
                 "generator:method": "wind_turbine"},
    # generator:* is in the keep vocabulary, plant:* is not; the matcher must
    # be able to tell a solar array from a power station.
    "SOLAR PANELS": {"power": "generator", "generator:source": "solar",
                     "generator:method": "photovoltaic"},
    "POWER PLANT": {"power": "plant"},
    "BLDG": {"building": "yes"},
    "HANGAR": {"building": "hangar"},
    "DOME": {"building": "yes", "roof:shape": "dome"},
    "STADIUM": {"building": "stadium"},
    "TANK": {"man_made": "storage_tank"},
    "SILO": {"man_made": "silo"},
    "ELEVATOR": {"man_made": "silo"},
    "GRAIN ELEVATOR": {"man_made": "silo"},
    "STACK": {"man_made": "chimney"},
    "COOL TWR": {"man_made": "cooling_tower"},
    "CRANE": {"man_made": "crane"},
    "PLANT": {"man_made": "works"},
    "REFINERY": {"man_made": "works", "industrial": "refinery"},
    "RIG": {"man_made": "offshore_platform"},
    "BRIDGE": {"bridge": "yes"},
    "DAM": {"waterway": "dam"},
    "LANDFILL": {"landuse": "landfill"},
    "MONUMENT": {"historic": "monument"},
    "ARCH": {"historic": "monument"},
    "LGTHOUSE": {"man_made": "lighthouse"},
    "AMUSEMENT PARK": {"tourism": "theme_park"},
}
# Airport-survey furniture and mobile or line-of-sight-invisible objects.
DROP_TYPES = frozenset({
    "POLE", "UTILITY POLE", "ELEC SYS", "FENCE", "SIGN", "NAVAID",
    "GEN UTIL", "AG EQUIP", "WALL", "GATE", "PIPELINE PIPE",
    "NATURAL GAS SYSTEM", "HEAT COOL SYSTEM", "WINDSOCK", "WIND INDICATOR",
    "BALLOON", "SHIP", "TRAMWAY",
})


def _field(line: str, name: str) -> str:
    start, stop = _FIELDS[name]
    return line[start:stop].strip()


def parse_dms(text: str) -> float:
    match = _DMS_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"unparseable DOF coordinate {text!r}")
    degrees, minutes, seconds, hemisphere = match.groups()
    value = int(degrees) + int(minutes) / 60.0 + float(seconds) / 3600.0
    return -value if hemisphere in "SW" else value


def julian_to_iso(text: str) -> str | None:
    """DOF action date YYYYDDD -> ISO date, or None when blank/invalid."""
    if not re.fullmatch(r"\d{7}", text):
        return None
    try:
        return (dt.date(int(text[:4]), 1, 1)
                + dt.timedelta(days=int(text[4:]) - 1)).isoformat()
    except ValueError:
        return None


def parse_currency_date(first_line: str) -> str:
    match = _CURRENCY_RE.search(first_line)
    if match is None:
        raise ValueError("DOF file does not start with a CURRENCY DATE header")
    month, day, year = match.groups()
    return dt.date(2000 + int(year), int(month), int(day)).isoformat()


def parse_record(line: str) -> dict:
    """One DOF detail line -> plain fields (no mapping, no filtering)."""
    return {
        "oas": _field(line, "oas"),
        "verified": _field(line, "verification") == "O",
        "state": _field(line, "state"),
        "city": _field(line, "city"),
        "lat": parse_dms(_field(line, "lat")),
        "lon": parse_dms(_field(line, "lon")),
        "type": _field(line, "type"),
        "quantity": _field(line, "quantity"),
        "agl_ft": int(_field(line, "agl_ft") or 0),
        "amsl_ft": int(_field(line, "amsl_ft") or 0),
        "lighting": _field(line, "lighting"),
        "accuracy_h": _field(line, "accuracy_h"),
        "accuracy_v": _field(line, "accuracy_v"),
        "marking": _field(line, "marking"),
        "study": _field(line, "study"),
        "action": _field(line, "action"),
        "action_date": julian_to_iso(_field(line, "julian_date")),
    }


def read_records(dat_path: Path) -> tuple[str, list[dict]]:
    """(currency date, detail records) of one DOF .Dat file."""
    lines = Path(dat_path).read_text(encoding="latin-1").splitlines()
    if not lines:
        raise ValueError(f"{dat_path} is empty")
    currency = parse_currency_date(lines[0])
    return currency, [parse_record(line) for line in lines
                      if _RECORD_RE.match(line)]


def record_tags(record: dict, currency: str) -> dict:
    """OSM-style tags plus the `faa:*` facts for one mappable record."""
    tags = dict(TYPE_TAGS[record["type"]])
    if record["agl_ft"] > 0:
        tags["height"] = f"{record['agl_ft'] * FEET_TO_M:.1f}"
    tags.update({
        "faa:oas": record["oas"],
        "faa:type": record["type"],
        "faa:verified": "yes" if record["verified"] else "no",
        "faa:quantity": record["quantity"],
        "faa:agl_ft": str(record["agl_ft"]),
        "faa:amsl_ft": str(record["amsl_ft"]),
        "faa:lighting": LIGHTING.get(record["lighting"], "unknown"),
        "faa:marking": MARKING.get(record["marking"], "unknown"),
        "faa:accuracy_h": record["accuracy_h"] or "unknown",
        "faa:accuracy_v": record["accuracy_v"] or "unknown",
        "faa:city": record["city"],
        "faa:currency_date": currency,
    })
    tolerance = POSITION_TOLERANCE_M.get(record["accuracy_h"])
    if tolerance is not None:
        tags["faa:position_tolerance_m"] = f"{tolerance:.1f}"
    if record["study"]:
        tags["faa:study"] = record["study"]
    if record["action_date"]:
        tags["faa:action_date"] = record["action_date"]
    return tags


def extract(records: list[dict], currency: str, bbox: tuple,
            verified_only: bool) -> tuple[gpd.GeoDataFrame, dict]:
    west, south, east, north = bbox
    dropped: Counter = Counter()
    unmapped: Counter = Counter()
    by_type: Counter = Counter()
    kept = []
    for record in records:
        if not (west <= record["lon"] <= east
                and south <= record["lat"] <= north):
            dropped["outside_bbox"] += 1
            continue
        if verified_only and not record["verified"]:
            dropped["unverified"] += 1
            continue
        if record["type"] in DROP_TYPES:
            dropped["dropped_type"] += 1
            continue
        if record["type"] not in TYPE_TAGS:
            unmapped[record["type"]] += 1
            continue
        by_type[record["type"]] += 1
        kept.append(record)
    kept.sort(key=lambda r: (-r["agl_ft"], r["oas"]))
    frame = schema.build_frame(
        ids=[r["oas"] for r in kept],
        geometries=[shapely.Point(r["lon"], r["lat"]) for r in kept],
        landmark_types=[LANDMARK_TYPE] * len(kept),
        tags=[record_tags(r, currency) for r in kept])
    report = {
        "rows_in": len(records),
        "rows_out": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "unmapped_types": dict(sorted(unmapped.items())),
        "by_type": dict(sorted(by_type.items())),
    }
    return frame, report


def main(dat_path: Path, bbox: tuple, verified_only: bool,
         output_path: Path) -> gpd.GeoDataFrame:
    dat_path = Path(dat_path)
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"bbox must be W S E N in WGS84 order: {bbox}")
    dat_sha256 = artifact.sha256_file(dat_path)
    feather_path = source_publication.output_paths(output_path)[0]
    currency, records = read_records(dat_path)
    provenance_base = {
        "tool": "farfield/dataset_tools/extract_landmarks_from_faa_dof.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "arguments": {
            "dat": str(dat_path.resolve()),
            "dat_sha256": dat_sha256,
            "currency_date": currency,
            "bbox": [float(value) for value in bbox],
            "verified_only": bool(verified_only),
            "landmark_type": LANDMARK_TYPE,
            "output_path": str(feather_path),
        },
    }

    def expected_provenance(_frame, document):
        report = document.get("report")
        if not isinstance(report, dict):
            raise ValueError("completed FAA DOF source lacks its report")
        return {**provenance_base, "report": report}

    completed = source_publication.reuse_completed(
        output_path, expected_provenance)
    if completed is not None:
        print(f"Reusing exact completed source {feather_path}")
        return completed
    source_publication.preflight_output(output_path)

    frame, report = extract(records, currency, bbox, verified_only)
    print(f"{report['rows_in']} DOF records (currency {currency}) -> "
          f"{report['rows_out']} landmarks")
    print(f"dropped: {report['dropped']}")
    print(f"by type: {report['by_type']}")
    if report["unmapped_types"]:
        print("types with no TYPE_TAGS entry (dropped; extend the table if "
              "any matter):")
        for name, count in sorted(report["unmapped_types"].items(),
                                  key=lambda item: -item[1]):
            print(f"  {count:5d}  {name}")

    if artifact.sha256_file(dat_path) != dat_sha256:
        raise RuntimeError(
            "DOF file changed during extraction; refusing to publish")
    feather_path, sidecar = source_publication.publish(
        frame, output_path, {**provenance_base, "report": report})
    print(f"Wrote {feather_path}")
    print(f"      {sidecar}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dat", required=True, type=Path,
                        help="one DOF state file (e.g. 23-ME.Dat) from the "
                             "56-day DOF zip")
    parser.add_argument("--bbox", required=True, type=float, nargs=4,
                        metavar=("W", "S", "E", "N"),
                        help="keep obstacles inside this WGS84 box")
    parser.add_argument("--verified_only", action="store_true",
                        help="drop records whose verification status is U")
    parser.add_argument("--output_path", required=True, type=Path,
                        help="output Feather stem, typically under "
                             "raw_material/catalog_sources/<scope>/")
    args = parser.parse_args()
    main(args.dat, tuple(args.bbox), args.verified_only, args.output_path)
