#!/usr/bin/env python3
"""Verify that a set of Geofabrik extracts really covers a request bbox.

Why this exists: the common OSM extractor indexes the PBF before
it can filter by bbox, so cost scales with the file rather than the area asked
for. That pushes you toward the smallest regional sub-extract -- and a
sub-extract that does not reach the whole request area produces a partial catalog
with no error at all. Folkestone makes the stakes concrete: the UK extract alone
yields 1,708 landmarks inside the bbox while the French side holds 465,571, so
silently dropping one side loses 99.6% of the catalog.

The check uses each extract's Geofabrik **.poly** clip boundary, not its PBF
HeaderBBox. The header bbox is only a bounding rectangle: united-kingdom's spans
W-14.86..E2.64, which geometrically contains the whole Dover Strait including
Calais, while the file holds no French data whatsoever. Testing against the
header bbox therefore reports full coverage for an extract that is missing
almost everything -- the exact failure this module exists to prevent.

Ported from swag/scripts/pbf_coverage.py; the run_farfield_collection coverage
gate now imports `check_coverage` from here as a proper package dependency (the
old bare `from pbf_coverage import ...` only resolved when both files shared a
directory, so under bazel the gate was unreachable). The .poly cache directory
is an argument everywhere -- the old module-level ~/scratch default is gone.

    bazel run //experimental/overhead_matching/swag/farfield/collection:pbf_coverage -- \\
        --poly_cache_dir <osm_cache>/poly europe/france/nord-pas-de-calais-latest.osm.pbf
    ... -- --poly_cache_dir <osm_cache>/poly --bbox W S E N <spec> [<spec> ...]
"""

import argparse
import math
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

GEOFABRIK = "https://download.geofabrik.de"
# A gap under this fraction of the request is treated as .poly boundary
# resolution rather than missing data.
# Allowed loss vs the reference set. Not tight, and deliberately so: national
# clip polygons claim territorial water that regional ones do not, so swapping
# whole-France for Nord-Pas-de-Calais "loses" 332 km2 of open Channel containing
# no French land (verified: Calais, Dunkirk and Gravelines all remain covered).
# A genuinely missing landmass looks nothing like that -- dropping the French
# extract entirely loses 71.8% -- so the two regimes are far apart.
COVERAGE_TOLERANCE_FRAC = 0.20

# An extract contributing less than this to the request is the wrong region for
# this bbox, whatever the totals say (e.g. picardie for the Dover Strait: 0.0%).
MIN_USEFUL_CONTRIBUTION_FRAC = 0.005


# ── Geofabrik .poly clip boundaries (the authority on what a file contains) ───

def poly_url_for(spec: str) -> str:
    """'europe/france/nord-pas-de-calais-latest.osm.pbf' -> its .poly URL."""
    stem = spec.rsplit("/", 1)[-1]
    for suffix in ("-latest.osm.pbf", ".osm.pbf"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parent = spec.rsplit("/", 1)[0] if "/" in spec else ""
    return f"{GEOFABRIK}/{parent + '/' if parent else ''}{stem}.poly"


def fetch_poly(spec: str, cache_dir: Path) -> Path:
    """Download (and cache) the .poly for an extract spec."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = poly_url_for(spec).rsplit("/", 1)[-1]
    dest = cache_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = poly_url_for(spec)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def parse_poly(path: Path):
    """Parse an osmosis .poly file into a shapely geometry."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    rings, holes, cur, is_hole = [], [], None, False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "END":
            if cur is not None:
                if len(cur) >= 3:
                    (holes if is_hole else rings).append(Polygon(cur))
                cur = None
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                cur = cur or []
                cur.append((float(parts[0]), float(parts[1])))
                continue
            except ValueError:
                pass
        # a section header: bare name, or !name for a hole
        is_hole = line.startswith("!")
        cur = None
    geom = unary_union(rings) if rings else None
    if geom is not None and holes:
        geom = geom.difference(unary_union(holes))
    return geom


# ── PBF HeaderBBox (reported for context only, never used as proof) ──────────

def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fnum, wtype = key >> 3, key & 7
        if wtype == 0:
            val, i = _varint(buf, i)
        elif wtype == 2:
            ln, i = _varint(buf, i)
            val, i = buf[i:i + ln], i + ln
        elif wtype == 5:
            val, i = buf[i:i + 4], i + 4
        elif wtype == 1:
            val, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wtype}")
        yield fnum, wtype, val


def header_bbox(pbf_path):
    """(west, south, east, north) from the PBF HeaderBBox, or None."""
    with open(pbf_path, "rb") as f:
        raw_len = f.read(4)
        if len(raw_len) != 4:
            raise ValueError(f"{pbf_path}: too short to be a PBF")
        (hdr_len,) = struct.unpack(">i", raw_len)
        btype = datasize = None
        for fnum, _, val in _fields(f.read(hdr_len)):
            if fnum == 1:
                btype = val.decode()
            elif fnum == 3:
                datasize = val
        if btype != "OSMHeader":
            raise ValueError(f"{pbf_path}: first blob is {btype!r}")
        blob = f.read(datasize)
    payload = None
    for fnum, _, val in _fields(blob):
        if fnum == 1:
            payload = val
        elif fnum == 3:
            payload = zlib.decompress(val)
    if payload is None:
        return None
    for fnum, _, val in _fields(payload):
        if fnum == 1:
            vals = {bf: ((bv >> 1) ^ -(bv & 1)) / 1e9 for bf, _, bv in _fields(val)}
            if len(vals) == 4:
                return (vals[1], vals[4], vals[2], vals[3])
    return None


# ── the check ────────────────────────────────────────────────────────────────

def _union_of(specs, cache_dir):
    from shapely.ops import unary_union
    geoms, problems = [], []
    for spec in specs:
        try:
            g = parse_poly(fetch_poly(spec, cache_dir))
            if g is None or g.is_empty:
                raise ValueError("parsed to an empty geometry")
            geoms.append((spec, g))
        except Exception as e:
            problems.append((spec, str(e)[:80]))
    return geoms, problems, (unary_union([g for _, g in geoms]) if geoms else None)


def check_coverage(specs, bbox, cache_dir: Path,
                   tolerance_frac=COVERAGE_TOLERANCE_FRAC,
                   reference_specs=None):
    """Do the extracts cover everything in bbox that we need?

    The pass/fail criterion is *relative to a reference* set of extracts when one
    is given -- typically the larger parent region that a small sub-extract was
    substituted for. That is the hazard worth failing on: swapping whole-France
    for Nord-Pas-de-Calais to avoid an OOM must not silently drop French land
    inside the bbox.

    An absolute test cannot be the criterion for water trajectories. Geofabrik
    clip polygons follow land/territorial boundaries, so open sea belongs to no
    extract at all: UK + Nord-Pas-de-Calais leaves 9.5% of the Dover Strait bbox
    "uncovered" purely because it is the middle of the Channel. That fraction is
    reported for information, never failed on. With no reference given, coverage
    is measured against the *mappable* part of the bbox (what any Geofabrik leaf
    claims), so open water is not counted as a gap.

    Returns (ok, message, details). Unverifiable is NOT ok: a missing .poly means
    a gap cannot be ruled out, and a silent partial catalog is worse than a stop.
    """
    from shapely.geometry import box

    west, south, east, north = bbox
    want = box(west, south, east, north)
    mid_lat = (south + north) / 2

    def km2(area):
        return area * 111.0 * 111.0 * math.cos(math.radians(mid_lat))

    geoms, problems, chosen = _union_of(specs, cache_dir)
    details = [{"spec": spec,
                "covers_frac_of_request": (round(g.intersection(want).area / want.area, 4)
                                           if want.area else 0.0)}
               for spec, g in geoms]
    if problems:
        return False, ("cannot verify coverage (refusing to assume): "
                       + "; ".join(f"{s} -> {e}" for s, e in problems)), details
    if chosen is None:
        return False, "no extracts given", details

    abs_gap = want.difference(chosen)
    abs_frac = abs_gap.area / want.area if want.area else 0.0
    info = (f"{100*abs_frac:.1f}% of the bbox is outside every clip polygon "
            f"(~{km2(abs_gap.area):.0f} km2) -- expected on water, where no land "
            f"extract claims the sea")

    if not reference_specs:
        # Compare against the mappable part of the bbox rather than the bbox
        # itself, so open water is not counted as a gap.
        target = mapped_area(bbox, cache_dir)
        if target is None or target.is_empty:
            return True, f"no mapped region overlaps this bbox; {info}", details
        missed = target.difference(chosen)
        missed_frac = missed.area / target.area
        details.append({"mapped_area_km2": round(km2(target.area), 1),
                        "missed_frac_of_mapped": round(missed_frac, 4)})
        if missed_frac <= tolerance_frac:
            return True, (f"extracts cover {100*(1-missed_frac):.1f}% of the mappable "
                          f"area in the bbox; {info}"), details
        mw, ms, me, mn = missed.bounds
        return False, (f"extracts miss {100*missed_frac:.1f}% "
                       f"(~{km2(missed.area):.0f} km2) of the mappable area in this "
                       f"bbox; missing bounds W{mw:.4f} S{ms:.4f} E{me:.4f} "
                       f"N{mn:.4f}. Run with --suggest to list the extracts "
                       f"needed."), details

    _, ref_problems, ref_union = _union_of(reference_specs, cache_dir)
    if ref_problems:
        return False, ("cannot verify against the reference extracts: "
                       + "; ".join(f"{s} -> {e}" for s, e in ref_problems)), details
    ref_in_bbox = want.intersection(ref_union)
    lost = ref_in_bbox.difference(chosen)
    lost_frac = lost.area / ref_in_bbox.area if ref_in_bbox.area else 0.0
    details.append({"reference": list(reference_specs),
                    "reference_area_in_bbox_km2": round(km2(ref_in_bbox.area), 1),
                    "lost_frac_of_reference": round(lost_frac, 4)})

    useless = [d["spec"] for d in details
               if "spec" in d
               and d["covers_frac_of_request"] < MIN_USEFUL_CONTRIBUTION_FRAC]
    if useless:
        return False, (f"extract(s) {useless} contribute essentially nothing to this "
                       f"bbox -- wrong region for this trajectory"), details

    if lost_frac <= tolerance_frac:
        return True, (f"chosen extracts retain {100*(1-lost_frac):.2f}% of the "
                      f"reference coverage inside the bbox (loss is offshore water, "
                      f"not land, when small); {info}"), details

    gw, gs, ge, gn = lost.bounds
    return False, (f"the chosen extracts LOSE {100*lost_frac:.1f}% "
                   f"(~{km2(lost.area):.0f} km2) of the area the reference extracts "
                   f"{list(reference_specs)} cover inside this bbox; missing bounds "
                   f"W{gw:.4f} S{gs:.4f} E{ge:.4f} N{gn:.4f}. Use a larger "
                   f"sub-extract or add another one."), details


GEOFABRIK_INDEX = f"{GEOFABRIK}/index-v1.json"


def load_index(cache_dir: Path):
    """Geofabrik's region index: every downloadable region with its geometry."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "index-v1.json"
    if not dest.exists() or dest.stat().st_size == 0:
        with urllib.request.urlopen(GEOFABRIK_INDEX, timeout=120) as r:
            dest.write_bytes(r.read())
    import json
    return json.loads(dest.read_text())


def leaf_regions(cache_dir: Path):
    """The smallest downloadable regions, with geometry: Geofabrik's leaves."""
    from shapely.geometry import shape

    index = load_index(cache_dir)
    parents = {f["properties"].get("parent") for f in index["features"]}
    out = []
    for feature in index["features"]:
        props = feature["properties"]
        url = (props.get("urls") or {}).get("pbf")
        if not url or props["id"] in parents:
            continue
        try:
            out.append({"id": props["id"],
                        "spec": url.split("/download.geofabrik.de/", 1)[-1],
                        "geom": shape(feature["geometry"])})
        except Exception:
            continue
    return out


def mapped_area(bbox, cache_dir: Path):
    """The part of bbox that any leaf region claims -- i.e. what is mappable.

    This is the right target to measure coverage against, and open water is the
    reason. Geofabrik polygons follow land and territorial boundaries, so the
    middle of the Dover Strait belongs to no leaf region at all; asking for the
    whole bbox to be covered forces continent-sized extracts whose only
    contribution is nominal sea. A 45 km buffer around the Channel crossing
    "needed" europe-latest before this, for 0 extra landmarks.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    want = box(*bbox)
    pieces = [r["geom"].intersection(want) for r in leaf_regions(cache_dir)]
    pieces = [p for p in pieces if not p.is_empty]
    return unary_union(pieces) if pieces else None


def suggest_extracts(bbox, cache_dir: Path):
    """Smallest set of Geofabrik extracts whose union covers the bbox's land.

    Picking sub-extracts by hand does not scale: a 45 km buffer around a Channel
    crossing reaches across three English counties and into Belgian waters, and
    guessing wrong is exactly what the coverage gate then rejects. So ask the
    index which regions actually intersect, and prefer the deepest (smallest)
    ones -- a region with children is always larger than the children needed.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union

    want = box(*bbox)
    target = mapped_area(bbox, cache_dir)
    if target is None or target.is_empty:
        return [], 0.0

    candidates = []
    for region in leaf_regions(cache_dir):
        overlap = region["geom"].intersection(want)
        if overlap.is_empty or overlap.area <= 0:
            continue
        candidates.append({**region, "overlap": overlap.area})

    # Greedy by ascending region size, not by overlap. Leaf-ness is not a size
    # proxy: Geofabrik ships convenience aggregates with no children, so
    # britain-and-ireland (~6 GB, the whole isles) is as much a "leaf" as kent
    # (51 MB). Taking the smallest regions first means counties are picked before
    # countries, and a big aggregate is only reached for a remainder nothing
    # smaller covers.
    chosen, covered = [], None
    remaining = target
    for cand in sorted(candidates, key=lambda c: c["geom"].area):
        gain = cand["geom"].intersection(remaining)
        if gain.is_empty or gain.area / target.area < 0.005:
            continue
        chosen.append(cand)
        covered = gain if covered is None else unary_union([covered, gain])
        remaining = target.difference(covered)
        if remaining.is_empty or remaining.area / target.area < 0.005:
            break
    return chosen, (covered.area / target.area if covered is not None else 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="*",
                    help="Geofabrik-relative spec, e.g. europe/france/nord-pas-de-calais-latest.osm.pbf")
    ap.add_argument("--poly_cache_dir", type=Path, required=True,
                    help="Directory for cached .poly clip boundaries and the "
                         "Geofabrik index (the collection orchestrator uses "
                         "<osm_cache_dir>/poly; the old hardcoded location was "
                         "~/scratch/osm_downloads/poly)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--suggest", action="store_true",
                    help="List the smallest Geofabrik extracts covering --bbox")
    ap.add_argument("--reference", nargs="+", default=None,
                    help="Parent extract(s) the chosen ones substitute for; "
                         "coverage is failed relative to these")
    args = ap.parse_args(argv)

    if args.suggest:
        if not args.bbox:
            print("--suggest needs --bbox")
            return 2
        chosen, frac = suggest_extracts(tuple(args.bbox), args.poly_cache_dir)
        print(f"extracts covering {100*frac:.1f}% of the bbox:")
        for c in chosen:
            print(f"    {c['spec']}")
        print("\nregistry form:")
        print("        \"osm\": [" + ",\n                ".join(
            f'"{c["spec"]}"' for c in chosen) + "],")
        return 0

    for spec in args.spec:
        try:
            g = parse_poly(fetch_poly(spec, args.poly_cache_dir))
            print(f"  {spec}\n      clip polygon bounds "
                  f"{tuple(round(v, 4) for v in g.bounds)}")
        except Exception as e:
            print(f"  {spec}\n      ERROR {e}")

    if args.bbox:
        ok, msg, details = check_coverage(args.spec, tuple(args.bbox),
                                          args.poly_cache_dir,
                                          reference_specs=args.reference)
        print(f"\n{'OK: ' if ok else 'FAIL: '}{msg}")
        for d in details:
            if "spec" in d:
                print(f"    {d['spec'].rsplit('/', 1)[-1]}: covers "
                      f"{100*d['covers_frac_of_request']:.1f}% of the request")
            elif "reference" in d:
                print(f"    reference {d['reference']}: "
                      f"{d['reference_area_in_bbox_km2']} km2 in bbox, "
                      f"lost {100*d['lost_frac_of_reference']:.2f}%")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
