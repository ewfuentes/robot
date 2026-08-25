#!/usr/bin/env python3
"""Verify that a set of Geofabrik extracts really covers a request bbox.

An OSM extractor silently returns only the portion of a request that lies
inside its input PBF. Requests that cross extract boundaries therefore need an
explicit coverage check; otherwise a partial catalog looks like a successful
one.

The check uses each extract's Geofabrik **.poly** clip boundary, not its PBF
HeaderBBox. A header bbox is only the clip's bounding rectangle, so concave
boundaries, holes, coastlines, and adjacent administrative areas inside that
rectangle are not evidence that the PBF contains those places.

    bazel run //experimental/overhead_matching/swag/farfield/collection:pbf_coverage -- \\
        --poly_cache_dir <osm_cache>/poly <geofabrik-spec>
    ... -- --poly_cache_dir <osm_cache>/poly --bbox W S E N <spec> [<spec> ...]
"""

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact

GEOFABRIK = "https://download.geofabrik.de"
# Collection must retain essentially all mappable land.  Loading a national
# extract is now fast enough that there is no alternate substitution policy.
NORMAL_COVERAGE_TOLERANCE_FRAC = 0.005


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
    dest = poly_cache_path(spec, cache_dir)
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_file() or dest.stat().st_size <= 0:
            raise ValueError(f"invalid cached Geofabrik polygon: {dest}")
        return dest
    url = poly_url_for(spec)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    if not data:
        raise ValueError(f"empty Geofabrik polygon response for {spec}")
    try:
        artifact.atomic_create_file(dest, data)
    except FileExistsError:
        if (dest.is_symlink() or not dest.is_file()
                or artifact.sha256_file(dest)
                != hashlib.sha256(data).hexdigest()):
            raise ValueError(f"concurrent polygon cache collision: {dest}")
    return dest


def poly_cache_path(spec: str, cache_dir: Path) -> Path:
    """Collision-safe cache path for one full Geofabrik relative spec."""
    basename = poly_url_for(spec).rsplit("/", 1)[-1]
    key = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{key}-{basename}"


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


def check_coverage(specs, bbox, cache_dir: Path, *, pbf_paths=None):
    """Do the extracts cover everything in bbox that we need?

    Geofabrik clip polygons follow land and territorial boundaries, so open sea
    may belong to no extract. Absolute bbox coverage is reported for context,
    while the pass/fail decision measures the *mappable* part of the bbox (what
    any Geofabrik leaf claims). Open water is therefore not counted as a gap.

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
    if pbf_paths is not None:
        if len(pbf_paths) != len(specs):
            problems.append(("PBF inputs", "spec/path count mismatch"))
        else:
            from shapely.geometry import box as shapely_box
            by_spec = {spec: geometry for spec, geometry in geoms}
            for spec, pbf_path in zip(specs, pbf_paths, strict=True):
                path = Path(pbf_path)
                try:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("not a regular file")
                    expected_name = spec.rsplit("/", 1)[-1]
                    if expected_name.endswith("-latest.osm.pbf"):
                        prefix = expected_name.removesuffix(
                            "latest.osm.pbf")
                        if not re.fullmatch(
                                re.escape(prefix) + r"\d{6}\.osm\.pbf",
                                path.name):
                            raise ValueError(
                                "filename does not identify the requested spec")
                    elif path.name != expected_name:
                        raise ValueError(
                            "filename does not identify the requested spec")
                    header = header_bbox(path)
                    if (header is None or len(header) != 4
                            or not all(math.isfinite(value) for value in header)
                            or header[0] >= header[2]
                            or header[1] >= header[3]):
                        raise ValueError("missing or invalid HeaderBBox")
                    geometry = by_spec.get(spec)
                    if geometry is None:
                        raise ValueError("clip polygon was not validated")
                    required = geometry.intersection(want)
                    if (not required.is_empty
                            and not shapely_box(*header).buffer(1e-9).covers(
                                required)):
                        raise ValueError(
                            "HeaderBBox does not cover the requested clip area")
                    identity = {
                        "path": str(path.resolve()),
                        "sha256": artifact.sha256_file(path),
                        "header_bbox": list(header),
                    }
                    next(item for item in details
                         if item["spec"] == spec)["pbf"] = identity
                except (OSError, ValueError, artifact.ArtifactError) as error:
                    problems.append((spec, f"invalid PBF {path}: {error}"))
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

    # Compare against the mappable part of the bbox rather than the bbox itself,
    # so open water is not counted as a gap.
    target = mapped_area(bbox, cache_dir)
    if target is None or target.is_empty:
        return False, (
            "cannot verify coverage: the Geofabrik index contains no mappable "
            f"region overlapping this bbox; {info}"), details
    missed = target.difference(chosen)
    missed_frac = missed.area / target.area
    details.append({"mapped_area_km2": round(km2(target.area), 1),
                    "missed_frac_of_mapped": round(missed_frac, 4)})
    details.append({
        "coverage_policy": "mappable_land",
        "tolerance_frac": NORMAL_COVERAGE_TOLERANCE_FRAC,
    })
    if missed_frac <= NORMAL_COVERAGE_TOLERANCE_FRAC:
        return True, (f"extracts cover {100*(1-missed_frac):.1f}% of the mappable "
                      f"area in the bbox; {info}"), details
    mw, ms, me, mn = missed.bounds
    return False, (f"extracts miss {100*missed_frac:.1f}% "
                   f"(~{km2(missed.area):.0f} km2) of the mappable area in this "
                   f"bbox; missing bounds W{mw:.4f} S{ms:.4f} E{me:.4f} "
                   f"N{mn:.4f}. Run with --suggest to list the extracts "
                   f"needed."), details


GEOFABRIK_INDEX = f"{GEOFABRIK}/index-v1.json"


def load_index(cache_dir: Path):
    """Geofabrik's region index: every downloadable region with its geometry."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "index-v1.json"
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_file() or dest.stat().st_size <= 0:
            raise ValueError(f"invalid cached Geofabrik index: {dest}")
    else:
        with urllib.request.urlopen(GEOFABRIK_INDEX, timeout=120) as r:
            data = r.read()
        if not data:
            raise ValueError("empty Geofabrik index response")
        try:
            artifact.atomic_create_file(dest, data)
        except FileExistsError:
            if (dest.is_symlink() or not dest.is_file()
                    or artifact.sha256_file(dest)
                    != hashlib.sha256(data).hexdigest()):
                raise ValueError(f"concurrent index cache collision: {dest}")
    try:
        return json.loads(
            dest.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid Geofabrik index {dest}: {error}") from error


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def leaf_regions(cache_dir: Path):
    """The smallest downloadable regions, with geometry: Geofabrik's leaves."""
    from shapely.geometry import shape

    index = load_index(cache_dir)
    if (not isinstance(index, dict)
            or not isinstance(index.get("features"), list)):
        raise ValueError("Geofabrik index must contain a features list")
    normalized = []
    for position, feature in enumerate(index["features"]):
        if not isinstance(feature, dict):
            raise ValueError(f"Geofabrik feature {position} is not an object")
        props = feature.get("properties")
        if (not isinstance(props, dict)
                or not isinstance(props.get("id"), str)
                or not props["id"]):
            raise ValueError(f"Geofabrik feature {position} has no valid id")
        parent = props.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError(f"Geofabrik feature {props['id']} has invalid parent")
        urls = props.get("urls")
        if urls is not None and not isinstance(urls, dict):
            raise ValueError(f"Geofabrik feature {props['id']} has invalid urls")
        normalized.append((feature, props))
    parents = {props.get("parent") for _, props in normalized}
    out = []
    for feature, props in normalized:
        url = (props.get("urls") or {}).get("pbf")
        if not url or props["id"] in parents:
            continue
        if not isinstance(url, str):
            raise ValueError(f"Geofabrik feature {props['id']} has invalid PBF URL")
        geometry = shape(feature.get("geometry"))
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Geofabrik feature {props['id']} has invalid geometry")
        out.append({"id": props["id"],
                    "spec": url.split("/download.geofabrik.de/", 1)[-1],
                    "geom": geometry})
    return out


def mapped_area(bbox, cache_dir: Path):
    """The part of bbox that any leaf region claims -- i.e. what is mappable.

    Geofabrik polygons follow land and territorial boundaries, so open water
    may belong to no leaf region. Measuring against the mapped area avoids
    requiring larger extracts merely to cover sea outside every source clip.
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
                         "Geofabrik index")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--suggest", action="store_true",
                    help="List the smallest Geofabrik extracts covering --bbox")
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
        ok, msg, details = check_coverage(
            args.spec, tuple(args.bbox), args.poly_cache_dir)
        print(f"\n{'OK: ' if ok else 'FAIL: '}{msg}")
        for d in details:
            if "spec" in d:
                print(f"    {d['spec'].rsplit('/', 1)[-1]}: covers "
                      f"{100*d['covers_frac_of_request']:.1f}% of the request")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
