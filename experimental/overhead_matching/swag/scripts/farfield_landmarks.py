"""Fetch the OSM classes a far-field observer could actually name, via Overpass.

This is deliberately *not* the landmark extractor the datasets use. That one runs
against a country PBF and keeps everything, because the filter's catalog wants
breadth (see `trim_landmark_feather`'s note on `harbor_catalog`). This one exists
to answer a screening question over regions no PBF has been downloaded for yet:
"is there anything tall enough to see from here?"

What belongs in a far-field catalog follows from curvature, not from taste. With
an observer 2 m up, `farfield_viewshed.horizon_range_km` gives:

    landmark          height    max range
    park bench           1 m        9 km      <- everything a harbour catalog
    house               10 m       18 km         is full of, and useless here
    water tower         30 m       26 km
    church spire        45 m       31 km
    chimney             80 m       40 km
    radio mast         150 m       52 km
    hill              1000 m      126 km
    alpine peak       3000 m      215 km

So the vocabulary below is everything that clears roughly 20 m, and nothing else.
Street furniture is not merely noisy at this range, it is geometrically incapable
of being seen.

Height handling is the part that silently ruins results if it is wrong, and it
splits the vocabulary in two:

  * **Terrain features** (`peak`, `volcano`, `hill`, `saddle`) are already in the
    DEM. Their height must be *read from* it, and `structure_height_m` must be 0.
    Trusting the `ele` tag instead double-counts, putting Mont Blanc at 9.6 km.
  * **Structures** (masts, chimneys, spires, turbines) are absent from the DEM,
    which models bare ground. Their height must be *added* to it, or they never
    clear a horizon they physically dominate.

`ele` on a peak is still carried through, in the `ele_m` field, purely as a
cross-check against the DEM -- never as the height. SRTM systematically
under-reads sharp summits (Mont Blanc comes back 4786 m against a true 4808 m),
so a shortfall of tens of metres is expected and one of hundreds means the node
is mistagged or misplaced.

    bazel run //experimental/overhead_matching/swag/scripts:farfield_landmarks -- \\
        --bbox 6.0 46.2 7.4 46.9 --output /tmp/geneva_landmarks.json
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import requests

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Not cosmetic -- see fetch_overpass. The default `python-requests/x.y.z` is
# refused with HTTP 406 by overpass-api.de.
OVERPASS_USER_AGENT = "robot-farfield-crossview/1.0 (research; landmark screening)"

# (osm_key, osm_value, class name, default height when untagged, already_in_dem).
#
# Default heights are priors for nodes with no `height` tag, chosen as rough
# class medians. They are a screening convenience and nothing more -- a real
# height tag always wins, and the output records which was used in
# `height_source` so a suspicious result can be traced.
VOCABULARY = [
    ("natural", "peak", "natural:peak", 0.0, True),
    ("natural", "volcano", "natural:volcano", 0.0, True),
    ("natural", "hill", "natural:hill", 0.0, True),

    # Deliberately absent: `natural=cliff`. It is a *linear* way, and this
    # catalog is points, so a cliff enters as the centroid of its line -- a
    # position the observer sees nothing at, on a feature with no name to match
    # against. It is also not a rounding error: over the Lake Geneva bbox
    # cliffs were 18,741 of 31,486 rows, 60% of the catalog, all of it inflating
    # visibility counts with things no matcher can associate. Linear landmarks
    # need a different representation than this module has.

    ("man_made", "mast", "man_made:mast", 100.0, False),
    ("man_made", "communications_tower", "man_made:communications_tower", 150.0, False),
    ("man_made", "tower", "man_made:tower", 40.0, False),
    ("man_made", "chimney", "man_made:chimney", 80.0, False),
    ("man_made", "water_tower", "man_made:water_tower", 30.0, False),
    ("man_made", "lighthouse", "man_made:lighthouse", 25.0, False),
    ("man_made", "silo", "man_made:silo", 25.0, False),
    ("man_made", "storage_tank", "man_made:storage_tank", 18.0, False),
    ("man_made", "windmill", "man_made:windmill", 20.0, False),
    ("man_made", "cooling_tower", "man_made:cooling_tower", 120.0, False),
    ("man_made", "crane", "man_made:crane", 50.0, False),
    # Most `man_made=bridge` outlines are ordinary road crossings, not the
    # 250 m-pylon kind, so the untagged default is modest and the ones that
    # actually break a skyline are expected to carry a real `height`.
    ("man_made", "bridge", "man_made:bridge", 15.0, False),

    # Wind turbines. Tagged as generators, and the useful height is blade tip
    # rather than hub -- the tip is what breaks a skyline. OSM records hub
    # height in `height` when it records anything, so the 150 m default is a
    # whole-turbine figure for a modern onshore unit.
    ("generator:source", "wind", "power:wind_turbine", 150.0, False),

    ("building", "church", "building:church", 35.0, False),
    ("building", "cathedral", "building:cathedral", 60.0, False),
    ("building", "minaret", "building:minaret", 40.0, False),
    ("tower:type", "spire", "tower:spire", 45.0, False),
    ("tower:type", "observation", "tower:observation", 60.0, False),
    ("tower:type", "cooling", "tower:cooling", 120.0, False),

    ("historic", "castle", "historic:castle", 25.0, False),
    ("historic", "fort", "historic:fort", 20.0, False),

    ("seamark:type", "landmark", "seamark:landmark", 25.0, False),
    ("seamark:type", "light_major", "seamark:light_major", 30.0, False),
    ("seamark:type", "light_minor", "seamark:light_minor", 12.0, False),
]

# Skyscrapers have no single tag; they are ordinary buildings with a big
# `height`. Queried separately with a height predicate so the result set stays
# small -- an unfiltered `building` query over a city is millions of rows.
MIN_TALL_BUILDING_M = 60.0


def _parse_length_m(value) -> float | None:
    """OSM height/ele -> metres, or None.

    Tolerates the units people actually write: bare metres, "12 m", "40'", and
    feet-inches like `123'4"`. A value that does not parse returns None so the
    caller falls back to the class default rather than silently reading 0.
    """
    if value is None:
        return None
    text = str(value).strip().lower().replace(",", ".")
    if not text:
        return None

    feet_inches = re.fullmatch(r"(\d+(?:\.\d+)?)'\s*(?:(\d+(?:\.\d+)?)\")?", text)
    if feet_inches:
        feet = float(feet_inches.group(1))
        inches = float(feet_inches.group(2) or 0.0)
        return (feet * 12 + inches) * 0.0254

    match = re.match(r"^(-?\d+(?:\.\d+)?)\s*(m|metres?|meters?|ft|feet)?$", text)
    if not match:
        return None
    magnitude = float(match.group(1))
    unit = match.group(2) or "m"
    return magnitude * (0.3048 if unit in ("ft", "feet") else 1.0)


def build_query(bbox, timeout_s: int = 180, tall_buildings: bool = True) -> str:
    """Overpass QL for one cell.

    Note the bbox transpose: Overpass filters are (south, west, north, east),
    while (west, south, east, north) is used everywhere else in this pipeline
    and by Mapillary. Reversing it queries a different part of the world
    without erroring.
    """
    west, south, east, north = bbox
    area = f"{south},{west},{north},{east}"
    clauses = []
    for key, value, _kind, _height, _in_dem in VOCABULARY:
        for element in ("node", "way"):
            clauses.append(f'{element}["{key}"="{value}"]({area});')
    if tall_buildings:
        # Filter on height server-side. Over Lausanne, 1,201 buildings carry a
        # height tag and 2 of them clear 60 m, so doing this in Python instead
        # would transfer 600x the data for the same answer -- and it is that
        # volume, not the tag lookup, that pushes a cell into a 504.
        clauses.append(
            f'way["building"]["height"]'
            f'(if: number(t["height"]) >= {MIN_TALL_BUILDING_M:g})({area});')
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout center tags;"


def _split_bbox(bbox, max_span_deg: float):
    """Grid a bbox so no cell exceeds max_span_deg on a side.

    Overpass has no hard area cap the way `/images` does, but it does have a
    wall-clock timeout, and a query that dies at 180 s returns nothing at all
    rather than a partial result. Splitting up front is cheaper than discovering
    that after three minutes.
    """
    west, south, east, north = bbox
    n_x = max(1, math.ceil((east - west) / max_span_deg))
    n_y = max(1, math.ceil((north - south) / max_span_deg))
    for i in range(n_x):
        for j in range(n_y):
            yield (west + (east - west) * i / n_x,
                   south + (north - south) * j / n_y,
                   west + (east - west) * (i + 1) / n_x,
                   south + (north - south) * (j + 1) / n_y)


def fetch_overpass(query: str, max_retries: int = 3, start_mirror: int = 0) -> dict:
    """Run one Overpass query, rotating mirrors on failure.

    Two failure modes that do not look like what they are:

      * **The User-Agent header is mandatory.** overpass-api.de rejects
        `requests`' default `python-requests/x.y.z` with HTTP 406 Not
        Acceptable, for every query, valid or not. The identical query with any
        other UA returns 200. 406 reads like a malformed-query error and sends
        you off debugging Overpass QL, so it is pinned here.
      * **Overload arrives as a 200** carrying an HTML error page, so the body
        is sniffed rather than trusting the status code alone.

    Queries go by POST: they run to several kilobytes with this vocabulary, and
    that is the documented transport for large ones.

    `start_mirror` lets the caller spread *successful* traffic across mirrors
    rather than only failing over. It matters: rotating on failure alone sends
    every cell to overpass-api.de first, and a multi-cell sweep gets throttled
    into minutes per cell while the other mirror answers the same query in 45 s.
    """
    headers = {"User-Agent": OVERPASS_USER_AGENT}
    last_error = None
    overloaded = False
    for attempt in range(max_retries):
        url = OVERPASS_URLS[(start_mirror + attempt) % len(OVERPASS_URLS)]
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=300)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 200 and resp.text.lstrip().startswith("{"):
            return resp.json()
        if resp.status_code in (504, 429):
            overloaded = True
        last_error = f"HTTP {resp.status_code} from {url}: {resp.text[:200]}"
        time.sleep(10 * (attempt + 1))
    if overloaded:
        raise OverpassOverloaded(last_error)
    raise RuntimeError(f"Overpass failed after {max_retries} attempts: {last_error}")


def _classify(tags: dict) -> tuple[str, float, bool] | None:
    """(kind, default_height_m, in_dem) for an element, or None if out of scope.

    First match in VOCABULARY order wins, which puts the specific classes ahead
    of `man_made=tower`; a node tagged both `man_made=tower` and
    `tower:type=communication` should not be filed as a generic 40 m tower.
    """
    for key, value, kind, height, in_dem in VOCABULARY:
        if tags.get(key) == value:
            return kind, height, in_dem
    if "building" in tags:
        parsed = _parse_length_m(tags.get("height"))
        if parsed is not None and parsed >= MIN_TALL_BUILDING_M:
            return "building:tall", parsed, False
    return None


def elements_to_landmarks(elements: list[dict]) -> list[dict]:
    seen_ids = set()
    out = []
    for element in elements:
        tags = element.get("tags") or {}
        classified = _classify(tags)
        if classified is None:
            continue
        kind, default_height, in_dem = classified

        if element["type"] == "node":
            lat, lon = element.get("lat"), element.get("lon")
        else:
            center = element.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue

        osm_id = f"{element['type']}/{element['id']}"
        if osm_id in seen_ids:
            continue
        seen_ids.add(osm_id)

        if in_dem:
            structure_height, height_source = 0.0, "dem"
        else:
            tagged = _parse_length_m(tags.get("height"))
            if tagged is None:
                tagged = _parse_length_m(tags.get("building:height"))
            if tagged is None:
                levels = _parse_length_m(tags.get("building:levels"))
                tagged = levels * 3.2 if levels else None
                height_source = "levels" if tagged else "default"
            else:
                height_source = "tag"
            structure_height = tagged if tagged else default_height

        out.append({
            "lat": float(lat), "lon": float(lon), "kind": kind,
            "name": tags.get("name") or tags.get("ref") or "",
            "structure_height_m": float(structure_height),
            "in_dem": bool(in_dem),
            "osm_id": osm_id,
            "height_source": height_source,
            "ele_m": _parse_length_m(tags.get("ele")),
        })
    return out


class OverpassOverloaded(RuntimeError):
    """The cell is too expensive to answer; split it rather than retrying.

    Distinguished from an ordinary failure the same way `mapillary_lib.api`
    separates `MapillaryQueryTooLarge` from a transient 500: retrying an
    overloaded query at the same size burns the timeout again and still fails,
    while halving the cell succeeds immediately.
    """


def _fetch_cell(cell, min_cell_deg: float, verbose: bool,
                start_mirror: int = 0) -> list[dict]:
    """One cell, subdividing on timeout until it fits.

    A 0.75 degree cell answers in ~20 s over the Alps, but density varies by
    more than an order of magnitude between an alpine massif and open water, so
    a fixed grid cannot be sized correctly in advance. Splitting on failure
    adapts to whatever is actually there.
    """
    try:
        data = fetch_overpass(build_query(cell), start_mirror=start_mirror)
        return elements_to_landmarks(data.get("elements", []))
    except OverpassOverloaded:
        west, south, east, north = cell
        if min(east - west, north - south) <= min_cell_deg:
            # Refusing here rather than returning [] matters: a silently empty
            # cell reads downstream as "nothing tall in this region", which is
            # the opposite of what an overload means.
            raise RuntimeError(f"cell {cell} still overloaded at minimum size")
        if verbose:
            print(f"    cell {cell} overloaded, splitting", file=sys.stderr, flush=True)
        mid_x, mid_y = (west + east) / 2, (south + north) / 2
        found = []
        for i, quadrant in enumerate(((west, south, mid_x, mid_y),
                                      (mid_x, south, east, mid_y),
                                      (west, mid_y, mid_x, north),
                                      (mid_x, mid_y, east, north))):
            found.extend(_fetch_cell(quadrant, min_cell_deg, verbose,
                                     start_mirror + i))
        return found


def fetch_landmarks(bbox, max_span_deg: float = 0.75, verbose: bool = True,
                    min_cell_deg: float = 0.05) -> list[dict]:
    cells = list(_split_bbox(bbox, max_span_deg))
    all_landmarks: dict[str, dict] = {}
    for i, cell in enumerate(cells, 1):
        found = _fetch_cell(cell, min_cell_deg, verbose, start_mirror=i)
        for landmark in found:
            all_landmarks[landmark["osm_id"]] = landmark
        if verbose:
            # flush=True: Python block-buffers a redirected stderr, so a long
            # Overpass sweep otherwise prints nothing for many minutes.
            print(f"  [{i}/{len(cells)}] {len(found):5d} landmarks "
                  f"({len(all_landmarks)} unique)", file=sys.stderr, flush=True)
    return list(all_landmarks.values())


def summarise(landmarks: list[dict]) -> dict:
    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for landmark in landmarks:
        counts[landmark["kind"]] = counts.get(landmark["kind"], 0) + 1
        sources[landmark["height_source"]] = sources.get(landmark["height_source"], 0) + 1
    named = sum(1 for landmark in landmarks if landmark["name"])
    return {
        "n": len(landmarks),
        "n_named": named,
        "by_kind": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "height_source": sources,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, required=True,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_span_deg", type=float, default=0.75)
    parser.add_argument("--min_structure_height_m", type=float, default=0.0,
                        help="drop structures shorter than this (peaks unaffected)")
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    print(f"Overpass over {bbox}", file=sys.stderr)
    landmarks = fetch_landmarks(bbox, args.max_span_deg)

    if args.min_structure_height_m > 0:
        before = len(landmarks)
        landmarks = [landmark for landmark in landmarks
                     if landmark["in_dem"]
                     or landmark["structure_height_m"] >= args.min_structure_height_m]
        print(f"  height filter: {before} -> {len(landmarks)}", file=sys.stderr)

    stats = summarise(landmarks)
    payload = {"bbox": list(bbox), "summary": stats, "landmarks": landmarks}
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
