#!/usr/bin/env python3
"""Find candidate far-field trajectories in a region, without a seed link.

Every trajectory in `farfield_trajectories.py` started as a Mapillary app URL
someone found by browsing. That does not scale, and it biases the collection
toward what is easy to stumble onto -- which is how 22 of 22 entries ended up
being boats.

This inverts it. Given a region, pull every sequence from the coverage vector
tiles, filter to the ones that could plausibly carry a far-field capture, and
emit them as candidates. Scoring happens next, in the repo:

    # 1. what tracks are there?
    bazel run //experimental/overhead_matching/swag/mapillary_tools:discover_tracks -- --region geneva_lakeshore --output /tmp/geneva_tracks.json

    # 2. what could they see? (bazel, needs no token)
    bazel run //experimental/overhead_matching/swag/scripts:farfield_landmarks -- \\
        --bbox 6.10 46.30 7.00 46.60 --output /tmp/geneva_landmarks.json
    bazel run //experimental/overhead_matching/swag/scripts:farfield_viewshed -- \\
        --tracks /tmp/geneva_tracks.json --landmarks /tmp/geneva_landmarks.json \\
        --output /tmp/geneva_scored.json

    # 3. registry stanzas for the winners
    bazel run //experimental/overhead_matching/swag/mapillary_tools:discover_tracks -- --region geneva_lakeshore --scored /tmp/geneva_scored.json \\
        --emit_registry --top 5

Why tiles and not `/images`: the Graph API rejects any bbox over 0.010 square
degrees and separately rejects dense areas on volume, both as HTTP 500. A region
sweep with it subdivides exponentially and never finishes -- the collection
README records depth 10 in SF, Seattle and London. One z12 tile covers ~10 km
with no cap at all. See `mapillary_lib/vector_tiles.py`.

The handoff back to the existing pipeline is `image_id`, which the sequence
layer carries: it is exactly the seed pKey that `seed_to_trajectory.py` stitches
from. So a discovered candidate drops straight into the registry with no manual
URL-copying step, and stitching still runs the same way -- which matters,
because a tile sequence is one Mapillary fragment, and the whole trip is
typically an order of magnitude longer (Folkestone: 500 images to 10,711).
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.farfield_regions import REGIONS, geometries
from experimental.overhead_matching.swag.mapillary_tools.vector_tiles import (
    VectorTileClient, merge_tile_features, polyline_length_km, tiles_for_bbox,
)

# z12 is ~10 km per tile at the equator, so a region is tens of tiles rather
# than the ~1,200 z14 would need. Geometry is quantised to the tile's 4096-unit
# grid, giving ~2.4 m positions -- far finer than anything screening cares
# about, and the tracks are re-fetched at full resolution by stage 1 anyway.
DEFAULT_ZOOM = 12


def fetch_region_sequences(client: VectorTileClient, bbox, zoom: int = DEFAULT_ZOOM,
                           workers: int = 8, verbose: bool = True) -> dict:
    """Every sequence intersecting bbox, reassembled across tile boundaries."""
    tiles = tiles_for_bbox(bbox, zoom)
    if verbose:
        print(f"  {len(tiles)} tiles at z{zoom}", file=sys.stderr)

    features = []
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.fetch, zoom, x, y): (x, y) for x, y in tiles}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                layers = future.result()
            except Exception as exc:
                errors += 1
                print(f"    tile {futures[future]} failed: {exc}", file=sys.stderr)
                continue
            features.extend(layers.get("sequence", []))
            if verbose and i % 25 == 0:
                print(f"    {i}/{len(tiles)} tiles, {len(features)} fragments",
                      file=sys.stderr)

    if errors:
        # Loud, because a dropped tile is a hole in the search that looks
        # identical to "no coverage there".
        print(f"  WARNING: {errors}/{len(tiles)} tiles failed", file=sys.stderr)

    merged = merge_tile_features(features)
    if verbose:
        print(f"  {len(features)} fragments -> {len(merged)} sequences",
              file=sys.stderr)
    return merged


def _clip_to_bbox(coords, bbox):
    """Keep only vertices inside the region.

    A sequence that merely clips the corner of the region arrives with its whole
    geometry, so its length and its sampled observer positions would describe a
    track mostly outside the area being searched.
    """
    west, south, east, north = bbox
    return [(lon, lat) for lon, lat in coords
            if west <= lon <= east and south <= lat <= north]


def filter_candidates(sequences: dict, bbox, min_length_km: float = 2.0,
                      pano_only: bool = False, exclude_foot: bool = True,
                      after_year: int = None, min_quality: float = 0.0,
                      max_fragments: int = None) -> list[dict]:
    """Screen sequences down to plausible far-field candidates.

    `exclude_foot` defaults on because the downstream pipeline wants a vehicle:
    mount-offset calibration solves for a fixed camera-to-travel-direction
    angle, and a handheld or body-worn capture has no such angle to find. It is
    a filter on usability, not on view quality -- a walker on a lakeshore sees
    exactly what a driver does.
    """
    out = []
    for seq_id, seq in sequences.items():
        props = seq["properties"]
        coords = _clip_to_bbox(seq["coords"], bbox)
        if len(coords) < 2:
            continue

        if pano_only and not props.get("is_pano"):
            continue
        if exclude_foot and props.get("foot"):
            continue
        if props.get("quality_score") is not None and props["quality_score"] < min_quality:
            continue
        if max_fragments is not None and seq["n_fragments"] > max_fragments:
            continue

        captured_ms = props.get("captured_at")
        year = None
        if captured_ms:
            year = datetime.fromtimestamp(captured_ms / 1000, timezone.utc).year
            if after_year and year < after_year:
                continue

        length_km = polyline_length_km(coords)
        if length_km < min_length_km:
            continue

        out.append({
            "sequence_id": seq_id,
            "seed_image_id": str(props.get("image_id", "")),
            "is_pano": bool(props.get("is_pano")),
            "foot": bool(props.get("foot")),
            "creator_id": str(props.get("creator_id", "")),
            "captured_ms": captured_ms,
            "captured_year": year,
            "quality_score": props.get("quality_score"),
            "length_km": round(length_km, 2),
            "n_fragments": seq["n_fragments"],
            "n_vertices": len(coords),
            "coords": [[round(lon, 6), round(lat, 6)] for lon, lat in coords],
        })
    out.sort(key=lambda r: -r["length_km"])
    return out


def summarise(candidates: list[dict]) -> dict:
    if not candidates:
        return {"n": 0}
    pano = [c for c in candidates if c["is_pano"]]
    years = [c["captured_year"] for c in candidates if c["captured_year"]]
    return {
        "n": len(candidates),
        "n_pano": len(pano),
        "total_km": round(sum(c["length_km"] for c in candidates), 1),
        "pano_km": round(sum(c["length_km"] for c in pano), 1),
        "longest_km": candidates[0]["length_km"],
        "year_range": [min(years), max(years)] if years else None,
        "n_creators": len({c["creator_id"] for c in candidates}),
    }


MAPILLARY_APP_URL = "https://www.mapillary.com/app/?pKey={pkey}&focus=photo&lat={lat:.5f}&lng={lon:.5f}&z=14"


def track_url(candidate: dict) -> str:
    """A Mapillary app link that opens this capture, with the map on it.

    `pKey` is the seed image; `lat`/`lng` only position the *viewport*. The
    collection guide records a trajectory mis-registered because the URL's
    lat/lng was read as the image position -- it is not, so these are set from
    the track's own midpoint purely so the map opens somewhere useful.
    """
    coords = candidate["coords"]
    mid = coords[len(coords) // 2]
    return MAPILLARY_APP_URL.format(pkey=candidate["seed_image_id"],
                                    lon=mid[0], lat=mid[1])


def _endpoints(candidate: dict):
    coords = candidate["coords"]
    return coords[0], coords[-1]


def _gap_km(a, b) -> float:
    import math
    mid_lat = math.radians((a[1] + b[1]) / 2)
    return math.hypot((b[0] - a[0]) * 111.320 * math.cos(mid_lat),
                      (b[1] - a[1]) * 110.574)


def cluster_tracks(candidates: list[dict], max_gap_km: float = 3.0,
                   max_time_gap_h: float = 2.0) -> list[list[dict]]:
    """Group candidates that are probably one trip.

    Three conditions, all required: **same creator**, **endpoints within
    max_gap_km**, and **start times within max_time_gap_h**. Proximity alone
    over-merges badly on a road like CO-82, where a dozen unrelated drivers
    cover the same 40 km and would collapse into one fictional trip. Requiring
    the creator to match keeps a cluster to what `seed_to_trajectory.py` could
    actually stitch, since Mapillary splits one capture at 500 or 1000 images
    and the pieces keep their uploader. Without the time condition a mapper who
    starts from home for eight years chains into one multi-year "trip" — the
    group is then a campaign, and its km total describes nothing collectable.

    The time window is hours, not the stitcher's 300 s, because tiles carry one
    timestamp per sequence — the capture start — so two fragments of one outing
    differ by the whole duration of the earlier fragment (a 500-image fragment
    at photo cadence runs tens of minutes; a slow ferry fragment can exceed an
    hour). 2 h tells outings apart from campaigns without splitting a long
    crossing. A candidate with no timestamp never merges (strict: an unknown
    time is not evidence of adjacency); tracks files saved before captured_ms
    was recorded therefore cluster per-sequence. max_time_gap_h <= 0 disables
    the condition.

    A timestamp repeated verbatim across many of one creator's sequences is
    treated as missing for the same reason. A working clock cannot stamp two
    fragment starts with the same millisecond, let alone 180 of them — which is
    exactly what a clock-less camera in Denver did (every fragment of a
    1,200 km campaign says 1989-05-29 06:01:00), and identical stamps satisfy
    "adjacent in time" vacuously. Same lesson as kurashiki's all-zero
    compass_angle: a field that is identical everywhere is not data.

    This is a hint for reading the list, not a claim: stitching is decided by
    sequence continuity in stage 1, which has the per-image timestamps and
    positions this does not.
    """
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    max_time_gap_ms = max_time_gap_h * 3600e3 if max_time_gap_h and \
        max_time_gap_h > 0 else None

    # 2-3 equal stamps could be rounding; more is a broken or defaulted clock.
    stamp_counts = Counter((c["creator_id"], c.get("captured_ms"))
                           for c in candidates
                           if c.get("captured_ms") is not None)

    def reliable_time(c):
        t = c.get("captured_ms")
        if t is None or stamp_counts[(c["creator_id"], t)] > 3:
            return None
        return t

    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if a["creator_id"] != b["creator_id"]:
                continue
            if max_time_gap_ms is not None:
                t_a, t_b = reliable_time(a), reliable_time(b)
                if t_a is None or t_b is None or abs(t_a - t_b) > max_time_gap_ms:
                    continue
            ends_a, ends_b = _endpoints(a), _endpoints(b)
            if min(_gap_km(x, y) for x in ends_a for y in ends_b) <= max_gap_km:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, candidate in enumerate(candidates):
        groups.setdefault(find(i), []).append(candidate)
    out = [sorted(g, key=lambda c: -c["length_km"]) for g in groups.values()]
    out.sort(key=lambda g: -sum(c["length_km"] for c in g))
    return out


def print_urls(candidates: list[dict], region: str, top: int = 15,
               max_gap_km: float = 3.0, urls_per_cluster: int = 3,
               max_time_gap_h: float = 2.0) -> None:
    """Mapillary links, grouped into probable trips, longest first."""
    clusters = cluster_tracks(candidates, max_gap_km, max_time_gap_h)
    print(f"\n# {region}: {len(candidates)} tracks in {len(clusters)} probable trips")
    time_rule = (f" + start times within {max_time_gap_h:g} h"
                 if max_time_gap_h and max_time_gap_h > 0 else "")
    print(f"# same creator + endpoints within {max_gap_km:g} km"
          f"{time_rule} = one group\n")

    for n, group in enumerate(clusters[:top], 1):
        total = sum(c["length_km"] for c in group)
        panos = sum(1 for c in group if c["is_pano"])
        years = sorted({c["captured_year"] for c in group if c["captured_year"]})
        if not years:
            span = "no-date"
        elif len(years) == 1:
            span = f"{years[0]}"
        else:
            span = f"{years[0]}-{years[-1]}"
        kind = f"{panos}/{len(group)} pano" if panos else "perspective"
        print(f"[{n:2d}] {total:6.1f} km  {len(group):2d} seq  {kind:16s} {span}  "
              f"creator {group[0]['creator_id']}")
        for candidate in group[:urls_per_cluster]:
            flag = "360" if candidate["is_pano"] else "   "
            print(f"       {flag} {candidate['length_km']:5.1f} km  {track_url(candidate)}")
        if len(group) > urls_per_cluster:
            print(f"           ... {len(group) - urls_per_cluster} more in this group")
        print()


def emit_registry_stanza(candidate: dict, region: str, scored: dict = None) -> str:
    """A `farfield_trajectories.py` entry for a discovered track.

    `osm` is left as a TODO rather than guessed: the collection guide is
    explicit that stage 4 refuses to build a partial catalog, and `pbf_coverage
    --suggest` is the tool that answers it correctly for the stitched extent,
    which is not known until stage 1 has run.
    """
    name = f"{region}_{candidate['sequence_id'][:6].lower()}"
    note = candidate.get("note", "")
    if scored:
        note = (f"discovered; {scored['n_far_union']} far landmarks, "
                f"max {scored['max_range_km']:.0f} km, "
                f"spread {scored['axial_spread_median']:.2f}")
    return f'''    "{name}": {{
        "seed_pkey": "{candidate['seed_image_id']}",
        "pano": {candidate['is_pano']},
        "osm": None,  # TODO: pbf_coverage --suggest on the stitched extent
        "enc_state": None,
        "note": "{note}",
    }},'''


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--region", help=f"one of: {', '.join(REGIONS)}")
    target.add_argument("--bbox", nargs=4, type=float,
                        metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    target.add_argument("--list_regions", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min_length_km", type=float, default=2.0)
    parser.add_argument("--pano_only", action="store_true")
    parser.add_argument("--include_foot", action="store_true")
    parser.add_argument("--after_year", type=int)
    parser.add_argument("--min_quality", type=float, default=0.0)
    parser.add_argument("--scored", type=Path,
                        help="viewshed output, to rank and annotate by")
    parser.add_argument("--tracks", type=Path,
                        help="the discovery output the scoring ran on; needed "
                             "with --scored --urls, which wants geometry")
    parser.add_argument("--emit_registry", action="store_true")
    parser.add_argument("--urls", action="store_true",
                        help="print Mapillary app links, grouped into probable trips")
    parser.add_argument("--max_gap_km", type=float, default=3.0,
                        help="endpoint gap under which same-creator tracks group")
    parser.add_argument("--max_time_gap_h", type=float, default=2.0,
                        help="start-time gap under which same-creator tracks "
                             "group; hours because tiles carry one timestamp "
                             "per sequence, so fragments of one outing differ "
                             "by the earlier fragment's whole duration. "
                             "<=0 disables the time condition")
    parser.add_argument("--urls_per_cluster", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.list_regions:
        for geometry in geometries():
            print(f"\n{geometry}")
            for name, cfg in REGIONS.items():
                if cfg["geometry"] == geometry:
                    print(f"  {name:24s} {cfg['note']}")
        return

    if args.region:
        if args.region not in REGIONS:
            parser.error(f"unknown region {args.region}; try --list_regions")
        bbox = REGIONS[args.region]["bbox"]
        region_name = args.region
    else:
        bbox = tuple(args.bbox)
        region_name = "bbox"

    if args.scored:
        scored = json.loads(args.scored.read_text())["tracks"]
        print(f"{'score':>6} {'far':>5} {'maxkm':>6} {'spread':>6} {'km':>6} "
              f"{'pano':>5}  sequence", file=sys.stderr)
        for track in scored[:args.top]:
            print(f"{track['score']:6.2f} {track['n_far_union']:5d} "
                  f"{track['max_range_km']:6.1f} {track['axial_spread_median']:6.2f} "
                  f"{track['length_km']:6.1f} {str(track['is_pano']):>5}  "
                  f"{track['sequence_id']}", file=sys.stderr)
        if args.urls:
            # The scored file drops `coords`, so re-read geometry from the
            # tracks file the scoring ran on and join on sequence_id.
            source = json.loads(args.tracks.read_text())["tracks"] if args.tracks \
                else []
            by_id = {t["sequence_id"]: t for t in source}
            enriched = [{**by_id.get(t["sequence_id"], {}), **t}
                        for t in scored[:args.top]
                        if t["sequence_id"] in by_id]
            if not enriched:
                print("--urls needs --tracks as well: the scored file has no "
                      "geometry to place a link on", file=sys.stderr)
            else:
                print_urls(enriched, region_name, top=args.top,
                           max_gap_km=args.max_gap_km,
                           urls_per_cluster=args.urls_per_cluster,
                           max_time_gap_h=args.max_time_gap_h)
        if args.emit_registry:
            print("\n# paste into farfield_trajectories.py TRAJECTORIES")
            for track in scored[:args.top]:
                print(emit_registry_stanza(track, region_name, track))
        return

    print(f"Region {region_name}: {bbox}", file=sys.stderr)
    client = VectorTileClient()
    start = time.time()
    sequences = fetch_region_sequences(client, bbox, args.zoom, args.workers)
    candidates = filter_candidates(
        sequences, bbox, min_length_km=args.min_length_km,
        pano_only=args.pano_only, exclude_foot=not args.include_foot,
        after_year=args.after_year, min_quality=args.min_quality)

    stats = summarise(candidates)
    print(f"\n{json.dumps(stats, indent=2)}", file=sys.stderr)
    print(f"tiles: {client.stats}  ({time.time() - start:.0f}s)", file=sys.stderr)

    print(f"\n{'km':>7} {'pano':>5} {'year':>5} {'qual':>5}  sequence", file=sys.stderr)
    for candidate in candidates[:args.top]:
        print(f"{candidate['length_km']:7.1f} {str(candidate['is_pano']):>5} "
              f"{str(candidate['captured_year']):>5} "
              f"{candidate['quality_score'] or 0:5.2f}  {candidate['sequence_id']}",
              file=sys.stderr)

    if args.urls:
        print_urls(candidates, region_name, top=args.top,
                   max_gap_km=args.max_gap_km,
                   urls_per_cluster=args.urls_per_cluster,
                   max_time_gap_h=args.max_time_gap_h)

    payload = {"region": region_name, "bbox": list(bbox),
               "summary": stats, "tracks": candidates}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
