#!/usr/bin/env python3
"""Resolve a Mapillary image pKey into the whole trajectory it belongs to.

A Mapillary app URL identifies one image (`pKey=...`), and that image's sequence
is usually only a fragment of the trip: long captures get split into multiple
sequences on upload (commonly at 500 or 1000 images). This script walks from a
seed image out to every sibling sequence of the same capture session and emits a
single merged manifest entry that `extract_stitch.py` can download in order.

    seed pKey -> seed sequence -> sibling sequences (same creator, same
    area, same day) -> stitch chain containing the seed -> merged manifest

Discovery is creator-scoped and local to the chain's two endpoints, widening
outward as the chain grows. It deliberately does not sweep the trajectory's
whole area: the /images endpoint rejects a large bbox on area (over 0.010 square
degrees) and separately rejects dense regions on result volume, both as HTTP
500, and subdividing a city-sized box far enough to satisfy the volume limit
fans out exponentially (measured to depth 10 in SF, Seattle and London).

Usage:
    # measure only, no manifest written (cheap; no images downloaded)
    python seed_to_trajectory.py --seed_pkey 298475668560052 --name folkestone_dover --report_only

    # write a manifest ready for extract_stitch.py
    python seed_to_trajectory.py --seed_pkey 298475668560052 --name folkestone_dover \
        -o manifests/farfield/folkestone_dover.json
"""

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from experimental.overhead_matching.swag.mapillary_tools.api import MapillaryClient
from experimental.overhead_matching.swag.mapillary_tools.models import BBox, PanoImage, PanoSequence, haversine
from experimental.overhead_matching.swag.mapillary_tools.tiling import SCAN_TILE_SIZE, adaptive_subdivide, generate_tiles
from experimental.overhead_matching.swag.mapillary_tools.sequences import write_sequence_manifest

# A boat or vehicle cannot plausibly exceed this between two stitched
# sequences; a seam implying more is a mis-stitch, not a trip.
MAX_PLAUSIBLE_SEAM_SPEED_MPS = 15.0

# Below this fraction of distinct GPS positions per image, the track is
# position-quantized: many frames share one fix, so consecutive frames give no
# triangulation baseline however many of them there are.
COARSE_GPS_DISTINCT_FRAC = 0.25

# A "trajectory" shorter than this is a stationary burst, not a trip.
MIN_USEFUL_TRACK_KM = 0.5


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def track_quality(images: list[PanoImage]) -> dict:
    """Per-frame GPS geometry of a capture, used to screen and to size seams.

    Mapillary sequences vary wildly in usable geometry: a 500-image sequence may
    be a 10 km ferry run with a fix per frame, or 17 seconds of video shot from a
    moored boat. Both look identical in an image count, so measure it.
    """
    steps = [haversine(images[i].lat, images[i].lng,
                       images[i + 1].lat, images[i + 1].lng) * 1000
             for i in range(len(images) - 1)]
    ss = sorted(steps)
    moved = [s for s in steps if s > 0.5]
    distinct = len({(round(i.lat, 6), round(i.lng, 6)) for i in images})
    ts = [i.captured_at for i in images if i.captured_at]
    duration_s = (max(ts) - min(ts)) / 1000.0 if len(ts) > 1 else 0.0
    length_km = sum(steps) / 1000.0
    distinct_frac = distinct / max(1, len(images))

    q = {
        "n_images": len(images),
        "duration_s": round(duration_s, 1),
        "s_per_frame": round(duration_s / max(1, len(images) - 1), 3),
        "length_km": round(length_km, 3),
        "distinct_positions": distinct,
        "distinct_frac": round(distinct_frac, 4),
        "step_median_m": round(_percentile(ss, 0.5), 1),
        "step_p95_m": round(_percentile(ss, 0.95), 1),
        "step_p99_m": round(_percentile(ss, 0.99), 1),
        "step_max_m": round(ss[-1], 1) if ss else 0.0,
        "frames_per_gps_fix": round(len(steps) / max(1, len(moved)), 1),
        "mean_speed_mps": round(length_km * 1000 / duration_s, 2) if duration_s else None,
    }
    warnings = []
    if distinct_frac < COARSE_GPS_DISTINCT_FRAC:
        warnings.append(
            f"coarse_gps: only {distinct} distinct positions for {len(images)} "
            f"images ({100*distinct_frac:.0f}%)")
    if length_km < MIN_USEFUL_TRACK_KM:
        warnings.append(f"short_track: {length_km:.2f} km")
    if q["s_per_frame"] and q["s_per_frame"] < 0.2:
        warnings.append(f"video_burst: {q['s_per_frame']}s between frames")
    q["warnings"] = warnings
    return q


def seam_allowance_m(quality: dict, time_gap_s: float, floor_m: float) -> float:
    """How far apart two sequences may be at a seam and still be one trip.

    A single fixed distance cannot express this, because the gap has two
    independent causes and only one of them scales with time:

      * GPS quantization — the device emits a fix every N frames, so endpoints
        land up to one whole step apart even with no elapsed time. On the
        Folkestone ferry that step is ~220 m, so a fixed 100 m threshold rejects
        every seam of an obviously continuous 9 km run.
      * real travel during a recording gap — at the track's own mean speed, a
        37 s gap legitimately covers a few hundred metres.

    So the allowance is (expected travel + one GPS step), with 1.5x headroom on
    each, floored at the caller's value. Judging seams by instantaneous implied
    speed does not work: dividing a quantized 200 m gap by a 0.3 s time gap
    yields a nonsensical 690 m/s.
    """
    mean_speed = quality.get("mean_speed_mps") or 0.0
    expected_travel = mean_speed * max(0.0, time_gap_s)
    return max(floor_m, 1.5 * expected_travel + 1.5 * quality["step_max_m"])


def build_chain(seed_seq: PanoSequence, sequences: list[PanoSequence], quality: dict,
                stitch_time: float, floor_m: float, require_same_resolution: bool = True):
    """Grow a chain outward from the seed in both directions.

    Bidirectional because a seed link often lands mid-trip: extending only
    forward would silently discard everything captured before it.

    At each step, among all sequences that could attach, take the one with the
    smallest time gap — the nearest neighbour in time is the actual next
    segment, whereas first-match-in-list order can attach a later segment and
    strand the ones between.
    """
    chain = [seed_seq]
    used = {seed_seq.id}
    seams = {}   # sequence id -> diagnostics for the seam that attached it
    blocked_by_resolution = []

    def compatible(a: PanoSequence, b: PanoSequence) -> bool:
        if not require_same_resolution:
            return True
        same = (a.min_width == b.min_width and a.min_height == b.min_height)
        if not same:
            blocked_by_resolution.append((a.id, b.id))
        return same

    def try_extend() -> bool:
        head, tail = chain[0], chain[-1]
        best = None
        for cand in sequences:
            if cand.id in used or not cand.images:
                continue
            # forward: cand starts after the tail ends
            dt = (cand.start_time - tail.end_time) / 1000.0
            if 0 <= dt <= stitch_time and compatible(tail, cand):
                dist = haversine(tail.images[-1].lat, tail.images[-1].lng,
                                 cand.images[0].lat, cand.images[0].lng) * 1000
                allowed = seam_allowance_m(quality, dt, floor_m)
                if dist <= allowed and (best is None or dt < best["time_gap_s"]):
                    best = {"seq": cand, "where": "after", "time_gap_s": round(dt, 1),
                            "dist_gap_m": round(dist, 1), "allowed_m": round(allowed, 1)}
            # backward: cand ends before the head starts
            dt_b = (head.start_time - cand.end_time) / 1000.0
            if 0 <= dt_b <= stitch_time and compatible(head, cand):
                dist = haversine(cand.images[-1].lat, cand.images[-1].lng,
                                 head.images[0].lat, head.images[0].lng) * 1000
                allowed = seam_allowance_m(quality, dt_b, floor_m)
                if dist <= allowed and (best is None or dt_b < best["time_gap_s"]):
                    best = {"seq": cand, "where": "before", "time_gap_s": round(dt_b, 1),
                            "dist_gap_m": round(dist, 1), "allowed_m": round(allowed, 1)}
        if best is None:
            return False
        seq = best.pop("seq")
        if best["where"] == "after":
            chain.append(seq)
        else:
            chain.insert(0, seq)
        used.add(seq.id)
        seams[seq.id] = best
        return True

    while try_extend():
        pass
    return chain, seams, blocked_by_resolution


def buffer_bbox(images: list[PanoImage], buffer_km: float) -> BBox:
    """Bbox covering the images, expanded by buffer_km in every direction."""
    lats = [i.lat for i in images]
    lngs = [i.lng for i in images]
    mid_lat = (min(lats) + max(lats)) / 2
    dlat = buffer_km / 111.0
    # cos(lat) guards against an over-narrow box at high latitude
    dlng = buffer_km / max(1e-6, 111.0 * math.cos(math.radians(mid_lat)))
    return BBox(
        west=min(lngs) - dlng,
        south=min(lats) - dlat,
        east=max(lngs) + dlng,
        north=max(lats) + dlat,
    )


# Search radii (degrees) tried in order around a chain endpoint. Small first
# because nearly every seam is within one GPS step; the wide ring is only needed
# for the occasional multi-minute recording gap.
ENDPOINT_SEARCH_RADII_DEG = (0.006, 0.022)


class EndpointScanner:
    """Finds sequences adjacent to a chain endpoint, with a query cache.

    Scanning the whole buffered trajectory area does not work: the /images
    endpoint refuses dense regions on data volume (not just bbox area), and
    subdividing a city-sized box to satisfy it fans out exponentially —
    measured to depth 10 in San Francisco, Seattle and London, which never
    finishes.

    Stitching does not need an area scan anyway. A sequence that continues the
    trip must begin or end near where this chain currently begins or ends, so
    only small neighbourhoods of the two endpoints are worth querying. That
    turns an area sweep into a couple of ~600 m lookups per extension step.
    """

    def __init__(self, client, username, after_ms, before_ms, workers=6, verbose=True):
        self.client = client
        self.username = username
        self.after_ms = after_ms
        self.before_ms = before_ms
        self.workers = workers
        self.verbose = verbose
        self._cache = {}          # bbox string -> images
        self.queries = 0
        self.errors = []

    def _query(self, bbox: BBox):
        key = bbox.to_string()
        if key in self._cache:
            return self._cache[key]

        def q(b):
            self.queries += 1
            return self.client.search_images(
                b, creator_username=self.username,
                after_ms=self.after_ms, before_ms=self.before_ms)

        try:
            images = adaptive_subdivide(bbox, q)
        except Exception as e:
            # A dropped region can only shorten the trajectory, so surface it.
            self.errors.append(f"{key}: {e}")
            images = []
        self._cache[key] = images
        return images

    def around(self, lat: float, lng: float, radius_deg: float) -> dict:
        """Sequence id -> image count near a point, within the time window."""
        dlng = radius_deg / max(1e-6, math.cos(math.radians(lat)))
        bbox = BBox(west=lng - dlng, south=lat - radius_deg,
                    east=lng + dlng, north=lat + radius_deg)
        found = {}
        for img in self._query(bbox):
            # Server-side time filtering is unreliable here, so enforce it.
            if img.captured_at and not (self.after_ms <= img.captured_at <= self.before_ms):
                continue
            if img.sequence_id:
                found[img.sequence_id] = found.get(img.sequence_id, 0) + 1
        return found

    def candidates_near_endpoints(self, chain: list, known: set,
                                  radius_index: int = 0) -> dict:
        """Unseen sequence ids adjacent to either end of the chain.

        Queries exactly one radius so the caller controls escalation. The caller
        must widen on a round that fails to *attach* anything, not merely one
        that finds nothing new: a round can surface an unrelated nearby sequence,
        fail to attach it, and still have a real continuation sitting just past
        the close-in radius. Stopping there silently truncated the Folkestone
        crossing by 2.7 km.
        """
        radius = ENDPOINT_SEARCH_RADII_DEG[radius_index]
        points = [(chain[0].images[0].lat, chain[0].images[0].lng),
                  (chain[-1].images[-1].lat, chain[-1].images[-1].lng)]
        found = {}
        for lat, lng in points:
            for sid, n in self.around(lat, lng, radius).items():
                if sid not in known:
                    found[sid] = found.get(sid, 0) + n
        return found


def seam_report(chain: list[PanoSequence], seams: dict, quality: dict) -> list[dict]:
    """Ordered seam diagnostics along the chain, with a plausibility verdict.

    Plausibility is judged against the allowance actually used, and separately
    reports whether the gap is explained by GPS quantization alone, so a
    quantized track's seams are not mistaken for teleportation.
    """
    out = []
    for seq in chain:
        s = seams.get(seq.id)
        if s is None:
            continue   # the seed itself attached to nothing
        d, t = s["dist_gap_m"], s["time_gap_s"]
        within_quantum = d <= quality["step_max_m"]
        speed = (d / t) if t and t > 0 else None
        out.append({
            "sequence": seq.id,
            "attached": s["where"],
            "time_gap_s": t,
            "dist_gap_m": d,
            "allowed_m": s["allowed_m"],
            "within_gps_quantum": within_quantum,
            "implied_speed_mps": round(speed, 2) if speed is not None else None,
            # Only meaningful once the gap exceeds the GPS quantum; below that
            # the "speed" is an artifact of position quantization.
            "implausible": bool(
                not within_quantum and speed is not None
                and t >= 2.0 and speed > MAX_PLAUSIBLE_SEAM_SPEED_MPS),
        })
    return out


def resolve_trajectory(client: MapillaryClient, seed_pkey: str, name: str,
                       buffer_km: float = 8.0, window_hours: float = 36.0,
                       stitch_time: float = 300.0, stitch_dist: float = 100.0,
                       tile_size: float = SCAN_TILE_SIZE, workers: int = 8,
                       verbose: bool = True) -> dict:
    """Seed image id -> merged trajectory + provenance.

    Returns a dict with the merged PanoSequence under "sequence", plus the
    component sequence ids, seam diagnostics, and the seed's camera model.
    """
    detail = client.get_image_detail(seed_pkey)
    seed_seq_id = detail.get("sequence")
    if not seed_seq_id:
        raise RuntimeError(f"image {seed_pkey} has no sequence")
    creator = (detail.get("creator") or {}).get("username", "")
    if not creator:
        raise RuntimeError(f"image {seed_pkey} has no creator username")
    seed_img = PanoImage.from_api(detail)

    if verbose:
        print(f"[{name}] seed {seed_pkey}: creator={creator} sequence={seed_seq_id}")
        print(f"  camera_type={seed_img.camera_type} "
              f"{seed_img.width}x{seed_img.height} "
              f"equirect={seed_img.is_equirectangular}")

    seed_seq = client.get_full_sequence(seed_seq_id)
    if not seed_seq.images:
        raise RuntimeError(f"sequence {seed_seq_id} returned no images")
    seed_quality = track_quality(seed_seq.images)
    zero_gap_allowance = seam_allowance_m(seed_quality, 0.0, stitch_dist)
    if verbose:
        print(f"  seed sequence: {seed_seq.image_count} images, {seed_seq.length_km:.2f} km, "
              f"{seed_quality['s_per_frame']}s/frame, "
              f"{seed_quality['distinct_positions']} distinct GPS positions")
        print(f"  GPS steps: median {seed_quality['step_median_m']}m "
              f"p99 {seed_quality['step_p99_m']}m max {seed_quality['step_max_m']}m "
              f"(~{seed_quality['frames_per_gps_fix']} frames per fix)")
        for w in seed_quality["warnings"]:
            print(f"  WARNING: {w}")
        if zero_gap_allowance > stitch_dist:
            print(f"  seam allowance at zero time gap: {zero_gap_allowance:.0f}m "
                  f"(floor {stitch_dist:.0f}m) to match this capture's GPS granularity")

    window_ms = int(window_hours * 3600 * 1000)
    after_ms = seed_seq.start_time - window_ms
    before_ms = seed_seq.end_time + window_ms

    # Grow outward from the seed, re-querying only around the current endpoints
    # after each successful attach. Sequences are fetched lazily for the same
    # reason discovery is endpoint-local: a full-area sweep of a dense city is
    # unaffordable, and almost all of what it would return is irrelevant.
    scanner = EndpointScanner(client, creator, after_ms, before_ms,
                              workers=workers, verbose=verbose)
    sequences = [seed_seq]
    known = {seed_seq_id}
    chain = [seed_seq]
    seam_info = {}
    res_blocked = []
    rounds = 0

    radius_index = 0
    while True:
        rounds += 1
        new_ids = scanner.candidates_near_endpoints(chain, known, radius_index)
        if not new_ids:
            if radius_index + 1 < len(ENDPOINT_SEARCH_RADII_DEG):
                radius_index += 1
                continue
            break
        known.update(new_ids)
        fetched = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(client.get_full_sequence, sid): sid for sid in new_ids}
            for fut in as_completed(futures):
                try:
                    seq = fut.result()
                except Exception as e:
                    print(f"    WARNING: could not fetch {futures[fut]}: {e}")
                    continue
                if seq.images:
                    fetched.append(seq)
        sequences.extend(fetched)

        before_len = len(chain)
        chain, seam_info, res_blocked = build_chain(
            seed_seq, sequences, seed_quality, stitch_time, stitch_dist)
        if verbose:
            print(f"  round {rounds}: +{len(new_ids)} candidates at r="
                  f"{ENDPOINT_SEARCH_RADII_DEG[radius_index]}° -> chain "
                  f"{before_len} -> {len(chain)} sequences")
        if len(chain) > before_len:
            # Endpoints moved; go back to the cheap close-in radius.
            radius_index = 0
        elif radius_index + 1 < len(ENDPOINT_SEARCH_RADII_DEG):
            radius_index += 1
        else:
            break

    if scanner.errors:
        print(f"  WARNING: {len(scanner.errors)} region query/queries failed; the "
              f"trajectory may be truncated:")
        for e in scanner.errors[:5]:
            print(f"    {e}")

    if len(chain) == 1 and verbose:
        print("  no stitchable siblings — trajectory is the seed sequence alone")

    merged_images = []
    for seq in chain:
        merged_images.extend(seq.images)
    merged = PanoSequence(id=name, images=merged_images)
    merged.compute_length()

    # A resolution change mid-trip blocks stitching, so report it rather than
    # letting the chain end silently at that boundary.
    res_seen = sorted({(s.min_width, s.min_height) for s in sequences})
    chain_res = sorted({(s.min_width, s.min_height) for s in chain})
    excluded_res = [r for r in res_seen if r not in chain_res]

    seams = seam_report(chain, seam_info, seed_quality)
    chain_quality = track_quality(merged_images)
    result = {
        "name": name,
        "seed_pkey": seed_pkey,
        "seed_sequence_id": seed_seq_id,
        "creator_username": creator,
        "camera_type": seed_img.camera_type,
        "is_equirectangular": seed_img.is_equirectangular,
        "camera_parameters": seed_img.camera_parameters,
        "seed_image_count": seed_seq.image_count,
        "seed_length_km": round(seed_seq.length_km, 3),
        "component_sequence_ids": [s.id for s in chain],
        "candidate_sequences_found": len(sequences),
        "discovery_queries": scanner.queries,
        "discovery_rounds": rounds,
        "chain_image_count": merged.image_count,
        "chain_length_km": round(merged.length_km, 3),
        "gain": round(merged.image_count / max(1, seed_seq.image_count), 3),
        "seed_quality": seed_quality,
        "chain_quality": chain_quality,
        "stitch_dist_floor_m": stitch_dist,
        "resolution_blocked_pairs": res_blocked,
        "seams": seams,
        "resolutions_in_window": [f"{w}x{h}" for w, h in res_seen],
        "resolutions_excluded_from_chain": [f"{w}x{h}" for w, h in excluded_res],
        "search_radii_deg": list(ENDPOINT_SEARCH_RADII_DEG),
        "window_hours": window_hours,
        "stitch_time_s": stitch_time,
        "stitch_dist_m": stitch_dist,
        "sequence": merged,
    }

    if verbose:
        print(f"  chain: {len(chain)} sequence(s), {merged.image_count} images "
              f"({result['gain']}x seed), {merged.length_km:.2f} km, "
              f"{chain_quality['distinct_positions']} distinct GPS positions")
        for w in chain_quality["warnings"]:
            print(f"  WARNING (whole chain): {w}")
        for s in seams:
            flag = "  <-- IMPLAUSIBLE" if s["implausible"] else ""
            q = " (within GPS quantum)" if s["within_gps_quantum"] else ""
            print(f"    seam {s['attached']:6s} {s['sequence'][:22]}: {s['time_gap_s']}s "
                  f"{s['dist_gap_m']}m / allowed {s['allowed_m']}m{q}{flag}")
        if excluded_res:
            print(f"    NOTE: resolutions {excluded_res} present nearby but excluded "
                  f"(stitching requires an exact resolution match)")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed_pkey", required=True, help="Mapillary image id (pKey from the app URL)")
    p.add_argument("--name", required=True, help="Trajectory name (becomes the merged sequence id)")
    p.add_argument("-o", "--output", help="Manifest JSON path")
    p.add_argument("--report_only", action="store_true",
                   help="Print the stitch report without writing a manifest")
    p.add_argument("--buffer_km", type=float, default=8.0,
                   help="Search buffer around the seed sequence (default: 8)")
    p.add_argument("--window_hours", type=float, default=36.0,
                   help="Capture-time window around the seed sequence (default: 36)")
    p.add_argument("--stitch_time", type=float, default=300.0,
                   help="Max seam time gap in seconds (default: 300)")
    p.add_argument("--stitch_dist", type=float, default=100.0,
                   help="Max seam spatial gap in meters (default: 100)")
    p.add_argument("--tile_size", type=float, default=SCAN_TILE_SIZE,
                   help=f"Discovery tile size in degrees (default: {SCAN_TILE_SIZE})")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    client = MapillaryClient()
    res = resolve_trajectory(
        client, args.seed_pkey, args.name,
        buffer_km=args.buffer_km, window_hours=args.window_hours,
        stitch_time=args.stitch_time, stitch_dist=args.stitch_dist,
        tile_size=args.tile_size, workers=args.workers)

    if args.report_only:
        summary = {k: v for k, v in res.items() if k != "sequence"}
        print(json.dumps(summary, indent=2))
        return

    out = args.output or f"manifests/farfield/{args.name}.json"
    extra = {"trajectory": {k: v for k, v in res.items() if k != "sequence"}}
    write_sequence_manifest(out, [res["sequence"]], area_name=args.name,
                            full_sequences=True, extra=extra)
    print(f"\nManifest written to {out}")
    print(f"Next: python extract_stitch.py --manifest {out} "
          f"--sequence {args.name} --output _raw/{args.name}")


if __name__ == "__main__":
    main()
