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
500. Subdividing a large dense region far enough to satisfy the volume limit
fans out exponentially.

`build_chain` is the canonical policy for deciding whether two Mapillary
sequences belong to one trip.

Usage:
    # measure only, no manifest written (cheap; no images downloaded)
    bazel run //experimental/overhead_matching/swag/farfield/collection:seed_to_trajectory -- \\
        --seed_pkey <image-id> --name <trajectory-name> \\
        --stitch_time 300 --stitch_dist 100 --window_hours 36 --report_only

    # write a manifest ready for extract_stitch.py
    bazel run //experimental/overhead_matching/swag/farfield/collection:seed_to_trajectory -- \\
        --seed_pkey <image-id> --name <trajectory-name> \\
        --stitch_time 300 --stitch_dist 100 --window_hours 36 \\
        --output <root>/raw_material/mapillary_manifests/<trajectory-name>.json
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from experimental.overhead_matching.swag.farfield.collection.api import MapillaryClient
from experimental.overhead_matching.swag.farfield.collection.models import (
    BBox, PanoImage, PanoSequence,
)
from experimental.overhead_matching.swag.farfield.collection.provenance_util import (
    provenance_record,
)
from experimental.overhead_matching.swag.farfield.collection.tiling import (
    adaptive_subdivide,
)
from experimental.overhead_matching.swag.farfield.geometry import haversine_m

# A boat or vehicle cannot plausibly exceed this between two stitched
# sequences; a seam implying more is a mis-stitch, not a trip.
MAX_PLAUSIBLE_SEAM_SPEED_MPS = 15.0

# Below this fraction of distinct GPS positions per image, the track is
# position-quantized: many frames share one fix, so consecutive frames give no
# triangulation baseline however many of them there are.
COARSE_GPS_DISTINCT_FRAC = 0.25

# A "trajectory" shorter than this is a stationary burst, not a trip.
MIN_USEFUL_TRACK_KM = 0.5

MANIFEST_INCOMPLETE_SUFFIX = ".incomplete"
_MANIFEST_KEYS = frozenset({"metadata", "sequences", "trajectory", "provenance"})
_METADATA_KEYS = frozenset({
    "area_name",
    "total_sequences",
    "total_images",
    "total_length_km",
    "created_at",
})
_SEQUENCE_KEYS = frozenset({
    "id",
    "length_km",
    "image_count",
    "start_time",
    "end_time",
    "camera_types",
    "min_width",
    "min_height",
    "images",
})
_IMAGE_REQUIRED_KEYS = frozenset({
    "id",
    "lat",
    "lng",
    "compass_angle",
    "computed_compass_angle",
    "captured_at",
    "camera_type",
    "height",
    "width",
    "sequence_id",
    "downloaded",
})
_IMAGE_OPTIONAL_KEYS = frozenset({
    "camera_parameters",
    "is_pano",
    "creator_username",
    "geometry_source",
})
_TRAJECTORY_IDENTITY_KEYS = frozenset({
    "name",
    "seed_pkey",
    "seed_sequence_id",
    "creator_username",
    "camera_type",
    "is_equirectangular",
    "camera_parameters",
    "seed_image_count",
    "seed_length_km",
    "component_sequence_ids",
    "chain_image_count",
    "chain_length_km",
})
_PROVENANCE_KEYS = frozenset({
    "schema",
    "generator",
    "git_commit",
    "argv",
    "created",
    "inputs",
    "config",
    "notes",
})
_GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/"
    "collection:seed_to_trajectory"
)


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def track_quality(images: list[PanoImage]) -> dict:
    """Per-frame GPS geometry of a capture, used to screen and to size seams.

    The same image count can describe either a moving capture with a GPS fix per
    frame or a stationary video burst. Per-frame geometry distinguishes them.
    """
    steps = [haversine_m(images[i].lat, images[i].lng,
                         images[i + 1].lat, images[i + 1].lng)
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
        land up to one whole GPS step apart even with no elapsed time. Coarse
        devices can make that step hundreds of metres.
      * real travel during a recording gap — at the track's own mean speed, a
        recording gap can legitimately cover additional distance.

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
                dist = haversine_m(tail.images[-1].lat, tail.images[-1].lng,
                                   cand.images[0].lat, cand.images[0].lng)
                allowed = seam_allowance_m(quality, dt, floor_m)
                if dist <= allowed and (best is None or dt < best["time_gap_s"]):
                    best = {"seq": cand, "where": "after", "time_gap_s": round(dt, 1),
                            "dist_gap_m": round(dist, 1), "allowed_m": round(allowed, 1)}
            # backward: cand ends before the head starts
            dt_b = (head.start_time - cand.end_time) / 1000.0
            if 0 <= dt_b <= stitch_time and compatible(head, cand):
                dist = haversine_m(cand.images[-1].lat, cand.images[-1].lng,
                                   head.images[0].lat, head.images[0].lng)
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


# Search radii (degrees) tried in order around a chain endpoint. Small first
# because nearly every seam is within one GPS step; the wide ring is only needed
# for the occasional multi-minute recording gap.
ENDPOINT_SEARCH_RADII_DEG = (0.006, 0.022)


class EndpointScanner:
    """Finds sequences adjacent to a chain endpoint, with a query cache.

    Scanning the whole buffered trajectory area does not work: the /images
    endpoint refuses dense regions on data volume (not just bbox area), and
    subdividing a large dense box to satisfy it fans out exponentially.

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
        the close-in radius. Stopping after that round can silently truncate the
        trajectory.
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: dict, expected: frozenset[str], where: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"{where}: missing={missing}, unknown={unknown}")


def _require_object(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _require_string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"{where} must be a {qualifier}string")
    return value


def _require_integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, where: str, *, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{where} must be <= {maximum}")
    return result


def _validate_json_value(value: Any, where: str) -> None:
    """Reject values that JSON accepts permissively but cannot identify."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string key")
            _validate_json_value(item, f"{where}.{key}")
        return
    raise ValueError(
        f"{where} contains non-JSON value {type(value).__name__}")


def _load_manifest_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"manifest is not a regular file: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid trajectory manifest {path}: {error}") from error
    return _require_object(document, f"trajectory manifest {path}")


def _image_length_km(images: list[dict]) -> float:
    return sum(
        haversine_m(
            images[index]["lat"], images[index]["lng"],
            images[index + 1]["lat"], images[index + 1]["lng"],
        )
        for index in range(len(images) - 1)
    ) / 1000.0


def _validate_image(image: Any, sequence_id: str, index: int,
                    seen_image_ids: set[str]) -> dict:
    where = f"sequence {sequence_id!r} image {index}"
    image = _require_object(image, where)
    actual = frozenset(image)
    missing = sorted(_IMAGE_REQUIRED_KEYS - actual)
    unknown = sorted(actual - _IMAGE_REQUIRED_KEYS - _IMAGE_OPTIONAL_KEYS)
    if missing or unknown:
        raise ValueError(f"{where}: missing={missing}, unknown={unknown}")
    image_id = _require_string(image["id"], f"{where} id")
    if image_id in seen_image_ids:
        raise ValueError(f"duplicate image id {image_id!r}")
    seen_image_ids.add(image_id)
    _require_finite(image["lat"], f"image {image_id} lat",
                    minimum=-90.0, maximum=90.0)
    _require_finite(image["lng"], f"image {image_id} lng",
                    minimum=-180.0, maximum=180.0)
    _require_finite(image["compass_angle"], f"image {image_id} compass_angle")
    _require_finite(
        image["computed_compass_angle"],
        f"image {image_id} computed_compass_angle")
    _require_integer(image["captured_at"], f"image {image_id} captured_at")
    _require_string(image["camera_type"], f"image {image_id} camera_type",
                    nonempty=False)
    _require_integer(image["height"], f"image {image_id} height", minimum=1)
    _require_integer(image["width"], f"image {image_id} width", minimum=1)
    _require_string(image["sequence_id"], f"image {image_id} sequence_id")
    if type(image["downloaded"]) is not bool:
        raise ValueError(f"image {image_id} downloaded must be a boolean")
    if "camera_parameters" in image:
        parameters = image["camera_parameters"]
        if not isinstance(parameters, list):
            raise ValueError(f"image {image_id} camera_parameters must be a list")
        for parameter_index, parameter in enumerate(parameters):
            _require_finite(
                parameter,
                f"image {image_id} camera_parameters[{parameter_index}]")
    if "is_pano" in image and type(image["is_pano"]) is not bool:
        raise ValueError(f"image {image_id} is_pano must be a boolean")
    for field in ("creator_username", "geometry_source"):
        if field in image:
            _require_string(image[field], f"image {image_id} {field}",
                            nonempty=False)
    return image


def _validate_sequence(sequence: Any, seen_sequence_ids: set[str],
                       seen_image_ids: set[str]) -> tuple[dict, float]:
    sequence = _require_object(sequence, "sequence")
    _require_exact_keys(sequence, _SEQUENCE_KEYS, "sequence")
    sequence_id = _require_string(sequence["id"], "sequence id")
    if sequence_id in seen_sequence_ids:
        raise ValueError(f"duplicate sequence id {sequence_id!r}")
    seen_sequence_ids.add(sequence_id)
    images_value = sequence["images"]
    if not isinstance(images_value, list) or not images_value:
        raise ValueError(f"sequence {sequence_id!r} images must be a non-empty list")
    images = [
        _validate_image(image, sequence_id, index, seen_image_ids)
        for index, image in enumerate(images_value)
    ]
    image_count = len(images)
    if _require_integer(sequence["image_count"],
                        f"sequence {sequence_id} image_count") != image_count:
        raise ValueError(f"sequence {sequence_id!r} image_count is not exact")
    start_time = min(image["captured_at"] for image in images)
    end_time = max(image["captured_at"] for image in images)
    recorded_start = _require_integer(
        sequence["start_time"], f"sequence {sequence_id} start_time")
    recorded_end = _require_integer(
        sequence["end_time"], f"sequence {sequence_id} end_time")
    if recorded_start != start_time or recorded_end != end_time:
        raise ValueError(f"sequence {sequence_id!r} timestamp bounds are not exact")
    camera_types = sorted({image["camera_type"] for image in images
                           if image["camera_type"]})
    if (not isinstance(sequence["camera_types"], list)
            or not all(isinstance(item, str)
                       for item in sequence["camera_types"])
            or sequence["camera_types"] != camera_types):
        raise ValueError(f"sequence {sequence_id!r} camera_types are not exact")
    recorded_min_width = _require_integer(
        sequence["min_width"], f"sequence {sequence_id} min_width", minimum=1)
    if recorded_min_width != min(image["width"] for image in images):
        raise ValueError(f"sequence {sequence_id!r} min_width is not exact")
    recorded_min_height = _require_integer(
        sequence["min_height"], f"sequence {sequence_id} min_height", minimum=1)
    if recorded_min_height != min(image["height"] for image in images):
        raise ValueError(f"sequence {sequence_id!r} min_height is not exact")
    length_km = _image_length_km(images)
    recorded_length = _require_finite(
        sequence["length_km"], f"sequence {sequence_id} length_km", minimum=0.0)
    if recorded_length != round(length_km, 3):
        raise ValueError(f"sequence {sequence_id!r} length_km is not exact")
    return sequence, length_km


def _validate_provenance(provenance: Any, sequence_id: str,
                         seed_pkey: str, trajectory: dict) -> None:
    provenance = _require_object(provenance, "provenance")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "provenance")
    if provenance["schema"] != "farfield_provenance/v1":
        raise ValueError("provenance has an unsupported schema")
    if provenance["generator"] != _GENERATOR:
        raise ValueError("provenance generator does not identify seed_to_trajectory")
    for field in ("git_commit", "created"):
        _require_string(provenance[field], f"provenance {field}")
    if not isinstance(provenance["argv"], list) or not all(
            isinstance(item, str) for item in provenance["argv"]):
        raise ValueError("provenance argv must be a list of strings")
    if provenance["inputs"] != {"seed_pkey": seed_pkey}:
        raise ValueError("provenance seed_pkey does not match trajectory identity")
    config = _require_object(provenance["config"], "provenance config")
    _require_exact_keys(
        config,
        frozenset({"name", "window_hours", "stitch_time_s", "stitch_dist_m"}),
        "provenance config")
    if config["name"] != sequence_id:
        raise ValueError("provenance name does not match sequence identity")
    for field, trajectory_field in (
            ("window_hours", "window_hours"),
            ("stitch_time_s", "stitch_time_s"),
            ("stitch_dist_m", "stitch_dist_m")):
        _require_finite(config[field], f"provenance config {field}", minimum=0.0)
        if config[field] != trajectory.get(trajectory_field):
            raise ValueError(
                f"provenance {field} does not match trajectory diagnostics")
    _require_string(provenance["notes"], "provenance notes", nonempty=False)


def _validate_manifest_document(
        document: Any, *, expected_sequence_id: str | None = None,
        expected_seed_pkey: str | None = None) -> dict:
    document = _require_object(document, "trajectory manifest")
    _validate_json_value(document, "trajectory manifest")
    _require_exact_keys(document, _MANIFEST_KEYS, "trajectory manifest")

    sequences_value = document["sequences"]
    if not isinstance(sequences_value, list) or len(sequences_value) != 1:
        raise ValueError("trajectory manifest must contain exactly one sequence")
    seen_sequence_ids: set[str] = set()
    seen_image_ids: set[str] = set()
    sequence, length_km = _validate_sequence(
        sequences_value[0], seen_sequence_ids, seen_image_ids)
    sequence_id = sequence["id"]
    if expected_sequence_id is not None and sequence_id != expected_sequence_id:
        raise ValueError(
            f"sequence identity mismatch: expected {expected_sequence_id!r}, "
            f"found {sequence_id!r}")

    metadata = _require_object(document["metadata"], "metadata")
    _require_exact_keys(metadata, _METADATA_KEYS, "metadata")
    if _require_string(metadata["area_name"], "metadata area_name") != sequence_id:
        raise ValueError("metadata area_name does not match sequence identity")
    if _require_integer(metadata["total_sequences"],
                        "metadata total_sequences", minimum=1) != 1:
        raise ValueError("metadata total_sequences is not exact")
    if _require_integer(metadata["total_images"],
                        "metadata total_images", minimum=1) != len(
                            sequence["images"]):
        raise ValueError("metadata total_images is not exact")
    if _require_finite(metadata["total_length_km"],
                       "metadata total_length_km", minimum=0.0) != round(length_km, 2):
        raise ValueError("metadata total_length_km is not exact")
    _require_integer(metadata["created_at"], "metadata created_at")

    trajectory = _require_object(document["trajectory"], "trajectory")
    missing_trajectory = sorted(_TRAJECTORY_IDENTITY_KEYS - frozenset(trajectory))
    if missing_trajectory:
        raise ValueError(f"trajectory missing identity fields {missing_trajectory}")
    if trajectory["name"] != sequence_id:
        raise ValueError("trajectory name does not match sequence identity")
    seed_pkey = _require_string(trajectory["seed_pkey"], "trajectory seed_pkey")
    if expected_seed_pkey is not None and seed_pkey != expected_seed_pkey:
        raise ValueError(
            f"seed image identity mismatch: expected {expected_seed_pkey!r}, "
            f"found {seed_pkey!r}")
    seed_sequence_id = _require_string(
        trajectory["seed_sequence_id"], "trajectory seed_sequence_id")
    components = trajectory["component_sequence_ids"]
    if not isinstance(components, list) or not components or not all(
            isinstance(item, str) and item for item in components):
        raise ValueError("trajectory component_sequence_ids must be non-empty strings")
    if len(set(components)) != len(components):
        raise ValueError("trajectory component_sequence_ids must be unique")
    images = sequence["images"]
    observed_components = []
    for image in images:
        source_sequence_id = image["sequence_id"]
        if not observed_components or observed_components[-1] != source_sequence_id:
            observed_components.append(source_sequence_id)
    if observed_components != components:
        raise ValueError(
            "trajectory component_sequence_ids do not exactly match image order")
    if seed_sequence_id not in components:
        raise ValueError("trajectory seed_sequence_id is not a component")
    seed_images = [image for image in images
                   if image["sequence_id"] == seed_sequence_id]
    seed_matches = [image for image in images if image["id"] == seed_pkey]
    if len(seed_matches) != 1:
        raise ValueError("trajectory seed_pkey does not identify exactly one image")
    seed_image = seed_matches[0]
    if seed_image["sequence_id"] != seed_sequence_id:
        raise ValueError("trajectory seed image belongs to the wrong sequence")
    if _require_integer(trajectory["chain_image_count"],
                        "trajectory chain_image_count", minimum=1) != len(images):
        raise ValueError("trajectory chain_image_count is not exact")
    if _require_finite(trajectory["chain_length_km"],
                       "trajectory chain_length_km", minimum=0.0) != round(length_km, 3):
        raise ValueError("trajectory chain_length_km is not exact")
    if _require_integer(trajectory["seed_image_count"],
                        "trajectory seed_image_count", minimum=1) != len(seed_images):
        raise ValueError("trajectory seed_image_count is not exact")
    seed_length_km = _image_length_km(seed_images)
    if _require_finite(trajectory["seed_length_km"],
                       "trajectory seed_length_km", minimum=0.0) != round(
                           seed_length_km, 3):
        raise ValueError("trajectory seed_length_km is not exact")
    if trajectory["camera_type"] != seed_image["camera_type"]:
        raise ValueError("trajectory camera_type does not match the seed image")
    if trajectory["camera_parameters"] != seed_image.get("camera_parameters"):
        raise ValueError("trajectory camera_parameters do not match the seed image")
    expected_equirectangular = PanoImage.from_dict(seed_image).is_equirectangular
    if trajectory["is_equirectangular"] is not expected_equirectangular:
        raise ValueError("trajectory panorama type does not match the seed image")
    if "creator_username" in seed_image and (
            trajectory["creator_username"] != seed_image["creator_username"]):
        raise ValueError("trajectory creator does not match the seed image")

    _validate_provenance(
        document["provenance"], sequence_id, seed_pkey, trajectory)
    return document


def validate_sequence_manifest(
        path: Path | str, *, expected_sequence_id: str | None = None,
        expected_seed_pkey: str | None = None) -> dict:
    """Strictly validate one completed seed-to-trajectory manifest.

    The collection orchestrator calls this before treating an existing stage-1
    output as complete.  Incomplete siblings are deliberately not consumable.
    """
    path = Path(path)
    if any(part.endswith(MANIFEST_INCOMPLETE_SUFFIX) for part in path.parts):
        raise ValueError(f"incomplete trajectory manifest cannot be consumed: {path}")
    return _validate_manifest_document(
        _load_manifest_json(path),
        expected_sequence_id=expected_sequence_id,
        expected_seed_pkey=expected_seed_pkey,
    )


def _directory_fd(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags)


def _fsync_directory(path: Path) -> None:
    descriptor = _directory_fd(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_incomplete(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _publish_file_no_clobber(staging: Path, destination: Path) -> None:
    """Publish one sibling file atomically without a replacement window."""
    if staging.parent.resolve() != destination.parent.resolve():
        raise ValueError("manifest staging and destination must be siblings")
    if staging.is_symlink() or not staging.is_file():
        raise ValueError(f"manifest staging file is not regular: {staging}")
    parent_fd = _directory_fd(destination.parent)
    try:
        # Linking is the portable atomic no-clobber publication primitive for
        # two names on the same filesystem.  A concurrent winner makes this
        # fail instead of replacing its manifest.
        os.link(staging, destination, follow_symlinks=False)
        os.fsync(parent_fd)
        staging.unlink()
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def write_sequence_manifest(path, sequences: list[PanoSequence],
                            area_name: str = "", extra: dict = None) -> None:
    """Validate, stage, and immutably publish an extract_stitch manifest.

    The exact sibling ``<manifest>.incomplete`` is retained on a write,
    validation, or publication failure.  Neither that diagnostic residue nor
    an existing completed manifest is ever overwritten.
    """
    total_images = sum(s.image_count for s in sequences)
    total_length = sum(s.length_km for s in sequences)
    manifest = {
        "metadata": {
            "area_name": area_name,
            "total_sequences": len(sequences),
            "total_images": total_images,
            "total_length_km": round(total_length, 2),
            "created_at": int(time.time() * 1000),
        },
        "sequences": [s.to_dict() for s in sequences],
    }
    if extra is not None:
        if not isinstance(extra, dict):
            raise ValueError("manifest extra fields must be an object")
        collisions = sorted(set(manifest) & set(extra))
        if collisions:
            raise ValueError(f"manifest extra fields shadow {collisions}")
        manifest.update(extra)

    # Validate the in-memory contract before creating any filesystem residue;
    # the disk copy is validated again before it becomes consumable.
    _validate_manifest_document(manifest)
    payload = (json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")
    destination = Path(path)
    staging = destination.with_name(
        destination.name + MANIFEST_INCOMPLETE_SUFFIX)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"completed manifest already exists: {destination}")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"incomplete manifest already exists: {staging}")
    _write_incomplete(staging, payload)
    _validate_manifest_document(_load_manifest_json(staging))
    _publish_file_no_clobber(staging, destination)


def resolve_trajectory(client: MapillaryClient, seed_pkey: str, name: str,
                       window_hours: float, stitch_time: float,
                       stitch_dist: float, workers: int = 8,
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed_pkey", required=True,
                   help="Mapillary image id (pKey from the app URL)")
    p.add_argument("--name", required=True,
                   help="Trajectory name (becomes the merged sequence id)")
    p.add_argument("-o", "--output", type=Path,
                   help="Manifest JSON path; required unless --report_only")
    p.add_argument("--report_only", action="store_true",
                   help="Print the stitch report without writing a manifest")
    p.add_argument("--window_hours", type=float, required=True,
                   help="Capture-time window around the seed sequence within "
                        "which siblings are considered")
    p.add_argument("--stitch_time", type=float, required=True,
                   help="Max seam time gap in seconds")
    p.add_argument("--stitch_dist", type=float, required=True,
                   help="Max seam spatial gap floor in meters; the allowance "
                        "grows with GPS quantization and speed, see "
                        "seam_allowance_m")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    if not args.report_only and not args.output:
        p.error("--output is required unless --report_only")

    client = MapillaryClient()
    res = resolve_trajectory(
        client, args.seed_pkey, args.name,
        window_hours=args.window_hours,
        stitch_time=args.stitch_time, stitch_dist=args.stitch_dist,
        workers=args.workers)

    if args.report_only:
        summary = {k: v for k, v in res.items() if k != "sequence"}
        print(json.dumps(summary, indent=2))
        return 0

    extra = {
        "trajectory": {k: v for k, v in res.items() if k != "sequence"},
        "provenance": provenance_record(
            generator=_GENERATOR,
            inputs={"seed_pkey": args.seed_pkey},
            config={"name": args.name, "window_hours": args.window_hours,
                    "stitch_time_s": args.stitch_time,
                    "stitch_dist_m": args.stitch_dist}),
    }
    write_sequence_manifest(args.output, [res["sequence"]],
                            area_name=args.name, extra=extra)
    print(f"\nManifest written to {args.output}")
    print(f"Next: bazel run //experimental/overhead_matching/swag/farfield/"
          f"collection:extract_stitch -- --manifest {args.output} "
          f"--sequence {args.name} --output <raw_dir>/{args.name} "
          f"--max_width ... --min_spacing_m ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
