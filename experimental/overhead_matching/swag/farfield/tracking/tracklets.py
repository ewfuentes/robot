"""The canonical audited-track and camera-bearing contract.

An accepted tracklet is one source track joined to one successful semantic
audit. Acceptance, audit-segment validation, identity, and bearing creation
live here so calibration, matching, and localization cannot quietly choose
different subsets of the data.

Bearings are camera-frame azimuths, clockwise positive. They remain attached
to their real keyframes. `epoch_fused_compat_v1` is the named reducer for
consumers whose contract requires one fused measurement per information epoch.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from experimental.overhead_matching.swag.farfield import geometry as geo


class TrackletContractError(ValueError):
    """A track/audit join cannot be represented without guessing."""


@dataclass(frozen=True)
class TrackletParams:
    """Recorded compatibility-reducer and observation-noise parameters."""

    epoch_keyframes: int
    bearing_sigma_deg: float

    def __post_init__(self):
        if (isinstance(self.epoch_keyframes, bool)
                or not isinstance(self.epoch_keyframes, int)
                or self.epoch_keyframes <= 0):
            raise TrackletContractError("epoch_keyframes must be a positive int")
        if (isinstance(self.bearing_sigma_deg, bool)
                or not isinstance(self.bearing_sigma_deg, (int, float))
                or not math.isfinite(self.bearing_sigma_deg)
                or self.bearing_sigma_deg <= 0.0):
            raise TrackletContractError(
                "bearing_sigma_deg must be finite and positive")


@dataclass(frozen=True)
class ValidSegment:
    """One inclusive audit segment in both relative and keyframe indices."""

    index: int
    start_t: int
    end_t: int
    start_keyframe_idx: int
    end_keyframe_idx: int


@dataclass(frozen=True)
class AcceptedTracklet:
    """One immutable-in-meaning source-track/audit join.

    tracklet_id is globally scoped by the bound source artifact identity and
    digest. local_id is the human-facing T<track_id> identifier retained by
    the v1 compatibility reducer.
    """

    tracklet_id: str
    local_id: str
    source_track: dict
    audit: dict
    valid_segments: tuple[ValidSegment, ...]
    provenance: dict
    quality: dict


@dataclass(frozen=True)
class CameraBearingObservation:
    tracklet_id: str
    keyframe_idx: int
    bearing_camera_cw_deg: float
    angular_width_deg: float
    sigma_deg: float
    correlation_group: str

    def __post_init__(self):
        if not isinstance(self.tracklet_id, str) or not self.tracklet_id:
            raise TrackletContractError(
                "observation tracklet_id must be a non-empty string")
        if (isinstance(self.keyframe_idx, bool)
                or not isinstance(self.keyframe_idx, int)
                or self.keyframe_idx < 0):
            raise TrackletContractError(
                "observation keyframe_idx must be a nonnegative integer")
        if (isinstance(self.bearing_camera_cw_deg, bool)
                or not isinstance(self.bearing_camera_cw_deg, (int, float))
                or not math.isfinite(self.bearing_camera_cw_deg)
                or not 0.0 <= self.bearing_camera_cw_deg < 360.0):
            raise TrackletContractError(
                "bearing_camera_cw_deg must be finite and within [0, 360)")
        if (isinstance(self.angular_width_deg, bool)
                or not isinstance(self.angular_width_deg, (int, float))
                or not math.isfinite(self.angular_width_deg)
                or not 0.0 < self.angular_width_deg <= 360.0):
            raise TrackletContractError(
                "angular_width_deg must be finite and within (0, 360]")
        if (isinstance(self.sigma_deg, bool)
                or not isinstance(self.sigma_deg, (int, float))
                or not math.isfinite(self.sigma_deg)
                or self.sigma_deg <= 0.0):
            raise TrackletContractError(
                "sigma_deg must be finite and positive")
        if (not isinstance(self.correlation_group, str)
                or not self.correlation_group):
            raise TrackletContractError(
                "correlation_group must be a non-empty string")


@dataclass(frozen=True)
class Measurement:
    """Epoch-fused compatibility shape returned at the export boundary."""

    tracklet_id: str
    anchor_keyframe_idx: int
    bearing_camera_cw_deg: float
    kappa: float


def _canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_id(track_id) -> str:
    return f"T{track_id}"


def tracklet_id(track: dict) -> str:
    """The run-local ID retained by existing matching/export artifacts."""
    return _local_id(track["track_id"])


def _integer(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrackletContractError(f"{field} must be an integer")
    return value


def _validate_track(track: dict, expected_id) -> tuple[int, int, list]:
    if not isinstance(track, dict):
        raise TrackletContractError(f"track {expected_id!r} is not an object")
    track_id = track.get("track_id")
    if track_id != expected_id:
        raise TrackletContractError(
            f"track mapping key {expected_id!r} disagrees with track_id "
            f"{track_id!r}")
    birth = _integer(track.get("birth_keyframe"),
                     f"track {track_id} birth_keyframe")
    end = _integer(track.get("end_keyframe"),
                   f"track {track_id} end_keyframe")
    if end < birth:
        raise TrackletContractError(
            f"track {track_id} ends before it is born ({birth}..{end})")
    # `end_keyframe` is the last geometrically supported keyframe and bounds
    # the evidence lifetime exposed to semantic audit.  The tracker may keep
    # propagating the mask through an unsupported tail before closing it;
    # `last_keyframe` bounds those lifecycle records.  Do not conflate the two
    # horizons or valid unsupported records make every real track malformed.
    raw_last = track.get("last_keyframe")
    last = end if raw_last is None else _integer(
        raw_last, f"track {track_id} last_keyframe")
    if last < end:
        raise TrackletContractError(
            f"track {track_id} last_keyframe {last} precedes its supported "
            f"end_keyframe {end}")
    records = track.get("records")
    if not isinstance(records, list) or not records:
        raise TrackletContractError(f"track {track_id} has no records")
    seen = set()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise TrackletContractError(
                f"track {track_id} record {i} is not an object")
        keyframe = _integer(record.get("keyframe"),
                            f"track {track_id} record {i} keyframe")
        if not birth <= keyframe <= last:
            raise TrackletContractError(
                f"track {track_id} record keyframe {keyframe} is outside "
                f"its lifecycle {birth}..{last}")
        if keyframe in seen:
            raise TrackletContractError(
                f"track {track_id} repeats keyframe {keyframe}")
        seen.add(keyframe)
    return birth, end, records


def normalize_valid_segments(track: dict, audit: dict) \
        -> tuple[ValidSegment, ...]:
    """Validate ordered, inclusive audit segments against track lifetime."""
    if not isinstance(audit, dict):
        raise TrackletContractError("audit must be an object")
    if "valid_segments" not in audit:
        raise TrackletContractError("audit has no valid_segments")
    raw_segments = audit["valid_segments"]
    if not isinstance(raw_segments, list):
        raise TrackletContractError("audit valid_segments must be a list")

    birth = _integer(track.get("birth_keyframe"), "birth_keyframe")
    end = _integer(track.get("end_keyframe"), "end_keyframe")
    lifetime = end - birth + 1
    normalized = []
    previous_end = -1
    for i, segment in enumerate(raw_segments):
        if not isinstance(segment, dict):
            raise TrackletContractError(
                f"valid_segments[{i}] must be an object")
        start_t = _integer(segment.get("start_t"),
                           f"valid_segments[{i}].start_t")
        end_t = _integer(segment.get("end_t"),
                         f"valid_segments[{i}].end_t")
        if start_t < 0 or end_t < start_t or end_t >= lifetime:
            raise TrackletContractError(
                f"valid segment {i} [{start_t}, {end_t}] is outside track "
                f"lifetime t0..t{lifetime - 1}")
        if start_t <= previous_end:
            raise TrackletContractError(
                f"valid segment {i} starts at t{start_t} before or within "
                f"the preceding segment ending at t{previous_end}")
        normalized.append(ValidSegment(
            index=i, start_t=start_t, end_t=end_t,
            start_keyframe_idx=birth + start_t,
            end_keyframe_idx=birth + end_t))
        previous_end = end_t
    return tuple(normalized)


def _audit_provenance(audits: Mapping, track_id) -> dict:
    by_track = getattr(audits, "provenance_by_track", {})
    value = by_track.get(track_id, {}) if isinstance(by_track, Mapping) else {}
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if not isinstance(value, Mapping):
        raise TrackletContractError(
            f"audit provenance for track {track_id} is not an object")
    return dict(value)


def _validate_audit_verdict(audit: dict, track_id):
    verdict = audit.get("verdict")
    single_object = audit.get("single_object")
    drop_reason = audit.get("drop_reason")
    if verdict == "keep":
        if single_object is not True or drop_reason != "none":
            raise TrackletContractError(
                f"audit for track {track_id!r}: keep requires "
                f"single_object=true and drop_reason=none")
    elif verdict == "keep_partial":
        if single_object is not False or drop_reason != "none":
            raise TrackletContractError(
                f"audit for track {track_id!r}: keep_partial requires "
                f"single_object=false and drop_reason=none")
    elif verdict == "drop":
        if not isinstance(drop_reason, str) or drop_reason == "none":
            raise TrackletContractError(
                f"audit for track {track_id!r}: drop requires a concrete "
                f"drop_reason")
    else:
        raise TrackletContractError(
            f"audit for track {track_id!r} has invalid verdict {verdict!r}")


def build_accepted_tracklets(tracks: Mapping, audits: Mapping) \
        -> list[AcceptedTracklet]:
    """Join tracks to audits under the one canonical acceptance policy.

    Missing audits are allowed: tracks below the recorded audit support bar
    were never requested. An audit referring to a missing track is stale and
    is an error. drop is always excluded; only valid keep and keep_partial
    records are returned. For either accepted verdict, valid_segments is the
    sole observation whitelist: keep records may trim unreliable spans while
    still asserting that every retained span belongs to one physical object.
    """
    if not isinstance(tracks, Mapping) or not isinstance(audits, Mapping):
        raise TrackletContractError("tracks and audits must be mappings")

    # A deterministic fallback is useful for pure/unit callers. Production
    # AuditResults supplies the stronger bound artifact ID and whole-file hash.
    tracks_fallback_digest = _canonical_sha256([
        tracks[key] for key in sorted(tracks, key=lambda value: str(value))])
    accepted = []
    for track_id in sorted(audits, key=lambda value: str(value)):
        if track_id not in tracks:
            raise TrackletContractError(
                f"audit for track {track_id!r} has no source track")
        track = tracks[track_id]
        _validate_track(track, track_id)
        audit = audits[track_id]
        if not isinstance(audit, dict):
            raise TrackletContractError(
                f"audit for track {track_id!r} is not an object")
        _validate_audit_verdict(audit, track_id)
        verdict = audit["verdict"]
        segments = normalize_valid_segments(track, audit)
        if verdict == "drop":
            continue
        if not segments:
            raise TrackletContractError(
                f"accepted audit for track {track_id!r} has no valid segment")
        local = _local_id(track_id)
        provenance = _audit_provenance(audits, track_id)
        actual_track_digest = _canonical_sha256(track)
        bound_track_digest = provenance.get("source_track_sha256")
        if (bound_track_digest is not None
                and bound_track_digest != actual_track_digest):
            raise TrackletContractError(
                f"track {track_id!r} does not match the source-track digest "
                f"bound by its audit")
        bound_key = provenance.get("audit_key")
        if bound_key is not None and bound_key != local:
            raise TrackletContractError(
                f"track {track_id!r} audit key {bound_key!r} does not match "
                f"its local ID {local!r}")
        source_digest = provenance.get(
            "source_tracks_sha256", tracks_fallback_digest)
        source_identity = provenance.get(
            "source_tracks_artifact_id", f"sha256:{source_digest}")
        global_id = f"{source_identity}@sha256:{source_digest}#{local}"
        provenance.setdefault("source_track_sha256", actual_track_digest)
        accepted.append(AcceptedTracklet(
            tracklet_id=global_id,
            local_id=local,
            source_track=track,
            audit=audit,
            valid_segments=segments,
            provenance=provenance,
            quality={
                "audit_confidence": audit.get("confidence"),
                "n_records": len(track["records"]),
                "n_valid_segments": len(segments),
            }))
    return accepted


def mask_boxes_by_keyframe(track: dict, valid_segments=None) -> dict:
    """Keyframe -> mask bbox in pano coordinates.

    Audit segments use relative time indices. None means an unrestricted
    raw-track query; an empty list means no usable observations.
    """
    birth = track["birth_keyframe"]
    spans = None
    if valid_segments is not None:
        spans = [(birth + segment["start_t"], birth + segment["end_t"])
                 for segment in valid_segments]
    out = {}
    for record in track["records"]:
        mask_box = record.get("mask_bbox_window")
        if mask_box is None:
            continue
        keyframe = record["keyframe"]
        if spans is not None and not any(
                start <= keyframe <= end for start, end in spans):
            continue
        origin_x, origin_y = record["window_origin"]
        out[keyframe] = (
            origin_x + mask_box[0], origin_y + mask_box[1],
            origin_x + mask_box[2], origin_y + mask_box[3])
    return out


def bearing_series(track: dict, pano_w: int, valid_segments=None) -> list:
    """(keyframe, azimuth_cw_deg, angular_width_deg) from mask boxes."""
    if isinstance(pano_w, bool) or not isinstance(pano_w, int) or pano_w <= 0:
        raise TrackletContractError("pano_w must be a positive integer")
    out = []
    for keyframe, box in sorted(
            mask_boxes_by_keyframe(track, valid_segments).items()):
        width_px = box[2] - box[0]
        if not math.isfinite(width_px) or width_px <= 0.0:
            raise TrackletContractError(
                f"track {track.get('track_id')} keyframe {keyframe} has "
                f"invalid mask-box width {width_px!r}")
        midpoint_x = (box[0] + box[2]) / 2.0
        azimuth = geo.azimuth_of_pano_column(midpoint_x, pano_w)
        angular_width = width_px / pano_w * 360.0
        out.append((keyframe, azimuth, angular_width))
    return out


def build_camera_bearing_observations(
        accepted_tracklets: list[AcceptedTracklet], pano_w: int,
        bearing_sigma_deg: float) -> list[CameraBearingObservation]:
    """Preserve every audit-valid bearing at its actual keyframe."""
    if (isinstance(bearing_sigma_deg, bool)
            or not isinstance(bearing_sigma_deg, (int, float))
            or not math.isfinite(bearing_sigma_deg)
            or bearing_sigma_deg <= 0.0):
        raise TrackletContractError(
            "bearing_sigma_deg must be finite and positive")
    observations = []
    for tracklet in accepted_tracklets:
        for segment in tracklet.valid_segments:
            raw_segment = [{"start_t": segment.start_t,
                            "end_t": segment.end_t}]
            correlation_group = (
                f"{tracklet.tracklet_id}/audit-segment-{segment.index}")
            for keyframe, azimuth, width in bearing_series(
                    tracklet.source_track, pano_w, raw_segment):
                observations.append(CameraBearingObservation(
                    tracklet_id=tracklet.tracklet_id,
                    keyframe_idx=keyframe,
                    bearing_camera_cw_deg=azimuth,
                    angular_width_deg=width,
                    sigma_deg=bearing_sigma_deg,
                    correlation_group=correlation_group))
    observations.sort(key=lambda obs: (obs.keyframe_idx, obs.tracklet_id,
                                       obs.correlation_group))
    return observations


def _fuse_group(observations: list[CameraBearingObservation],
                epoch_keyframes: int) -> list[Measurement]:
    fused = []
    bucket = []
    start_keyframe = observations[0].keyframe_idx

    def flush():
        if not bucket:
            return
        mean_azimuth = geo.circular_mean_deg(
            [obs.bearing_camera_cw_deg for obs in bucket])
        mean_width = sum(obs.angular_width_deg for obs in bucket) / len(bucket)
        # sigma_deg is constant today, but averaging makes the reducer's
        # behavior explicit if observation-level noise is introduced later.
        mean_sigma = sum(obs.sigma_deg for obs in bucket) / len(bucket)
        anchor = bucket[len(bucket) // 2].keyframe_idx
        sigma = math.hypot(mean_sigma, mean_width / 4.0)
        fused.append(Measurement(
            tracklet_id=bucket[0].tracklet_id,
            anchor_keyframe_idx=anchor,
            bearing_camera_cw_deg=mean_azimuth,
            kappa=1.0 / math.radians(sigma) ** 2))

    for observation in observations:
        if observation.keyframe_idx - start_keyframe >= epoch_keyframes:
            flush()
            bucket = []
            start_keyframe = observation.keyframe_idx
        bucket.append(observation)
    flush()
    return fused


def epoch_fused_compat_v1(
        observations: list[CameraBearingObservation],
        params: TrackletParams) -> list[Measurement]:
    """Reproduce the pre-observation-contract epoch fusion.

    Epoch buckets never cross an audit segment/correlation-group boundary.
    The bearing is the circular mean, the middle real keyframe is the anchor,
    angular width is averaged, and observation count does not increase kappa.
    """
    grouped = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, CameraBearingObservation):
            raise TrackletContractError(
                "epoch_fused_compat_v1 expects CameraBearingObservation")
        grouped[(observation.tracklet_id,
                 observation.correlation_group)].append(observation)
    fused = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda obs: obs.keyframe_idx)
        if len({obs.keyframe_idx for obs in group}) != len(group):
            raise TrackletContractError(
                f"duplicate keyframe in correlation group {key!r}")
        fused.extend(_fuse_group(group, params.epoch_keyframes))
    fused.sort(key=lambda measurement: (
        measurement.anchor_keyframe_idx, measurement.tracklet_id))
    return fused


def fuse_bearings(series: list, params: TrackletParams) -> list:
    """Tuple-form representation of the compatibility reducer output."""
    observations = [CameraBearingObservation(
        tracklet_id="compat", keyframe_idx=keyframe,
        bearing_camera_cw_deg=azimuth, angular_width_deg=width,
        sigma_deg=params.bearing_sigma_deg, correlation_group="compat")
        for keyframe, azimuth, width in series]
    return [(measurement.anchor_keyframe_idx,
             measurement.bearing_camera_cw_deg, measurement.kappa)
            for measurement in epoch_fused_compat_v1(observations, params)]


def build_measurements(tracks: Mapping, audits: Mapping, pano_w: int,
                       params: TrackletParams) -> list[Measurement]:
    """Named fusion boundary preserving artifact-scoped tracklet IDs."""
    accepted = build_accepted_tracklets(tracks, audits)
    observations = build_camera_bearing_observations(
        accepted, pano_w, params.bearing_sigma_deg)
    return epoch_fused_compat_v1(observations, params)
