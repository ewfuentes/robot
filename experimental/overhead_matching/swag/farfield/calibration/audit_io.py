"""Strict reader for complete, source-bound semantic-audit artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    semantic_audit,
)


META_SCHEMA = "farfield_semantic_audit_meta/v2"
# Manifest `config.audit_source` of an audit artifact that restates single VLM
# detections verbatim instead of reviewing tracks (the no-tracking ablation).
# Matching keys its Set 1 prompt and formatting off this value.
DETECTION_PASSTHROUGH_SOURCE = "detection_passthrough_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_META_KEYS = frozenset({"schema", "source_tracks", "requests"})
_SOURCE_KEYS = frozenset({"artifact_id", "file", "sha256"})
_REQUEST_REQUIRED_KEYS = frozenset({"track_id", "source_track_sha256"})
_REQUEST_OPTIONAL_KEYS = frozenset({
    "range",
    "birth_keyframe",
    "n_supports",
    "support_obs_by_t",
    "chips",
})


class AuditArtifactError(ValueError):
    """The semantic-audit artifact is missing, stale, or incomplete."""


class AuditResults(dict):
    """track_id -> audit, with source identities and join provenance."""

    def __init__(self, values, *, provenance_by_track, source_tracks,
                 tracks_ref, semantic_audits_ref):
        super().__init__(values)
        self.provenance_by_track = provenance_by_track
        self.source_tracks = source_tracks
        self.tracks_ref = tracks_ref
        self.semantic_audits_ref = semantic_audits_ref


def canonical_sha256(value) -> str:
    """SHA-256 of canonical finite JSON for individual track binding."""
    try:
        return artifact.sha256_json(value)
    except artifact.ArtifactError as error:
        raise AuditArtifactError(f"cannot hash source track: {error}") \
            from error


def file_sha256(path: Path) -> str:
    """Compatibility spelling for the shared strict file hasher."""
    try:
        return artifact.sha256_file(path)
    except artifact.ArtifactError as error:
        raise AuditArtifactError(str(error)) from error


def source_artifact_id(ref: artifact.ArtifactRef) -> str:
    return (f"{paths_lib.OBJECT_TRACKS}:{ref.dataset}:{ref.version}"
            f"@sha256:{ref.content_digest}")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditArtifactError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise AuditArtifactError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant)
    except AuditArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditArtifactError(f"cannot read valid JSON from {path}: {error}") \
            from error


def _required_object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise AuditArtifactError(f"{label} must be an object")
    return value


def _require_exact_keys(value: dict, expected, label: str) -> None:
    actual = frozenset(value)
    if actual != frozenset(expected):
        missing = sorted(frozenset(expected) - actual)
        unknown = sorted(actual - frozenset(expected))
        raise AuditArtifactError(
            f"{label} has invalid fields; missing={missing}, unknown={unknown}")


def _required_string(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuditArtifactError(f"{label} must be a non-empty string")
    return value


def _required_sha256(value, label: str) -> str:
    value = _required_string(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise AuditArtifactError(
            f"{label} must be a lowercase 64-character SHA-256 digest")
    return value


def _required_track_id(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditArtifactError(f"{label} must be a nonnegative integer")
    return value


def _open_artifact(path: Path, *, kind: str,
                   dataset_name: str | None = None) -> artifact.ArtifactRef:
    try:
        return artifact.open_artifact(
            path, expected_kind=kind, expected_dataset=dataset_name)
    except artifact.ArtifactError as error:
        raise AuditArtifactError(f"invalid completed {kind} artifact: {error}") \
            from error


def _load_one_tracks_artifact(tracks_dir: Path):
    paths = sorted(Path(tracks_dir).glob("tracks_*.json"))
    if len(paths) != 1:
        raise AuditArtifactError(
            f"{tracks_dir} must contain exactly one tracks_*.json file; "
            f"found {len(paths)}")
    path = paths[0]
    document = _required_object(_load_json(path), str(path))
    tracks_value = document.get("tracks")
    if not isinstance(tracks_value, list):
        raise AuditArtifactError(f"{path}: tracks must be a list")
    range_value = _required_object(document.get("range"), f"{path}: range")
    range_name = _required_string(
        range_value.get("name"), f"{path}: range.name")
    tracks = {}
    for index, track in enumerate(tracks_value):
        track = _required_object(track, f"{path}: tracks[{index}]")
        track_id = _required_track_id(
            track.get("track_id"), f"{path}: tracks[{index}].track_id")
        if track_id in tracks:
            raise AuditArtifactError(
                f"{path}: duplicate source track_id {track_id}")
        tracks[track_id] = track
    return path, tracks, range_name


def _validate_optional_request_fields(request_meta: dict, key: str) -> None:
    if "n_supports" in request_meta:
        _required_track_id(
            request_meta["n_supports"],
            f"audit_meta.requests[{key!r}].n_supports")
    if "support_obs_by_t" in request_meta:
        supports = _required_object(
            request_meta["support_obs_by_t"],
            f"audit_meta.requests[{key!r}].support_obs_by_t")
        for relative_t, observation_id in supports.items():
            _required_string(relative_t, f"{key} support relative time")
            _required_string(observation_id, f"{key} support observation id")
    if "chips" in request_meta:
        chips = request_meta["chips"]
        if not isinstance(chips, list):
            raise AuditArtifactError(f"{key}: chips must be a list")
        for chip in chips:
            chip = _required_string(chip, f"{key} chip path")
            if Path(chip).is_absolute() or ".." in Path(chip).parts:
                raise AuditArtifactError(
                    f"{key}: chip paths must be artifact-relative")


def _validate_bound_meta(meta: dict, tracks_path: Path, tracks: dict,
                         range_name: str, tracks_ref: artifact.ArtifactRef):
    _require_exact_keys(meta, _META_KEYS, "audit_meta.json")
    if meta["schema"] != META_SCHEMA:
        raise AuditArtifactError(
            f"audit_meta.json schema must be {META_SCHEMA!r}; "
            f"got {meta['schema']!r}")
    source = _required_object(
        meta["source_tracks"], "audit_meta.source_tracks")
    _require_exact_keys(source, _SOURCE_KEYS, "audit_meta.source_tracks")
    artifact_id = _required_string(
        source["artifact_id"], "audit_meta.source_tracks.artifact_id")
    expected_artifact_id = source_artifact_id(tracks_ref)
    if artifact_id != expected_artifact_id:
        raise AuditArtifactError(
            "audit metadata source artifact identity mismatch: "
            f"expected {expected_artifact_id!r}, found {artifact_id!r}")
    source_file = _required_string(
        source["file"], "audit_meta.source_tracks.file")
    if Path(source_file).name != source_file:
        raise AuditArtifactError(
            "audit_meta.source_tracks.file must be a basename")
    if source_file != tracks_path.name:
        raise AuditArtifactError(
            f"audit metadata binds {source_file!r}, but the only source file "
            f"is {tracks_path.name!r}")
    expected_file_digest = _required_sha256(
        source["sha256"], "audit_meta.source_tracks.sha256")
    actual_file_digest = file_sha256(tracks_path)
    if expected_file_digest != actual_file_digest:
        raise AuditArtifactError(
            f"source track file digest mismatch for {tracks_path}: "
            f"metadata has {expected_file_digest}, bytes have "
            f"{actual_file_digest}")

    requests = _required_object(meta["requests"], "audit_meta.requests")
    seen_track_ids = set()
    allowed = _REQUEST_REQUIRED_KEYS | _REQUEST_OPTIONAL_KEYS
    for key, request_meta in requests.items():
        _required_string(key, "audit request key")
        request_meta = _required_object(
            request_meta, f"audit_meta.requests[{key!r}]")
        actual_keys = frozenset(request_meta)
        missing = _REQUEST_REQUIRED_KEYS - actual_keys
        unknown = actual_keys - allowed
        if missing or unknown:
            raise AuditArtifactError(
                f"audit_meta.requests[{key!r}] has invalid fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}")
        track_id = _required_track_id(
            request_meta["track_id"],
            f"audit_meta.requests[{key!r}].track_id")
        if key != f"T{track_id}":
            raise AuditArtifactError(
                f"audit key {key!r} does not identify track {track_id}")
        if track_id in seen_track_ids:
            raise AuditArtifactError(
                f"more than one audit request resolves to track {track_id}")
        seen_track_ids.add(track_id)
        track = tracks.get(track_id)
        if track is None:
            raise AuditArtifactError(
                f"audit key {key!r} refers to missing source track {track_id}")
        expected_track_digest = _required_sha256(
            request_meta["source_track_sha256"],
            f"audit_meta.requests[{key!r}].source_track_sha256")
        if expected_track_digest != canonical_sha256(track):
            raise AuditArtifactError(
                f"audit key {key!r} source-track digest mismatch")
        if ("birth_keyframe" in request_meta
                and request_meta["birth_keyframe"]
                != track.get("birth_keyframe")):
            raise AuditArtifactError(
                f"audit key {key!r} birth_keyframe does not match its track")
        if ("range" in request_meta
                and request_meta["range"] != range_name):
            raise AuditArtifactError(
                f"audit key {key!r} range does not match its track artifact")
        _validate_optional_request_fields(request_meta, key)

    return requests, {
        "source_tracks_artifact_id": artifact_id,
        "source_tracks_file": source_file,
        "source_tracks_sha256": actual_file_digest,
        "source_tracks_manifest_sha256": tracks_ref.manifest_digest,
        "source_tracks_content_sha256": tracks_ref.content_digest,
    }


def _require_exact_response_shape(raw, canonical, path="audit"):
    if isinstance(canonical, dict):
        if not isinstance(raw, dict) or set(raw) != set(canonical):
            raw_keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise AuditArtifactError(
                f"{path} fields must be exactly {sorted(canonical)}; "
                f"found {raw_keys}")
        for key, value in canonical.items():
            _require_exact_response_shape(raw[key], value, f"{path}.{key}")
    elif isinstance(canonical, list):
        if not isinstance(raw, list) or len(raw) != len(canonical):
            raise AuditArtifactError(f"{path} has the wrong list shape")
        for index, value in enumerate(canonical):
            _require_exact_response_shape(
                raw[index], value, f"{path}[{index}]")


def _load_result_record(line: str, path: Path, line_number: int):
    try:
        record = json.loads(
            line, object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant)
        record = _required_object(record, f"{path}:{line_number}")
        _require_exact_keys(
            record, {"key", "response"}, f"{path}:{line_number}")
        key = _required_string(record["key"], f"{path}:{line_number}.key")
        response = _required_object(
            record["response"], f"{path}:{line_number}.response")
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        if not isinstance(text, str):
            raise AuditArtifactError("audit response text must be a string")
        raw = json.loads(
            text, object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant)
        audit_value = semantic_audit.TrackAudit.model_validate(raw).model_dump()
        _require_exact_response_shape(raw, audit_value)
        return key, audit_value
    except AuditArtifactError:
        raise
    except Exception as error:
        raise AuditArtifactError(
            f"{path}:{line_number}: invalid canonical audit result: {error}") \
            from error


def _validate_weight(value, label: str):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise AuditArtifactError(f"{label} must be finite and within [0, 1]")


def _validate_audit_semantics(audit_value: dict, key: str):
    verdict = audit_value["verdict"]
    segments = audit_value["valid_segments"]
    single_object = audit_value["single_object"]
    drop_reason = audit_value["drop_reason"]
    if verdict == "keep":
        if not single_object or drop_reason != "none" or not segments:
            raise AuditArtifactError(
                f"{key}: keep requires single_object=true, drop_reason=none, "
                "and at least one valid segment")
    elif verdict == "keep_partial":
        if single_object or drop_reason != "none" or not segments:
            raise AuditArtifactError(
                f"{key}: keep_partial requires single_object=false, "
                "drop_reason=none, and at least one valid segment")
    elif drop_reason == "none":
        raise AuditArtifactError(
            f"{key}: verdict=drop requires a concrete drop_reason")

    for index, item in enumerate(audit_value["primary_object"]["tags"]):
        _validate_weight(
            item["weight"], f"{key}: primary_object.tags[{index}].weight")
    for index, item in enumerate(
            audit_value["primary_object"]["name_candidates"]):
        _validate_weight(
            item["weight"],
            f"{key}: primary_object.name_candidates[{index}].weight")
    for secondary_index, secondary in enumerate(
            audit_value["secondary_objects"]):
        for tag_index, item in enumerate(secondary["tags"]):
            _validate_weight(
                item["weight"],
                f"{key}: secondary_objects[{secondary_index}]."
                f"tags[{tag_index}].weight")


def load_audits(tracks_dir: Path,
                semantic_audits_dir: Path) -> AuditResults:
    """Load one canonical audit artifact bound to one tracks artifact."""
    tracks_dir = Path(tracks_dir)
    semantic_audits_dir = Path(semantic_audits_dir)
    tracks_ref = _open_artifact(
        tracks_dir, kind=paths_lib.OBJECT_TRACKS)
    semantic_audits_ref = _open_artifact(
        semantic_audits_dir, kind=paths_lib.SEMANTIC_AUDITS,
        dataset_name=tracks_ref.dataset)
    audit_manifest = artifact.load_manifest(semantic_audits_dir)
    if tracks_ref not in audit_manifest.upstreams:
        raise AuditArtifactError(
            "semantic_audits manifest is not bound to the supplied "
            "object_tracks artifact")

    tracks_path, tracks, range_name = _load_one_tracks_artifact(tracks_dir)
    meta_path = semantic_audits_dir / "audit_meta.json"
    results_path = semantic_audits_dir / "results.jsonl"
    meta = _required_object(_load_json(meta_path), str(meta_path))
    requests, source_provenance = _validate_bound_meta(
        meta, tracks_path, tracks, range_name, tracks_ref)
    expected_keys = set(requests)
    config = audit_manifest.config
    if (config.get("phase") != "canonical_results"
            or config.get("coverage") != "complete"
            or config.get("n_expected") != len(expected_keys)
            or config.get("n_successful") != len(expected_keys)):
        raise AuditArtifactError(
            "semantic_audits manifest does not attest complete canonical "
            "coverage for its metadata request set")

    successes = {}
    try:
        stream = results_path.open(encoding="utf-8")
    except OSError as error:
        raise AuditArtifactError(
            f"cannot open canonical audit results {results_path}: {error}") \
            from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise AuditArtifactError(
                    f"{results_path}:{line_number}: blank result record")
            key, audit_value = _load_result_record(
                line, results_path, line_number)
            if key not in expected_keys:
                raise AuditArtifactError(
                    f"{results_path}:{line_number}: unexpected result key "
                    f"{key!r}")
            if key in successes:
                raise AuditArtifactError(
                    f"{results_path}:{line_number}: duplicate canonical "
                    f"result for {key!r}")
            _validate_audit_semantics(audit_value, key)
            successes[key] = audit_value
    missing = sorted(expected_keys - set(successes))
    if missing:
        raise AuditArtifactError(
            f"semantic audit lacks canonical results for {missing}")

    audits = {}
    provenance_by_track = {}
    meta_digest = file_sha256(meta_path)
    results_digest = file_sha256(results_path)
    for key in sorted(expected_keys):
        request_meta = requests[key]
        track_id = request_meta["track_id"]
        audit_value = successes[key]
        audits[track_id] = audit_value
        provenance_by_track[track_id] = {
            **source_provenance,
            "source_track_sha256": request_meta["source_track_sha256"],
            "audit_key": key,
            "audit_payload_sha256": canonical_sha256(audit_value),
            "audit_meta_sha256": meta_digest,
            "audit_results_sha256": results_digest,
            "semantic_audits_manifest_sha256": (
                semantic_audits_ref.manifest_digest),
            "semantic_audits_content_sha256": (
                semantic_audits_ref.content_digest),
            "result_attempts": 1,
            "failed_attempts": (),
        }
    return AuditResults(
        audits,
        provenance_by_track=provenance_by_track,
        source_tracks=tracks,
        tracks_ref=tracks_ref,
        semantic_audits_ref=semantic_audits_ref)
