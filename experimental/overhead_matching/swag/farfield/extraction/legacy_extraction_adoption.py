"""Verify legacy paid extraction before planning typed, zero-call adoption.

This module is deliberately a verifier, not a compatibility reader and not a
publisher.  It reads an explicitly enumerated legacy request/result history,
proves that every request used the bytes in one already-typed pinhole artifact,
and builds the exact current canonical results in memory.  The command prints a
deterministic publication plan; it never calls a provider or writes an artifact.

Only the two result shapes observed in the retained farfield runs are accepted:

* Vertex batch: ``{key, processed_time, request, response, status}``
* Online retry: ``{error, key, request, response}``

Every raw source and every failed attempt remains represented by a content
digest in the report.  A small, isolated compatibility boundary may discard a
geometrically invalid bounding box, but every such discard is recorded in the
sanitation ledger and the resulting prediction is revalidated by the normal
current extraction validator.  Normal artifact readers are unchanged.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    llm_lifecycle,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    extract_landmarks,
    prompts,
)


SPEC_SCHEMA = "farfield.legacy_extraction_adoption_spec/v1"
REPORT_SCHEMA = "farfield.legacy_extraction_adoption_report/v1"
ATTEMPT_PROVENANCE_SCHEMA = "farfield.legacy_extraction_attempt_source/v1"

REQUEST_ROLE_PRIMARY = "primary"
REQUEST_ROLE_RETRY = "retry"
REQUEST_ROLES = frozenset({REQUEST_ROLE_PRIMARY, REQUEST_ROLE_RETRY})

RESULT_FORMAT_VERTEX_BATCH = "vertex_batch_v1"
RESULT_FORMAT_ONLINE_RETRY = "online_retry_v1"
RESULT_FORMATS = frozenset({
    RESULT_FORMAT_VERTEX_BATCH,
    RESULT_FORMAT_ONLINE_RETRY,
})

_SPEC_KEYS = frozenset({
    "schema", "dataset", "request_set", "pinhole_dir",
    "request_sources", "result_sources", "empty_error_sidecars",
})
_REQUEST_SOURCE_KEYS = frozenset({"id", "path", "role"})
_RESULT_SOURCE_KEYS = frozenset({"id", "path", "format"})
_SIDECAR_SOURCE_KEYS = frozenset({"id", "path"})
_VERTEX_BATCH_KEYS = frozenset({
    "key", "processed_time", "request", "response", "status",
})
_ONLINE_RETRY_KEYS = frozenset({"error", "key", "request", "response"})
_REQUEST_RECORD_KEYS = frozenset({"key", "request"})
_BOX_KEYS = frozenset({"yaw_angle", "xmin", "ymin", "xmax", "ymax"})
_VALID_YAWS = frozenset({"0", "90", "180", "270"})

# Exact provider decorations observed in the retained sources. These are not
# general compatibility aliases: removal requires this path and exact value.
_PRIMARY_NULL_PATHS = (
    ("contents", 0, "parts", 0, "text"),
    ("contents", 0, "parts", 1, "text"),
    ("contents", 0, "parts", 2, "text"),
    ("contents", 0, "parts", 3, "text"),
    ("contents", 0, "parts", 4, "inline_data"),
    ("contents", 0, "parts", 4, "media_resolution"),
)
_RETRY_PROPERTY_ORDERING = (
    (("generationConfig", "responseSchema", "properties", "landmarks",
      "items", "property_ordering"),
     ["primary_tag", "additional_tags", "confidence", "bounding_boxes",
      "description"]),
    (("generationConfig", "responseSchema", "properties", "landmarks",
      "items", "properties", "primary_tag", "property_ordering"),
     ["key", "value"]),
    (("generationConfig", "responseSchema", "properties", "landmarks",
      "items", "properties", "additional_tags", "items",
      "property_ordering"),
     ["key", "value"]),
    (("generationConfig", "responseSchema", "properties", "landmarks",
      "items", "properties", "bounding_boxes", "items",
      "property_ordering"),
     ["yaw_angle", "ymin", "xmin", "ymax", "xmax"]),
)


class AdoptionError(ValueError):
    """Legacy material cannot be adopted without guessing or losing lineage."""


@dataclass(frozen=True)
class LegacyRequestSource:
    source_id: str
    path: Path
    role: str


@dataclass(frozen=True)
class LegacyResultSource:
    source_id: str
    path: Path
    result_format: str


@dataclass(frozen=True)
class EmptyErrorSidecar:
    source_id: str
    path: Path


@dataclass(frozen=True)
class AdoptionPlan:
    """Verified in-memory inputs for a future transactional publisher."""

    attempts: tuple[llm_lifecycle.Attempt, ...]
    canonical_results: tuple[llm_lifecycle.CanonicalResult, ...]
    canonical_results_bytes: bytes
    predictions_bytes: bytes
    report: dict[str, Any]


@dataclass(frozen=True)
class AdoptionSpec:
    dataset: str
    request_set_path: Path
    pinhole_dir: Path
    request_sources: tuple[LegacyRequestSource, ...]
    result_sources: tuple[LegacyResultSource, ...]
    empty_error_sidecars: tuple[EmptyErrorSidecar, ...]
    spec_sha256: str


@dataclass(frozen=True)
class _ResultPrecursor:
    source_id: str
    source_path: str
    source_sha256: str
    result_format: str
    line_number: int
    raw_line_sha256: str
    key: str
    raw_echoed_request_sha256: str
    normalized_echoed_request_sha256: str
    bound_request_role: str
    bound_request_raw_sha256: str
    bound_request_normalized_sha256: str
    current_request_sha256: str
    bound_request_snapshots: tuple[dict[str, Any], ...]
    normalization_events: tuple[dict[str, Any], ...]
    transport_metadata_sha256: str
    transport_metadata_fields: tuple[str, ...]
    response: dict[str, Any] | None
    error: Any
    outcome: str


@dataclass(frozen=True)
class _RequestSnapshot:
    source_id: str
    source_path: str
    source_sha256: str
    source_role: str
    line_number: int
    raw_line_sha256: str
    key: str
    raw_request_sha256: str
    normalized_request_sha256: str


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                where: str) -> None:
    actual = frozenset(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing {missing}")
    if unknown:
        details.append(f"unknown {unknown}")
    raise AdoptionError(f"{where} has an invalid shape: {', '.join(details)}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise AdoptionError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise AdoptionError(f"non-finite JSON constant {value!r}")


def _strict_json_bytes(data: bytes, where: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AdoptionError as error:
        raise AdoptionError(f"{where}: invalid strict JSON: {error}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AdoptionError(f"{where}: invalid strict JSON: {error}") from error


def _json_clone(value: Any) -> Any:
    return json.loads(artifact.canonical_json_bytes(value))


def _json_pointer(path: Sequence[str | int]) -> str:
    return "/" + "/".join(str(component) for component in path)


def _remove_exact_path(value: Any, path: Sequence[str | int], expected: Any,
                       action: str) -> dict[str, Any] | None:
    parent = value
    for component in path[:-1]:
        if isinstance(component, int):
            if (not isinstance(parent, list)
                    or not 0 <= component < len(parent)):
                return None
            parent = parent[component]
        else:
            if not isinstance(parent, dict) or component not in parent:
                return None
            parent = parent[component]
    leaf = path[-1]
    if not isinstance(leaf, str) or not isinstance(parent, dict):
        return None
    if leaf not in parent or parent[leaf] != expected:
        return None
    del parent[leaf]
    return {"action": action, "path": _json_pointer(path), "value": expected}


def _normalize_preserved_request(
        request: Mapping[str, Any], role: str,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = _json_clone(request)
    events = []
    if role == REQUEST_ROLE_PRIMARY:
        for path in _PRIMARY_NULL_PATHS:
            event = _remove_exact_path(
                normalized, path, None, "remove_exact_primary_null")
            if event is None:
                raise AdoptionError(
                    "preserved primary request lacks the exact observed null "
                    f"decoration at {_json_pointer(path)}")
            events.append(event)
    return normalized, events


def _normalize_provider_echo(
        request: Mapping[str, Any], result_format: str,
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = _json_clone(request)
    events = []
    if result_format == RESULT_FORMAT_ONLINE_RETRY:
        for path, expected in _RETRY_PROPERTY_ORDERING:
            event = _remove_exact_path(
                normalized, path, expected,
                "remove_exact_retry_property_ordering")
            if event is None:
                raise AdoptionError(
                    "online retry provider echo lacks the exact observed "
                    "property_ordering decoration at "
                    f"{_json_pointer(path)}")
            events.append(event)
    return normalized, events


def _regular_file(path: Path, where: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AdoptionError(f"{where} is not a regular non-symlink file: {path}")
    return path


def _regular_directory(path: Path, where: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise AdoptionError(
            f"{where} is not a regular non-symlink directory: {path}")
    return path


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdoptionError(f"{where} must be a non-empty string")
    return value


def _source_path(base: Path, value: Any, where: str) -> Path:
    text = _nonempty_string(value, where)
    path = Path(text)
    return path if path.is_absolute() else base / path


def _load_source_list(value: Any, *, base: Path, where: str,
                      expected_keys: frozenset[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AdoptionError(f"{where} must be a list")
    records = []
    for index, item in enumerate(value):
        item_where = f"{where}[{index}]"
        if not isinstance(item, dict):
            raise AdoptionError(f"{item_where} must be an object")
        _exact_keys(item, expected_keys, item_where)
        record = dict(item)
        record["id"] = _nonempty_string(record["id"], f"{item_where}.id")
        record["path"] = _source_path(
            base, record["path"], f"{item_where}.path")
        records.append(record)
    return records


def load_spec(path: Path | str) -> AdoptionSpec:
    """Load a strict, explicit source inventory for a report-only run."""
    spec_path = _regular_file(Path(path), "adoption spec")
    data = spec_path.read_bytes()
    value = _strict_json_bytes(data, str(spec_path))
    if not isinstance(value, dict):
        raise AdoptionError("adoption spec must be a JSON object")
    _exact_keys(value, _SPEC_KEYS, "adoption spec")
    if value["schema"] != SPEC_SCHEMA:
        raise AdoptionError(
            f"unsupported adoption spec schema {value['schema']!r}")
    dataset = artifact.require_identifier(value["dataset"], "dataset")
    base = spec_path.parent

    request_records = _load_source_list(
        value["request_sources"], base=base, where="request_sources",
        expected_keys=_REQUEST_SOURCE_KEYS)
    result_records = _load_source_list(
        value["result_sources"], base=base, where="result_sources",
        expected_keys=_RESULT_SOURCE_KEYS)
    sidecar_records = _load_source_list(
        value["empty_error_sidecars"], base=base,
        where="empty_error_sidecars", expected_keys=_SIDECAR_SOURCE_KEYS)

    requests = []
    for index, record in enumerate(request_records):
        role = record["role"]
        if role not in REQUEST_ROLES:
            raise AdoptionError(
                f"request_sources[{index}].role must be one of "
                f"{sorted(REQUEST_ROLES)}")
        requests.append(LegacyRequestSource(
            source_id=record["id"], path=record["path"], role=role))

    results = []
    for index, record in enumerate(result_records):
        result_format = record["format"]
        if result_format not in RESULT_FORMATS:
            raise AdoptionError(
                f"result_sources[{index}].format must be one of "
                f"{sorted(RESULT_FORMATS)}")
        results.append(LegacyResultSource(
            source_id=record["id"], path=record["path"],
            result_format=result_format))

    sidecars = tuple(EmptyErrorSidecar(
        source_id=record["id"], path=record["path"])
        for record in sidecar_records)
    all_sources: Sequence[Any] = (*requests, *results, *sidecars)
    ids = [source.source_id for source in all_sources]
    if len(ids) != len(set(ids)):
        raise AdoptionError("every legacy source id must be globally unique")
    normalized_paths = [str(Path(source.path).absolute())
                        for source in all_sources]
    if len(normalized_paths) != len(set(normalized_paths)):
        raise AdoptionError("every legacy source path must be listed exactly once")
    if not requests or not any(source.role == REQUEST_ROLE_PRIMARY
                               for source in requests):
        raise AdoptionError("at least one primary request source is required")
    if not results:
        raise AdoptionError("at least one result source is required")

    return AdoptionSpec(
        dataset=dataset,
        request_set_path=_source_path(
            base, value["request_set"], "request_set"),
        pinhole_dir=_source_path(base, value["pinhole_dir"], "pinhole_dir"),
        request_sources=tuple(requests),
        result_sources=tuple(results),
        empty_error_sidecars=sidecars,
        spec_sha256=hashlib.sha256(data).hexdigest(),
    )


def _request_contract(
        dataset: str, request_set: llm_lifecycle.RequestSet,
        ) -> tuple[artifact.ArtifactRef, dict[str, dict[str, Any]]]:
    if request_set.stage != "frame_landmark_extraction":
        raise AdoptionError(
            "request set is not a frame_landmark_extraction workload")
    if len(request_set.upstreams) != 1:
        raise AdoptionError("request set must bind exactly one pinhole artifact")
    pinhole_ref = request_set.upstreams[0]
    if (pinhole_ref.kind != paths_lib.PINHOLE_IMAGES
            or pinhole_ref.dataset != dataset):
        raise AdoptionError(
            "request set binds the wrong pinhole artifact kind or dataset")

    value = request_set.to_dict()
    media = value["media_settings"]
    expected_media_keys = {
        "prompt_type", "pinhole_resolution", "media_resolution",
        "thinking_level", "face_order",
    }
    if not isinstance(media, dict) or set(media) != expected_media_keys:
        raise AdoptionError("request set has an invalid extraction media contract")
    prompt_type = media["prompt_type"]
    if prompt_type not in prompts.SYSTEM_PROMPTS:
        raise AdoptionError(f"unknown prompt type {prompt_type!r}")
    if value["system_prompt"] != prompts.SYSTEM_PROMPTS[prompt_type]:
        raise AdoptionError("request set system prompt differs from the registry")
    if value["response_schema"] != prompts.response_schema():
        raise AdoptionError("request set response schema differs from current v1")
    if media["face_order"] != list(prompts.PINHOLE_FACES):
        raise AdoptionError("request set face order is not canonical")
    if (type(media["pinhole_resolution"]) is not int
            or media["pinhole_resolution"] <= 0):
        raise AdoptionError("pinhole resolution must be a positive integer")

    expected = {}
    for index, unit in enumerate(value["units"]):
        key = unit["key"]
        if unit["metadata"] != {"panorama_stem": key}:
            raise AdoptionError(
                f"request unit {key!r} has noncanonical panorama metadata")
        try:
            semantic = prompts.semantic_request_from_batch(
                key, unit["request"])
        except Exception as error:
            raise AdoptionError(
                f"request unit {key!r} is not a canonical batch request: "
                f"{error}") from error
        if (semantic.system_instruction != value["system_prompt"]
                or semantic.response_schema != value["response_schema"]
                or semantic.thinking_level != media["thinking_level"]
                or semantic.media_resolution != media["media_resolution"]):
            raise AdoptionError(
                f"request unit {key!r} disagrees with request-set semantics")
        if (len(semantic.parts) != len(prompts.PINHOLE_FACES) + 1
                or semantic.parts[-1] != {"text": prompts.USER_PROMPT}
                or any("inline_data" not in part
                       for part in semantic.parts[:-1])):
            raise AdoptionError(
                f"request unit {key!r} does not carry four faces then the "
                "canonical user prompt")
        if key in expected:
            raise AdoptionError(f"duplicate request-set key {key!r}")
        expected[key] = {
            "index": index,
            "request": unit["request"],
            "request_sha256": artifact.sha256_json(unit["request"]),
            "semantic": semantic,
        }
    if not expected:
        raise AdoptionError("request set contains no panorama units")
    return pinhole_ref, expected


def _pinhole_inventory(
        root: Path, expected: Mapping[str, Mapping[str, Any]],
        resolution: int, pinhole_ref: artifact.ArtifactRef,
        ) -> dict[str, Any]:
    root = _regular_directory(root, "pinhole directory")
    try:
        validated_ref = artifact.open_artifact(
            root,
            expected_kind=pinhole_ref.kind,
            expected_dataset=pinhole_ref.dataset,
            expected_version=pinhole_ref.version,
        )
    except artifact.ArtifactError as error:
        raise AdoptionError(
            f"pinhole directory is not a valid completed typed artifact: "
            f"{error}") from error
    if validated_ref.to_dict() != pinhole_ref.to_dict():
        raise AdoptionError(
            "validated pinhole artifact does not exactly match the request-set "
            "upstream identity, including path and manifest/content digests")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    allowed_files = {artifact.MANIFEST_NAME}
    actual_dirs = set()
    ancillary = []
    for entry in entries:
        if entry.is_symlink():
            raise AdoptionError(f"pinhole directory contains symlink {entry}")
        if entry.is_dir():
            actual_dirs.add(entry.name)
        elif entry.is_file() and entry.name in allowed_files:
            ancillary.append({
                "path": entry.name,
                "size": entry.stat().st_size,
                "sha256": artifact.sha256_file(entry),
            })
        else:
            raise AdoptionError(
                f"unexpected entry in pinhole directory: {entry.name!r}")
    wanted = set(expected)
    if actual_dirs != wanted:
        raise AdoptionError(
            "pinhole stems do not exactly cover request keys: "
            f"missing {sorted(wanted - actual_dirs)}, "
            f"unknown {sorted(actual_dirs - wanted)}")

    records = []
    total_bytes = 0
    for key, contract in expected.items():
        stem_dir = _regular_directory(root / key, f"pinhole stem {key!r}")
        wanted_names = {f"{face}.jpg" for face in prompts.PINHOLE_FACES}
        stem_entries = sorted(stem_dir.iterdir(), key=lambda item: item.name)
        actual_names = {entry.name for entry in stem_entries}
        if actual_names != wanted_names:
            raise AdoptionError(
                f"pinhole stem {key!r} must contain exactly "
                f"{sorted(wanted_names)}; found {sorted(actual_names)}")
        semantic = contract["semantic"]
        for face, part in zip(prompts.PINHOLE_FACES, semantic.parts[:-1]):
            face_path = stem_dir / f"{face}.jpg"
            _regular_file(face_path, f"pinhole face {key}/{face}")
            inline = part["inline_data"]
            if inline["mime_type"] != "image/jpeg":
                raise AdoptionError(
                    f"request {key!r} {face} has MIME "
                    f"{inline['mime_type']!r}, expected image/jpeg")
            try:
                request_bytes = base64.b64decode(
                    inline["data"].encode("ascii"), validate=True)
            except (UnicodeError, ValueError) as error:
                raise AdoptionError(
                    f"request {key!r} {face} has invalid base64 media") from error
            face_bytes = face_path.read_bytes()
            if request_bytes != face_bytes:
                raise AdoptionError(
                    f"request media differs from retained pinhole bytes: "
                    f"{key}/{face}.jpg")
            try:
                with Image.open(io.BytesIO(face_bytes)) as image:
                    if image.format != "JPEG" or image.size != (
                            resolution, resolution):
                        raise AdoptionError(
                            f"{key}/{face}.jpg must be a {resolution}x"
                            f"{resolution} JPEG; found {image.format} "
                            f"{image.size}")
                    image.verify()
            except AdoptionError:
                raise
            except OSError as error:
                raise AdoptionError(
                    f"cannot decode pinhole face {key}/{face}.jpg: {error}") \
                    from error
            relative = f"{key}/{face}.jpg"
            digest = hashlib.sha256(face_bytes).hexdigest()
            records.append({
                "path": relative,
                "size": len(face_bytes),
                "sha256": digest,
            })
            total_bytes += len(face_bytes)

    records.sort(key=lambda record: record["path"])
    content_digest = artifact.sha256_json(records)
    if content_digest != pinhole_ref.content_digest:
        raise AdoptionError(
            "retained pinhole content digest does not match the request-set "
            f"upstream ({content_digest} != {pinhole_ref.content_digest})")
    return {
        "path": str(root),
        "artifact_ref": validated_ref.to_dict(),
        "n_panoramas": len(expected),
        "n_faces": len(records),
        "n_bytes": total_bytes,
        "content_digest": content_digest,
        "request_media_match": "complete",
        "ancillary_source_files": ancillary,
    }


def _request_sources(
        sources: Sequence[LegacyRequestSource],
        expected: Mapping[str, Mapping[str, Any]],
        ) -> tuple[list[dict[str, Any]],
                   dict[tuple[str, str], tuple[_RequestSnapshot, ...]],
                   list[dict[str, Any]]]:
    expected_keys = list(expected)
    primary_keys = []
    reports = []
    snapshots: dict[tuple[str, str], list[_RequestSnapshot]] = {}
    normalization_ledger = []
    for source in sources:
        if source.role not in REQUEST_ROLES:
            raise AdoptionError(
                f"request source {source.source_id!r} has unknown role "
                f"{source.role!r}")
        keys = []
        seen = set()
        digest = hashlib.sha256()
        size = 0
        pending = []
        path = _regular_file(source.path, f"request source {source.source_id}")
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                size += len(raw_line)
                if not raw_line.strip():
                    raise AdoptionError(f"{path}:{line_number}: blank request")
                value = _strict_json_bytes(raw_line, f"{path}:{line_number}")
                if not isinstance(value, dict):
                    raise AdoptionError(
                        f"{path}:{line_number}: request must be an object")
                _exact_keys(value, _REQUEST_RECORD_KEYS,
                            f"{path}:{line_number} request")
                key = _nonempty_string(
                    value["key"], f"{path}:{line_number} key")
                if key not in expected:
                    raise AdoptionError(
                        f"{path}:{line_number}: unknown request key {key!r}")
                if key in seen:
                    raise AdoptionError(
                        f"{path}: duplicate request key {key!r} within one "
                        "source")
                seen.add(key)
                request = value["request"]
                if not isinstance(request, dict):
                    raise AdoptionError(
                        f"{path}:{line_number}: request payload must be an "
                        "object")
                raw_digest = artifact.sha256_json(request)
                normalized, events = _normalize_preserved_request(
                    request, source.role)
                normalized_digest = artifact.sha256_json(normalized)
                if normalized_digest != expected[key]["request_sha256"]:
                    raise AdoptionError(
                        f"{path}:{line_number}: request {key!r} conflicts "
                        "with the current request set after exact allowlisted "
                        "normalization")
                raw_line_digest = hashlib.sha256(raw_line).hexdigest()
                contextual_events = [{
                    "scope": "preserved_request",
                    "source_id": source.source_id,
                    "source_path": str(path),
                    "source_role": source.role,
                    "line_number": line_number,
                    "key": key,
                    **event,
                } for event in events]
                normalization_ledger.extend(contextual_events)
                pending.append({
                    "line_number": line_number,
                    "raw_line_sha256": raw_line_digest,
                    "key": key,
                    "raw_request_sha256": raw_digest,
                    "normalized_request_sha256": normalized_digest,
                    "n_normalization_events": len(events),
                })
                keys.append(key)
        if not keys:
            raise AdoptionError(f"request source {path} is empty")
        if source.role == REQUEST_ROLE_PRIMARY:
            primary_keys.extend(keys)
        source_sha256 = digest.hexdigest()
        for record in pending:
            snapshot = _RequestSnapshot(
                source_id=source.source_id,
                source_path=str(path),
                source_sha256=source_sha256,
                source_role=source.role,
                line_number=record["line_number"],
                raw_line_sha256=record["raw_line_sha256"],
                key=record["key"],
                raw_request_sha256=record["raw_request_sha256"],
                normalized_request_sha256=record[
                    "normalized_request_sha256"],
            )
            snapshots.setdefault((source.role, record["key"]), []).append(
                snapshot)
        reports.append({
            "id": source.source_id,
            "path": str(path),
            "role": source.role,
            "sha256": source_sha256,
            "size": size,
            "n_records": len(keys),
            "keys_sha256": artifact.sha256_json(keys),
            "records": pending,
            "requests_match_current_set": True,
            "matching_rule": "exact_after_allowlisted_role_normalization",
        })
    if primary_keys != expected_keys or len(primary_keys) != len(set(primary_keys)):
        raise AdoptionError(
            "primary request sources must exactly cover request-set order once: "
            f"expected {expected_keys}, found {primary_keys}")
    return (reports,
            {key: tuple(value) for key, value in snapshots.items()},
            normalization_ledger)


def _snapshot_ref(snapshot: _RequestSnapshot) -> dict[str, Any]:
    return {
        "source_id": snapshot.source_id,
        "source_path": snapshot.source_path,
        "source_sha256": snapshot.source_sha256,
        "source_role": snapshot.source_role,
        "line_number": snapshot.line_number,
        "raw_line_sha256": snapshot.raw_line_sha256,
        "raw_request_sha256": snapshot.raw_request_sha256,
        "normalized_request_sha256": snapshot.normalized_request_sha256,
    }


def _empty_sidecars(
        sources: Sequence[EmptyErrorSidecar]) -> list[dict[str, Any]]:
    reports = []
    for source in sources:
        path = _regular_file(source.path, f"error sidecar {source.source_id}")
        size = path.stat().st_size
        digest = artifact.sha256_file(path)
        if size != 0:
            raise AdoptionError(
                f"error sidecar {path} is nonempty; its unobserved record "
                "shape requires explicit support before adoption")
        reports.append({
            "id": source.source_id,
            "path": str(path),
            "sha256": digest,
            "size": 0,
            "shape": "verified_empty",
        })
    return reports


def _response_payload(response: Mapping[str, Any]) -> tuple[dict[str, Any], dict]:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise AdoptionError("response must contain exactly one candidate")
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, Mapping) else None
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list) or len(parts) != 1:
        raise AdoptionError("response candidate must contain exactly one part")
    part = parts[0]
    text = part.get("text") if isinstance(part, Mapping) else None
    if not isinstance(text, str) or not text:
        raise AdoptionError("response part must contain non-empty JSON text")
    payload = _strict_json_bytes(text.encode("utf-8"), "response text")
    if not isinstance(payload, dict):
        raise AdoptionError("response text must encode an object")
    # Detached ordinary JSON lets us replace only the response text below.
    detached = json.loads(artifact.canonical_json_bytes(response))
    return payload, detached


def _invalid_box_reason(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "not_object"
    actual = frozenset(value)
    if actual != _BOX_KEYS:
        if _BOX_KEYS - actual:
            return "missing_fields"
        return "unknown_fields"
    if value["yaw_angle"] not in _VALID_YAWS:
        return "invalid_yaw"
    for name in ("xmin", "ymin", "xmax", "ymax"):
        coordinate = value[name]
        if type(coordinate) is not int:
            return "non_integer_coordinate"
        if not 0 <= coordinate <= 1000:
            return "coordinate_out_of_range"
    if value["xmin"] >= value["xmax"] or value["ymin"] >= value["ymax"]:
        return "nonpositive_extent"
    return None


def _sanitize_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list]:
    if set(payload) != {"location_type", "landmarks"}:
        raise AdoptionError(
            "prediction must have exact keys location_type and landmarks")
    landmarks = payload["landmarks"]
    if not isinstance(landmarks, list):
        raise AdoptionError("prediction.landmarks must be an array")
    sanitized = json.loads(artifact.canonical_json_bytes(payload))
    kept_landmarks = []
    events = []
    for landmark_index, landmark in enumerate(sanitized["landmarks"]):
        if not isinstance(landmark, dict) or "bounding_boxes" not in landmark:
            raise AdoptionError(
                f"prediction.landmarks[{landmark_index}] is not a landmark "
                "object with bounding_boxes")
        boxes = landmark["bounding_boxes"]
        if not isinstance(boxes, list):
            raise AdoptionError(
                f"prediction.landmarks[{landmark_index}].bounding_boxes must "
                "be an array")
        kept_boxes = []
        for box_index, box in enumerate(boxes):
            reason = _invalid_box_reason(box)
            if reason is None:
                kept_boxes.append(box)
                continue
            events.append({
                "action": "drop_bbox",
                "landmark_index": landmark_index,
                "box_index": box_index,
                "reason": reason,
                "raw_sha256": artifact.sha256_json(box),
            })
        landmark["bounding_boxes"] = kept_boxes
        if kept_boxes:
            kept_landmarks.append(landmark)
        elif boxes:
            events.append({
                "action": "drop_landmark_no_valid_boxes",
                "landmark_index": landmark_index,
                "reason": "all_boxes_rejected",
                "raw_sha256": artifact.sha256_json(
                    payload["landmarks"][landmark_index]),
            })
        else:
            events.append({
                "action": "drop_landmark_empty_boxes",
                "landmark_index": landmark_index,
                "reason": "model_authored_empty_bounding_boxes",
                "raw_sha256": artifact.sha256_json(
                    payload["landmarks"][landmark_index]),
            })
    sanitized["landmarks"] = kept_landmarks
    return sanitized, events


def _validate_with_sanitation(
        key: str, response: Mapping[str, Any]) -> tuple[dict[str, Any], list]:
    try:
        return extract_landmarks.validate_response(key, response), []
    except Exception as original_error:
        payload, detached_response = _response_payload(response)
        sanitized, events = _sanitize_payload(payload)
        if not events:
            raise AdoptionError(str(original_error)) from original_error
        text = artifact.canonical_json_bytes(sanitized).decode("utf-8")
        detached_response["candidates"][0]["content"]["parts"][0]["text"] = text
        try:
            canonical = extract_landmarks.validate_response(
                key, detached_response)
        except Exception as error:
            raise AdoptionError(
                "response remains invalid after bbox-only sanitation: "
                f"{error}") from error
        return canonical, events


def _status_is_empty(value: Any) -> bool:
    return (value is None or value == ""
            or isinstance(value, (dict, list)) and not value)


def _read_result_source(
        source: LegacyResultSource,
        expected: Mapping[str, Mapping[str, Any]],
        request_snapshots: Mapping[
            tuple[str, str], Sequence[_RequestSnapshot]],
        ) -> tuple[list[_ResultPrecursor], dict[str, Any]]:
    if source.result_format not in RESULT_FORMATS:
        raise AdoptionError(
            f"result source {source.source_id!r} has unknown format "
            f"{source.result_format!r}")
    path = _regular_file(source.path, f"result source {source.source_id}")
    digest = hashlib.sha256()
    raw_records = []
    size = 0
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            digest.update(raw_line)
            size += len(raw_line)
            if not raw_line.strip():
                raise AdoptionError(f"{path}:{line_number}: blank result")
            value = _strict_json_bytes(raw_line, f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise AdoptionError(
                    f"{path}:{line_number}: result must be an object")
            is_batch = source.result_format == RESULT_FORMAT_VERTEX_BATCH
            expected_shape = (_VERTEX_BATCH_KEYS if is_batch
                              else _ONLINE_RETRY_KEYS)
            _exact_keys(value, expected_shape,
                        f"{path}:{line_number} result")
            key = _nonempty_string(value["key"], f"{path}:{line_number} key")
            if key not in expected:
                raise AdoptionError(
                    f"{path}:{line_number}: unknown result key {key!r}")
            request = value["request"]
            if not isinstance(request, dict):
                raise AdoptionError(
                    f"{path}:{line_number}: echoed request must be an object")
            raw_echo_digest = artifact.sha256_json(request)
            normalized_echo, echo_events = _normalize_provider_echo(
                request, source.result_format)
            normalized_echo_digest = artifact.sha256_json(normalized_echo)
            required_role = (REQUEST_ROLE_PRIMARY if is_batch
                             else REQUEST_ROLE_RETRY)
            matches = tuple(
                snapshot
                for snapshot in request_snapshots.get((required_role, key), ())
                if snapshot.raw_request_sha256 == normalized_echo_digest)
            if not matches:
                raise AdoptionError(
                    f"{path}:{line_number}: provider echo for {key!r} does "
                    f"not bind a preserved {required_role} raw request "
                    "snapshot")
            current_digest = expected[key]["request_sha256"]
            if any(snapshot.normalized_request_sha256 != current_digest
                   for snapshot in matches):
                raise AdoptionError(
                    f"{path}:{line_number}: bound preserved request for "
                    f"{key!r} conflicts with the current request set")
            contextual_events = [{
                "scope": "provider_echo",
                "source_id": source.source_id,
                "source_path": str(path),
                "result_format": source.result_format,
                "line_number": line_number,
                "key": key,
                **event,
            } for event in echo_events]

            response = value["response"]
            error = None
            outcome = "response"
            if is_batch:
                if not isinstance(value["processed_time"], str):
                    raise AdoptionError(
                        f"{path}:{line_number}: processed_time must be a string")
                transport_metadata = {
                    "processed_time": value["processed_time"],
                    "status": value["status"],
                }
                if not _status_is_empty(value["status"]):
                    if isinstance(response, dict):
                        try:
                            _validate_with_sanitation(key, response)
                        except Exception:
                            pass
                        else:
                            raise AdoptionError(
                                f"{path}:{line_number}: nonempty status "
                                "conflicts with a valid response")
                    error = {
                        "legacy_status": value["status"],
                        "raw_response_sha256": artifact.sha256_json(response),
                    }
                    response = None
                    outcome = "transport_error"
                elif not isinstance(response, dict):
                    error = {
                        "classification": "non_object_response",
                        "raw_response_sha256": artifact.sha256_json(response),
                    }
                    response = None
                    outcome = "invalid_transport_response"
            else:
                raw_error = value["error"]
                transport_metadata = {"error": raw_error}
                if raw_error is not None:
                    if response is not None:
                        raise AdoptionError(
                            f"{path}:{line_number}: retry result has both "
                            "response and error")
                    error = {"legacy_error": raw_error}
                    outcome = "transport_error"
                elif not isinstance(response, dict):
                    raise AdoptionError(
                        f"{path}:{line_number}: successful retry response "
                        "must be an object")
            transport_fields = tuple(transport_metadata)
            transport_digest = artifact.sha256_json(transport_metadata)
            contextual_events.append({
                "scope": "result_wrapper",
                "source_id": source.source_id,
                "source_path": str(path),
                "result_format": source.result_format,
                "line_number": line_number,
                "key": key,
                "action": "preserve_transport_metadata_digest",
                "paths": [f"/{field}" for field in transport_fields],
                "sha256": transport_digest,
            })
            raw_records.append({
                "line_number": line_number,
                "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
                "key": key,
                "raw_echoed_request_sha256": raw_echo_digest,
                "normalized_echoed_request_sha256": normalized_echo_digest,
                "bound_request_role": required_role,
                "bound_request_raw_sha256": normalized_echo_digest,
                "bound_request_normalized_sha256": current_digest,
                "current_request_sha256": current_digest,
                "bound_request_snapshots": tuple(
                    _snapshot_ref(snapshot) for snapshot in matches),
                "normalization_events": tuple(contextual_events),
                "transport_metadata_sha256": transport_digest,
                "transport_metadata_fields": transport_fields,
                "response": response,
                "error": error,
                "outcome": outcome,
            })
    if not raw_records:
        raise AdoptionError(f"result source {path} is empty")
    source_sha256 = digest.hexdigest()
    precursors = [_ResultPrecursor(
        source_id=source.source_id,
        source_path=str(path),
        source_sha256=source_sha256,
        result_format=source.result_format,
        line_number=record["line_number"],
        raw_line_sha256=record["raw_line_sha256"],
        key=record["key"],
        raw_echoed_request_sha256=record["raw_echoed_request_sha256"],
        normalized_echoed_request_sha256=record[
            "normalized_echoed_request_sha256"],
        bound_request_role=record["bound_request_role"],
        bound_request_raw_sha256=record["bound_request_raw_sha256"],
        bound_request_normalized_sha256=record[
            "bound_request_normalized_sha256"],
        current_request_sha256=record["current_request_sha256"],
        bound_request_snapshots=record["bound_request_snapshots"],
        normalization_events=record["normalization_events"],
        transport_metadata_sha256=record["transport_metadata_sha256"],
        transport_metadata_fields=record["transport_metadata_fields"],
        response=record["response"],
        error=record["error"],
        outcome=record["outcome"],
    ) for record in raw_records]
    report = {
        "id": source.source_id,
        "path": str(path),
        "format": source.result_format,
        "sha256": source_sha256,
        "size": size,
        "n_records": len(raw_records),
        "provider_echo_match": "complete_role_specific_two_stage",
    }
    return precursors, report


def _attempt_id(precursor: _ResultPrecursor) -> str:
    identity = {
        "source_id": precursor.source_id,
        "source_sha256": precursor.source_sha256,
        "line_number": precursor.line_number,
        "raw_line_sha256": precursor.raw_line_sha256,
    }
    return f"legacy-{artifact.sha256_json(identity)[:32]}"


def verify_adoption(
        *, dataset: str, request_set: llm_lifecycle.RequestSet,
        pinhole_dir: Path | str,
        request_sources: Sequence[LegacyRequestSource],
        result_sources: Sequence[LegacyResultSource],
        empty_error_sidecars: Sequence[EmptyErrorSidecar] = (),
        spec_sha256: str | None = None,
        ) -> AdoptionPlan:
    """Prove exact legacy lineage and return a zero-provider-call plan.

    No files are created.  The returned current ``Attempt`` and
    ``CanonicalResult`` values are in-memory publication inputs for a separate,
    explicitly authorized transactional publisher.
    """
    dataset = artifact.require_identifier(dataset, "dataset")
    all_sources: Sequence[Any] = (
        *request_sources, *result_sources, *empty_error_sidecars)
    source_ids = [source.source_id for source in all_sources]
    if len(source_ids) != len(set(source_ids)):
        raise AdoptionError("every legacy source id must be globally unique")
    source_paths = [str(Path(source.path).absolute()) for source in all_sources]
    if len(source_paths) != len(set(source_paths)):
        raise AdoptionError("every legacy source path must be listed exactly once")
    if not request_sources or not any(
            source.role == REQUEST_ROLE_PRIMARY for source in request_sources):
        raise AdoptionError("at least one primary request source is required")
    if not result_sources:
        raise AdoptionError("at least one result source is required")

    pinhole_ref, expected = _request_contract(dataset, request_set)
    request_set_value = request_set.to_dict()
    pinhole_report = _pinhole_inventory(
        Path(pinhole_dir), expected,
        request_set_value["media_settings"]["pinhole_resolution"],
        pinhole_ref)
    request_reports, request_snapshots, normalization_ledger = \
        _request_sources(request_sources, expected)
    sidecar_reports = _empty_sidecars(empty_error_sidecars)

    precursors = []
    result_reports = []
    for source in result_sources:
        source_precursors, source_report = _read_result_source(
            source, expected, request_snapshots)
        precursors.extend(source_precursors)
        normalization_ledger.extend(
            event for precursor in source_precursors
            for event in precursor.normalization_events)
        result_reports.append(source_report)

    attempts = []
    attempt_reports = []
    valid_by_key: dict[str, list[llm_lifecycle.CanonicalResult]] = {
        key: [] for key in expected}
    failures_by_key: dict[str, list[str]] = {key: [] for key in expected}
    sanitation = []
    seen_attempt_ids = set()
    for precursor in precursors:
        attempt_id = _attempt_id(precursor)
        if attempt_id in seen_attempt_ids:
            raise AdoptionError(
                f"legacy source inventory creates duplicate attempt id "
                f"{attempt_id!r}")
        seen_attempt_ids.add(attempt_id)
        metadata = {
            "legacy_source": {
                "schema": ATTEMPT_PROVENANCE_SCHEMA,
                "source_id": precursor.source_id,
                "source_path": precursor.source_path,
                "source_sha256": precursor.source_sha256,
                "result_format": precursor.result_format,
                "line_number": precursor.line_number,
                "raw_line_sha256": precursor.raw_line_sha256,
                "raw_echoed_request_sha256":
                    precursor.raw_echoed_request_sha256,
                "normalized_echoed_request_sha256":
                    precursor.normalized_echoed_request_sha256,
                "echo_binding": {
                    "request_role": precursor.bound_request_role,
                    "bound_raw_request_sha256":
                        precursor.bound_request_raw_sha256,
                    "bound_normalized_request_sha256":
                        precursor.bound_request_normalized_sha256,
                    "current_request_sha256":
                        precursor.current_request_sha256,
                    "snapshots": list(precursor.bound_request_snapshots),
                },
                "transport_metadata": {
                    "fields": list(precursor.transport_metadata_fields),
                    "sha256": precursor.transport_metadata_sha256,
                },
            },
        }
        attempt = llm_lifecycle.Attempt(
            request_set_fingerprint=request_set.fingerprint,
            key=precursor.key,
            attempt_id=attempt_id,
            response=precursor.response,
            error=precursor.error,
            metadata=metadata,
        )
        attempts.append(attempt)
        attempt_report = {
            "attempt_id": attempt_id,
            "key": precursor.key,
            "source_id": precursor.source_id,
            "line_number": precursor.line_number,
            "raw_line_sha256": precursor.raw_line_sha256,
            "raw_echoed_request_sha256":
                precursor.raw_echoed_request_sha256,
            "normalized_echoed_request_sha256":
                precursor.normalized_echoed_request_sha256,
            "bound_request_role": precursor.bound_request_role,
            "bound_request_raw_sha256":
                precursor.bound_request_raw_sha256,
            "bound_request_normalized_sha256":
                precursor.bound_request_normalized_sha256,
            "current_request_sha256": precursor.current_request_sha256,
            "transport_metadata_sha256":
                precursor.transport_metadata_sha256,
            "outcome": precursor.outcome,
        }
        if precursor.error is not None:
            failures_by_key[precursor.key].append(
                f"{attempt_id}: {precursor.outcome}")
            attempt_reports.append(attempt_report)
            continue
        assert precursor.response is not None
        try:
            result, events = _validate_with_sanitation(
                precursor.key, precursor.response)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            failures_by_key[precursor.key].append(
                f"{attempt_id}: {message}")
            attempt_report["outcome"] = "invalid_response"
            attempt_report["validation_error"] = message
            attempt_reports.append(attempt_report)
            continue
        canonical = llm_lifecycle.CanonicalResult(
            key=precursor.key, attempt_id=attempt_id, result=result)
        valid_by_key[precursor.key].append(canonical)
        attempt_report["outcome"] = (
            "valid_sanitized_response" if events else "valid_response")
        attempt_report["canonical_result_sha256"] = artifact.sha256_json(result)
        attempt_report["n_sanitation_events"] = len(events)
        attempt_reports.append(attempt_report)
        for event in events:
            sanitation.append({
                "key": precursor.key,
                "attempt_id": attempt_id,
                "source_id": precursor.source_id,
                **event,
            })

    problems = []
    selected = []
    for key in expected:
        valid = valid_by_key[key]
        if not valid:
            details = "; ".join(failures_by_key[key]) or "no attempt"
            problems.append(f"{key}: no valid response ({details})")
        elif len(valid) > 1:
            problems.append(
                f"{key}: duplicate valid responses "
                f"{[item.attempt_id for item in valid]}")
        else:
            selected.append(valid[0])
    if problems:
        raise AdoptionError(
            "legacy extraction does not have complete, unique successful "
            "coverage: " + " | ".join(problems))

    canonical_results = tuple(selected)
    canonical_bytes = llm_lifecycle.canonical_results_bytes(
        request_set, canonical_results)
    prediction_bytes = extract_landmarks.predictions_bytes(
        request_set, canonical_results)
    report = {
        "schema": REPORT_SCHEMA,
        "dataset": dataset,
        "status": "ready_for_explicit_publication",
        "provider_calls": 0,
        "spec_sha256": spec_sha256,
        "request_set": {
            "fingerprint": request_set.fingerprint,
            "model": request_set.model,
            "prompt_sha256": hashlib.sha256(
                request_set.system_prompt.encode("utf-8")).hexdigest(),
            "response_schema_sha256": artifact.sha256_json(
                request_set_value["response_schema"]),
            "n_expected": len(expected),
            "coverage": "complete",
        },
        "pinhole_images": pinhole_report,
        "request_sources": request_reports,
        "result_sources": result_reports,
        "empty_error_sidecars": sidecar_reports,
        "attempts": attempt_reports,
        "normalization_ledger": normalization_ledger,
        "attempt_summary": {
            "n_total": len(attempts),
            "n_valid": sum(1 for record in attempt_reports
                           if record["outcome"] in (
                               "valid_response", "valid_sanitized_response")),
            "n_failed_or_invalid": sum(1 for record in attempt_reports
                                       if record["outcome"] not in (
                                           "valid_response",
                                           "valid_sanitized_response")),
            "raw_provenance": "complete_by_source_and_line_digest",
        },
        "sanitation_ledger": sanitation,
        "canonical_outputs": {
            "n_results": len(canonical_results),
            "canonical_results_sha256": hashlib.sha256(
                canonical_bytes).hexdigest(),
            "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "coverage": "complete_unique_in_request_order",
        },
        "publication_plan": {
            "mode": "transactional_typed_adoption",
            "requires_explicit_write_authorization": True,
            "provider_calls": 0,
            "normal_reader_compatibility_fallback": False,
            "raw_source_policy": "retain_verbatim_by_reported_sha256",
            "artifacts": [
                {
                    "kind": paths_lib.PINHOLE_IMAGES,
                    "action": "adopt_verified_exact_face_bytes",
                    "content_digest": pinhole_ref.content_digest,
                },
                {
                    "kind": llm_lifecycle.REQUEST_ARTIFACT_KIND,
                    "action": "publish_exact_current_request_set",
                    "request_set_fingerprint": request_set.fingerprint,
                },
                {
                    "kind": llm_lifecycle.RESULT_ARTIFACT_KIND,
                    "action": "publish_complete_unique_canonical_results",
                    "content_sha256": hashlib.sha256(
                        canonical_bytes).hexdigest(),
                },
                {
                    "kind": paths_lib.FRAME_LANDMARKS,
                    "action": "publish_sanitized_current_predictions",
                    "content_sha256": hashlib.sha256(
                        prediction_bytes).hexdigest(),
                },
            ],
        },
    }
    return AdoptionPlan(
        attempts=tuple(attempts),
        canonical_results=canonical_results,
        canonical_results_bytes=canonical_bytes,
        predictions_bytes=prediction_bytes,
        report=report,
    )


def verify_spec(spec: AdoptionSpec) -> AdoptionPlan:
    request_set = llm_lifecycle.load_request_set(spec.request_set_path)
    return verify_adoption(
        dataset=spec.dataset,
        request_set=request_set,
        pinhole_dir=spec.pinhole_dir,
        request_sources=spec.request_sources,
        result_sources=spec.result_sources,
        empty_error_sidecars=spec.empty_error_sidecars,
        spec_sha256=spec.spec_sha256,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify paid legacy landmark extraction and print a zero-call "
            "typed-publication plan. No artifact is written."))
    parser.add_argument(
        "--spec", required=True, type=Path,
        help="Strict JSON source inventory (relative paths resolve beside it)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        plan = verify_spec(load_spec(args.spec))
    except (AdoptionError, artifact.ArtifactError,
            llm_lifecycle.LlmLifecycleError, OSError) as error:
        print(f"legacy extraction adoption rejected: {error}", file=sys.stderr)
        return 1
    print(json.dumps(plan.report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
