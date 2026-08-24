"""Strict, reusable lifecycle for LLM-backed farfield stages.

The request set is an immutable, content-addressed snapshot.  Every transport
attempt is retained as a separate atomically created immutable shard, while
downstream stages consume a separate canonical result artifact compiled from
exactly one valid response for every expected request unit.  Partial coverage
is never publishable.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from experimental.overhead_matching.swag.farfield import artifact


REQUEST_SET_SCHEMA = "farfield.llm_request_set/v1"
ATTEMPT_SCHEMA = "farfield.llm_attempt/v1"
CANONICAL_RESULTS_SCHEMA = "farfield.llm_canonical_results/v1"
REQUEST_ARTIFACT_KIND = "llm_requests"
RESULT_ARTIFACT_KIND = "llm_results"
REQUEST_SET_NAME = "request_set.json"
REQUESTS_NAME = "requests.jsonl"
CANONICAL_RESULTS_NAME = "canonical_results.jsonl"
ATTEMPTS_DIR_NAME = "attempts"

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_REQUEST_SET_KEYS = frozenset({
    "schema", "stage", "model", "system_prompt", "response_schema",
    "media_settings", "input_digests", "upstreams", "units", "fingerprint",
})
_UNIT_KEYS = frozenset({"key", "request", "metadata"})
_ATTEMPT_KEYS = frozenset({
    "schema", "request_set_fingerprint", "key", "attempt_id", "response",
    "error", "metadata",
})
_ATTEMPT_SHARD_RE = re.compile(r"attempt-[0-9a-f]{64}\.json\Z")
_CANONICAL_KEYS = frozenset({
    "schema", "request_set_fingerprint", "key", "attempt_id", "result",
})


class LlmLifecycleError(ValueError):
    """An LLM lifecycle artifact is malformed or incomplete."""


class IncompleteCoverageError(LlmLifecycleError):
    """Canonical publication cannot cover every expected request exactly once."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                what: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise LlmLifecycleError(f"invalid {what}: " + ", ".join(details))


def _json_clone(value: Any, what: str) -> Any:
    """Validate a finite JSON value and detach it from caller-owned objects."""
    try:
        encoded = artifact.canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, artifact.ArtifactError) as error:
        raise LlmLifecycleError(f"{what} must be finite JSON: {error}") from error


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nonempty_string(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise LlmLifecycleError(f"{what} must be a non-empty string")
    return value


def _digest(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise LlmLifecycleError(f"{what} must be a lowercase SHA-256 digest")
    return value


def _ref_identity(ref: artifact.ArtifactRef) -> dict[str, str]:
    """Identity fields only: an informational path must not affect a hash."""
    return {
        "kind": ref.kind,
        "dataset": ref.dataset,
        "version": ref.version,
        "manifest_digest": ref.manifest_digest,
        "content_digest": ref.content_digest,
    }


@dataclass(frozen=True)
class RequestUnit:
    """One stable key, exact transport request, and bound stage context."""

    key: str
    request: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        _nonempty_string(self.key, "request unit key")
        cloned = _json_clone(self.request, f"request unit {self.key!r}")
        metadata = _json_clone(self.metadata,
                               f"request unit {self.key!r} metadata")
        if not isinstance(cloned, dict):
            raise LlmLifecycleError("request unit request must be a JSON object")
        if not isinstance(metadata, dict):
            raise LlmLifecycleError("request unit metadata must be a JSON object")
        object.__setattr__(self, "request", _freeze(cloned))
        object.__setattr__(self, "metadata", _freeze(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "request": _thaw(self.request),
                "metadata": _thaw(self.metadata)}

    @classmethod
    def from_dict(cls, value: Any) -> "RequestUnit":
        if not isinstance(value, dict):
            raise LlmLifecycleError("request unit must be a JSON object")
        _exact_keys(value, _UNIT_KEYS, "request unit")
        return cls(key=value["key"], request=value["request"],
                   metadata=value["metadata"])


@dataclass(frozen=True)
class RequestSet:
    """Complete immutable identity of an ordered LLM workload."""

    stage: str
    model: str
    system_prompt: str
    response_schema: Mapping[str, Any]
    media_settings: Mapping[str, Any]
    input_digests: Mapping[str, str]
    upstreams: tuple[artifact.ArtifactRef, ...]
    units: tuple[RequestUnit, ...]
    fingerprint: str
    schema: str = REQUEST_SET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REQUEST_SET_SCHEMA:
            raise LlmLifecycleError(
                f"unsupported request-set schema {self.schema!r}")
        _nonempty_string(self.stage, "request-set stage")
        _nonempty_string(self.model, "request-set model")
        _nonempty_string(self.system_prompt, "request-set system prompt")
        response_schema = _json_clone(self.response_schema, "response schema")
        media_settings = _json_clone(self.media_settings, "media settings")
        input_digests = _json_clone(self.input_digests, "input digests")
        if not isinstance(response_schema, dict):
            raise LlmLifecycleError("response schema must be a JSON object")
        if not isinstance(media_settings, dict):
            raise LlmLifecycleError("media settings must be a JSON object")
        if not isinstance(input_digests, dict):
            raise LlmLifecycleError("input digests must be a JSON object")
        for name, digest in input_digests.items():
            _nonempty_string(name, "input digest name")
            _digest(digest, f"input digest {name!r}")
        if not isinstance(self.upstreams, tuple) or not all(
                isinstance(ref, artifact.ArtifactRef) for ref in self.upstreams):
            raise LlmLifecycleError("upstreams must be a tuple of ArtifactRef")
        if not input_digests and not self.upstreams:
            raise LlmLifecycleError(
                "request set must bind at least one upstream artifact or "
                "explicit input content digest")
        identities = [artifact.sha256_json(_ref_identity(ref))
                      for ref in self.upstreams]
        if len(identities) != len(set(identities)):
            raise LlmLifecycleError("upstream artifact identities must be unique")
        if not isinstance(self.units, tuple) or not self.units:
            raise LlmLifecycleError("request set must contain at least one unit")
        if not all(isinstance(unit, RequestUnit) for unit in self.units):
            raise LlmLifecycleError("request-set units must be RequestUnit values")
        keys = [unit.key for unit in self.units]
        if len(keys) != len(set(keys)):
            raise LlmLifecycleError("request unit keys must be unique")
        object.__setattr__(self, "response_schema", _freeze(response_schema))
        object.__setattr__(self, "media_settings", _freeze(media_settings))
        object.__setattr__(self, "input_digests", _freeze(input_digests))
        expected = artifact.sha256_json(self._identity_dict())
        if self.fingerprint != expected:
            raise LlmLifecycleError(
                "request-set fingerprint mismatch: "
                f"expected {expected}, found {self.fingerprint}")

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "response_schema": _thaw(self.response_schema),
            "media_settings": _thaw(self.media_settings),
            "input_digests": _thaw(self.input_digests),
            "upstreams": [_ref_identity(ref) for ref in self.upstreams],
            "units": [unit.to_dict() for unit in self.units],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._identity_dict()
        value["upstreams"] = [ref.to_dict() for ref in self.upstreams]
        value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def create(cls, *, stage: str, model: str, system_prompt: str,
               response_schema: Mapping[str, Any],
               media_settings: Mapping[str, Any],
               input_digests: Mapping[str, str],
               upstreams: Iterable[artifact.ArtifactRef],
               units: Iterable[RequestUnit]) -> "RequestSet":
        upstream_tuple = tuple(upstreams)
        unit_tuple = tuple(units)
        identity = {
            "schema": REQUEST_SET_SCHEMA,
            "stage": stage,
            "model": model,
            "system_prompt": system_prompt,
            "response_schema": _json_clone(response_schema, "response schema"),
            "media_settings": _json_clone(media_settings, "media settings"),
            "input_digests": _json_clone(input_digests, "input digests"),
            "upstreams": [_ref_identity(ref) for ref in upstream_tuple],
            "units": [unit.to_dict() for unit in unit_tuple],
        }
        return cls(
            stage=stage,
            model=model,
            system_prompt=system_prompt,
            response_schema=response_schema,
            media_settings=media_settings,
            input_digests=input_digests,
            upstreams=upstream_tuple,
            units=unit_tuple,
            fingerprint=artifact.sha256_json(identity),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "RequestSet":
        if not isinstance(value, dict):
            raise LlmLifecycleError("request set must be a JSON object")
        _exact_keys(value, _REQUEST_SET_KEYS, "request set")
        upstreams = value["upstreams"]
        units = value["units"]
        if not isinstance(upstreams, list) or not isinstance(units, list):
            raise LlmLifecycleError("request-set upstreams and units must be lists")
        return cls(
            schema=value["schema"],
            stage=value["stage"],
            model=value["model"],
            system_prompt=value["system_prompt"],
            response_schema=value["response_schema"],
            media_settings=value["media_settings"],
            input_digests=value["input_digests"],
            upstreams=tuple(artifact.ArtifactRef.from_dict(item)
                            for item in upstreams),
            units=tuple(RequestUnit.from_dict(item) for item in units),
            fingerprint=value["fingerprint"],
        )


def load_request_set(path: Path | str) -> RequestSet:
    """Load a strict request snapshot, rejecting duplicate object keys."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_reject_duplicate_keys,
                              parse_constant=_reject_constant)
    except LlmLifecycleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LlmLifecycleError(f"invalid request set {path}: {error}") from error
    return RequestSet.from_dict(value)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = {}
    for key, value in pairs:
        if key in out:
            raise LlmLifecycleError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise LlmLifecycleError(f"non-finite JSON constant {value!r}")


def publish_request_set(destination: Path | str, *, request_set: RequestSet,
                        dataset: str, version: str, generator: str,
                        git_commit: str, arguments: Iterable[str] = (),
                        extra_config: Mapping[str, Any] | None = None,
                        ) -> artifact.ArtifactRef:
    """Transactionally publish a request snapshot and transport JSONL."""
    config = dict(extra_config or {})
    config["request_set_fingerprint"] = request_set.fingerprint
    with artifact.ArtifactDirectoryBuilder(
            destination,
            kind=REQUEST_ARTIFACT_KIND,
            dataset=dataset,
            version=version,
            generator=generator,
            git_commit=git_commit,
            arguments=arguments,
            upstreams=request_set.upstreams,
            config=config,
            declared_outputs=(REQUEST_SET_NAME, REQUESTS_NAME)) as builder:
        artifact.atomic_write_json(
            builder.output_path(REQUEST_SET_NAME), request_set.to_dict())
        artifact.atomic_write_file(
            builder.output_path(REQUESTS_NAME),
            transport_requests_bytes(request_set))
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def transport_requests_bytes(request_set: RequestSet,
                             keys: Sequence[str] | None = None) -> bytes:
    """Encode all or an exact subset of provider inputs in snapshot order."""
    if not isinstance(request_set, RequestSet):
        raise TypeError("request_set must be a RequestSet")
    units = request_set.units
    if keys is not None:
        requested = tuple(keys)
        if len(requested) != len(set(requested)):
            raise LlmLifecycleError("transport request keys must be unique")
        expected = {unit.key for unit in units}
        unknown = sorted(set(requested) - expected)
        if unknown:
            raise LlmLifecycleError(
                f"unknown transport request keys {unknown}")
        selected = set(requested)
        units = tuple(unit for unit in units if unit.key in selected)
    return b"".join(
        artifact.canonical_json_bytes({
            "key": unit.key, "request": _thaw(unit.request)}) + b"\n"
        for unit in units)


@dataclass(frozen=True)
class Attempt:
    request_set_fingerprint: str
    key: str
    attempt_id: str
    response: Mapping[str, Any] | None
    error: Any
    metadata: Mapping[str, Any]
    schema: str = ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ATTEMPT_SCHEMA:
            raise LlmLifecycleError(f"unsupported attempt schema {self.schema!r}")
        _digest(self.request_set_fingerprint, "attempt request-set fingerprint")
        _nonempty_string(self.key, "attempt key")
        _nonempty_string(self.attempt_id, "attempt id")
        if (self.response is None) == (self.error is None):
            raise LlmLifecycleError(
                "attempt must contain exactly one of response or error")
        response = _json_clone(self.response, "attempt response")
        metadata = _json_clone(self.metadata, "attempt metadata")
        error = _json_clone(self.error, "attempt error")
        if response is not None and not isinstance(response, dict):
            raise LlmLifecycleError("attempt response must be a JSON object")
        if not isinstance(metadata, dict):
            raise LlmLifecycleError("attempt metadata must be a JSON object")
        object.__setattr__(self, "response",
                           None if response is None else _freeze(response))
        object.__setattr__(self, "metadata", _freeze(metadata))
        object.__setattr__(self, "error", _freeze(error))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_set_fingerprint": self.request_set_fingerprint,
            "key": self.key,
            "attempt_id": self.attempt_id,
            "response": _thaw(self.response),
            "error": _thaw(self.error),
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "Attempt":
        if not isinstance(value, dict):
            raise LlmLifecycleError("attempt must be a JSON object")
        _exact_keys(value, _ATTEMPT_KEYS, "attempt")
        return cls(
            schema=value["schema"],
            request_set_fingerprint=value["request_set_fingerprint"],
            key=value["key"],
            attempt_id=value["attempt_id"],
            response=value["response"],
            error=value["error"],
            metadata=value["metadata"],
        )


def _attempt_shard_name(attempt_id: str) -> str:
    return f"attempt-{artifact.sha256_json(attempt_id)}.json"


def _load_attempt_shard(path: Path) -> Attempt:
    if path.is_symlink() or not path.is_file():
        raise LlmLifecycleError(f"attempt shard is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except LlmLifecycleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LlmLifecycleError(
            f"invalid attempt shard {path}: {error}") from error
    attempt = Attempt.from_dict(value)
    expected_name = _attempt_shard_name(attempt.attempt_id)
    if path.name != expected_name:
        raise LlmLifecycleError(
            f"attempt shard filename does not bind attempt_id: expected "
            f"{expected_name!r}, found {path.name!r}")
    return attempt


def load_attempts(attempts_dir: Path | str) -> tuple[Attempt, ...]:
    """Strictly load immutable attempt shards in deterministic filename order."""
    attempts_dir = Path(attempts_dir)
    if attempts_dir.is_symlink() or not attempts_dir.is_dir():
        raise LlmLifecycleError(
            f"attempt store is not a directory: {attempts_dir}")
    try:
        entries = sorted(attempts_dir.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise LlmLifecycleError(
            f"cannot scan attempt store {attempts_dir}: {error}") from error
    attempts = []
    seen_ids = set()
    for entry in entries:
        if not _ATTEMPT_SHARD_RE.fullmatch(entry.name):
            raise LlmLifecycleError(
                f"unexpected entry in attempt store {attempts_dir}: "
                f"{entry.name!r}")
        attempt = _load_attempt_shard(entry)
        if attempt.attempt_id in seen_ids:
            raise LlmLifecycleError(
                f"duplicate attempt_id {attempt.attempt_id!r} in "
                f"{attempts_dir}")
        seen_ids.add(attempt.attempt_id)
        attempts.append(attempt)
    return tuple(attempts)


def _atomic_create_attempt_shard(path: Path, data: bytes) -> None:
    """Publish one no-clobber shard; temporary bytes stay outside the store."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.parent.name}-attempt-",
        suffix=".tmp",
        dir=path.parent.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def publish_attempt(attempts_dir: Path | str, attempt: Attempt) -> Path:
    """Atomically create one immutable shard, refusing duplicate attempt ids."""
    if not isinstance(attempt, Attempt):
        raise TypeError("attempt must be an Attempt")
    attempts_dir = Path(attempts_dir)
    attempts_dir.parent.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    if attempts_dir.is_symlink() or not attempts_dir.is_dir():
        raise LlmLifecycleError(
            f"attempt store is not a directory: {attempts_dir}")
    shard_path = attempts_dir / _attempt_shard_name(attempt.attempt_id)
    try:
        _atomic_create_attempt_shard(
            shard_path,
            artifact.canonical_json_bytes(attempt.to_dict()) + b"\n",
        )
    except FileExistsError as error:
        existing = _load_attempt_shard(shard_path)
        raise LlmLifecycleError(
            f"duplicate attempt_id {existing.attempt_id!r} in "
            f"{attempts_dir}") from error
    return shard_path


def import_transport_results(transport_path: Path | str,
                             attempts_dir: Path | str,
                             request_set: RequestSet,
                             *, transport_source: str | None = None) -> int:
    """Import the provider boundary ``{key,response|error}`` into attempts.

    The provider file is mutable transport state, never an aggregation input.
    Stable ids make importing an unchanged prefix idempotent while preserving
    genuinely repeated records as distinct attempts.
    """
    transport_path = Path(transport_path)
    attempts_dir = Path(attempts_dir)
    if transport_source is None:
        transport_source = transport_path.name
    _nonempty_string(transport_source, "transport source")
    try:
        lines = transport_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise LlmLifecycleError(
            f"cannot read transport results {transport_path}: {error}") from error
    existing = load_attempts(attempts_dir) if attempts_dir.exists() else ()
    for item in existing:
        if item.request_set_fingerprint != request_set.fingerprint:
            raise LlmLifecycleError(
                f"attempt {item.attempt_id!r} targets a different request set")
    existing_ids = {item.attempt_id for item in existing}
    occurrences: dict[str, int] = {}
    imported = 0
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise LlmLifecycleError(
                f"blank transport result at {transport_path}:{line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys,
                               parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise LlmLifecycleError(
                f"invalid transport result at {transport_path}:{line_number}: "
                f"{error}") from error
        if not isinstance(value, dict):
            raise LlmLifecycleError("transport result must be a JSON object")
        has_response = frozenset(value) == frozenset({"key", "response"})
        has_error = frozenset(value) == frozenset({"key", "error"})
        if not has_response and not has_error:
            raise LlmLifecycleError(
                "transport result must have exact keys {key,response} or "
                "{key,error}")
        key = _nonempty_string(value["key"], "transport result key")
        record_digest = artifact.sha256_json({
            "request_set_fingerprint": request_set.fingerprint,
            "transport_source": transport_source,
            "record": value,
        })
        occurrence = occurrences.get(record_digest, 0)
        occurrences[record_digest] = occurrence + 1
        attempt_id = f"transport-{record_digest[:24]}-{occurrence}"
        if attempt_id in existing_ids:
            continue
        item = Attempt(
            request_set_fingerprint=request_set.fingerprint,
            key=key,
            attempt_id=attempt_id,
            response=value.get("response") if has_response else None,
            error=value.get("error") if has_error else None,
            metadata={
                "transport_source": transport_source,
                "transport_record": line_number,
            },
        )
        publish_attempt(attempts_dir, item)
        existing_ids.add(attempt_id)
        imported += 1
    return imported


@dataclass(frozen=True)
class CanonicalResult:
    key: str
    attempt_id: str
    result: Any


Validator = Callable[[str, Mapping[str, Any]], Any]


def _classify_attempts(request_set: RequestSet,
                       attempts: Sequence[Attempt], validator: Validator):
    expected = {unit.key for unit in request_set.units}
    candidates: dict[str, list[CanonicalResult]] = {
        unit.key: [] for unit in request_set.units}
    failures: dict[str, list[str]] = {unit.key: [] for unit in request_set.units}
    for item in attempts:
        if item.request_set_fingerprint != request_set.fingerprint:
            raise LlmLifecycleError(
                f"attempt {item.attempt_id!r} targets a different request set")
        if item.key not in expected:
            raise LlmLifecycleError(
                f"attempt {item.attempt_id!r} has unknown key {item.key!r}")
        if item.error is not None:
            failures[item.key].append(f"{item.attempt_id}: transport error")
            continue
        assert item.response is not None
        try:
            result = _json_clone(
                validator(item.key, _thaw(item.response)),
                f"validated result for {item.key!r}")
        except Exception as error:  # validator defines stage-specific failures
            failures[item.key].append(f"{item.attempt_id}: {error}")
            continue
        candidates[item.key].append(CanonicalResult(
            key=item.key, attempt_id=item.attempt_id, result=result))
    return candidates, failures


def pending_request_keys(request_set: RequestSet,
                         attempts: Sequence[Attempt],
                         validator: Validator) -> tuple[str, ...]:
    """Return units lacking a valid success; reject existing ambiguity."""
    candidates, _ = _classify_attempts(request_set, attempts, validator)
    duplicates = {
        key: [item.attempt_id for item in values]
        for key, values in candidates.items() if len(values) > 1
    }
    if duplicates:
        raise IncompleteCoverageError(
            f"request set has duplicate valid responses: {duplicates}")
    return tuple(unit.key for unit in request_set.units
                 if not candidates[unit.key])


def compile_canonical_results(request_set: RequestSet,
                              attempts: Sequence[Attempt],
                              validator: Validator,
                              ) -> tuple[CanonicalResult, ...]:
    """Select exactly one valid success for each expected request, in order.

    Failed or malformed historical attempts may be retried because the log is
    append-only.  A unit is publishable only when it has exactly one successful
    response accepted by ``validator``.  Multiple valid successes are
    ambiguous and therefore rejected rather than selected by file order.
    """
    candidates, failures = _classify_attempts(
        request_set, attempts, validator)

    problems = []
    for unit in request_set.units:
        valid = candidates[unit.key]
        if not valid:
            details = "; ".join(failures[unit.key]) or "no attempt"
            problems.append(f"{unit.key}: no valid response ({details})")
        elif len(valid) > 1:
            ids = [item.attempt_id for item in valid]
            problems.append(f"{unit.key}: duplicate valid responses {ids}")
    if problems:
        raise IncompleteCoverageError(
            "request set does not have complete, unique successful coverage: "
            + " | ".join(problems))
    return tuple(candidates[unit.key][0] for unit in request_set.units)


def publish_canonical_results(
        destination: Path | str, *, request_set: RequestSet,
        request_artifact: artifact.ArtifactRef,
        results: Sequence[CanonicalResult], dataset: str, version: str,
        generator: str, git_commit: str, arguments: Iterable[str] = (),
        extra_config: Mapping[str, Any] | None = None,
        ) -> artifact.ArtifactRef:
    """Transactionally publish complete canonical results in request order."""
    encoded_results = canonical_results_bytes(request_set, results)
    expected = [unit.key for unit in request_set.units]
    config = dict(extra_config or {})
    config.update({
        "request_set_fingerprint": request_set.fingerprint,
        "n_expected": len(expected),
        "n_successful": len(results),
        "coverage": "complete",
    })
    with artifact.ArtifactDirectoryBuilder(
            destination,
            kind=RESULT_ARTIFACT_KIND,
            dataset=dataset,
            version=version,
            generator=generator,
            git_commit=git_commit,
            arguments=arguments,
            upstreams=(request_artifact,),
            config=config,
            declared_outputs=(CANONICAL_RESULTS_NAME,)) as builder:
        artifact.atomic_write_file(
            builder.output_path(CANONICAL_RESULTS_NAME), encoded_results)
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def canonical_results_bytes(request_set: RequestSet,
                            results: Sequence[CanonicalResult]) -> bytes:
    """Encode complete canonical results in exact request-set order."""
    expected = [unit.key for unit in request_set.units]
    actual = [item.key for item in results]
    if actual != expected or len(actual) != len(set(actual)):
        raise IncompleteCoverageError(
            f"canonical result keys must exactly equal request order {expected}; "
            f"found {actual}")
    return b"".join(
        artifact.canonical_json_bytes({
            "schema": CANONICAL_RESULTS_SCHEMA,
            "request_set_fingerprint": request_set.fingerprint,
            "key": item.key,
            "attempt_id": item.attempt_id,
            "result": item.result,
        }) + b"\n"
        for item in results)


def load_canonical_results(path: Path | str,
                           request_set: RequestSet,
                           ) -> tuple[CanonicalResult, ...]:
    """Load canonical JSONL and require exact ordered coverage."""
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise LlmLifecycleError(f"cannot read canonical results {path}: {error}") \
            from error
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise LlmLifecycleError(
                f"blank canonical result at {path}:{line_number}")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys,
                               parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise LlmLifecycleError(
                f"invalid canonical result at {path}:{line_number}: {error}") \
                from error
        if not isinstance(value, dict):
            raise LlmLifecycleError("canonical result must be a JSON object")
        _exact_keys(value, _CANONICAL_KEYS, "canonical result")
        if value["schema"] != CANONICAL_RESULTS_SCHEMA:
            raise LlmLifecycleError(
                f"unsupported canonical result schema {value['schema']!r}")
        if value["request_set_fingerprint"] != request_set.fingerprint:
            raise LlmLifecycleError(
                "canonical result targets a different request set")
        records.append(CanonicalResult(
            key=_nonempty_string(value["key"], "canonical result key"),
            attempt_id=_nonempty_string(
                value["attempt_id"], "canonical result attempt id"),
            result=_json_clone(value["result"], "canonical result payload"),
        ))
    expected = [unit.key for unit in request_set.units]
    actual = [item.key for item in records]
    if actual != expected or len(actual) != len(set(actual)):
        raise IncompleteCoverageError(
            f"canonical results do not exactly cover request order: "
            f"expected {expected}, found {actual}")
    return tuple(records)
