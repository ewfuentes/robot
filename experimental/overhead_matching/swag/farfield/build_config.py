"""Immutable orchestration recipes for farfield artifact builds.

``build_dir`` is mutable orchestration state only.  Scientific outputs live
in versioned artifact directories and completed localization executions live
under ``runs/``.  This module records the requested recipe once so every
stage resolves the same values; each published artifact independently records
the subset it actually consumed in its manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experimental.overhead_matching.swag.farfield import artifact, provenance


BUILD_CONFIG_NAME = "build_config.json"
SCHEMA = "farfield_build_config/v1"

_DOCUMENT_KEYS = frozenset({
    "schema", "dataset", "generator", "git_commit", "created", "inputs",
    "config", "notes", "build_identity",
})
_MISSING = object()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON constant {value!r}")


class MissingConfigValue(ValueError):
    """A required result-shaping value is absent from the build recipe."""


class InvalidConfigValue(ValueError):
    """A resolved value has the wrong type or lies outside its valid domain."""


@dataclass(frozen=True)
class ValueSpec:
    """Type and scalar-domain contract for one resolved config leaf.

    Types are checked exactly, so ``True`` is never accepted as the integer
    ``1``.  ``minimum`` and ``maximum`` are inclusive; their
    ``exclusive_*`` counterparts express strict bounds.  ``None`` is
    accepted only when the schema explicitly opts into it.
    """

    types: tuple[type, ...]
    minimum: float | None = None
    exclusive_minimum: float | None = None
    maximum: float | None = None
    exclusive_maximum: float | None = None
    choices: tuple[Any, ...] | None = None
    allow_none: bool = False
    nonempty: bool = False

    def __post_init__(self) -> None:
        if not self.types or not all(isinstance(kind, type)
                                     for kind in self.types):
            raise TypeError("ValueSpec.types must be a non-empty tuple of types")
        if self.minimum is not None and self.exclusive_minimum is not None:
            raise ValueError(
                "ValueSpec cannot combine minimum and exclusive_minimum")
        if self.maximum is not None and self.exclusive_maximum is not None:
            raise ValueError(
                "ValueSpec cannot combine maximum and exclusive_maximum")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError("ValueSpec minimum exceeds maximum")
        if (self.exclusive_minimum is not None and self.maximum is not None
                and self.exclusive_minimum >= self.maximum):
            raise ValueError("ValueSpec exclusive_minimum must be below maximum")

        if (self.minimum is not None
                and self.exclusive_maximum is not None
                and self.minimum >= self.exclusive_maximum):
            raise ValueError("ValueSpec minimum must be below exclusive_maximum")
        if (self.exclusive_minimum is not None
                and self.exclusive_maximum is not None
                and self.exclusive_minimum >= self.exclusive_maximum):
            raise ValueError("ValueSpec exclusive bounds are empty")

    def validate(self, path: str, value: Any) -> None:
        if value is None and self.allow_none:
            return
        if type(value) not in self.types:
            expected = "/".join(kind.__name__ for kind in self.types)
            raise InvalidConfigValue(
                f"{path} must be {expected}, found {type(value).__name__}")
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise InvalidConfigValue(f"{path} must be finite")
            if self.minimum is not None and value < self.minimum:
                raise InvalidConfigValue(
                    f"{path} must be >= {self.minimum}, found {value}")
            if (self.exclusive_minimum is not None
                    and value <= self.exclusive_minimum):
                raise InvalidConfigValue(
                    f"{path} must be > {self.exclusive_minimum}, found {value}")
            if self.maximum is not None and value > self.maximum:
                raise InvalidConfigValue(
                    f"{path} must be <= {self.maximum}, found {value}")
            if (self.exclusive_maximum is not None
                    and value >= self.exclusive_maximum):
                raise InvalidConfigValue(
                    f"{path} must be < {self.exclusive_maximum}, found {value}")
        if self.nonempty and isinstance(value, str) and not value.strip():
            raise InvalidConfigValue(f"{path} must be a non-empty string")
        if self.choices is not None and value not in self.choices:
            raise InvalidConfigValue(
                f"{path} must be one of {self.choices!r}, found {value!r}")


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict):
        return {prefix}
    if not value:
        return {prefix}
    leaves = set()
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise InvalidConfigValue("config keys must be non-empty strings")
        path = f"{prefix}.{key}" if prefix else key
        leaves.update(_leaf_paths(child, path))
    return leaves


def validate_resolved(config: dict,
                      schema: Mapping[str, ValueSpec],
                      optional: Mapping[str, ValueSpec] | None = None) -> None:
    """Validate an exact, fully resolved config tree.

    Every `schema` leaf must be present and every config leaf must be
    described. Rejecting unknown values matters: an ignored typo in a
    result-shaping key is indistinguishable from silently taking a default.

    `optional` is for keys where that reasoning does not apply -- presentation
    settings, which shape a rendered page and cannot change a scientific
    result. They are validated when present and rejected when misspelled,
    like everything else; they are simply not *required*, because demanding
    them would make every build recipe recorded before the key existed
    unreadable. A result-shaping key must never go here: absence there is
    exactly the silent default this function exists to prevent.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    optional = dict(optional or {})
    overlap = sorted(set(schema) & set(optional))
    if overlap:
        raise InvalidConfigValue(
            f"keys cannot be both required and optional: {overlap}")
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    expected = set(schema)
    actual = _leaf_paths(config)
    missing = sorted(path for path in expected
                     if _get(config, path, _MISSING) is _MISSING)
    unknown = sorted(actual - expected - set(optional))
    if missing:
        raise MissingConfigValue(
            "build config is missing required values (no defaults are "
            "supplied):\n" + "\n".join(f"  {path}" for path in missing))
    if unknown:
        raise InvalidConfigValue(
            "build config contains unknown values:\n" +
            "\n".join(f"  {path}" for path in unknown))
    for path, spec in schema.items():
        spec.validate(path, _get(config, path, _MISSING))
    for path, spec in optional.items():
        value = _get(config, path, _MISSING)
        if value is not _MISSING:
            spec.validate(path, value)
    try:
        _canonical_bytes(config)
    except (TypeError, ValueError) as exc:
        raise InvalidConfigValue(f"config must contain only finite JSON: {exc}") \
            from exc


def _identity(dataset: str, config: dict, inputs: dict) -> str:
    return hashlib.sha256(_canonical_bytes({
        "schema": SCHEMA,
        "dataset": dataset,
        "config": config,
        "inputs": inputs,
    })).hexdigest()


def create(build_dir: Path, *, dataset: str, config: dict,
           required: tuple = (), schema: Mapping[str, ValueSpec] | None = None,
           optional: Mapping[str, ValueSpec] | None = None,
           generator: str, inputs: dict, notes: str = "") -> Path:
    """Validate and immutably record a new build recipe.

    The target must not already contain data. This prevents stage products
    from being attributed to a different configuration.
    """
    build_dir = Path(build_dir)
    path = build_dir / BUILD_CONFIG_NAME
    if build_dir.is_symlink():
        raise FileExistsError(
            f"{build_dir} is a symlink; refusing to publish an immutable "
            "build configuration through it")
    if build_dir.exists() and any(build_dir.iterdir()):
        raise FileExistsError(
            f"{build_dir} is not empty; refusing to attach a new build "
            "configuration to existing state")
    artifact.require_identifier(dataset, "build config dataset")
    if not isinstance(config, dict):
        raise TypeError("config must be a JSON object")
    if schema is not None:
        validate_resolved(config, schema, optional=optional)
    missing = [key for key in required
               if _get(config, key, _MISSING) is _MISSING]
    if missing:
        raise MissingConfigValue(
            "build config is missing required values (no defaults are "
            "supplied):\n" + "\n".join(f"  {key}" for key in missing))
    if not isinstance(generator, str) or not generator.strip():
        raise ValueError("generator must be a non-empty string")
    if not isinstance(inputs, dict) or not all(
            isinstance(key, str) and key for key in inputs):
        raise TypeError("inputs must be an object with non-empty string keys")
    if not isinstance(notes, str):
        raise TypeError("notes must be a string")
    normalized_inputs = {key: str(value) for key, value in inputs.items()}
    document = {
        "schema": SCHEMA,
        "dataset": dataset,
        "generator": generator,
        "git_commit": provenance.git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": normalized_inputs,
        "config": config,
        "notes": notes,
        "build_identity": _identity(dataset, config, normalized_inputs),
    }
    artifact.atomic_create_json(path, document)
    return path


def load(build_dir: Path) -> dict:
    """Load and verify an immutable build recipe."""
    build_dir = Path(build_dir)
    path = build_dir / BUILD_CONFIG_NAME
    if build_dir.is_symlink():
        raise ValueError(
            f"invalid build config {path}: build directory is a symlink")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"{path} is not a regular, non-symlink file; this is not a "
            "farfield build directory")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid build config {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"invalid build config {path}: root must be an object")
    missing = sorted(_DOCUMENT_KEYS - set(document))
    unknown = sorted(set(document) - _DOCUMENT_KEYS)
    if missing or unknown:
        raise ValueError(
            f"invalid build config {path}: missing={missing}, unknown={unknown}")
    if document.get("schema") != SCHEMA:
        raise ValueError(
            f"invalid build config {path}: schema must be {SCHEMA!r}")
    if not isinstance(document.get("config"), dict):
        raise ValueError(f"invalid build config {path}: config must be an object")
    if not isinstance(document.get("inputs"), dict):
        raise ValueError(f"invalid build config {path}: inputs must be an object")
    for field in ("dataset", "generator", "git_commit", "created"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(
                f"invalid build config {path}: {field} must be non-empty")
    try:
        artifact.require_identifier(document["dataset"], "build config dataset")
    except artifact.ArtifactValidationError as exc:
        raise ValueError(f"invalid build config {path}: {exc}") from exc
    if not all(isinstance(key, str) and isinstance(value, str)
               for key, value in document["inputs"].items()):
        raise ValueError(
            f"invalid build config {path}: inputs must map strings to strings")
    if not isinstance(document.get("notes"), str):
        raise ValueError(f"invalid build config {path}: notes must be a string")
    expected = _identity(document.get("dataset"), document["config"],
                         document["inputs"])
    if document.get("build_identity") != expected:
        raise ValueError(f"invalid build config {path}: identity mismatch")
    return document


def value(build_dir_or_document, key: str):
    """Return one dotted config value without inventing a default."""
    if isinstance(build_dir_or_document, dict):
        document, where = build_dir_or_document, "<in-memory build config>"
    else:
        document = load(build_dir_or_document)
        where = str(Path(build_dir_or_document) / BUILD_CONFIG_NAME)
    result = _get(document.get("config", {}), key, _MISSING)
    if result is _MISSING:
        raise MissingConfigValue(f"{where} does not record {key!r}")
    return result


def _get(config: dict, dotted: str, default=None):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
