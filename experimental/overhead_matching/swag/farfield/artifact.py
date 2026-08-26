"""Strict identities and transactional publication for farfield artifacts.

An artifact is a directory whose ``manifest.json`` is written only after every
declared output has been produced and validated.  Writers build in a sibling
``.incomplete`` directory and publish with a single rename.  Readers never
infer identity from a directory layout: the typed manifest is authoritative.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from experimental.overhead_matching.swag.farfield import code_provenance


SCHEMA = "farfield.artifact.v1"
MANIFEST_NAME = "manifest.json"
INCOMPLETE_SUFFIX = ".incomplete"

_MANIFEST_KEYS = frozenset({
    "schema",
    "kind",
    "dataset",
    "version",
    "generator",
    "git_commit",
    "created",
    "arguments",
    "content_digest",
    "upstreams",
    "config",
    "declared_outputs",
    "complete",
})
# Manifest keys that may be absent. `code_provenance` was added after
# artifacts were already on disk, and it is provenance rather than contract:
# refusing to read a manifest for lacking it would strand every artifact
# published before it existed, for no gain -- nothing depends on it being
# there. `_require_exact_keys` therefore checks the required set only.
_OPTIONAL_MANIFEST_KEYS = frozenset(
    {"code_provenance", "artifact_identity", "recipe"})
_REF_KEYS = frozenset({
    "path",
    "kind",
    "dataset",
    "version",
    "manifest_digest",
    "content_digest",
})
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactError(ValueError):
    """Base class for an invalid artifact contract."""


class ArtifactValidationError(ArtifactError):
    """Raised when an artifact does not match its manifest."""


class ArtifactExistsError(FileExistsError):
    """Raised when publication would reuse or replace an artifact directory."""


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                        what: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ArtifactValidationError(f"invalid {what}: " + ", ".join(details))


def require_identifier(value: Any, field_name: str) -> str:
    """Return one valid manifest/path identifier or reject it.

    Artifact identity components are also used as directory names by the
    orchestration layer. Keeping this validator public prevents those two
    boundaries from drifting apart.
    """
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ArtifactValidationError(
            f"{field_name} must be a non-empty path-free identifier")
    return value


def _require_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ArtifactValidationError(
            f"{field_name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _validate_json_value(value: Any, where: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{where} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(
                    f"{where} contains a non-string object key")
            _validate_json_value(item, f"{where}.{key}")
        return
    raise ArtifactValidationError(
        f"{where} contains non-JSON value of type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value deterministically for identity computation."""
    _validate_json_value(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 digest of one regular, non-symlink file."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ArtifactValidationError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_files(root: Path, excluded: frozenset[str]) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactValidationError(f"not a regular directory: {root}")
    files = []
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if entry.is_symlink():
            raise ArtifactValidationError(
                f"artifact content cannot contain a symlink: {relative}")
        if entry.is_file():
            files.append(entry)
        elif not entry.is_dir():
            raise ArtifactValidationError(
                f"artifact content is not a regular file: {relative}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


# Annotations ABOUT a manifest rather than content of it, and both derived
# from the same recipe. Excluded from `manifest_digest` so an artifact
# published before they existed can still gain them -- see the note in
# `manifest_digest`. Tampering with either is caught by a stronger check than
# a digest: `artifact_recipe.verify_self_describing` recomputes the identity
# from the recipe and requires it to equal the recorded identity, so they
# cannot be edited independently of each other.
_DIGEST_EXCLUDED_KEYS = frozenset({"artifact_identity", "recipe"})


def manifest_digest(manifest_path: Path | str) -> str:
    """The digest every downstream `ArtifactRef` records for this artifact.

    Deliberately computed over the manifest MINUS the annotation keys in
    `_DIGEST_EXCLUDED_KEYS`, not over the file's bytes. Two reasons, and the
    second is the load-bearing one:

    - The identity is derived FROM the manifest's own contents. Hashing it
      into the digest that names the manifest is circular.
    - An artifact published before identity existed cannot otherwise ever
      gain one. `manifest_digest` is recorded by every downstream ref, and by
      the frozen request sets and work snapshots inside downstream artifacts,
      which are covered by `content_digest` and so cannot be rewritten at
      all. Measured on the real root: 32 of the 56 unattributed artifacts had
      their manifest digest baked into a downstream artifact's immutable
      content, so adding a field to their manifests the naive way would have
      broken checks that compare a stored ref to a live one -- and the only
      alternative was a permanent second lookup path beside the data.

    Excluding one annotation costs nothing and closes both. Every manifest on
    disk is written by `atomic_write_json`, so its bytes already equal
    canonical JSON plus a newline; with no `artifact_identity` key present
    this returns exactly what `sha256_file` returned before, for all 115
    artifacts on the real root. The migration that verified that is in the
    decision journal.
    """
    manifest_path = Path(manifest_path)
    return manifest_digest_of_document(
        json.loads(manifest_path.read_text(encoding="utf-8")))


def manifest_digest_of_document(document: Any) -> str:
    """`manifest_digest` for a manifest already in hand.

    The one definition, so a caller that has the document cannot compute it a
    slightly different way -- the trailing newline `atomic_write_json` adds is
    exactly the kind of detail two implementations disagree about.
    """
    if isinstance(document, dict):
        document = {key: value for key, value in document.items()
                    if key not in _DIGEST_EXCLUDED_KEYS}
    return hashlib.sha256(canonical_json_bytes(document) + b"\n").hexdigest()


def sha256_directory(path: Path | str,
                     *, exclude: Iterable[str] = (MANIFEST_NAME,)) -> str:
    """Hash the relative names, sizes, and bytes of a directory's files.

    The manifest is excluded by default so its ``content_digest`` is not
    self-referential.  Symlinks are rejected because their targets are mutable
    and may lie outside the artifact.
    """
    root = Path(path)
    excluded = frozenset(PurePosixPath(item).as_posix() for item in exclude)
    records = []
    for item in _directory_files(root, excluded):
        records.append({
            "path": item.relative_to(root).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    return sha256_json(records)


# Verb-first aliases are useful at call sites that group digest operations.
canonical_json_sha256 = sha256_json
file_sha256 = sha256_file
directory_sha256 = sha256_directory


def _normalize_output(value: str | Path) -> str:
    if isinstance(value, Path):
        value = value.as_posix()
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError("declared output must be a non-empty string")
    if "\\" in value:
        raise ArtifactValidationError(
            f"declared output must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or path.as_posix() != value
            or any(part in ("", ".", "..") for part in path.parts)):
        raise ArtifactValidationError(
            f"declared output must be a normalized relative path: {value!r}")
    normalized = path.as_posix()
    if normalized == MANIFEST_NAME:
        raise ArtifactValidationError(
            f"{MANIFEST_NAME} is reserved and cannot be a declared output")
    return normalized


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable artifact identity plus an informational current path.

    ``path`` is deliberately excluded from equality and hashing.  Moving an
    immutable artifact does not change its identity; the two digests do.
    """

    kind: str
    dataset: str
    version: str
    manifest_digest: str
    content_digest: str
    path: str = field(compare=False)

    def __post_init__(self) -> None:
        require_identifier(self.kind, "artifact ref kind")
        require_identifier(self.dataset, "artifact ref dataset")
        require_identifier(self.version, "artifact ref version")
        _require_digest(self.manifest_digest, "artifact ref manifest_digest")
        _require_digest(self.content_digest, "artifact ref content_digest")
        if not isinstance(self.path, str) or not self.path:
            raise ArtifactValidationError(
                "artifact ref path must be a non-empty informational string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "dataset": self.dataset,
            "version": self.version,
            "manifest_digest": self.manifest_digest,
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactRef":
        if not isinstance(value, dict):
            raise ArtifactValidationError("artifact ref must be a JSON object")
        _require_exact_keys(value, _REF_KEYS, "artifact ref")
        return cls(
            path=value["path"],
            kind=value["kind"],
            dataset=value["dataset"],
            version=value["version"],
            manifest_digest=value["manifest_digest"],
            content_digest=value["content_digest"],
        )


def records_same_artifact(document: Any, reference: "ArtifactRef") -> bool:
    """Whether a stored ref document names the same artifact as `reference`.

    For the many places that recorded an `ArtifactRef` as JSON and later have
    to check an artifact against it. Comparing the raw dicts is wrong: it
    reinstates `path`, which `ArtifactRef` deliberately excludes from
    equality because moving an immutable artifact does not change what it is.
    A relocated data root would fail every one of those bindings while every
    byte still agreed.

    Malformed stored data is not the same artifact -- it is not an error
    either, since callers use this inside a boolean chain whose job is to
    decide whether a recorded claim still holds.
    """
    try:
        return ArtifactRef.from_dict(document) == reference
    except (ArtifactError, ValueError, TypeError):
        return False


@dataclass(frozen=True)
class ArtifactManifest:
    """The complete, versioned contract stored in ``manifest.json``."""

    kind: str
    dataset: str
    version: str
    generator: str
    git_commit: str
    created: str
    arguments: tuple[str, ...]
    content_digest: str
    upstreams: tuple[ArtifactRef, ...]
    config: Mapping[str, Any]
    declared_outputs: tuple[str, ...]
    complete: bool = True
    schema: str = SCHEMA
    # The commit and working diff of the code that produced this artifact.
    # Recorded, never enforced: identity is data lineage (see
    # `artifact_identity`), and this is what makes "was it made by different
    # code?" answerable whenever someone asks. `None` on artifacts published
    # before it existed.
    code_provenance: Mapping[str, Any] | None = None
    # The data-lineage digest of this artifact, supplied by whatever drove
    # the build. Optional for the same reason as `code_provenance`: every
    # artifact published before it existed lacks it, and those read as
    # UNATTRIBUTED rather than as corrupt (see `artifact_identity`).
    artifact_identity: str | None = None
    # Everything needed to recompute this artifact's identity, and to know
    # what to run to reproduce it, WITHOUT joining to a build directory.
    # Two terms of the identity are not otherwise recoverable from a
    # manifest -- the stage's resolved config and the build inputs the stage
    # read -- and `builds/` is mutable orchestration state that nothing
    # protects. See `artifact_recipe`.
    recipe: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ArtifactValidationError(
                f"unsupported artifact schema {self.schema!r}; expected {SCHEMA!r}")
        require_identifier(self.kind, "artifact kind")
        require_identifier(self.dataset, "artifact dataset")
        require_identifier(self.version, "artifact version")
        for field_name, value in (("generator", self.generator),
                                  ("git_commit", self.git_commit),
                                  ("created", self.created)):
            if not isinstance(value, str) or not value:
                raise ArtifactValidationError(
                    f"artifact {field_name} must be a non-empty string")
        if not isinstance(self.arguments, tuple) or not all(
                isinstance(value, str) for value in self.arguments):
            raise ArtifactValidationError(
                "artifact arguments must be a tuple of strings")
        _require_digest(self.content_digest, "artifact content_digest")
        if not isinstance(self.upstreams, tuple) or not all(
                isinstance(item, ArtifactRef) for item in self.upstreams):
            raise ArtifactValidationError(
                "artifact upstreams must be a tuple of ArtifactRef values")
        if len(set(self.upstreams)) != len(self.upstreams):
            raise ArtifactValidationError(
                "artifact upstream identities must be unique")
        if not isinstance(self.config, Mapping):
            raise ArtifactValidationError("artifact config must be a JSON object")
        _validate_json_value(dict(self.config), "artifact config")
        if not isinstance(self.declared_outputs, tuple):
            raise ArtifactValidationError(
                "artifact declared_outputs must be a tuple of strings")
        normalized = tuple(_normalize_output(item)
                           for item in self.declared_outputs)
        if normalized != tuple(sorted(set(normalized))):
            raise ArtifactValidationError(
                "artifact declared_outputs must be unique and sorted")
        if type(self.complete) is not bool or not self.complete:
            raise ArtifactValidationError(
                "a published artifact manifest must have complete=true")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "kind": self.kind,
            "dataset": self.dataset,
            "version": self.version,
            "generator": self.generator,
            "git_commit": self.git_commit,
            "created": self.created,
            "arguments": list(self.arguments),
            "content_digest": self.content_digest,
            "upstreams": [item.to_dict() for item in self.upstreams],
            "config": dict(self.config),
            "declared_outputs": list(self.declared_outputs),
            "complete": self.complete,
        }
        # Omitted rather than written as null when absent, so a manifest read
        # from an older artifact round-trips to the bytes it came from.
        if self.code_provenance is not None:
            value["code_provenance"] = dict(self.code_provenance)
        if self.artifact_identity is not None:
            value["artifact_identity"] = self.artifact_identity
        if self.recipe is not None:
            value["recipe"] = dict(self.recipe)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> "ArtifactManifest":
        if not isinstance(value, dict):
            raise ArtifactValidationError("artifact manifest must be a JSON object")
        _require_exact_keys(
            {key: item for key, item in value.items()
             if key not in _OPTIONAL_MANIFEST_KEYS},
            _MANIFEST_KEYS, "artifact manifest")
        code = value.get("code_provenance")
        if code is not None and not isinstance(code, dict):
            raise ArtifactValidationError(
                "artifact code_provenance must be a JSON object")
        identity = value.get("artifact_identity")
        if identity is not None:
            _require_digest(identity, "artifact artifact_identity")
        recipe = value.get("recipe")
        if recipe is not None and not isinstance(recipe, dict):
            raise ArtifactValidationError(
                "artifact recipe must be a JSON object")
        upstreams = value["upstreams"]
        if not isinstance(upstreams, list):
            raise ArtifactValidationError("artifact upstreams must be a JSON list")
        config = value["config"]
        if not isinstance(config, dict):
            raise ArtifactValidationError("artifact config must be a JSON object")
        outputs = value["declared_outputs"]
        if not isinstance(outputs, list):
            raise ArtifactValidationError(
                "artifact declared_outputs must be a JSON list")
        arguments = value["arguments"]
        if not isinstance(arguments, list):
            raise ArtifactValidationError(
                "artifact arguments must be a JSON list")
        return cls(
            schema=value["schema"],
            kind=value["kind"],
            dataset=value["dataset"],
            version=value["version"],
            generator=value["generator"],
            git_commit=value["git_commit"],
            created=value["created"],
            arguments=tuple(arguments),
            content_digest=value["content_digest"],
            code_provenance=code,
            artifact_identity=identity,
            recipe=recipe,
            upstreams=tuple(ArtifactRef.from_dict(item) for item in upstreams),
            config=config,
            declared_outputs=tuple(outputs),
            complete=value["complete"],
        )


def atomic_write_file(path: Path | str, data: bytes) -> None:
    """Atomically replace one file using a unique temporary sibling."""
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_file data must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON with a trailing newline."""
    atomic_write_file(path, canonical_json_bytes(value) + b"\n")


def atomic_create_file(path: Path | str, data: bytes) -> None:
    """Atomically create one file without replacing an existing entry."""
    path = Path(path)
    if not isinstance(data, bytes):
        raise TypeError("atomic_create_file data must be bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is an atomic no-clobber operation. A file or
        # symlink created by a concurrent writer causes FileExistsError.
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path | str, value: Any) -> None:
    """Atomically create deterministic JSON without clobbering a recipe."""
    atomic_create_file(path, canonical_json_bytes(value) + b"\n")


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


def _fsync_artifact_tree(root: Path) -> None:
    """Flush artifact files and directory entries before final publication."""
    files = _directory_files(root, frozenset())
    for path in files:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [root]
    directories.extend(
        path for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink())
    for path in sorted(directories, key=lambda item: len(item.parts),
                       reverse=True):
        _fsync_directory(path)


def publish_directory_no_clobber(staging: Path | str,
                                 destination: Path | str) -> None:
    """Durably publish a staged directory without replacing a winner.

    Every regular file and directory entry in the staging tree is flushed
    before the rename, then the parent directory is flushed after it. Keeping
    the complete durability boundary here makes this primitive safe for typed
    artifacts and untyped diagnostic side outputs alike.
    """
    staging = Path(staging)
    destination = Path(destination)
    if staging.is_symlink() or not staging.is_dir():
        raise ArtifactValidationError(
            f"staging path is not a regular directory: {staging}")
    if staging.parent.resolve() != destination.parent.resolve():
        raise ArtifactValidationError(
            "staging and destination must be sibling directories")
    parent_fd = _directory_fd(destination.parent)
    try:
        fcntl.flock(parent_fd, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise ArtifactExistsError(
                f"completed artifact appeared during publication: {destination}")
        _fsync_artifact_tree(staging)
        os.rename(staging, destination)
        os.fsync(parent_fd)
    finally:
        try:
            fcntl.flock(parent_fd, fcntl.LOCK_UN)
        finally:
            os.close(parent_fd)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _is_incomplete_path(path: Path) -> bool:
    return any(part.endswith(INCOMPLETE_SUFFIX) for part in path.parts)


def _manifest_path(path: Path | str) -> tuple[Path, Path]:
    path = Path(path)
    if path.name == MANIFEST_NAME:
        return path.parent, path
    return path, path / MANIFEST_NAME


def _load_manifest(path: Path | str, *, allow_incomplete: bool) -> ArtifactManifest:
    artifact_dir, manifest_path = _manifest_path(path)
    if not allow_incomplete and _is_incomplete_path(artifact_dir):
        raise ArtifactValidationError(
            f"incomplete artifact cannot be consumed: {artifact_dir}")
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ArtifactValidationError(
            f"artifact directory does not exist: {artifact_dir}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ArtifactValidationError(
            f"artifact manifest does not exist: {manifest_path}")
    try:
        document = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactValidationError(
                    f"invalid non-finite JSON constant {value!r}")),
        )
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            f"invalid artifact manifest {manifest_path}: {error}") from error
    return ArtifactManifest.from_dict(document)


def load_manifest(path: Path | str) -> ArtifactManifest:
    """Strictly decode a completed artifact manifest without inferring fields."""
    return _load_manifest(path, allow_incomplete=False)


def _validate_declared_outputs(root: Path, outputs: Sequence[str]) -> None:
    expected = set(outputs)
    actual = {
        item.relative_to(root).as_posix()
        for item in _directory_files(root, frozenset({MANIFEST_NAME}))
    }
    missing = sorted(expected - actual)
    undeclared = sorted(actual - expected)
    if missing or undeclared:
        details = []
        if missing:
            details.append(f"missing declared outputs {missing}")
        if undeclared:
            details.append(f"undeclared outputs {undeclared}")
        raise ArtifactValidationError(
            f"artifact output mismatch in {root}: " + ", ".join(details))


def _validate_artifact(path: Path | str, *, expected_kind: str | None,
                       expected_dataset: str | None,
                       expected_version: str | None,
                       allow_incomplete: bool) -> ArtifactRef:
    artifact_dir, manifest_path = _manifest_path(path)
    manifest = _load_manifest(artifact_dir, allow_incomplete=allow_incomplete)
    for field_name, expected, actual in (
            ("kind", expected_kind, manifest.kind),
            ("dataset", expected_dataset, manifest.dataset),
            ("version", expected_version, manifest.version)):
        if expected is not None and actual != expected:
            raise ArtifactValidationError(
                f"artifact {field_name} mismatch: expected {expected!r}, "
                f"found {actual!r}")
    _validate_declared_outputs(artifact_dir, manifest.declared_outputs)
    actual_digest = sha256_directory(artifact_dir)
    if actual_digest != manifest.content_digest:
        raise ArtifactValidationError(
            "artifact content digest mismatch: expected "
            f"{manifest.content_digest}, found {actual_digest}")
    return ArtifactRef(
        path=str(artifact_dir.resolve()),
        kind=manifest.kind,
        dataset=manifest.dataset,
        version=manifest.version,
        manifest_digest=manifest_digest(manifest_path),
        content_digest=manifest.content_digest,
    )


def validate_artifact(path: Path | str, *, expected_kind: str | None = None,
                      expected_dataset: str | None = None,
                      expected_version: str | None = None) -> ArtifactRef:
    """Validate a published artifact and return its immutable identity."""
    return _validate_artifact(
        path,
        expected_kind=expected_kind,
        expected_dataset=expected_dataset,
        expected_version=expected_version,
        allow_incomplete=False,
    )


# ``open_artifact`` makes the rejection boundary explicit at reader call sites.
open_artifact = validate_artifact


class ArtifactDirectoryBuilder:
    """Build and atomically publish one immutable artifact directory."""

    def __init__(self, destination: Path | str, *, kind: str, dataset: str,
                 version: str, generator: str,
                 git_commit: str = "unknown",
                 arguments: Iterable[str] | None = None,
                 upstreams: Iterable[ArtifactRef] = (),
                 config: Mapping[str, Any] | None = None,
                 artifact_identity: str | None = None,
                 recipe: Mapping[str, Any] | None = None,
                 declared_outputs: Iterable[str | Path]) -> None:
        self.destination = Path(destination)
        if self.destination.name.endswith(INCOMPLETE_SUFFIX):
            raise ArtifactValidationError(
                "artifact destination must not itself end in .incomplete")
        self.staging_dir = self.destination.with_name(
            self.destination.name + INCOMPLETE_SUFFIX)
        self.path = self.staging_dir
        self.kind = require_identifier(kind, "artifact kind")
        self.dataset = require_identifier(dataset, "artifact dataset")
        self.version = require_identifier(version, "artifact version")
        if not isinstance(generator, str) or not generator:
            raise ArtifactValidationError("artifact generator must be non-empty")
        if not isinstance(git_commit, str) or not git_commit:
            raise ArtifactValidationError("artifact git_commit must be non-empty")
        self.generator = generator
        self.git_commit = git_commit
        self.arguments = tuple(sys.argv if arguments is None else arguments)
        if not all(isinstance(value, str) for value in self.arguments):
            raise ArtifactValidationError("artifact arguments must be strings")
        self.upstreams = tuple(upstreams)
        if not all(isinstance(item, ArtifactRef) for item in self.upstreams):
            raise ArtifactValidationError(
                "artifact upstreams must contain only ArtifactRef values")
        self.config = dict(config or {})
        _validate_json_value(self.config, "artifact config")
        # Supplied by the orchestrator that resolved this build's recipe --
        # the identity formula needs per-stage knowledge (which recorded
        # inputs the stage reads) that a producer does not have. Absent when
        # a producer is driven by hand, and then the artifact is honestly
        # UNATTRIBUTED rather than carrying a digest nobody computed.
        if artifact_identity is not None:
            _require_digest(artifact_identity, "artifact artifact_identity")
        self.artifact_identity = artifact_identity
        if recipe is not None:
            _validate_json_value(recipe, "artifact recipe")
        self.recipe = dict(recipe) if recipe is not None else None
        outputs = tuple(_normalize_output(item) for item in declared_outputs)
        if len(outputs) != len(set(outputs)):
            raise ArtifactValidationError("declared outputs must be unique")
        self.declared_outputs = tuple(sorted(outputs))
        self.artifact_ref: ArtifactRef | None = None
        self._entered = False

    def __enter__(self) -> "ArtifactDirectoryBuilder":
        if self._entered:
            raise RuntimeError("artifact builder cannot be entered twice")
        if self.destination.exists() or self.destination.is_symlink():
            raise ArtifactExistsError(
                f"completed artifact already exists: {self.destination}")
        if self.staging_dir.exists() or self.staging_dir.is_symlink():
            raise ArtifactExistsError(
                f"incomplete artifact already exists: {self.staging_dir}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.staging_dir.mkdir()
        except FileExistsError as error:
            raise ArtifactExistsError(
                f"incomplete artifact already exists: {self.staging_dir}") from error
        self._entered = True
        return self

    def __exit__(self, error_type: type[BaseException] | None,
                 error: BaseException | None, traceback: Any) -> bool:
        del error, traceback
        if error_type is None and self.artifact_ref is None:
            self.publish()
        return False

    def output_path(self, relative_path: str | Path) -> Path:
        """Return the staging path for one declared output."""
        relative = _normalize_output(relative_path)
        if relative not in self.declared_outputs:
            raise ArtifactValidationError(
                f"output was not declared for this artifact: {relative}")
        return self.staging_dir / relative

    def publish(self) -> ArtifactRef:
        """Validate, write the manifest last, then rename into final position."""
        if not self._entered:
            raise RuntimeError("artifact builder must be entered before publication")
        if self.artifact_ref is not None:
            return self.artifact_ref
        if self.destination.exists() or self.destination.is_symlink():
            raise ArtifactExistsError(
                f"completed artifact already exists: {self.destination}")
        _validate_declared_outputs(self.staging_dir, self.declared_outputs)
        manifest = ArtifactManifest(
            kind=self.kind,
            dataset=self.dataset,
            version=self.version,
            generator=self.generator,
            git_commit=self.git_commit,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            arguments=self.arguments,
            content_digest=sha256_directory(self.staging_dir),
            upstreams=self.upstreams,
            config=self.config,
            declared_outputs=self.declared_outputs,
            # Stamped here rather than by each producer. There are nine of
            # them and every one publishes through this class, so this is the
            # only place that cannot be forgotten -- and a producer that
            # forgot would be indistinguishable from one whose code was never
            # recorded.
            code_provenance=code_provenance.record(),
            artifact_identity=self.artifact_identity,
            recipe=self.recipe,
        )
        # This is intentionally the final write in the staging directory.
        atomic_write_json(self.staging_dir / MANIFEST_NAME, manifest.to_dict())
        reference = _validate_artifact(
            self.staging_dir,
            expected_kind=self.kind,
            expected_dataset=self.dataset,
            expected_version=self.version,
            allow_incomplete=True,
        )
        publish_directory_no_clobber(self.staging_dir, self.destination)
        self.artifact_ref = ArtifactRef(
            path=str(self.destination.resolve()),
            kind=reference.kind,
            dataset=reference.dataset,
            version=reference.version,
            manifest_digest=reference.manifest_digest,
            content_digest=reference.content_digest,
        )
        return self.artifact_ref


def transactional_directory(
        destination: Path | str, *, kind: str, dataset: str, version: str,
        generator: str, git_commit: str = "unknown",
        arguments: Iterable[str] | None = None,
        upstreams: Iterable[ArtifactRef] = (),
        config: Mapping[str, Any] | None = None,
        declared_outputs: Iterable[str | Path]) -> ArtifactDirectoryBuilder:
    """Convenience constructor for :class:`ArtifactDirectoryBuilder`."""
    return ArtifactDirectoryBuilder(
        destination,
        kind=kind,
        dataset=dataset,
        version=version,
        generator=generator,
        git_commit=git_commit,
        arguments=arguments,
        upstreams=upstreams,
        config=config,
        declared_outputs=declared_outputs,
    )
