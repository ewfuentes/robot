"""Transactional publication for loose, typed catalog source Feathers.

Collection builds use loose source Feathers as intermediate raw material and
then publish the selected result as an immutable CATALOGS artifact. A source
Feather is complete only when its strict compact schema and provenance have
both validated. The Feather is therefore linked into place last and is the
completion marker; every final path is no-clobber.
"""

import fcntl
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.catalog import schema

SOURCE_PROVENANCE_SCHEMA = "farfield.catalog_source.v1"
_PUBLICATION_KEYS = frozenset({
    "schema", "output", "output_sha256", "rows_out", "complete",
})


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def output_paths(output: Path) -> tuple[Path, Path, Path]:
    feather = Path(output).with_suffix(".feather")
    sidecar = feather.with_suffix(".provenance.json")
    staging = feather.with_name(f".{feather.name}{artifact.INCOMPLETE_SUFFIX}")
    return feather, sidecar, staging


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid source provenance {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"source provenance must be an object: {path}")
    return document


def require_exact_provenance(
        document: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Require every producer-owned field and no unknown sidecar fields."""
    expected_keys = frozenset(expected) | _PUBLICATION_KEYS
    actual_keys = frozenset(document)
    if actual_keys != expected_keys:
        raise ValueError(
            "completed source provenance fields differ: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unknown={sorted(actual_keys - expected_keys)}")
    disagreements = sorted(
        key for key, value in expected.items()
        if document.get(key) != value)
    if disagreements:
        raise ValueError(
            "completed source provenance differs for fields "
            f"{disagreements}")


def _validate_completed_pair(
        feather: Path, sidecar: Path) -> tuple[Any, dict[str, Any]]:
    if (feather.is_symlink() or not feather.is_file()
            or sidecar.is_symlink() or not sidecar.is_file()):
        raise FileExistsError(
            f"completed source pair is not two regular files: "
            f"{feather}, {sidecar}")
    document = _load_document(sidecar)
    if (document.get("schema") != SOURCE_PROVENANCE_SCHEMA
            or document.get("complete") is not True
            or document.get("output") != str(feather)):
        raise ValueError(
            f"completed source provenance has invalid identity: {sidecar}")
    before = artifact.sha256_file(feather)
    if document.get("output_sha256") != before:
        raise ValueError(
            f"completed source payload digest mismatch: {feather}")
    frame = schema.read_frame(feather)
    if (isinstance(document.get("rows_out"), bool)
            or not isinstance(document.get("rows_out"), int)
            or document["rows_out"] != len(frame)):
        raise ValueError(
            f"completed source row count mismatch: {feather}")
    if artifact.sha256_file(feather) != before:
        raise ValueError(
            f"completed source payload changed during validation: {feather}")
    return frame, document


def validate_completed_pair(
        feather: Path, sidecar: Path) -> tuple[Any, dict[str, Any]]:
    """Strictly reopen a completed source: (compact frame, sidecar document)."""
    return _validate_completed_pair(Path(feather), Path(sidecar))


def reuse_completed(
        output: Path,
        expected_factory: Callable[[Any, dict[str, Any]], Mapping[str, Any]],
) -> Any | None:
    """Return an exact completed source, or ``None`` when no output exists.

    Any partial, corrupt, or different completed identity fails closed. The
    producer callback receives the strictly reopened compact frame and full
    sidecar and must reconstruct every producer-owned provenance field.
    """
    feather, sidecar, staging = output_paths(output)
    feather.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        feather.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if (not feather.exists() and not feather.is_symlink()
                and (sidecar.exists() or sidecar.is_symlink())
                and (staging.exists() or staging.is_symlink())):
            _recover_staged_only(feather, sidecar, staging)
        present = [
            path for path in (feather, sidecar, staging)
            if path.exists() or path.is_symlink()
        ]
        if not present:
            return None
        if present != [feather, sidecar]:
            raise FileExistsError(
                f"refusing incomplete or ambiguous source output: {present}")
        frame, document = _validate_completed_pair(feather, sidecar)
        expected = expected_factory(frame, document)
        if not isinstance(expected, Mapping):
            raise TypeError("expected source provenance must be a mapping")
        require_exact_provenance(document, expected)
        return frame
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def preflight_output(output: Path) -> tuple[Path, Path, Path]:
    feather, sidecar, staging = output_paths(output)
    feather.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        feather.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _preflight_locked(feather, sidecar, staging)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return feather, sidecar, staging


def _preflight_locked(feather: Path, sidecar: Path, staging: Path) -> None:
    # A process can die after the provenance hard link but before the payload
    # hard link.  That state is not a completed output: validate that it is
    # exactly our intact private staging pair, then discard it so the explicit
    # rerun can recompute and publish normally.  Ambiguous/corrupt residue is
    # retained for diagnosis and still fails closed.
    if (not feather.exists() and not feather.is_symlink()
            and (sidecar.exists() or sidecar.is_symlink())
            and (staging.exists() or staging.is_symlink())):
        _recover_staged_only(feather, sidecar, staging)
    for path in (feather, sidecar, staging):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite completed or incomplete output: {path}")


def _recover_staged_only(feather: Path, sidecar: Path, staging: Path) -> None:
    if staging.is_symlink() or not staging.is_dir():
        raise FileExistsError(
            f"refusing ambiguous incomplete output: {staging}")
    staged_feather = staging / "catalog.feather"
    staged_sidecar = staging / "provenance.json"
    entries = list(staging.iterdir())
    if ({entry.name for entry in entries}
            != {staged_feather.name, staged_sidecar.name}
            or any(entry.is_symlink() or not entry.is_file()
                   for entry in entries)):
        raise FileExistsError(
            f"refusing corrupt incomplete output: {staging}")
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink() or not sidecar.is_file():
            raise FileExistsError(
                f"refusing ambiguous final provenance: {sidecar}")
        if sidecar.read_bytes() != staged_sidecar.read_bytes():
            raise FileExistsError(
                f"final provenance does not match staging: {sidecar}")
    try:
        document = json.loads(
            staged_sidecar.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise FileExistsError(
            f"invalid staged provenance {staged_sidecar}: {error}") from error
    if (not isinstance(document, dict)
            or document.get("schema") != SOURCE_PROVENANCE_SCHEMA
            or document.get("complete") is not True
            or document.get("output") != str(feather)
            or document.get("output_sha256")
            != artifact.sha256_file(staged_feather)):
        raise FileExistsError(
            f"staged provenance does not identify its payload: {staging}")
    reopened = schema.read_frame(staged_feather)
    if (isinstance(document.get("rows_out"), bool)
            or not isinstance(document.get("rows_out"), int)
            or document["rows_out"] != len(reopened)):
        raise FileExistsError(
            f"staged provenance row count is not exact: {staging}")

    if sidecar.exists():
        sidecar.unlink()
    shutil.rmtree(staging)
    _flush_directory(feather.parent)


def _flush(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(frame, output: Path, provenance_document: dict) -> tuple[Path, Path]:
    """Strictly validate, stage, and no-clobber a source Feather + sidecar."""
    feather, sidecar, staging = preflight_output(output)
    descriptor = os.open(
        feather.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        # Two producers can both perform the early preflight before either
        # starts writing. Recheck under the publication lock.
        for path in (feather, sidecar, staging):
            if path.exists() or path.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite completed or incomplete output: {path}")
        schema.tag_dicts(frame)
        staging.mkdir()
        staged_feather = staging / "catalog.feather"
        staged_sidecar = staging / "provenance.json"
        frame.to_feather(staged_feather)
        reopened = schema.read_frame(staged_feather)
        if len(reopened) != len(frame):
            raise RuntimeError("staged Feather row count changed on strict reopen")
        payload_sha256 = artifact.sha256_file(staged_feather)
        document = {
            **dict(provenance_document),
            "schema": SOURCE_PROVENANCE_SCHEMA,
            "output": str(feather),
            "output_sha256": payload_sha256,
            "rows_out": int(len(reopened)),
            "complete": True,
        }
        artifact.atomic_write_json(staged_sidecar, document)
        _flush(staged_feather)
        _flush(staged_sidecar)

        # Provenance first, payload last. Consumers only recognize the Feather;
        # a crash before the final hard link cannot expose a partial payload.
        os.link(staged_sidecar, sidecar, follow_symlinks=False)
        try:
            os.link(staged_feather, feather, follow_symlinks=False)
        except BaseException:
            sidecar.unlink(missing_ok=True)
            raise
        _flush_directory(feather.parent)
        shutil.rmtree(staging)
        _flush_directory(feather.parent)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return feather, sidecar
