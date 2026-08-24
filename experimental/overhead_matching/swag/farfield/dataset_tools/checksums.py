"""The one implementation of a dataset's `checksums.sha256` regeneration.

Datasets are immutable outside explicit dataset-mutating tools.
``trim_dataset`` calls ``regenerate`` here so
the manifest format and exclusion list have one owner. Calibration diagnostics
never mutate dataset metadata.  The approved ``nominal_forward.json`` is an
immutable build input in the dataset root, is covered by this manifest, and is
published together with a refreshed manifest by the human-review finalizer.

Format matches `sha256sum` output with `./`-relative paths sorted as bytes
(C locale), covering everything except:

- the manifest itself;
- the `panorama/` symlink tree (it aliases `frames/`, which is covered);
- derived per-dataset products (`_manifests/`, `catalog_cache/`,
  `__pycache__/`): these are rebuildable and rewritten whenever a triage tool
  runs, so checksumming them would report every tool run as corruption.
"""

import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat

CHECKSUM_FILE = "checksums.sha256"
# Rebuildable, tool-rewritten directories. `_manifests/` holds the triage
# sidecars (recording_seams.json, vehicle_anchor.json, regenerated views);
# it is derived data living beside the frozen definition, not part of it.
EXCLUDED_DIRS = frozenset({"catalog_cache", "__pycache__", "_manifests"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(relative: Path) -> bool:
    return (relative.parts[0] != "panorama"
            and relative.name != CHECKSUM_FILE
            and not set(relative.parts) & EXCLUDED_DIRS)


def _replacement_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"checksum replacement path is invalid: {value!r}")
    parsed = PurePosixPath(value)
    if (parsed.is_absolute() or parsed.as_posix() != value
            or any(part in ("", ".", "..") for part in parsed.parts)):
        raise ValueError(f"checksum replacement path is invalid: {value!r}")
    relative = Path(*parsed.parts)
    if not _included(relative):
        raise ValueError(
            f"checksum replacement path is excluded from the manifest: {value}")
    return relative


def manifest_bytes(dataset_base: Path, *,
                   replacements: dict[str, bytes] | None = None) -> bytes:
    """Render the canonical manifest, optionally substituting future bytes.

    This is the read-only owner of manifest enumeration and formatting.  A
    dataset mutation can stage the bytes it intends to publish and ask this
    function for the exact resulting checksum before changing the dataset.
    """
    dataset_base = Path(dataset_base)
    if dataset_base.is_symlink() or not dataset_base.is_dir():
        raise ValueError(
            f"dataset must be a regular, non-symlink directory: {dataset_base}")
    normalized = {}
    for raw_path, payload in (replacements or {}).items():
        relative = _replacement_path(raw_path)
        if not isinstance(payload, bytes):
            raise TypeError(
                f"replacement payload for {raw_path!r} must be bytes")
        normalized[relative.as_posix()] = payload

    entries = {}
    for path in dataset_base.rglob("*"):
        relative = path.relative_to(dataset_base)
        if (len(relative.parts) == 1
                and relative.name.startswith(
                    f".{CHECKSUM_FILE}.incomplete-")):
            raise ValueError(
                f"incomplete checksum publication requires inspection: {path}")
        if not _included(relative):
            continue
        if path.is_symlink():
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unsupported filesystem entry: {path}")
        key = relative.as_posix()
        payload = normalized.pop(key, None)
        entries["./" + key] = (
            hashlib.sha256(payload).hexdigest()
            if payload is not None else file_sha256(path))
    for key, payload in normalized.items():
        entries["./" + key] = hashlib.sha256(payload).hexdigest()
    return "".join(
        f"{entries[key]}  {key}\n"
        for key in sorted(entries, key=lambda item: item.encode())).encode(
            "utf-8")


def verify(dataset_base: Path) -> int:
    """Require the existing manifest to exactly cover current dataset bytes."""
    dataset_base = Path(dataset_base)
    target = dataset_base / CHECKSUM_FILE
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"checksum manifest is not a regular file: {target}")
    expected = manifest_bytes(dataset_base)
    actual = target.read_bytes()
    if actual != expected:
        raise ValueError(
            f"checksum manifest is stale, incomplete, or noncanonical: {target}")
    return len(expected.splitlines())


def _atomic_replace(path: Path, payload: bytes) -> None:
    staging = path.parent / f".{path.name}.incomplete-{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
    descriptor = os.open(staging, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
        parent = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if staging.exists() or staging.is_symlink():
            staging.unlink()


def regenerate(dataset_base: Path) -> int | None:
    """Rewrite `checksums.sha256` over every real file in the dataset.

    Returns the number of manifest lines, or None when the dataset carries no
    manifest (nothing is invented: a dataset that never had integrity checking
    does not gain it as a side effect of an unrelated tool).
    """
    dataset_base = Path(dataset_base)
    target = dataset_base / CHECKSUM_FILE
    if not target.exists() and not target.is_symlink():
        return None
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"checksum manifest is not a regular file: {target}")
    payload = manifest_bytes(dataset_base)
    _atomic_replace(target, payload)
    return len(payload.splitlines())
