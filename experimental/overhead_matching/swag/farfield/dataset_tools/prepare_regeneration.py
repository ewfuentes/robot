"""Inventory and transactionally retire the inactive Mapillary datasets.

The default invocation is strictly read-only: it prints a complete JSON plan
whose digest binds the source trees that were inspected.  Applying a plan is
an intentionally separate operation and requires spelling its digest back to
the tool.  This prevents a reviewed plan from silently expanding to include a
new directory that appeared later.

The retirement location is a collection *inside* the datasets lane, alongside
``unvetted``::

    datasets/out_of_date_but_usable_mapillary_datasets/<dataset>/

Only the explicit allowlist below can move.  In particular, the eight active
datasets and ``unvetted/london_thames`` are protected inputs.  Artifact lanes
are inventoried but never modified by this tool.

Examples::

    # Read-only. Save the reviewed plan outside the data root.
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:\
prepare_regeneration -- --output /tmp/farfield-retirement-plan.json

    # Separate, explicit mutation after reviewing the file and digest.
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:\
prepare_regeneration -- \
      --apply_plan /tmp/farfield-retirement-plan.json \
      --confirm_plan_digest <sha256 printed by the planning command>

Publication is rename-only on one filesystem.  A journal and an unpublished
``.incomplete`` collection make an interruption recoverable.  No source file
or artifact is copied, deleted, or overwritten.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime
import errno
import fcntl
import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance


PLAN_SCHEMA = "farfield_regeneration_preparation_plan/v1"
RETIREMENT_SCHEMA = "farfield_retired_dataset_collection/v1"
JOURNAL_SCHEMA = "farfield_dataset_retirement_transaction/v1"
RETIREMENT_DIRNAME = "out_of_date_but_usable_mapillary_datasets"

ACTIVE_DATASETS = (
    "boston_harbor_leg1",
    "boston_harbor_leg2",
    "boston_harbor_leg3",
    "charles_river_20260727",
    "mount_washington_20260815_leg1",
    "mount_washington_20260815_leg2",
    "mount_washington_20260815_leg3",
    "pohang_canal_04",
)

# This is deliberately not inferred as "everything except ACTIVE_DATASETS".
# A newly collected directory must never become eligible for a move merely by
# appearing under datasets/.
RETIRED_MAPILLARY_DATASETS = (
    "flevoland_polder",
    "folkestone_dover",
    "franconia_notch",
    "friesland_workum",
    "fukuoka_yumechan_a",
    "fukuyama_yasunari",
    "innsbruck_inn_valley",
    "kagoshima_matoken",
    "kumamoto_yumechan_b",
    "miami_beach",
    "mississippi_rural",
    "mt_washington_auto_road",
    "nyc_east_river",
    "nyc_inner_harbor",
    "portsmouth_navalbase",
    "seattle",
    "tangier_morocco",
    "tokyo_bay",
    "zermatt_ski_b",
)

# Containers, not dataset names. Their entire trees are snapshotted so an
# apply can prove that this tool left them alone.
UNTOUCHED_COLLECTIONS = ("unvetted",)

CRITICAL_DATASET_FILES = (
    "pipeline_metadata.json",
    "frames_gps.csv",
    "pano_id_mapping.csv",
    "extraction_log.csv",
    "intrinsics.csv",
    "checksums.sha256",
)


class PreparationError(RuntimeError):
    """The data tree or reviewed plan does not permit a safe operation."""


class _StrictJsonError(ValueError):
    """A JSON document used a representation we refuse to canonicalize."""


def _strict_json_loads(payload: str | bytes, label: str) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _StrictJsonError(f"duplicate object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _StrictJsonError(f"non-finite number {value!r}")

    try:
        return json.loads(
            payload, object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJsonError) as exc:
        raise PreparationError(f"{label} is invalid strict JSON: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use Linux ``renameat2`` so a racing destination is never overwritten."""
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise PreparationError(
            "this platform lacks renameat2; refusing a non-atomic retirement")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise PreparationError(
            f"refusing to overwrite rename destination: {destination}")
    raise OSError(error, os.strerror(error), str(source), str(destination))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise PreparationError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PreparationError(
            f"{label} must be a real, non-symlink directory: {path}")


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PreparationError(f"{label} escapes data root: {path}") from exc


def _walk_tree(root: Path) -> list[dict[str, Any]]:
    """Return a content-bound tree without following symlinks."""
    _require_real_directory(root, "inventory root")
    records: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            raise PreparationError(f"cannot inventory {directory}: {exc}") \
                from exc
        for child in children:
            rel = child.relative_to(root).as_posix()
            metadata = child.lstat()
            mode = metadata.st_mode
            permissions = stat.S_IMODE(mode)
            if stat.S_ISLNK(mode):
                records.append({
                    "path": rel,
                    "type": "symlink",
                    "mode": permissions,
                    "target": os.readlink(child),
                })
            elif stat.S_ISDIR(mode):
                records.append({
                    "path": rel,
                    "type": "directory",
                    "mode": permissions,
                })
                pending.append(child)
            elif stat.S_ISREG(mode):
                records.append({
                    "path": rel,
                    "type": "file",
                    "mode": permissions,
                    "size": metadata.st_size,
                    "sha256": _sha256_file(child),
                })
            else:
                raise PreparationError(
                    f"unsupported filesystem entry in dataset: {child}")
    return sorted(records, key=lambda record: record["path"])


def _tree_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        kind: sum(record["type"] == kind for record in records)
        for kind in ("file", "directory", "symlink")
    }
    return {
        "tree_digest": _digest(records),
        "entries": len(records),
        "regular_files": counts["file"],
        "directories": counts["directory"],
        "symlinks": counts["symlink"],
        "regular_file_bytes": sum(
            record.get("size", 0) for record in records
            if record["type"] == "file"),
    }


def _critical_files(dataset: Path) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for name in CRITICAL_DATASET_FILES:
        path = dataset / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            result[name] = None
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise PreparationError(
                f"critical dataset input must be a regular file: {path}")
        result[name] = {
            "size": metadata.st_size,
            "sha256": _sha256_file(path),
        }
    return result


def _lexical_target(link: Path, raw_target: str) -> Path:
    """Normalize ``..`` lexically without following the target's symlinks."""
    return Path(os.path.abspath(os.path.join(link.parent, raw_target)))


def _target_identity(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    if stat.S_ISREG(metadata.st_mode):
        kind = "file"
    elif stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    else:
        raise PreparationError(f"symlink resolves to unsupported entry: {path}")
    return {"device": metadata.st_dev, "inode": metadata.st_ino,
            "type": kind}


def _symlink_plan(dataset: Path, destination: Path, data_root: Path,
                  records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root_resolved = data_root.resolve(strict=True)
    source_resolved = dataset.resolve(strict=True)
    links = []
    for record in records:
        if record["type"] != "symlink":
            continue
        rel = PurePosixPath(record["path"])
        link = dataset.joinpath(*rel.parts)
        raw = record["target"]
        if Path(raw).is_absolute():
            raise PreparationError(f"absolute symlink is not relocatable: "
                                   f"{link} -> {raw}")
        lexical = _lexical_target(link, raw)
        _relative(lexical, root_resolved,
                  f"symlink target for {record['path']}")
        try:
            resolved = link.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as exc:
            raise PreparationError(
                f"dangling or cyclic symlink: {link} -> {raw}") from exc
        _relative(resolved, root_resolved,
                  f"resolved symlink target for {record['path']}")
        try:
            lexical.relative_to(source_resolved)
            internal = True
        except ValueError:
            internal = False
        new_link = destination.joinpath(*rel.parts)
        planned = (raw if internal else
                   os.path.relpath(lexical, start=new_link.parent))
        links.append({
            "path": record["path"],
            "original_target": raw,
            "retired_target": planned,
            "scope": "dataset" if internal else "data_root",
            "target_path": _relative(lexical, root_resolved,
                                     f"symlink target for {record['path']}"),
            "resolved_target": _relative(
                resolved, root_resolved,
                f"resolved symlink target for {record['path']}"),
            "target_identity": _target_identity(link),
        })
    return links


def _expected_retired_digest(records: list[dict[str, Any]],
                             links: list[dict[str, Any]]) -> str:
    targets = {link["path"]: link["retired_target"] for link in links}
    transformed = []
    for record in records:
        item = dict(record)
        if item["type"] == "symlink":
            item["target"] = targets[item["path"]]
        transformed.append(item)
    return _digest(transformed)


def _is_manifest_or_provenance(path: Path) -> bool:
    name = path.name.lower()
    return (name == "manifest.json" or name.endswith("_manifest.json")
            or name == "provenance.json"
            or name.endswith(".provenance.json"))


def _manifest_provenance_files(directory: Path) -> list[dict[str, Any]]:
    """Hash only small identity records, never artifact payloads."""
    records = []
    pending = [directory]
    while pending:
        parent = pending.pop()
        for entry in sorted(parent.iterdir(), key=lambda path: path.name):
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                if not _is_manifest_or_provenance(entry):
                    continue
                try:
                    target = entry.resolve(strict=True)
                except (FileNotFoundError, RuntimeError) as exc:
                    raise PreparationError(
                        f"dangling artifact identity record: {entry}") from exc
                if not target.is_file():
                    raise PreparationError(
                        f"artifact identity link is not a file: {entry}")
                records.append({
                    "path": entry.relative_to(directory).as_posix(),
                    "type": "symlink",
                    "target": os.readlink(entry),
                    "size": target.stat().st_size,
                    "sha256": _sha256_file(target),
                })
            elif stat.S_ISDIR(metadata.st_mode):
                pending.append(entry)
            elif stat.S_ISREG(metadata.st_mode) and (
                    _is_manifest_or_provenance(entry)):
                records.append({
                    "path": entry.relative_to(directory).as_posix(),
                    "type": "file",
                    "size": metadata.st_size,
                    "sha256": _sha256_file(entry),
                })
    return sorted(records, key=lambda record: record["path"])


def _artifact_inventory(root: Path, dataset: str) -> list[dict[str, Any]]:
    artifacts = root / "artifacts"
    output = []
    for kind in sorted(artifacts.iterdir(), key=lambda path: path.name):
        if kind.is_symlink() or not kind.is_dir():
            continue
        dataset_dir = kind / dataset
        if not dataset_dir.exists():
            continue
        _require_real_directory(dataset_dir, "dataset artifact lane")
        versions, loose_entries = [], []
        for child in sorted(dataset_dir.iterdir(), key=lambda path: path.name):
            metadata = child.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                identity_files = _manifest_provenance_files(child)
                versions.append({
                    "version": child.name,
                    "path": child.relative_to(root).as_posix(),
                    "manifest_provenance_files": identity_files,
                    "manifest_provenance_digest": _digest(identity_files),
                })
            else:
                loose_entries.append({
                    "name": child.name,
                    "type": ("symlink" if stat.S_ISLNK(metadata.st_mode)
                             else "file" if stat.S_ISREG(metadata.st_mode)
                             else "other"),
                    "size": metadata.st_size,
                })
        lane_identity_files = _manifest_provenance_files(dataset_dir)
        output.append({
            "kind": kind.name,
            "path": _relative(dataset_dir, root, "artifact path"),
            "versions": versions,
            "loose_entries": loose_entries,
            "manifest_provenance_files": lane_identity_files,
            "manifest_provenance_digest": _digest(lane_identity_files),
        })
    return output


def _mapillary_manifest(root: Path, dataset: str) -> dict[str, Any] | None:
    path = (root / "raw_material" / "mapillary_manifests"
            / f"{dataset}.json")
    if not path.exists():
        return None
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PreparationError(
            f"Mapillary source manifest must be a regular file: {path}")
    return {
        "path": _relative(path, root, "Mapillary source manifest"),
        "size": metadata.st_size,
        "sha256": _sha256_file(path),
    }


def _dataset_snapshot(root: Path, name: str, destination: Path | None = None
                      ) -> dict[str, Any]:
    dataset = root / "datasets" / name
    _require_real_directory(dataset, f"dataset {name!r}")
    records = _walk_tree(dataset)
    result = {
        "name": name,
        "source": _relative(dataset, root, "dataset source"),
        **_tree_summary(records),
        "critical_files": _critical_files(dataset),
        "artifact_lanes": _artifact_inventory(root, name),
        "mapillary_manifest": _mapillary_manifest(root, name),
    }
    if destination is not None:
        links = _symlink_plan(dataset, destination, root, records)
        result.update({
            "destination": _relative(destination, root,
                                     "dataset destination"),
            "links": links,
            "retired_tree_digest": _expected_retired_digest(records, links),
        })
    return result


def _top_level_directories(datasets: Path) -> set[str]:
    names = set()
    for entry in datasets.iterdir():
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PreparationError(
                f"top-level dataset entry must not be a symlink: {entry}")
        if stat.S_ISDIR(metadata.st_mode) and not entry.name.startswith("."):
            names.add(entry.name)
    return names


def _preflight_layout(root: Path, *, target_must_be_absent: bool) -> None:
    _require_real_directory(root, "data root")
    datasets = root / "datasets"
    artifacts = root / "artifacts"
    _require_real_directory(datasets, "datasets lane")
    _require_real_directory(artifacts, "artifacts lane")
    expected = (set(ACTIVE_DATASETS) | set(RETIRED_MAPILLARY_DATASETS)
                | set(UNTOUCHED_COLLECTIONS))
    actual = _top_level_directories(datasets)
    target = datasets / RETIREMENT_DIRNAME
    if target_must_be_absent and target.exists():
        raise PreparationError(f"retirement destination already exists: {target}")
    if not target_must_be_absent:
        actual.discard(RETIREMENT_DIRNAME)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise PreparationError(
            "top-level dataset inventory differs from the reviewed scope: "
            f"missing={missing}, unexpected={unexpected}")
    devices = {datasets.stat().st_dev}
    devices.update((datasets / name).stat().st_dev
                   for name in RETIRED_MAPILLARY_DATASETS)
    if len(devices) != 1:
        raise PreparationError(
            "retirement requires every source and destination parent on one "
            "filesystem; copy-and-delete fallback is forbidden")


def build_plan(data_root: Path, *, created: str | None = None,
               git_commit: str | None = None) -> dict[str, Any]:
    """Build the complete read-only baseline and retirement plan."""
    root = Path(data_root).resolve(strict=True)
    _preflight_layout(root, target_must_be_absent=True)
    destination = root / "datasets" / RETIREMENT_DIRNAME
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created": created or datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit or provenance.git_commit(),
        "data_root": str(root),
        "retirement_collection": _relative(
            destination, root, "retirement collection"),
        "active_datasets": [
            _dataset_snapshot(root, name) for name in ACTIVE_DATASETS
        ],
        "untouched_collections": [
            {
                "name": name,
                "path": f"datasets/{name}",
                **_tree_summary(_walk_tree(root / "datasets" / name)),
            }
            for name in UNTOUCHED_COLLECTIONS
        ],
        "retirements": [
            _dataset_snapshot(root, name, destination / name)
            for name in RETIRED_MAPILLARY_DATASETS
        ],
        "policy": {
            "move_method": "same_filesystem_rename_only",
            "artifacts": "inventory_only_never_modified",
            "overwrite": False,
            "delete": False,
            "copy": False,
        },
    }
    plan["plan_digest"] = _digest(plan)
    return plan


def _validate_plan(plan: Any) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise PreparationError(f"plan schema must be {PLAN_SCHEMA!r}")
    digest = plan.get("plan_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PreparationError("plan has no valid plan_digest")
    payload = dict(plan)
    payload.pop("plan_digest", None)
    if _digest(payload) != digest:
        raise PreparationError("plan_digest does not match plan content")
    names = [item.get("name") for item in plan.get("retirements", [])]
    if tuple(names) != RETIRED_MAPILLARY_DATASETS:
        raise PreparationError(
            "plan retirement set/order differs from the explicit allowlist")
    active = [item.get("name") for item in plan.get("active_datasets", [])]
    if tuple(active) != ACTIVE_DATASETS:
        raise PreparationError(
            "plan active set/order differs from the protected allowlist")
    untouched = [item.get("name")
                 for item in plan.get("untouched_collections", [])]
    if tuple(untouched) != UNTOUCHED_COLLECTIONS:
        raise PreparationError("plan does not protect the expected collections")
    if plan.get("retirement_collection") != (
            f"datasets/{RETIREMENT_DIRNAME}"):
        raise PreparationError("plan names a different retirement destination")
    for item, name in zip(plan["retirements"], RETIRED_MAPILLARY_DATASETS):
        if item.get("source") != f"datasets/{name}":
            raise PreparationError(
                f"plan names a different source for {name!r}")
        if item.get("destination") != (
                f"datasets/{RETIREMENT_DIRNAME}/{name}"):
            raise PreparationError(
                f"plan names a different destination for {name!r}")
        for link in item.get("links", []):
            relative = link.get("path")
            if not isinstance(relative, str):
                raise PreparationError(f"invalid link path for {name!r}")
            parsed = PurePosixPath(relative)
            if (parsed.is_absolute() or parsed.as_posix() != relative
                    or any(part in ("", ".", "..") for part in parsed.parts)):
                raise PreparationError(
                    f"unsafe link path for {name!r}: {relative!r}")


def _same_snapshot(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    keys = ("name", "source", "tree_digest", "entries", "regular_files",
            "directories", "symlinks", "regular_file_bytes",
            "critical_files", "artifact_lanes", "mapillary_manifest")
    return all(expected.get(key) == actual.get(key) for key in keys)


def _verify_plan_is_current(root: Path, plan: dict[str, Any]) -> None:
    _preflight_layout(root, target_must_be_absent=True)
    destination = root / "datasets" / RETIREMENT_DIRNAME
    for expected in plan["active_datasets"]:
        current = _dataset_snapshot(root, expected["name"])
        if not _same_snapshot(expected, current):
            raise PreparationError(
                f"active dataset changed after plan: {expected['name']}")
    for expected in plan["retirements"]:
        current = _dataset_snapshot(
            root, expected["name"], destination / expected["name"])
        if current != expected:
            raise PreparationError(
                f"retirement source changed after plan: {expected['name']}")
    for expected in plan["untouched_collections"]:
        current = {
            "name": expected["name"],
            "path": expected["path"],
            **_tree_summary(_walk_tree(root / expected["path"])),
        }
        if current != expected:
            raise PreparationError(
                f"protected collection changed after plan: "
                f"{expected['name']}")


def _atomic_symlink(link: Path, target: str) -> None:
    temporary = link.with_name(f".{link.name}.retirement-link.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise PreparationError(f"temporary link path already exists: {temporary}")
    temporary.symlink_to(target)
    try:
        os.replace(temporary, link)
        _fsync_directory(link.parent)
    except BaseException:
        if temporary.is_symlink():
            temporary.unlink()
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    """Flush every real directory entry without following dataset links."""
    directories = [root]
    for parent, child_names, _ in os.walk(root, followlinks=False):
        parent_path = Path(parent)
        for name in list(child_names):
            child = parent_path / name
            if child.is_symlink():
                child_names.remove(name)
            else:
                directories.append(child)
    for directory in sorted(directories, key=lambda path: len(path.parts),
                            reverse=True):
        _fsync_directory(directory)


def _write_journal(path: Path, *, plan: dict[str, Any], status: str,
                   moved: Iterable[str], rebased: Iterable[str],
                   error: str = "") -> None:
    artifact.atomic_write_json(path, {
        "schema": JOURNAL_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "status": status,
        "moved": list(moved),
        "rebased": list(rebased),
        "error": error,
    })
    _fsync_directory(path.parent)


def _durable_rename_noreplace(source: Path, destination: Path) -> None:
    """Rename without overwrite and durably publish both directory updates."""
    source_parent = source.parent
    destination_parent = destination.parent
    _rename_noreplace(source, destination)
    _fsync_directory(destination_parent)
    if source_parent != destination_parent:
        _fsync_directory(source_parent)


def _unretire_note(plan: dict[str, Any]) -> str:
    names = "\n".join(f"- `{item['name']}`" for item in plan["retirements"])
    return f"""# Out-of-date but usable Mapillary datasets

These {len(plan['retirements'])} datasets were retired from the active dataset
lane by a reviewed, rename-only transaction. Their source bytes were not
rewritten. Relative links that leave a dataset were rebased only so they still
resolve from this additional directory level.

{names}

`unvetted/london_thames` is intentionally not part of this collection.
`mt_washington_auto_road` is the old Mapillary collection, not one of the
three active 2026 Mount Washington recording legs.

## Unretiring one dataset

1. Review `retirement_manifest.json`, verify the dataset tree digest, and move
   the directory back to `datasets/<dataset>` with a guarded same-filesystem
   rename. Rebase external relative links to their recorded original targets.
2. Migrate `pipeline_metadata.json` and `intrinsics.csv` to the current camera
   frame contract; do not reinterpret an old fitted mount offset as a human
   nominal-forward annotation.
3. Human-approve nominal forward and review reverse-motion ranges for the
   dataset's fixed mounting/recording leg.
4. Build a fresh typed full map catalog with successful source coverage, then
   publish the required trim. Old catalog artifacts are regression evidence,
   not current typed inputs.
5. Re-attest preserved paid extraction only if its exact request, response,
   prompt, model, and image identities pass the importer; otherwise rerun only
   the failed keys.
6. Regenerate tracking, semantic audit, bearings, complete matching, identity
   review, diagnostics, localization inputs, localization, and viewers in
   that order, checking the stage report after each publication.

Artifact directories were deliberately left under `artifacts/<kind>/<dataset>`
and are listed in the manifest. They predate the current contracts and must
not be selected merely because they still exist.
"""


def _retirement_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": RETIREMENT_SCHEMA,
        "applied": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "generator": "farfield/dataset_tools/prepare_regeneration.py",
        "git_commit": provenance.git_commit(),
        "plan_digest": plan["plan_digest"],
        "source_plan_git_commit": plan["git_commit"],
        "retirement_collection": plan["retirement_collection"],
        "datasets": plan["retirements"],
        "protected_active_datasets": list(ACTIVE_DATASETS),
        "protected_untouched_collections": list(UNTOUCHED_COLLECTIONS),
        "artifacts_moved": False,
    }


@contextmanager
def _datasets_lock(datasets: Path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(datasets, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _restore_link(link: Path, original: str, retired: str) -> None:
    if not link.is_symlink():
        raise PreparationError(f"cannot roll back missing symlink: {link}")
    current = os.readlink(link)
    if current == original:
        return
    if current != retired:
        raise PreparationError(
            f"cannot roll back unexpectedly changed symlink {link}: {current}")
    _atomic_symlink(link, original)


def _rollback(root: Path, plan: dict[str, Any], staging: Path,
              journal: Path, cause: BaseException) -> None:
    errors = []
    # First restore every link while all moved datasets remain in staging.  A
    # single failed restoration leaves *all* dataset directories in staging;
    # returning even one tree to its shallower live path with archive-relative
    # link text would silently redirect its targets.
    for item in reversed(plan["retirements"]):
        moved = staging / item["name"]
        if moved.is_dir() and not moved.is_symlink():
            for link in reversed(item["links"]):
                try:
                    _restore_link(
                        moved / link["path"], link["original_target"],
                        link["retired_target"])
                except BaseException as exc:  # Preserve every recovery error.
                    errors.append(str(exc))
    if errors:
        _write_journal(
            journal, plan=plan, status="recovery_required", moved=(),
            rebased=(), error=f"{cause}; rollback errors={errors}")
        raise PreparationError(
            "retirement failed and link rollback was incomplete; no dataset "
            f"was returned to its live path: {errors}") from cause

    # Only after all link text is proven restorable may directories return to
    # their shallower source paths.
    for item in reversed(plan["retirements"]):
        moved = staging / item["name"]
        source = root / item["source"]
        if moved.is_dir() and not moved.is_symlink():
            if source.exists() or source.is_symlink():
                errors.append(f"rollback source is occupied: {source}")
            else:
                try:
                    _durable_rename_noreplace(moved, source)
                except BaseException as exc:
                    errors.append(str(exc))
    # Only generated transaction records may remain in staging at this point.
    if staging.is_dir():
        for name in ("retirement_manifest.json", "UNRETIRE.md"):
            generated = staging / name
            if generated.is_file() and not generated.is_symlink():
                generated.unlink()
                _fsync_directory(staging)
        try:
            staging.rmdir()
            _fsync_directory(staging.parent)
        except OSError as exc:
            errors.append(str(exc))
    status = "recovery_required" if errors else "rolled_back"
    _write_journal(journal, plan=plan, status=status, moved=(), rebased=(),
                   error=f"{cause}; rollback errors={errors}")
    if errors:
        raise PreparationError(
            "retirement failed and automatic rollback was incomplete; leave "
            f"the journal and staging tree untouched: {errors}") from cause


def apply_plan(plan: dict[str, Any], *, confirm_plan_digest: str,
               hook: Callable[[str, str], None] | None = None) -> Path:
    """Apply one exact reviewed plan; ``hook`` exists only for fault tests."""
    _validate_plan(plan)
    if confirm_plan_digest != plan["plan_digest"]:
        raise PreparationError(
            "--confirm_plan_digest does not match the reviewed plan")
    root = Path(plan["data_root"])
    if root.resolve(strict=True) != root:
        raise PreparationError("plan data_root must be a resolved real path")
    datasets = root / "datasets"
    target = datasets / RETIREMENT_DIRNAME
    prefix = plan["plan_digest"][:12]
    staging = datasets / f".{RETIREMENT_DIRNAME}.incomplete-{prefix}"
    journal = datasets / f".{RETIREMENT_DIRNAME}.journal-{prefix}.json"

    with _datasets_lock(datasets):
        if journal.exists() or staging.exists():
            raise PreparationError(
                "an earlier transaction record exists; inspect it before "
                f"retrying: journal={journal}, staging={staging}")
        _verify_plan_is_current(root, plan)
        staging.mkdir(mode=0o755)
        moved: list[str] = []
        rebased: list[str] = []
        _write_journal(journal, plan=plan, status="moving", moved=moved,
                       rebased=rebased)
        try:
            for item in plan["retirements"]:
                source = root / item["source"]
                destination = staging / item["name"]
                if hook:
                    hook("before_move", item["name"])
                _durable_rename_noreplace(source, destination)
                moved.append(item["name"])
                _write_journal(journal, plan=plan, status="moving",
                               moved=moved, rebased=rebased)

            for item in plan["retirements"]:
                dataset = staging / item["name"]
                for link in item["links"]:
                    if link["original_target"] == link["retired_target"]:
                        continue
                    label = f"{item['name']}/{link['path']}"
                    if hook:
                        hook("before_rebase", label)
                    path = dataset / link["path"]
                    if not path.is_symlink() or os.readlink(path) != (
                            link["original_target"]):
                        raise PreparationError(
                            f"symlink changed before rebase: {path}")
                    _atomic_symlink(path, link["retired_target"])
                    rebased.append(label)
                    _write_journal(journal, plan=plan, status="rebasing",
                                   moved=moved, rebased=rebased)

            for item in plan["retirements"]:
                dataset = staging / item["name"]
                snapshot = _tree_summary(_walk_tree(dataset))
                if snapshot["tree_digest"] != item["retired_tree_digest"]:
                    raise PreparationError(
                        f"retired tree differs after rename: {item['name']}")
                for link in item["links"]:
                    path = dataset / link["path"]
                    if _target_identity(path) != link["target_identity"]:
                        raise PreparationError(
                            f"symlink target identity changed: {path}")

            artifact.atomic_write_json(
                staging / "retirement_manifest.json",
                _retirement_manifest(plan))
            artifact.atomic_write_file(
                staging / "UNRETIRE.md", _unretire_note(plan).encode("utf-8"))
            _write_journal(journal, plan=plan, status="publishing",
                           moved=moved, rebased=rebased)
            if hook:
                hook("before_publish", RETIREMENT_DIRNAME)
            _fsync_directory_tree(staging)
            if target.exists() or target.is_symlink():
                raise PreparationError(
                    f"retirement destination appeared during apply: {target}")
            _durable_rename_noreplace(staging, target)
            # Preserve rather than delete the complete transaction record.
            _durable_rename_noreplace(
                journal, target / "transaction_journal.json")
            return target
        except BaseException as exc:
            if target.is_dir() and not target.is_symlink():
                # Publication itself succeeded. Never move a visible completed
                # collection backwards merely because the journal move failed.
                raise PreparationError(
                    f"retirement published at {target}, but final journal "
                    f"handling failed: {exc}") from exc
            _rollback(root, plan, staging, journal, exc)
            raise


def _load_plan(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"plan must be a regular non-symlink file: {path}")
    try:
        value = _strict_json_loads(
            path.read_bytes(), f"reviewed plan {path}")
    except OSError as exc:
        raise PreparationError(f"cannot read plan {path}: {exc}") from exc
    _validate_plan(value)
    return value


def _write_plan(path: Path, plan: dict[str, Any], root: Path) -> None:
    destination = path.resolve(strict=False)
    try:
        destination.relative_to(root.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise PreparationError(
            "read-only planning output must be outside the data root")
    if path.exists() or path.is_symlink():
        raise PreparationError(f"refusing to overwrite plan output: {path}")
    artifact.atomic_create_json(path, plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=None,
                        help=f"default: ${paths_lib.ROOT_ENV_VAR} or "
                             f"{paths_lib.DEFAULT_ROOT}")
    parser.add_argument("--output", type=Path,
                        help="optional plan output outside the data root")
    parser.add_argument("--apply_plan", type=Path,
                        help="apply this already-reviewed plan")
    parser.add_argument("--confirm_plan_digest",
                        help="required with --apply_plan")
    args = parser.parse_args(argv)

    try:
        if args.apply_plan:
            if args.output:
                parser.error("--output and --apply_plan are mutually exclusive")
            if not args.confirm_plan_digest:
                parser.error("--apply_plan requires --confirm_plan_digest")
            plan = _load_plan(args.apply_plan)
            if args.data_root is not None and Path(args.data_root).resolve() != (
                    Path(plan["data_root"])):
                parser.error("--data_root differs from the reviewed plan")
            destination = apply_plan(
                plan, confirm_plan_digest=args.confirm_plan_digest)
            print(f"published: {destination}")
            return 0

        if args.confirm_plan_digest:
            parser.error("--confirm_plan_digest requires --apply_plan")
        root = args.data_root or paths_lib.default_root()
        plan = build_plan(root)
        if args.output:
            _write_plan(args.output, plan, Path(plan["data_root"]))
            print(f"plan: {args.output}")
        else:
            json.dump(plan, sys.stdout, indent=2, sort_keys=True)
            print()
        print(f"plan_digest: {plan['plan_digest']}", file=sys.stderr)
        print("read-only plan; nothing moved", file=sys.stderr)
        return 0
    except PreparationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
