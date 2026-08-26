r"""Plan and apply the active far-field dataset contract migration.

This is deliberately narrower than a general migration framework.  It knows
the seven active self-collected datasets, the active public Pohang dataset,
and no others.  Planning is
read-only and is the default.  The plan binds every source file that will be
changed, the legacy ``landmarks/`` symlink tree, and the complete regenerated
checksum manifest.  Applying is a separate invocation which requires both
the saved plan and its digest::

    # Read-only; save the plan outside the data root.
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:\
migrate_active_datasets -- --output /tmp/active-dataset-migration.json

    # Only after reviewing the plan.
    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:\
migrate_active_datasets -- \\
      --apply_plan /tmp/active-dataset-migration.json \\
      --confirm_plan_digest <sha256 printed while planning>

The migration does not touch panorama pixels, frame/GPS tables, source video,
or derived artifacts.  For each active dataset it:

* archives the original ``pipeline_metadata.json``, ``intrinsics.csv``, any
  pre-existing ``checksums.sha256``, and the legacy ``landmarks/`` symlink
  directory, rebasing its links for the deeper archive path while recording
  the exact original text and resolved target content;
* stamps the exact current camera-frame convention and removes the old
  mount/course-derived heading authority;
* replaces the legacy intrinsics heading columns with the five current shape
  columns, all empty (no heading is inferred or relabelled);
* publishes a newly computed checksum manifest only after every other dataset
  mutation has completed.

The archive is published at
``archive/active_dataset_contract_migrations/<plan digest>/``.  Renames are
same-filesystem and no destination is overwritten.  An exception triggers an
automatic rollback.  If the process itself is interrupted, ``--rollback_plan``
uses the same reviewed plan to recover the ``.incomplete`` transaction without
guessing which side of a rename is authoritative.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import datetime
import errno
import fcntl
import hashlib
import io
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.dataset_tools import checksums


PLAN_SCHEMA = "farfield_active_dataset_contract_migration_plan/v1"
JOURNAL_SCHEMA = "farfield_active_dataset_contract_migration_transaction/v1"
GENERATOR = "farfield/dataset_tools/migrate_active_datasets.py"
ARCHIVE_PARENT = Path("archive/active_dataset_contract_migrations")

# Deliberately explicit.  Never infer migration targets as "everything not
# retired"; a newly collected directory must not become mutable by appearing
# under datasets/.
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

# Provenance is part of target identity, not a broad category accepted by the
# tool.  Pohang is a public third-party collect; relabelling it self-collected
# would lose its source/license meaning.
EXPECTED_SOURCE_BY_DATASET = {
    "boston_harbor_leg1": "self_collect",
    "boston_harbor_leg2": "self_collect",
    "boston_harbor_leg3": "self_collect",
    "charles_river_20260727": "self_collect",
    "mount_washington_20260815_leg1": "self_collect",
    "mount_washington_20260815_leg2": "self_collect",
    "mount_washington_20260815_leg3": "self_collect",
    "pohang_canal_04": "third_party_public",
}
SELF_COLLECT_AZIMUTH_FRAME = "camera (as captured)"
POHANG_AZIMUTH_FRAME = (
    "level-heading: gravity-stabilised, bow at column width/2")
EXPECTED_AZIMUTH_FRAME_BY_DATASET = {
    name: (POHANG_AZIMUTH_FRAME if name == "pohang_canal_04"
           else SELF_COLLECT_AZIMUTH_FRAME)
    for name in ACTIVE_DATASETS
}
EXPECTED_BEARING_INCREASES = "left_to_right"

METADATA_FILE = "pipeline_metadata.json"
INTRINSICS_FILE = "intrinsics.csv"
CHECKSUM_FILE = checksums.CHECKSUM_FILE
LEGACY_CHECKSUM_PREFIX = CHECKSUM_FILE + "."
LANDMARKS_DIR = "landmarks"

# These fields were written by the original self-collect ingest from GPS
# course and, sometimes, a fitted mount offset.  They are not human-approved
# physical orientation.  The original bytes remain in the migration archive.
LEGACY_METADATA_AUTHORITY_FIELDS = (
    "mount_offset",
    "heading_source",
    "heading_reliable",
    "heading_note",
)
LEGACY_AZIMUTH_AUTHORITY_FIELDS = (
    "heading_deg_is_bearing_of",
    "formula",
    "heading_per_frame",
    "mount_offset_frame",
    # These occur in intermediate contract versions.  A self-collected
    # dataset has no measured true-north camera heading, so references to
    # those columns would be an authority claim even though the columns below
    # are intentionally blank.
    "raw_mapillary_fields_reference",
    "selected_heading_per_frame",
    "column0_per_frame",
    "column0_from_optical_axis_formula",
    "world_bearing_formula",
)
LEGACY_INTRINSICS_HEADING_FIELDS = (
    "heading_deg",
    "heading_reference",
    "heading_source",
)
HEADING_SHAPE_FIELDS = (
    "computed_compass_angle_true_deg",
    "compass_angle_true_deg",
    "heading_optical_axis_true_deg",
    "heading_column0_true_deg",
    "selected_heading_source",
)

BOSTON_SOURCE_VIDEOS = {
    "boston_harbor_leg1": (
        "raw_material/boston_harbor_20260712/videos/"
        "long_wharf_to_hull_wharf.mp4 (not retained; ~38 GB originals)"),
    "boston_harbor_leg2": (
        "raw_material/boston_harbor_20260712/videos/"
        "hull_wharf_to_hingham_wharf.mp4 (not retained; ~38 GB originals)"),
    "boston_harbor_leg3": (
        "raw_material/boston_harbor_20260712/videos/"
        "hingham_wharf_to_rowes_wharf.mp4 (not retained; ~38 GB originals)"),
}
BOSTON_FALSE_NOT_RETAINED_SUFFIX = " (not retained; ~38 GB originals)"
POHANG_PENDING = [
    "landmarks/ (OSM catalogue for South Korea; no NOAA ENC coverage for "
    "Korean waters)",
    "checksums.sha256 (written at freeze time, after landmarks)",
]
POHANG_POST_INGEST_FIXUPS = [{
    "file": "intrinsics.csv",
    "column": "heading_source",
    "from": "gps_course_minus_mount_prior",
    "to": "slam_baseline_yaw_minus_calibrated_mount",
    "rows": 1450,
    "why": (
        "the generator infers this string from its own CLI shape and cannot "
        "express a calibrated mount offset or an attitude-derived heading"),
    "script": (
        "raw_material/pohang_canal_20210716/scripts/"
        "fix_heading_source.py"),
}]


class MigrationError(RuntimeError):
    """The reviewed plan or filesystem state does not permit a safe change."""


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
        raise MigrationError(f"{label} is invalid strict JSON: {exc}") from exc


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without the overwrite behavior of ``os.rename``.

    The data pipeline runs on Linux.  Requiring ``renameat2`` is preferable
    to emulating no-replace with a check-then-rename race, especially for a
    tool whose only purpose is safely changing the durable data root.
    """
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise MigrationError(
            "this platform lacks renameat2; refusing a non-atomic migration")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = function(
        at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination),
        rename_noreplace)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise MigrationError(
            f"refusing to overwrite rename destination: {destination}")
    raise OSError(error, os.strerror(error), str(source), str(destination))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_identity(payload: bytes) -> dict[str, Any]:
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _file_sha256(path: Path) -> str:
    return checksums.file_sha256(path)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise MigrationError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise MigrationError(
            f"{label} must be a real, non-symlink directory: {path}")


def _read_regular(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MigrationError(f"{label} must be a regular file: {path}")
    payload = path.read_bytes()
    identity = _bytes_identity(payload)
    identity["mode"] = stat.S_IMODE(metadata.st_mode)
    return payload, identity


def _optional_regular_identity(path: Path, label: str) -> dict[str, Any] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MigrationError(f"{label} must be a regular file: {path}")
    return {
        "size": metadata.st_size,
        "sha256": _file_sha256(path),
        "mode": stat.S_IMODE(metadata.st_mode),
    }


def _legacy_checksum_backups(dataset: Path, dataset_name: str) \
        -> dict[str, dict[str, Any]]:
    """Bind only direct-child files with the exact checksum backup prefix."""
    backups = {}
    for path in sorted(dataset.iterdir(), key=lambda item: item.name):
        if not path.name.startswith(LEGACY_CHECKSUM_PREFIX):
            continue
        # A matching symlink/directory is not silently ignored: a plan that
        # cannot archive the exact legacy backup shape must fail closed.
        identity = _optional_regular_identity(
            path, f"{dataset_name}/{path.name}")
        if identity is None:  # ``iterdir`` proved it existed; defensive only.
            raise MigrationError(f"legacy checksum backup vanished: {path}")
        backups[path.name] = identity
    return backups


def _parse_metadata(payload: bytes, dataset_name: str) -> dict[str, Any]:
    metadata = _strict_json_loads(
        payload, f"{dataset_name}/{METADATA_FILE}")
    if not isinstance(metadata, dict):
        raise MigrationError(
            f"{dataset_name}/{METADATA_FILE} must contain a JSON object")
    if metadata.get("dataset_name") != dataset_name:
        raise MigrationError(
            f"{dataset_name}/{METADATA_FILE}: dataset_name must be exactly "
            f"{dataset_name!r}, got {metadata.get('dataset_name')!r}")
    expected_source = EXPECTED_SOURCE_BY_DATASET[dataset_name]
    if metadata.get("source") != expected_source:
        raise MigrationError(
            f"{dataset_name}: source must be exactly {expected_source!r}, got "
            f"{metadata.get('source')!r}")
    if metadata.get("is_equirectangular") is not True:
        raise MigrationError(f"{dataset_name}: expected equirectangular imagery")
    if metadata.get("north_aligned") is not False:
        raise MigrationError(
            f"{dataset_name}: refuses to reinterpret north-aligned imagery")
    convention = metadata.get("azimuth_convention")
    if not isinstance(convention, dict):
        raise MigrationError(
            f"{dataset_name}: azimuth_convention must be an object")
    if convention.get("images_rotated") is not False:
        raise MigrationError(
            f"{dataset_name}: images_rotated must already be false")
    expected_frame = EXPECTED_AZIMUTH_FRAME_BY_DATASET[dataset_name]
    if convention.get("frame") != expected_frame:
        raise MigrationError(
            f"{dataset_name}: azimuth_convention.frame must already be "
            f"exactly {expected_frame!r}, got {convention.get('frame')!r}")
    if convention.get("bearing_increases") != EXPECTED_BEARING_INCREASES:
        raise MigrationError(
            f"{dataset_name}: azimuth_convention.bearing_increases must "
            f"already be exactly {EXPECTED_BEARING_INCREASES!r}, got "
            f"{convention.get('bearing_increases')!r}")
    return metadata


def _migrate_metadata(payload: bytes, dataset_name: str) \
        -> tuple[bytes, dict[str, Any]]:
    metadata = copy.deepcopy(_parse_metadata(payload, dataset_name))
    removed = []
    for field in LEGACY_METADATA_AUTHORITY_FIELDS:
        if field in metadata:
            metadata.pop(field)
            removed.append(field)

    convention = metadata["azimuth_convention"]
    for field in LEGACY_AZIMUTH_AUTHORITY_FIELDS:
        if field in convention:
            convention.pop(field)
            removed.append(f"azimuth_convention.{field}")
    convention["images_rotated"] = False
    preserved_convention = {
        "azimuth_convention.frame": convention["frame"],
        "azimuth_convention.bearing_increases":
            convention["bearing_increases"],
    }
    convention["camera_frame"] = geo.CAMERA_FRAME

    corrected: dict[str, Any] = {}
    if dataset_name in BOSTON_SOURCE_VIDEOS:
        video = metadata.get("video")
        if not isinstance(video, dict):
            raise MigrationError(f"{dataset_name}: video must be an object")
        expected = BOSTON_SOURCE_VIDEOS[dataset_name]
        if video.get("source_video") != expected:
            raise MigrationError(
                f"{dataset_name}: refuses to correct unexpected "
                f"video.source_video {video.get('source_video')!r}")
        if "retained" in video and video["retained"] is not False:
            raise MigrationError(
                f"{dataset_name}: refuses unexpected legacy video.retained "
                f"value {video['retained']!r}")
        corrected_path = expected.removesuffix(
            BOSTON_FALSE_NOT_RETAINED_SUFFIX)
        if corrected_path == expected:
            raise MigrationError(
                f"{dataset_name}: expected false not-retained suffix is absent")
        video["source_video"] = corrected_path
        video["retained"] = True
        corrected.update({
            "video.source_video": corrected_path,
            "video.retained": True,
        })

    if dataset_name == "pohang_canal_04":
        if metadata.get("pending") != POHANG_PENDING:
            raise MigrationError(
                f"{dataset_name}: pending differs from the exact stale list")
        if metadata.get("post_ingest_fixups") != POHANG_POST_INGEST_FIXUPS:
            raise MigrationError(
                f"{dataset_name}: post_ingest_fixups differs from the exact "
                "completed fixup")
        metadata.pop("pending")
        metadata.pop("post_ingest_fixups")
        removed.extend(("pending", "post_ingest_fixups"))

    # The original self-collect metadata omitted this even though the file
    # was always present.  This is a path correction, not a heading claim.
    existing_intrinsics = metadata.get("intrinsics_csv")
    if existing_intrinsics not in (None, INTRINSICS_FILE):
        raise MigrationError(
            f"{dataset_name}: refuses to replace unexpected intrinsics_csv "
            f"value {existing_intrinsics!r}")
    metadata["intrinsics_csv"] = INTRINSICS_FILE

    migrated = (json.dumps(metadata, indent=2, ensure_ascii=False,
                           allow_nan=False) + "\n").encode("utf-8")
    _validate_migrated_metadata(migrated, dataset_name)
    set_values = {
        "azimuth_convention.camera_frame": geo.CAMERA_FRAME,
        "intrinsics_csv": INTRINSICS_FILE,
        **corrected,
    }
    return migrated, {
        "removed_authority_paths": sorted(removed),
        "set": set_values,
        "preserved": preserved_convention,
    }


def _validate_migrated_metadata(payload: bytes, dataset_name: str) -> None:
    metadata = _parse_metadata(payload, dataset_name)
    for field in LEGACY_METADATA_AUTHORITY_FIELDS:
        if field in metadata:
            raise MigrationError(
                f"{dataset_name}: migrated metadata retained {field}")
    convention = metadata["azimuth_convention"]
    for field in LEGACY_AZIMUTH_AUTHORITY_FIELDS:
        if field in convention:
            raise MigrationError(
                f"{dataset_name}: migrated convention retained {field}")
    if convention.get("camera_frame") != geo.CAMERA_FRAME:
        raise MigrationError(
            f"{dataset_name}: migrated metadata lacks canonical camera frame")
    if metadata.get("intrinsics_csv") != INTRINSICS_FILE:
        raise MigrationError(
            f"{dataset_name}: migrated metadata lacks intrinsics_csv")
    if dataset_name in BOSTON_SOURCE_VIDEOS:
        video = metadata.get("video")
        expected = BOSTON_SOURCE_VIDEOS[dataset_name].removesuffix(
            BOSTON_FALSE_NOT_RETAINED_SUFFIX)
        if (not isinstance(video, dict)
                or video.get("source_video") != expected
                or video.get("retained") is not True):
            raise MigrationError(
                f"{dataset_name}: migrated metadata lacks exact retained video")
    if dataset_name == "pohang_canal_04":
        if "pending" in metadata or "post_ingest_fixups" in metadata:
            raise MigrationError(
                f"{dataset_name}: migrated metadata retained stale status")
        if convention.get("bearing_increases") != "left_to_right":
            raise MigrationError(
                f"{dataset_name}: migrated bearing convention changed")


def _source_video_identity(data_root: Path, metadata_payload: bytes,
                           dataset_name: str) -> dict[str, Any]:
    """Bind the normalized in-root retained source video used by later stages."""
    metadata = _parse_metadata(metadata_payload, dataset_name)
    video = metadata.get("video")
    if not isinstance(video, dict):
        raise MigrationError(f"{dataset_name}: video must be an object")
    raw = video.get("source_video")
    if not isinstance(raw, str):
        raise MigrationError(
            f"{dataset_name}: video.source_video must be a string")
    parsed = PurePosixPath(raw)
    if (parsed.is_absolute() or parsed.as_posix() != raw
            or any(part in ("", ".", "..") for part in parsed.parts)):
        raise MigrationError(
            f"{dataset_name}: video.source_video is not a normalized "
            f"root-relative path: {raw!r}")
    candidate = data_root.joinpath(*parsed.parts)
    try:
        metadata_stat = candidate.lstat()
    except FileNotFoundError as exc:
        raise MigrationError(
            f"{dataset_name}: retained source video is missing: {raw}") from exc
    if stat.S_ISLNK(metadata_stat.st_mode) or not stat.S_ISREG(
            metadata_stat.st_mode):
        raise MigrationError(
            f"{dataset_name}: retained source video must be a regular "
            f"non-symlink file: {raw}")
    root_resolved = data_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise MigrationError(
            f"{dataset_name}: retained source video escapes data root: {raw}") \
            from exc
    return {
        "path": raw,
        "size": metadata_stat.st_size,
        "mode": stat.S_IMODE(metadata_stat.st_mode),
        "sha256": _file_sha256(candidate),
    }


def _parse_csv(payload: bytes, dataset_name: str) \
        -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or not reader.fieldnames:
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} has no header")
    fields = list(reader.fieldnames)
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} has duplicate/empty columns")
    rows = list(reader)
    if not rows:
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} has no data rows")
    if any(None in row for row in rows):
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} has rows wider than its header")
    if "idx" not in fields:
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} lacks idx")
    try:
        indices = [int(row["idx"]) for row in rows]
    except (TypeError, ValueError) as exc:
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} has invalid idx") from exc
    if indices != list(range(len(rows))):
        raise MigrationError(
            f"{dataset_name}/{INTRINSICS_FILE} idx is not contiguous 0..N-1")
    return fields, rows


def _migrate_intrinsics(payload: bytes, dataset_name: str) \
        -> tuple[bytes, dict[str, Any]]:
    fields, rows = _parse_csv(payload, dataset_name)
    heading_like = set(LEGACY_INTRINSICS_HEADING_FIELDS) | set(
        HEADING_SHAPE_FIELDS)
    positions = [index for index, field in enumerate(fields)
                 if field in heading_like]
    insert_at = min(positions) if positions else len(fields)
    kept_fields = [field for field in fields if field not in heading_like]
    insert_at = min(insert_at, len(kept_fields))
    output_fields = (kept_fields[:insert_at] + list(HEADING_SHAPE_FIELDS)
                     + kept_fields[insert_at:])

    nonempty_replaced = {
        field: sum(bool((row.get(field) or "").strip()) for row in rows)
        for field in fields if field in heading_like
    }
    output_rows = []
    for row in rows:
        migrated = {field: row.get(field, "") for field in kept_fields}
        for field in HEADING_SHAPE_FIELDS:
            migrated[field] = ""
        output_rows.append(migrated)

    sink = io.StringIO(newline="")
    writer = csv.DictWriter(
        sink, fieldnames=output_fields, lineterminator="\n",
        extrasaction="raise")
    writer.writeheader()
    writer.writerows(output_rows)
    migrated_payload = sink.getvalue().encode("utf-8")
    _validate_migrated_intrinsics(migrated_payload, dataset_name)
    return migrated_payload, {
        "rows": len(rows),
        "removed_columns": [field for field in fields
                            if field in LEGACY_INTRINSICS_HEADING_FIELDS],
        "heading_shape_columns": list(HEADING_SHAPE_FIELDS),
        "nonempty_heading_values_archived_then_cleared": nonempty_replaced,
    }


def _validate_migrated_intrinsics(payload: bytes, dataset_name: str) -> None:
    fields, rows = _parse_csv(payload, dataset_name)
    missing = [field for field in HEADING_SHAPE_FIELDS if field not in fields]
    legacy = [field for field in LEGACY_INTRINSICS_HEADING_FIELDS
              if field in fields]
    if missing or legacy:
        raise MigrationError(
            f"{dataset_name}: migrated intrinsics missing={missing}, "
            f"legacy={legacy}")
    populated = [
        (index, field) for index, row in enumerate(rows)
        for field in HEADING_SHAPE_FIELDS if (row.get(field) or "").strip()
    ]
    if populated:
        raise MigrationError(
            f"{dataset_name}: migrated heading fields are not empty: "
            f"{populated[:3]}")


def _landmarks_inventory(path: Path, dataset_name: str, data_root: Path,
                         archive_path: Path) -> list[dict[str, Any]]:
    """Plan a relocatable, content-bound legacy landmark symlink tree."""
    _require_real_directory(path, f"{dataset_name}/{LANDMARKS_DIR}")
    root_resolved = data_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    pending = [path]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            rel = child.relative_to(path).as_posix()
            metadata = child.lstat()
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                original_target = os.readlink(child)
                if Path(original_target).is_absolute():
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR} contains an absolute "
                        f"symlink: {child} -> {original_target}")
                try:
                    resolved = child.resolve(strict=True)
                except (FileNotFoundError, RuntimeError) as exc:
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR} contains a dangling "
                        f"or cyclic symlink: {child} -> {original_target}") \
                        from exc
                try:
                    resolved_relative = resolved.relative_to(root_resolved)
                except ValueError as exc:
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR} symlink escapes data "
                        f"root: {child} -> {resolved}") from exc
                target_stat = resolved.lstat()
                if not stat.S_ISREG(target_stat.st_mode):
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR} symlink must resolve "
                        f"to a regular file: {child} -> {resolved}")
                archived_link = archive_path / rel
                archive_target = os.path.relpath(
                    resolved, start=archived_link.parent)
                records.append({
                    "path": rel,
                    "type": "symlink",
                    "mode": stat.S_IMODE(mode),
                    "original_target": original_target,
                    "archive_target": archive_target,
                    "resolved_target": resolved_relative.as_posix(),
                    "target_identity": {
                        "size": target_stat.st_size,
                        "mode": stat.S_IMODE(target_stat.st_mode),
                        "sha256": _file_sha256(resolved),
                    },
                })
            elif stat.S_ISDIR(mode):
                records.append({
                    "path": rel,
                    "type": "directory",
                    "mode": stat.S_IMODE(mode),
                })
                pending.append(child)
            else:
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR} is expected to contain "
                    f"only directories and symlinks; refusing real entry {child}")
    if not any(record["type"] == "symlink" for record in records):
        raise MigrationError(
            f"{dataset_name}/{LANDMARKS_DIR} has no legacy symlinks")
    return sorted(records, key=lambda record: record["path"])


def _validate_landmarks_tree(
        path: Path, dataset_name: str, data_root: Path,
        expected: dict[str, Any], *, location: str,
        allow_mixed_archive_targets: bool = False) -> None:
    """Validate exact tree shape, link text, and resolved target bytes."""
    _require_real_directory(path, f"{dataset_name}/{LANDMARKS_DIR}")
    if location not in ("live", "archive"):
        raise ValueError(f"unsupported landmarks location {location!r}")
    expected_records = expected["records"]
    expected_by_path = {record["path"]: record
                        for record in expected_records}
    actual_paths: set[str] = set()
    pending = [path]
    while pending:
        directory = pending.pop()
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            rel = child.relative_to(path).as_posix()
            actual_paths.add(rel)
            record = expected_by_path.get(rel)
            if record is None:
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR} gained {rel}")
            metadata = child.lstat()
            if stat.S_IMODE(metadata.st_mode) != record["mode"]:
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR}/{rel} mode changed")
            if record["type"] == "directory":
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
                        metadata.st_mode):
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR}/{rel} changed type")
                pending.append(child)
                continue
            if not stat.S_ISLNK(metadata.st_mode):
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR}/{rel} changed type")
            raw_target = os.readlink(child)
            desired = (record["original_target"] if location == "live"
                       else record["archive_target"])
            permitted = {desired}
            if location == "archive" and allow_mixed_archive_targets:
                permitted.add(record["original_target"])
            if raw_target not in permitted:
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR}/{rel} has unexpected "
                    f"target {raw_target!r}")

            resolved = data_root / record["resolved_target"]
            if not _file_matches(resolved, record["target_identity"]):
                raise MigrationError(
                    f"{dataset_name}/{LANDMARKS_DIR}/{rel} resolved target "
                    "content changed")
            # Original link text is expected to be temporarily dangling from
            # the deeper archive path.  Every other accepted state must itself
            # resolve to the content-bound target.
            if not (location == "archive"
                    and raw_target == record["original_target"]):
                try:
                    actual_resolved = child.resolve(strict=True)
                except (FileNotFoundError, RuntimeError) as exc:
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR}/{rel} is dangling") \
                        from exc
                if actual_resolved != resolved.resolve(strict=True):
                    raise MigrationError(
                        f"{dataset_name}/{LANDMARKS_DIR}/{rel} resolves to an "
                        "unexpected path")
    expected_paths = set(expected_by_path)
    if actual_paths != expected_paths:
        raise MigrationError(
            f"{dataset_name}/{LANDMARKS_DIR} paths changed: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}")
    if _digest(expected_records) != expected["tree_digest"]:
        raise MigrationError(
            f"{dataset_name}/{LANDMARKS_DIR} reviewed tree digest is invalid")


def _checksum_manifest(dataset: Path,
                       replacements: dict[str, bytes]) -> bytes:
    """Build the canonical manifest without writing into *dataset*."""
    entries: dict[str, str] = {}
    for path in dataset.rglob("*"):
        rel = path.relative_to(dataset)
        if not rel.parts:
            continue
        if rel.parts[0] in ("panorama", LANDMARKS_DIR):
            continue
        is_legacy_checksum_backup = (
            len(rel.parts) == 1
            and rel.name.startswith(LEGACY_CHECKSUM_PREFIX))
        if (rel.name == CHECKSUM_FILE or is_legacy_checksum_backup
                or set(rel.parts) & checksums.EXCLUDED_DIRS):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise MigrationError(
                f"unsupported filesystem entry while checksumming: {path}")
        key = rel.as_posix()
        if key in replacements:
            entries[key] = hashlib.sha256(replacements[key]).hexdigest()
        else:
            entries[key] = _file_sha256(path)
    for rel, payload in replacements.items():
        if rel == CHECKSUM_FILE:
            continue
        entries.setdefault(rel, hashlib.sha256(payload).hexdigest())
    lines = [f"{digest}  ./{rel}\n" for rel, digest in sorted(
        entries.items(), key=lambda item: ("./" + item[0]).encode())]
    return "".join(lines).encode("utf-8")


def _dataset_plan(data_root: Path, dataset_name: str) -> dict[str, Any]:
    dataset = data_root / "datasets" / dataset_name
    _require_real_directory(dataset, f"active dataset {dataset_name}")

    metadata_payload, metadata_source = _read_regular(
        dataset / METADATA_FILE, f"{dataset_name}/{METADATA_FILE}")
    intrinsics_payload, intrinsics_source = _read_regular(
        dataset / INTRINSICS_FILE, f"{dataset_name}/{INTRINSICS_FILE}")
    old_checksum = _optional_regular_identity(
        dataset / CHECKSUM_FILE, f"{dataset_name}/{CHECKSUM_FILE}")
    old_checksum_backups = _legacy_checksum_backups(dataset, dataset_name)
    metadata_output, metadata_changes = _migrate_metadata(
        metadata_payload, dataset_name)
    source_video = _source_video_identity(
        data_root, metadata_output, dataset_name)
    intrinsics_output, intrinsics_changes = _migrate_intrinsics(
        intrinsics_payload, dataset_name)
    replacements = {
        METADATA_FILE: metadata_output,
        INTRINSICS_FILE: intrinsics_output,
    }
    checksum_output = _checksum_manifest(dataset, replacements)
    archive_landmarks = (data_root / ARCHIVE_PARENT / "_plan_digest_"
                         / "datasets" / dataset_name / LANDMARKS_DIR)
    landmarks = _landmarks_inventory(
        dataset / LANDMARKS_DIR, dataset_name, data_root,
        archive_landmarks)

    metadata_output_identity = _bytes_identity(metadata_output)
    metadata_output_identity["mode"] = metadata_source["mode"]
    intrinsics_output_identity = _bytes_identity(intrinsics_output)
    intrinsics_output_identity["mode"] = intrinsics_source["mode"]
    checksum_output_identity = _bytes_identity(checksum_output)
    checksum_output_identity.update({
        "mode": (old_checksum["mode"] if old_checksum is not None else 0o644),
        "entries": checksum_output.count(b"\n"),
    })

    return {
        "dataset": dataset_name,
        "source": {
            METADATA_FILE: metadata_source,
            INTRINSICS_FILE: intrinsics_source,
            CHECKSUM_FILE: old_checksum,
            "legacy_checksum_backups": old_checksum_backups,
            "source_video": source_video,
            LANDMARKS_DIR: {
                "tree_digest": _digest(landmarks),
                "entries": len(landmarks),
                "symlinks": sum(record["type"] == "symlink"
                                for record in landmarks),
                "records": landmarks,
            },
        },
        "output": {
            METADATA_FILE: metadata_output_identity,
            INTRINSICS_FILE: intrinsics_output_identity,
            CHECKSUM_FILE: checksum_output_identity,
            LANDMARKS_DIR: None,
        },
        "changes": {
            "metadata": metadata_changes,
            "intrinsics": intrinsics_changes,
            "checksums": {
                "action": "regenerated after all other mutations",
                "legacy_backups_archived": sorted(old_checksum_backups),
            },
            "landmarks": (
                "moved to the content-addressed archive with exact original "
                "link text recorded and archive-relative link text rebased"),
        },
    }


def build_plan(data_root: Path) -> dict[str, Any]:
    """Inspect the exact active set and return a deterministic reviewed plan."""
    data_root = Path(data_root).resolve(strict=True)
    _require_real_directory(data_root, "data root")
    _require_real_directory(data_root / "datasets", "datasets lane")
    _require_real_directory(data_root / "archive", "archive lane")
    datasets = [_dataset_plan(data_root, name) for name in ACTIVE_DATASETS]
    unsigned = {
        "schema": PLAN_SCHEMA,
        "generator": GENERATOR,
        "generator_git_commit": provenance.git_commit(),
        "data_root": str(data_root),
        "archive_parent": ARCHIVE_PARENT.as_posix(),
        "active_datasets": list(ACTIVE_DATASETS),
        "datasets": datasets,
    }
    return {**unsigned, "plan_digest": _digest(unsigned)}


def _validate_plan_document(plan: Any) -> str:
    if not isinstance(plan, dict):
        raise MigrationError("reviewed plan must contain a JSON object")
    digest = plan.get("plan_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MigrationError("reviewed plan has no valid plan_digest")
    unsigned = dict(plan)
    unsigned.pop("plan_digest", None)
    actual = _digest(unsigned)
    if actual != digest:
        raise MigrationError(
            f"reviewed plan digest mismatch: recorded {digest}, actual {actual}")
    if plan.get("schema") != PLAN_SCHEMA:
        raise MigrationError(
            f"unsupported reviewed plan schema {plan.get('schema')!r}")
    if plan.get("generator") != GENERATOR:
        raise MigrationError("reviewed plan names a different generator")
    if plan.get("active_datasets") != list(ACTIVE_DATASETS):
        raise MigrationError("reviewed plan does not name the exact active set")
    entries = plan.get("datasets")
    entry_names = ([item.get("dataset") for item in entries
                    if isinstance(item, dict)]
                   if isinstance(entries, list) else None)
    if entry_names != list(ACTIVE_DATASETS):
        raise MigrationError(
            "reviewed plan dataset entries do not match the exact active set")
    if plan.get("archive_parent") != ARCHIVE_PARENT.as_posix():
        raise MigrationError("reviewed plan names an unexpected archive lane")
    return digest


def load_reviewed_plan(path: Path, confirmed_digest: str) -> dict[str, Any]:
    payload, _ = _read_regular(Path(path), "reviewed plan")
    plan = _strict_json_loads(payload, "reviewed plan")
    digest = _validate_plan_document(plan)
    if confirmed_digest != digest:
        raise MigrationError(
            "--confirm_plan_digest must exactly match the reviewed plan: "
            f"expected {digest}, got {confirmed_digest!r}")
    return plan


def _assert_plan_still_current(plan: dict[str, Any]) -> Path:
    data_root = Path(plan["data_root"])
    current = build_plan(data_root)
    if _canonical_bytes(current) != _canonical_bytes(plan):
        raise MigrationError(
            "the data tree or generator changed after this plan was reviewed; "
            "generate and review a new plan")
    return data_root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_parents_durable(path: Path) -> None:
    """Create a real directory chain and fsync every new parent entry."""
    if path.exists() or path.is_symlink():
        _require_real_directory(path, "transaction directory")
        return
    _mkdir_parents_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        pass
    _require_real_directory(path, "transaction directory")
    _fsync_directory(path.parent)


def _write_new_file(path: Path, payload: bytes, mode: int = 0o600) -> None:
    _mkdir_parents_durable(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _replace_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if temporary.exists() or temporary.is_symlink():
        raise MigrationError(f"transaction journal temporary exists: {temporary}")
    _write_new_file(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _durable_rename_noreplace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    destination_parent = destination.parent
    _rename_noreplace(source, destination)
    _fsync_directory(destination_parent)
    if source_parent != destination_parent:
        _fsync_directory(source_parent)


def _replace_symlink(path: Path, expected: str, replacement: str) -> None:
    if not path.is_symlink() or os.readlink(path) != expected:
        raise MigrationError(
            f"refusing to replace unexpectedly changed symlink: {path}")
    temporary = path.with_name(f".{path.name}.migration-link.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise MigrationError(f"temporary symlink already exists: {temporary}")
    temporary.symlink_to(replacement)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.is_symlink():
            temporary.unlink()
            _fsync_directory(path.parent)
        raise


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


@contextmanager
def _migration_lock(data_root: Path) -> Iterator[None]:
    datasets = data_root / "datasets"
    descriptor = os.open(
        datasets, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _transaction_paths(data_root: Path, digest: str) -> tuple[Path, Path]:
    parent = data_root / ARCHIVE_PARENT
    return parent / f".{digest}.incomplete", parent / digest


def _stage_transaction(plan: dict[str, Any], transaction: Path) -> dict[str, Any]:
    transaction.mkdir(mode=0o700)
    _fsync_directory(transaction.parent)
    _write_new_file(
        transaction / "plan.json",
        json.dumps(plan, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    _write_new_file(
        transaction / "README.md",
        ("# Active dataset contract migration archive\n\n"
         f"Plan digest: `{plan['plan_digest']}`\n\n"
         "`datasets/<name>/` contains the exact pre-migration metadata, "
         "intrinsics, and any checksum manifest that existed. The legacy "
         "landmarks symlink tree is relocated with its exact original link "
         "text and target identities in `plan.json`; published archive links "
         "are rebased only for the deeper location. Datasets whose old "
         "checksum was absent are "
         "identified in `plan.json`; absence is not silently rewritten as an "
         "old file.\n").encode("utf-8"))

    data_root = Path(plan["data_root"])
    for item in plan["datasets"]:
        name = item["dataset"]
        dataset = data_root / "datasets" / name
        metadata_payload = (dataset / METADATA_FILE).read_bytes()
        intrinsics_payload = (dataset / INTRINSICS_FILE).read_bytes()
        metadata_output, _ = _migrate_metadata(metadata_payload, name)
        intrinsics_output, _ = _migrate_intrinsics(intrinsics_payload, name)
        checksum_output = _checksum_manifest(dataset, {
            METADATA_FILE: metadata_output,
            INTRINSICS_FILE: intrinsics_output,
        })
        outputs = {
            METADATA_FILE: metadata_output,
            INTRINSICS_FILE: intrinsics_output,
            CHECKSUM_FILE: checksum_output,
        }
        for filename, payload in outputs.items():
            if _bytes_identity(payload) != {
                    key: value for key, value in item["output"][filename].items()
                    if key in ("size", "sha256")}:
                raise MigrationError(
                    f"{name}/{filename}: staged output does not match plan")
            _write_new_file(
                transaction / ".staged" / name / filename, payload,
                mode=item["output"][filename]["mode"])

    journal = {
        "schema": JOURNAL_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "status": "staged",
        "started_at": _utc_now(),
        "renames": [],
        "rebased": [],
    }
    _replace_json(transaction / "transaction.json", journal)
    return journal


def _rename_no_overwrite(source: Path, destination: Path,
                         journal: dict[str, Any], journal_path: Path,
                         fail_after_renames: int | None) -> None:
    if not (source.exists() or source.is_symlink()):
        raise MigrationError(f"rename source vanished: {source}")
    if destination.exists() or destination.is_symlink():
        raise MigrationError(f"refusing to overwrite rename destination: "
                             f"{destination}")
    _mkdir_parents_durable(destination.parent)
    if (fail_after_renames is not None
            and len(journal["renames"]) >= fail_after_renames):
        raise OSError("injected migration failure")
    _durable_rename_noreplace(source, destination)
    journal["renames"].append({"source": str(source),
                               "destination": str(destination)})
    _replace_json(journal_path, journal)


def _file_matches(path: Path, identity: dict[str, Any]) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return (stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
            and metadata.st_size == identity["size"]
            and ("mode" not in identity
                 or stat.S_IMODE(metadata.st_mode) == identity["mode"])
            and _file_sha256(path) == identity["sha256"])


def _move_aside(path: Path, failed: Path, name: str) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    destination = failed / name
    if destination.exists() or destination.is_symlink():
        raise MigrationError(
            f"rollback refuses to overwrite diagnostic output: {destination}")
    _mkdir_parents_durable(destination.parent)
    _durable_rename_noreplace(path, destination)


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validate_rollback_sources(plan: dict[str, Any], transaction: Path) -> None:
    """Prove every recovery source before any live file is overwritten."""
    data_root = Path(plan["data_root"])
    errors = []
    for item in plan["datasets"]:
        name = item["dataset"]
        dataset = data_root / "datasets" / name
        archived = transaction / "datasets" / name
        try:
            video = item["source"]["source_video"]
            if not _file_matches(data_root / video["path"], video):
                raise MigrationError(
                    f"retained source video changed for {name}")
            for filename in (METADATA_FILE, INTRINSICS_FILE, CHECKSUM_FILE):
                source_identity = item["source"][filename]
                output_identity = item["output"][filename]
                live = dataset / filename
                original = archived / filename
                if _path_present(original):
                    if source_identity is None or not _file_matches(
                            original, source_identity):
                        raise MigrationError(
                            f"archive cannot prove {name}/{filename}")
                    if _path_present(live) and not _file_matches(
                            live, output_identity):
                        raise MigrationError(
                            f"live replacement is not reviewed output for "
                            f"{name}/{filename}")
                elif source_identity is None:
                    if _path_present(live) and not _file_matches(
                            live, output_identity):
                        raise MigrationError(
                            f"unexpected live file for originally absent "
                            f"{name}/{filename}")
                elif not _file_matches(live, source_identity):
                    raise MigrationError(
                        f"original {name}/{filename} is not provable on "
                        "either side")

            for filename, source_identity in item["source"][
                    "legacy_checksum_backups"].items():
                live = dataset / filename
                original = archived / filename
                if _path_present(original):
                    if not _file_matches(original, source_identity):
                        raise MigrationError(
                            f"archive cannot prove {name}/{filename}")
                    if _path_present(live):
                        raise MigrationError(
                            f"legacy checksum backup exists on both sides for "
                            f"{name}/{filename}")
                elif not _file_matches(live, source_identity):
                    raise MigrationError(
                        f"original {name}/{filename} is not provable")

            live_landmarks = dataset / LANDMARKS_DIR
            archived_landmarks = archived / LANDMARKS_DIR
            if _path_present(live_landmarks) == _path_present(
                    archived_landmarks):
                raise MigrationError(
                    f"landmarks must exist on exactly one side for {name}")
            if _path_present(archived_landmarks):
                _validate_landmarks_tree(
                    archived_landmarks, name, data_root,
                    item["source"][LANDMARKS_DIR], location="archive",
                    allow_mixed_archive_targets=True)
            else:
                _validate_landmarks_tree(
                    live_landmarks, name, data_root,
                    item["source"][LANDMARKS_DIR], location="live")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise MigrationError(
            "rollback prevalidation failed before live overwrite: "
            + "; ".join(errors))


def _restore_archived_landmark_links(
        archived_landmarks: Path, item: dict[str, Any]) -> None:
    for record in reversed(item["source"][LANDMARKS_DIR]["records"]):
        if record["type"] != "symlink":
            continue
        link = archived_landmarks / record["path"]
        current = os.readlink(link) if link.is_symlink() else None
        if current == record["original_target"]:
            continue
        _replace_symlink(
            link, record["archive_target"], record["original_target"])


def _validate_restored(plan: dict[str, Any], transaction: Path) -> None:
    data_root = Path(plan["data_root"])
    for item in plan["datasets"]:
        name = item["dataset"]
        dataset = data_root / "datasets" / name
        archived = transaction / "datasets" / name
        for filename in (METADATA_FILE, INTRINSICS_FILE, CHECKSUM_FILE):
            expected = item["source"][filename]
            live = dataset / filename
            if expected is None:
                if _path_present(live):
                    raise MigrationError(
                        f"rollback invented {name}/{filename}")
            elif not _file_matches(live, expected):
                raise MigrationError(
                    f"rollback did not restore {name}/{filename}")
            if _path_present(archived / filename):
                raise MigrationError(
                    f"rollback left archived {name}/{filename}")
        for filename, expected in item["source"][
                "legacy_checksum_backups"].items():
            if not _file_matches(dataset / filename, expected):
                raise MigrationError(
                    f"rollback did not restore {name}/{filename}")
            if _path_present(archived / filename):
                raise MigrationError(
                    f"rollback left archived {name}/{filename}")
        _validate_landmarks_tree(
            dataset / LANDMARKS_DIR, name, data_root,
            item["source"][LANDMARKS_DIR], location="live")
        if _path_present(archived / LANDMARKS_DIR):
            raise MigrationError(
                f"rollback left archived {name}/{LANDMARKS_DIR}")


def _rollback_incomplete(plan: dict[str, Any], transaction: Path,
                         reason: str) -> None:
    """Restore the exact reviewed source state, retaining failed new files."""
    data_root = Path(plan["data_root"])
    failed = transaction / ".failed_new"
    journal_path = transaction / "transaction.json"
    try:
        _validate_rollback_sources(plan, transaction)
    except Exception as exc:
        journal = {
            "schema": JOURNAL_SCHEMA,
            "plan_digest": plan["plan_digest"],
            "status": "rollback_failed",
            "rolled_back_at": _utc_now(),
            "reason": reason,
            "errors": [str(exc)],
        }
        _replace_json(journal_path, journal)
        raise

    # Restore every archived link to its original live-relative text before
    # returning any landmark tree or overwriting any live file.
    try:
        for item in reversed(plan["datasets"]):
            archived_landmarks = (transaction / "datasets" / item["dataset"]
                                  / LANDMARKS_DIR)
            if _path_present(archived_landmarks):
                _restore_archived_landmark_links(archived_landmarks, item)
    except Exception as exc:
        journal = {
            "schema": JOURNAL_SCHEMA,
            "plan_digest": plan["plan_digest"],
            "status": "rollback_failed",
            "rolled_back_at": _utc_now(),
            "reason": reason,
            "errors": [f"landmark link restoration failed: {exc}"],
        }
        _replace_json(journal_path, journal)
        raise MigrationError(
            "landmark link restoration failed before live overwrite") from exc

    errors = []
    for item in reversed(plan["datasets"]):
        name = item["dataset"]
        dataset = data_root / "datasets" / name
        archived = transaction / "datasets" / name
        try:
            archived_landmarks = archived / LANDMARKS_DIR
            live_landmarks = dataset / LANDMARKS_DIR
            if archived_landmarks.exists() or archived_landmarks.is_symlink():
                if live_landmarks.exists() or live_landmarks.is_symlink():
                    raise MigrationError(
                        f"rollback found landmarks on both sides for {name}")
                _mkdir_parents_durable(live_landmarks.parent)
                _durable_rename_noreplace(
                    archived_landmarks, live_landmarks)

            for filename in (CHECKSUM_FILE, INTRINSICS_FILE, METADATA_FILE):
                live = dataset / filename
                original = archived / filename
                source_identity = item["source"][filename]
                if original.exists() or original.is_symlink():
                    _move_aside(live, failed / name, filename)
                    _durable_rename_noreplace(original, live)
                elif source_identity is None:
                    # The dataset originally had no checksum.  Any new one is
                    # diagnostic output from the failed transaction.
                    _move_aside(live, failed / name, filename)
                elif not _file_matches(live, source_identity):
                    raise MigrationError(
                        f"rollback cannot prove original {name}/{filename}")
            for filename, source_identity in reversed(list(
                    item["source"]["legacy_checksum_backups"].items())):
                live = dataset / filename
                original = archived / filename
                if original.exists() or original.is_symlink():
                    _move_aside(live, failed / name, filename)
                    _durable_rename_noreplace(original, live)
                elif not _file_matches(live, source_identity):
                    raise MigrationError(
                        f"rollback cannot prove original {name}/{filename}")
        except Exception as exc:  # Continue restoring other datasets.
            errors.append(f"{name}: {exc}")

    if not errors:
        try:
            _validate_restored(plan, transaction)
        except Exception as exc:
            errors.append(f"restored-state validation: {exc}")
    journal = {
        "schema": JOURNAL_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "status": "rollback_failed" if errors else "rolled_back",
        "rolled_back_at": _utc_now(),
        "reason": reason,
        "errors": errors,
    }
    _replace_json(journal_path, journal)
    if errors:
        raise MigrationError("rollback was incomplete: " + "; ".join(errors))


def _validate_applied(plan: dict[str, Any], transaction: Path) -> None:
    data_root = Path(plan["data_root"])
    for item in plan["datasets"]:
        name = item["dataset"]
        dataset = data_root / "datasets" / name
        video = item["source"]["source_video"]
        if not _file_matches(data_root / video["path"], video):
            raise MigrationError(f"retained source video changed for {name}")
        if (dataset / LANDMARKS_DIR).exists() \
                or (dataset / LANDMARKS_DIR).is_symlink():
            raise MigrationError(f"{name}: landmarks remained in dataset")
        metadata_payload, _ = _read_regular(
            dataset / METADATA_FILE, f"{name}/{METADATA_FILE}")
        intrinsics_payload, _ = _read_regular(
            dataset / INTRINSICS_FILE, f"{name}/{INTRINSICS_FILE}")
        checksum_payload, _ = _read_regular(
            dataset / CHECKSUM_FILE, f"{name}/{CHECKSUM_FILE}")
        _validate_migrated_metadata(metadata_payload, name)
        _validate_migrated_intrinsics(intrinsics_payload, name)
        for filename, payload in (
                (METADATA_FILE, metadata_payload),
                (INTRINSICS_FILE, intrinsics_payload),
                (CHECKSUM_FILE, checksum_payload)):
            expected = item["output"][filename]
            if (_bytes_identity(payload)["sha256"] != expected["sha256"]
                    or len(payload) != expected["size"]):
                raise MigrationError(
                    f"{name}/{filename}: published bytes differ from plan")
            published_mode = stat.S_IMODE(
                (dataset / filename).lstat().st_mode)
            if published_mode != expected["mode"]:
                raise MigrationError(
                    f"{name}/{filename}: published mode differs from plan")
        recomputed = _checksum_manifest(dataset, {
            METADATA_FILE: metadata_payload,
            INTRINSICS_FILE: intrinsics_payload,
        })
        if recomputed != checksum_payload:
            raise MigrationError(
                f"{name}/{CHECKSUM_FILE}: published manifest does not verify")
        archived = transaction / "datasets" / name
        for filename in (METADATA_FILE, INTRINSICS_FILE):
            if not _file_matches(archived / filename,
                                 item["source"][filename]):
                raise MigrationError(
                    f"archive does not preserve {name}/{filename}")
        old_checksum = item["source"][CHECKSUM_FILE]
        archived_checksum = archived / CHECKSUM_FILE
        if old_checksum is None:
            if archived_checksum.exists() or archived_checksum.is_symlink():
                raise MigrationError(
                    f"archive invented old checksum for {name}")
        elif not _file_matches(archived_checksum, old_checksum):
            raise MigrationError(
                f"archive does not preserve {name}/{CHECKSUM_FILE}")
        for filename, source_identity in item["source"][
                "legacy_checksum_backups"].items():
            if (dataset / filename).exists() or (dataset / filename).is_symlink():
                raise MigrationError(
                    f"{name}: legacy checksum backup remained live: {filename}")
            if not _file_matches(archived / filename, source_identity):
                raise MigrationError(
                    f"archive does not preserve {name}/{filename}")
        _validate_landmarks_tree(
            archived / LANDMARKS_DIR, name, data_root,
            item["source"][LANDMARKS_DIR], location="archive")


def _rebase_archived_landmarks(
        data_root: Path, archived: Path, item: dict[str, Any],
        journal: dict[str, Any], journal_path: Path,
        fail_after_rebases: int | None) -> None:
    name = item["dataset"]
    landmarks = archived / LANDMARKS_DIR
    for record in item["source"][LANDMARKS_DIR]["records"]:
        if record["type"] != "symlink":
            continue
        if (fail_after_rebases is not None
                and len(journal["rebased"]) >= fail_after_rebases):
            raise OSError("injected landmark rebase failure")
        link = landmarks / record["path"]
        _replace_symlink(
            link, record["original_target"], record["archive_target"])
        journal["rebased"].append({
            "dataset": name,
            "path": record["path"],
        })
        _replace_json(journal_path, journal)
    _validate_landmarks_tree(
        landmarks, name, data_root, item["source"][LANDMARKS_DIR],
        location="archive")


def apply_reviewed_plan(
        plan: dict[str, Any], *, fail_after_renames: int | None = None,
        fail_after_rebases: int | None = None) -> Path:
    """Apply *plan*; failure counters exist only for transaction tests."""
    digest = _validate_plan_document(plan)
    data_root = Path(plan["data_root"])
    with _migration_lock(data_root):
        _assert_plan_still_current(plan)
        parent = data_root / ARCHIVE_PARENT
        _mkdir_parents_durable(parent)
        _require_real_directory(parent, "migration archive parent")
        _fsync_directory(parent)
        _fsync_directory(parent.parent)
        transaction, final_archive = _transaction_paths(data_root, digest)
        if transaction.exists() or transaction.is_symlink():
            raise MigrationError(
                f"incomplete transaction exists; inspect or roll it back: "
                f"{transaction}")
        if final_archive.exists() or final_archive.is_symlink():
            raise MigrationError(
                f"refusing to overwrite migration archive: {final_archive}")

        journal: dict[str, Any] | None = None
        try:
            journal = _stage_transaction(plan, transaction)
            journal_path = transaction / "transaction.json"
            journal["status"] = "committing"
            _replace_json(journal_path, journal)

            # Phase one: archive old contract files, publish metadata and
            # intrinsics, and relocate legacy landmarks.  Checksums are
            # intentionally absent during this phase rather than lying about
            # partially migrated content.
            for item in plan["datasets"]:
                name = item["dataset"]
                dataset = data_root / "datasets" / name
                archive = transaction / "datasets" / name
                staged = transaction / ".staged" / name
                for filename in (METADATA_FILE, INTRINSICS_FILE):
                    _rename_no_overwrite(
                        dataset / filename, archive / filename,
                        journal, journal_path, fail_after_renames)
                    _rename_no_overwrite(
                        staged / filename, dataset / filename,
                        journal, journal_path, fail_after_renames)
                if item["source"][CHECKSUM_FILE] is not None:
                    _rename_no_overwrite(
                        dataset / CHECKSUM_FILE, archive / CHECKSUM_FILE,
                        journal, journal_path, fail_after_renames)
                for filename in item["source"]["legacy_checksum_backups"]:
                    _rename_no_overwrite(
                        dataset / filename, archive / filename,
                        journal, journal_path, fail_after_renames)
                _rename_no_overwrite(
                    dataset / LANDMARKS_DIR, archive / LANDMARKS_DIR,
                    journal, journal_path, fail_after_renames)
                _rebase_archived_landmarks(
                    data_root, archive, item, journal, journal_path,
                    fail_after_rebases)

            # Phase two and the last dataset mutation: publish every newly
            # generated checksum manifest.
            for item in plan["datasets"]:
                name = item["dataset"]
                _rename_no_overwrite(
                    transaction / ".staged" / name / CHECKSUM_FILE,
                    data_root / "datasets" / name / CHECKSUM_FILE,
                    journal, journal_path, fail_after_renames)

            _validate_applied(plan, transaction)
            journal["status"] = "committed"
            journal["committed_at"] = _utc_now()
            _replace_json(journal_path, journal)
            # Only empty staging directories remain.
            for name in ACTIVE_DATASETS:
                (transaction / ".staged" / name).rmdir()
                _fsync_directory(transaction / ".staged")
            (transaction / ".staged").rmdir()
            _fsync_directory(transaction)
            _durable_rename_noreplace(transaction, final_archive)
            return final_archive
        except Exception as exc:
            if transaction.exists() and journal is not None:
                try:
                    _rollback_incomplete(plan, transaction, str(exc))
                except Exception as rollback_exc:
                    raise MigrationError(
                        f"migration failed ({exc}); rollback also failed "
                        f"({rollback_exc}); preserve {transaction} for "
                        "manual recovery") from rollback_exc
            raise


def rollback_reviewed_plan(plan: dict[str, Any]) -> Path:
    """Recover an interrupted/incomplete apply, never a committed archive."""
    digest = _validate_plan_document(plan)
    data_root = Path(plan["data_root"])
    with _migration_lock(data_root):
        transaction, final_archive = _transaction_paths(data_root, digest)
        if final_archive.exists() or final_archive.is_symlink():
            raise MigrationError(
                f"migration is committed; refusing rollback: {final_archive}")
        _require_real_directory(transaction, "incomplete transaction")
        _rollback_incomplete(plan, transaction, "explicit rollback request")
        return transaction


def _write_plan_output(path: Path, plan: dict[str, Any]) -> None:
    data_root = Path(plan["data_root"]).resolve(strict=True)
    destination = Path(path).absolute()
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise MigrationError(
            "reviewed plan must be saved outside the data root")
    _mkdir_parents_durable(resolved_destination.parent)
    resolved_parent = resolved_destination.parent.resolve(strict=True)
    resolved_destination = resolved_parent / destination.name
    try:
        resolved_destination.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise MigrationError(
            "reviewed plan must be saved outside the data root")
    _write_new_file(
        resolved_destination,
        json.dumps(plan, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_root", type=Path, default=None,
        help=f"default: ${paths_lib.ROOT_ENV_VAR} or {paths_lib.DEFAULT_ROOT}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply_plan", type=Path,
                      help="apply this previously saved reviewed plan")
    mode.add_argument("--rollback_plan", type=Path,
                      help="recover this plan's interrupted transaction")
    parser.add_argument("--confirm_plan_digest",
                        help="required exact digest for apply/rollback")
    parser.add_argument("--output", type=Path,
                        help="planning only: save with no overwrite")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.apply_plan or args.rollback_plan:
        if args.output:
            parser.error("--output is only valid while planning")
        if not args.confirm_plan_digest:
            parser.error("apply/rollback requires --confirm_plan_digest")
        plan_path = args.apply_plan or args.rollback_plan
        plan = load_reviewed_plan(plan_path, args.confirm_plan_digest)
        if args.data_root is not None:
            requested = str(args.data_root.resolve(strict=True))
            if requested != plan["data_root"]:
                parser.error(
                    f"--data_root {requested} differs from reviewed plan "
                    f"{plan['data_root']}")
        if args.apply_plan:
            archive = apply_reviewed_plan(plan)
            print(f"migration committed: {archive}")
        else:
            transaction = rollback_reviewed_plan(plan)
            print(f"migration rolled back; diagnostics retained: {transaction}")
        return 0

    if args.confirm_plan_digest:
        parser.error("--confirm_plan_digest is only valid for apply/rollback")
    data_root = args.data_root or paths_lib.default_root()
    plan = build_plan(data_root)
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        _write_plan_output(args.output, plan)
        print(f"reviewed plan written: {args.output}")
        print(f"plan digest: {plan['plan_digest']}")
    else:
        sys.stdout.write(rendered)
        print(f"plan digest: {plan['plan_digest']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
