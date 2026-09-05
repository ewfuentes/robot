"""Human review and explicit authoring for nominal-forward calibration.

The default command creates evidence only: an annotated contact sheet, a
content-bound manifest, and a deliberately non-authoritative record template.
It never proposes an angle and cannot write an approved calibration.  The
``finalize`` subcommand is a separate, explicit human-approval boundary.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import dataclasses
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

from experimental.overhead_matching.swag.farfield import (
    dataset as dataset_lib,
    geometry,
    nominal_forward,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import checksums


BUNDLE_SCHEMA = "farfield_nominal_forward_review_bundle/v1"
TEMPLATE_SCHEMA = "farfield_nominal_forward_review_template/v1"
DEFAULT_EVIDENCE_COUNT = 5
DEFAULT_GRID_STEP_DEG = 30.0
DEFAULT_DISPLAY_WIDTH = 1600
NOMINAL_FORWARD_NAME = "nominal_forward.json"
_TRANSACTION_SCHEMA = "farfield_nominal_forward_finalize_transaction/v1"
_TRANSACTION_RELATIVE = (Path("_manifests") /
                         "nominal_forward_finalize_transaction")
_REVIEW_DIGEST_PREFIX = "review_manifest_sha256="
_HEADING_FIELDS = (
    "computed_compass_angle_true_deg",
    "compass_angle_true_deg",
    "heading_optical_axis_true_deg",
    "heading_column0_true_deg",
    "selected_heading_source",
)


class ReviewError(RuntimeError):
    """Dataset or requested review operation is unsafe or inconsistent."""


@dataclasses.dataclass(frozen=True)
class PanoramaEvidence:
    pano_id: str
    pano_stem: str
    path: Path
    width: int
    height: int
    size: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class ValidatedDataset:
    base: Path
    name: str
    width: int
    height: int
    panoramas: tuple[PanoramaEvidence, ...]
    source_digests: dict[str, str]
    dataset_digest: str


def _duplicate_safe_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ReviewError(f"non-finite JSON constant {value!r} is forbidden")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(document) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReviewError(f"{label} must be a regular, non-symlink file: {path}")


def _load_metadata(base: Path) -> dict:
    path = base / "pipeline_metadata.json"
    _require_regular_file(path, "pipeline metadata")
    try:
        metadata = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read strict metadata {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ReviewError(f"{path}: root must be an object")
    if metadata.get("dataset_name") != base.name:
        raise ReviewError(
            f"{path}: dataset_name must equal directory name {base.name!r}")
    if metadata.get("intrinsics_csv") != "intrinsics.csv":
        raise ReviewError(
            f"{path}: intrinsics_csv must be exactly 'intrinsics.csv'")
    try:
        dataset_lib.require_camera_frame_panoramas(metadata, base)
    except dataset_lib.ContractViolation as exc:
        raise ReviewError(str(exc)) from exc
    return metadata


def _validate_panorama_root(base: Path) -> Path:
    panorama = base / "panorama"
    if panorama.is_symlink():
        target = Path(os.readlink(panorama))
        if target.is_absolute():
            raise ReviewError(f"{panorama}: absolute symlink is forbidden")
        try:
            resolved = panorama.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ReviewError(f"{panorama}: dangling or cyclic symlink") from exc
        try:
            resolved.relative_to(base.resolve(strict=True))
        except ValueError as exc:
            raise ReviewError(f"{panorama}: symlink escapes the dataset") from exc
        if not resolved.is_dir():
            raise ReviewError(f"{panorama}: symlink target is not a directory")
        return resolved
    if not panorama.is_dir():
        raise ReviewError(f"{panorama}: panorama directory does not exist")
    return panorama.resolve(strict=True)


def _strict_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    _require_regular_file(path, label)
    try:
        payload = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ReviewError(f"{path}: must be valid UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(payload, newline=""), strict=True)
    fields = list(reader.fieldnames or ())
    if (not fields or len(fields) != len(set(fields))
            or any(not item for item in fields)):
        raise ReviewError(f"{path}: header has missing or duplicate fields")
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ReviewError(f"{path}: invalid CSV: {exc}") from exc
    if not rows or any(None in row for row in rows):
        raise ReviewError(f"{path}: requires non-empty, header-width rows")
    return fields, rows


def _positive_integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"{label} must be an integer") from exc
    if result <= 0 or str(result) != str(value).strip():
        raise ReviewError(f"{label} must be a canonical positive integer")
    return result


def validate_dataset(dataset_base: Path) -> ValidatedDataset:
    """Fully validate and content-bind a post-migration camera-frame dataset."""
    base = Path(dataset_base)
    if base.is_symlink() or not base.is_dir():
        raise ReviewError(f"dataset must be a regular, non-symlink directory: {base}")
    base = base.resolve(strict=True)
    metadata = _load_metadata(base)
    panorama_root = _validate_panorama_root(base)
    frames_path = base / "frames_gps.csv"
    _require_regular_file(frames_path, "frames GPS table")
    try:
        frames = dataset_lib.load_frames(base)
    except (dataset_lib.ContractViolation, OSError, ValueError) as exc:
        raise ReviewError(f"dataset frame contract failed: {exc}") from exc
    if not frames:
        raise ReviewError(f"{base}: dataset has no panorama frames")

    expected_paths = {frame.pano_stem + ".jpg" for frame in frames}
    entries = list(panorama_root.iterdir())
    unexpected = sorted(
        entry.name for entry in entries
        if entry.name not in expected_paths or entry.name.startswith("."))
    if unexpected:
        raise ReviewError(
            f"{panorama_root}: entries outside the exact JPEG frame set: "
            f"{unexpected[:10]}")
    if len(entries) != len(expected_paths):
        raise ReviewError(f"{panorama_root}: duplicate or missing frame entries")

    panorama_records = []
    dimensions = set()
    for frame in frames:
        path = base / "panorama" / f"{frame.pano_stem}.jpg"
        _require_regular_file(path, f"panorama {frame.pano_id}")
        try:
            with Image.open(path) as opened:
                if opened.format != "JPEG":
                    raise ReviewError(f"{path}: content is not JPEG")
                dimensions.add(opened.size)
                opened.verify()
        except ReviewError:
            raise
        except Exception as exc:
            raise ReviewError(f"{path}: corrupt panorama JPEG: {exc}") from exc
        width, height = next(iter(dimensions)) if len(dimensions) == 1 else (0, 0)
        panorama_records.append(PanoramaEvidence(
            pano_id=frame.pano_id,
            pano_stem=frame.pano_stem,
            path=path,
            width=width,
            height=height,
            size=path.stat().st_size,
            sha256=_sha256(path),
        ))
    if len(dimensions) != 1:
        raise ReviewError(f"{base}: panorama dimensions disagree: {sorted(dimensions)}")
    width, height = next(iter(dimensions))
    if width != 2 * height:
        raise ReviewError(
            f"{base}: panoramas must have exact 2:1 equirectangular shape, "
            f"got {width}x{height}")
    # Replace provisional dimensions now that agreement is established.
    panorama_records = [dataclasses.replace(item, width=width, height=height)
                        for item in panorama_records]

    intrinsics_path = base / "intrinsics.csv"
    fields, rows = _strict_csv(intrinsics_path, "intrinsics table")
    required = {"idx", "pano_id", "projection", "width", "height", *_HEADING_FIELDS}
    missing = sorted(required.difference(fields))
    if missing:
        raise ReviewError(f"{intrinsics_path}: missing required fields {missing}")
    if len(rows) != len(frames):
        raise ReviewError(
            f"{intrinsics_path}: {len(rows)} rows for {len(frames)} panoramas")
    for index, (row, frame) in enumerate(zip(rows, frames)):
        try:
            row_index = int(row["idx"])
        except (TypeError, ValueError) as exc:
            raise ReviewError(
                f"{intrinsics_path}: invalid idx at row {index + 2}") from exc
        if row_index != index or row["pano_id"] != frame.pano_id:
            raise ReviewError(
                f"{intrinsics_path}: row {index + 2} does not join frame "
                f"{frame.pano_id}")
        if row["projection"] != "equirectangular":
            raise ReviewError(
                f"{intrinsics_path}: row {index + 2} projection is not equirectangular")
        if (_positive_integer(
                row["width"], f"intrinsics row {index + 2} width") != width
                or _positive_integer(
                    row["height"],
                    f"intrinsics row {index + 2} height") != height):
            raise ReviewError(
                f"{intrinsics_path}: row {index + 2} dimensions disagree with JPEGs")
        populated = [field for field in _HEADING_FIELDS if row[field].strip()]
        if populated and metadata.get("source") != "mapillary":
            raise ReviewError(
                f"{intrinsics_path}: row {index + 2} contains unapproved "
                f"heading authority in {populated}")
        if (metadata.get("source") == "mapillary" and populated
                and len(populated) != len(_HEADING_FIELDS)):
            raise ReviewError(
                f"{intrinsics_path}: row {index + 2} must fully populate or "
                "fully omit Mapillary orientation diagnostics")

    source_digests = {
        "pipeline_metadata.json": _sha256(base / "pipeline_metadata.json"),
        "frames_gps.csv": _sha256(frames_path),
        "intrinsics.csv": _sha256(intrinsics_path),
    }
    identity = {
        "dataset": base.name,
        "width": width,
        "height": height,
        "sources": source_digests,
        "panoramas": [{
            "pano_id": item.pano_id,
            "pano_stem": item.pano_stem,
            "size": item.size,
            "sha256": item.sha256,
        } for item in panorama_records],
    }
    return ValidatedDataset(
        base=base,
        name=base.name,
        width=width,
        height=height,
        panoramas=tuple(panorama_records),
        source_digests=source_digests,
        dataset_digest=_json_digest(identity),
    )


def select_evidence(validated: ValidatedDataset, *,
                    evidence_frame_ids: tuple[str, ...] = (),
                    evidence_count: int = DEFAULT_EVIDENCE_COUNT) \
        -> tuple[tuple[PanoramaEvidence, ...], str]:
    by_id = {item.pano_id: item for item in validated.panoramas}
    if evidence_frame_ids:
        if len(set(evidence_frame_ids)) != len(evidence_frame_ids):
            raise ReviewError("explicit evidence frame IDs must be unique")
        missing = [item for item in evidence_frame_ids if item not in by_id]
        if missing:
            raise ReviewError(f"unknown evidence frame IDs: {missing}")
        return tuple(by_id[item] for item in evidence_frame_ids), "explicit"
    if (isinstance(evidence_count, bool)
            or not isinstance(evidence_count, int) or evidence_count <= 0):
        raise ReviewError("evidence_count must be a positive integer")
    count = min(evidence_count, len(validated.panoramas))
    if count == 1:
        indices = [len(validated.panoramas) // 2]
    else:
        last = len(validated.panoramas) - 1
        indices = [int(math.floor(index * last / (count - 1) + 0.5))
                   for index in range(count)]
    return (tuple(validated.panoramas[index] for index in indices),
            "deterministic_evenly_spaced_v1")


def column_grid(width: int, step_deg: float = DEFAULT_GRID_STEP_DEG) -> list[dict]:
    if (not math.isfinite(step_deg) or step_deg <= 0.0
            or step_deg > 180.0):
        raise ReviewError("grid_step_deg must be finite and in (0, 180]")
    count = 360.0 / step_deg
    if abs(count - round(count)) > 1e-12:
        raise ReviewError("grid_step_deg must divide 360 exactly")
    by_column = {}
    for index in range(int(round(count))):
        target = index * step_deg
        x, _ = geometry.pano_px_from_direction(target, 0.0, width, 1)
        column = int(math.floor(x + 0.5)) % width
        bearing = float(geometry.azimuth_of_pano_column(column, width)) % 360.0
        by_column[column] = {
            "panorama_column": column,
            "bearing_camera_cw_deg": bearing,
        }
    return [by_column[column] for column in sorted(by_column)]


def _render_contact_sheet(evidence: tuple[PanoramaEvidence, ...], grid: list[dict],
                          display_width: int) -> Image.Image:
    if display_width < 600:
        raise ReviewError("display_width must be at least 600 pixels")
    source_width, source_height = evidence[0].width, evidence[0].height
    display_height = int(round(display_width * source_height / source_width))
    header_height = 72
    panel_height = header_height + display_height
    sheet = Image.new("RGB", (display_width, panel_height * len(evidence)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for panel, item in enumerate(evidence):
        top = panel * panel_height
        with Image.open(item.path) as opened:
            image = opened.convert("RGB").resize(
                (display_width, display_height), resampling)
        sheet.paste(image, (0, top + header_height))
        draw.text((6, top + 3),
                  f"{item.pano_id} | {item.pano_stem} | "
                  f"source {source_width}x{source_height}",
                  fill="black", font=font)
        for marker_index, marker in enumerate(grid):
            column = marker["panorama_column"]
            x = min(display_width - 1,
                    int(round(column * display_width / source_width)))
            color = (255, 45, 45) if marker_index % 2 == 0 else (35, 220, 255)
            draw.line((x, top + header_height, x, top + panel_height - 1),
                      fill=color, width=2)
            label = (f"x={column}\n"
                     f"{marker['bearing_camera_cw_deg']:.3f} deg CW")
            label_y = top + 20 + 24 * (marker_index % 2)
            box = draw.multiline_textbbox((0, 0), label, font=font, spacing=0)
            label_width = box[2] - box[0]
            label_x = max(0, min(display_width - label_width - 2,
                                 x - label_width // 2))
            draw.multiline_text((label_x, label_y), label, fill="black",
                                font=font, spacing=0, align="center")
    return sheet


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise ReviewError(
            "platform lacks renameat2; refusing non-atomic bundle publication")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100,
                      os.fsencode(destination), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error_number, os.strerror(error_number), str(destination))


def create_review_bundle(dataset_base: Path, output_dir: Path, *,
                         evidence_frame_ids: tuple[str, ...] = (),
                         evidence_count: int = DEFAULT_EVIDENCE_COUNT,
                         grid_step_deg: float = DEFAULT_GRID_STEP_DEG,
                         display_width: int = DEFAULT_DISPLAY_WIDTH) -> dict:
    """Create an atomic, no-overwrite evidence bundle with no angle proposal."""
    validated = validate_dataset(dataset_base)
    evidence, selection = select_evidence(
        validated, evidence_frame_ids=evidence_frame_ids,
        evidence_count=evidence_count)
    grid = column_grid(validated.width, grid_step_deg)
    destination = Path(output_dir)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ReviewError(f"output parent must be a regular directory: {parent}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    staging = parent / f".{destination.name}.incomplete-{secrets.token_hex(8)}"
    staging.mkdir(mode=0o755)
    try:
        sheet = _render_contact_sheet(evidence, grid, display_width)
        sink = io.BytesIO()
        sheet.save(sink, format="PNG", compress_level=9)
        _write_file(staging / "contact_sheet.png", sink.getvalue())
        template = {
            "schema": TEMPLATE_SCHEMA,
            "calibration_authority": False,
            "nominal_forward_schema": nominal_forward.SCHEMA,
            "nominal_forward_frame": nominal_forward.FRAME,
            "dataset": validated.name,
            "panorama_width": validated.width,
            "evidence_frame_ids_reviewed": [item.pano_id for item in evidence],
            "human_must_enter_explicitly_at_finalize": [
                "version", "mounting_id", "bearing_camera_cw_deg OR panorama_column",
                "uncertainty_deg", "operator", "approved_at", "notes",
                "evidence_frame_ids",
            ],
            "version": None,
            "mounting_id": None,
            "bearing_camera_cw_deg": None,
            "panorama_column": None,
            "uncertainty_deg": None,
            "operator": None,
            "approved_at": None,
            "notes": None,
            "evidence_frame_ids": None,
            "warning": (
                "This is review evidence, not calibration authority. The "
                "finalize command never imports a candidate angle from this bundle."),
        }
        template_payload = (json.dumps(
            template, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False) + "\n").encode("utf-8")
        _write_file(staging / "record_template.json", template_payload)
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "calibration_authority": False,
            "dataset": validated.name,
            "dataset_digest": validated.dataset_digest,
            "source_digests": validated.source_digests,
            "panorama": {
                "count": len(validated.panoramas),
                "width": validated.width,
                "height": validated.height,
            },
            "selection": {
                "method": selection,
                "evidence_frame_ids": [item.pano_id for item in evidence],
            },
            "evidence": [{
                "pano_id": item.pano_id,
                "pano_stem": item.pano_stem,
                "size": item.size,
                "sha256": item.sha256,
            } for item in evidence],
            "column_grid": grid,
            "outputs": {
                "contact_sheet.png": _sha256(staging / "contact_sheet.png"),
                "record_template.json": hashlib.sha256(template_payload).hexdigest(),
            },
            "safety": {
                "contains_approved_nominal_forward": False,
                "diagnostic_candidate_angle_copied": False,
            },
        }
        payload = (json.dumps(
            manifest, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False) + "\n").encode("utf-8")
        _write_file(staging / "review_manifest.json", payload)
        _fsync_directory(staging)
        _rename_directory_no_replace(staging, destination)
        _fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def _contains_approved_true(value) -> bool:
    if isinstance(value, dict):
        return (value.get("approved") is True
                or any(_contains_approved_true(item)
                       for item in value.values()))
    if isinstance(value, list):
        return any(_contains_approved_true(item) for item in value)
    return False


def validate_review_manifest(validated: ValidatedDataset,
                             review_manifest: Path,
                             evidence_frame_ids: tuple[str, ...]) -> dict:
    """Bind final approval to the exact current bundle and reviewed frames."""
    path = Path(review_manifest)
    _require_regular_file(path, "review manifest")
    raw_payload = path.read_bytes()
    try:
        manifest = json.loads(
            raw_payload.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid review manifest {path}: {exc}") from exc
    fields = {
        "schema", "calibration_authority", "dataset", "dataset_digest",
        "source_digests", "panorama", "selection", "evidence",
        "column_grid", "outputs", "safety",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ReviewError("review manifest fields do not match its v1 schema")
    if (manifest["schema"] != BUNDLE_SCHEMA
            or manifest["calibration_authority"] is not False
            or _contains_approved_true(manifest)):
        raise ReviewError(
            "review manifest must be non-authoritative bundle evidence")
    if (manifest["dataset"] != validated.name
            or manifest["dataset_digest"] != validated.dataset_digest
            or manifest["source_digests"] != validated.source_digests):
        raise ReviewError(
            "review manifest dataset/content identity is no longer current")
    if manifest["panorama"] != {
            "count": len(validated.panoramas),
            "width": validated.width,
            "height": validated.height,
    }:
        raise ReviewError("review manifest panorama identity is no longer current")

    selection = manifest["selection"]
    if (not isinstance(selection, dict)
            or set(selection) != {"method", "evidence_frame_ids"}
            or selection["method"] not in {
                "explicit", "deterministic_evenly_spaced_v1"}
            or selection["evidence_frame_ids"] != list(evidence_frame_ids)):
        raise ReviewError(
            "explicit evidence_frame_ids must exactly match the reviewed bundle")
    by_id = {item.pano_id: item for item in validated.panoramas}
    expected_evidence = []
    for pano_id in evidence_frame_ids:
        item = by_id.get(pano_id)
        if item is None:
            raise ReviewError(f"review evidence frame is no longer present: {pano_id}")
        expected_evidence.append({
            "pano_id": item.pano_id,
            "pano_stem": item.pano_stem,
            "size": item.size,
            "sha256": item.sha256,
        })
    if manifest["evidence"] != expected_evidence:
        raise ReviewError("review manifest evidence bytes are no longer current")

    grid = manifest["column_grid"]
    if not isinstance(grid, list) or not grid:
        raise ReviewError("review manifest column grid must be non-empty")
    columns = []
    for marker in grid:
        if (not isinstance(marker, dict)
                or set(marker) != {
                    "panorama_column", "bearing_camera_cw_deg"}):
            raise ReviewError("review manifest has an invalid column-grid marker")
        column = marker["panorama_column"]
        bearing = marker["bearing_camera_cw_deg"]
        if (isinstance(column, bool) or not isinstance(column, int)
                or not 0 <= column < validated.width):
            raise ReviewError("review manifest has an invalid panorama column")
        expected_bearing = float(geometry.azimuth_of_pano_column(
            column, validated.width)) % 360.0
        try:
            difference = abs(float(geometry.circular_diff_deg(
                _finite_real(bearing, "grid bearing"), expected_bearing)))
        except ReviewError as exc:
            raise ReviewError("review manifest has an invalid grid bearing") from exc
        if difference > 1e-9:
            raise ReviewError(
                "review manifest grid bearing disagrees with its exact column")
        columns.append(column)
    if columns != sorted(set(columns)):
        raise ReviewError("review manifest grid columns must be sorted and unique")

    expected_safety = {
        "contains_approved_nominal_forward": False,
        "diagnostic_candidate_angle_copied": False,
    }
    if manifest["safety"] != expected_safety:
        raise ReviewError("review manifest safety declaration is invalid")
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {
            "contact_sheet.png", "record_template.json"}:
        raise ReviewError("review manifest output identities are invalid")
    contact_sheet = path.parent / "contact_sheet.png"
    template_path = path.parent / "record_template.json"
    _require_regular_file(contact_sheet, "review contact sheet")
    _require_regular_file(template_path, "review record template")
    if (_sha256(contact_sheet) != outputs["contact_sheet.png"]
            or _sha256(template_path) != outputs["record_template.json"]):
        raise ReviewError("review bundle output bytes disagree with its manifest")
    try:
        with Image.open(contact_sheet) as opened:
            if opened.format != "PNG":
                raise ReviewError("review contact sheet must be PNG")
            opened.verify()
    except ReviewError:
        raise
    except Exception as exc:
        raise ReviewError(f"review contact sheet is corrupt: {exc}") from exc
    try:
        template = json.loads(
            template_path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid review record template: {exc}") from exc
    if (not isinstance(template, dict)
            or template.get("schema") != TEMPLATE_SCHEMA
            or template.get("calibration_authority") is not False
            or template.get("bearing_camera_cw_deg") is not None
            or template.get("panorama_column") is not None
            or _contains_approved_true(template)):
        raise ReviewError("review record template is not safely non-authoritative")
    return {
        "path": path,
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "manifest": manifest,
    }


def _finite_real(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ReviewError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ReviewError(f"{label} must be a finite real number")
    return result


def _atomic_replace_file(path: Path, payload: bytes) -> None:
    staging = path.parent / f".{path.name}.incomplete-{secrets.token_hex(8)}"
    try:
        _write_file(staging, payload)
        os.replace(staging, path)
        _fsync_directory(path.parent)
    finally:
        if staging.exists() or staging.is_symlink():
            staging.unlink()


def _transaction_path(dataset_base: Path) -> Path:
    return dataset_base / _TRANSACTION_RELATIVE


def _load_finalize_transaction(dataset_base: Path) -> dict:
    transaction = _transaction_path(dataset_base)
    if transaction.is_symlink() or not transaction.is_dir():
        raise ReviewError(
            f"nominal-forward transaction is not a regular directory: "
            f"{transaction}")
    expected_files = {
        "journal.json", "record.json", "old_checksums.sha256",
        "new_checksums.sha256",
    }
    actual_files = {path.name for path in transaction.iterdir()}
    if actual_files != expected_files or any(
            path.is_symlink() or not path.is_file()
            for path in transaction.iterdir()):
        raise ReviewError(
            f"nominal-forward transaction has unexpected contents: "
            f"{transaction}")
    try:
        journal = json.loads(
            (transaction / "journal.json").read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError(f"invalid finalization journal: {exc}") from exc
    fields = {
        "schema", "dataset", "output", "record_sha256",
        "old_checksums_sha256", "new_checksums_sha256",
    }
    if not isinstance(journal, dict) or set(journal) != fields:
        raise ReviewError("finalization journal fields do not match its schema")
    if (journal["schema"] != _TRANSACTION_SCHEMA
            or journal["dataset"] != dataset_base.name
            or journal["output"] != NOMINAL_FORWARD_NAME):
        raise ReviewError("finalization journal identity is invalid")
    payloads = {
        "record": (transaction / "record.json").read_bytes(),
        "old_checksum": (transaction / "old_checksums.sha256").read_bytes(),
        "new_checksum": (transaction / "new_checksums.sha256").read_bytes(),
    }
    identities = {
        "record_sha256": hashlib.sha256(payloads["record"]).hexdigest(),
        "old_checksums_sha256": hashlib.sha256(
            payloads["old_checksum"]).hexdigest(),
        "new_checksums_sha256": hashlib.sha256(
            payloads["new_checksum"]).hexdigest(),
    }
    if any(journal[key] != value for key, value in identities.items()):
        raise ReviewError("finalization transaction payload digest mismatch")
    nominal_forward.load(
        transaction / "record.json", expected_dataset=dataset_base.name)
    expected_new = checksums.manifest_bytes(
        dataset_base, replacements={NOMINAL_FORWARD_NAME: payloads["record"]})
    if payloads["new_checksum"] != expected_new:
        raise ReviewError(
            "finalization transaction checksum does not bind its record")
    return {"path": transaction, "journal": journal, **payloads}


def _cleanup_finalize_transaction(transaction: Path) -> None:
    parent = transaction.parent
    shutil.rmtree(transaction)
    _fsync_directory(parent)


def _record_state(path: Path, expected: bytes) -> str:
    if not path.exists() and not path.is_symlink():
        return "absent"
    if path.is_symlink() or not path.is_file():
        return "unexpected"
    return "expected" if path.read_bytes() == expected else "unexpected"


def _recover_finalize_transaction(dataset_base: Path) -> None:
    transaction_path = _transaction_path(dataset_base)
    if not transaction_path.exists() and not transaction_path.is_symlink():
        return
    transaction = _load_finalize_transaction(dataset_base)
    output = dataset_base / NOMINAL_FORWARD_NAME
    record_state = _record_state(output, transaction["record"])
    checksum_path = dataset_base / checksums.CHECKSUM_FILE
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ReviewError(
            f"cannot recover without a regular checksum manifest: "
            f"{checksum_path}")
    live_checksum = checksum_path.read_bytes()
    checksum_state = (
        "old" if live_checksum == transaction["old_checksum"] else
        "new" if live_checksum == transaction["new_checksum"] else
        "unexpected")
    if record_state == "absent" and checksum_state == "old":
        _cleanup_finalize_transaction(transaction["path"])
        return
    if record_state == "expected" and checksum_state == "old":
        count = checksums.regenerate(dataset_base)
        if count is None or checksum_path.read_bytes() != transaction[
                "new_checksum"]:
            raise ReviewError(
                "could not roll forward interrupted nominal-forward transaction")
        _cleanup_finalize_transaction(transaction["path"])
        return
    if record_state == "expected" and checksum_state == "new":
        _cleanup_finalize_transaction(transaction["path"])
        return
    raise ReviewError(
        "nominal-forward transaction cannot be recovered automatically; "
        f"record={record_state}, checksums={checksum_state}, "
        f"transaction={transaction['path']}")


def _prepare_finalize_transaction(dataset_base: Path, payload: bytes) -> dict:
    manifests = dataset_base / "_manifests"
    if manifests.exists() or manifests.is_symlink():
        if manifests.is_symlink() or not manifests.is_dir():
            raise ReviewError(
                f"derived manifest lane must be a regular directory: {manifests}")
    else:
        manifests.mkdir(mode=0o755)
        _fsync_directory(dataset_base)
    transaction = _transaction_path(dataset_base)
    transaction.mkdir(mode=0o700)
    try:
        checksum_path = dataset_base / checksums.CHECKSUM_FILE
        old_checksum = checksum_path.read_bytes()
        new_checksum = checksums.manifest_bytes(
            dataset_base, replacements={NOMINAL_FORWARD_NAME: payload})
        _write_file(transaction / "record.json", payload)
        _write_file(transaction / "old_checksums.sha256", old_checksum)
        _write_file(transaction / "new_checksums.sha256", new_checksum)
        journal = {
            "schema": _TRANSACTION_SCHEMA,
            "dataset": dataset_base.name,
            "output": NOMINAL_FORWARD_NAME,
            "record_sha256": hashlib.sha256(payload).hexdigest(),
            "old_checksums_sha256": hashlib.sha256(old_checksum).hexdigest(),
            "new_checksums_sha256": hashlib.sha256(new_checksum).hexdigest(),
        }
        _write_file(
            transaction / "journal.json",
            (json.dumps(journal, sort_keys=True, indent=2,
                        allow_nan=False) + "\n").encode("utf-8"))
        _fsync_directory(transaction)
        _fsync_directory(manifests)
    except BaseException:
        if transaction.exists():
            shutil.rmtree(transaction)
        raise
    return _load_finalize_transaction(dataset_base)


def _rollback_finalize_transaction(dataset_base: Path,
                                   transaction: dict) -> None:
    output = dataset_base / NOMINAL_FORWARD_NAME
    state = _record_state(output, transaction["record"])
    if state == "unexpected":
        raise ReviewError(
            "refusing to roll back an unexpected nominal-forward output")
    if state == "expected":
        output.unlink()
        _fsync_directory(dataset_base)
    _atomic_replace_file(
        dataset_base / checksums.CHECKSUM_FILE,
        transaction["old_checksum"])
    _cleanup_finalize_transaction(transaction["path"])


def _publish_dataset_record(dataset_base: Path, output: Path, payload: bytes):
    expected_output = dataset_base / NOMINAL_FORWARD_NAME
    if Path(output).resolve() != expected_output.resolve():
        raise ReviewError(
            "approved nominal-forward output must be the checksummed dataset "
            f"record {expected_output}")
    output = expected_output
    _recover_finalize_transaction(dataset_base)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    try:
        checksums.verify(dataset_base)
    except ValueError as exc:
        raise ReviewError(
            "refusing to mask a stale pre-finalization checksum manifest: "
            f"{exc}") from exc
    transaction = _prepare_finalize_transaction(dataset_base, payload)
    try:
        os.link(transaction["path"] / "record.json", output,
                follow_symlinks=False)
    except Exception:
        _cleanup_finalize_transaction(transaction["path"])
        raise
    try:
        _fsync_directory(dataset_base)
        count = checksums.regenerate(dataset_base)
        checksum_path = dataset_base / checksums.CHECKSUM_FILE
        if count is None or checksum_path.read_bytes() != transaction[
                "new_checksum"]:
            raise ReviewError(
                "shared checksum owner did not publish the reviewed manifest")
        loaded = nominal_forward.load(
            output, expected_dataset=dataset_base.name)
    except Exception:
        _rollback_finalize_transaction(dataset_base, transaction)
        raise
    _cleanup_finalize_transaction(transaction["path"])
    return loaded


def finalize_nominal_forward(dataset_base: Path, output: Path, *, version: str,
                             mounting_id: str,
                             review_manifest: Path,
                             bearing_camera_cw_deg: float | None,
                             panorama_column: float | None,
                             uncertainty_deg: float, operator: str,
                             approved_at: str, notes: str,
                             evidence_frame_ids: tuple[str, ...],
                             approve_as_authority: bool) -> dict:
    """Write one explicitly approved v1 record; never infer a candidate."""
    if approve_as_authority is not True:
        raise ReviewError(
            "finalization requires explicit approve_as_authority acknowledgment")
    if (bearing_camera_cw_deg is None) == (panorama_column is None):
        raise ReviewError(
            "provide exactly one of bearing_camera_cw_deg or panorama_column")
    validated = validate_dataset(dataset_base)
    if (not evidence_frame_ids
            or len(set(evidence_frame_ids)) != len(evidence_frame_ids)):
        raise ReviewError("evidence_frame_ids must be non-empty and unique")
    known = {item.pano_id for item in validated.panoramas}
    missing = [item for item in evidence_frame_ids if item not in known]
    if missing:
        raise ReviewError(f"unknown evidence frame IDs: {missing}")
    review = validate_review_manifest(
        validated, review_manifest, evidence_frame_ids)
    if not isinstance(notes, str):
        raise ReviewError("notes must be a string")
    if _REVIEW_DIGEST_PREFIX in notes:
        raise ReviewError(
            f"notes must not contain reserved provenance marker "
            f"{_REVIEW_DIGEST_PREFIX!r}")
    separator = "" if not notes or notes.endswith("\n") else "\n"
    approved_notes = (
        notes + separator + _REVIEW_DIGEST_PREFIX + review["sha256"])
    if panorama_column is not None:
        column = _finite_real(panorama_column, "panorama_column")
        if not 0.0 <= column < validated.width:
            raise ReviewError("panorama_column must be in [0, panorama_width)")
        bearing = float(geometry.azimuth_of_pano_column(
            column, validated.width)) % 360.0
    else:
        bearing = _finite_real(
            bearing_camera_cw_deg, "bearing_camera_cw_deg")
        if not 0.0 <= bearing < 360.0:
            raise ReviewError("bearing_camera_cw_deg must be in [0, 360)")
        column, _ = geometry.pano_px_from_direction(
            bearing, 0.0, validated.width, validated.height)
    document = {
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "dataset": validated.name,
        "version": version,
        "mounting_id": mounting_id,
        "panorama_column": column,
        "panorama_width": validated.width,
        "bearing_camera_cw_deg": bearing,
        "uncertainty_deg": _finite_real(uncertainty_deg, "uncertainty_deg"),
        "evidence_frame_ids": list(evidence_frame_ids),
        "operator": operator,
        "approved_at": approved_at,
        "approved": True,
        "notes": approved_notes,
    }
    parsed = nominal_forward.parse(
        document, expected_dataset=validated.name, source="explicit finalize arguments")
    payload = (json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=False,
        allow_nan=False) + "\n").encode("utf-8")
    loaded = _publish_dataset_record(validated.base, Path(output), payload)
    if loaded != parsed:
        raise ReviewError("nominal-forward parse/load validation disagreed")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    bundle = subcommands.add_parser(
        "bundle", help="create review evidence only (the safe default mode)")
    bundle.add_argument(
        "--dataset_base", type=Path, required=True,
        help="post-migration camera-frame dataset directory")
    bundle.add_argument(
        "--output_dir", type=Path, required=True,
        help="new review-bundle directory outside the frozen dataset")
    bundle.add_argument(
        "--evidence_frame_id", action="append", default=[],
        help="exact pano_id to review; repeat to set explicit ordered evidence")
    bundle.add_argument("--evidence_count", type=int,
                        default=DEFAULT_EVIDENCE_COUNT)
    bundle.add_argument("--grid_step_deg", type=float,
                        default=DEFAULT_GRID_STEP_DEG)
    bundle.add_argument("--display_width", type=int,
                        default=DEFAULT_DISPLAY_WIDTH)

    finalize = subcommands.add_parser(
        "finalize", help="explicitly author an approved nominal-forward v1 record")
    finalize.add_argument(
        "--dataset_base", type=Path, required=True,
        help="post-migration camera-frame dataset directory")
    finalize.add_argument(
        "--output", type=Path, required=True,
        help="must be DATASET_BASE/nominal_forward.json; never overwritten")
    finalize.add_argument(
        "--review_manifest", type=Path, required=True,
        help="review_manifest.json from the exact inspected bundle")
    finalize.add_argument("--version", required=True)
    finalize.add_argument("--mounting_id", required=True)
    direction = finalize.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--bearing_camera_cw_deg", type=float,
        help="human-selected degrees clockwise from camera forward in [0,360)")
    direction.add_argument(
        "--panorama_column", type=float,
        help="human-selected zero-based source-pixel x in [0, panorama width)")
    finalize.add_argument("--uncertainty_deg", type=float, required=True)
    finalize.add_argument("--operator", required=True)
    finalize.add_argument(
        "--approved_at", required=True,
        help="ISO-8601 human approval time with UTC offset or Z")
    finalize.add_argument("--notes", required=True)
    finalize.add_argument(
        "--evidence_frame_id", action="append", required=True,
        help="repeat the reviewed bundle pano_ids in exactly the same order")
    finalize.add_argument(
        "--approve_as_authority", action="store_true", required=True,
        help="acknowledge that this human choice becomes calibration authority")
    return parser


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Omitting a mode can only select the non-authoritative bundle operation.
    if arguments and arguments[0] not in ("bundle", "finalize", "-h", "--help"):
        arguments.insert(0, "bundle")
    args = _parser().parse_args(arguments)
    if args.mode == "bundle":
        manifest = create_review_bundle(
            args.dataset_base, args.output_dir,
            evidence_frame_ids=tuple(args.evidence_frame_id),
            evidence_count=args.evidence_count,
            grid_step_deg=args.grid_step_deg,
            display_width=args.display_width)
        print(json.dumps({
            "mode": "review_only",
            "output_dir": str(args.output_dir),
            "dataset": manifest["dataset"],
            "evidence_frame_ids": manifest["selection"]["evidence_frame_ids"],
        }, sort_keys=True))
        return 0
    document = finalize_nominal_forward(
        args.dataset_base, args.output,
        version=args.version,
        mounting_id=args.mounting_id,
        review_manifest=args.review_manifest,
        bearing_camera_cw_deg=args.bearing_camera_cw_deg,
        panorama_column=args.panorama_column,
        uncertainty_deg=args.uncertainty_deg,
        operator=args.operator,
        approved_at=args.approved_at,
        notes=args.notes,
        evidence_frame_ids=tuple(args.evidence_frame_id),
        approve_as_authority=args.approve_as_authority)
    print(json.dumps({
        "mode": "approved_nominal_forward",
        "output": str(args.output),
        "dataset": document["dataset"],
        "bearing_camera_cw_deg": document["bearing_camera_cw_deg"],
        "panorama_column": document["panorama_column"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
