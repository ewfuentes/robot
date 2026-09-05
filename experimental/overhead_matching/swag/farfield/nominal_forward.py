"""Human-approved camera-to-nominal-forward calibration.

Nominal forward is the platform's fixed longitudinal forward axis.  It is
not GPS course and it is never inferred from a sun check or a landmark sweep.
Those tools may produce reviewable alignment diagnostics, but only a record
validated here may rotate localization bearings.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import json
import math
from pathlib import Path
import re

from experimental.overhead_matching.swag.farfield import geometry


SCHEMA = "farfield_nominal_forward/v1"
FRAME = "camera_centre_column_nominal_forward_axis_v1"
_FIELDS = frozenset({
    "schema",
    "frame",
    "dataset",
    "version",
    "mounting_id",
    "panorama_column",
    "panorama_width",
    "bearing_camera_cw_deg",
    "uncertainty_deg",
    "evidence_frame_ids",
    "operator",
    "approved_at",
    "approved",
    "notes",
})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


@dataclasses.dataclass(frozen=True)
class NominalForward:
    dataset: str
    version: str
    mounting_id: str
    panorama_column: float
    panorama_width: int
    bearing_camera_cw_deg: float
    uncertainty_deg: float
    evidence_frame_ids: tuple[str, ...]
    operator: str
    approved_at: str
    notes: str = ""


def _real(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number, not {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _duplicate_safe_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _text_field(document: dict, name: str, source: str) -> str:
    value = document[name]
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{source}: {name} must be a non-empty string without leading "
            "or trailing whitespace")
    return value


def _identifier_field(document: dict, name: str, source: str) -> str:
    value = _text_field(document, name, source)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{source}: {name} must match {_IDENTIFIER.pattern!r}")
    return value


def _approval_timestamp(document: dict, source: str) -> str:
    value = _text_field(document, "approved_at", source)
    try:
        parsed = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ValueError(
            f"{source}: approved_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"{source}: approved_at must include a UTC offset or Z")
    return value


def parse(document, *, expected_dataset: str | None = None,
          source: str = "<memory>") -> NominalForward:
    if not isinstance(document, dict):
        raise ValueError(f"{source}: nominal-forward record must be an object")
    missing = _FIELDS.difference(document)
    unknown = set(document).difference(_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(
                sorted(repr(key) for key in unknown)))
        raise ValueError(
            f"{source}: nominal-forward fields do not match the v1 schema; "
            + "; ".join(details))
    if document["schema"] != SCHEMA:
        raise ValueError(f"{source}: schema must be {SCHEMA!r}")
    if document["frame"] != FRAME:
        raise ValueError(f"{source}: frame must be {FRAME!r}")
    if document["approved"] is not True:
        raise ValueError(
            f"{source}: approved must be the boolean true; diagnostics and "
            "review candidates are not calibration authority")

    dataset = _text_field(document, "dataset", source)
    if expected_dataset is not None and dataset != expected_dataset:
        raise ValueError(
            f"{source}: record belongs to dataset {dataset!r}, expected "
            f"{expected_dataset!r}")
    width_raw = document["panorama_width"]
    if isinstance(width_raw, bool) or not isinstance(width_raw, int) \
            or width_raw <= 0:
        raise ValueError(f"{source}: panorama_width must be a positive integer")
    column = _real(document["panorama_column"], "panorama_column")
    if not 0.0 <= column < width_raw:
        raise ValueError(
            f"{source}: panorama_column must be in [0, panorama_width)")
    bearing = _real(document["bearing_camera_cw_deg"],
                    "bearing_camera_cw_deg")
    if not 0.0 <= bearing < 360.0:
        raise ValueError(
            f"{source}: bearing_camera_cw_deg must be in [0, 360)")
    expected_bearing = float(
        geometry.azimuth_of_pano_column(column, width_raw)) % 360.0
    if abs(float(geometry.circular_diff_deg(bearing, expected_bearing))) > 1e-9:
        raise ValueError(
            f"{source}: bearing_camera_cw_deg={bearing} disagrees with "
            f"panorama column/width derivation {expected_bearing}")
    uncertainty = _real(document["uncertainty_deg"], "uncertainty_deg")
    if uncertainty < 0.0:
        raise ValueError(f"{source}: uncertainty_deg must be nonnegative")
    evidence = document["evidence_frame_ids"]
    if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item and item == item.strip()
            for item in evidence):
        raise ValueError(
            f"{source}: evidence_frame_ids must be a non-empty list of "
            "non-empty strings without surrounding whitespace")
    if len(set(evidence)) != len(evidence):
        raise ValueError(f"{source}: evidence_frame_ids must be unique")
    notes = document["notes"]
    if not isinstance(notes, str):
        raise ValueError(f"{source}: notes must be a string")
    return NominalForward(
        dataset=dataset,
        version=_identifier_field(document, "version", source),
        mounting_id=_identifier_field(document, "mounting_id", source),
        panorama_column=column,
        panorama_width=width_raw,
        bearing_camera_cw_deg=bearing % 360.0,
        uncertainty_deg=uncertainty,
        evidence_frame_ids=tuple(evidence),
        operator=_text_field(document, "operator", source),
        approved_at=_approval_timestamp(document, source),
        notes=notes,
    )


def load(path: Path, *, expected_dataset: str | None = None) -> NominalForward:
    path = Path(path)
    try:
        document = json.loads(
            path.read_text(), object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_nonfinite_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read nominal-forward record {path}: {exc}") from exc
    return parse(document, expected_dataset=expected_dataset, source=str(path))


def camera_to_forward_cw_deg(bearing_camera_cw_deg: float,
                             calibration: NominalForward) -> float:
    """Rotate a camera-frame bearing into the fixed nominal-forward frame."""
    bearing = _real(bearing_camera_cw_deg, "bearing_camera_cw_deg")
    return (bearing - calibration.bearing_camera_cw_deg) % 360.0
