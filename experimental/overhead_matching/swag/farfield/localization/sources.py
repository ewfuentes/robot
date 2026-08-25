"""Canonical tracker/audit evidence for the localization viewer.

Only current typed artifacts are supported.  The supplied object-track and
semantic-audit artifacts must be the exact ancestors of the viewed run, and
their join is delegated to the same strict readers used by matching and
localization export.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.tracking import tracklets


DEFAULT_CHIP_BUDGET_BYTES = 3 * 1024 * 1024
LOCALIZATION_RUN_KIND = "localization_run"


class SourceContractError(ValueError):
    """Viewer evidence is stale, incomplete, or not an ancestor of the run."""


@dataclass(frozen=True)
class TrackletSource:
    tracklet_id: str
    local_id: str
    source_track_id: int
    keyframe_span: tuple[int, int]
    n_supports: int
    verdict: str
    confidence: str
    valid_segments: tuple[tuple[int, int], ...]
    name: str | None
    tags: tuple[str, ...]
    description: str
    features: tuple[str, ...]
    unresolved: str
    chip_data_uri: str | None


@dataclass(frozen=True)
class SourceBundle:
    tracklets: dict[str, TrackletSource]
    tracks_ref: artifact.ArtifactRef
    audits_ref: artifact.ArtifactRef
    notes: tuple[str, ...] = ()

    def get(self, tracklet_id: str) -> TrackletSource | None:
        return self.tracklets.get(tracklet_id)


def _one_upstream(manifest, kind: str, label: str) -> artifact.ArtifactRef:
    matches = [ref for ref in manifest.upstreams if ref.kind == kind]
    if len(matches) != 1:
        raise SourceContractError(
            f"{label} must bind exactly one {kind} artifact; found "
            f"{len(matches)}")
    return matches[0]


def _validate_run_ancestry(
        run_dir: Path,
        tracks_ref: artifact.ArtifactRef,
        audits_ref: artifact.ArtifactRef) -> None:
    try:
        run_ref = artifact.open_artifact(
            run_dir, expected_kind=LOCALIZATION_RUN_KIND,
            expected_dataset=tracks_ref.dataset)
        run_manifest = artifact.load_manifest(run_ref.path)
        inputs_ref = _one_upstream(
            run_manifest, paths_lib.LOCALIZATION_INPUTS, "localization run")
        inputs_manifest = artifact.load_manifest(inputs_ref.path)
        matching_ref = _one_upstream(
            inputs_manifest, paths_lib.LANDMARK_MATCHES,
            "localization_inputs artifact")
        matching_manifest = artifact.load_manifest(matching_ref.path)
    except artifact.ArtifactError as error:
        raise SourceContractError(
            f"cannot validate viewer source ancestry: {error}") from error
    for expected, label in ((tracks_ref, paths_lib.OBJECT_TRACKS),
                            (audits_ref, paths_lib.SEMANTIC_AUDITS)):
        if matching_manifest.upstreams.count(expected) != 1:
            raise SourceContractError(
                f"viewed run's matching artifact is not bound to the supplied "
                f"{label} artifact")


def _load_meta(audit_dir: Path) -> dict:
    path = Path(audit_dir) / "audit_meta.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        message = f"cannot read validated {path}: {error}"
        raise SourceContractError(message) from error
    requests = value.get("requests") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or value.get("schema") != audit_io.META_SCHEMA
            or not isinstance(requests, dict)):
        raise SourceContractError(
            f"{path} is not a {audit_io.META_SCHEMA} document")
    return requests


def _chip_path(audit_dir: Path, paths: list[str]) -> Path | None:
    root = Path(audit_dir).resolve()
    for relative in paths:
        path = Path(audit_dir) / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise SourceContractError(
                f"audit chip escapes or is absent from its artifact: {relative}")
        if path.is_symlink() or not path.is_file():
            raise SourceContractError(
                f"audit chip must be a regular non-symlink file: {path}")
        return path
    return None


def _best_name(primary: dict) -> str | None:
    candidates = primary["name_candidates"]
    if not candidates:
        return None
    return max(
        candidates, key=lambda item: (float(item["weight"]), item["name"])
    )["name"]


def _tags(primary: dict) -> tuple[str, ...]:
    return tuple(item["tag"] for item in sorted(
        primary["tags"],
        key=lambda item: (-float(item["weight"]), item["tag"]))[:6])


def _data_uri(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return (f"data:{media_type};base64,"
            + base64.b64encode(raw).decode("ascii"), len(raw))


def load(run_dir: Path, tracks_dir: Path, audit_dir: Path, tracklet_ids,
         *, embed_chips: bool = True,
         chip_budget_bytes: int = DEFAULT_CHIP_BUDGET_BYTES) -> SourceBundle:
    """Load exact current-schema source evidence for the requested tracklets."""
    if (isinstance(chip_budget_bytes, bool)
            or not isinstance(chip_budget_bytes, int)
            or chip_budget_bytes < 0):
        raise SourceContractError("chip_budget_bytes must be nonnegative")
    try:
        audits = audit_io.load_audits(tracks_dir, audit_dir)
        accepted = tracklets.build_accepted_tracklets(
            audits.source_tracks, audits)
    except (artifact.ArtifactError, audit_io.AuditArtifactError,
            tracklets.TrackletContractError) as error:
        message = f"invalid canonical viewer sources: {error}"
        raise SourceContractError(message) from error
    _validate_run_ancestry(run_dir, audits.tracks_ref,
                           audits.semantic_audits_ref)

    wanted = set(tracklet_ids)
    by_id = {item.tracklet_id: item for item in accepted}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SourceContractError(
            "run names tracklets absent from the exact audited-track join: "
            + ", ".join(missing[:4]))
    request_meta = _load_meta(Path(audit_dir))

    result = {}
    notes = []
    spent = 0
    skipped = []
    for tracklet_id in sorted(wanted):
        item = by_id[tracklet_id]
        track = item.source_track
        audit = item.audit
        meta = request_meta[item.local_id]
        records = track["records"]
        keyframes = [record["keyframe"] for record in records]
        support_count = meta.get(
            "n_supports",
            sum(len(record.get("supports", ())) for record in records))
        chip_uri = None
        if embed_chips:
            path = _chip_path(Path(audit_dir), meta.get("chips", []))
            if path is not None:
                uri, size = _data_uri(path)
                if spent + size <= chip_budget_bytes:
                    chip_uri = uri
                    spent += size
                else:
                    skipped.append(tracklet_id)
        primary = audit["primary_object"]
        result[tracklet_id] = TrackletSource(
            tracklet_id=tracklet_id,
            local_id=item.local_id,
            source_track_id=track["track_id"],
            keyframe_span=(min(keyframes), max(keyframes)),
            n_supports=support_count,
            verdict=audit["verdict"],
            confidence=audit["confidence"],
            valid_segments=tuple(
                (segment.start_keyframe_idx, segment.end_keyframe_idx)
                for segment in item.valid_segments),
            name=_best_name(primary),
            tags=_tags(primary),
            description=primary["description"],
            features=tuple(primary["distinctive_features"]),
            unresolved=audit["unresolved"],
            chip_data_uri=chip_uri)
    if skipped:
        notes.append(
            f"source-chip budget reached: {len(skipped)} tracklet chip(s) "
            "were not embedded")
    return SourceBundle(
        result, audits.tracks_ref, audits.semantic_audits_ref, tuple(notes))
