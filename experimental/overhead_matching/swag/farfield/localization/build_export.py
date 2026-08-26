"""Publish the immutable inputs consumed by bearing-only localization.

This is an artifact-to-artifact boundary.  It consumes lossless camera-frame
bearing observations, a completely aggregated landmark-matching artifact, and
one catalog artifact.  Camera bearings are rotated only by a human-approved,
dataset-bound nominal-forward calibration.  GPS is used to manufacture an
explicitly labelled dead-reckoning input and diagnostic truth; it is never a
camera calibration source.

There is deliberately no ``run_dir`` discovery, mount-offset override, or
sidecar fallback.  Every scientific input is explicit and every published
output is covered by the localization_inputs artifact manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import msgspec
import numpy as np

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import (
    artifact,
    configured_lane,
    build_config,
    dataset as dataset_lib,
    geometry as geo,
    nominal_forward,
    paths as paths_lib,
    publication,
    provenance,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    audit_io,
    heading,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    gps_to_odometry,
    run_io,
    structs,
)
from experimental.overhead_matching.swag.farfield.matching import identity_review
from experimental.overhead_matching.swag.farfield.tracking import tracklets


GENERATOR = ("//experimental/overhead_matching/swag/farfield/"
             "localization:build_export")
OBSERVATIONS_NAME = "observations.jsonl"
COMPATIBILITY_NAME = "compatibility.json"
CATALOG_NAME = "catalog.feather"
UNINFORMATIVE_MATCHER = "uninformative_v1"
_OBSERVATION_KEYS = frozenset({
    "tracklet_id",
    "keyframe_idx",
    "bearing_camera_cw_deg",
    "angular_width_deg",
    "sigma_deg",
    "correlation_group",
})

# A coarse display type for viewer glyphs.  The localization likelihood uses
# identity, geometry, and the one explicit positional sigma, not this label.
TYPE_TAGS = ("seamark:type", "man_made", "leisure", "amenity", "natural",
             "building", "place", "highway")


class LocalizationInputError(ValueError):
    """An upstream cannot support a reproducible localization export."""


def type_key(tags: dict) -> str:
    for key in TYPE_TAGS:
        if key in tags:
            return f"{key}={tags[key]}"
    return "landmark"


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalizationInputError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise LocalizationInputError(f"{name} must be {qualifier}")
    return value


def _exact_upstream(manifest: artifact.ArtifactManifest,
                    kind: str) -> artifact.ArtifactRef:
    matches = [ref for ref in manifest.upstreams if ref.kind == kind]
    if len(matches) != 1:
        raise LocalizationInputError(
            f"{manifest.kind} manifest must contain exactly one {kind} "
            f"upstream; found {len(matches)}")
    return matches[0]


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LocalizationInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_observation_records(path: Path) \
        -> list[tracklets.CameraBearingObservation]:
    records = []
    try:
        stream = path.open()
    except OSError as exc:
        raise LocalizationInputError(f"cannot read {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise LocalizationInputError(
                    f"{path}:{line_number}: blank records are not canonical")
            try:
                value = json.loads(
                    line, object_pairs_hook=_reject_duplicate_keys,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        LocalizationInputError(
                            f"non-finite JSON constant {token!r}")))
            except json.JSONDecodeError as exc:
                raise LocalizationInputError(
                    f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict) or set(value) != _OBSERVATION_KEYS:
                actual = set(value) if isinstance(value, dict) else set()
                raise LocalizationInputError(
                    f"{path}:{line_number}: observation keys differ; "
                    f"missing={sorted(_OBSERVATION_KEYS - actual)}, "
                    f"unknown={sorted(actual - _OBSERVATION_KEYS)}")
            try:
                records.append(tracklets.CameraBearingObservation(**value))
            except (TypeError, tracklets.TrackletContractError) as exc:
                raise LocalizationInputError(
                    f"{path}:{line_number}: {exc}") from exc
    keys = [(item.tracklet_id, item.keyframe_idx) for item in records]
    if keys != sorted(keys):
        raise LocalizationInputError(
            f"{path}: observations must be sorted by "
            "(tracklet_id, keyframe_idx)")
    if len(keys) != len(set(keys)):
        raise LocalizationInputError(
            f"{path}: duplicate (tracklet_id, keyframe_idx)")
    return records


def load_observations(observations_dir: Path, *, dataset_name: str,
                      panorama_width: int, expected_versions: dict[str, str],
                      build_identity: str,
                      target_build_dir: Path | None = None):
    """Load and independently verify a complete lossless bearing artifact.

    Coverage and geometry are rebuilt from the bound tracks + audit, rather
    than trusting a producer-supplied percentage or count.
    """
    observations_dir = Path(observations_dir)
    observations_ref = artifact.open_artifact(
        observations_dir, expected_kind=paths_lib.BEARING_OBSERVATIONS,
        expected_dataset=dataset_name,
        expected_version=expected_versions[paths_lib.BEARING_OBSERVATIONS])
    # `open_artifact` above proved this is the artifact it claims to be at
    # the version the recipe names. Which generation it belongs to is the
    # orchestrator's question, answered by `artifact_identity`.
    target_document = (build_config.load(target_build_dir)
                       if target_build_dir is not None else None)
    manifest = (configured_lane.require(
        observations_ref, document=target_document,
        kind=paths_lib.BEARING_OBSERVATIONS)
        if target_document is not None
        else artifact.load_manifest(observations_dir))
    if manifest.config.get("build_identity") != build_identity:
        raise LocalizationInputError(
            "bearing_observations belongs to a different immutable build")
    expected_upstream_kinds = (
        paths_lib.OBJECT_TRACKS, paths_lib.SEMANTIC_AUDITS)
    if tuple(ref.kind for ref in manifest.upstreams) != expected_upstream_kinds:
        raise LocalizationInputError(
            "bearing_observations upstreams must be exactly object_tracks "
            "then semantic_audits")
    tracks_ref = _exact_upstream(manifest, paths_lib.OBJECT_TRACKS)
    audits_ref = _exact_upstream(manifest, paths_lib.SEMANTIC_AUDITS)
    for ref in (tracks_ref, audits_ref):
        if ref.version != expected_versions[ref.kind]:
            raise LocalizationInputError(
                f"{ref.kind} version disagrees with immutable build config")
    tracks_manifest = (configured_lane.require(
        tracks_ref, document=target_document, kind=paths_lib.OBJECT_TRACKS)
        if target_document is not None
        else artifact.load_manifest(tracks_ref.path))
    if tracks_manifest.config.get("build_identity") != build_identity:
        raise LocalizationInputError(
            f"{paths_lib.OBJECT_TRACKS} belongs to a different immutable build")
    audits_manifest = (configured_lane.require(
        audits_ref, document=target_document, kind=paths_lib.SEMANTIC_AUDITS)
        if target_document is not None
        else artifact.load_manifest(audits_ref.path))
    if audits_manifest.config.get("build_identity") != build_identity:
        raise LocalizationInputError(
            f"{paths_lib.SEMANTIC_AUDITS} belongs to a different immutable build")
    try:
        audits = audit_io.load_audits(
            Path(tracks_ref.path), Path(audits_ref.path))
    except (audit_io.AuditArtifactError, artifact.ArtifactError) as exc:
        raise LocalizationInputError(
            f"bearing observations have invalid bound tracks/audits: {exc}") \
            from exc
    if (audits.tracks_ref.to_dict() != tracks_ref.to_dict()
            or audits.semantic_audits_ref.to_dict() != audits_ref.to_dict()):
        raise LocalizationInputError(
            "bearing-observation upstream references do not resolve to their "
            "recorded artifact identities")

    accepted = tracklets.build_accepted_tracklets(
        audits.source_tracks, audits)
    expected = {}
    for item in accepted:
        for segment in item.valid_segments:
            raw_segment = [{"start_t": segment.start_t,
                            "end_t": segment.end_t}]
            group = f"{item.tracklet_id}/audit-segment-{segment.index}"
            for keyframe, bearing, width in tracklets.bearing_series(
                    item.source_track, panorama_width, raw_segment):
                key = (item.tracklet_id, keyframe)
                if key in expected:
                    raise LocalizationInputError(
                        f"accepted-track contract repeats observation {key}")
                expected[key] = (bearing % 360.0, width, group)

    records = _load_observation_records(
        observations_dir / OBSERVATIONS_NAME)
    actual = {(item.tracklet_id, item.keyframe_idx): item for item in records}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise LocalizationInputError(
            "bearing observations do not have complete accepted-tracklet "
            f"coverage: missing={missing[:5]}, extra={extra[:5]}")
    for key, (bearing, width, group) in expected.items():
        item = actual[key]
        if abs(float(geo.circular_diff_deg(
                item.bearing_camera_cw_deg, bearing))) > 1e-9:
            raise LocalizationInputError(
                f"observation {key} bearing differs from its source track")
        if not math.isclose(item.angular_width_deg, width,
                            rel_tol=0.0, abs_tol=1e-9):
            raise LocalizationInputError(
                f"observation {key} angular width differs from its source track")
        if item.correlation_group != group:
            raise LocalizationInputError(
                f"observation {key} has stale correlation-group identity")

    sigmas = {item.sigma_deg for item in records}
    if len(sigmas) > 1:
        raise LocalizationInputError(
            "bearing observation sigma must be uniform in this artifact")
    return (observations_ref, records,
            {item.tracklet_id for item in accepted}, tracks_ref, audits_ref)


def reduce_observations(
        observations: list[tracklets.CameraBearingObservation],
        epoch_keyframes: int) -> list[tracklets.Measurement]:
    """Apply the named epoch_fused_compat_v1 reducer once, at this seam."""
    if isinstance(epoch_keyframes, bool) or not isinstance(epoch_keyframes, int) \
            or epoch_keyframes <= 0:
        raise LocalizationInputError(
            "reducer_epoch_keyframes must be a positive integer")
    if not observations:
        return []
    sigmas = {item.sigma_deg for item in observations}
    if len(sigmas) != 1:
        raise LocalizationInputError(
            "epoch_fused_compat_v1 requires one recorded observation sigma")
    params = tracklets.TrackletParams(
        epoch_keyframes=epoch_keyframes,
        bearing_sigma_deg=next(iter(sigmas)))
    return tracklets.epoch_fused_compat_v1(observations, params)


def forward_frame_measurements(
        camera_measurements: list[tracklets.Measurement],
        calibration: nominal_forward.NominalForward) \
        -> list[structs.TrackletMeasurement]:
    """Rotate camera-CW bearings into human-approved nominal-forward CW."""
    seen = set()
    result = []
    for measurement in camera_measurements:
        key = (measurement.tracklet_id, measurement.anchor_keyframe_idx)
        if key in seen:
            raise LocalizationInputError(f"duplicate information epoch {key}")
        seen.add(key)
        result.append(structs.TrackletMeasurement(
            tracklet_id=measurement.tracklet_id,
            anchor_keyframe_idx=measurement.anchor_keyframe_idx,
            bearing_forward_cw_deg=nominal_forward.camera_to_forward_cw_deg(
                measurement.bearing_camera_cw_deg, calibration),
            kappa=measurement.kappa))
    return result


def _validate_table(table: structs.CompatibilityTable) -> None:
    if not isinstance(table.tracklet_id, str) or not table.tracklet_id:
        raise LocalizationInputError("compatibility tracklet_id is empty")
    if not isinstance(table.matcher_version, str) or not table.matcher_version:
        raise LocalizationInputError(
            f"table {table.tracklet_id!r} has no matcher_version")
    for value, name in ((table.default_log_lr, "default_log_lr"),
                        (table.clip_lo, "clip_lo"),
                        (table.clip_hi, "clip_hi")):
        _finite(value, f"table {table.tracklet_id!r} {name}")
    if table.clip_lo >= table.clip_hi:
        raise LocalizationInputError(
            f"table {table.tracklet_id!r} has an empty clip interval")
    if table.status not in ("fast", "refined"):
        raise LocalizationInputError(
            f"table {table.tracklet_id!r} has invalid status {table.status!r}")
    landmark_ids = []
    for entry in table.entries:
        if not isinstance(entry.landmark_id, str) or not entry.landmark_id:
            raise LocalizationInputError(
                f"table {table.tracklet_id!r} has an empty landmark id")
        _finite(entry.log_lr,
                f"table {table.tracklet_id!r} entry log_lr")
        landmark_ids.append(entry.landmark_id)
    if len(landmark_ids) != len(set(landmark_ids)):
        raise LocalizationInputError(
            f"table {table.tracklet_id!r} repeats a landmark id")


def load_matching(matching_dir: Path, *, dataset_name: str,
                  accepted_tracklet_ids: set[str],
                  tracks_ref: artifact.ArtifactRef,
                  audits_ref: artifact.ArtifactRef,
                  catalog_ref: artifact.ArtifactRef,
                  expected_version: str, build_identity: str):
    """Load only a complete matcher result covering every accepted tracklet."""
    matching_dir = Path(matching_dir)
    matching_ref = artifact.open_artifact(
        matching_dir, expected_kind=paths_lib.LANDMARK_MATCHES,
        expected_dataset=dataset_name, expected_version=expected_version)
    manifest = artifact.load_manifest(matching_dir)
    if manifest.config.get("build_identity") != build_identity:
        raise LocalizationInputError(
            "landmark_matches belongs to a different immutable build")
    expected_kinds = (paths_lib.OBJECT_TRACKS,
                      paths_lib.SEMANTIC_AUDITS, paths_lib.CATALOGS)
    if tuple(ref.kind for ref in manifest.upstreams) != expected_kinds:
        raise LocalizationInputError(
            "landmark_matches upstreams must be exactly object_tracks, "
            "semantic_audits, then catalogs")
    for expected in (tracks_ref, audits_ref, catalog_ref):
        recorded = _exact_upstream(manifest, expected.kind)
        if recorded.to_dict() != expected.to_dict():
            raise LocalizationInputError(
                f"matching artifact is bound to a different {expected.kind} "
                "artifact")
    if manifest.config.get("coverage") != "complete":
        raise LocalizationInputError(
            "matching manifest must attest coverage='complete'")
    n_expected = manifest.config.get("n_expected")
    n_successful = manifest.config.get("n_successful")
    if (isinstance(n_expected, bool) or not isinstance(n_expected, int)
            or isinstance(n_successful, bool)
            or not isinstance(n_successful, int)
            or n_expected < 0 or n_successful != n_expected):
        raise LocalizationInputError(
            "matching manifest does not attest one successful result per "
            "expected request")
    try:
        tables = msgspec.json.decode(
            (matching_dir / COMPATIBILITY_NAME).read_bytes(),
            type=list[structs.CompatibilityTable], dec_hook=msgspec_dec_hook)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise LocalizationInputError(
            f"cannot decode matching compatibility tables: {exc}") from exc
    if not tables:
        raise LocalizationInputError(
            "matching compatibility table list is empty; a complete matching "
            "artifact must publish one table per accepted tracklet")
    table_ids = [table.tracklet_id for table in tables]
    if len(table_ids) != len(set(table_ids)):
        raise LocalizationInputError("matching repeats a compatibility table")
    actual_ids = set(table_ids)
    if actual_ids != accepted_tracklet_ids:
        raise LocalizationInputError(
            "matching tables do not completely cover accepted tracklets: "
            f"missing={sorted(accepted_tracklet_ids - actual_ids)[:5]}, "
            f"extra={sorted(actual_ids - accepted_tracklet_ids)[:5]}")
    for table in tables:
        _validate_table(table)
    versions = {table.matcher_version for table in tables}
    matcher_version = (next(iter(versions)) if len(versions) == 1
                       else "empty" if not versions else None)
    if matcher_version is None:
        raise LocalizationInputError(
            f"matching tables mix matcher versions {sorted(versions)}")
    return matching_ref, tables, matcher_version


def apply_identity_review(
        tables: list[structs.CompatibilityTable],
        review: identity_review.IdentityReview,
        review_version: str) -> tuple[list[structs.CompatibilityTable], str]:
    """Apply explicit human precedence without mutating machine artifacts.

    Confirmed decisions replace the machine shortlist with exactly the
    reviewed ids at the table's upper clip. Rejected decisions remove exactly
    the reviewed ids. Ambiguous decisions preserve machine scores. Every table
    receives one composite matcher version so downstream cannot mix reviewed
    and unreviewed score identities silently.
    """
    by_tracklet = {table.tracklet_id: table for table in tables}
    decisions = {item.tracklet_id: item for item in review.decisions}
    unknown = set(decisions) - set(by_tracklet)
    if unknown:
        raise LocalizationInputError(
            "identity review names a tracklet absent from compatibility "
            f"tables: {sorted(unknown)[0]!r}")
    machine_versions = {table.matcher_version for table in tables}
    if len(machine_versions) != 1:
        raise LocalizationInputError(
            "cannot apply human review to mixed machine matcher versions")
    machine_version = next(iter(machine_versions))
    combined_version = (
        f"{machine_version}+human_identity_review_v1:{review_version}")
    result = []
    for table in tables:
        decision = decisions.get(table.tracklet_id)
        entries = list(table.entries)
        status = table.status
        if decision is not None:
            if decision.decision == "confirmed":
                entries = [structs.CompatibilityEntry(
                    landmark_id=landmark_id, log_lr=float(table.clip_hi))
                           for landmark_id in sorted(decision.landmark_ids)]
            elif decision.decision == "rejected":
                rejected = set(decision.landmark_ids)
                entries = [entry for entry in entries
                           if entry.landmark_id not in rejected]
            elif decision.decision != "ambiguous":
                raise LocalizationInputError(
                    f"unsupported identity decision {decision.decision!r}")
            status = "refined"
        result.append(structs.CompatibilityTable(
            tracklet_id=table.tracklet_id,
            matcher_version=combined_version,
            entries=entries,
            default_log_lr=table.default_log_lr,
            clip_lo=table.clip_lo,
            clip_hi=table.clip_hi,
            status=status,
        ))
    return result, combined_version


def uninformative_tables(tracklet_ids: set[str], default_log_lr: float,
                         clip: float) -> list[structs.CompatibilityTable]:
    default_log_lr = _finite(default_log_lr, "default_log_compatibility")
    clip = _finite(clip, "compatibility_clip", positive=True)
    return [structs.CompatibilityTable(
        tracklet_id=tracklet_id,
        matcher_version=UNINFORMATIVE_MATCHER,
        entries=[],
        default_log_lr=default_log_lr,
        clip_lo=-clip,
        clip_hi=clip,
        status="fast") for tracklet_id in sorted(tracklet_ids)]


def landmark_entries(catalog_path: Path, anchor_lat: float, anchor_lon: float,
                     position_sigma_m: float) -> list[structs.LandmarkEntry]:
    """Load the whole catalog with one uniform explicit map uncertainty."""
    entries = catalog_lib.load_catalog(
        catalog_path, anchor_lat, anchor_lon,
        position_sigma_m=position_sigma_m, keep_hulls=True)
    frame = geo.RegionFrame(anchor_lat, anchor_lon)
    result = []
    for entry in entries:
        lat, lon = frame.latlon_from_enu(entry.east_m, entry.north_m)
        result.append(structs.LandmarkEntry(
            landmark_id=entry.landmark_id,
            lat_deg=float(lat),
            lon_deg=float(lon),
            type_key=type_key(entry.tags),
            position_sigma_m=entry.position_sigma_m,
            hull_east_m=[float(value) for value in entry.hull_east_m],
            hull_north_m=[float(value) for value in entry.hull_north_m]))
    return result


def _panorama_width(dataset_base: Path, frames: list[dataset_lib.Frame]) -> int:
    from PIL import Image

    widths = set()
    for frame in frames:
        path = Path(dataset_base) / "panorama" / f"{frame.pano_stem}.jpg"
        try:
            with Image.open(path) as image:
                widths.add(image.size[0])
        except OSError as exc:
            raise LocalizationInputError(f"cannot inspect {path}: {exc}") \
                from exc
    if len(widths) != 1:
        raise LocalizationInputError(
            f"dataset panoramas do not have one width: {sorted(widths)}")
    return widths.pop()


def _load_build_document(path: Path, dataset_name: str) -> dict:
    path = Path(path)
    if (path.name != build_config.BUILD_CONFIG_NAME or not path.is_file()
            or path.is_symlink()):
        raise LocalizationInputError(
            f"--build_config must name a regular, non-symlink "
            f"{build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(path.parent)
    if document["dataset"] != dataset_name:
        raise LocalizationInputError(
            f"build config belongs to {document['dataset']!r}, not "
            f"{dataset_name!r}")
    return document


def _config(document: dict, key: str):
    try:
        return build_config.value(document, key)
    except (build_config.MissingConfigValue,
            build_config.InvalidConfigValue) as exc:
        raise LocalizationInputError(str(exc)) from exc


def _require_configured_path(document: dict, key: str, supplied: Path) -> None:
    configured = Path(_config(document, key)).resolve()
    if Path(supplied).resolve() != configured:
        raise LocalizationInputError(
            f"--{key.rsplit('.', 1)[-1]} resolves to {Path(supplied).resolve()}, "
            f"but the immutable build config records {configured}")


def _require_input_path(document: dict, key: str, supplied: Path) -> Path:
    recorded = document["inputs"].get(key)
    resolved = Path(supplied).resolve()
    if not isinstance(recorded, str) or resolved != Path(recorded).resolve():
        raise LocalizationInputError(
            f"--{key} disagrees with immutable build input: {resolved} != "
            f"{Path(recorded or '').resolve()}")
    return resolved


def _flatten_config(value, prefix: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        return {prefix: value}
    result = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise LocalizationInputError(
                f"{prefix} contains a non-string or empty key")
        result.update(_flatten_config(child, f"{prefix}.{key}"))
    return result


def orchestration_contract(document: dict) -> dict:
    """Recompute the pipeline's exact localization-input config selection."""
    config = document.get("config")
    if not isinstance(config, dict):
        raise LocalizationInputError("build config has no config object")
    selected = {}
    for prefix in ("localization_inputs", "gps_course"):
        value = config.get(prefix)
        if not isinstance(value, dict):
            raise LocalizationInputError(
                f"build config does not record {prefix!r}")
        selected.update(_flatten_config(value, prefix))
    selected["artifacts.localization_inputs_version"] = build_config.value(
        document, "artifacts.localization_inputs_version")
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "localization_inputs",
        "config_digest": artifact.sha256_json(selected),
    }


def _nominal_forward_meta(path: Path,
                          calibration: nominal_forward.NominalForward) -> dict:
    return {
        "file": "nominal_forward.json",
        "source_path": str(Path(path).resolve()),
        "content_sha256": artifact.sha256_file(path),
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "dataset": calibration.dataset,
        "version": calibration.version,
        "mounting_id": calibration.mounting_id,
        "panorama_column": calibration.panorama_column,
        "panorama_width": calibration.panorama_width,
        "bearing_camera_cw_deg": calibration.bearing_camera_cw_deg,
        "uncertainty_deg": calibration.uncertainty_deg,
        "evidence_frame_ids": list(calibration.evidence_frame_ids),
        "operator": calibration.operator,
        "approved_at": calibration.approved_at,
        "notes": calibration.notes,
    }


def build(args) -> artifact.ArtifactRef:
    """Build and publish one localization_inputs artifact."""
    config_path = Path(args.build_config)
    document = _load_build_document(config_path, args.dataset)
    target_git_commit = provenance.git_commit()
    dataset_base = _require_input_path(
        document, "dataset_base", args.dataset_base)
    if dataset_base.is_symlink() or not dataset_base.is_dir():
        raise LocalizationInputError(
            f"--dataset_base must be a regular directory: {dataset_base}")
    try:
        dataset_digests = paths_lib.dataset_source_digests(dataset_base)
    except paths_lib.MissingInput as exc:
        raise LocalizationInputError(str(exc)) from exc
    mismatched_sources = [
        key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
        if document["inputs"].get(key) != dataset_digests[key]
    ]
    if mismatched_sources:
        raise LocalizationInputError(
            "dataset source bytes differ from the immutable build recipe: "
            f"{mismatched_sources}")
    orchestration = orchestration_contract(document)
    if args.orchestration_config_digest != orchestration["config_digest"]:
        raise LocalizationInputError(
            "--orchestration_config_digest does not match the immutable "
            "localization-inputs stage config selection")
    output_version = _config(
        document, "artifacts.localization_inputs_version")
    if Path(args.output_dir).name != output_version:
        raise LocalizationInputError(
            f"--output_dir must end in configured version {output_version!r}")

    metadata = dataset_lib.load_metadata(dataset_base)
    if metadata["dataset_name"] != args.dataset:
        raise LocalizationInputError(
            f"dataset metadata names {metadata['dataset_name']!r}, expected "
            f"{args.dataset!r}")
    dataset_lib.require_camera_frame_panoramas(metadata, dataset_base)
    _require_configured_path(
        document, "localization_inputs.motion_source", args.motion_source)
    _require_configured_path(
        document, "localization_inputs.nominal_forward_calibration",
        args.nominal_forward_calibration)
    motion_source = _require_input_path(
        document, "motion_source", args.motion_source)
    calibration_source = _require_input_path(
        document, "nominal_forward_calibration",
        args.nominal_forward_calibration)
    expected_motion = (dataset_base / "frames_gps.csv").resolve()
    if Path(args.motion_source).resolve() != expected_motion:
        raise LocalizationInputError(
            "the current motion contract consumes dataset frames_gps.csv; "
            f"the explicit source resolves to {Path(args.motion_source).resolve()}")

    motion_sha = artifact.sha256_file(motion_source)
    if motion_sha != document["inputs"].get("motion_source_sha256"):
        raise LocalizationInputError(
            "motion source bytes disagree with immutable build input")
    calibration_sha = artifact.sha256_file(calibration_source)
    if calibration_sha != document["inputs"].get("nominal_forward_sha256"):
        raise LocalizationInputError(
            "nominal-forward bytes disagree with immutable build input")

    frames = sorted(dataset_lib.load_frames(dataset_base),
                    key=lambda frame: frame.frame_idx)
    if len(frames) < 2:
        raise LocalizationInputError("localization needs at least two frames")
    if [frame.frame_idx for frame in frames] != list(range(len(frames))):
        raise LocalizationInputError("frame indices must be contiguous 0..N")
    if any(frame.time_s is None for frame in frames):
        raise LocalizationInputError("motion frames must carry timestamps")
    times = np.asarray([frame.time_s for frame in frames], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise LocalizationInputError(
            "motion timestamps must be finite and strictly increasing")
    anchor_lat, anchor_lon = dataset_lib.fill_enu(frames)
    east = np.asarray([frame.x_m for frame in frames], dtype=np.float64)
    north = np.asarray([frame.y_m for frame in frames], dtype=np.float64)

    calibration = nominal_forward.load(
        args.nominal_forward_calibration, expected_dataset=args.dataset)
    panorama_width = _panorama_width(dataset_base, frames)
    if calibration.panorama_width != panorama_width:
        raise LocalizationInputError(
            "nominal-forward panorama_width does not match this dataset: "
            f"{calibration.panorama_width} != {panorama_width}")

    (observations_ref, observations, accepted_ids, tracks_ref,
     audits_ref) = \
        load_observations(
            args.observations_dir, dataset_name=args.dataset,
            panorama_width=panorama_width,
            expected_versions={
                kind: _config(document, f"artifacts.{kind}_version")
                for kind in (paths_lib.BEARING_OBSERVATIONS,
                             paths_lib.OBJECT_TRACKS,
                             paths_lib.SEMANTIC_AUDITS)
            }, build_identity=document["build_identity"],
            target_build_dir=config_path.parent)
    for observation in observations:
        if observation.keyframe_idx >= len(frames):
            raise LocalizationInputError(
                f"observation keyframe {observation.keyframe_idx} lies "
                f"outside 0..{len(frames) - 1}")

    catalog_ref = artifact.open_artifact(
        args.catalog_dir, expected_kind=paths_lib.CATALOGS,
        expected_dataset=args.dataset,
        expected_version=_config(document, "artifacts.catalogs_version"))
    configured_lane.require(
        catalog_ref, document=document, kind=paths_lib.CATALOGS)
    if (catalog_ref.manifest_digest
            != document["inputs"].get("catalog_manifest_digest")
            or catalog_ref.content_digest
            != document["inputs"].get("catalog_content_digest")):
        raise LocalizationInputError(
            "catalog artifact identity disagrees with immutable build input")
    matching_candidate = artifact.open_artifact(
        args.matching_dir, expected_kind=paths_lib.LANDMARK_MATCHES,
        expected_dataset=args.dataset,
        expected_version=_config(
            document, "artifacts.landmark_matches_version"))
    configured_lane.require(
        matching_candidate, document=document,
        kind=paths_lib.LANDMARK_MATCHES)
    matching_ref, matched_tables, matched_version = load_matching(
        args.matching_dir,
        dataset_name=args.dataset,
        accepted_tracklet_ids=accepted_ids,
        tracks_ref=tracks_ref,
        audits_ref=audits_ref,
        catalog_ref=catalog_ref,
        expected_version=_config(
            document, "artifacts.landmark_matches_version"),
        build_identity=document["build_identity"])
    if matching_ref.to_dict() != matching_candidate.to_dict():
        raise LocalizationInputError(
            "matching loader did not retain the exact configured lane")

    config = dict(document["config"]["localization_inputs"])
    epoch_keyframes = _config(
        document, "localization_inputs.reducer_epoch_keyframes")
    camera_measurements = reduce_observations(observations, epoch_keyframes)
    measurements = forward_frame_measurements(
        camera_measurements, calibration)

    review_dir = getattr(args, "identity_review_dir", None)
    review_ref = None
    review = None
    if review_dir is not None:
        if _config(document, "localization_inputs.use_uninformative_tables"):
            raise LocalizationInputError(
                "--identity_review_dir cannot be combined with the "
                "uninformative-table control")
        try:
            review_ref, review = identity_review.load(
                review_dir, expected_matching_ref=matching_ref,
                matching_dir=args.matching_dir)
        except (identity_review.IdentityReviewError,
                artifact.ArtifactError) as error:
            raise LocalizationInputError(
                f"invalid human identity review: {error}") from error

    if _config(document, "localization_inputs.use_uninformative_tables"):
        tables = uninformative_tables(
            accepted_ids,
            _config(document,
                    "localization_inputs.default_log_compatibility"),
            _config(document, "localization_inputs.compatibility_clip"))
        matcher_version = UNINFORMATIVE_MATCHER
    else:
        tables = matched_tables
        matcher_version = matched_version
        if review is not None:
            tables, matcher_version = apply_identity_review(
                tables, review, review_ref.version)

    sigma_pair_m = _finite(
        _config(document, "localization_inputs.odometry_sigma_pair_m"),
        "odometry_sigma_pair_m", positive=True)
    displacement_gate_m = _finite(
        _config(document, "localization_inputs.displacement_gate_m"),
        "displacement_gate_m", positive=True)
    stationary_sigma_m = _finite(
        _config(document, "localization_inputs.stationary_sigma_m"),
        "stationary_sigma_m", positive=True)
    slow_yaw_sigma_deg = _finite(
        _config(document, "localization_inputs.slow_yaw_sigma_deg"),
        "slow_yaw_sigma_deg", positive=True)
    reverse_ranges = _config(
        document, "localization_inputs.reverse_keyframe_ranges")
    reverse_source = _config(
        document, "localization_inputs.reverse_annotation_source")
    course_min_displacement_m = _finite(
        _config(document, "gps_course.min_displacement_m"),
        "course_min_displacement_m", positive=True)
    course_smooth_window_s = _finite(
        _config(document, "gps_course.smooth_window_s"),
        "course_smooth_window_s")
    if course_smooth_window_s < 0.0:
        raise LocalizationInputError(
            "course_smooth_window_s must be nonnegative")

    odometry = gps_to_odometry.derive_increments(
        east, north,
        sigma_pair_m=sigma_pair_m,
        displacement_gate_m=displacement_gate_m,
        stationary_sigma_m=stationary_sigma_m,
        slow_yaw_sigma_deg=slow_yaw_sigma_deg,
        reverse_keyframe_ranges=reverse_ranges,
        extra_sigma_m=0.0,
        extra_yaw_sigma_deg=0.0,
        noise_seed=0)
    course_model = heading.gps_course_model_from_positions(
        east, north, times,
        min_displacement_m=course_min_displacement_m,
        smooth_window_s=course_smooth_window_s)
    if course_model is None:
        truth = []
        course_status = "abstained_insufficient_displacement"
    else:
        truth = [structs.TruthPose(
            keyframe_idx=index,
            east_m=float(east[index]),
            north_m=float(north[index]),
            course_world_cw_deg=float(course_model.course_world_cw_deg_at(
                times[index])) % 360.0)
                 for index in range(len(frames))]
        course_status = "gps_course_diagnostic_only"

    position_sigma_m = _finite(
        args.landmark_position_sigma_m,
        "landmark_position_sigma_m", positive=True)
    configured_position_sigma_m = _finite(
        _config(document, "localization_inputs.landmark_position_sigma_m"),
        "configured landmark_position_sigma_m", positive=True)
    if position_sigma_m != configured_position_sigma_m:
        raise LocalizationInputError(
            "--landmark_position_sigma_m disagrees with the immutable build "
            f"config: {position_sigma_m} != {configured_position_sigma_m}")
    landmarks = landmark_entries(
        Path(args.catalog_dir) / CATALOG_NAME, anchor_lat, anchor_lon,
        position_sigma_m)
    known_landmarks = {entry.landmark_id for entry in landmarks}
    for table in tables:
        unknown = [entry.landmark_id for entry in table.entries
                   if entry.landmark_id not in known_landmarks]
        if unknown:
            raise LocalizationInputError(
                f"table {table.tracklet_id!r} refers to catalog-absent "
                f"landmark {unknown[0]!r}")

    max_visible_range_m = _finite(
        _config(document, "localization_inputs.max_visible_range_m"),
        "max_visible_range_m", positive=True)
    nominal_meta = _nominal_forward_meta(
        args.nominal_forward_calibration, calibration)
    meta = {
        "schema_version": export_ingest.EXPORT_SCHEMA,
        "message_schema_version": structs.SCHEMA_VERSION,
        "dataset": args.dataset,
        "scenario_name": _config(document, "experiment.name"),
        "anchor_lat_deg": anchor_lat,
        "anchor_lon_deg": anchor_lon,
        "n_keyframes": len(frames),
        "matcher_version": matcher_version,
        "matching_coverage": "complete",
        "max_visible_range_m": max_visible_range_m,
        "landmark_position_sigma_m": position_sigma_m,
        "nominal_forward": nominal_meta,
        "motion": {
            "file": "motion_source.csv",
            "source_path": str(Path(args.motion_source).resolve()),
            "content_sha256": motion_sha,
            "course_heading_status": course_status,
            "reverse_annotation_source": reverse_source,
        },
        "reducer": {
            "name": "epoch_fused_compat_v1",
            "epoch_keyframes": epoch_keyframes,
            "input_frame": "camera_cw_deg",
            "output_frame": "nominal_forward_cw_deg",
        },
    }
    outputs = [
        "export_meta.json",
        "landmarks.json",
        "motion_source.csv",
        "nominal_forward.json",
        "tier1_measurements.jsonl",
        "tier1_odometry.jsonl",
        "tier1_tables.json",
        "truth.jsonl",
    ]
    if review_ref is not None:
        outputs.append(identity_review.REVIEW_NAME)
    with publication.published_artifact(
            args.output_dir,
            kind=paths_lib.LOCALIZATION_INPUTS,
            dataset=args.dataset,
            version=output_version,
            generator=GENERATOR,
            git_commit=target_git_commit,
            arguments=sys.argv,
            upstreams=(observations_ref, matching_ref, catalog_ref)
            + (() if review_ref is None else (review_ref,)),
            config={
                "orchestration": orchestration,
                "build_identity": document["build_identity"],
                "localization_inputs": config,
                "gps_course": dict(document["config"]["gps_course"]),
                "nominal_forward_sha256": nominal_meta["content_sha256"],
                "motion_source_sha256": motion_sha,
                "identity_review": (None if review_ref is None else {
                    "artifact": review_ref.to_dict(),
                    "content_digest": review_ref.content_digest,
                    "precedence_policy": "human_identity_over_machine_v1",
                    "n_decisions": len(review.decisions),
                }),
                "source_digests": {
                    "build_config": artifact.sha256_file(config_path),
                    **dataset_digests,
                    "motion_source": motion_sha,
                    "nominal_forward": calibration_sha,
                    paths_lib.BEARING_OBSERVATIONS:
                        observations_ref.content_digest,
                    paths_lib.LANDMARK_MATCHES:
                        matching_ref.content_digest,
                    paths_lib.CATALOGS: catalog_ref.content_digest,
                    identity_review.IDENTITY_REVIEW_KIND: (
                        None if review_ref is None
                        else review_ref.content_digest),
                },
                "matching_coverage": "complete",
                "matching_n_expected": artifact.load_manifest(
                    args.matching_dir).config["n_expected"],
                "matching_n_successful": artifact.load_manifest(
                    args.matching_dir).config["n_successful"],
                "reducer": meta["reducer"],
            },
            declared_outputs=tuple(outputs)) as builder:
        artifact.atomic_write_json(builder.output_path("export_meta.json"), meta)
        artifact.atomic_write_file(
            builder.output_path("landmarks.json"),
            msgspec.json.encode(landmarks, enc_hook=msgspec_enc_hook))
        artifact.atomic_write_file(
            builder.output_path("tier1_tables.json"),
            msgspec.json.encode(tables, enc_hook=msgspec_enc_hook))
        for name, records in (
                ("tier1_measurements.jsonl", measurements),
                ("tier1_odometry.jsonl", odometry),
                ("truth.jsonl", truth)):
            run_io.write_jsonl(builder.output_path(name), records)
        shutil.copyfile(
            motion_source, builder.output_path("motion_source.csv"))
        shutil.copyfile(
            calibration_source,
            builder.output_path("nominal_forward.json"))
        if review_ref is not None:
            shutil.copyfile(
                Path(review_dir) / identity_review.REVIEW_NAME,
                builder.output_path(identity_review.REVIEW_NAME))
    assert builder.artifact_ref is not None
    # Exercise the consumer boundary before reporting success.
    export_ingest.load(args.output_dir, expected_dataset=args.dataset)
    return builder.artifact_ref


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--observations_dir", type=Path, required=True)
    parser.add_argument("--matching_dir", type=Path, required=True)
    parser.add_argument(
        "--identity_review_dir", type=Path,
        help="Optional immutable human identity review bound to the exact "
             "matching artifact; decisive reviews override machine tables")
    parser.add_argument("--catalog_dir", type=Path, required=True)
    parser.add_argument("--motion_source", type=Path, required=True)
    parser.add_argument("--nominal_forward_calibration", type=Path,
                        required=True)
    parser.add_argument("--landmark_position_sigma_m", type=float,
                        required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    args = parser.parse_args()
    try:
        ref = build(args)
    except (LocalizationInputError, artifact.ArtifactError,
            dataset_lib.ContractViolation, ValueError) as exc:
        parser.error(str(exc))
    print(f"published immutable localization inputs: {ref.path}")


if __name__ == "__main__":
    main()
