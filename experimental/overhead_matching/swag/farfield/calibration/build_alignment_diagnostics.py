"""Build review-only camera/GPS-course alignment diagnostics.

This stage consumes the lossless ``bearing_observations`` artifact and the
frozen dataset sources to produce two independent *candidates* for the
camera-frame bearing of GPS course: a static-landmark triangulation sweep and
an optional absolute sun check.  Its output is evidence for a human review;
it is deliberately not a nominal-forward calibration and is never an input
to localization.

The GPS-course parameters are read from the object-tracks manifest bound by
the observations artifact.  They are not re-selected from CLI flags or from
the current dataset metadata.  This preserves the exact course model under
which the source bearings were tracked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    dataset,
    geometry,
    nominal_forward,
    paths as paths_lib,
    publication,
    provenance,
    stage_reuse,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    heading,
    mount_offset_sweep as sweep_lib,
    sun_offset_check as sun_lib,
)


GENERATOR = ("//experimental/overhead_matching/swag/farfield/calibration:"
             "build_alignment_diagnostics")
SCHEMA = "farfield_alignment_diagnostics/v1"
OUTPUT_NAME = "alignment_diagnostics.json"
SUN_REVIEW_NAME = "sun_review_contact_sheet.jpg"
RESULT_KIND = "effective_camera_to_course_v1"
RESULT_FRAME = "camera_centre_column_cw_positive"
RESULT_FIELD = "effective_camera_to_course_offset_deg"
AUTHORITY = {
    "classification": "diagnostic_only",
    "calibration_use": "prohibited",
    "automatic_promotion": "prohibited",
    "localization_use": "prohibited",
}

_OBSERVATION_KEYS = frozenset({
    "tracklet_id",
    "keyframe_idx",
    "bearing_camera_cw_deg",
    "angular_width_deg",
    "sigma_deg",
    "correlation_group",
})
_OBSERVATION_CONFIG_KEYS = frozenset({
    "orchestration",
    "build_identity",
    "schema",
    "pano_width",
    "bearing_sigma_deg",
    "n_accepted_tracklets",
    "n_observations",
    "coverage",
    "source_digests",
})
_OBSERVATION_SOURCE_KEYS = frozenset({
    "build_config",
    "dataset_tracking_inputs",
    paths_lib.OBJECT_TRACKS,
    paths_lib.SEMANTIC_AUDITS,
})
_SUN_KEYS = frozenset({
    "n_frames", "min_speed_mps", "elevation_tolerance_deg", "work_width",
})
_SWEEP_KEYS = frozenset({
    "coarse_step_deg", "fine_step_deg", "fine_halfwidth_deg",
    "min_observations", "min_arc_deg", "max_condition", "min_tracklets",
    "min_support_frac",
})
_COURSE_KEYS = frozenset({"min_displacement_m", "smooth_window_s"})


class AlignmentDiagnosticError(ValueError):
    """An input cannot support a provenance-safe diagnostic artifact."""


def _exact_keys(value: Any, expected: frozenset[str], where: str) -> dict:
    if not isinstance(value, dict):
        raise AlignmentDiagnosticError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise AlignmentDiagnosticError(
            f"{where} has missing={missing}, unknown={unknown}")
    return value


def _finite(value: Any, where: str, *, minimum: float | None = None,
            maximum: float | None = None, strictly_positive: bool = False) \
        -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentDiagnosticError(f"{where} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise AlignmentDiagnosticError(f"{where} must be finite")
    if strictly_positive and result <= 0.0:
        raise AlignmentDiagnosticError(f"{where} must be positive")
    if minimum is not None and result < minimum:
        raise AlignmentDiagnosticError(f"{where} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise AlignmentDiagnosticError(f"{where} must be <= {maximum}")
    return result


def _positive_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AlignmentDiagnosticError(f"{where} must be a positive integer")
    return value


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {prefix: value}
    result = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise AlignmentDiagnosticError(
                f"{prefix} contains an empty or non-string key")
        result.update(_flatten(child, f"{prefix}.{key}"))
    return result


def orchestration_contract(document: dict) -> dict:
    """Recompute the pipeline's exact diagnostics-stage config selection."""
    config = document.get("config")
    if not isinstance(config, dict):
        raise AlignmentDiagnosticError("build config has no config object")
    diagnostics = config.get("alignment_diagnostics")
    if not isinstance(diagnostics, dict):
        raise AlignmentDiagnosticError(
            "build config does not record alignment_diagnostics")
    selected = _flatten(diagnostics, "alignment_diagnostics")
    selected["localization_inputs.nominal_forward_calibration"] = \
        build_config.value(
            document, "localization_inputs.nominal_forward_calibration")
    selected["artifacts.alignment_diagnostics_version"] = build_config.value(
        document, "artifacts.alignment_diagnostics_version")
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "diagnostics",
        "config_digest": artifact.sha256_json(selected),
    }


def _same_path(actual: Path, recorded: str, what: str) -> Path:
    resolved = Path(actual).resolve()
    if resolved != Path(recorded).resolve():
        raise AlignmentDiagnosticError(
            f"{what} disagrees with immutable build config: {resolved} != "
            f"{Path(recorded).resolve()}")
    return resolved


def dataset_source_digest(dataset_base: Path) -> str:
    """Identity of metadata, GPS fixes, and camera-frame panoramas.

    This intentionally matches the object-tracks producer's source digest.
    Comparing the result prevents a keyframe join against dataset bytes other
    than those used to build the observations' recorded source tracks.
    """
    try:
        return artifact.sha256_json(
            paths_lib.dataset_source_digests(dataset_base))
    except paths_lib.MissingInput as error:
        raise AlignmentDiagnosticError(str(error)) from error


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AlignmentDiagnosticError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_observations(observations_dir: Path, manifest: artifact.ArtifactManifest,
                      *, n_frames: int) -> list[dict]:
    """Strictly decode the single canonical observation stream."""
    if manifest.declared_outputs != ("observations.jsonl",):
        raise AlignmentDiagnosticError(
            "bearing_observations must declare only observations.jsonl")
    expected_config_keys = _OBSERVATION_CONFIG_KEYS | (
        frozenset({"stage_reuse"})
        if "stage_reuse" in manifest.config else frozenset())
    config = _exact_keys(
        manifest.config, expected_config_keys,
        "bearing_observations manifest config")
    if config["schema"] != "farfield_bearing_observations/v1":
        raise AlignmentDiagnosticError(
            "unsupported bearing_observations payload schema")
    if config["coverage"] != "complete":
        raise AlignmentDiagnosticError(
            "bearing_observations coverage must be complete")
    build_identity = config["build_identity"]
    if (not isinstance(build_identity, str) or len(build_identity) != 64
            or any(character not in "0123456789abcdef"
                   for character in build_identity)):
        raise AlignmentDiagnosticError(
            "bearing_observations build_identity must be lowercase SHA-256")
    sources = _exact_keys(
        config["source_digests"], _OBSERVATION_SOURCE_KEYS,
        "bearing_observations source_digests")
    for name, value in sources.items():
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            raise AlignmentDiagnosticError(
                f"bearing_observations source_digests.{name} must be "
                "lowercase SHA-256")
    orchestration = _exact_keys(
        config["orchestration"],
        frozenset({"schema", "stage", "config_digest"}),
        "bearing_observations orchestration")
    if (orchestration["schema"] != "farfield_pipeline_stage/v1"
            or orchestration["stage"] != "bearings"):
        raise AlignmentDiagnosticError(
            "bearing_observations has an invalid producer stage contract")
    digest = orchestration["config_digest"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise AlignmentDiagnosticError(
            "bearing_observations config_digest must be lowercase SHA-256")
    _positive_integer(config["pano_width"],
                      "bearing_observations pano_width")
    _finite(config["bearing_sigma_deg"],
            "bearing_observations bearing_sigma_deg", strictly_positive=True)
    expected_count = config["n_observations"]
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) \
            or expected_count < 0:
        raise AlignmentDiagnosticError(
            "bearing_observations n_observations must be nonnegative integer")
    expected_tracks = config["n_accepted_tracklets"]
    if isinstance(expected_tracks, bool) or not isinstance(expected_tracks, int) \
            or expected_tracks < 0:
        raise AlignmentDiagnosticError(
            "bearing_observations n_accepted_tracklets must be nonnegative integer")

    path = Path(observations_dir) / "observations.jsonl"
    records = []
    seen = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise AlignmentDiagnosticError(
            f"cannot read canonical observations {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: blank JSONL record")
        try:
            record = json.loads(
                line, object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    AlignmentDiagnosticError(
                        f"invalid non-finite JSON constant {value!r}")))
        except AlignmentDiagnosticError:
            raise
        except json.JSONDecodeError as error:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: malformed JSON: {error}") from error
        _exact_keys(record, _OBSERVATION_KEYS,
                    f"{path}:{line_number} observation")
        tracklet_id = record["tracklet_id"]
        if not isinstance(tracklet_id, str) or not tracklet_id:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: tracklet_id must be non-empty")
        keyframe = record["keyframe_idx"]
        if isinstance(keyframe, bool) or not isinstance(keyframe, int) \
                or not 0 <= keyframe < n_frames:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: keyframe_idx {keyframe!r} is outside "
                f"the dataset's 0..{n_frames - 1} range")
        bearing = _finite(record["bearing_camera_cw_deg"],
                          f"{path}:{line_number} bearing_camera_cw_deg")
        if not 0.0 <= bearing < 360.0:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: bearing_camera_cw_deg must be in "
                "[0, 360)")
        _finite(record["angular_width_deg"],
                f"{path}:{line_number} angular_width_deg",
                strictly_positive=True, maximum=360.0)
        sigma = _finite(record["sigma_deg"],
                        f"{path}:{line_number} sigma_deg",
                        strictly_positive=True)
        if not math.isclose(sigma, float(config["bearing_sigma_deg"]),
                            rel_tol=0.0, abs_tol=1e-12):
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: sigma_deg disagrees with manifest")
        group = record["correlation_group"]
        if not isinstance(group, str) or not group:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: correlation_group must be non-empty")
        key = (tracklet_id, keyframe)
        if key in seen:
            raise AlignmentDiagnosticError(
                f"{path}:{line_number}: duplicate tracklet/keyframe {key!r}")
        seen.add(key)
        records.append(record)
    if len(records) != expected_count:
        raise AlignmentDiagnosticError(
            "observations.jsonl count disagrees with manifest: "
            f"{len(records)} != {expected_count}")
    if len({record["tracklet_id"] for record in records}) != expected_tracks:
        raise AlignmentDiagnosticError(
            "observed tracklet coverage disagrees with manifest")
    ordered = sorted(records,
                     key=lambda item: (item["tracklet_id"],
                                       item["keyframe_idx"]))
    if records != ordered:
        raise AlignmentDiagnosticError(
            "observations.jsonl must be canonically ordered by global "
            "tracklet_id and keyframe_idx")
    return records


def _diagnostic_config(document: dict) -> dict:
    value = document.get("config", {}).get("alignment_diagnostics")
    value = _exact_keys(value, frozenset({"sun", "sweep"}),
                        "build config alignment_diagnostics")
    sun = dict(_exact_keys(value["sun"], _SUN_KEYS,
                           "build config alignment_diagnostics.sun"))
    sweep = dict(_exact_keys(value["sweep"], _SWEEP_KEYS,
                             "build config alignment_diagnostics.sweep"))
    _positive_integer(sun["n_frames"], "alignment_diagnostics.sun.n_frames")
    _finite(sun["min_speed_mps"],
            "alignment_diagnostics.sun.min_speed_mps", minimum=0.0)
    _finite(sun["elevation_tolerance_deg"],
            "alignment_diagnostics.sun.elevation_tolerance_deg",
            strictly_positive=True, maximum=90.0)
    if _positive_integer(sun["work_width"],
                         "alignment_diagnostics.sun.work_width") < 2:
        raise AlignmentDiagnosticError(
            "alignment_diagnostics.sun.work_width must be >= 2")
    for key in ("coarse_step_deg", "fine_step_deg", "fine_halfwidth_deg",
                "max_condition"):
        _finite(sweep[key], f"alignment_diagnostics.sweep.{key}",
                strictly_positive=True)
    _positive_integer(sweep["min_observations"],
                      "alignment_diagnostics.sweep.min_observations")
    _finite(sweep["min_arc_deg"],
            "alignment_diagnostics.sweep.min_arc_deg", minimum=0.0,
            maximum=360.0)
    _positive_integer(sweep["min_tracklets"],
                      "alignment_diagnostics.sweep.min_tracklets")
    _finite(sweep["min_support_frac"],
            "alignment_diagnostics.sweep.min_support_frac", minimum=0.0,
            maximum=1.0)
    return {"sun": sun, "sweep": sweep}


def _recorded_course(tracks_manifest: artifact.ArtifactManifest) -> dict:
    config = tracks_manifest.config
    if config.get("schema") != "farfield_object_tracks/v1":
        raise AlignmentDiagnosticError(
            "bearing observations source is not object_tracks schema v1")
    if config.get("coverage") != "complete":
        raise AlignmentDiagnosticError("source object_tracks is not complete")
    resolved = config.get("resolved")
    if not isinstance(resolved, dict):
        raise AlignmentDiagnosticError(
            "source object_tracks does not record resolved settings")
    course = dict(_exact_keys(
        resolved.get("gps_course"), _COURSE_KEYS,
        "source object_tracks resolved.gps_course"))
    _finite(course["min_displacement_m"],
            "recorded gps_course.min_displacement_m", strictly_positive=True)
    _finite(course["smooth_window_s"],
            "recorded gps_course.smooth_window_s", minimum=0.0)
    return course


def _load_inputs(args) -> dict:
    config_path = Path(args.build_config)
    if (config_path.name != build_config.BUILD_CONFIG_NAME
            or not config_path.is_file() or config_path.is_symlink()):
        raise AlignmentDiagnosticError(
            f"--build_config must name a regular, non-symlink "
            f"{build_config.BUILD_CONFIG_NAME}")
    document = build_config.load(config_path.parent)
    if document["dataset"] != args.dataset:
        raise AlignmentDiagnosticError(
            "--dataset disagrees with immutable build config")
    dataset_base = _same_path(
        args.dataset_base, document["inputs"].get("dataset_base", ""),
        "--dataset_base")
    if dataset_base.is_symlink() or not dataset_base.is_dir():
        raise AlignmentDiagnosticError(
            f"--dataset_base must be a regular directory: {dataset_base}")
    metadata = dataset.load_metadata(dataset_base)
    if metadata["dataset_name"] != args.dataset:
        raise AlignmentDiagnosticError(
            "dataset metadata disagrees with --dataset")
    dataset.require_camera_frame_panoramas(metadata, dataset_base)
    try:
        dataset_digests = paths_lib.dataset_source_digests(dataset_base)
    except paths_lib.MissingInput as error:
        raise AlignmentDiagnosticError(str(error)) from error
    mismatched_sources = [
        key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
        if document["inputs"].get(key) != dataset_digests[key]
    ]
    if mismatched_sources:
        raise AlignmentDiagnosticError(
            "dataset source bytes differ from the immutable build recipe: "
            f"{mismatched_sources}")

    output_version = build_config.value(
        document, "artifacts.alignment_diagnostics_version")
    if Path(args.output_dir).name != output_version:
        raise AlignmentDiagnosticError(
            f"--output_dir must end in configured version {output_version!r}")
    orchestration = orchestration_contract(document)
    if args.orchestration_config_digest != orchestration["config_digest"]:
        raise AlignmentDiagnosticError(
            "--orchestration_config_digest does not match the immutable "
            "diagnostics-stage config selection")
    calibration_path = _same_path(
        args.nominal_forward_calibration,
        build_config.value(
            document, "localization_inputs.nominal_forward_calibration"),
        "--nominal_forward_calibration")
    if calibration_path.is_symlink() or not calibration_path.is_file():
        raise AlignmentDiagnosticError(
            "--nominal_forward_calibration must be a regular, non-symlink "
            f"file: {calibration_path}")
    try:
        approved_nominal_forward = nominal_forward.load(
            calibration_path, expected_dataset=args.dataset)
    except ValueError as error:
        raise AlignmentDiagnosticError(
            f"invalid approved nominal-forward record: {error}") from error
    calibration_sha256 = artifact.sha256_file(calibration_path)

    observations_ref = artifact.open_artifact(
        args.observations_dir,
        expected_kind=paths_lib.BEARING_OBSERVATIONS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.bearing_observations_version"))
    authorization = stage_reuse.load_proof(config_path.parent)
    stage_reuse.require_target_checkout(
        config_path.parent, document=document, authorization=authorization)
    observations_manifest = stage_reuse.require_configured_artifact(
        observations_ref, target_build_dir=config_path.parent,
        kind=paths_lib.BEARING_OBSERVATIONS, document=document)
    if observations_manifest.config.get(
            "build_identity") != document["build_identity"]:
        raise AlignmentDiagnosticError(
            "bearing_observations belongs to a different immutable build")
    if len(observations_manifest.upstreams) != 2:
        raise AlignmentDiagnosticError(
            "bearing_observations must bind object_tracks and semantic_audits")
    tracks_recorded, audits_recorded = observations_manifest.upstreams
    if (tracks_recorded.kind != paths_lib.OBJECT_TRACKS
            or audits_recorded.kind != paths_lib.SEMANTIC_AUDITS
            or tracks_recorded.dataset != args.dataset
            or audits_recorded.dataset != args.dataset):
        raise AlignmentDiagnosticError(
            "bearing_observations upstream order must be object_tracks, "
            "semantic_audits for this dataset")
    tracks_ref = artifact.open_artifact(
        tracks_recorded.path, expected_kind=paths_lib.OBJECT_TRACKS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.object_tracks_version"))
    if tracks_ref.to_dict() != tracks_recorded.to_dict():
        raise AlignmentDiagnosticError(
            "recorded object_tracks path no longer resolves to the bound "
            "artifact identity")
    audits_ref = artifact.open_artifact(
        audits_recorded.path, expected_kind=paths_lib.SEMANTIC_AUDITS,
        expected_dataset=args.dataset,
        expected_version=build_config.value(
            document, "artifacts.semantic_audits_version"))
    if audits_ref.to_dict() != audits_recorded.to_dict():
        raise AlignmentDiagnosticError(
            "recorded semantic_audits path no longer resolves to the bound "
            "artifact identity")
    tracks_manifest = stage_reuse.require_configured_artifact(
        tracks_ref, target_build_dir=config_path.parent,
        kind=paths_lib.OBJECT_TRACKS, document=document)
    audits_manifest = stage_reuse.require_configured_artifact(
        audits_ref, target_build_dir=config_path.parent,
        kind=paths_lib.SEMANTIC_AUDITS, document=document)
    track_bridge = stage_reuse.require_compatible_artifact(
        tracks_ref, tracks_manifest, target_build_dir=config_path.parent,
        owner_stage="track", authorization=authorization)
    stage_reuse.require_recorded_bridge(
        observations_manifest.config.get("stage_reuse"), track_bridge)
    stage_reuse.require_recorded_bridge(
        audits_manifest.config.get("stage_reuse"), track_bridge,
        required_artifacts=(tracks_ref,),
        additional_artifacts=tuple(
            reference for reference in (authorization.refs
                                         if authorization is not None else ())
            if reference.kind == paths_lib.FRAME_LANDMARKS))
    if audits_manifest.config.get("build_identity") != document["build_identity"]:
        raise AlignmentDiagnosticError(
            f"{paths_lib.SEMANTIC_AUDITS} belongs to a different immutable build")
    if sum(ref.to_dict() == tracks_ref.to_dict()
           for ref in audits_manifest.upstreams) != 1:
        raise AlignmentDiagnosticError(
            "semantic_audits must bind the exact object_tracks artifact once")
    course = _recorded_course(tracks_manifest)
    source_digests = tracks_manifest.config.get("source_digests")
    if not isinstance(source_digests, dict):
        raise AlignmentDiagnosticError(
            "source object_tracks does not record source_digests")
    recorded_dataset_digest = source_digests.get("dataset_tracking_inputs")
    current_dataset_digest = dataset_source_digest(dataset_base)
    if recorded_dataset_digest != current_dataset_digest:
        raise AlignmentDiagnosticError(
            "dataset metadata/GPS/panorama bytes disagree with the source "
            "object_tracks artifact")
    observation_sources = observations_manifest.config["source_digests"]
    expected_observation_sources = {
        "build_config": artifact.sha256_file(config_path),
        "dataset_tracking_inputs": current_dataset_digest,
        paths_lib.OBJECT_TRACKS: tracks_ref.content_digest,
        paths_lib.SEMANTIC_AUDITS: audits_ref.content_digest,
    }
    if observation_sources != expected_observation_sources:
        raise AlignmentDiagnosticError(
            "bearing_observations source digests disagree with its exact "
            "build and upstream artifacts")

    frames = dataset.load_frames(dataset_base)
    if not frames:
        raise AlignmentDiagnosticError("dataset has no panorama frames")
    dataset.fill_enu(frames)
    records = load_observations(
        args.observations_dir, observations_manifest, n_frames=len(frames))
    expected_pano_width = observations_manifest.config["pano_width"]
    for frame in frames:
        pano_path = dataset_base / "panorama" / f"{frame.pano_stem}.jpg"
        try:
            with Image.open(pano_path) as image:
                actual_width = image.size[0]
        except OSError as error:
            raise AlignmentDiagnosticError(
                f"cannot inspect source panorama {pano_path}: {error}") from error
        if actual_width != expected_pano_width:
            raise AlignmentDiagnosticError(
                "bearing_observations pano_width disagrees with source "
                f"panorama {pano_path.name}: {expected_pano_width} != "
                f"{actual_width}")
    return {
        "document": document,
        "build_config_sha256": artifact.sha256_file(config_path),
        "dataset_base": dataset_base,
        "metadata": metadata,
        "frames": frames,
        "records": records,
        "observations_ref": observations_ref,
        "observations_manifest": observations_manifest,
        "tracks_ref": tracks_ref,
        "course": course,
        "diagnostics": _diagnostic_config(document),
        "dataset_source_sha256": current_dataset_digest,
        "output_version": output_version,
        "orchestration": orchestration,
        "nominal_forward": approved_nominal_forward,
        "nominal_forward_path": calibration_path,
        "nominal_forward_sha256": calibration_sha256,
        "stage_reuse": track_bridge,
        "reuse_authorization": authorization,
    }


def _course_model(frames: list[dataset.Frame], course: dict):
    return heading.gps_course_model_from_positions(
        [frame.x_m for frame in frames], [frame.y_m for frame in frames],
        [frame.time_s for frame in frames],
        min_displacement_m=course["min_displacement_m"],
        smooth_window_s=course["smooth_window_s"])


def _no_candidate(method: str, reason: str, evidence: dict) -> dict:
    return {
        "method": method,
        "result_kind": RESULT_KIND,
        "frame": RESULT_FRAME,
        "status": "no_candidate",
        "reason": reason,
        "evidence": evidence,
    }


def build_sweep_candidate(records: list[dict], frames: list[dataset.Frame],
                          model, config: dict) -> dict:
    """Run the relative static-landmark diagnostic on strict observations."""
    if model is None:
        return _no_candidate(
            "static_landmark_triangulation",
            "recorded GPS-course model abstained for inadequate displacement",
            {"n_observations": len(records), "n_tracklets": len({
                item["tracklet_id"] for item in records})})
    by_tracklet = defaultdict(list)
    for record in records:
        frame = frames[record["keyframe_idx"]]
        by_tracklet[record["tracklet_id"]].append((
            frame.x_m,
            frame.y_m,
            record["bearing_camera_cw_deg"],
            float(model.course_world_cw_deg_at(frame.time_s)),
            record["keyframe_idx"],
        ))
    enough = {
        key: values for key, values in by_tracklet.items()
        if len(values) >= config["min_observations"]
    }
    kept = {
        key: values for key, values in enough.items()
        if sweep_lib.arc_deg(values) >= config["min_arc_deg"]
    }
    evidence = {
        "n_input_observations": len(records),
        "n_input_tracklets": len(by_tracklet),
        "n_tracklets_meeting_observation_count": len(enough),
        "n_tracklets_meeting_course_plus_camera_arc": len(kept),
    }
    if not kept:
        widest = max((sweep_lib.arc_deg(values)
                      for values in enough.values()), default=None)
        if widest is not None:
            evidence["widest_course_plus_camera_arc_deg"] = round(widest, 6)
        return _no_candidate(
            "static_landmark_triangulation",
            "no tracklet meets the configured observation-count and "
            "course-plus-camera arc gates", evidence)

    coarse = sweep_lib.sweep(
        kept, config["max_condition"], 0.0, 360.0,
        config["coarse_step_deg"])
    if not coarse:
        return _no_candidate(
            "static_landmark_triangulation",
            "every course-alignment candidate failed triangulation", evidence)
    coarse_eligible, support_floor = sweep_lib.eligible(
        coarse, config["min_support_frac"])
    if not coarse_eligible:
        return _no_candidate(
            "static_landmark_triangulation",
            "no course-alignment candidate meets the support gate", evidence)
    coarse_best = min(coarse_eligible, key=lambda item: item[1])
    fine = sweep_lib.sweep(
        kept, config["max_condition"],
        coarse_best[0] - config["fine_halfwidth_deg"],
        coarse_best[0] + config["fine_halfwidth_deg"]
        + config["fine_step_deg"],
        config["fine_step_deg"])
    fine_eligible = [item for item in fine if item[2] >= support_floor]
    best = min(fine_eligible or [coarse_best], key=lambda item: item[1])
    candidate, residual, n_support = best
    verdict, detail, internally_supported = sweep_lib.assess(
        coarse_eligible, min(item[1] for item in coarse_eligible), n_support,
        config["min_tracklets"])
    evidence.update({
        "support_floor_tracklets": support_floor,
        "winning_well_conditioned_tracklets": n_support,
        "median_course_plus_camera_arc_deg": round(float(np.median([
            sweep_lib.arc_deg(values) for values in kept.values()])), 6),
        "winning_median_bearing_residual_deg": round(residual, 9),
        "curve_assessment": {
            "classification": verdict.lower().replace(" ", "_"),
            "internally_supported": internally_supported,
            "detail": detail,
        },
        "coarse_curve": [{
            RESULT_FIELD: round(angle, 9),
            "median_bearing_residual_deg": round(value, 9),
            "n_well_conditioned_tracklets": count,
            "meets_support_gate": count >= support_floor,
        } for angle, value, count in coarse],
        "fine_curve": [{
            RESULT_FIELD: round(angle, 9),
            "median_bearing_residual_deg": round(value, 9),
            "n_well_conditioned_tracklets": count,
            "meets_support_gate": count >= support_floor,
        } for angle, value, count in fine],
    })
    return {
        "method": "static_landmark_triangulation",
        "result_kind": RESULT_KIND,
        "frame": RESULT_FRAME,
        "status": "candidate_reported",
        RESULT_FIELD: round(candidate % 360.0, 9),
        "evidence": evidence,
    }


def build_sun_candidate(dataset_base: Path, metadata: dict,
                        frames: list[dataset.Frame], model,
                        config: dict) -> dict:
    """Run the optional absolute sun diagnostic without publishing authority."""
    if model is None:
        return _no_candidate(
            "solar_ephemeris",
            "recorded GPS-course model abstained for inadequate displacement",
            {"n_dataset_frames": len(frames)})
    try:
        start = sun_lib.log_start_utc(metadata)
    except (TypeError, ValueError) as error:
        raise AlignmentDiagnosticError(
            f"invalid dataset log_start_utc: {error}") from error
    if start is None:
        return _no_candidate(
            "solar_ephemeris",
            "dataset metadata has no absolute log_start_utc",
            {"n_dataset_frames": len(frames)})

    moving = []
    for previous, frame in zip(frames, frames[1:]):
        dt = frame.time_s - previous.time_s
        if dt <= 0.0:
            raise AlignmentDiagnosticError(
                "dataset frame times must be strictly increasing")
        speed = math.hypot(frame.x_m - previous.x_m,
                           frame.y_m - previous.y_m) / dt
        if speed >= config["min_speed_mps"]:
            moving.append(frame)
    if not moving:
        return _no_candidate(
            "solar_ephemeris",
            "no frame meets the configured GPS speed gate",
            {"n_dataset_frames": len(frames), "n_frames_meeting_speed": 0})
    step = max(1, len(moving) // config["n_frames"])
    sampled = moving[::step][:config["n_frames"]]
    loaded = []
    for frame in sampled:
        when = start + timedelta(seconds=frame.time_s)
        sun_world_cw_deg, sun_elevation_deg = sun_lib.solar_position(
            when, frame.lat, frame.lon)
        if sun_elevation_deg <= 5.0:
            continue
        image_path = dataset_base / "panorama" / f"{frame.pano_stem}.jpg"
        try:
            with Image.open(image_path) as source:
                source_width, source_height = source.size
                grey = source.convert("L").resize(
                    (config["work_width"], config["work_width"] // 2),
                    Image.Resampling.BILINEAR)
                pixels = np.asarray(grey, dtype=np.float32)
        except (OSError, ValueError) as error:
            raise AlignmentDiagnosticError(
                f"cannot read diagnostic panorama {image_path}: {error}") \
                from error
        loaded.append((frame, when, sun_world_cw_deg,
                       sun_elevation_deg, pixels,
                       source_width, source_height))
    if not loaded:
        return _no_candidate(
            "solar_ephemeris",
            "no sampled frame has the sun above the elevation floor",
            {"n_dataset_frames": len(frames),
             "n_frames_meeting_speed": len(moving),
             "n_frames_sampled": len(sampled)})

    courses = [float(model.course_world_cw_deg_at(item[0].time_s))
               for item in loaded]
    _, course_concentration = sun_lib.circular_stats(courses)
    mask = sun_lib.rig_mask([item[4] for item in loaded])
    internal_rows = []
    output_rows = []
    for (frame, when, sun_world, sun_elevation, pixels,
         source_width, source_height) in loaded:
        found = sun_lib.brightest_blob_in_band(
            pixels, sun_elevation, config["elevation_tolerance_deg"],
            mask=mask)
        if found is None:
            continue
        (sun_camera, measured_elevation, n_pixels,
         work_x, work_y) = found
        source_x = work_x * source_width / pixels.shape[1]
        source_y = work_y * source_height / pixels.shape[0]
        course_world = float(model.course_world_cw_deg_at(frame.time_s))
        candidate = (course_world + sun_camera - sun_world) % 360.0
        internal_rows.append({
            "candidate_deg": candidate,
            "sun_az_camera_deg": sun_camera,
        })
        output_rows.append({
            "keyframe_idx": frame.frame_idx,
            "pano_id": frame.pano_id,
            "pano_stem": frame.pano_stem,
            "utc": when.isoformat(),
            "gps_course_world_cw_deg": round(course_world, 9),
            "sun_world_cw_deg": round(sun_world, 9),
            "sun_camera_cw_deg": round(sun_camera, 9),
            "sun_elevation_world_deg": round(sun_elevation, 9),
            "sun_elevation_camera_deg": round(measured_elevation, 9),
            RESULT_FIELD: round(candidate, 9),
            "bright_blob_center_px": {
                "x": round(source_x, 3),
                "y": round(source_y, 3),
                "image_width": source_width,
                "image_height": source_height,
            },
            "bright_blob_pixels": n_pixels,
        })
    if not internal_rows:
        return _no_candidate(
            "solar_ephemeris",
            "no sampled panorama produced a compact bright blob in the "
            "sun-elevation band",
            {"n_dataset_frames": len(frames),
             "n_frames_meeting_speed": len(moving),
             "n_frames_sampled": len(sampled),
             "n_frames_sun_above_floor": len(loaded)})

    candidate, concentration = sun_lib.circular_stats(
        [item["candidate_deg"] for item in internal_rows])
    _, fixed_concentration = sun_lib.circular_stats(
        [item["sun_az_camera_deg"] for item in internal_rows])
    if fixed_concentration > concentration and course_concentration < 0.98:
        classification = "fixed_camera_structure_more_likely"
        internally_supported = False
        detail = ("camera-fixed bright-blob model has greater concentration "
                  "than the solar alignment model while GPS course varies")
    elif concentration >= sun_lib.R_TRUSTWORTHY:
        classification = "internally_consistent"
        internally_supported = True
        detail = "per-frame effective camera-to-course candidates agree"
    elif concentration >= sun_lib.R_USELESS:
        classification = "weak"
        internally_supported = False
        detail = "per-frame candidates are consistent but diffuse"
    else:
        classification = "scattered"
        internally_supported = False
        detail = "bright blobs do not support one camera-to-course candidate"
    return {
        "method": "solar_ephemeris",
        "result_kind": RESULT_KIND,
        "frame": RESULT_FRAME,
        "status": "candidate_reported",
        RESULT_FIELD: round(candidate, 9),
        "evidence": {
            "n_dataset_frames": len(frames),
            "n_frames_meeting_speed": len(moving),
            "n_frames_sampled": len(sampled),
            "n_frames_sun_above_floor": len(loaded),
            "n_frames_with_bright_blob": len(output_rows),
            "candidate_concentration": round(concentration, 9),
            "fixed_camera_blob_concentration": round(fixed_concentration, 9),
            "gps_course_concentration": round(course_concentration, 9),
            "assessment": {
                "classification": classification,
                "internally_supported": internally_supported,
                "detail": detail,
            },
            "frames": output_rows,
        },
    }


def _attach_nominal_forward_comparison(
        method: dict,
        calibration: nominal_forward.NominalForward) -> None:
    if method.get("status") != "candidate_reported":
        return
    candidate = method[RESULT_FIELD]
    method["comparison_to_approved_nominal_forward"] = {
        "nominal_forward_bearing_camera_cw_deg": round(
            calibration.bearing_camera_cw_deg, 9),
        "candidate_minus_nominal_forward_cw_deg": round(float(
            geometry.circular_diff_deg(
                candidate, calibration.bearing_camera_cw_deg)), 9),
        "interpretation": (
            "diagnostic only; this difference may include GPS-course error, "
            "crab/current, timing error, or diagnostic error and cannot "
            "modify the approved nominal-forward record"),
    }


def _write_sun_contact_sheet(report: dict, dataset_base: Path,
                             output_path: Path) -> None:
    """Write review crops for the exact solar detections in the report."""
    solar = next(
        (method for method in report["methods"]
         if method["method"] == "solar_ephemeris"), None)
    rows = [] if solar is None else solar.get("evidence", {}).get("frames", [])
    cards = []
    for row in rows:
        source_path = (Path(dataset_base) / "panorama" /
                       f"{row['pano_stem']}.jpg")
        try:
            with Image.open(source_path) as opened:
                source = opened.convert("RGB")
        except OSError as error:
            raise AlignmentDiagnosticError(
                f"cannot create sun review crop from {source_path}: {error}") \
                from error
        location = row["bright_blob_center_px"]
        x, y = float(location["x"]), float(location["y"])
        width, height = source.size
        crop_width = min(320, width)
        crop_height = min(180, height)
        wrapped = Image.new("RGB", (width * 3, height))
        for offset in range(3):
            wrapped.paste(source, (offset * width, 0))
        centre_x = x + width
        crop = wrapped.crop((
            round(centre_x - crop_width / 2),
            round(y - crop_height / 2),
            round(centre_x + crop_width / 2),
            round(y + crop_height / 2),
        ))
        card = Image.new("RGB", (340, 225), color=(18, 24, 31))
        card.paste(crop, ((340 - crop.width) // 2, 30))
        draw = ImageDraw.Draw(card)
        draw.text((8, 7),
                  f"{row['pano_id']}  kf {row['keyframe_idx']}  "
                  f"x={x:.1f}, y={y:.1f}", fill=(235, 240, 245))
        draw.text((8, 210),
                  f"predicted elevation {row['sun_elevation_world_deg']:.1f}°",
                  fill=(190, 205, 220))
        cards.append(card)
    if not cards:
        sheet = Image.new("RGB", (640, 100), color=(18, 24, 31))
        ImageDraw.Draw(sheet).text(
            (12, 38), "No reviewable solar detections were produced.",
            fill=(235, 240, 245))
    else:
        sheet = Image.new("RGB", (340, 225 * len(cards)),
                          color=(18, 24, 31))
        for index, card in enumerate(cards):
            sheet.paste(card, (0, 225 * index))
    sheet.save(output_path, format="JPEG", quality=92)


def build_report(resolved: dict) -> dict:
    model = _course_model(resolved["frames"], resolved["course"])
    methods = [
        build_sweep_candidate(
            resolved["records"], resolved["frames"], model,
            resolved["diagnostics"]["sweep"]),
        build_sun_candidate(
            resolved["dataset_base"], resolved["metadata"],
            resolved["frames"], model, resolved["diagnostics"]["sun"]),
    ]
    for method in methods:
        _attach_nominal_forward_comparison(
            method, resolved["nominal_forward"])
    observation_ref = resolved["observations_ref"]
    return {
        "schema": SCHEMA,
        "dataset": observation_ref.dataset,
        "authority": dict(AUTHORITY),
        "quantity": {
            "kind": RESULT_KIND,
            "name": RESULT_FIELD,
            "frame": RESULT_FRAME,
            "definition": (
                "GPS course axis expressed as a clockwise camera-frame "
                "bearing; diagnostic proxy, not fixed nominal forward"),
            "range": "[0,360)",
        },
        "source": {
            "bearing_observations": {
                "kind": observation_ref.kind,
                "dataset": observation_ref.dataset,
                "version": observation_ref.version,
                "manifest_digest": observation_ref.manifest_digest,
                "content_digest": observation_ref.content_digest,
            },
            "dataset_source_sha256": resolved["dataset_source_sha256"],
            "approved_nominal_forward": {
                "path": str(resolved["nominal_forward_path"]),
                "sha256": resolved["nominal_forward_sha256"],
                "version": resolved["nominal_forward"].version,
                "mounting_id": resolved["nominal_forward"].mounting_id,
                "bearing_camera_cw_deg":
                    resolved["nominal_forward"].bearing_camera_cw_deg,
            },
            "gps_course": {
                "source": "bound_object_tracks_manifest",
                "object_tracks_manifest_digest":
                    resolved["tracks_ref"].manifest_digest,
                "parameters": dict(resolved["course"]),
            },
        },
        "review_media": {"sun_contact_sheet": SUN_REVIEW_NAME},
        "methods": methods,
    }


def publish(resolved: dict, output_dir: Path, *,
            arguments: tuple[str, ...] = ()) -> artifact.ArtifactRef:
    report = build_report(resolved)
    manifest_config = {
        "schema": SCHEMA,
        "authority": dict(AUTHORITY),
        "coverage": "complete_diagnostic_processing",
        "orchestration": resolved["orchestration"],
        "build_identity": resolved["document"]["build_identity"],
        "resolved": {
            "alignment_diagnostics": resolved["diagnostics"],
            "gps_course_from_object_tracks": resolved["course"],
            "approved_nominal_forward": {
                "path": str(resolved["nominal_forward_path"]),
                "sha256": resolved["nominal_forward_sha256"],
                "version": resolved["nominal_forward"].version,
                "mounting_id": resolved["nominal_forward"].mounting_id,
                "bearing_camera_cw_deg":
                    resolved["nominal_forward"].bearing_camera_cw_deg,
            },
        },
        "source_digests": {
            "build_config": resolved["build_config_sha256"],
            "dataset_tracking_inputs": resolved["dataset_source_sha256"],
            "nominal_forward": resolved["nominal_forward_sha256"],
        },
        **({"stage_reuse": resolved["stage_reuse"]}
           if resolved["stage_reuse"] is not None else {}),
    }
    with publication.published_artifact(
            output_dir, kind=paths_lib.ALIGNMENT_DIAGNOSTICS,
            dataset=resolved["observations_ref"].dataset,
            version=resolved["output_version"], generator=GENERATOR,
            git_commit=resolved["document"]["git_commit"],
            arguments=arguments,
            upstreams=(resolved["observations_ref"],),
            config=manifest_config,
            declared_outputs=(OUTPUT_NAME, SUN_REVIEW_NAME)) as builder:
        artifact.atomic_write_json(builder.output_path(OUTPUT_NAME), report)
        _write_sun_contact_sheet(
            report, resolved["dataset_base"],
            builder.output_path(SUN_REVIEW_NAME))
    return builder.artifact_ref


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--observations_dir", type=Path, required=True)
    parser.add_argument(
        "--nominal_forward_calibration", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    args = parser.parse_args()
    try:
        resolved = _load_inputs(args)
        reference = publish(
            resolved, args.output_dir, arguments=tuple(sys.argv))
        stage_reuse.require_output_commit(
            reference, target_build_dir=Path(args.build_config).parent,
            document=resolved["document"],
            authorization=resolved["reuse_authorization"])
    except (AlignmentDiagnosticError, artifact.ArtifactError,
            build_config.InvalidConfigValue,
            build_config.MissingConfigValue, dataset.ContractViolation,
            OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
