"""Exact, prefix-bounded authorization for truthful immutable-stage reuse.

The global build identity covers the whole experiment.  A completed prefix
artifact may nevertheless be byte-for-byte reusable by a successor whose
changes are first consumed downstream.  This module is the sole authority for
that exception.  It never rewrites an old manifest: it revalidates both build
recipes, current inputs, producer-specific source artifacts, and exact paths,
then records the bridge in canonical ``stage_reuse.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    llm_lifecycle,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    extract_landmarks,
    legacy_extraction_adoption,
)


PROOF_NAME = "stage_reuse.json"
PROOF_SCHEMA = "farfield_pipeline_stage_reuse/v2"
ATTESTATION_SCHEMA = "farfield_prefix_code_compatibility_review/v1"
BRIDGE_SCHEMA = "farfield_stage_reuse_bridge/v1"
THROUGH_STAGE = "track"

TRACKING_GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/tracking:run_tracking")
TRACKS_NAME = "tracks_full.json"

_DIGEST_CHARS = frozenset("0123456789abcdef")
_PREFIX_OWNER = {
    paths_lib.PINHOLE_IMAGES: "extract",
    paths_lib.FRAME_LANDMARKS: "extract",
    paths_lib.OBJECT_TRACKS: "track",
}
_PREFIX_CONFIG_KEYS = (
    "extraction", "execution", "cost", "ingest", "tracking", "gps_course",
)
_PREFIX_VERSION_KEYS = (
    "pinhole_images_version", "frame_landmarks_version",
    "object_tracks_version",
)
# The sections a build config must contain. `pipeline` cannot own this list
# (it imports this module, not the other way round), so it is declared here
# and `pipeline_test` asserts it equals the top level of `CONFIG_SCHEMA` --
# adding a section without updating this then fails as a plain test, not as a
# confusing StageReuseError raised deep inside reuse-proof creation.
CONFIG_TOP_LEVEL = frozenset({
    "experiment", "artifacts", "extraction", "execution", "cost", "ingest",
    "tracking", "audit", "matching", "bearing_observations", "gps_course",
    "alignment_diagnostics", "localization_inputs", "localization", "viewer",
})
_ARTIFACT_CONFIG_KEYS = frozenset(
    f"{kind}_version" for kind in paths_lib.ARTIFACT_KINDS)

_REQUIRED_INPUTS = frozenset({
    "farfield_root", "dataset_base", "source_config", "source_config_sha256",
    "sam2_checkpoint", "sam2_checkpoint_sha256", "motion_source",
    "motion_source_sha256", "nominal_forward_calibration",
    "nominal_forward_sha256", "catalog_manifest_digest",
    "catalog_content_digest", *paths_lib.DATASET_SOURCE_DIGEST_KEYS,
})
_OPTIONAL_INPUT_PAIRS = (
    frozenset({"video", "video_sha256"}),
    frozenset({"identity_review_output_dir", "identity_review_phase"}),
)
_KNOWN_INPUTS = _REQUIRED_INPUTS | frozenset().union(*_OPTIONAL_INPUT_PAIRS)
_PROVENANCE_ONLY_INPUTS = frozenset({
    "source_config", "source_config_sha256",
})
_PATH_INPUTS = frozenset({
    "farfield_root", "dataset_base", "source_config", "sam2_checkpoint",
    "motion_source", "nominal_forward_calibration", "video",
    "identity_review_output_dir",
})
_PREFIX_INPUTS = frozenset({
    "farfield_root", "dataset_base", "sam2_checkpoint",
    "sam2_checkpoint_sha256", *paths_lib.DATASET_SOURCE_DIGEST_KEYS,
    "video", "video_sha256",
})

_PROOF_KEYS = frozenset({
    "schema", "dataset", "through_stage", "source_build", "target_build",
    "config_changed_leaves", "input_changed_keys", "source_catalog",
    "target_catalog", "prefix_code_compatibility", "compatible_artifacts",
})
_BUILD_DESCRIPTOR_KEYS = frozenset({
    "path", "build_identity", "build_config_sha256", "git_commit",
})
_ATTESTATION_KEYS = frozenset({
    "schema", "decision", "source_git_commit", "target_git_commit",
    "reviewed_by", "reviewed_at", "note",
})


class StageReuseError(ValueError):
    """A stage-reuse proof or attempted bridge is invalid."""


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and set(value) <= _DIGEST_CHARS)


def _exact_keys(value: Any, expected: frozenset[str], what: str) -> dict:
    if not isinstance(value, dict):
        raise StageReuseError(f"{what} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise StageReuseError(
            f"{what} has invalid shape: missing={missing}, unknown={unknown}")
    return value


def _same_ref(left: artifact.ArtifactRef,
              right: artifact.ArtifactRef) -> bool:
    """ArtifactRef equality deliberately omits path; reuse must not."""
    return left.to_dict() == right.to_dict()


def _strict_json_bytes(path: Path, what: str) -> tuple[bytes, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                label = ("stage-reuse" if what == "stage-reuse proof"
                         else what)
                raise StageReuseError(f"duplicate {label} JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        label = "stage-reuse" if what == "stage-reuse proof" else what
        raise StageReuseError(f"invalid {label} JSON constant {value!r}")

    try:
        raw = Path(path).read_bytes()
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, StageReuseError) as exc:
        raise StageReuseError(f"invalid {what} {path}: {exc}") from exc
    return raw, value


def _changed_leaves(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(_changed_leaves(left[key], right[key], path))
        return result
    return set() if left == right else {prefix}


def _flatten(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {prefix: value}
    result = {}
    for key in sorted(value):
        if not isinstance(key, str) or not key:
            raise StageReuseError(f"{prefix} has an invalid key")
        result.update(_flatten(value[key], f"{prefix}.{key}"))
    return result


def _stage_contract(document: dict, stage: str) -> dict:
    selected = {}
    if stage == "extract":
        prefixes = ("extraction", "execution", "cost")
        versions = ("pinhole_images_version", "frame_landmarks_version")
    elif stage == "track":
        prefixes = ("ingest", "tracking", "gps_course")
        versions = ("object_tracks_version",)
    else:
        raise StageReuseError(f"unsupported prefix owner {stage!r}")
    config = document["config"]
    for prefix in prefixes:
        selected.update(_flatten(config[prefix], prefix))
    for name in versions:
        selected[f"artifacts.{name}"] = config["artifacts"][name]
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": stage,
        "config_digest": artifact.sha256_json(selected),
    }


def _validate_input_schema(document: dict, label: str) -> None:
    inputs = document["inputs"]
    missing = sorted(_REQUIRED_INPUTS - set(inputs))
    unknown = sorted(set(inputs) - _KNOWN_INPUTS)
    partial = [sorted(pair & set(inputs)) for pair in _OPTIONAL_INPUT_PAIRS
               if 0 < len(pair & set(inputs)) < len(pair)]
    if missing or unknown or partial:
        raise StageReuseError(
            f"{label} build input schema is invalid: missing={missing}, "
            f"unknown={unknown}, partial_optional_pairs={partial}")
    if ("identity_review_phase" in inputs
            and inputs["identity_review_phase"] != "post_match_gate"):
        raise StageReuseError(
            f"{label} build has an invalid identity-review input phase")
    for key in ("source_config_sha256", "sam2_checkpoint_sha256",
                "motion_source_sha256", "nominal_forward_sha256",
                "catalog_manifest_digest", "catalog_content_digest",
                *paths_lib.DATASET_SOURCE_DIGEST_KEYS):
        if not _is_digest(inputs[key]):
            raise StageReuseError(f"{label} build input {key!r} is not SHA-256")
    if "video_sha256" in inputs and not _is_digest(inputs["video_sha256"]):
        raise StageReuseError(f"{label} build video_sha256 is not SHA-256")
    invalid_paths = sorted(
        key for key in _PATH_INPUTS & set(inputs)
        if not isinstance(inputs[key], str)
        or not Path(inputs[key]).is_absolute())
    if invalid_paths:
        raise StageReuseError(
            f"{label} build path inputs must be absolute: {invalid_paths}")


def _resolve_build(build_dir: Path) -> tuple[paths_lib.FarfieldPaths, dict]:
    build_dir = Path(build_dir)
    if build_dir.is_symlink() or not build_dir.is_dir():
        raise StageReuseError(
            f"build directory is not a regular, non-symlink directory: "
            f"{build_dir}")
    try:
        document = build_config.load(build_dir)
    except (OSError, ValueError) as exc:
        raise StageReuseError(f"cannot load build {build_dir}: {exc}") from exc
    config_path = build_dir / build_config.BUILD_CONFIG_NAME
    try:
        if config_path.read_bytes() != artifact.canonical_json_bytes(
                document) + b"\n":
            raise StageReuseError(
                f"build config is not canonical JSON: {config_path}")
    except OSError as exc:
        raise StageReuseError(
            f"cannot reread build config {config_path}: {exc}") from exc
    config = document["config"]
    if set(config) != CONFIG_TOP_LEVEL:
        raise StageReuseError(
            f"build config top-level schema differs: "
            f"missing={sorted(CONFIG_TOP_LEVEL - set(config))}, "
            f"unknown={sorted(set(config) - CONFIG_TOP_LEVEL)}")
    if (not isinstance(config.get("artifacts"), dict)
            or set(config["artifacts"]) != _ARTIFACT_CONFIG_KEYS):
        raise StageReuseError("build artifact-version config has invalid shape")
    for key in _PREFIX_CONFIG_KEYS:
        if not isinstance(config.get(key), dict):
            raise StageReuseError(f"build config {key!r} must be an object")
    _validate_input_schema(document, str(build_dir))
    versions = {
        kind: config["artifacts"][f"{kind}_version"]
        for kind in paths_lib.ARTIFACT_KINDS
    }
    paths = paths_lib.FarfieldPaths(
        dataset=document["dataset"], root=Path(document["inputs"]["farfield_root"]),
        versions=versions,
        overrides={"dataset_base": Path(document["inputs"]["dataset_base"])})
    expected = paths.build_dir(build_dir.name).resolve()
    if build_dir.resolve() != expected:
        raise StageReuseError(
            f"build path disagrees with immutable root/dataset: "
            f"{build_dir.resolve()} != {expected}")
    return paths, document


def reviewed_prefix_code_compatibility(
        *, source_git_commit: str, target_git_commit: str,
        reviewed_by: str, reviewed_at: str, note: str) -> dict[str, str]:
    """Construct the required human-reviewed source/target code attestation."""
    for name, value in (("source_git_commit", source_git_commit),
                        ("target_git_commit", target_git_commit),
                        ("reviewed_by", reviewed_by),
                        ("reviewed_at", reviewed_at), ("note", note)):
        if not isinstance(value, str) or not value.strip():
            raise StageReuseError(f"prefix-code attestation {name} is empty")
    return {
        "schema": ATTESTATION_SCHEMA,
        "decision": "compatible_no_prefix_computation_change",
        "source_git_commit": source_git_commit,
        "target_git_commit": target_git_commit,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at,
        "note": note,
    }


def _validate_attestation(value: Any, source: dict, target: dict) -> dict:
    value = dict(_exact_keys(value, _ATTESTATION_KEYS,
                             "prefix-code compatibility attestation"))
    expected = reviewed_prefix_code_compatibility(
        source_git_commit=source["git_commit"],
        target_git_commit=target["git_commit"],
        reviewed_by=value["reviewed_by"], reviewed_at=value["reviewed_at"],
        note=value["note"])
    if value != expected:
        raise StageReuseError(
            "prefix-code compatibility attestation is not bound to the "
            "source/target commits or compatible decision")
    return value


def _require_running_target_commit(
        target: dict,
        authorization: StageReuseAuthorization | None = None) -> str:
    expected = target["git_commit"]
    current = require_checkout_commit(expected)
    if (authorization is not None
            and authorization.target_git_commit != expected):
        raise StageReuseError(
            "stage-reuse authorization targets a different code commit")
    return current


def require_checkout_commit(expected_commit: str) -> str:
    """Require the executing repository HEAD to equal one recorded commit."""
    current = provenance.git_commit()
    if current != expected_commit:
        raise StageReuseError(
            "executing checkout differs from immutable target build commit: "
            f"{current!r} != {expected_commit!r}")
    return current


def _build_descriptor(build_dir: Path, document: dict) -> dict[str, str]:
    return {
        "path": str(Path(build_dir).resolve()),
        "build_identity": document["build_identity"],
        "build_config_sha256": artifact.sha256_file(
            Path(build_dir) / build_config.BUILD_CONFIG_NAME),
        "git_commit": document["git_commit"],
    }


def _open_exact_ref(reference: artifact.ArtifactRef, *, expected_path: Path,
                    label: str) -> artifact.ArtifactManifest:
    expected_text = str(Path(expected_path).resolve())
    if reference.path != expected_text or Path(reference.path) != Path(expected_path):
        raise StageReuseError(
            f"{label} reference path is not the exact configured lane: "
            f"{reference.path!r} != {str(expected_path)!r}")
    path = Path(reference.path)
    if path.is_symlink() or not path.is_dir():
        raise StageReuseError(f"{label} path is not a regular directory")
    try:
        reopened = artifact.open_artifact(
            path, expected_kind=reference.kind,
            expected_dataset=reference.dataset,
            expected_version=reference.version)
        manifest = artifact.load_manifest(path)
    except (OSError, artifact.ArtifactError) as exc:
        raise StageReuseError(f"cannot validate {label}: {exc}") from exc
    if not _same_ref(reopened, reference):
        raise StageReuseError(f"{label} no longer resolves to its exact ref")
    try:
        current_manifest_digest = artifact.sha256_file(
            path / artifact.MANIFEST_NAME)
    except (OSError, artifact.ArtifactError) as exc:
        raise StageReuseError(
            f"cannot recheck {label} manifest identity: {exc}") from exc
    if current_manifest_digest != reference.manifest_digest:
        raise StageReuseError(
            f"{label} manifest changed while its ref was validated")
    return manifest


def _load_just_opened_manifest(
        reference: artifact.ArtifactRef, *, label: str,
        ) -> artifact.ArtifactManifest:
    """Read metadata after a full producer/open check without rehashing data."""
    path = Path(reference.path)
    try:
        manifest = artifact.load_manifest(path)
        manifest_digest = artifact.sha256_file(path / artifact.MANIFEST_NAME)
    except (OSError, artifact.ArtifactError) as exc:
        raise StageReuseError(
            f"cannot reread just-validated {label}: {exc}") from exc
    if manifest_digest != reference.manifest_digest:
        raise StageReuseError(
            f"{label} manifest changed after its content was validated")
    return manifest


def _configured_ref(paths: paths_lib.FarfieldPaths, document: dict,
                    kind: str) -> artifact.ArtifactRef:
    version = document["config"]["artifacts"][f"{kind}_version"]
    path = paths.artifact(kind, version)
    if path.is_symlink() or not path.is_dir():
        raise StageReuseError(
            f"configured {kind} path is not a regular directory")
    try:
        reference = artifact.open_artifact(
            path, expected_kind=kind, expected_dataset=document["dataset"],
            expected_version=version)
    except (OSError, artifact.ArtifactError) as exc:
        raise StageReuseError(
            f"configured {kind} artifact is invalid: {exc}") from exc
    if reference.path != str(path.resolve()):
        raise StageReuseError(f"configured {kind} ref has a non-exact path")
    return reference


def validate_catalog_binding(paths: paths_lib.FarfieldPaths,
                             document: dict) -> artifact.ArtifactRef:
    """Reopen the configured catalog and match both immutable build digests."""
    reference = _configured_ref(paths, document, paths_lib.CATALOGS)
    inputs = document["inputs"]
    if (reference.manifest_digest != inputs["catalog_manifest_digest"]
            or reference.content_digest != inputs["catalog_content_digest"]):
        raise StageReuseError(
            "configured catalog differs from immutable build inputs")
    return reference


def validate_target_catalog(target_build_dir: Path) -> artifact.ArtifactRef:
    """Consumer-side catalog revalidation, performed at point of use."""
    paths, document = _resolve_build(target_build_dir)
    return validate_catalog_binding(paths, document)


def _load_json_payload(path: Path, what: str) -> Any:
    raw, value = _strict_json_bytes(path, what)
    if raw != artifact.canonical_json_bytes(value) + b"\n":
        raise StageReuseError(f"{what} is not canonical JSON")
    return value


def _validate_track(reference: artifact.ArtifactRef,
                    manifest: artifact.ArtifactManifest,
                    source: dict, source_build_dir: Path,
                    pinhole_ref: artifact.ArtifactRef,
                    frame_ref: artifact.ArtifactRef) -> None:
    config = dict(manifest.config)
    expected_keys = {
        "orchestration", "schema", "coverage", "build_identity", "range",
        "resolved", "source_digests",
    }
    source_inputs = source["inputs"]
    expected_range = {
        "name": "full", "k_start": source["config"]["tracking"]["range"]["k_start"],
        "k_end": source["config"]["tracking"]["range"]["k_end"],
    }
    expected_digests = {
        "build_config": artifact.sha256_file(
            Path(source_build_dir) / build_config.BUILD_CONFIG_NAME),
        "dataset_tracking_inputs": artifact.sha256_json({
            key: source_inputs[key] for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS}),
        "sam2_checkpoint": source_inputs["sam2_checkpoint_sha256"],
        paths_lib.PINHOLE_IMAGES: pinhole_ref.content_digest,
        paths_lib.FRAME_LANDMARKS: frame_ref.content_digest,
    }
    if "video_sha256" in source_inputs:
        expected_digests["video"] = source_inputs["video_sha256"]
    if (manifest.generator != TRACKING_GENERATOR
            or manifest.git_commit != source["git_commit"]
            or set(config) != expected_keys
            or config["orchestration"] != _stage_contract(source, "track")
            or config["schema"] != "farfield_object_tracks/v1"
            or config["coverage"] != "complete"
            or config["build_identity"] != source["build_identity"]
            or config["range"] != expected_range
            or config["source_digests"] != expected_digests
            or len(manifest.upstreams) != 2
            or not _same_ref(manifest.upstreams[0], pinhole_ref)
            or not _same_ref(manifest.upstreams[1], frame_ref)
            or TRACKS_NAME not in manifest.declared_outputs
            or "index.html" not in manifest.declared_outputs):
        raise StageReuseError(
            "source object_tracks fails its exact producer contract")
    resolved = config["resolved"]
    if (not isinstance(resolved, dict)
            or set(resolved) != {"ingest", "tracking", "gps_course"}
            or resolved["ingest"] != source["config"]["ingest"]
            or resolved["gps_course"] != source["config"]["gps_course"]
            or resolved["tracking"] != {
                key: value for key, value in source["config"]["tracking"].items()
                if key != "range"}):
        raise StageReuseError("source object_tracks resolved config is stale")
    payload = _load_json_payload(Path(reference.path) / TRACKS_NAME,
                                 "tracks_full.json")
    if (not isinstance(payload, dict)
            or set(payload) != {
                "range", "config", "tracks", "rejected_births",
                "track_overlaps"}
            or payload["range"] != expected_range
            or payload["config"] != {
                key: value for key, value in resolved["tracking"].items()
                if key != "sam2_checkpoint"}
            or any(not isinstance(payload[name], list) for name in (
                "tracks", "rejected_births", "track_overlaps"))):
        raise StageReuseError("source tracks_full.json has a stale shape")
    track_ids = [item.get("track_id") for item in payload["tracks"]
                 if isinstance(item, dict)]
    if (len(track_ids) != len(payload["tracks"])
            or any(type(value) is not int for value in track_ids)
            or len(track_ids) != len(set(track_ids))
            or any(not isinstance(item.get("records"), list)
                   or not item["records"] for item in payload["tracks"])):
        raise StageReuseError("source tracks_full.json has invalid tracks")


def _validate_source_prefix(paths: paths_lib.FarfieldPaths, source: dict,
                            source_build_dir: Path) -> list[artifact.ArtifactRef]:
    configured_paths = {
        kind: paths.artifact(
            kind, source["config"]["artifacts"][f"{kind}_version"])
        for kind in (paths_lib.PINHOLE_IMAGES, paths_lib.FRAME_LANDMARKS)
    }
    try:
        context = extract_landmarks.load_artifact_validation_context(
            build_config_path=(Path(source_build_dir)
                               / build_config.BUILD_CONFIG_NAME),
            dataset=source["dataset"],
            dataset_base=Path(source["inputs"]["dataset_base"]))
        extraction_args = argparse.Namespace(
            dataset=source["dataset"],
            pinhole_output_dir=configured_paths[paths_lib.PINHOLE_IMAGES],
            output_dir=configured_paths[paths_lib.FRAME_LANDMARKS])
        producer_pinhole = (
            extract_landmarks.validate_existing_pinhole_artifact(
                extraction_args, context))
        producer_frame = extract_landmarks.validate_existing_frame_artifact(
            extraction_args, context, producer_pinhole)
        frame_manifest = _load_just_opened_manifest(
            producer_frame, label="source frame_landmarks")
        if (len(frame_manifest.upstreams) != 2
                or not _same_ref(
                    frame_manifest.upstreams[0],
                    producer_pinhole)):
            raise ValueError(
                "frame_landmarks does not bind the exact configured "
                "pinhole lane")
        if (frame_manifest.config.get("legacy_adoption_schema")
                == extract_landmarks.LEGACY_ADOPTION_CONFIG_SCHEMA):
            result_ref = frame_manifest.upstreams[1]
            result_manifest = artifact.load_manifest(result_ref.path)
            request_ref = result_manifest.upstreams[0]
            request_manifest = artifact.load_manifest(request_ref.path)
            proof = request_manifest.config["legacy_adoption"]
            report = proof["verification_report"]
            request_set = llm_lifecycle.load_request_set(
                Path(request_ref.path) / llm_lifecycle.REQUEST_SET_NAME)
            plan = legacy_extraction_adoption.reverify_published_report(
                report, dataset=source["dataset"],
                request_set=request_set,
                pinhole_dir=Path(producer_pinhole.path))
            if (plan.report_sha256
                    != proof["verification_report_sha256"]
                    or (Path(result_ref.path)
                        / llm_lifecycle.CANONICAL_RESULTS_NAME).read_bytes()
                    != plan.canonical_results_bytes
                    or (Path(producer_frame.path)
                        / extract_landmarks.PREDICTIONS_NAME).read_bytes()
                    != plan.predictions_bytes):
                raise ValueError(
                    "adopted extraction lineage does not reproduce its "
                    "published typed bytes")
        else:
            result_manifest = artifact.load_manifest(
                frame_manifest.upstreams[1].path)
            request_manifest = artifact.load_manifest(
                result_manifest.upstreams[0].path)
            if any(manifest.git_commit != source["git_commit"]
                   for manifest in (
                       request_manifest, result_manifest, frame_manifest)):
                raise ValueError(
                    "normal extraction lineage commit differs from its "
                    "source build")
    except (OSError, ValueError, artifact.ArtifactError) as exc:
        raise StageReuseError(
            f"source extraction prefix fails its real producer validator: "
            f"{exc}") from exc
    if any(
            reference.path != str(configured_paths[kind].resolve())
            or Path(reference.path) != configured_paths[kind]
            for kind, reference in (
                (paths_lib.PINHOLE_IMAGES, producer_pinhole),
                (paths_lib.FRAME_LANDMARKS, producer_frame))):
        raise StageReuseError(
            "source extraction validator returned a non-configured lane")
    track_ref = _configured_ref(
        paths, source, paths_lib.OBJECT_TRACKS)
    track_manifest = _load_just_opened_manifest(
        track_ref, label=f"source {paths_lib.OBJECT_TRACKS}")
    _validate_track(
        track_ref, track_manifest, source, source_build_dir,
        producer_pinhole, producer_frame)
    return [producer_pinhole, producer_frame, track_ref]


def _validate_current_common_prefix_inputs(source: dict, target: dict) -> None:
    """Hash each equal protected physical input exactly once."""
    left, right = source["inputs"], target["inputs"]
    unequal = sorted(key for key in _PREFIX_INPUTS
                     if left.get(key) != right.get(key))
    if unequal:
        raise StageReuseError(
            f"stage-reuse prefix inputs are not identical: {unequal}")
    try:
        current_dataset = paths_lib.dataset_source_digests(
            Path(left["dataset_base"]))
        mismatch = sorted(key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
                          if current_dataset[key] != left[key])
        if mismatch:
            raise StageReuseError(
                f"current dataset differs from prefix inputs: {mismatch}")
        checkpoint = Path(left["sam2_checkpoint"])
        if artifact.sha256_file(checkpoint) != left["sam2_checkpoint_sha256"]:
            raise StageReuseError("current SAM2 checkpoint digest differs")
        if "video" in left and artifact.sha256_file(
                Path(left["video"])) != left["video_sha256"]:
            raise StageReuseError("current source video digest differs")
    except (OSError, artifact.ArtifactError, paths_lib.MissingInput) as exc:
        raise StageReuseError(
            f"cannot validate current prefix input bytes: {exc}") from exc


def proof_document(source_build_dir: Path, target_build_dir: Path, *,
                   through_stage: str,
                   prefix_code_compatibility: dict) -> dict:
    if through_stage != THROUGH_STAGE:
        raise StageReuseError(
            f"stage reuse through {through_stage!r} is unsupported")
    source_build_dir, target_build_dir = (Path(source_build_dir),
                                          Path(target_build_dir))
    source_paths, source = _resolve_build(source_build_dir)
    target_paths, target = _resolve_build(target_build_dir)
    _require_running_target_commit(target)
    if source_build_dir.resolve() == target_build_dir.resolve():
        raise StageReuseError("stage reuse requires two different builds")
    if source["dataset"] != target["dataset"]:
        raise StageReuseError("stage reuse cannot cross datasets")
    if source["build_identity"] == target["build_identity"]:
        raise StageReuseError("stage reuse is unnecessary for one build identity")
    attestation = _validate_attestation(
        prefix_code_compatibility, source, target)
    config_changes = _changed_leaves(source["config"], target["config"])
    invalid_config = sorted(
        key for key in config_changes
        if any(key == prefix or key.startswith(prefix + ".")
               for prefix in _PREFIX_CONFIG_KEYS)
        or key in {f"artifacts.{name}" for name in _PREFIX_VERSION_KEYS})
    if invalid_config:
        raise StageReuseError(
            "builds differ in configuration consumed by the reused prefix: "
            f"{invalid_config}")
    input_changes = _changed_leaves(source["inputs"], target["inputs"])
    invalid_inputs = sorted(
        key for key in input_changes
        if key in _PREFIX_INPUTS and key not in _PROVENANCE_ONLY_INPUTS)
    if invalid_inputs:
        raise StageReuseError(
            "builds differ in inputs consumed by the reused prefix: "
            f"{invalid_inputs}")
    _validate_current_common_prefix_inputs(source, target)
    source_catalog = validate_catalog_binding(source_paths, source)
    target_catalog = validate_catalog_binding(target_paths, target)
    refs = _validate_source_prefix(source_paths, source, source_build_dir)
    for reference in refs:
        target_path = target_paths.artifact(
            reference.kind,
            target["config"]["artifacts"][
                f"{reference.kind}_version"])
        # Prefix roots and versions were already proven equal, and the real
        # source producer validator just hashed this exact directory.  A
        # second target-side content hash would read the same large artifact
        # again without adding an independent identity check.
        if (reference.path != str(target_path.resolve())
                or Path(reference.path) != target_path):
            raise StageReuseError(
                f"target does not resolve exact reused {reference.kind} ref")
    return {
        "schema": PROOF_SCHEMA,
        "dataset": target["dataset"],
        "through_stage": THROUGH_STAGE,
        "source_build": _build_descriptor(source_build_dir, source),
        "target_build": _build_descriptor(target_build_dir, target),
        "config_changed_leaves": sorted(config_changes),
        "input_changed_keys": sorted(input_changes),
        "source_catalog": source_catalog.to_dict(),
        "target_catalog": target_catalog.to_dict(),
        "prefix_code_compatibility": attestation,
        "compatible_artifacts": [reference.to_dict() for reference in refs],
    }


def create_proof(source_build_dir: Path, target_build_dir: Path, *,
                 through_stage: str,
                 prefix_code_compatibility: dict) -> Path:
    document = proof_document(
        source_build_dir, target_build_dir, through_stage=through_stage,
        prefix_code_compatibility=prefix_code_compatibility)
    path = Path(target_build_dir) / PROOF_NAME
    try:
        artifact.atomic_create_json(path, document)
    except (OSError, ValueError) as exc:
        raise StageReuseError(f"cannot create stage-reuse proof: {exc}") from exc
    return path


@dataclass(frozen=True)
class StageReuseAuthorization:
    """Recomputed exact refs accepted for one successor build."""

    target_build_identity: str
    source_build_identity: str
    source_git_commit: str
    target_git_commit: str
    source_build_path: str
    target_build_path: str
    through_stage: str
    refs: tuple[artifact.ArtifactRef, ...]
    proof_sha256: str

    def __post_init__(self) -> None:
        if (not _is_digest(self.target_build_identity)
                or not _is_digest(self.source_build_identity)
                or not _is_digest(self.proof_sha256)
                or self.through_stage != THROUGH_STAGE):
            raise StageReuseError("invalid stage-reuse authorization identity")
        serialized = [reference.to_dict() for reference in self.refs]
        if len(serialized) != len({artifact.sha256_json(item)
                                  for item in serialized}):
            raise StageReuseError("stage-reuse refs must be exactly unique")

    def accepts(self, reference: artifact.ArtifactRef, *, owner_stage: str,
                target_build_identity: str) -> bool:
        return (
            target_build_identity == self.target_build_identity
            and owner_stage in {"extract", "track"}
            and _PREFIX_OWNER.get(reference.kind) == owner_stage
            and any(_same_ref(reference, item) for item in self.refs)
        )

    def bridge_provenance(
            self, references: Iterable[artifact.ArtifactRef]) -> dict:
        refs = tuple(references)
        if not refs or any(not any(_same_ref(ref, item) for item in self.refs)
                           for ref in refs):
            raise StageReuseError(
                "cannot record bridge provenance for an unauthorized ref")
        return {
            "schema": BRIDGE_SCHEMA,
            "proof_sha256": self.proof_sha256,
            "source_build_identity": self.source_build_identity,
            "target_build_identity": self.target_build_identity,
            "source_git_commit": self.source_git_commit,
            "target_git_commit": self.target_git_commit,
            "source_build_path": self.source_build_path,
            "target_build_path": self.target_build_path,
            "through_stage": self.through_stage,
            "adopted_artifacts": [
                reference.to_dict() for reference in sorted(
                    refs, key=lambda item: (item.kind, item.path))],
        }


def load_proof(target_build_dir: Path) -> StageReuseAuthorization | None:
    target_build_dir = Path(target_build_dir)
    path = target_build_dir / PROOF_NAME
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise StageReuseError(
            f"stage-reuse proof is not a regular file: {path}")
    raw, stored = _strict_json_bytes(path, "stage-reuse proof")
    try:
        stored = _exact_keys(stored, _PROOF_KEYS, "stage-reuse proof")
        _exact_keys(stored["source_build"], _BUILD_DESCRIPTOR_KEYS,
                    "stage-reuse source build")
        _exact_keys(stored["target_build"], _BUILD_DESCRIPTOR_KEYS,
                    "stage-reuse target build")
        source_path = Path(stored["source_build"]["path"])
        attestation = stored["prefix_code_compatibility"]
        expected = proof_document(
            source_path, target_build_dir,
            through_stage=stored["through_stage"],
            prefix_code_compatibility=attestation)
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, StageReuseError):
            raise
        raise StageReuseError(f"invalid stage-reuse proof {path}: {exc}") from exc
    if stored != expected:
        raise StageReuseError(
            f"stage-reuse proof does not exactly reproduce: {path}")
    canonical = artifact.canonical_json_bytes(expected) + b"\n"
    if raw != canonical:
        raise StageReuseError(
            f"stage-reuse proof is not canonical JSON: {path}")
    try:
        refs = tuple(artifact.ArtifactRef.from_dict(value)
                     for value in expected["compatible_artifacts"])
    except artifact.ArtifactError as exc:
        raise StageReuseError(
            f"stage-reuse proof contains an invalid artifact ref: {exc}") \
            from exc
    return StageReuseAuthorization(
        target_build_identity=expected["target_build"]["build_identity"],
        source_build_identity=expected["source_build"]["build_identity"],
        source_git_commit=expected["source_build"]["git_commit"],
        target_git_commit=expected["target_build"]["git_commit"],
        source_build_path=expected["source_build"]["path"],
        target_build_path=expected["target_build"]["path"],
        through_stage=expected["through_stage"], refs=refs,
        proof_sha256=hashlib.sha256(raw).hexdigest())


def require_target_checkout(
        target_build_dir: Path, *, document: dict | None = None,
        authorization: StageReuseAuthorization | None = None) -> str:
    """Require the code executing a producer to be the target recipe code."""
    if document is None:
        try:
            document = build_config.load(target_build_dir)
        except (OSError, ValueError) as exc:
            raise StageReuseError(
                f"cannot load target build for checkout validation: {exc}") \
                from exc
    return _require_running_target_commit(document, authorization)


def require_configured_artifact(
        reference: artifact.ArtifactRef, *, target_build_dir: Path,
        kind: str, document: dict | None = None,
        ) -> artifact.ArtifactManifest:
    """Reopen one ref only at its exact configured target artifact lane."""
    if document is None:
        try:
            document = build_config.load(target_build_dir)
        except (OSError, ValueError) as exc:
            raise StageReuseError(
                f"cannot load target build for lane validation: {exc}") \
                from exc
    if reference.kind != kind or reference.dataset != document["dataset"]:
        raise StageReuseError(
            f"configured-lane check received the wrong {kind} ref")
    inputs = document.get("inputs")
    root = inputs.get("farfield_root") if isinstance(inputs, dict) else None
    try:
        version = document["config"]["artifacts"][f"{kind}_version"]
    except (KeyError, TypeError) as exc:
        raise StageReuseError(
            f"target build does not configure an exact {kind} lane") from exc
    if not isinstance(root, str) or not root:
        raise StageReuseError(
            "target build does not record inputs.farfield_root")
    expected_path = (
        Path(root) / "artifacts" / kind / document["dataset"] / version)
    if reference.version != version:
        raise StageReuseError(
            f"{kind} ref does not use the configured target version")
    manifest = _open_exact_ref(
        reference, expected_path=expected_path,
        label=f"configured target {kind}")
    return manifest


def require_output_commit(
        reference: artifact.ArtifactRef, *, target_build_dir: Path,
        document: dict | None = None,
        authorization: StageReuseAuthorization | None = None) -> None:
    """Require a newly published configured output to name target code."""
    if document is None:
        try:
            document = build_config.load(target_build_dir)
        except (OSError, ValueError) as exc:
            raise StageReuseError(
                f"cannot load target build for output validation: {exc}") \
                from exc
    expected = require_target_checkout(
        target_build_dir, document=document, authorization=authorization)
    manifest = require_configured_artifact(
        reference, target_build_dir=target_build_dir, kind=reference.kind,
        document=document)
    if manifest.git_commit != expected:
        raise StageReuseError(
            f"new {reference.kind} output commit differs from target build")


def require_manifest_commit(
        reference: artifact.ArtifactRef, expected_commit: str) -> None:
    """Reopen any typed output and require its recorded producer commit."""
    manifest = _open_exact_ref(
        reference, expected_path=Path(reference.path),
        label=f"new {reference.kind} output")
    if manifest.git_commit != expected_commit:
        raise StageReuseError(
            f"new {reference.kind} output commit differs from target build")


def require_compatible_artifact(
        reference: artifact.ArtifactRef,
        manifest: artifact.ArtifactManifest, *, target_build_dir: Path,
        owner_stage: str,
        authorization: StageReuseAuthorization | None = None,
        ) -> dict | None:
    """Validate a direct old-prefix consumer and return bridge provenance."""
    try:
        target = build_config.load(target_build_dir)
    except (OSError, ValueError) as exc:
        raise StageReuseError(
            f"cannot load target build for stage reuse: {exc}") from exc
    target_identity = target["build_identity"]
    reopened_manifest = _open_exact_ref(
        reference, expected_path=Path(reference.path),
        label=f"supplied {reference.kind}")
    if reopened_manifest.to_dict() != manifest.to_dict():
        raise StageReuseError(
            f"supplied {reference.kind} manifest does not match its exact ref")
    manifest = reopened_manifest
    if manifest.config.get("build_identity") == target_identity:
        return None
    # The complete exact schema is required only when exercising the reuse
    # exception. Ordinary same-build consumers retain their own narrower
    # producer-specific build schema tests.
    _, target = _resolve_build(target_build_dir)
    authorization = authorization or load_proof(target_build_dir)
    _require_running_target_commit(target, authorization)
    if authorization is None or not authorization.accepts(
            reference, owner_stage=owner_stage,
            target_build_identity=target_identity):
        raise StageReuseError(
            f"{reference.kind} belongs to a different immutable build and "
            "has no exact stage-reuse authorization")
    _open_exact_ref(
        reference, expected_path=Path(reference.path),
        label=f"authorized {reference.kind}")
    return authorization.bridge_provenance((reference,))


def combine_bridge_provenance(*values: dict | None) -> dict | None:
    """Merge direct-consumer bridge records from one recomputed proof."""
    present = [value for value in values if value is not None]
    if not present:
        return None
    identity_keys = {
        "schema", "proof_sha256", "source_build_identity",
        "target_build_identity", "source_git_commit", "target_git_commit",
        "source_build_path", "target_build_path", "through_stage",
    }
    base = {key: present[0][key] for key in identity_keys}
    if any({key: item[key] for key in identity_keys} != base
           for item in present[1:]):
        raise StageReuseError("cannot combine different stage-reuse proofs")
    by_digest = {}
    for item in present:
        for reference in item["adopted_artifacts"]:
            by_digest[artifact.sha256_json(reference)] = reference
    return {
        **base,
        "adopted_artifacts": sorted(
            by_digest.values(), key=lambda item: (item["kind"], item["path"])),
    }


def require_recorded_bridge(
        recorded: Any, expected: dict | None, *,
        required_artifacts: Iterable[artifact.ArtifactRef] = (),
        additional_artifacts: Iterable[artifact.ArtifactRef] = ()) -> None:
    """Validate bridge provenance copied through a downstream manifest.

    A directly authorized consumer recomputes ``expected`` from the proof.
    This check prevents an intermediate producer from dropping, replacing, or
    widening that authorization while still allowing a combined audit bridge
    to name the FRAME authorization in addition to the required TRACK ref.
    """
    if expected is None:
        if recorded is not None:
            raise StageReuseError(
                "manifest records stage-reuse provenance without an active "
                "authorization")
        return
    keys = frozenset({
        "schema", "proof_sha256", "source_build_identity",
        "target_build_identity", "source_git_commit", "target_git_commit",
        "source_build_path", "target_build_path", "through_stage",
        "adopted_artifacts",
    })
    value = _exact_keys(recorded, keys, "recorded stage-reuse bridge")
    expected_value = _exact_keys(
        expected, keys, "recomputed stage-reuse bridge")
    identity_keys = keys - {"adopted_artifacts"}
    if any(value[key] != expected_value[key] for key in identity_keys):
        raise StageReuseError(
            "recorded stage-reuse bridge names a different proof")
    try:
        recorded_refs = tuple(
            artifact.ArtifactRef.from_dict(item)
            for item in value["adopted_artifacts"])
        expected_refs = tuple(
            artifact.ArtifactRef.from_dict(item)
            for item in expected_value["adopted_artifacts"])
    except (TypeError, artifact.ArtifactError) as exc:
        raise StageReuseError(
            f"recorded stage-reuse bridge has invalid artifact refs: {exc}") \
            from exc
    canonical = tuple(sorted(
        recorded_refs, key=lambda item: (item.kind, item.path)))
    if (recorded_refs != canonical
            or len({artifact.sha256_json(item.to_dict())
                    for item in recorded_refs}) != len(recorded_refs)):
        raise StageReuseError(
            "recorded stage-reuse bridge artifacts must be unique and sorted")
    recorded_dicts = {artifact.sha256_json(item.to_dict())
                      for item in recorded_refs}
    additional = tuple(additional_artifacts)
    required = tuple(expected_refs) + tuple(required_artifacts) + additional
    if any(artifact.sha256_json(item.to_dict()) not in recorded_dicts
           for item in required):
        raise StageReuseError(
            "recorded stage-reuse bridge omits an authorized artifact")
    allowed_by_identity = {
        artifact.sha256_json(item.to_dict()): item
        for item in required
    }
    allowed = tuple(sorted(
        allowed_by_identity.values(),
        key=lambda item: (item.kind, item.path)))
    if recorded_refs != allowed:
        raise StageReuseError(
            "recorded stage-reuse bridge changes the authorized artifacts")
