"""Publish lossless, audited camera-bearing observations.

This stage is the only owner of the accepted-tracklet-to-bearing conversion.
It consumes one source-bound semantic-audit artifact, retains every usable
keyframe, and publishes correlation groups so downstream reducers cannot
quietly treat observations from the same audited segment as independent.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    dataset,
    paths as paths_lib,
    publication,
    provenance,
    stage_reuse,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.tracking import tracklets


GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking:"
             "build_bearing_observations")
OUTPUT_NAME = "observations.jsonl"


class BearingObservationError(ValueError):
    """The accepted tracks cannot produce one lossless bearing artifact."""


def _canonical_jsonl(records: list[dict]) -> bytes:
    return b"".join(artifact.canonical_json_bytes(record) + b"\n"
                    for record in records)


def publish_observations(
        output_dir: Path, *, dataset_name: str, version: str,
        tracks_ref: artifact.ArtifactRef,
        audits_ref: artifact.ArtifactRef,
        accepted_tracklets: list[tracklets.AcceptedTracklet],
        pano_width: int, bearing_sigma_deg: float,
        orchestration: dict, build_identity: str, source_digests: dict,
        stage_reuse_provenance: dict | None = None,
        git_commit: str | None = None,
        arguments: tuple[str, ...] = ()) -> artifact.ArtifactRef:
    """Build, validate, and atomically publish one observation artifact."""
    observations = tracklets.build_camera_bearing_observations(
        accepted_tracklets, pano_width, bearing_sigma_deg)
    accepted_ids = {item.tracklet_id for item in accepted_tracklets}
    observed_ids = {item.tracklet_id for item in observations}
    missing = sorted(accepted_ids - observed_ids)
    if missing:
        raise BearingObservationError(
            "accepted tracklets have no bearing-capable record in their "
            f"valid segments: {missing}")
    observations.sort(key=lambda item: (item.tracklet_id, item.keyframe_idx))
    keys = [(item.tracklet_id, item.keyframe_idx) for item in observations]
    if len(keys) != len(set(keys)):
        raise BearingObservationError(
            "an accepted tracklet has more than one bearing at a keyframe")
    records = [dataclasses.asdict(item) for item in observations]
    config = {
        "orchestration": orchestration,
        "build_identity": build_identity,
        "schema": "farfield_bearing_observations/v1",
        "pano_width": pano_width,
        "bearing_sigma_deg": bearing_sigma_deg,
        "n_accepted_tracklets": len(accepted_tracklets),
        "n_observations": len(records),
        "coverage": "complete",
        "source_digests": source_digests,
        **({"stage_reuse": stage_reuse_provenance}
           if stage_reuse_provenance is not None else {}),
    }
    with publication.published_artifact(
            output_dir, kind=paths_lib.BEARING_OBSERVATIONS,
            dataset=dataset_name, version=version, generator=GENERATOR,
            git_commit=(provenance.git_commit()
                        if git_commit is None else git_commit),
            arguments=arguments,
            upstreams=(tracks_ref, audits_ref), config=config,
            declared_outputs=(OUTPUT_NAME,)) as builder:
        artifact.atomic_write_file(
            builder.output_path(OUTPUT_NAME), _canonical_jsonl(records))
    return builder.artifact_ref


def _load_build_document(path: Path) -> dict:
    path = Path(path)
    if (path.name != build_config.BUILD_CONFIG_NAME or not path.is_file()
            or path.is_symlink()):
        raise BearingObservationError(
            f"--build_config must name a regular, non-symlink "
            f"{build_config.BUILD_CONFIG_NAME}")
    return build_config.load(path.parent)


def orchestration_contract(document: dict) -> dict:
    """Recompute the pipeline's exact bearings-stage config selection."""
    keys = (
        "bearing_observations.bearing_sigma_deg",
        "tracking.reference_pano_width",
        "artifacts.bearing_observations_version",
    )
    selected = {key: build_config.value(document, key) for key in keys}
    return {
        "schema": "farfield_pipeline_stage/v1",
        "stage": "bearings",
        "config_digest": artifact.sha256_json(selected),
    }


def _same_path(actual: Path, recorded: str, what: str) -> Path:
    resolved = Path(actual).resolve()
    if resolved != Path(recorded).resolve():
        raise BearingObservationError(
            f"{what} disagrees with immutable build config: {resolved} != "
            f"{Path(recorded).resolve()}")
    return resolved


def load_inputs(args):
    """Resolve and verify every source before converting any observations."""
    config_path = Path(args.build_config)
    document = _load_build_document(config_path)
    if document["dataset"] != args.dataset:
        raise BearingObservationError(
            "--dataset disagrees with the immutable build config")

    dataset_base = _same_path(
        args.dataset_base, document["inputs"].get("dataset_base", ""),
        "--dataset_base")
    if dataset_base.is_symlink() or not dataset_base.is_dir():
        raise BearingObservationError(
            f"--dataset_base must be a regular directory: {dataset_base}")
    metadata = dataset.load_metadata(dataset_base)
    if metadata["dataset_name"] != args.dataset:
        raise BearingObservationError(
            "dataset metadata disagrees with --dataset")
    dataset.require_camera_frame_panoramas(metadata, dataset_base)
    try:
        dataset_digests = paths_lib.dataset_source_digests(dataset_base)
    except paths_lib.MissingInput as error:
        raise BearingObservationError(str(error)) from error
    mismatched_sources = [
        key for key in paths_lib.DATASET_SOURCE_DIGEST_KEYS
        if document["inputs"].get(key) != dataset_digests[key]
    ]
    if mismatched_sources:
        raise BearingObservationError(
            "dataset source bytes differ from the immutable build recipe: "
            f"{mismatched_sources}")

    output_version = build_config.value(
        document, "artifacts.bearing_observations_version")
    if Path(args.output_dir).name != output_version:
        raise BearingObservationError(
            f"--output_dir must end in configured version {output_version!r}")
    orchestration = orchestration_contract(document)
    if args.orchestration_config_digest != orchestration["config_digest"]:
        raise BearingObservationError(
            "--orchestration_config_digest does not match the immutable "
            "bearings-stage config selection")

    expected_versions = {
        paths_lib.OBJECT_TRACKS: build_config.value(
            document, "artifacts.object_tracks_version"),
        paths_lib.SEMANTIC_AUDITS: build_config.value(
            document, "artifacts.semantic_audits_version"),
    }
    tracks_ref = artifact.open_artifact(
        args.tracks_dir, expected_kind=paths_lib.OBJECT_TRACKS,
        expected_dataset=args.dataset,
        expected_version=expected_versions[paths_lib.OBJECT_TRACKS])
    audits_ref = artifact.open_artifact(
        args.audit_dir, expected_kind=paths_lib.SEMANTIC_AUDITS,
        expected_dataset=args.dataset,
        expected_version=expected_versions[paths_lib.SEMANTIC_AUDITS])
    authorization = stage_reuse.load_proof(config_path.parent)
    stage_reuse.require_target_checkout(
        config_path.parent, document=document, authorization=authorization)
    tracks_manifest = stage_reuse.require_configured_artifact(
        tracks_ref, target_build_dir=config_path.parent,
        kind=paths_lib.OBJECT_TRACKS, document=document)
    track_bridge = stage_reuse.require_compatible_artifact(
        tracks_ref, tracks_manifest, target_build_dir=config_path.parent,
        owner_stage="track", authorization=authorization)
    audit_manifest = stage_reuse.require_configured_artifact(
        audits_ref, target_build_dir=config_path.parent,
        kind=paths_lib.SEMANTIC_AUDITS, document=document)
    stage_reuse.require_recorded_bridge(
        audit_manifest.config.get("stage_reuse"), track_bridge,
        required_artifacts=(tracks_ref,),
        additional_artifacts=tuple(
            reference for reference in (authorization.refs
                                         if authorization is not None else ())
            if reference.kind == paths_lib.FRAME_LANDMARKS))
    if audit_manifest.config.get("build_identity") != document["build_identity"]:
        raise BearingObservationError(
            f"{paths_lib.SEMANTIC_AUDITS} belongs to a different immutable build")
    if sum(ref.to_dict() == tracks_ref.to_dict()
           for ref in audit_manifest.upstreams) != 1:
        raise BearingObservationError(
            "semantic_audits must bind the exact object_tracks artifact once")
    audits = audit_io.load_audits(args.tracks_dir, args.audit_dir)
    if (audits.tracks_ref.to_dict() != tracks_ref.to_dict()
            or audits.semantic_audits_ref.to_dict() != audits_ref.to_dict()):
        raise BearingObservationError(
            "audit loader did not retain the exact authorized upstream refs")

    dataset_source_digest = artifact.sha256_json(dataset_digests)
    recorded_sources = tracks_manifest.config.get("source_digests")
    if (not isinstance(recorded_sources, dict)
            or recorded_sources.get("dataset_tracking_inputs")
            != dataset_source_digest):
        raise BearingObservationError(
            "object_tracks does not bind the current frozen dataset sources")
    return {
        "document": document,
        "audits": audits,
        "output_version": output_version,
        "orchestration": orchestration,
        "source_digests": {
            "build_config": artifact.sha256_file(config_path),
            "dataset_tracking_inputs": dataset_source_digest,
            paths_lib.OBJECT_TRACKS: audits.tracks_ref.content_digest,
            paths_lib.SEMANTIC_AUDITS:
                audits.semantic_audits_ref.content_digest,
        },
        "stage_reuse": track_bridge,
        "reuse_authorization": authorization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--tracks_dir", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--build_config", type=Path, required=True)
    parser.add_argument("--orchestration_config_digest", required=True)
    args = parser.parse_args()

    try:
        resolved = load_inputs(args)
        document = resolved["document"]
        audits = resolved["audits"]
        accepted = tracklets.build_accepted_tracklets(
            audits.source_tracks, audits)
        reference = publish_observations(
            args.output_dir, dataset_name=args.dataset,
            version=resolved["output_version"],
            tracks_ref=audits.tracks_ref,
            audits_ref=audits.semantic_audits_ref,
            accepted_tracklets=accepted,
            pano_width=build_config.value(
                document, "tracking.reference_pano_width"),
            bearing_sigma_deg=build_config.value(
                document, "bearing_observations.bearing_sigma_deg"),
            orchestration=resolved["orchestration"],
            build_identity=document["build_identity"],
            source_digests=resolved["source_digests"],
            stage_reuse_provenance=resolved["stage_reuse"],
            git_commit=document["git_commit"],
            arguments=tuple(sys.argv))
        stage_reuse.require_output_commit(
            reference, target_build_dir=Path(args.build_config).parent,
            document=document,
            authorization=resolved["reuse_authorization"])
    except (artifact.ArtifactError, audit_io.AuditArtifactError,
            BearingObservationError, dataset.ContractViolation,
            tracklets.TrackletContractError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
