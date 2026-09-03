"""Publish detections-as-tracks and passthrough audits for the no-tracking ablation.

Every ingested VLM detection becomes one single-record ``object_tracks`` track
and one deterministic ``semantic_audits`` record carrying that detection's own
tags verbatim (weight 1.0, its name as the sole name candidate). Nothing is
linked across keyframes, pooled, reviewed, or gated on support. The bearing is
the detection's pano-box midpoint, exactly the box the SAM2 tracker would have
been seeded with.

Downstream stages run unchanged from ``pipeline run --from bearings``:
bearings read the box like a mask box, matching recognises the audit source
(``audit_io.DETECTION_PASSTHROUGH_SOURCE``) and switches to the
single-detection prompt with one Set 1 entry per distinct tag bundle, and the
filter sees one single-epoch tracklet per detection.

Both artifacts are published into the lanes the build config pins and record
the identities the orchestrator computes for the ``track`` and ``audit``
stages, so the pipeline accepts them as those stages' outputs. The versions
must therefore be dedicated to this ablation: the manifests name this
generator, but the recorded stage config is the SAM2 recipe the real tracker
would have run under.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    dataset as dataset_lib,
    geometry as geo,
    paths as paths_lib,
    pipeline,
    provenance,
    publication,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.tracking import (
    semantic_audit as sa,
    track_builder as tb,
    tracklets,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking:"
             "detections_as_tracks")
TRACKS_FILE = "tracks_full.json"
RANGE_NAME = "full"
CLOSE_REASON = "single_detection"
AUDIT_OUTPUTS = ("audit_meta.json", "results.jsonl", "settings.json")

_VEGETATION = frozenset({"wood", "tree", "tree_row", "scrub", "heath"})


class DetectionTrackError(ValueError):
    """The build cannot be represented as detections-only tracks."""


def landmark_kind(primary_key: str, primary_value: str) -> str:
    if primary_key.startswith("seamark:"):
        return "navigation_aid"
    if primary_key == "natural":
        return "vegetation" if primary_value in _VEGETATION else "terrain"
    return "fixed_structure"


def detection_tags(obs) -> tuple[list[str], list[str]]:
    """(identity tags in the detection's order, names) of one observation."""
    tags = [f"{obs.primary_tag_key}={obs.primary_tag_value}"]
    names = []
    for key, value in obs.additional_tags:
        if key == "name":
            if value not in names:
                names.append(value)
        elif key not in sa.NON_IDENTITY_TAG_KEYS:
            tag = f"{key}={value}"
            if tag not in tags:
                tags.append(tag)
    return tags, names


def passthrough_audit(obs) -> dict:
    """A canonical TrackAudit that restates one detection and decides nothing."""
    tags, names = detection_tags(obs)
    audit = {
        "landmark_kind": landmark_kind(
            obs.primary_tag_key, obs.primary_tag_value),
        "single_object": True,
        "valid_segments": [{"start_t": 0, "end_t": 0}],
        "verdict": "keep",
        "drop_reason": "none",
        "primary_object": {
            "tags": [{"tag": tag, "weight": 1.0} for tag in tags],
            "name_candidates": [
                {"name": name, "weight": 1.0,
                 "basis": "reported_by_detections"}
                for name in names],
            "name_aliases": [],
            "description": obs.description,
            "distinctive_features": [],
            "extent": "small_extended",
        },
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": obs.confidence,
        "unresolved": "",
    }
    return sa.TrackAudit.model_validate(audit).model_dump()


def detection_track(track_id: int, obs, *, pano_w: int, pano_h: int,
                    fov_deg: float) -> dict:
    """One closed single-record track whose mask box is the detection box."""
    box = [float(value) for value in geo.pano_bbox_for_observation(
        obs.boxes, pano_w, pano_h, fov_deg)]
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width <= 0.0 or height <= 0.0:
        raise DetectionTrackError(
            f"observation {obs.obs_id} has a degenerate pano box {box}")
    area = width * height
    keyframe = obs.frame_idx
    tags, names = detection_tags(obs)
    label = tags[0] + (f" '{names[0]}'" if names else "")
    return {
        "track_id": track_id,
        "birth_keyframe": keyframe,
        "end_keyframe": keyframe,
        "last_keyframe": keyframe,
        "birth_obs_id": obs.obs_id,
        "modal_label": label,
        "n_supported_keyframes": 0,
        "status": "closed",
        "close_reason": CLOSE_REASON,
        "records": [{
            "keyframe": keyframe,
            "action": "birth",
            "health": {"ok": True, "area": area, "coverage": 1.0,
                       "spill_frac": 0.0, "dominant_cc_frac": 1.0,
                       "n_components": 1, "reason": ""},
            "mask_area": area,
            "mask_bbox_window": box,
            "supports": [],
            "window_origin": [0, 0],
            "window_px": pano_w,
        }],
    }


def _panorama_size(dataset_base: Path, frames) -> tuple[int, int]:
    frame = min(frames, key=lambda item: item.frame_idx)
    with Image.open(
            Path(dataset_base) / "panorama" / f"{frame.pano_stem}.jpg") as im:
        return im.size


def _result_line(key: str, audit: dict) -> bytes:
    return artifact.canonical_json_bytes({
        "key": key,
        "response": {"candidates": [{"content": {"parts": [
            {"text": json.dumps(audit, sort_keys=True)}]}}]},
    }) + b"\n"


def build(build_dir: Path) -> tuple[artifact.ArtifactRef, artifact.ArtifactRef]:
    paths, document = pipeline.resolve_build(build_dir)
    config = document["config"]
    build_inputs = pipeline.build_inputs_of(document)
    pinhole_ref, frames_ref = pipeline.expected_upstream_refs(
        paths, config, "track", build_inputs=build_inputs)

    ingest = dataset_lib.run_ingest(
        paths.dataset_base, Path(frames_ref.path),
        dataset_lib.IngestParams(**config["ingest"]))
    if ingest.frame_landmarks_ref != frames_ref:
        raise DetectionTrackError(
            "ingest resolved a different frame_landmarks artifact than the "
            "configured lane")
    k_start = config["tracking"]["range"]["k_start"]
    k_end = config["tracking"]["range"]["k_end"]
    pano_w, pano_h = _panorama_size(paths.dataset_base, ingest.frames)
    if pano_w != config["tracking"]["reference_pano_width"]:
        raise DetectionTrackError(
            f"panorama width {pano_w} differs from "
            f"tracking.reference_pano_width")
    fov_deg = config["ingest"]["fov_deg"]
    observations = sorted(
        (obs for obs in ingest.observations
         if k_start <= obs.frame_idx <= k_end),
        key=lambda obs: (obs.frame_idx, obs.landmark_idx, obs.local_obs_id))
    if not observations:
        raise DetectionTrackError("no detections inside the tracking range")
    tracks = [detection_track(index, obs, pano_w=pano_w, pano_h=pano_h,
                              fov_deg=fov_deg)
              for index, obs in enumerate(observations)]
    builder_cfg = dataclasses.asdict(tb.TrackBuilderConfig(**{
        field.name: config["tracking"][field.name]
        for field in dataclasses.fields(tb.TrackBuilderConfig)}))
    tracks_document = {
        "range": {"name": RANGE_NAME, "k_start": k_start, "k_end": k_end},
        "config": builder_cfg,
        "tracks": tracks,
        "rejected_births": [],
        "track_overlaps": [],
    }
    dataset_digests = paths_lib.dataset_source_digests(paths.dataset_base)
    config_sha = artifact.sha256_file(
        Path(build_dir) / build_config.BUILD_CONFIG_NAME)
    git_commit = provenance.git_commit()

    tracks_dir = paths.artifact(paths_lib.OBJECT_TRACKS)
    tracks_config = {
        "orchestration": pipeline.stage_contract("track", config),
        "schema": "farfield_object_tracks/v1",
        "coverage": "complete",
        "build_identity": document["build_identity"],
        "range": tracks_document["range"],
        "producer": audit_io.DETECTION_PASSTHROUGH_SOURCE,
        "producer_note": (
            "no-tracking ablation: one single-record track per VLM "
            "detection; SAM2 was not run, the recorded tracking config is "
            "the recipe this lane stands in for"),
        "resolved": {
            "ingest": dict(config["ingest"]),
            "tracking": {**builder_cfg,
                         "sam2_checkpoint": config["tracking"][
                             "sam2_checkpoint"]},
            "gps_course": dict(config["gps_course"]),
        },
        "source_digests": {
            "build_config": config_sha,
            "dataset_tracking_inputs": artifact.sha256_json(dataset_digests),
            paths_lib.PINHOLE_IMAGES: pinhole_ref.content_digest,
            paths_lib.FRAME_LANDMARKS: frames_ref.content_digest,
        },
        "n_observations_ingested": len(ingest.observations),
        "n_tracks": len(tracks),
    }
    with publication.published_artifact(
            tracks_dir, kind=paths_lib.OBJECT_TRACKS, dataset=paths.dataset,
            version=paths.version(paths_lib.OBJECT_TRACKS),
            generator=GENERATOR, git_commit=git_commit,
            arguments=tuple(sys.argv), upstreams=(pinhole_ref, frames_ref),
            config=tracks_config,
            artifact_identity=pipeline.expected_artifact_identity(
                paths, config, paths_lib.OBJECT_TRACKS,
                build_inputs=build_inputs),
            recipe=pipeline.stage_recipe(
                paths, config, "track", build_inputs=build_inputs),
            declared_outputs=(TRACKS_FILE,)) as builder:
        artifact.atomic_write_json(
            builder.output_path(TRACKS_FILE), tracks_document)
    tracks_ref = builder.artifact_ref
    print(f"published {len(tracks)} detection tracks: {tracks_dir}")

    requests = {}
    result_lines = []
    observation_by_key = {}
    for track, obs in zip(tracks, observations):
        key = f"T{track['track_id']}"
        requests[key] = {
            "track_id": track["track_id"],
            "source_track_sha256": artifact.sha256_json(track),
            "range": RANGE_NAME,
            "birth_keyframe": track["birth_keyframe"],
            "n_supports": 0,
            "support_obs_by_t": {},
        }
        result_lines.append(_result_line(key, passthrough_audit(obs)))
        observation_by_key[key] = obs.obs_id
    audit_meta = {
        "schema": audit_io.META_SCHEMA,
        "source_tracks": {
            "artifact_id": audit_io.source_artifact_id(tracks_ref),
            "file": TRACKS_FILE,
            "sha256": artifact.sha256_file(tracks_dir / TRACKS_FILE),
        },
        "requests": requests,
    }
    settings = {
        "generator": GENERATOR,
        "git_commit": git_commit,
        "argv": list(sys.argv),
        "audit_source": audit_io.DETECTION_PASSTHROUGH_SOURCE,
        "rule": ("every detection track is kept over its single keyframe; "
                 "tags carry weight 1.0, the detection's name tag is the "
                 "sole name candidate; no provider was queried"),
        "n_requests": len(requests),
        "observation_by_key": observation_by_key,
    }
    audits_dir = paths.artifact(paths_lib.SEMANTIC_AUDITS)
    audits_config = {
        "orchestration": pipeline.stage_contract("audit", config),
        "build_identity": document["build_identity"],
        "phase": "canonical_results",
        "coverage": "complete",
        "n_expected": len(requests),
        "n_successful": len(requests),
        "audit_source": audit_io.DETECTION_PASSTHROUGH_SOURCE,
    }
    with publication.published_artifact(
            audits_dir, kind=paths_lib.SEMANTIC_AUDITS, dataset=paths.dataset,
            version=paths.version(paths_lib.SEMANTIC_AUDITS),
            generator=GENERATOR, git_commit=git_commit,
            arguments=tuple(sys.argv), upstreams=(tracks_ref, frames_ref),
            config=audits_config,
            artifact_identity=pipeline.expected_artifact_identity(
                paths, config, paths_lib.SEMANTIC_AUDITS,
                build_inputs=build_inputs),
            recipe=pipeline.stage_recipe(
                paths, config, "audit", build_inputs=build_inputs),
            declared_outputs=AUDIT_OUTPUTS) as builder:
        artifact.atomic_write_json(
            builder.output_path("audit_meta.json"), audit_meta)
        artifact.atomic_write_file(
            builder.output_path("results.jsonl"), b"".join(result_lines))
        artifact.atomic_write_json(
            builder.output_path("settings.json"), settings)
    audits_ref = builder.artifact_ref

    # The consumers' own readers are the acceptance test.
    audits = audit_io.load_audits(tracks_dir, audits_dir)
    accepted = tracklets.build_accepted_tracklets(audits.source_tracks, audits)
    if len(accepted) != len(tracks):
        raise DetectionTrackError(
            f"{len(accepted)} of {len(tracks)} detection tracks were accepted "
            "by the tracklet contract")
    print(f"published {len(requests)} passthrough audits: {audits_dir}")
    return tracks_ref, audits_ref


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build_dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        build(args.build_dir)
    except (artifact.ArtifactError, audit_io.AuditArtifactError,
            dataset_lib.ContractViolation, DetectionTrackError,
            pipeline.StageContractError, pipeline.StageDependencyError,
            tracklets.TrackletContractError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
