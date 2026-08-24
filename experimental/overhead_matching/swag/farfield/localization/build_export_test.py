import dataclasses
import json
import tempfile
import types
import unittest
from pathlib import Path

import msgspec
import shapely

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    geometry as geo,
    nominal_forward,
    paths as paths_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.localization import (
    build_export,
    export_ingest,
    structs,
)
from experimental.overhead_matching.swag.farfield.tracking import tracklets


DATASET = "tiny_harbor"
PANO_W = 64


def source_track():
    return {
        "track_id": 1,
        "birth_keyframe": 0,
        "end_keyframe": 3,
        "records": [{
            "keyframe": keyframe,
            "mask_bbox_window": [28.0, 5.0, 36.0, 15.0],
            "window_origin": [0.0, 0.0],
        } for keyframe in range(4)],
    }


def audit_payload():
    return {
        "landmark_kind": "fixed_structure",
        "single_object": True,
        "valid_segments": [{"start_t": 0, "end_t": 3}],
        "verdict": "keep",
        "drop_reason": "none",
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9}],
            "name_candidates": [{
                "name": "Example Light",
                "weight": 0.8,
                "basis": "reported_by_detections",
            }],
            "name_aliases": [],
            "description": "A fixed lighthouse.",
            "distinctive_features": ["white tower"],
            "extent": "point_like",
        },
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": "high",
        "unresolved": "",
    }


def result_line(payload):
    return json.dumps({
        "key": "T1",
        "response": {"candidates": [{"content": {"parts": [{
            "text": json.dumps(payload),
        }]}}]},
    })


def write_tracks_and_audits(root: Path, build_identity: str,
                            dataset_source_digest: str):
    track = source_track()
    tracks_dir = root / "tracks"
    tracks_document = {
        "range": {"name": "full", "k_start": 0, "k_end": 3},
        "tracks": [track],
    }
    with artifact.ArtifactDirectoryBuilder(
            tracks_dir, kind=paths_lib.OBJECT_TRACKS, dataset=DATASET,
            version="v1", generator="test", git_commit="test",
            arguments=(), config={
                "build_identity": build_identity,
                "source_digests": {
                    "dataset_tracking_inputs": dataset_source_digest,
                },
            }, declared_outputs=("tracks_full.json",)) as builder:
        artifact.atomic_write_json(
            builder.output_path("tracks_full.json"), tracks_document)
    tracks_ref = artifact.open_artifact(tracks_dir)
    tracks_path = tracks_dir / "tracks_full.json"
    meta = {
        "schema": audit_io.META_SCHEMA,
        "source_tracks": {
            "artifact_id": audit_io.source_artifact_id(tracks_ref),
            "file": tracks_path.name,
            "sha256": artifact.sha256_file(tracks_path),
        },
        "requests": {
            "T1": {
                "track_id": 1,
                "range": "full",
                "birth_keyframe": 0,
                "source_track_sha256": artifact.sha256_json(track),
            },
        },
    }
    audits_dir = root / "audits"
    with artifact.ArtifactDirectoryBuilder(
            audits_dir, kind=paths_lib.SEMANTIC_AUDITS, dataset=DATASET,
            version="v1", generator="test", git_commit="test", arguments=(),
            upstreams=(tracks_ref,), config={
                "phase": "canonical_results", "coverage": "complete",
                "n_expected": 1, "n_successful": 1,
                "build_identity": build_identity,
            }, declared_outputs=("audit_meta.json", "results.jsonl")) as builder:
        artifact.atomic_write_json(builder.output_path("audit_meta.json"), meta)
        artifact.atomic_write_file(
            builder.output_path("results.jsonl"),
            (result_line(audit_payload()) + "\n").encode())
    return tracks_dir, audits_dir


def write_observations(root: Path, tracks_dir: Path, audits_dir: Path,
                       build_identity: str):
    audits = audit_io.load_audits(tracks_dir, audits_dir)
    accepted = tracklets.build_accepted_tracklets(audits.source_tracks, audits)
    observations = tracklets.build_camera_bearing_observations(
        accepted, PANO_W, 1.0)
    observations.sort(key=lambda item: (item.tracklet_id, item.keyframe_idx))
    observations_dir = root / "observations"
    payload = b"".join(
        artifact.canonical_json_bytes(dataclasses.asdict(item)) + b"\n"
        for item in observations)
    with artifact.ArtifactDirectoryBuilder(
            observations_dir, kind=paths_lib.BEARING_OBSERVATIONS,
            dataset=DATASET, version="v1", generator="test",
            git_commit="test", arguments=(),
            upstreams=(audits.tracks_ref, audits.semantic_audits_ref),
            config={
                "coverage": "complete", "bearing_sigma_deg": 1.0,
                "build_identity": build_identity,
            },
            declared_outputs=("observations.jsonl",)) as builder:
        artifact.atomic_write_file(
            builder.output_path("observations.jsonl"), payload)
    return observations_dir, accepted[0].tracklet_id


def write_catalog(root: Path, *, node_id="node:1"):
    catalog_dir = root / "catalog"
    with artifact.ArtifactDirectoryBuilder(
            catalog_dir, kind=paths_lib.CATALOGS, dataset=DATASET,
            version="v1", generator="test", git_commit="test", arguments=(),
            declared_outputs=("catalog.feather",)) as builder:
        schema.build_frame(
            ids=[node_id],
            geometries=[shapely.Point(testing.ANCHOR_LON,
                                      testing.ANCHOR_LAT + 0.01)],
            landmark_types=["osm"],
            tags=[{"man_made": "lighthouse"}],
        ).to_feather(builder.output_path("catalog.feather"))
    return catalog_dir, artifact.open_artifact(catalog_dir)


def write_matching(root: Path, tracks_dir: Path, audits_dir: Path,
                   catalog_ref, tracklet_id: str, *, coverage="complete",
                   table_tracklet_id=None, build_identity=None):
    tracks_ref = artifact.open_artifact(tracks_dir)
    audits_ref = artifact.open_artifact(audits_dir)
    if build_identity is None:
        build_identity = artifact.load_manifest(
            tracks_dir).config["build_identity"]
    matching_dir = root / f"matching_{coverage}_{len(list(root.iterdir()))}"
    table = structs.CompatibilityTable(
        tracklet_id=table_tracklet_id or tracklet_id,
        matcher_version="matcher-v1",
        entries=[structs.CompatibilityEntry("osm:node:1", 1.0)],
        default_log_lr=-1.0,
        clip_lo=-4.0,
        clip_hi=4.0,
        status="fast")
    with artifact.ArtifactDirectoryBuilder(
            matching_dir, kind=paths_lib.LANDMARK_MATCHES, dataset=DATASET,
            version="v1", generator="test", git_commit="test", arguments=(),
            upstreams=(tracks_ref, audits_ref, catalog_ref),
            config={"phase": "canonical_results", "coverage": coverage,
                    "n_expected": 1, "n_successful": 1,
                    "build_identity": build_identity},
            declared_outputs=("compatibility.json",)) as builder:
        artifact.atomic_write_file(
            builder.output_path("compatibility.json"),
            msgspec.json.encode([table], enc_hook=msgspec_enc_hook))
    return matching_dir


def write_nominal_forward(path: Path):
    document = {
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "approved": True,
        "dataset": DATASET,
        "version": "v1",
        "mounting_id": "rig-a",
        "panorama_column": 16.0,
        "panorama_width": PANO_W,
        "bearing_camera_cw_deg": float(
            geo.azimuth_of_pano_column(16.0, PANO_W)) % 360.0,
        "uncertainty_deg": 0.5,
        "evidence_frame_ids": ["f0000"],
        "operator": "reviewer",
        "approved_at": "2026-08-23T00:00:00Z",
        "notes": "human annotation",
    }
    path.write_text(json.dumps(document))
    return path


def build_fixture(root: Path):
    base = testing.make_dataset(
        root / "datasets" / DATASET, n_frames=4,
        pano_size=(PANO_W, PANO_W // 2))
    catalog_dir, catalog_ref = write_catalog(root)
    calibration = write_nominal_forward(base / "nominal_forward.json")
    config = {
        "experiment": {"name": "test-experiment"},
        "artifacts": {
            "object_tracks_version": "v1",
            "semantic_audits_version": "v1",
            "bearing_observations_version": "v1",
            "landmark_matches_version": "v1",
            "catalogs_version": "v1",
            "localization_inputs_version": "v1",
        },
        "gps_course": {"min_displacement_m": 2.0,
                       "smooth_window_s": 0.0},
        "localization_inputs": {
            "motion_source": str((base / "frames_gps.csv").resolve()),
            "nominal_forward_calibration": str(calibration.resolve()),
            "use_uninformative_tables": False,
            "default_log_compatibility": 0.0,
            "compatibility_clip": 4.0,
            "reducer_epoch_keyframes": 2,
            "odometry_sigma_pair_m": 1.0,
            "displacement_gate_m": 2.0,
            "stationary_sigma_m": 3.0,
            "slow_yaw_sigma_deg": 30.0,
            "reverse_keyframe_ranges": [],
            "reverse_annotation_source": "reviewer: no reverse motion",
            "max_visible_range_m": 10000.0,
            "landmark_position_sigma_m": 25.0,
        },
    }
    dataset_digests = paths_lib.dataset_source_digests(base)
    build_dir = root / "builds" / DATASET / "b001"
    config_path = build_config.create(
        build_dir, dataset=DATASET, config=config, generator="test",
        inputs={
            "dataset_base": base,
            "motion_source": str((base / "frames_gps.csv").resolve()),
            "motion_source_sha256": artifact.sha256_file(
                base / "frames_gps.csv"),
            "nominal_forward_calibration": str(calibration.resolve()),
            "nominal_forward_sha256": artifact.sha256_file(calibration),
            "catalog_manifest_digest": catalog_ref.manifest_digest,
            "catalog_content_digest": catalog_ref.content_digest,
            **dataset_digests,
        })
    document = build_config.load(build_dir)
    build_identity = document["build_identity"]
    tracks_dir, audits_dir = write_tracks_and_audits(
        root, build_identity, artifact.sha256_json(dataset_digests))
    observations_dir, tracklet_id = write_observations(
        root, tracks_dir, audits_dir, build_identity)
    matching_dir = write_matching(
        root, tracks_dir, audits_dir, catalog_ref, tracklet_id,
        build_identity=build_identity)
    output = root / "v1"
    args = types.SimpleNamespace(
        dataset=DATASET,
        dataset_base=base,
        observations_dir=observations_dir,
        matching_dir=matching_dir,
        catalog_dir=catalog_dir,
        motion_source=base / "frames_gps.csv",
        nominal_forward_calibration=calibration,
        landmark_position_sigma_m=25.0,
        output_dir=output,
        build_config=config_path,
        orchestration_config_digest=(
            build_export.orchestration_contract(document)["config_digest"]),
    )
    return args, tracklet_id


class ReducerTest(unittest.TestCase):
    def test_rotation_uses_approved_nominal_forward(self):
        record = nominal_forward.parse({
            "schema": nominal_forward.SCHEMA,
            "frame": nominal_forward.FRAME,
            "approved": True,
            "dataset": DATASET,
            "version": "v1",
            "mounting_id": "rig",
            "panorama_column": 16.0,
            "panorama_width": PANO_W,
            "bearing_camera_cw_deg": float(
                geo.azimuth_of_pano_column(16.0, PANO_W)) % 360.0,
            "uncertainty_deg": 1.0,
            "evidence_frame_ids": ["f0"],
            "operator": "reviewer",
            "approved_at": "2026-08-23T00:00:00Z",
            "notes": "human annotation",
        })
        measurement = tracklets.Measurement(
            "artifact@sha256:" + "1" * 64 + "#T1", 3, 0.0, 10.0)
        result = build_export.forward_frame_measurements(
            [measurement], record)
        self.assertAlmostEqual(
            result[0].bearing_forward_cw_deg,
            nominal_forward.camera_to_forward_cw_deg(0.0, record))


class EndToEndTest(unittest.TestCase):
    def test_publishes_valid_strict_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, tracklet_id = build_fixture(Path(temporary))
            reference = build_export.build(args)
            data = export_ingest.load(
                args.output_dir, expected_dataset=DATASET)
            self.assertEqual(reference.kind, paths_lib.LOCALIZATION_INPUTS)
            self.assertEqual(set(data.tables), {tracklet_id})
            self.assertEqual({item.tracklet_id for item in data.measurements},
                             {tracklet_id})
            self.assertTrue(all(
                item.position_sigma_m == 25.0 for item in data.landmarks))
            self.assertTrue(all(
                0.0 <= item.bearing_forward_cw_deg < 360.0
                for item in data.measurements))
            manifest = artifact.load_manifest(args.output_dir)
            self.assertEqual(
                [item.kind for item in manifest.upstreams],
                [paths_lib.BEARING_OBSERVATIONS,
                 paths_lib.LANDMARK_MATCHES, paths_lib.CATALOGS])
            self.assertEqual(manifest.config["matching_coverage"], "complete")
            self.assertEqual(
                manifest.config["orchestration"],
                build_export.orchestration_contract(
                    build_config.load(args.build_config.parent)))

    def test_stale_recipe_digest_and_wrong_output_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = build_fixture(Path(temporary))
            args.orchestration_config_digest = "0" * 64
            with self.assertRaisesRegex(
                    build_export.LocalizationInputError,
                    "orchestration_config_digest"):
                build_export.build(args)
            args.orchestration_config_digest = (
                build_export.orchestration_contract(
                    build_config.load(args.build_config.parent))[
                        "config_digest"])
            args.output_dir = Path(temporary) / "wrong-version"
            with self.assertRaisesRegex(
                    build_export.LocalizationInputError, "output_dir"):
                build_export.build(args)

    def test_dataset_mutation_is_rejected_against_build_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _ = build_fixture(Path(temporary))
            with Path(args.motion_source).open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(
                    build_export.LocalizationInputError,
                    "dataset source bytes"):
                build_export.build(args)

    def test_substituted_catalog_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, _ = build_fixture(root)
            substitute_dir, _ = write_catalog(
                root / "substitute", node_id="node:other")
            args.catalog_dir = substitute_dir
            with self.assertRaisesRegex(
                    build_export.LocalizationInputError,
                    "catalog artifact identity"):
                build_export.build(args)

    def test_matching_partial_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, tracklet_id = build_fixture(root)
            args.matching_dir = write_matching(
                root, artifact.load_manifest(args.observations_dir).upstreams[0].path,
                artifact.load_manifest(args.observations_dir).upstreams[1].path,
                artifact.open_artifact(args.catalog_dir), tracklet_id,
                coverage="partial")
            with self.assertRaisesRegex(
                    build_export.LocalizationInputError, "coverage='complete'"):
                build_export.build(args)


if __name__ == "__main__":
    unittest.main()
