import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import msgspec

from common.python.serialization import msgspec_enc_hook
from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
    nominal_forward,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    run_io,
    structs,
)


DATASET = "tiny_harbor"
GLOBAL_ID = "object_tracks:tiny_harbor:v1@sha256:" + "a" * 64 + "#T1"
MOTION_BYTES = b"idx,latitude,longitude,dist_m,video_t_s\n"


def nominal_document():
    width = 64
    column = 32.0
    return {
        "schema": nominal_forward.SCHEMA,
        "frame": nominal_forward.FRAME,
        "approved": True,
        "dataset": DATASET,
        "version": "v1",
        "mounting_id": "rig-a",
        "panorama_column": column,
        "panorama_width": width,
        "bearing_camera_cw_deg": float(
            geo.azimuth_of_pano_column(column, width)) % 360.0,
        "uncertainty_deg": 0.5,
        "evidence_frame_ids": ["f0000"],
        "operator": "reviewer",
        "approved_at": "2026-08-23T00:00:00Z",
        "notes": "approved record",
    }


def publish_placeholder(root: Path, kind: str):
    destination = root / kind
    with artifact.ArtifactDirectoryBuilder(
            destination, kind=kind, dataset=DATASET, version="v1",
            generator="test", git_commit="test", arguments=(),
            declared_outputs=("payload.json",)) as builder:
        artifact.atomic_write_json(builder.output_path("payload.json"), {})
    return artifact.open_artifact(destination)


def valid_meta(nominal_bytes: bytes, **overrides):
    document = nominal_document()
    meta = {
        "schema_version": export_ingest.EXPORT_SCHEMA,
        "message_schema_version": structs.SCHEMA_VERSION,
        "dataset": DATASET,
        "scenario_name": "test-experiment",
        "anchor_lat_deg": 42.35,
        "anchor_lon_deg": -71.05,
        "n_keyframes": 3,
        "matcher_version": "matcher-v1",
        "matching_coverage": "complete",
        "max_visible_range_m": 10000.0,
        "landmark_position_sigma_m": 25.0,
        "nominal_forward": {
            "file": "nominal_forward.json",
            "source_path": "/source/nominal_forward.json",
            "content_sha256": hashlib.sha256(nominal_bytes).hexdigest(),
            "schema": document["schema"],
            "frame": document["frame"],
            "dataset": document["dataset"],
            "version": document["version"],
            "mounting_id": document["mounting_id"],
            "panorama_column": document["panorama_column"],
            "panorama_width": document["panorama_width"],
            "bearing_camera_cw_deg": document["bearing_camera_cw_deg"],
            "uncertainty_deg": document["uncertainty_deg"],
            "evidence_frame_ids": document["evidence_frame_ids"],
            "operator": document["operator"],
            "approved_at": document["approved_at"],
            "notes": document["notes"],
        },
        "motion": {
            "file": "motion_source.csv",
            "source_path": "/source/frames_gps.csv",
            "content_sha256": hashlib.sha256(MOTION_BYTES).hexdigest(),
            "course_heading_status": "gps_course_diagnostic_only",
            "reverse_annotation_source": "reviewer: no reverse motion",
        },
        "reducer": {
            "name": "epoch_fused_compat_v1",
            "epoch_keyframes": 2,
            "input_frame": "camera_cw_deg",
            "output_frame": "nominal_forward_cw_deg",
        },
    }
    meta.update(overrides)
    return meta


def write_export(root: Path, *, meta_mutator=None, landmarks=None,
                 measurements=None, tables=None, odometry=None, truth=None,
                 manifest_mutator=None):
    refs = tuple(publish_placeholder(root, kind) for kind in (
        paths_lib.BEARING_OBSERVATIONS,
        paths_lib.LANDMARK_MATCHES,
        paths_lib.CATALOGS,
    ))
    document = nominal_document()
    nominal_bytes = json.dumps(document, sort_keys=True).encode()
    meta = valid_meta(nominal_bytes)
    if meta_mutator is not None:
        meta_mutator(meta)
    if landmarks is None:
        landmarks = [
            structs.LandmarkEntry(
                "osm:node:1", 42.36, -71.05, "man_made=lighthouse", 25.0),
            structs.LandmarkEntry(
                "osm:node:2", 42.35, -71.03, "natural=peak", 25.0),
        ]
    if tables is None:
        tables = [structs.CompatibilityTable(
            tracklet_id=GLOBAL_ID,
            matcher_version="matcher-v1",
            entries=[structs.CompatibilityEntry("osm:node:1", 1.5)],
            default_log_lr=0.0,
            clip_lo=-4.0,
            clip_hi=4.0,
            status="fast")]
    if measurements is None:
        measurements = [structs.TrackletMeasurement(
            tracklet_id=GLOBAL_ID,
            anchor_keyframe_idx=1,
            bearing_forward_cw_deg=45.0,
            kappa=100.0)]
    if odometry is None:
        odometry = [structs.OdometryDelta(
            keyframe_idx=keyframe,
            forward_m=40.0,
            left_m=0.0,
            delta_yaw_cw_rad=0.0,
            sigma_m=1.0,
            sigma_yaw_rad=0.02) for keyframe in (1, 2)]
    if truth is None:
        truth = [structs.TruthPose(
            keyframe_idx=keyframe,
            east_m=float(keyframe),
            north_m=float(keyframe),
            course_world_cw_deg=45.0) for keyframe in range(3)]
    config = {
        "localization_inputs": {
            "max_visible_range_m": meta["max_visible_range_m"],
            "landmark_position_sigma_m": meta["landmark_position_sigma_m"],
            "reducer_epoch_keyframes": meta["reducer"]["epoch_keyframes"],
        },
        "nominal_forward_sha256": hashlib.sha256(nominal_bytes).hexdigest(),
        "motion_source_sha256": hashlib.sha256(MOTION_BYTES).hexdigest(),
        "matching_coverage": "complete",
        "matching_n_expected": 2,
        "matching_n_successful": 2,
        "reducer": meta["reducer"],
    }
    if manifest_mutator is not None:
        manifest_mutator(config)
    destination = root / "localization_inputs"
    outputs = (
        "export_meta.json",
        "landmarks.json",
        "motion_source.csv",
        "nominal_forward.json",
        "tier1_measurements.jsonl",
        "tier1_odometry.jsonl",
        "tier1_tables.json",
        "truth.jsonl",
    )
    with artifact.ArtifactDirectoryBuilder(
            destination, kind=paths_lib.LOCALIZATION_INPUTS,
            dataset=DATASET, version="v1", generator="test",
            git_commit="test", arguments=(), upstreams=refs,
            config=config, declared_outputs=outputs) as builder:
        artifact.atomic_write_json(builder.output_path("export_meta.json"), meta)
        artifact.atomic_write_file(
            builder.output_path("nominal_forward.json"), nominal_bytes)
        artifact.atomic_write_file(
            builder.output_path("motion_source.csv"), MOTION_BYTES)
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
    return destination


class LoadTest(unittest.TestCase):
    def test_loads_a_valid_manifest_owned_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = write_export(Path(temporary))
            data = export_ingest.load(export, expected_dataset=DATASET)
            self.assertEqual(data.n_keyframes, 3)
            self.assertEqual(data.catalog.n, 2)
            self.assertEqual(float(data.catalog.max_visible_range_m[0]),
                             10000.0)
            self.assertEqual(float(data.catalog.position_sigma_m[0]), 25.0)
            self.assertIn("complete coverage", export_ingest.describe(data))

    def test_unknown_legacy_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = write_export(
                Path(temporary),
                meta_mutator=lambda meta: meta.update({
                    "mount_offset_deg": 214.0,
                }))
            with self.assertRaisesRegex(ValueError, "unknown field"):
                export_ingest.load(export)

    def test_nominal_forward_digest_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = write_export(
                Path(temporary),
                meta_mutator=lambda meta: meta["nominal_forward"].update({
                    "content_sha256": "0" * 64,
                }))
            with self.assertRaisesRegex(ValueError, "nominal-forward digest"):
                export_ingest.load(export)

    def test_matching_must_attest_complete_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = write_export(
                Path(temporary), manifest_mutator=lambda config: config.update({
                    "matching_n_successful": 1,
                }))
            with self.assertRaisesRegex(ValueError, "complete successful"):
                export_ingest.load(export)

    def test_position_sigma_must_be_uniform_positive(self):
        with tempfile.TemporaryDirectory() as temporary:
            landmarks = [
                structs.LandmarkEntry(
                    "osm:node:1", 42.36, -71.05, "tower", 25.0),
                structs.LandmarkEntry(
                    "osm:node:2", 42.35, -71.03, "peak", 10.0),
            ]
            export = write_export(Path(temporary), landmarks=landmarks)
            with self.assertRaisesRegex(ValueError, "uniform recorded value"):
                export_ingest.load(export)

    def test_cw_bearing_range_and_global_tracklet_id_are_enforced(self):
        cases = (
            (structs.TrackletMeasurement(GLOBAL_ID, 1, -1.0, 10.0),
             r"\[0, 360\)"),
            (structs.TrackletMeasurement("T1", 1, 45.0, 10.0),
             "global tracklet id"),
        )
        for measurement, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    export = write_export(
                        Path(temporary), measurements=[measurement])
                    with self.assertRaisesRegex(ValueError, message):
                        export_ingest.load(export)

    def test_odometry_indices_are_contiguous(self):
        with tempfile.TemporaryDirectory() as temporary:
            odometry = [structs.OdometryDelta(
                keyframe_idx=2,
                forward_m=1.0,
                left_m=0.0,
                delta_yaw_cw_rad=0.0,
                sigma_m=1.0,
                sigma_yaw_rad=0.02)]
            export = write_export(Path(temporary), odometry=odometry)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                export_ingest.load(export)

    def test_course_abstention_requires_empty_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            export = write_export(
                Path(temporary), truth=[],
                meta_mutator=lambda meta: meta["motion"].update({
                    "course_heading_status":
                    "abstained_insufficient_displacement",
                }))
            data = export_ingest.load(export)
            self.assertEqual(data.truth, [])

    def test_region_box_rejects_negative_margin(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = export_ingest.load(write_export(Path(temporary)))
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                export_ingest.region_box(data, -1.0)


if __name__ == "__main__":
    unittest.main()
