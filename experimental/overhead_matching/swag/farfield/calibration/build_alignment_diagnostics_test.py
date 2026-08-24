import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    dataset,
    geometry,
    nominal_forward,
    paths,
)
from experimental.overhead_matching.swag.farfield.calibration import (
    build_alignment_diagnostics as subject,
)


class AlignmentDiagnosticsArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_base = self.root / "dataset"
        self.dataset_base.mkdir()
        (self.dataset_base / "panorama").mkdir()
        (self.dataset_base / "pipeline_metadata.json").write_text(json.dumps({
            "dataset_name": "ds",
            "is_equirectangular": True,
            "north_aligned": False,
            "azimuth_convention": {
                "images_rotated": False,
                "camera_frame": geometry.CAMERA_FRAME,
            },
        }))
        rows = ["idx,latitude,longitude,dist_m,video_t_s"]
        for index in range(8):
            latitude = 42.0 + index * 0.0001
            longitude = -71.0
            rows.append(
                f"{index},{latitude:.7f},{longitude:.7f},{index * 11.1},"
                f"{index}.0")
            Image.new("RGB", (360, 180), color=(0, 0, 0)).save(
                self.dataset_base / "panorama" /
                f"f{index:04d},{latitude:.7f},{longitude:.7f},.jpg")
        (self.dataset_base / "frames_gps.csv").write_text("\n".join(rows) + "\n")

        self.nominal_forward_path = self.dataset_base / "nominal_forward.json"
        self.nominal_forward_path.write_text(json.dumps({
            "schema": nominal_forward.SCHEMA,
            "frame": nominal_forward.FRAME,
            "dataset": "ds",
            "version": "nominal-v1",
            "mounting_id": "rig-1",
            "panorama_column": 200.0,
            "panorama_width": 360,
            "bearing_camera_cw_deg": 20.0,
            "uncertainty_deg": 1.0,
            "evidence_frame_ids": ["f0000"],
            "operator": "reviewer",
            "approved_at": "2026-08-24T12:00:00Z",
            "approved": True,
            "notes": "test fixture",
        }, sort_keys=True) + "\n")

        self.build_dir = self.root / "build"
        self.config = {
            "artifacts": {
                "object_tracks_version": "tracks-v1",
                "semantic_audits_version": "audits-v1",
                "bearing_observations_version": "obs-v1",
                "alignment_diagnostics_version": "diag-v1",
            },
            # Deliberately different: diagnostics must use the settings bound
            # by object_tracks, not independently re-resolve this subtree.
            "gps_course": {
                "min_displacement_m": 9.0,
                "smooth_window_s": 4.0,
            },
            "localization_inputs": {
                "nominal_forward_calibration":
                    str(self.nominal_forward_path.resolve()),
            },
            "alignment_diagnostics": {
                "sun": {
                    "n_frames": 4,
                    "min_speed_mps": 0.0,
                    "elevation_tolerance_deg": 3.0,
                    "work_width": 360,
                },
                "sweep": {
                    "coarse_step_deg": 10.0,
                    "fine_step_deg": 1.0,
                    "fine_halfwidth_deg": 5.0,
                    "min_observations": 3,
                    "min_arc_deg": 0.0,
                    "max_condition": 1000000000.0,
                    "min_tracklets": 2,
                    "min_support_frac": 0.5,
                },
            },
        }
        self.dataset_digests = paths.dataset_source_digests(
            self.dataset_base)
        self.build_path = build_config.create(
            self.build_dir, dataset="ds", config=self.config,
            generator="test", inputs={
                "dataset_base": self.dataset_base,
                **self.dataset_digests,
            })
        document = build_config.load(self.build_dir)
        self.build_identity = document["build_identity"]
        self.course = {"min_displacement_m": 1.0, "smooth_window_s": 0.0}
        self.tracks_ref = self._publish_tracks()
        self.audits_ref = self._publish_artifact(
            self.root / "audits-v1", paths.SEMANTIC_AUDITS, "audits-v1",
            upstreams=(self.tracks_ref,),
            config={"build_identity": self.build_identity})
        self.observations_dir = self.root / "obs-v1"
        self.observations_ref = self._publish_observations(
            self.observations_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def _publish_artifact(self, destination, kind, version, *, upstreams=(),
                          config=None):
        with artifact.ArtifactDirectoryBuilder(
                destination, kind=kind, dataset="ds", version=version,
                generator="test", git_commit="test", arguments=(),
                upstreams=upstreams, config=config or {},
                declared_outputs=("payload.json",)) as builder:
            builder.output_path("payload.json").write_text("{}\n")
        return artifact.open_artifact(destination)

    def _publish_tracks(self):
        return self._publish_artifact(
            self.root / "tracks-v1", paths.OBJECT_TRACKS, "tracks-v1",
            config={
                "schema": "farfield_object_tracks/v1",
                "coverage": "complete",
                "build_identity": self.build_identity,
                "resolved": {"gps_course": dict(self.course)},
                "source_digests": {
                    "dataset_tracking_inputs":
                        subject.dataset_source_digest(self.dataset_base),
                },
            })

    def _observation_records(self):
        frames = dataset.load_frames(self.dataset_base)
        dataset.fill_enu(frames)
        model = subject._course_model(frames, self.course)
        records = []
        candidate = 30.0
        landmarks = ((45.0, 120.0), (-55.0, 145.0), (70.0, 175.0))
        for track_index, (landmark_east, landmark_north) in enumerate(landmarks):
            tracklet_id = f"object_tracks:ds:tracks-v1#T{track_index}"
            for frame in frames:
                world = geometry.compass_bearing_deg(
                    landmark_east - frame.x_m, landmark_north - frame.y_m)
                course = float(model.course_world_cw_deg_at(frame.time_s))
                records.append({
                    "tracklet_id": tracklet_id,
                    "keyframe_idx": frame.frame_idx,
                    "bearing_camera_cw_deg":
                        (world - course + candidate) % 360.0,
                    "angular_width_deg": 2.0,
                    "sigma_deg": 1.0,
                    "correlation_group": f"{tracklet_id}/segment0",
                })
        return sorted(records, key=lambda item: (
            item["tracklet_id"], item["keyframe_idx"]))

    def _publish_observations(self, destination, *, count_delta=0,
                              version=None):
        records = self._observation_records()
        config = {
            "orchestration": {
                "schema": "farfield_pipeline_stage/v1",
                "stage": "bearings",
                "config_digest": "a" * 64,
            },
            "build_identity": self.build_identity,
            "schema": "farfield_bearing_observations/v1",
            "pano_width": 360,
            "bearing_sigma_deg": 1.0,
            "n_accepted_tracklets": 3,
            "n_observations": len(records) + count_delta,
            "coverage": "complete",
            "source_digests": {
                "build_config": artifact.sha256_file(self.build_path),
                "dataset_tracking_inputs":
                    subject.dataset_source_digest(self.dataset_base),
                paths.OBJECT_TRACKS: self.tracks_ref.content_digest,
                paths.SEMANTIC_AUDITS: self.audits_ref.content_digest,
            },
        }
        with artifact.ArtifactDirectoryBuilder(
                destination, kind=paths.BEARING_OBSERVATIONS,
                dataset="ds", version=version or destination.name,
                generator="test", git_commit="test", arguments=(),
                upstreams=(self.tracks_ref, self.audits_ref), config=config,
                declared_outputs=("observations.jsonl",)) as builder:
            builder.output_path("observations.jsonl").write_text("".join(
                json.dumps(record, sort_keys=True) + "\n" for record in records))
        return artifact.open_artifact(destination)

    def _args(self, **changes):
        document = build_config.load(self.build_dir)
        values = {
            "dataset": "ds",
            "dataset_base": self.dataset_base,
            "observations_dir": self.observations_dir,
            "nominal_forward_calibration": self.nominal_forward_path,
            "output_dir": self.root / "diag-v1",
            "build_config": self.build_dir / build_config.BUILD_CONFIG_NAME,
            "orchestration_config_digest":
                subject.orchestration_contract(document)["config_digest"],
        }
        values.update(changes)
        return SimpleNamespace(**values)

    @staticmethod
    def _all_keys(value):
        keys = set()
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                keys.update(AlignmentDiagnosticsArtifactTest._all_keys(child))
        elif isinstance(value, list):
            for child in value:
                keys.update(AlignmentDiagnosticsArtifactTest._all_keys(child))
        return keys

    def test_publishes_one_candidate_only_transactional_artifact(self):
        resolved = subject._load_inputs(self._args())
        reference = subject.publish(resolved, self.root / "diag-v1")
        self.assertEqual(reference.kind, paths.ALIGNMENT_DIAGNOSTICS)
        manifest = artifact.load_manifest(self.root / "diag-v1")
        self.assertEqual(manifest.upstreams, (self.observations_ref,))
        self.assertEqual(manifest.declared_outputs,
                         (subject.OUTPUT_NAME, subject.SUN_REVIEW_NAME))
        self.assertEqual(manifest.config["authority"], subject.AUTHORITY)
        self.assertEqual(
            manifest.config["resolved"]["gps_course_from_object_tracks"],
            self.course)
        report = json.loads(
            (self.root / "diag-v1" / subject.OUTPUT_NAME).read_text())
        self.assertEqual(report["authority"], subject.AUTHORITY)
        self.assertEqual(report["source"]["gps_course"]["parameters"],
                         self.course)
        self.assertNotIn("approved", self._all_keys(report))
        self.assertNotIn("usable", self._all_keys(report))
        sweep = report["methods"][0]
        self.assertEqual(sweep["status"], "candidate_reported")
        self.assertAlmostEqual(
            sweep[subject.RESULT_FIELD], 30.0,
            delta=2.0)
        self.assertAlmostEqual(
            sweep["comparison_to_approved_nominal_forward"]
                 ["candidate_minus_nominal_forward_cw_deg"],
            10.0, delta=2.0)
        self.assertEqual(sweep["result_kind"], subject.RESULT_KIND)
        self.assertEqual(sweep["frame"], subject.RESULT_FRAME)
        self.assertEqual(report["quantity"]["name"], subject.RESULT_FIELD)
        review_path = self.root / "diag-v1" / subject.SUN_REVIEW_NAME
        with Image.open(review_path) as review:
            self.assertEqual(review.format, "JPEG")
            self.assertGreater(review.width, 0)
            self.assertGreater(review.height, 0)
        self.assertEqual(report["methods"][1]["status"], "no_candidate")
        self.assertIn("log_start_utc", report["methods"][1]["reason"])

    def test_dataset_mutation_is_rejected_against_tracks_source_digest(self):
        (self.dataset_base / "frames_gps.csv").write_text(
            (self.dataset_base / "frames_gps.csv").read_text() + "\n")
        with self.assertRaisesRegex(
                subject.AlignmentDiagnosticError, "dataset source bytes"):
            subject._load_inputs(self._args())

    def test_observation_count_mismatch_fails_closed(self):
        bad_dir = self.root / "bad-obs"
        self._publish_observations(bad_dir, count_delta=1, version="obs-v1")
        with self.assertRaisesRegex(
                subject.AlignmentDiagnosticError, "count disagrees"):
            subject._load_inputs(self._args(observations_dir=bad_dir))

    def test_orchestration_digest_is_required_exactly(self):
        with self.assertRaisesRegex(
                subject.AlignmentDiagnosticError,
                "orchestration_config_digest"):
            subject._load_inputs(self._args(
                orchestration_config_digest="0" * 64))

    def test_nominal_forward_path_is_bound_to_build_config(self):
        other = self.root / "other-nominal-forward.json"
        other.write_bytes(self.nominal_forward_path.read_bytes())
        with self.assertRaisesRegex(
                subject.AlignmentDiagnosticError,
                "--nominal_forward_calibration"):
            subject._load_inputs(self._args(
                nominal_forward_calibration=other))


if __name__ == "__main__":
    unittest.main()
