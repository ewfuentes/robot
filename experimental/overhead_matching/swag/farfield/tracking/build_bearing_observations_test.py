import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    paths,
    testing,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    build_bearing_observations as subject,
    tracklets,
)


class BearingObservationArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tracks_ref = self.publish_upstream("object_tracks", "tracks")
        self.audits_ref = self.publish_upstream(
            "semantic_audits", "audits", upstreams=(self.tracks_ref,))

    def tearDown(self):
        self.temp.cleanup()

    def publish_upstream(self, kind, version, upstreams=()):
        destination = self.root / version
        with artifact.ArtifactDirectoryBuilder(
                destination, kind=kind, dataset="ds", version=version,
                generator="test", git_commit="test", arguments=(),
                upstreams=upstreams, declared_outputs=("payload.json",)) \
                as builder:
            builder.output_path("payload.json").write_text("{}\n")
        return artifact.open_artifact(destination)

    @staticmethod
    def accepted(*, with_box=True):
        record = {"keyframe": 4, "window_origin": [100, 0]}
        if with_box:
            record["mask_bbox_window"] = [20, 0, 60, 20]
        return tracklets.AcceptedTracklet(
            tracklet_id="object_tracks:ds:tracks@sha256:abc#T7",
            local_id="T7",
            source_track={
                "track_id": 7,
                "birth_keyframe": 4,
                "end_keyframe": 4,
                "records": [record],
            },
            audit={"verdict": "keep"},
            valid_segments=(tracklets.ValidSegment(
                index=0, start_t=0, end_t=0,
                start_keyframe_idx=4, end_keyframe_idx=4),),
            provenance={}, quality={})


    def test_accepted_track_without_bearing_fails_closed(self):
        with self.assertRaisesRegex(subject.BearingObservationError,
                                    "no bearing-capable record"):
            subject.publish_observations(
                self.root / "bad", dataset_name="ds", version="obs-v1",
                tracks_ref=self.tracks_ref, audits_ref=self.audits_ref,
                accepted_tracklets=[self.accepted(with_box=False)],
                pano_width=1000, bearing_sigma_deg=1.0,
                orchestration={
                    "schema": "farfield_pipeline_stage/v1",
                    "stage": "bearings",
                    "config_digest": "b" * 64,
                },
                build_identity="c" * 64, source_digests={})


class BearingObservationInputBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset_base = testing.make_dataset(
            self.root / "ds", n_frames=2, pano_size=(64, 32))
        self.dataset_digests = paths.dataset_source_digests(
            self.dataset_base)
        self.build_path = build_config.create(
            self.root / "build", dataset="ds", generator="test",
            config={
                "artifacts": {
                    "frame_landmarks_version": "fl-v1",
                    "object_tracks_version": "tracks-v1",
                    "semantic_audits_version": "audits-v1",
                    "bearing_observations_version": "obs-v1",
                },
                "ingest": {"fov_deg": 90.0, "seam_gap_norm": 25.0,
                           "seam_min_y_iou": 0.3},
                "tracking": {"reference_pano_width": 64},
                "bearing_observations": {"bearing_sigma_deg": 1.5},
            },
            inputs={
                "farfield_root": str(self.root.resolve()),
                "dataset_base": str(self.dataset_base.resolve()),
                **self.dataset_digests,
            })
        document = build_config.load(self.build_path.parent)
        self.build_identity = document["build_identity"]
        self.frame_ref = self._publish(
            self.root / "artifacts" / paths.FRAME_LANDMARKS / "ds"
            / "fl-v1", paths.FRAME_LANDMARKS, "fl-v1", config={})
        self.tracks_ref = self._publish(
            self.root / "artifacts" / paths.OBJECT_TRACKS / "ds"
            / "tracks-v1", paths.OBJECT_TRACKS, "tracks-v1",
            upstreams=(self.frame_ref,),
            config={
                "build_identity": self.build_identity,
                "source_digests": {
                    "dataset_tracking_inputs":
                        artifact.sha256_json(self.dataset_digests),
                },
            })
        self.audits_ref = self._publish(
            self.root / "artifacts" / paths.SEMANTIC_AUDITS / "ds"
            / "audits-v1", paths.SEMANTIC_AUDITS, "audits-v1",
            upstreams=(self.tracks_ref,),
            config={"build_identity": self.build_identity})
        self.fake_audits = types.SimpleNamespace(
            tracks_ref=self.tracks_ref,
            semantic_audits_ref=self.audits_ref)

    def _publish(self, destination, kind, version, *, upstreams=(), config):
        with artifact.ArtifactDirectoryBuilder(
                destination, kind=kind, dataset="ds", version=version,
                generator="test", git_commit="test", arguments=(),
                upstreams=upstreams, config=config,
                declared_outputs=("payload.json",)) as builder:
            artifact.atomic_write_json(
                builder.output_path("payload.json"), {})
        return artifact.open_artifact(destination)

    def _args(self, **updates):
        document = build_config.load(self.build_path.parent)
        values = {
            "dataset": "ds",
            "dataset_base": self.dataset_base,
            "tracks_dir": Path(self.tracks_ref.path),
            "audit_dir": Path(self.audits_ref.path),
            "frame_landmarks_dir": Path(self.frame_ref.path),
            "output_dir": self.root / "artifacts"
                / paths.BEARING_OBSERVATIONS / "ds" / "obs-v1",
            "build_config": self.build_path,
            "orchestration_config_digest":
                subject.orchestration_contract(document)["config_digest"],
        }
        values.update(updates)
        return types.SimpleNamespace(**values)


    def test_dataset_mutation_fails_against_immutable_build(self):
        with (self.dataset_base / "frames_gps.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(
                subject.BearingObservationError, "dataset source bytes"):
            subject.load_inputs(self._args())


    def test_wrong_digest_and_output_version_are_rejected(self):
        with self.assertRaisesRegex(
                subject.BearingObservationError, "output_dir"):
            subject.load_inputs(self._args(output_dir=self.root / "wrong"))
        with self.assertRaisesRegex(
                subject.BearingObservationError,
                "orchestration_config_digest"):
            subject.load_inputs(self._args(
                orchestration_config_digest="0" * 64))

    def test_valid_inputs_load_and_bind_tracks_sources(self):
        detection = types.SimpleNamespace(
            obs_id="obs-1", additional_tags=[["distance_estimate",
                                              "under_100m"]])
        fake_ingest = types.SimpleNamespace(
            frame_landmarks_ref=self.frame_ref, observations=[detection])
        with (mock.patch.object(subject.audit_io, "load_audits",
                                return_value=self.fake_audits),
              mock.patch.object(subject.dataset, "run_ingest",
                                return_value=fake_ingest) as run_ingest):
            loaded = subject.load_inputs(self._args())
        self.assertIs(loaded["audits"], self.fake_audits)
        self.assertEqual(
            loaded["source_digests"]["dataset_tracking_inputs"],
            artifact.sha256_json(self.dataset_digests))
        self.assertEqual(loaded["source_digests"][paths.FRAME_LANDMARKS],
                         self.frame_ref.content_digest)
        self.assertEqual(loaded["obs_by_id"], {"obs-1": detection})
        params = run_ingest.call_args.args[2]
        self.assertEqual((params.fov_deg, params.seam_gap_norm,
                          params.seam_min_y_iou), (90.0, 25.0, 0.3))

    def test_frame_landmarks_must_be_the_one_the_tracks_bind(self):
        other = self._publish(
            self.root / "artifacts" / paths.FRAME_LANDMARKS / "ds"
            / "fl-other", paths.FRAME_LANDMARKS, "fl-other",
            config={"note": "different"})
        document = build_config.load(self.build_path.parent)
        document["config"]["artifacts"]["frame_landmarks_version"] = (
            "fl-other")
        with (mock.patch.object(subject.audit_io, "load_audits",
                                return_value=self.fake_audits),
              mock.patch.object(subject.build_config, "load",
                                return_value=document)):
            with self.assertRaisesRegex(
                    subject.BearingObservationError,
                    "bind the exact frame_landmarks"):
                subject.load_inputs(self._args(
                    frame_landmarks_dir=Path(other.path)))

    def test_tracks_missing_source_binding_is_rejected(self):
        stale_tracks = self._publish(
            self.root / "artifacts" / paths.OBJECT_TRACKS / "ds"
            / "tracks-stale", paths.OBJECT_TRACKS, "tracks-stale",
            config={"build_identity": self.build_identity})
        stale_audits = self._publish(
            self.root / "artifacts" / paths.SEMANTIC_AUDITS / "ds"
            / "audits-stale", paths.SEMANTIC_AUDITS, "audits-stale",
            upstreams=(stale_tracks,),
            config={"build_identity": self.build_identity})
        fake_audits = types.SimpleNamespace(
            tracks_ref=stale_tracks,
            semantic_audits_ref=stale_audits)
        document = build_config.load(self.build_path.parent)
        document["config"]["artifacts"]["object_tracks_version"] = (
            "tracks-stale")
        document["config"]["artifacts"]["semantic_audits_version"] = (
            "audits-stale")
        with (mock.patch.object(subject.audit_io, "load_audits",
                                return_value=fake_audits),
              mock.patch.object(subject.build_config, "load",
                                return_value=document)):
            with self.assertRaisesRegex(
                    subject.BearingObservationError,
                    "does not bind the current frozen dataset sources"):
                subject.load_inputs(self._args(
                    tracks_dir=Path(stale_tracks.path),
                    audit_dir=Path(stale_audits.path)))


if __name__ == "__main__":
    unittest.main()
