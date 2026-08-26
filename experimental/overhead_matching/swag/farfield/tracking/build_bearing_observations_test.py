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

    def test_publishes_exact_lossless_schema_and_upstreams(self):
        destination = self.root / "observations"
        ref = subject.publish_observations(
            destination, dataset_name="ds", version="obs-v1",
            tracks_ref=self.tracks_ref, audits_ref=self.audits_ref,
            accepted_tracklets=[self.accepted()], pano_width=1000,
            bearing_sigma_deg=2.5,
            orchestration={
                "schema": "farfield_pipeline_stage/v1",
                "stage": "bearings",
                "config_digest": "a" * 64,
            },
            build_identity="b" * 64, source_digests={})
        self.assertEqual(ref.kind, paths.BEARING_OBSERVATIONS)
        manifest = artifact.load_manifest(destination)
        self.assertEqual(manifest.upstreams,
                         (self.tracks_ref, self.audits_ref))
        self.assertEqual(manifest.config["coverage"], "complete")
        self.assertEqual(manifest.config["build_identity"], "b" * 64)
        self.assertNotIn("stage_reuse", manifest.config)
        records = [json.loads(line) for line in
                   (destination / subject.OUTPUT_NAME).read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(set(records[0]), {
            "tracklet_id", "keyframe_idx", "bearing_camera_cw_deg",
            "angular_width_deg", "sigma_deg", "correlation_group"})
        self.assertEqual(records[0]["keyframe_idx"], 4)
        self.assertEqual(records[0]["sigma_deg"], 2.5)
        self.assertNotEqual(records[0]["tracklet_id"], "T7")

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
                    "object_tracks_version": "tracks-v1",
                    "semantic_audits_version": "audits-v1",
                    "bearing_observations_version": "obs-v1",
                },
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
        self.tracks_ref = self._publish(
            self.root / "artifacts" / paths.OBJECT_TRACKS / "ds"
            / "tracks-v1", paths.OBJECT_TRACKS, "tracks-v1",
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
            "output_dir": self.root / "artifacts"
                / paths.BEARING_OBSERVATIONS / "ds" / "obs-v1",
            "build_config": self.build_path,
            "orchestration_config_digest":
                subject.orchestration_contract(document)["config_digest"],
        }
        values.update(updates)
        return types.SimpleNamespace(**values)

    def test_resolves_exact_configured_sources_and_versions(self):
        bridge = {"schema": "farfield_stage_reuse_bridge/v1"}
        with (mock.patch.object(
                subject.audit_io, "load_audits",
                return_value=self.fake_audits),
              mock.patch.object(
                  subject.stage_reuse, "require_compatible_artifact",
                  return_value=bridge) as require_reuse,
              mock.patch.object(
                  subject.stage_reuse, "require_recorded_bridge")):
            resolved = subject.load_inputs(self._args())
        self.assertEqual(resolved["output_version"], "obs-v1")
        self.assertEqual(
            resolved["source_digests"]["dataset_tracking_inputs"],
            artifact.sha256_json(self.dataset_digests))
        self.assertIs(resolved["stage_reuse"], bridge)
        self.assertEqual(require_reuse.call_args.kwargs["owner_stage"], "track")

    def test_dataset_mutation_fails_against_immutable_build(self):
        with (self.dataset_base / "frames_gps.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(
                subject.BearingObservationError, "dataset source bytes"):
            subject.load_inputs(self._args())

    def test_byte_identical_audit_copy_outside_configured_lane_is_rejected(self):
        alias = self.root / "alternate-audits"
        shutil.copytree(Path(self.audits_ref.path), alias)
        with self.assertRaisesRegex(ValueError, "exact configured lane"):
            subject.load_inputs(self._args(audit_dir=alias))

    def test_wrong_digest_and_output_version_are_rejected(self):
        with self.assertRaisesRegex(
                subject.BearingObservationError, "output_dir"):
            subject.load_inputs(self._args(output_dir=self.root / "wrong"))
        with self.assertRaisesRegex(
                subject.BearingObservationError,
                "orchestration_config_digest"):
            subject.load_inputs(self._args(
                orchestration_config_digest="0" * 64))


if __name__ == "__main__":
    unittest.main()
