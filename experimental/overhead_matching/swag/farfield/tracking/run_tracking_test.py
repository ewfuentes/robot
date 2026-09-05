"""Strict immutable object-tracks producer tests."""

import argparse
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experimental.overhead_matching.swag.farfield import (
    artifact,
    build_config,
    paths as paths_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    range_runner as rr,
    run_tracking as rt,
    track_builder as tb,
)


DATASET = "tracking_test"
PANO_W = 64
PANO_H = 32
K_START = 0
K_END = 1
PINHOLE_RES = 24


def write_artifact(path: Path, kind: str, version: str, *, upstreams=(),
                   config=None) -> Path:
    with artifact.ArtifactDirectoryBuilder(
            path,
            kind=kind,
            dataset=DATASET,
            version=version,
            generator="run_tracking_test",
            git_commit="test",
            arguments=(),
            upstreams=upstreams,
            config=config or {},
            declared_outputs=("payload.txt",)) as builder:
        builder.output_path("payload.txt").write_text(kind)
    return path


def tracking_config(checkpoint: Path, *, clean_iou=0.67) -> dict:
    builder = dataclasses.asdict(
        tb.TrackBuilderConfig(reference_pano_width=PANO_W))
    builder["clean_iou"] = clean_iou
    return {
        "artifacts": {
            "frame_landmarks_version": "landmarks-v1",
            "pinhole_images_version": "pinholes-v1",
            "object_tracks_version": "tracks-v1",
        },
        "ingest": {
            "fov_deg": 91.0,
            "seam_gap_norm": 24.0,
            "seam_min_y_iou": 0.31,
        },
        "tracking": {
            **builder,
            "sam2_checkpoint": str(checkpoint.resolve()),
            "range": {"k_start": K_START, "k_end": K_END},
        },
        "gps_course": {
            "min_displacement_m": 2.5,
            "smooth_window_s": 6.0,
        },
    }


class ProducerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.dataset_base = testing.make_dataset(
            root / "datasets" / DATASET, n_frames=2,
            pano_size=(PANO_W, PANO_H))
        self.checkpoint = root / "models" / "sam2.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"fixed-sam2-weights")
        self.dataset_digests = paths_lib.dataset_source_digests(
            self.dataset_base)
        self.output = (
            root / "artifacts" / paths_lib.OBJECT_TRACKS
            / DATASET / "tracks-v1")
        self.build_dir = root / "builds" / DATASET / "b001"
        self.config = tracking_config(self.checkpoint)
        self.build_path = build_config.create(
            self.build_dir,
            dataset=DATASET,
            config=self.config,
            generator="run_tracking_test",
            inputs={
                "dataset_base": str(self.dataset_base.resolve()),
                "sam2_checkpoint": str(self.checkpoint.resolve()),
                "sam2_checkpoint_sha256": artifact.sha256_file(
                    self.checkpoint),
                **self.dataset_digests,
            })
        document = build_config.load(self.build_dir)
        self.build_identity = document["build_identity"]
        self.digest = rt.orchestration_contract(document)["config_digest"]
        extraction_config = paths_lib.pinhole_manifest_config(
            self.dataset_digests,
            resolution=PINHOLE_RES,
            panorama_keys=sorted(
                path.stem for path in
                (self.dataset_base / "panorama").glob("*.jpg")))
        self.pinholes = write_artifact(
            root / "artifacts" / paths_lib.PINHOLE_IMAGES
            / DATASET / "pinholes-v1",
            paths_lib.PINHOLE_IMAGES, "pinholes-v1",
            config=extraction_config)
        pinhole_ref = artifact.open_artifact(self.pinholes)
        self.frame_landmarks = write_artifact(
            root / "artifacts" / paths_lib.FRAME_LANDMARKS
            / DATASET / "landmarks-v1",
            paths_lib.FRAME_LANDMARKS, "landmarks-v1",
            upstreams=(pinhole_ref,),
            config={"build_identity": self.build_identity})

    def args(self, **updates):
        values = {
            "frame_landmarks_dir": self.frame_landmarks,
            "pinhole_dir": self.pinholes,
            "checkpoint": self.checkpoint,
            "output_dir": self.output,
            "dataset": DATASET,
            "dataset_base": self.dataset_base,
            "k_start": K_START,
            "k_end": K_END,
            "build_config": self.build_path,
            "orchestration_config_digest": self.digest,
            "video": None,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def context(self):
        return {
            "result": SimpleNamespace(frames=[
                SimpleNamespace(frame_idx=K_START),
                SimpleNamespace(frame_idx=K_END),
            ]),
            "pano_w": PANO_W,
            "pano_h": PANO_H,
            "obs_by_frame": {},
            "obs_by_id": {},
            "det_pano_boxes": {},
            "model": None,
            "provider": object(),
            "backend": object(),
        }

    def payload(self):
        return {
            "range": {"name": "full", "k_start": K_START,
                      "k_end": K_END},
            "config": dataclasses.asdict(
                tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                      clean_iou=0.67)),
            "tracks": [],
            "rejected_births": [],
            "track_overlaps": [],
        }


class TrackingConfigTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ProducerFixture(Path(self.temporary.name))

    def test_all_scientific_values_come_from_build_config(self):
        resolved = rt.load_tracking_config(self.fixture.args())
        self.assertEqual(resolved["builder_cfg"].clean_iou, 0.67)
        self.assertEqual(resolved["ingest_params"].fov_deg, 91.0)
        self.assertEqual(resolved["course"], {
            "min_displacement_m": 2.5,
            "smooth_window_s": 6.0,
        })
        self.assertEqual(
            [ref.kind for ref in resolved["upstreams"]],
            [paths_lib.PINHOLE_IMAGES, paths_lib.FRAME_LANDMARKS])
        self.assertNotIn(
            "build_identity",
            artifact.load_manifest(self.fixture.pinholes).config)
        self.assertRegex(resolved["dataset_source_sha256"], r"^[0-9a-f]{64}$")

    def test_cli_range_must_equal_immutable_recipe(self):
        with self.assertRaisesRegex(
                rt.TrackingContractError, "k_start/--k_end disagree"):
            rt.load_tracking_config(self.fixture.args(k_end=2))

    def test_supplied_orchestration_digest_is_recomputed(self):
        with self.assertRaisesRegex(
                rt.TrackingContractError, "orchestration_config_digest"):
            rt.load_tracking_config(self.fixture.args(
                orchestration_config_digest="0" * 64))

    def test_checkpoint_content_must_match_recorded_digest(self):
        self.fixture.checkpoint.write_bytes(b"different weights")
        with self.assertRaisesRegex(
                rt.TrackingContractError, "checkpoint content digest"):
            rt.load_tracking_config(self.fixture.args())

    def test_dataset_mutation_cannot_mix_old_extraction_with_new_tracking(self):
        panorama = next(self.fixture.dataset_base.glob("panorama/*.jpg"))
        panorama.write_bytes(panorama.read_bytes() + b"changed")
        with self.assertRaisesRegex(
                rt.TrackingContractError, "dataset source bytes differ"):
            rt.load_tracking_config(self.fixture.args())

    def test_frame_landmarks_must_bind_the_exact_pinhole_artifact(self):
        manifest_path = self.fixture.frame_landmarks / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["upstreams"] = []
        artifact.atomic_write_json(manifest_path, document)
        with self.assertRaisesRegex(
                rt.TrackingContractError, "exact pinhole artifact"):
            rt.load_tracking_config(self.fixture.args())

    def test_frame_landmarks_from_another_build_is_accepted(self):
        """A build may plug in extraction another build paid for. Which
        generation an input belongs to is the orchestrator's question (see
        docs/farfield/decisions.md, 2026-09-02); the producer only requires
        that it binds the configured pinholes."""
        manifest_path = self.fixture.frame_landmarks / artifact.MANIFEST_NAME
        document = json.loads(manifest_path.read_text())
        document["config"]["build_identity"] = "0" * 64
        artifact.atomic_write_json(manifest_path, document)
        rt.load_tracking_config(self.fixture.args())

    def test_upstream_artifact_identity_is_exact(self):
        wrong = write_artifact(
            Path(self.temporary.name) / "wrong-frame-version",
            paths_lib.FRAME_LANDMARKS, "landmarks-v2")
        with self.assertRaisesRegex(
                artifact.ArtifactValidationError, "version mismatch"):
            rt.load_tracking_config(
                self.fixture.args(frame_landmarks_dir=wrong))

    def test_only_new_explicit_cli_is_exposed(self):
        options = rt.build_parser()._option_string_actions
        required = {
            "--frame_landmarks_dir", "--pinhole_dir", "--checkpoint",
            "--output_dir", "--dataset", "--dataset_base", "--k_start",
            "--k_end", "--build_config", "--orchestration_config_digest",
        }
        self.assertTrue(required.issubset(options))
        self.assertIn("--video", options)
        for retired in ("--run_name", "--runs_root", "--range",
                        "--skip_existing_ranges", "--force", "--fov_deg"):
            self.assertNotIn(retired, options)

    def test_automatic_viewer_is_derived_and_best_effort(self):
        tracks_ref = SimpleNamespace(path=str(self.fixture.output))
        args = self.fixture.args()
        expected = SimpleNamespace(path="viewer")
        with mock.patch.object(
                rt.keyframe_viewer, "publish_viewer",
                return_value=expected) as publish:
            self.assertIs(rt.publish_viewer_sidecar(args, tracks_ref), expected)
        viewer_args = publish.call_args.args[0]
        self.assertEqual(viewer_args.tracks_dir, self.fixture.output)
        self.assertEqual(viewer_args.dataset_base, self.fixture.dataset_base)
        self.assertEqual(
            viewer_args.frame_landmarks_dir, self.fixture.frame_landmarks)
        self.assertIsNone(viewer_args.output_dir)

        with mock.patch.object(
                rt.keyframe_viewer, "publish_viewer",
                side_effect=RuntimeError("render failed")):
            self.assertIsNone(rt.publish_viewer_sidecar(args, tracks_ref))


class TrackingPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = ProducerFixture(Path(self.temporary.name))

    def test_one_full_payload_and_manifest_publish_atomically(self):
        context = self.fixture.context()
        payload = self.fixture.payload()
        with mock.patch.object(rt.rr, "load_context", return_value=context) \
                as load_context, \
             mock.patch.object(rt.rr, "run_range",
                               return_value=(object(), payload)) as run_range, \
             mock.patch.object(rt.vc, "load_font", return_value=None):
            reference = rt.publish_tracking(
                self.fixture.args(), arguments=("run_tracking_test",))

        self.assertEqual(reference.kind, paths_lib.OBJECT_TRACKS)
        self.assertFalse(Path(str(self.fixture.output) + ".incomplete").exists())
        self.assertEqual(
            [path.name for path in self.fixture.output.glob("tracks_*.json")],
            ["tracks_full.json"])
        self.assertEqual(
            json.loads((self.fixture.output / "tracks_full.json").read_text()),
            payload)
        manifest = artifact.load_manifest(self.fixture.output)
        self.assertEqual(
            [ref.kind for ref in manifest.upstreams],
            [paths_lib.PINHOLE_IMAGES, paths_lib.FRAME_LANDMARKS])
        self.assertEqual(
            manifest.config["orchestration"]["config_digest"],
            self.fixture.digest)
        self.assertEqual(manifest.config["range"], payload["range"])
        self.assertIn("dataset_tracking_inputs",
                      manifest.config["source_digests"])
        self.assertEqual(set(manifest.declared_outputs),
                         {"index.html", "tracks_full.json"})

        _, load_args, load_kwargs = load_context.mock_calls[0]
        self.assertEqual(load_args[0], self.fixture.dataset_base)
        self.assertEqual(load_kwargs["course_min_displacement_m"], 2.5)
        self.assertEqual(load_kwargs["course_smooth_window_s"], 6.0)
        _, range_args, _ = run_range.mock_calls[0]
        self.assertEqual(range_args[:3], ("full", K_START, K_END))
        self.assertEqual(range_args[3].clean_iou, 0.67)

    def test_crash_leaves_only_incomplete_directory(self):
        with mock.patch.object(
                rt.rr, "load_context", return_value=self.fixture.context()), \
             mock.patch.object(rt.rr, "run_range",
                               side_effect=RuntimeError("GPU failed")), \
             mock.patch.object(rt.vc, "load_font", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                rt.publish_tracking(self.fixture.args())
        self.assertFalse(self.fixture.output.exists())
        incomplete = Path(str(self.fixture.output) + ".incomplete")
        self.assertTrue(incomplete.is_dir())
        self.assertFalse((incomplete / artifact.MANIFEST_NAME).exists())

    def test_existing_completed_artifact_is_never_reused(self):
        self.fixture.output.mkdir(parents=True)
        with mock.patch.object(
                rt.rr, "load_context", return_value=self.fixture.context()):
            with self.assertRaises(artifact.ArtifactExistsError):
                rt.publish_tracking(self.fixture.args())

    def test_track_without_records_cannot_publish(self):
        payload = self.fixture.payload()
        payload["tracks"] = [{"track_id": 7, "records": []}]
        with mock.patch.object(
                rt.rr, "load_context", return_value=self.fixture.context()), \
             mock.patch.object(
                 rt.rr, "run_range", return_value=(object(), payload)), \
             mock.patch.object(rt.vc, "load_font", return_value=None):
            with self.assertRaisesRegex(
                    rt.TrackingContractError, "track 7 has no records"):
                rt.publish_tracking(self.fixture.args())
        self.assertFalse(self.fixture.output.exists())


class RangeRunnerTerminalBirthTest(unittest.TestCase):
    def test_terminal_detection_is_not_serialized_without_an_interval(self):
        terminal = SimpleNamespace(obs_id="terminal")
        provider = SimpleNamespace(frames_between=lambda *_: [])
        result = SimpleNamespace(frames=[
            SimpleNamespace(frame_idx=0, time_s=0.0),
            SimpleNamespace(frame_idx=1, time_s=1.0),
        ])
        _, document = rr.run_range(
            "full", 0, 1,
            tb.TrackBuilderConfig(reference_pano_width=PANO_W),
            object(), provider, None, result, {1: [terminal]},
            {terminal.obs_id: [0.0, 0.0, 4.0, 4.0]}, PANO_W, PANO_H,
            Path("unused"), log=lambda *_: None)
        self.assertEqual(document["tracks"], [])

    def test_single_frame_range_does_not_seed_an_empty_track(self):
        only = SimpleNamespace(obs_id="only")
        result = SimpleNamespace(frames=[
            SimpleNamespace(frame_idx=0, time_s=0.0),
        ])
        _, document = rr.run_range(
            "full", 0, 0,
            tb.TrackBuilderConfig(reference_pano_width=PANO_W),
            object(), object(), None, result, {0: [only]},
            {only.obs_id: [0.0, 0.0, 4.0, 4.0]}, PANO_W, PANO_H,
            Path("unused"), log=lambda *_: None)
        self.assertEqual(document["tracks"], [])


class RangeRunnerCourseAbstentionTest(unittest.TestCase):
    def test_no_course_model_means_zero_relative_rotation(self):
        class Track:
            track_id = 1
            center_x = 8.0
            center_y = 8.0
            birth_obs_id = "obs"
            birth_keyframe = 0
            status = "alive"
            close_reason = ""
            end_keyframe = None
            last_keyframe = 0
            records = []

            @staticmethod
            def modal_label():
                return "test"

        class Builder:
            def __init__(self, *_args, **_kwargs):
                self.tracks = [Track()]
                self.rejected_births = []
                self.track_overlaps = []

            def seed_unassigned(self, *_args):
                pass

            def alive_tracks(self):
                return self.tracks

            def step(self, _keyframe, crops_fn, *_args, **_kwargs):
                crops, origins = crops_fn(self.tracks[0], 8)
                self.test_crops = crops
                self.test_origins = origins

        provider = SimpleNamespace(frames_between=lambda *_: [
            (0, 0.5, np.zeros((PANO_H, PANO_W, 3), dtype=np.uint8))])
        result = SimpleNamespace(frames=[
            SimpleNamespace(frame_idx=0, time_s=0.0),
            SimpleNamespace(frame_idx=1, time_s=1.0),
        ])
        with mock.patch.object(rr.tb, "TrackBuilder", Builder):
            builder, _ = rr.run_range(
                "full", 0, 1,
                tb.TrackBuilderConfig(reference_pano_width=PANO_W),
                object(), provider, None, result, {}, {}, PANO_W, PANO_H,
                Path("unused"), log=lambda *_: None)
        self.assertEqual(len(builder.test_crops), 1)
        self.assertEqual(len(builder.test_origins), 1)


if __name__ == "__main__":
    unittest.main()
