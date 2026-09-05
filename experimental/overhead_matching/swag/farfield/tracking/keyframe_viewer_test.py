import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experimental.overhead_matching.swag.farfield import (
    artifact as artifact_lib,
    dataset,
    paths as paths_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer as kv,
    track_builder as tb,
)

PANO_W = 7680


def make_artifact(range_name, tracks, config=None, rejected=None):
    """A tracks_*.json-shaped dict, in the tracklets_test fixture style."""
    cfg = config or tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    return {
        "range": {"name": range_name, "k_start": 0, "k_end": 20},
        "config": dataclasses.asdict(cfg),
        "tracks": tracks,
        "rejected_births": rejected or [],
        "track_overlaps": [],
    }


def make_track(track_id, birth_obs_id, records):
    return {
        "track_id": track_id,
        "birth_obs_id": birth_obs_id,
        "birth_keyframe": records[0]["keyframe"] if records else 0,
        "status": "closed", "close_reason": "starved",
        "end_keyframe": None, "last_keyframe": None,
        "modal_label": "man_made=tower", "n_supported_keyframes": 0,
        "records": records,
    }


def record(keyframe, supports=(), mask_bbox=None, origin=(100.0, 200)):
    return {
        "keyframe": keyframe, "action": "continue_mask",
        "window_origin": list(origin), "window_px": 1024,
        "mask_area": 100, "mask_bbox_window": mask_bbox,
        "supports": list(supports),
    }


def support(obs_id, iou, iom, iob):
    return {"obs_id": obs_id, "class": "recorded-at-run-time",
            "box_window": [0, 0, 10, 10], "iou": iou,
            "inter_over_mask": iom, "inter_over_box": iob}


class LoadTrackArtifactsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tracks_dir = self.root / "tracks"
        self.addCleanup(self._tmp.cleanup)

    def publish(self, payloads, *, range_record=None, kind=None,
                upstreams=(), config_extra=None):
        range_record = range_record or {
            "name": "full", "k_start": 0, "k_end": 20}
        config = {
            "schema": "farfield_object_tracks/v1",
            "coverage": "complete",
            "range": range_record,
        }
        config.update(config_extra or {})
        with artifact_lib.ArtifactDirectoryBuilder(
                self.tracks_dir,
                kind=kind or paths_lib.OBJECT_TRACKS,
                dataset="viewer_test",
                version="v1",
                generator="keyframe_viewer_test",
                git_commit="test",
                upstreams=upstreams,
                config=config,
                declared_outputs=tuple(payloads)) as builder:
            for filename, document in payloads.items():
                artifact_lib.atomic_write_json(
                    builder.output_path(filename), document)

    def test_raw_directory_is_not_accepted_as_tracks(self):
        self.tracks_dir.mkdir()
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.tracks_dir)
        self.assertIn("completed object_tracks", str(ctx.exception))

    def test_loads_exact_single_full_payload(self):
        self.publish({"tracks_full.json": make_artifact("full", [])})
        self.assertEqual(list(kv.load_track_artifacts(self.tracks_dir)),
                         ["full"])

    def test_missing_payload_is_rejected(self):
        self.publish({})
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.tracks_dir)
        self.assertIn("exactly one tracks_full.json", str(ctx.exception))

    def test_legacy_multi_range_payloads_are_rejected(self):
        self.publish({
            "tracks_a.json": make_artifact("a", []),
            "tracks_b.json": make_artifact("b", []),
        })
        with self.assertRaisesRegex(SystemExit, "exactly one tracks_full"):
            kv.load_track_artifacts(self.tracks_dir)

    def test_payload_range_must_equal_manifest_range(self):
        self.publish({"tracks_full.json": make_artifact("other", [])})
        with self.assertRaises(SystemExit) as ctx:
            kv.load_track_artifacts(self.tracks_dir)
        self.assertIn("single full range", str(ctx.exception))

    def test_wrong_artifact_kind_is_rejected(self):
        self.publish(
            {"tracks_full.json": make_artifact("full", [])}, kind="other")
        with self.assertRaisesRegex(SystemExit, "kind mismatch"):
            kv.load_track_artifacts(self.tracks_dir)

    def test_viewer_inputs_come_from_bound_artifacts_and_manifest_config(self):
        dataset_base = testing.make_dataset(
            self.root / "viewer_test", n_frames=1)
        stem = next((dataset_base / "panorama").glob("*.jpg")).stem
        frames_dir = testing.make_predictions(
            self.root / "frames", {stem: []}, dataset_name="viewer_test")
        frame_ref = artifact_lib.open_artifact(frames_dir)
        source_digest = artifact_lib.sha256_json(
            paths_lib.dataset_source_digests(dataset_base))
        self.publish(
            {"tracks_full.json": make_artifact("full", [])},
            upstreams=(frame_ref,),
            config_extra={
                "source_digests": {
                    "dataset_tracking_inputs": source_digest,
                },
                "resolved": {
                    "ingest": {
                        "fov_deg": 91.0,
                        "seam_gap_norm": 24.0,
                        "seam_min_y_iou": 0.31,
                    },
                },
            })
        inputs = kv.load_viewer_inputs(
            self.tracks_dir, dataset_base, frames_dir)
        self.assertEqual(
            inputs.ingest_params,
            dataset.IngestParams(
                fov_deg=91.0, seam_gap_norm=24.0,
                seam_min_y_iou=0.31))
        self.assertEqual(inputs.frame_landmarks_dir, frames_dir.resolve())

        other_frames = testing.make_predictions(
            self.root / "other_frames",
            {stem: [testing.landmark("Different", [(0, 1, 1, 2, 2)])]},
            dataset_name="viewer_test")
        with self.assertRaisesRegex(
                SystemExit, "not the exact artifact bound by tracks"):
            kv.load_viewer_inputs(
                self.tracks_dir, dataset_base, other_frames)

        panorama = next((dataset_base / "panorama").glob("*.jpg"))
        panorama.write_bytes(panorama.read_bytes() + b"changed")
        with self.assertRaisesRegex(SystemExit, "bytes do not match"):
            kv.load_viewer_inputs(self.tracks_dir, dataset_base, frames_dir)


class CliContractTest(unittest.TestCase):
    def test_cli_has_explicit_typed_inputs_and_no_legacy_run_or_ingest_flags(self):
        args = kv.build_parser().parse_args([
            "--tracks_dir", "/artifacts/tracks",
            "--dataset_base", "/datasets/ds",
            "--frame_landmarks_dir", "/artifacts/frames",
            "--output_dir", "/derived/keyframes",
        ])
        self.assertFalse(hasattr(args, "run_dir"))
        self.assertFalse(hasattr(args, "fov_deg"))
        self.assertEqual(args.tracks_dir, Path("/artifacts/tracks"))

    def test_output_defaults_to_a_commit_versioned_frame_sidecar(self):
        inputs = SimpleNamespace(
            frame_ref=SimpleNamespace(version="frames-v1"),
            tracks_ref=SimpleNamespace(version="tracks-v2"),
            frame_landmarks_dir=Path("/artifacts/frame_landmarks/ds/frames-v1"),
        )
        self.assertEqual(
            kv.default_output_dir(inputs, "0123456789abcdef"),
            Path("/artifacts/frame_landmarks/ds/"
                 "frames-v1--tracks-tracks-v2--viewer-0123456789ab"))

    def test_output_inside_completed_tracks_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            tracks_dir = Path(temp) / "tracks"
            tracks_dir.mkdir()
            with self.assertRaisesRegex(SystemExit, "inside it"):
                kv.prepare_output_directory(
                    tracks_dir, tracks_dir / "keyframes")


class RecordedConfigTest(unittest.TestCase):
    def test_reconstructs_the_recorded_dataclass(self):
        cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                    clean_iou=0.30)
        artifact = make_artifact("leg", [], config=cfg)
        self.assertEqual(kv.recorded_config(artifact), cfg)

    def test_missing_config_is_an_error_not_a_default(self):
        artifact = make_artifact("leg", [])
        del artifact["config"]
        with self.assertRaises(SystemExit):
            kv.recorded_config(artifact)


class SidecarPublicationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset_base = testing.make_dataset(
            self.root / "datasets" / "viewer_test", n_frames=1,
            pano_size=(64, 32))
        stem = next((self.dataset_base / "panorama").glob("*.jpg")).stem
        self.frames_dir = testing.make_predictions(
            self.root / "artifacts" / "frame_landmarks" / "viewer_test"
            / "frames-v1",
            {stem: []}, dataset_name="viewer_test", version="frames-v1")
        frame_ref = artifact_lib.open_artifact(self.frames_dir)
        self.tracks_dir = (
            self.root / "artifacts" / "object_tracks" / "viewer_test"
            / "tracks-v1")
        source_digest = artifact_lib.sha256_json(
            paths_lib.dataset_source_digests(self.dataset_base))
        payload = make_artifact("full", [make_track(0, "seed", [record(0)])])
        outputs = (
            "index.html", "track_full_T0.html", "tracks_full.json",
            "thumbs/full_T0.jpg", "videos/full_T0.mp4")
        with artifact_lib.ArtifactDirectoryBuilder(
                self.tracks_dir,
                kind=paths_lib.OBJECT_TRACKS,
                dataset="viewer_test",
                version="tracks-v1",
                generator="keyframe_viewer_test",
                git_commit="test",
                upstreams=(frame_ref,),
                config={
                    "schema": "farfield_object_tracks/v1",
                    "coverage": "complete",
                    "range": payload["range"],
                    "source_digests": {
                        "dataset_tracking_inputs": source_digest,
                    },
                    "resolved": {"ingest": {
                        "fov_deg": 91.0,
                        "seam_gap_norm": 24.0,
                        "seam_min_y_iou": 0.31,
                    }},
                },
                declared_outputs=outputs) as builder:
            builder.output_path("index.html").write_text(
                "<html><body><img src='thumbs/full_T0.jpg'></body></html>")
            builder.output_path("track_full_T0.html").write_text(
                "<html><body><table><tr><td class='kf'>f0000</td></tr>"
                "</table><video src='videos/full_T0.mp4'></video></body></html>")
            artifact_lib.atomic_write_json(
                builder.output_path("tracks_full.json"), payload)
            builder.output_path("thumbs/full_T0.jpg").parent.mkdir()
            builder.output_path("thumbs/full_T0.jpg").write_bytes(b"thumb")
            builder.output_path("videos/full_T0.mp4").parent.mkdir()
            builder.output_path("videos/full_T0.mp4").write_bytes(b"video")

    def test_sidecar_is_typed_bidirectional_and_leaves_inputs_unchanged(self):
        frame_before = artifact_lib.open_artifact(self.frames_dir)
        track_before = artifact_lib.open_artifact(self.tracks_dir)
        output = (self.frames_dir.parent
                  / "frames-v1--tracks-tracks-v1--viewer-test")
        args = SimpleNamespace(
            tracks_dir=self.tracks_dir,
            dataset_base=self.dataset_base,
            frame_landmarks_dir=self.frames_dir,
            output_dir=output,
            pano_width=64,
            kf_start=None,
            kf_end=None,
            image_workers=1,
        )
        reference = kv.publish_viewer(args, arguments=("test",))
        self.assertEqual(reference.kind, kv.VIEWER_KIND)
        manifest = artifact_lib.load_manifest(output)
        self.assertEqual(manifest.upstreams, (frame_before, track_before))
        self.assertEqual(manifest.config["schema"], kv.VIEWER_SCHEMA)
        self.assertIn("f0000.html", manifest.declared_outputs)
        self.assertIn("tracks/track_full_T0.html",
                      manifest.declared_outputs)
        frame_page = (output / "f0000.html").read_text()
        self.assertIn("tracks/index.html", frame_page)
        track_page = (output / "tracks" / "track_full_T0.html").read_text()
        self.assertIn("href='../f0000.html'", track_page)
        self.assertNotIn("immutable track artifact", track_page)
        self.assertIn("/videos/full_T0.mp4", track_page)
        self.assertEqual(artifact_lib.open_artifact(self.frames_dir),
                         frame_before)
        self.assertEqual(artifact_lib.open_artifact(self.tracks_dir),
                         track_before)


class TrackAssociationsTest(unittest.TestCase):
    def test_supports_reclassified_under_the_recorded_config(self):
        # Recorded run used clean_iou=0.30: iou 0.35/iom 0.9 is a clean
        # continuation THERE, while today's default (0.45) would demote it to
        # weak. The viewer must show the run as it was built.
        recorded = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                         clean_iou=0.30)
        track = make_track(0, "seed_obs", [
            record(5, supports=[support("f0005__lm1__box0",
                                        iou=0.35, iom=0.9, iob=0.5)]),
        ])
        artifacts = {"leg": make_artifact("leg", [track], config=recorded)}
        by_obs, _, _, _ = kv.track_associations(artifacts)
        self.assertEqual(by_obs[(5, "f0005__lm1__box0")],
                         [("leg_T0", "continue_clean")])
        # Sanity: a fresh default config would have said "weak" -- the value
        # the old viewer (which built TrackBuilderConfig()) would have shown.
        self.assertEqual(
            tb.classify_support(
                {"iou": 0.35, "inter_over_mask": 0.9, "inter_over_box": 0.5},
                tb.TrackBuilderConfig(reference_pano_width=PANO_W)),
            "weak")

    def test_each_range_classifies_under_its_own_config(self):
        loose = tb.TrackBuilderConfig(reference_pano_width=PANO_W,
                                      clean_iou=0.30)
        strict = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
        sup = support("obs", iou=0.35, iom=0.9, iob=0.5)
        artifacts = {
            "loose": make_artifact(
                "loose", [make_track(0, "a", [record(1, supports=[sup])])],
                config=loose),
            "strict": make_artifact(
                "strict", [make_track(0, "b", [record(1, supports=[sup])])],
                config=strict),
        }
        by_obs, _, _, _ = kv.track_associations(artifacts)
        self.assertEqual(sorted(by_obs[(1, "obs")]),
                         [("loose_T0", "continue_clean"),
                          ("strict_T0", "weak")])

    def test_track_keys_carry_the_range_name(self):
        # Two ranges may reuse track_id 0; keys must not collide.
        artifacts = {
            "legA": make_artifact(
                "legA", [make_track(0, "obsA",
                                    [record(2, mask_bbox=[10, 20, 30, 40])])]),
            "legB": make_artifact(
                "legB", [make_track(0, "obsB",
                                    [record(2, mask_bbox=[1, 2, 3, 4])])]),
        }
        _, masks, seeded, _ = kv.track_associations(artifacts)
        self.assertEqual({key for key, _, _ in masks[2]},
                         {"legA_T0", "legB_T0"})
        self.assertEqual(seeded["obsA"], ["legA_T0"])
        self.assertEqual(seeded["obsB"], ["legB_T0"])

    def test_mask_boxes_shift_by_window_origin(self):
        track = make_track(0, "obs", [
            record(3, mask_bbox=[10, 20, 30, 40], origin=(100.0, 200)),
        ])
        _, masks, _, _ = kv.track_associations(
            {"leg": make_artifact("leg", [track])})
        self.assertEqual(masks[3], [("leg_T0", "continue_mask",
                                     (110.0, 220, 130.0, 240))])

    def test_rejected_births_surface_their_health(self):
        artifact = make_artifact(
            "leg", [], rejected=[{"obs_id": "bad_obs", "keyframe": 0,
                                  "health": {"ok": False,
                                             "reason": "fragmented"}}])
        _, _, _, rejected = kv.track_associations({"leg": artifact})
        self.assertEqual(rejected["bad_obs"]["reason"], "fragmented")


if __name__ == "__main__":
    unittest.main()
