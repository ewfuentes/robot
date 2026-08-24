"""Strict source binding and complete semantic-audit publication tests."""

import argparse
import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact as artifact_lib,
    build_config,
    dataset,
    llm_lifecycle as llm,
    paths as paths_lib,
    pipeline,
    testing,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    prompts as request_adapter,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    audit_requests as ar,
    semantic_audit as sa,
    track_builder as tb,
)

DATASET = "tiny_harbor"
PANO_W = 256
CLEAN = {"iou": 0.8, "inter_over_mask": 0.9, "inter_over_box": 0.9}


def audit_stage_config():
    return {
        "artifacts": {
            "frame_landmarks_version": "v1",
            "object_tracks_version": "v1",
            "semantic_audits_version": "v1",
        },
        "audit": {
            "model": "test-model",
            "min_supports": 2,
            "thinking_level": "LOW",
            "max_support_chips": 2,
            "max_context_chips": 2,
            "max_description_samples": 5,
            "chip_height_px": 64,
        },
        "ingest": {
            "fov_deg": 90.0,
            "seam_gap_norm": 25.0,
            "seam_min_y_iou": 0.3,
        },
        "execution": {
            "llm_transport": "batch",
            "batch_gcs_prefix": "gs://test/farfield/audit",
            "approve_cost": False,
        },
        "cost": {"limit_usd": 10.0},
    }


def support(keyframe, obs_id):
    return {
        "obs_id": obs_id,
        "class": "recorded-at-run-time",
        "box_window": [40.0, 30.0, 80.0, 60.0],
        **CLEAN,
    }


def record(keyframe, obs_id):
    return {
        "keyframe": keyframe,
        "action": "continue_mask",
        "window_origin": [0.0, 0],
        "window_px": PANO_W,
        "mask_area": 100,
        "mask_bbox_window": [40, 30, 80, 60],
        "supports": [support(keyframe, obs_id)],
    }


def birth_record(keyframe):
    return {
        "keyframe": keyframe,
        "action": "birth",
        "window_origin": [0.0, 0],
        "window_px": PANO_W,
        "health": {"ok": True},
    }


def track(track_id, birth_kf, support_kfs, obs_by_frame):
    return {
        "track_id": track_id,
        "birth_obs_id": obs_by_frame[birth_kf],
        "birth_keyframe": birth_kf,
        "status": "closed",
        "close_reason": "starved",
        "end_keyframe": max(support_kfs),
        "last_keyframe": max(support_kfs),
        "modal_label": "man_made=tower 'Graves Light'",
        "n_supported_keyframes": len(support_kfs),
        "records": [birth_record(birth_kf)] + [
            record(keyframe, obs_by_frame[keyframe])
            for keyframe in support_kfs
        ],
    }


def track_payload(tracks):
    cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    return {
        "range": {"name": "full", "k_start": 0, "k_end": 4},
        "config": dataclasses.asdict(cfg),
        "tracks": tracks,
        "rejected_births": [],
        "track_overlaps": [],
    }


def write_tracks_artifact(path: Path, payloads: dict[str, dict]):
    with artifact_lib.ArtifactDirectoryBuilder(
            path,
            kind=paths_lib.OBJECT_TRACKS,
            dataset=DATASET,
            version="v1",
            generator="audit_requests_test",
            git_commit="test",
            declared_outputs=sorted(payloads)) as builder:
        for name, payload in payloads.items():
            artifact_lib.atomic_write_json(builder.output_path(name), payload)
    return path


def make_inputs(root: Path):
    base = testing.make_dataset(
        root / "datasets" / DATASET,
        n_frames=5,
        pano_size=(PANO_W, 128))
    stems = sorted(path.stem for path in (base / "panorama").glob("*.jpg"))
    frame_landmarks = (
        root / "artifacts" / paths_lib.FRAME_LANDMARKS / DATASET / "v1")
    testing.make_predictions(
        frame_landmarks,
        {stem: [testing.landmark(
            "Graves Light", [(0, 400, 300, 600, 500)])]
         for stem in stems},
        dataset_name=DATASET)
    ingest = dataset.run_ingest(
        base, frame_landmarks,
        dataset.IngestParams(
            fov_deg=90.0, seam_gap_norm=25.0, seam_min_y_iou=0.3))
    obs_by_frame = {
        observation.frame_idx: observation.obs_id
        for observation in ingest.observations
    }
    tracks = [
        track(1, 0, [1, 2, 3], obs_by_frame),
        track(2, 2, [3], obs_by_frame),
        track(5, 1, [2, 3], obs_by_frame),
    ]
    tracks_dir = (
        root / "artifacts" / paths_lib.OBJECT_TRACKS / DATASET / "v1")
    write_tracks_artifact(
        tracks_dir, {"tracks_full.json": track_payload(tracks)})
    return base, frame_landmarks, tracks_dir, tracks


def request_args(request_dir, document, selected, orchestration):
    ingest_params = dataset.IngestParams(
        fov_deg=selected["ingest.fov_deg"],
        seam_gap_norm=selected["ingest.seam_gap_norm"],
        seam_min_y_iou=selected["ingest.seam_min_y_iou"])
    return argparse.Namespace(
        dataset=DATASET,
        dataset_base=Path(document["inputs"]["dataset_base"]),
        output_dir=request_dir,
        output_version="v1.requests",
        model=selected["audit.model"],
        min_supports=selected["audit.min_supports"],
        thinking_level=selected["audit.thinking_level"],
        max_support_chips=selected["audit.max_support_chips"],
        max_context_chips=selected["audit.max_context_chips"],
        max_description_samples=selected["audit.max_description_samples"],
        chip_height_px=selected["audit.chip_height_px"],
        fov_deg=selected["ingest.fov_deg"],
        seam_gap_norm=selected["ingest.seam_gap_norm"],
        seam_min_y_iou=selected["ingest.seam_min_y_iou"],
        ingest_params=ingest_params,
        build_identity=document["build_identity"],
        orchestration=orchestration,
        resolved_stage_config=selected)


def audit_payload():
    return {
        "landmark_kind": "fixed_structure",
        "single_object": True,
        "valid_segments": [{"start_t": 0, "end_t": 3}],
        "verdict": "keep",
        "drop_reason": "none",
        "primary_object": {
            "tags": [{"tag": "man_made=tower", "weight": 1.0}],
            "name_candidates": [{
                "name": "Graves Light",
                "weight": 1.0,
                "basis": "reported_by_detections",
            }],
            "name_aliases": [],
            "description": "A fixed tower.",
            "distinctive_features": ["red top"],
            "extent": "point_like",
        },
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": "high",
        "unresolved": "",
    }


def success(request_set, key, attempt_id, payload=None):
    response = {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps(payload or audit_payload())
        }]}}]
    }
    return llm.Attempt(
        request_set_fingerprint=request_set.fingerprint,
        key=key,
        attempt_id=attempt_id,
        response=response,
        error=None,
        metadata={"transport": "test"})


class AuditRequestsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        (cls.dataset_base, cls.frame_landmarks, cls.tracks_dir,
         cls.tracks) = make_inputs(cls.root)
        cls.config = audit_stage_config()
        cls.build_config_path = build_config.create(
            cls.root / "build",
            dataset=DATASET,
            config=cls.config,
            generator="audit_requests_test",
            inputs={"dataset_base": cls.dataset_base})
        cls.document = build_config.load(cls.build_config_path.parent)
        cls.orchestration = pipeline.stage_contract("audit", cls.config)
        cls.selected = {
            key: build_config.value(cls.document, key)
            for key in ar.AUDIT_CONFIG_KEYS
        }
        cls.work_dir = ar.audit_work_dir(cls.root / "audits" / "v1")
        cls.request_dir = cls.work_dir / "requests"
        cls.paths = argparse.Namespace(
            dataset=DATASET, dataset_base=cls.dataset_base)
        cls.source = ar.load_source_tracks(cls.tracks_dir, DATASET)
        cls.ingest_result = dataset.run_ingest(
            cls.dataset_base, cls.frame_landmarks,
            dataset.IngestParams(
                fov_deg=cls.selected["ingest.fov_deg"],
                seam_gap_norm=cls.selected["ingest.seam_gap_norm"],
                seam_min_y_iou=cls.selected["ingest.seam_min_y_iou"]))
        cls.args = request_args(
            cls.request_dir, cls.document, cls.selected, cls.orchestration)
        ar.build_request_artifact(
            cls.args, cls.paths, *cls.source, cls.ingest_result)
        cls.request_ref = artifact_lib.open_artifact(
            cls.request_dir,
            expected_kind=ar.REQUEST_ARTIFACT_KIND,
            expected_dataset=DATASET)
        cls.request_set = llm.load_request_set(
            cls.request_dir / llm.REQUEST_SET_NAME)
        cls.requests = {
            record["key"]: record["request"]
            for record in map(json.loads, (
                cls.request_dir / llm.REQUESTS_NAME).read_text().splitlines())
        }
        cls.meta = json.loads(
            (cls.request_dir / ar.AUDIT_META_NAME).read_text())
        cls.settings = json.loads(
            (cls.request_dir / ar.SETTINGS_NAME).read_text())

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_request_set_covers_every_eligible_track(self):
        self.assertEqual(set(self.requests), {"T1", "T5"})
        self.assertEqual(set(self.meta["requests"]), {"T1", "T5"})
        self.assertEqual(
            [unit.key for unit in self.request_set.units], ["T1", "T5"])

    def test_request_shape_and_media_are_frozen(self):
        request = self.requests["T1"]
        self.assertEqual(
            request["systemInstruction"]["parts"][0]["text"],
            sa.SYSTEM_PROMPT)
        generation = request["generationConfig"]
        self.assertEqual(
            generation["thinkingConfig"]["thinkingLevel"], "LOW")
        self.assertEqual(generation["responseMimeType"], "application/json")
        images = [part for part in request["contents"][0]["parts"]
                  if "inline_data" in part]
        self.assertEqual(
            len(images), len(self.meta["requests"]["T1"]["chips"]))
        self.assertGreaterEqual(len(images), 1)
        self.assertEqual(self.request_set.stage, "semantic_audit")
        self.assertEqual(self.request_set.model, "test-model")
        online = request_adapter.online_request_from_batch("T1", request)
        self.assertEqual(
            online["config"]["system_instruction"], sa.SYSTEM_PROMPT)
        self.assertEqual(
            online["config"]["thinking_config"], {
                "thinking_level": "LOW",
            })
        self.assertNotIn("media_resolution", online["config"])

    def test_meta_is_v2_and_binds_file_artifact_and_each_track(self):
        self.assertEqual(self.meta["schema"], ar.AUDIT_META_SCHEMA)
        source_ref = artifact_lib.open_artifact(self.tracks_dir)
        self.assertEqual(self.meta["source_tracks"], {
            "artifact_id": ar.source_artifact_id(source_ref),
            "file": "tracks_full.json",
            "sha256": artifact_lib.sha256_file(
                self.tracks_dir / "tracks_full.json"),
        })
        tracks_by_id = {track["track_id"]: track for track in self.tracks}
        for key, request_meta in self.meta["requests"].items():
            self.assertEqual(
                request_meta["source_track_sha256"],
                artifact_lib.sha256_json(
                    tracks_by_id[request_meta["track_id"]]))
            unit = next(unit for unit in self.request_set.units
                        if unit.key == key)
            self.assertEqual(request_meta, unit.to_dict()["metadata"])

    def test_settings_and_manifest_record_whole_recipe(self):
        settings = self.settings
        self.assertEqual(settings["model"], "test-model")
        self.assertEqual(settings["thinking_level"], "LOW")
        self.assertEqual(settings["min_supports"], 2)
        self.assertEqual(
            settings["system_prompt_sha256"],
            hashlib.sha256(sa.SYSTEM_PROMPT.encode()).hexdigest())
        self.assertEqual(
            settings["classifier"]["reference_pano_width"], PANO_W)
        self.assertEqual(settings["source_tracks_file"], "tracks_full.json")
        self.assertEqual(
            (settings["n_tracks_total"], settings["n_eligible"],
             settings["n_requests"]),
            (3, 2, 2))
        manifest = artifact_lib.load_manifest(self.request_dir)
        self.assertEqual(manifest.config["phase"], "requests")
        self.assertEqual(len(manifest.upstreams), 2)

    def test_preview_uses_artifact_relative_chips(self):
        page = (self.request_dir / "preview" / "index.html").read_text()
        self.assertIn("id='T1'", page)
        self.assertIn("id='T5'", page)
        self.assertIn("../chips/T1_t", page)
        for chip in self.meta["requests"]["T1"]["chips"]:
            self.assertFalse(Path(chip).is_absolute())
            self.assertTrue((self.request_dir / chip).is_file())

    def test_published_request_set_is_not_reused_or_overwritten(self):
        with self.assertRaises(artifact_lib.ArtifactExistsError):
            ar.build_request_artifact(
                self.args, self.paths, *self.source, self.ingest_result)

    def test_config_loader_matches_the_pipeline_stage_contract(self):
        load_args = argparse.Namespace(
            build_config=self.build_config_path,
            dataset=DATASET,
            dataset_base=self.dataset_base,
            orchestration_config_digest=self.orchestration["config_digest"])
        document, selected, orchestration = ar.load_audit_config(load_args)
        self.assertEqual(document["build_identity"],
                         self.document["build_identity"])
        self.assertEqual(selected, self.selected)
        self.assertEqual(orchestration, self.orchestration)
        self.assertNotIn("artifacts.object_tracks_version", selected)
        self.assertNotIn("artifacts.frame_landmarks_version", selected)
        execution_args = argparse.Namespace(
            parallel=8,
            poll_interval=120,
            online=False,
            gcs_prefix="gs://test/farfield/audit",
            approve_cost=False,
            cost_limit=10.0,
            model=None)
        ar.validate_execution_args(execution_args, selected)
        self.assertEqual(execution_args.model, "test-model")
        execution_args.parallel = 0
        with self.assertRaisesRegex(ValueError, "parallel must be positive"):
            ar.validate_execution_args(execution_args, selected)

    def test_request_snapshot_resume_uses_scientific_fingerprint(self):
        old_argv = sys.argv
        sys.argv = ["audit_requests", "--aggregate_only"]
        try:
            request_dir = ar._prepare_request_artifact(  # noqa: SLF001
                self.args, self.selected, self.document, self.orchestration,
                self.source, self.ingest_result, self.work_dir,
                "v1.requests")
        finally:
            sys.argv = old_argv
        self.assertEqual(request_dir, self.request_dir)
        self.assertEqual(
            llm.load_request_set(
                request_dir / llm.REQUEST_SET_NAME).fingerprint,
            self.request_set.fingerprint)

    def test_more_than_one_tracks_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracks_dir = Path(temporary) / "tracks"
            payload = track_payload([])
            write_tracks_artifact(tracks_dir, {
                "tracks_a.json": payload,
                "tracks_b.json": payload,
            })
            with self.assertRaisesRegex(ValueError, "exactly one"):
                ar.load_source_tracks(tracks_dir, DATASET)

    def test_incomplete_or_legacy_results_cannot_publish(self):
        only_one = (success(self.request_set, "T1", "one"),)
        with self.assertRaisesRegex(
                llm.IncompleteCoverageError, "T5"):
            ar.compile_audit_results(self.request_set, only_one)

        legacy = audit_payload()
        legacy["primary_object"]["name"] = "Graves Light"
        del legacy["primary_object"]["name_candidates"]
        attempts = (
            success(self.request_set, "T1", "legacy", legacy),
            success(self.request_set, "T5", "valid"),
        )
        with self.assertRaisesRegex(
                llm.IncompleteCoverageError, "T1"):
            ar.compile_audit_results(self.request_set, attempts)

    def test_complete_results_publish_one_canonical_record_per_request(self):
        attempts_dir = self.root / llm.ATTEMPTS_DIR_NAME
        for key in ("T1", "T5"):
            llm.publish_attempt(
                attempts_dir,
                success(self.request_set, key, f"success-{key}"))
        destination = self.root / "audits" / "v1"
        result_ref = ar.publish_audit_results(
            destination,
            request_dir=self.request_dir,
            tracks_dir=self.tracks_dir,
            attempts_dir=attempts_dir,
            dataset_name=DATASET,
            version="v1",
            arguments=("test",))
        self.assertEqual(result_ref, artifact_lib.open_artifact(
            destination,
            expected_kind=paths_lib.SEMANTIC_AUDITS,
            expected_dataset=DATASET,
            expected_version="v1"))
        manifest = artifact_lib.load_manifest(destination)
        self.assertEqual(manifest.config["coverage"], "complete")
        self.assertEqual(manifest.config["n_successful"], 2)
        for chip in self.meta["requests"]["T1"]["chips"]:
            self.assertTrue((destination / chip).is_file())
        records = [json.loads(line) for line in
                   (destination / "results.jsonl").read_text().splitlines()]
        self.assertEqual([record["key"] for record in records], ["T1", "T5"])
        for record in records:
            key, parsed, error = sa.parse_result_line(record)
            self.assertIn(key, {"T1", "T5"})
            self.assertIsNone(error)
            self.assertEqual(parsed["verdict"], "keep")


if __name__ == "__main__":
    unittest.main()
