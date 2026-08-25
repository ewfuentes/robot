"""Strict source binding and complete semantic-audit publication tests."""

import argparse
import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import (
    artifact as artifact_lib,
    build_config,
    dataset,
    llm_lifecycle as llm,
    paths as paths_lib,
    pipeline,
    testing,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.extraction import (
    prompts as request_adapter,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    audit_requests as ar,
    semantic_audit as sa,
    track_builder as tb,
    tracklets,
)

DATASET = "tiny_harbor"
PANO_W = 256
CLEAN = {"iou": 0.8, "inter_over_mask": 0.9, "inter_over_box": 0.9}
LEGACY_REQUEST_SET_FINGERPRINT = (
    "c2aa323319dd6915ab971b7de4c13a5126f3d09b571415a2c3f6b05772b1cfc9")


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
        resolved_stage_config=selected,
        stage_reuse={"schema": "farfield_stage_reuse_bridge/v1"})


def audit_payload(decision="keep_single", segments=None):
    if segments is None:
        segments = [{"start_t": 0, "end_t": 3}]
    return {
        "landmark_kind": "fixed_structure",
        "decision": decision,
        "valid_segments": segments,
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


def legacy_correlated_payload(
        verdict, single_object, drop_reason, segments=None):
    payload = audit_payload(segments=segments)
    del payload["decision"]
    payload.update({
        "verdict": verdict,
        "single_object": single_object,
        "drop_reason": drop_reason,
    })
    return payload


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


def provider_response(payload):
    return {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps(payload),
        }]}}],
    }


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
        self.assertEqual(
            generation["responseSchema"], sa.get_provider_audit_schema())
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
        self.assertEqual(
            online["config"]["response_schema"],
            sa.get_provider_audit_schema())
        self.assertNotIn("media_resolution", online["config"])

    def test_observed_invalid_cross_field_results_are_rejected(self):
        observed = (
            ("boston_harbor_leg1/T1", True, "none", [(0, 19)], ""),
            ("boston_harbor_leg1/T40", False, "identity_broken",
             [(0, 0)], ""),
            ("boston_harbor_leg1/T123", True, "none", [(0, 14)], ""),
            ("boston_harbor_leg1/T147", True, "none", [(0, 15)], ""),
            ("boston_harbor_leg1/T172", True, "none",
             [(0, 1), (3, 3)],
             "mask drifted into sky at t2 then re-anchored at t3"),
            ("boston_harbor_leg1/T214", True, "none", [(0, 5)], ""),
            ("mount_washington_20260815_leg1/T77", True, "none",
             [(0, 4)],
             "mask drifts after t4 but remains on the same mountain massif"),
        )
        for (key, single_object, drop_reason, raw_segments,
             unresolved) in observed:
            with self.subTest(key=key):
                payload = legacy_correlated_payload(
                    "keep_partial", single_object, drop_reason,
                    [
                        {"start_t": start, "end_t": end}
                        for start, end in raw_segments
                    ])
                payload["unresolved"] = unresolved
                with self.assertRaisesRegex(
                        ValueError, "invalid ProviderTrackAudit response"):
                    ar._validate_audit_response(  # noqa: SLF001
                        key, provider_response(payload))

    def test_legacy_correlated_provider_shape_is_always_rejected(self):
        reasons = (
            "none", "dynamic_object", "not_a_physical_landmark",
            "identity_broken", "insufficient_evidence")
        segment_options = ([], [{"start_t": 0, "end_t": 3}])
        for verdict in ("keep", "keep_partial", "drop"):
            for single_object in (False, True):
                for drop_reason in reasons:
                    for segments in segment_options:
                        payload = legacy_correlated_payload(
                            verdict, single_object, drop_reason, segments)
                        response = provider_response(payload)
                        with self.subTest(
                                verdict=verdict,
                                single_object=single_object,
                                drop_reason=drop_reason,
                                has_segments=bool(segments)):
                            with self.assertRaises(ValueError):
                                ar._validate_audit_response(  # noqa: SLF001
                                    "T-contract", response)

    def test_provider_shape_rejects_reintroduced_correlated_fields(self):
        for field, value in (
                ("verdict", "keep"),
                ("single_object", True),
                ("drop_reason", "none")):
            with self.subTest(field=field):
                payload = audit_payload()
                payload[field] = value
                with self.assertRaisesRegex(
                        ValueError, "fields must be exactly"):
                    ar._validate_audit_response(  # noqa: SLF001
                        "T-extra", provider_response(payload))

    def test_each_provider_decision_maps_to_one_canonical_variant(self):
        for decision, expected in sa.PROVIDER_DECISION_TO_CANONICAL.items():
            is_accepted = decision.startswith("keep_")
            segment_options = (
                ([{"start_t": 0, "end_t": 3}],)
                if is_accepted else
                ([], [{"start_t": 0, "end_t": 1}]))
            for segments in segment_options:
                with self.subTest(decision=decision, segments=segments):
                    payload = audit_payload(decision, segments)
                    canonical = ar._validate_audit_response(  # noqa: SLF001
                        "T-valid", provider_response(payload))
                    self.assertEqual(
                        (canonical["verdict"], canonical["single_object"],
                         canonical["drop_reason"]), expected)
                    self.assertEqual(canonical["valid_segments"], segments)
                    self.assertNotIn("decision", canonical)

    def test_accepted_provider_decisions_require_nonempty_segments(self):
        for decision in ("keep_single", "keep_partial_identity_switch"):
            with self.subTest(decision=decision):
                with self.assertRaisesRegex(
                        ValueError, "requires at least one valid segment"):
                    ar._validate_audit_response(  # noqa: SLF001
                        "T-empty",
                        provider_response(audit_payload(decision, [])))

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
        self.assertEqual(
            manifest.config["stage_reuse"],
            {"schema": "farfield_stage_reuse_bridge/v1"})
        self.assertEqual(len(manifest.upstreams), 2)
        self.assertNotEqual(
            self.request_set.fingerprint, LEGACY_REQUEST_SET_FINGERPRINT)

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

    def test_direct_prefix_inputs_cross_the_shared_authorization_boundary(self):
        args = argparse.Namespace(
            build_config=self.build_config_path,
            dataset=DATASET,
            tracks_dir=self.tracks_dir,
            frame_landmarks_dir=self.frame_landmarks)
        authorization = object()
        combined = {"schema": "farfield_stage_reuse_bridge/v1"}
        with (mock.patch.object(
                ar.stage_reuse, "load_proof", return_value=authorization)
              as load_proof,
              mock.patch.object(
                  ar.stage_reuse, "require_target_checkout",
                  return_value=self.document["git_commit"]),
              mock.patch.object(
                  ar.stage_reuse, "require_configured_artifact",
                  side_effect=lambda reference, **unused:
                  artifact_lib.load_manifest(reference.path)),
              mock.patch.object(
                  ar.stage_reuse, "require_compatible_artifact",
                  side_effect=({"track": True}, {"frame": True}))
              as require_reuse,
              mock.patch.object(
                  ar.stage_reuse, "combine_bridge_provenance",
                  return_value=combined) as combine):
            (track_ref, frame_ref, bridge, returned_authorization,
             target_git_commit) = ar.authorize_prefix_inputs(
                 args, self.document)
        self.assertEqual(track_ref.path, str(self.tracks_dir.resolve()))
        self.assertEqual(frame_ref.path, str(self.frame_landmarks.resolve()))
        self.assertIs(bridge, combined)
        self.assertIs(returned_authorization, authorization)
        self.assertEqual(target_git_commit, self.document["git_commit"])
        load_proof.assert_called_once_with(self.build_config_path.parent)
        self.assertEqual(
            [call.kwargs["owner_stage"]
             for call in require_reuse.call_args_list],
            ["track", "extract"])
        self.assertTrue(all(
            call.kwargs["authorization"] is authorization
            for call in require_reuse.call_args_list))
        combine.assert_called_once_with({"track": True}, {"frame": True})

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

    def test_track_without_records_cannot_reach_audit_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            ghost = {
                "track_id": 99,
                "birth_obs_id": self.tracks[0]["birth_obs_id"],
                "birth_keyframe": 4,
                "status": "alive",
                "close_reason": "",
                "end_keyframe": None,
                "last_keyframe": None,
                "modal_label": "man_made=tower",
                "n_supported_keyframes": 0,
                "records": [],
            }
            tracks_dir = temporary / "tracks"
            write_tracks_artifact(
                tracks_dir,
                {"tracks_full.json": track_payload([
                    self.tracks[0], ghost])})
            source = ar.load_source_tracks(tracks_dir, DATASET)
            args = request_args(
                temporary / "requests", self.document, self.selected,
                self.orchestration)
            with self.assertRaisesRegex(
                    ValueError, "source track 99 has no records"):
                ar.build_request_artifact(
                    args, self.paths, *source, self.ingest_result)
            self.assertFalse(args.output_dir.exists())

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
            self.assertNotIn("decision", parsed)

    def test_provider_trimmed_keep_reaches_bearings_without_excluded_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            attempts_dir = temporary / llm.ATTEMPTS_DIR_NAME
            trimmed = audit_payload(
                "keep_single",
                [{"start_t": 0, "end_t": 1},
                 {"start_t": 3, "end_t": 3}])
            full_second_track = audit_payload(
                "keep_single", [{"start_t": 0, "end_t": 2}])
            llm.publish_attempt(
                attempts_dir,
                success(self.request_set, "T1", "trimmed", trimmed))
            llm.publish_attempt(
                attempts_dir,
                success(
                    self.request_set, "T5", "full", full_second_track))
            audit_dir = temporary / "semantic_audits"
            ar.publish_audit_results(
                audit_dir,
                request_dir=self.request_dir,
                tracks_dir=self.tracks_dir,
                attempts_dir=attempts_dir,
                dataset_name=DATASET,
                version="trimmed-keep-v1",
                arguments=("test",))

            audits = audit_io.load_audits(self.tracks_dir, audit_dir)
            accepted = tracklets.build_accepted_tracklets(
                audits.source_tracks, audits)
            accepted_t1 = next(
                item for item in accepted if item.local_id == "T1")
            self.assertEqual(accepted_t1.audit["verdict"], "keep")
            self.assertTrue(accepted_t1.audit["single_object"])
            self.assertEqual(
                [(segment.start_keyframe_idx, segment.end_keyframe_idx)
                 for segment in accepted_t1.valid_segments],
                [(0, 1), (3, 3)])

            observations = tracklets.build_camera_bearing_observations(
                accepted, PANO_W, bearing_sigma_deg=1.25)
            t1_observations = [
                observation for observation in observations
                if observation.tracklet_id.endswith("#T1")
            ]
            self.assertEqual(
                [observation.keyframe_idx for observation in t1_observations],
                [1, 3])
            self.assertNotIn(
                2,
                [observation.keyframe_idx
                 for observation in t1_observations])


if __name__ == "__main__":
    unittest.main()
