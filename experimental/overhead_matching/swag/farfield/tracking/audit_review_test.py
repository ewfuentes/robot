"""Review-page tests over explicit immutable track and audit artifacts."""

import argparse
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    dataset,
    llm_lifecycle as llm,
    paths as paths_lib,
    testing,
)
from experimental.overhead_matching.swag.farfield.calibration import audit_io
from experimental.overhead_matching.swag.farfield.tracking import (
    audit_requests as ar,
    audit_review as av,
    track_builder as tb,
)

DATASET = "tiny_harbor"
PANO_W = 256
CLEAN = {"iou": 0.8, "inter_over_mask": 0.9, "inter_over_box": 0.9}


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = old


def support(keyframe, observation_id):
    return {
        "obs_id": observation_id,
        "class": "recorded-at-run-time",
        "box_window": [40.0, 30.0, 80.0, 60.0],
        **CLEAN,
    }


def record(keyframe, observation_id):
    return {
        "keyframe": keyframe,
        "action": "continue_mask",
        "window_origin": [0.0, 0],
        "window_px": PANO_W,
        "mask_area": 100,
        "mask_bbox_window": [40, 30, 80, 60],
        "supports": [support(keyframe, observation_id)],
    }


def track(track_id, birth_keyframe, support_keyframes, observations):
    return {
        "track_id": track_id,
        "birth_obs_id": observations[birth_keyframe],
        "birth_keyframe": birth_keyframe,
        "status": "closed",
        "close_reason": "starved",
        "end_keyframe": max(support_keyframes),
        "last_keyframe": max(support_keyframes),
        "modal_label": "man_made=tower 'Graves Light'",
        "n_supported_keyframes": len(support_keyframes),
        "records": [{
            "keyframe": birth_keyframe,
            "mask_area": 100,
            "mask_bbox_window": [40, 30, 80, 60],
            "supports": [],
            "action": "birth",
            "window_origin": [0.0, 0],
            "window_px": PANO_W,
            "health": {"ok": True},
        }] + [
            record(keyframe, observations[keyframe])
            for keyframe in support_keyframes
        ],
    }


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
    observations = {
        observation.frame_idx: observation.obs_id
        for observation in ingest.observations
    }
    tracks = [
        track(1, 0, [1, 2, 3], observations),
        track(5, 1, [2, 3], observations),
    ]
    cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    document = {
        "range": {"name": "full", "k_start": 0, "k_end": 4},
        "config": dataclasses.asdict(cfg),
        "tracks": tracks,
        "rejected_births": [],
        "track_overlaps": [],
    }
    frame_ref = artifact.open_artifact(frame_landmarks)
    source_digest = artifact.sha256_json(
        paths_lib.dataset_source_digests(base))
    ingest = {"fov_deg": 90.0, "seam_gap_norm": 25.0,
              "seam_min_y_iou": 0.3}
    tracks_dir = (
        root / "artifacts" / paths_lib.OBJECT_TRACKS / DATASET / "v1")
    with artifact.ArtifactDirectoryBuilder(
            tracks_dir,
            kind=paths_lib.OBJECT_TRACKS,
            dataset=DATASET,
            version="v1",
            generator="audit_review_test",
            git_commit="test",
            config={
                "schema": "farfield_object_tracks/v1",
                "coverage": "complete",
                "range": document["range"],
                "source_digests": {
                    "dataset_tracking_inputs": source_digest,
                },
                "resolved": {
                    "ingest": ingest,
                },
            },
            upstreams=(frame_ref,),
            declared_outputs=("tracks_full.json",)) as builder:
        artifact.atomic_write_json(
            builder.output_path("tracks_full.json"), document)
    return base, frame_landmarks, tracks_dir


def audit_payload(*, with_edits):
    payload = {
        "landmark_kind": "fixed_structure",
        "decision": "keep_single",
        "valid_segments": [{"start_t": 0, "end_t": 2}],
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9}],
            "name_candidates": [{
                "name": "Graves Light",
                "weight": 0.8,
                "basis": "both",
            }],
            "name_aliases": ["The Graves"],
            "description": "white conical masonry tower",
            "distinctive_features": ["black lantern"],
            "extent": "point_like",
        },
        "strike_votes": [],
        "secondary_objects": [],
        "confidence": "high",
        "unresolved": "",
    }
    if with_edits:
        payload["valid_segments"] = [{"start_t": 0, "end_t": 3}]
        payload["strike_votes"] = [
            {"t": 2, "reason": "different building"}]
        payload["secondary_objects"] = [{
            "tags": [{"tag": "man_made=crane", "weight": 0.5}],
            "name": "",
            "description": "a crane described only in text",
            "ts": [3],
            "relation": "adjacent",
            "worth_own_landmark": False,
        }]
    return payload


def response(payload):
    return {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps(payload)
        }]}}]
    }


def build_request_artifact(dataset_base, frame_landmarks, tracks_dir,
                           request_dir):
    ingest_params = dataset.IngestParams(
        fov_deg=90.0, seam_gap_norm=25.0, seam_min_y_iou=0.3)
    args = argparse.Namespace(
        output_dir=request_dir,
        output_version="requests-v1",
        model="test-model",
        min_supports=2,
        thinking_level="LOW",
        max_support_chips=2,
        max_context_chips=2,
        max_description_samples=5,
        chip_height_px=64,
        fov_deg=ingest_params.fov_deg,
        seam_gap_norm=ingest_params.seam_gap_norm,
        seam_min_y_iou=ingest_params.seam_min_y_iou,
        ingest_params=ingest_params,
        build_identity="audit-review-test",
        orchestration={
            "schema": "farfield_pipeline_stage/v1",
            "stage": "audit",
            "config_digest": "0" * 64,
        },
        resolved_stage_config={"fixture": "audit-review-test"})
    paths = argparse.Namespace(dataset=DATASET, dataset_base=dataset_base)
    source = ar.load_source_tracks(tracks_dir, DATASET)
    ingest_result = dataset.run_ingest(
        dataset_base, frame_landmarks, ingest_params)
    return ar.build_request_artifact(
        args, paths, *source, ingest_result)


class AuditReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        (cls.dataset_base, cls.frame_landmarks,
         cls.tracks_dir) = make_inputs(cls.root)
        cls.request_dir = cls.root / "audit_requests" / "v1"
        build_request_artifact(
            cls.dataset_base, cls.frame_landmarks, cls.tracks_dir,
            cls.request_dir)
        request_set = llm.load_request_set(
            cls.request_dir / llm.REQUEST_SET_NAME)
        cls.attempts = cls.root / llm.ATTEMPTS_DIR_NAME
        payloads = {"T1": audit_payload(with_edits=True),
                    "T5": audit_payload(with_edits=False)}
        for unit in request_set.units:
            llm.publish_attempt(cls.attempts, llm.Attempt(
                request_set_fingerprint=request_set.fingerprint,
                key=unit.key,
                attempt_id=f"success-{unit.key}",
                response=response(payloads[unit.key]),
                error=None,
                metadata={"transport": "test"}))
        cls.audit_dir = (
            cls.root / "artifacts" / paths_lib.SEMANTIC_AUDITS
            / DATASET / "v1")
        ar.publish_audit_results(
            cls.audit_dir,
            request_dir=cls.request_dir,
            tracks_dir=cls.tracks_dir,
            attempts_dir=cls.attempts,
            dataset_name=DATASET,
            version="v1",
            arguments=("test",))
        cls.audit_ref_before_review = artifact.open_artifact(cls.audit_dir)
        cls.review_dir = cls.root / "reviews" / "audit-v1"
        cls.review_flags = [
            "--tracks_dir", str(cls.tracks_dir),
            "--semantic_audits_dir", str(cls.audit_dir),
            "--dataset_base", str(cls.dataset_base),
            "--frame_landmarks_dir", str(cls.frame_landmarks),
            "--output_dir", str(cls.review_dir),
        ]
        run_main(av, cls.review_flags)
        cls.page = (cls.review_dir / "index.html").read_text()

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_page_sections_and_complete_coverage(self):
        self.assertIn("id='T1'", self.page)
        self.assertIn("id='T5'", self.page)
        self.assertIn("complete canonical coverage", self.page)
        self.assertIn("v_keep", self.page)
        self.assertIn("Graves Light", self.page)
        self.assertIn("range: full", self.page)
        self.assertIn("preview/index.html#T1", self.page)
        self.assertIn("test-model", self.page)
        self.assertIn("man_made=crane", self.page)
        self.assertNotIn("result errors", self.page)

    def test_strike_gets_extra_chip_in_review_output(self):
        chip = self.review_dir / "chips" / "T1_t0002_extra.jpg"
        self.assertTrue(chip.exists())
        self.assertIn("chips/T1_t0002_extra.jpg", self.page)
        self.assertIn("different building", self.page)

    def test_request_chips_are_copied_not_linked_through_mutable_artifact(self):
        meta = json.loads(
            (self.audit_dir / "audit_meta.json").read_text())["requests"]
        for request_meta in meta.values():
            for relative in request_meta["chips"]:
                self.assertTrue(
                    (self.review_dir / "chips" / Path(relative).name).is_file())

    def test_review_does_not_mutate_semantic_audit_artifact(self):
        self.assertEqual(
            artifact.open_artifact(self.audit_dir),
            self.audit_ref_before_review)
        self.assertFalse((self.audit_dir / "review").exists())

    def test_no_extra_chips_still_renders(self):
        output = self.root / "reviews" / "without-extra"
        flags = list(self.review_flags)
        flags[flags.index("--output_dir") + 1] = str(output)
        flags.append("--no_extra_chips")
        run_main(av, flags)
        page = (output / "index.html").read_text()
        self.assertIn("id='T1'", page)
        self.assertFalse((output / "chips" / "T1_t0002_extra.jpg").exists())

    def test_output_inside_immutable_artifact_is_refused(self):
        flags = list(self.review_flags)
        output_index = flags.index("--output_dir") + 1
        flags[output_index] = str(self.audit_dir / "review")
        with self.assertRaises(SystemExit):
            run_main(av, flags)

    def test_invalidated_input_artifact_is_refused(self):
        results = self.audit_dir / "results.jsonl"
        original = results.read_bytes()
        results.write_bytes(original + b"\n")
        try:
            with self.assertRaises(audit_io.AuditArtifactError):
                run_main(av, self.review_flags)
        finally:
            results.write_bytes(original)
        self.assertEqual(
            artifact.open_artifact(self.audit_dir),
            self.audit_ref_before_review)

    def test_cli_has_no_fresh_ingest_or_legacy_path_flags(self):
        args = av.build_parser().parse_args([
            "--tracks_dir", "/artifacts/tracks",
            "--semantic_audits_dir", "/artifacts/audits",
            "--dataset_base", "/datasets/ds",
            "--frame_landmarks_dir", "/artifacts/frames",
            "--output_dir", "/reviews/audit",
        ])
        self.assertFalse(hasattr(args, "fov_deg"))
        self.assertFalse(hasattr(args, "seam_gap_norm"))
        self.assertFalse(hasattr(args, "landmark_base"))

    def test_recorded_ingest_requires_the_exact_shape(self):
        settings = {
            "ingest": {"fov_deg": 91.0, "seam_gap_norm": 24.0,
                       "seam_min_y_iou": 0.31}}
        self.assertEqual(
            av.recorded_ingest(settings),
            dataset.IngestParams(
                fov_deg=91.0, seam_gap_norm=24.0,
                seam_min_y_iou=0.31))
        settings["ingest"]["unknown"] = 1
        with self.assertRaisesRegex(SystemExit, "exact recorded ingest"):
            av.recorded_ingest(settings)

if __name__ == "__main__":
    unittest.main()
