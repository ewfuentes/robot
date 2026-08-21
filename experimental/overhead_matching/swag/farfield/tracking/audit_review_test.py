"""Review-page tests: requests are built with no network, results are
fabricated, and the page is rendered under the RECORDED settings."""

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import testing
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


def support(keyframe):
    return {"obs_id": f"f{keyframe:04d}__lm0__box0",
            "class": "recorded-at-run-time",
            "box_window": [40.0, 30.0, 80.0, 60.0], **CLEAN}


def record(keyframe):
    return {"keyframe": keyframe, "action": "continue_mask",
            "window_origin": [0.0, 0], "window_px": PANO_W,
            "mask_area": 100, "mask_bbox_window": [40, 30, 80, 60],
            "supports": [support(keyframe)]}


def track(track_id, birth_kf, support_kfs):
    return {
        "track_id": track_id,
        "birth_obs_id": f"f{birth_kf:04d}__lm0__box0",
        "birth_keyframe": birth_kf,
        "status": "closed", "close_reason": "starved",
        "end_keyframe": max(support_kfs),
        "last_keyframe": max(support_kfs),
        "modal_label": "man_made=tower 'Graves Light'",
        "n_supported_keyframes": len(support_kfs),
        "records": [{"keyframe": birth_kf, "action": "birth",
                     "window_origin": [0.0, 0], "window_px": PANO_W,
                     "health": {"ok": True}}]
        + [record(kf) for kf in support_kfs],
    }


def make_run(root: Path):
    base = testing.make_dataset(root / "datasets" / DATASET, n_frames=5,
                                pano_size=(PANO_W, 128))
    stems = sorted(p.stem for p in (base / "panorama").glob("*.jpg"))
    frame_landmarks = root / "artifacts" / "frame_landmarks" / DATASET / "v1"
    testing.make_predictions(
        frame_landmarks,
        {stem: [testing.landmark("Graves Light",
                                 [(0, 400, 300, 600, 500)])]
         for stem in stems})
    run_dir = (root / "artifacts" / "object_tracks" / DATASET / "v1"
               / "runs" / "r001")
    run_dir.mkdir(parents=True)
    cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    (run_dir / "tracks_seg0.json").write_text(json.dumps(
        {"range": {"name": "seg0", "k_start": 0, "k_end": 4},
         "config": dataclasses.asdict(cfg),
         "tracks": [track(1, 0, [1, 2, 3]), track(5, 1, [2, 3])],
         "rejected_births": [], "track_overlaps": []}))
    return frame_landmarks, run_dir


def audit_payload():
    """Schema-valid TrackAudit dict for T1 with one strike (t2, a keyframe
    whose chip was NOT in the request: max_support_chips=2 keeps t1+t3) and
    one text-only secondary object."""
    return {
        "landmark_kind": "fixed_structure",
        "single_object": True,
        "valid_segments": [{"start_t": 0, "end_t": 3}],
        "verdict": "keep",
        "drop_reason": "none",
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9}],
            "name_candidates": [{"name": "Graves Light", "weight": 0.8,
                                 "basis": "both"}],
            "name_aliases": ["The Graves"],
            "description": "white conical masonry tower",
            "distinctive_features": ["black lantern"],
            "extent": "point_like"},
        "strike_votes": [{"t": 2, "reason": "different building"}],
        "secondary_objects": [{
            "tags": [{"tag": "man_made=crane", "weight": 0.5}],
            "name": "", "description": "a crane described only in text",
            "ts": [3], "relation": "adjacent",
            "worth_own_landmark": False}],
        "confidence": "high",
        "unresolved": "",
    }


def result_line(key, payload):
    return json.dumps({
        "key": key,
        "response": {"candidates": [{"content": {"parts": [
            {"text": json.dumps(payload)}]}}]}})


class AuditReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.frame_landmarks, cls.run_dir = make_run(root)
        run_main(ar, ["--run_dir", str(cls.run_dir),
                      "--landmark_base", str(cls.frame_landmarks),
                      "--min_supports", "2",
                      "--thinking_level", "LOW",
                      "--max_support_chips", "2",
                      "--max_context_chips", "2",
                      "--max_description_samples", "5",
                      "--chip_height_px", "64",
                      "--fov_deg", "90.0",
                      "--seam_gap_norm", "25",
                      "--seam_min_y_iou", "0.3",
                      "--model", "test-model"])
        cls.audit_dir = cls.run_dir / "semantic_audit"
        (cls.audit_dir / "results.jsonl").write_text(
            result_line("T1", audit_payload()) + "\n"
            + json.dumps({"key": "T5", "error": "quota exceeded"}) + "\n")
        cls.review_flags = ["--run_dir", str(cls.run_dir),
                            "--landmark_base", str(cls.frame_landmarks),
                            "--fov_deg", "90.0",
                            "--seam_gap_norm", "25",
                            "--seam_min_y_iou", "0.3"]
        run_main(av, cls.review_flags)
        cls.page = (cls.audit_dir / "review" / "index.html").read_text()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_page_sections(self):
        self.assertIn("id='T1'", self.page)
        self.assertIn("v_keep", self.page)
        self.assertIn("Graves Light", self.page)
        # Track links carry the track's own range name.
        self.assertIn("track_seg0_T1.html", self.page)
        self.assertIn("../preview/index.html#T1", self.page)
        # The recorded provenance is displayed.
        self.assertIn("test-model", self.page)
        # Secondary object rendered even without a chip.
        self.assertIn("man_made=crane", self.page)

    def test_errors_come_from_the_reparse_not_the_canonical_reader(self):
        self.assertIn("result errors", self.page)
        self.assertIn("quota exceeded", self.page)
        # The errored key gets no track section.
        self.assertNotIn("id='T5'", self.page)

    def test_strike_gets_an_extra_chip_under_recorded_config(self):
        # t2's chip was not in the request (max_support_chips=2 kept t1+t3),
        # so the strike triggers an on-demand render.
        chip = self.audit_dir / "chips" / "T1_t0002_extra.jpg"
        self.assertTrue(chip.exists())
        self.assertIn("T1_t0002_extra.jpg", self.page)
        self.assertIn("different building", self.page)

    def test_no_extra_chips_still_renders(self):
        run_main(av, self.review_flags + ["--no_extra_chips"])
        page = (self.audit_dir / "review" / "index.html").read_text()
        self.assertIn("id='T1'", page)
        # Restore the full page for any later-ordered test.
        run_main(av, self.review_flags)

    def test_missing_settings_is_refused_not_defaulted(self):
        settings = self.audit_dir / "settings.json"
        aside = settings.with_suffix(".aside")
        settings.rename(aside)
        try:
            with self.assertRaises(SystemExit) as ctx:
                run_main(av, self.review_flags)
            self.assertIn("settings", str(ctx.exception))
        finally:
            aside.rename(settings)

    def test_missing_results_is_refused_with_pointer(self):
        results = self.audit_dir / "results.jsonl"
        aside = results.with_suffix(".aside")
        results.rename(aside)
        try:
            with self.assertRaises(SystemExit) as ctx:
                run_main(av, self.review_flags)
            self.assertIn("results.jsonl", str(ctx.exception))
        finally:
            aside.rename(results)


if __name__ == "__main__":
    unittest.main()
