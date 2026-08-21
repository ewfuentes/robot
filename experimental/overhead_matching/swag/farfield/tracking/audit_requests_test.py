"""Request-builder tests on a tiny synthetic run: requests are built with no
network (no --submit), then gating, settings/provenance content, and the
preview page are checked."""

import dataclasses
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    provenance,
    testing,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    audit_requests as ar,
    semantic_audit as sa,
    track_builder as tb,
)

DATASET = "tiny_harbor"
PANO_W = 256

# continue_clean under the default thresholds recorded in the artifact.
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


def record(keyframe, supported=True):
    return {"keyframe": keyframe, "action": "continue_mask",
            "window_origin": [0.0, 0], "window_px": PANO_W,
            "mask_area": 100, "mask_bbox_window": [40, 30, 80, 60],
            "supports": [support(keyframe)] if supported else []}


def birth_record(keyframe):
    return {"keyframe": keyframe, "action": "birth",
            "window_origin": [0.0, 0], "window_px": PANO_W,
            "health": {"ok": True}}


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
        "records": [birth_record(birth_kf)]
        + [record(kf) for kf in support_kfs],
    }


def artifact(range_name, tracks):
    cfg = tb.TrackBuilderConfig(reference_pano_width=PANO_W)
    return {"range": {"name": range_name, "k_start": 0, "k_end": 4},
            "config": dataclasses.asdict(cfg),
            "tracks": tracks, "rejected_births": [], "track_overlaps": []}


def make_run(root: Path):
    """Dataset + frame_landmarks + a two-range tracking run."""
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
    # Two ranges: the stage must consume ALL tracks_*.json, not next(glob).
    (run_dir / "tracks_seg0.json").write_text(json.dumps(artifact(
        "seg0", [track(1, 0, [1, 2, 3]),    # 3 supports: above the bar
                 track(2, 2, [3])])))       # 1 support: below the bar
    (run_dir / "tracks_seg1.json").write_text(json.dumps(artifact(
        "seg1", [track(5, 1, [2, 3])])))    # 2 supports: at the bar
    return frame_landmarks, run_dir


def flags(frame_landmarks, run_dir):
    return ["--run_dir", str(run_dir),
            "--landmark_base", str(frame_landmarks),
            "--min_supports", "2",
            "--thinking_level", "LOW",
            "--max_support_chips", "2",
            "--max_context_chips", "2",
            "--max_description_samples", "5",
            "--chip_height_px", "64",
            "--fov_deg", "90.0",
            "--seam_gap_norm", "25",
            "--seam_min_y_iou", "0.3",
            "--model", "test-model"]


class AuditRequestsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.frame_landmarks, cls.run_dir = make_run(root)
        run_main(ar, flags(cls.frame_landmarks, cls.run_dir))
        cls.audit_dir = cls.run_dir / "semantic_audit"
        cls.requests = {}
        with open(cls.audit_dir / "requests.jsonl") as f:
            for line in f:
                r = json.loads(line)
                cls.requests[r["key"]] = r["request"]
        cls.meta = json.loads((cls.audit_dir / "audit_meta.json").read_text())
        cls.settings = json.loads(
            (cls.audit_dir / "settings.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_requests_gate_on_min_supports_across_all_ranges(self):
        # T1 (3 supports, seg0) and T5 (2 supports, seg1) are audited; T2
        # (1 support) is below the bar and carried no further.
        self.assertEqual(set(self.requests), {"T1", "T5"})
        self.assertEqual(set(self.meta), {"T1", "T5"})

    def test_request_shape(self):
        req = self.requests["T1"]
        self.assertEqual(
            req["systemInstruction"]["parts"][0]["text"], sa.SYSTEM_PROMPT)
        gen = req["generationConfig"]
        self.assertEqual(gen["thinkingConfig"]["thinkingLevel"], "LOW")
        self.assertEqual(gen["responseMimeType"], "application/json")
        self.assertIn("required", gen["responseSchema"])
        parts = req["contents"][0]["parts"]
        self.assertTrue(parts[0]["text"].startswith("TRACK EVIDENCE"))
        images = [p for p in parts if "inline_data" in p]
        self.assertEqual(len(images), len(self.meta["T1"]["chips"]))
        self.assertGreaterEqual(len(images), 1)

    def test_meta_join_info(self):
        m1 = self.meta["T1"]
        self.assertEqual(m1["track_id"], 1)
        self.assertEqual(m1["range"], "seg0")
        self.assertEqual(m1["birth_keyframe"], 0)
        self.assertEqual(m1["n_supports"], 3)
        self.assertEqual(m1["support_obs_by_t"],
                         {"1": "f0001__lm0__box0", "2": "f0002__lm0__box0",
                          "3": "f0003__lm0__box0"})
        self.assertEqual(self.meta["T5"]["range"], "seg1")
        for chip in m1["chips"]:
            self.assertTrue((self.audit_dir / "chips"
                             / Path(chip).name).exists())

    def test_settings_record_the_whole_recipe(self):
        s = self.settings
        self.assertEqual(s["model"], "test-model")
        self.assertEqual(s["thinking_level"], "LOW")
        self.assertEqual(s["min_supports"], 2)
        self.assertEqual(
            s["system_prompt_sha256"],
            hashlib.sha256(sa.SYSTEM_PROMPT.encode()).hexdigest())
        self.assertEqual(s["audit_config"], {
            "min_supports": 2, "max_support_chips": 2,
            "max_context_chips": 2, "max_description_samples": 5,
            "chip_height_px": 64, "thinking_level": "LOW"})
        # The classifier is each range's RECORDED TrackBuilderConfig.
        self.assertEqual(set(s["classifier_by_range"]), {"seg0", "seg1"})
        self.assertEqual(
            s["classifier_by_range"]["seg0"]["reference_pano_width"], PANO_W)
        self.assertEqual(s["ingest"], {"fov_deg": 90.0,
                                       "seam_gap_norm": 25.0,
                                       "seam_min_y_iou": 0.3})
        self.assertEqual(s["tracks_files"],
                         ["tracks_seg0.json", "tracks_seg1.json"])
        self.assertIn("--model", s["argv"])
        self.assertIn("git_commit", s)
        self.assertEqual(s["dataset"], DATASET)
        self.assertFalse(s["submitted_by_stage"])
        self.assertEqual((s["n_tracks_total"], s["n_eligible"],
                          s["n_requests"]), (3, 2, 2))

    def test_provenance_manifest(self):
        manifest = provenance.read(self.audit_dir)
        self.assertEqual(manifest["schema"], "farfield_provenance/v1")
        self.assertEqual(manifest["config"]["model"], "test-model")
        self.assertIn("run_dir", manifest["inputs"])

    def test_preview_page(self):
        page = (self.audit_dir / "preview" / "index.html").read_text()
        self.assertIn("id='T1'", page)
        self.assertIn("id='T5'", page)
        self.assertIn("You audit object tracks", page)  # system prompt shown
        self.assertIn("../chips/T1_t", page)
        self.assertIn("audit_requests", page)  # provenance footer


if __name__ == "__main__":
    unittest.main()
