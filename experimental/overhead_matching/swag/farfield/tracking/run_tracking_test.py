import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import dataset
from experimental.overhead_matching.swag.farfield.tracking import (
    run_tracking as rt,
    track_builder as tb,
)

RANGES = [("legA", 0, 10), ("legB", 11, 20)]


def make_run(run_dir: Path):
    return rt.write_run_meta(
        run_dir, run_name="r001", dataset_name="test_ds",
        notes="unit test",
        inputs={
            "dataset_base": Path("/data/x/datasets/test_ds"),
            "frame_landmarks": Path("/data/x/artifacts/frame_landmarks/v1"),
            "video": None,  # keyframe-only dataset: recorded as known-absent
            "sam2_checkpoint": Path("/data/x/models/sam2/ckpt.pt"),
        },
        builder_cfg=tb.TrackBuilderConfig(reference_pano_width=7680),
        ingest_params=dataset.IngestParams(fov_deg=90.0, seam_gap_norm=25.0,
                                           seam_min_y_iou=0.3),
        ranges=RANGES)


class RunMetaTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "r001"
        self.addCleanup(self._tmp.cleanup)

    def test_records_every_input_including_checkpoint(self):
        # Audit finding B: the old stage omitted the SAM2 checkpoint from
        # run_meta, so a run could not say which weights built its masks.
        doc = json.loads(make_run(self.run_dir).read_text())
        self.assertEqual(doc["inputs"]["sam2_checkpoint"],
                         "/data/x/models/sam2/ckpt.pt")
        self.assertEqual(doc["inputs"]["dataset_base"],
                         "/data/x/datasets/test_ds")
        self.assertEqual(doc["inputs"]["frame_landmarks"],
                         "/data/x/artifacts/frame_landmarks/v1")
        # video is recorded as known-absent, not silently omitted.
        self.assertIn("video", doc["inputs"])
        self.assertIsNone(doc["inputs"]["video"])
        self.assertNotEqual(doc["git_commit"], "")

    def test_records_full_configs_verbatim(self):
        doc = json.loads(make_run(self.run_dir).read_text())
        track_cfg = doc["config"]["track_builder"]
        self.assertEqual(track_cfg["reference_pano_width"], 7680)
        self.assertEqual(track_cfg["clean_iou"], 0.45)
        # The recorded dict must round-trip into the dataclass (the reader
        # contract): every field present, none extra.
        self.assertEqual(tb.TrackBuilderConfig(**track_cfg),
                         tb.TrackBuilderConfig(reference_pano_width=7680))
        self.assertEqual(doc["config"]["ingest"],
                         {"fov_deg": 90.0, "seam_gap_norm": 25.0,
                          "seam_min_y_iou": 0.3})
        self.assertEqual([r["name"] for r in doc["ranges"]],
                         ["legA", "legB"])

    def test_run_meta_makes_no_completion_claim(self):
        # The P0 fix: run_meta.json is written BEFORE tracking and must never
        # read as "done". It carries no completed-ranges field, and its
        # completion note points at the real marker.
        doc = json.loads(make_run(self.run_dir).read_text())
        self.assertNotIn("completed", doc)
        self.assertIn(rt.TRACKS_COMPLETE, doc["completion"])


class CompletionMarkerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "r001"
        self.addCleanup(self._tmp.cleanup)
        make_run(self.run_dir)

    def test_fresh_run_has_every_range_unfinished(self):
        self.assertEqual(rt.completed_ranges(self.run_dir), {})
        self.assertEqual(rt.unfinished_ranges(self.run_dir),
                         ["legA", "legB"])

    def test_marks_accumulate_per_range(self):
        rt.mark_range_complete(self.run_dir, "legA")
        self.assertEqual(rt.unfinished_ranges(self.run_dir), ["legB"])
        self.assertEqual(list(rt.completed_ranges(self.run_dir)), ["legA"])
        rt.mark_range_complete(self.run_dir, "legB")
        self.assertEqual(rt.unfinished_ranges(self.run_dir), [])
        # Timestamps are per range, ISO-shaped.
        completed = rt.completed_ranges(self.run_dir)
        for name in ("legA", "legB"):
            self.assertRegex(completed[name], r"^\d{4}-\d{2}-\d{2}T")

    def test_marking_is_idempotent(self):
        rt.mark_range_complete(self.run_dir, "legA")
        rt.mark_range_complete(self.run_dir, "legA")
        self.assertEqual(list(rt.completed_ranges(self.run_dir)), ["legA"])

    def test_rerun_meta_write_does_not_erase_marks(self):
        # A resumed run rewrites run_meta.json first; the completion marker
        # must survive so --skip_existing_ranges can trust it.
        rt.mark_range_complete(self.run_dir, "legA")
        make_run(self.run_dir)
        self.assertEqual(rt.unfinished_ranges(self.run_dir), ["legB"])

    def test_corrupt_marker_reads_as_nothing_done(self):
        # Fail safe: an unreadable marker means "re-run", never "done".
        (self.run_dir / rt.TRACKS_COMPLETE).write_text("{not json")
        self.assertEqual(rt.completed_ranges(self.run_dir), {})
        self.assertEqual(rt.unfinished_ranges(self.run_dir),
                         ["legA", "legB"])


if __name__ == "__main__":
    unittest.main()
