"""Viewer tests on a tiny synthetic run: matcher outputs are produced with
--build_only + fabricated results (no network), then the page is rendered
with and without the map pane."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import shapely

from experimental.overhead_matching.swag.farfield import testing
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
    match_viewer as mv,
)

DATASET = "tiny_harbor"


def write_feather(path: Path) -> Path:
    frame = schema.build_frame(
        ids=["('node', 101)", "('node', 102)", "('node', 103)"],
        geometries=[
            shapely.Point(testing.ANCHOR_LON + 0.001,
                          testing.ANCHOR_LAT + 0.001),
            shapely.Point(testing.ANCHOR_LON + 0.002,
                          testing.ANCHOR_LAT + 0.001),
            shapely.Point(testing.ANCHOR_LON + 0.002,
                          testing.ANCHOR_LAT + 0.002),
        ],
        landmark_types=["osm", "osm", "osm"],
        tags=[
            {"man_made": "lighthouse", "name": "Graves Light"},
            {"man_made": "tower"},
            {"man_made": "tower"},
        ],
    )
    frame.to_feather(path)
    return path


def track(track_id: int, keyframes) -> dict:
    return {
        "track_id": track_id,
        "birth_keyframe": min(keyframes),
        "close_reason": "end_of_range",
        "records": [
            {"keyframe": kf, "action": "propagate",
             "window_origin": [0, 0],
             "mask_bbox_window": [30, 10, 34, 14],
             "supports": []}
            for kf in keyframes
        ],
    }


def audit(verdict="keep") -> dict:
    return {
        "verdict": verdict,
        "landmark_kind": "fixed_structure",
        "valid_segments": [{"start_t": 0, "end_t": 10}],
        "unresolved": "",
        "primary_object": {
            "tags": [{"tag": "man_made=lighthouse", "weight": 0.9}],
            "name_candidates": [
                {"name": "Graves Light", "weight": 0.8, "basis": "both"}],
            "name_aliases": [],
            "description": "white conical tower",
            "distinctive_features": [],
            "extent": "point_like",
        },
    }


def write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tracks_seg0.json").write_text(json.dumps(
        {"range": {"name": "seg0"},
         "tracks": [track(1, [0, 1, 2]), track(2, [1, 2, 3])]}))
    audit_dir = run_dir / "semantic_audit"
    audit_dir.mkdir(exist_ok=True)
    meta = {}
    with open(audit_dir / "results.jsonl", "w") as f:
        for tid in (1, 2):
            key = f"T{tid}"
            meta[key] = {"track_id": tid, "birth_keyframe": 0,
                         "n_supports": 3, "chips": []}
            f.write(json.dumps({
                "key": key,
                "response": {"candidates": [{"content": {"parts": [
                    {"text": json.dumps(audit())}]}}]},
            }) + "\n")
    (audit_dir / "audit_meta.json").write_text(json.dumps(meta))


def fabricate_results(match_dir: Path, feather: Path, chunk_size: int) -> None:
    meta = json.loads((match_dir / "request_meta.json").read_text())
    sigs = sorted(ml.build_map_signatures(feather))
    chunks = [sigs[i:i + chunk_size]
              for i in range(0, len(sigs), chunk_size)]
    lighthouse = "man_made=lighthouse; name=Graves Light"
    with open(match_dir / "results.jsonl", "w") as f:
        for key, record in meta.items():
            chunk = chunks[record["chunk_index"]]
            entries = []
            for i, tid in enumerate(record["batch_keys"]):
                set_2 = []
                if tid == "T1" and lighthouse in chunk:
                    set_2.append({"set_2_id": chunk.index(lighthouse),
                                  "match_type": "instance",
                                  "confidence": 0.9})
                entries.append({"set_1_id": i, "set_2_matches": set_2,
                                "no_match_confidence": 0.2 if set_2 else 0.9,
                                "uniqueness_score": 5})
            f.write(json.dumps({
                "key": key,
                "response": {"candidates": [{"content": {"parts": [
                    {"text": json.dumps({"matches": entries})}]}}]},
            }) + "\n")


def run_main(module, argv):
    old = sys.argv
    sys.argv = [module.__name__] + argv
    try:
        module.main()
    finally:
        sys.argv = old


class MountOffsetResolutionTest(unittest.TestCase):
    def test_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            offset, source = mv.resolve_mount_offset(Path(tmp), 33.0)
            self.assertEqual((offset, source), (33.0, "--offset_deg"))

    def test_sun_outranks_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "sun_offset_check.json").write_text(json.dumps(
                {"offset_deg": 10.0, "usable": True, "verdict": "AGREEING"}))
            (run / "mount_offset_sweep.json").write_text(json.dumps(
                {"mount_offset_deg": 55.0, "usable": True,
                 "verdict": "CLEAN", "tracklets_used": 9}))
            offset, source = mv.resolve_mount_offset(run, None)
            self.assertEqual(offset, 10.0)
            self.assertIn("sun_offset_check.json", source)

    def test_unusable_sun_falls_back_to_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "sun_offset_check.json").write_text(json.dumps(
                {"offset_deg": 10.0, "usable": False,
                 "verdict": "FIXED-OBJECT"}))
            (run / "mount_offset_sweep.json").write_text(json.dumps(
                {"mount_offset_deg": 55.0, "usable": True,
                 "verdict": "CLEAN", "tracklets_used": 9}))
            offset, source = mv.resolve_mount_offset(run, None)
            self.assertEqual(offset, 55.0)
            self.assertIn("mount_offset_sweep.json", source)

    def test_nothing_usable_is_an_error_not_a_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                mv.resolve_mount_offset(Path(tmp), None)


class ViewerTest(unittest.TestCase):
    """Renders the page off a real (tiny) matching artifact."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        testing.make_dataset(root / "datasets" / DATASET, n_frames=6)
        cls.feather = write_feather(root / "cat.feather")
        # Inside the artifact lane so the dataset is inferred from the path.
        cls.run_dir = (root / "artifacts" / "object_tracks" / DATASET /
                       "v1" / "runs" / "r001")
        write_run(cls.run_dir)
        (cls.run_dir / "sun_offset_check.json").write_text(json.dumps(
            {"offset_deg": 10.0, "usable": True, "verdict": "AGREEING"}))

        flags = [
            "--run_dir", str(cls.run_dir),
            "--feather", str(cls.feather),
            "--query_batch", "2",
            "--chunk_size", "2",
            "--thinking_level", "HIGH",
            "--confidence_floor", "0.05",
            "--instance_max_rows", "1",
            "--model", "test-model",
        ]
        run_main(ml, flags + ["--build_only"])
        fabricate_results(cls.run_dir / "matching", cls.feather, chunk_size=2)
        run_main(ml, flags + ["--aggregate_only"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _page(self):
        return (self.run_dir / "matching" / "review" / "index.html")

    def test_no_map_page_renders(self):
        run_main(mv, ["--run_dir", str(self.run_dir), "--no_map"])
        page = self._page().read_text()
        self.assertIn("map skipped", page)
        self.assertIn("data-key='T1'", page)
        # Track links use each track's own range name (all ranges loaded,
        # not next(glob)).
        self.assertIn("track_seg0_T1.html", page)
        self.assertIn("semantic_audit/review/index.html#T1", page)

    def test_map_page_draws_recorded_catalog_and_sidecar_offset(self):
        run_main(mv, ["--run_dir", str(self.run_dir),
                      "--epoch_keyframes", "5",
                      "--bearing_sigma_deg", "1.0"])
        page = self._page().read_text()
        self.assertIn("MAP_DATA", page)
        # The offset came from the run's sidecar, not dataset metadata (which
        # testing.default_metadata records as 214 -- must NOT appear).
        self.assertIn("sun_offset_check.json (AGREEING, absolute)", page)
        payload = json.loads(
            page.split("MAP_DATA=", 1)[1].split(";</script>", 1)[0]
            .replace("<\\/", "</"))
        self.assertEqual(payload["offset"]["deg"], 10.0)
        self.assertIn("T1", payload["tracklets"])
        self.assertTrue(payload["tracklets"]["T1"]["rays"])
        target_ids = [t[4] for t in payload["tracklets"]["T1"]["targets"]]
        self.assertIn("osm:node:101", target_ids)
        # Self-contained page: provenance manifest beside it.
        self.assertTrue((self._page().parent / "manifest.json").exists())

    def test_map_without_fusion_params_is_refused(self):
        with self.assertRaises(SystemExit):
            run_main(mv, ["--run_dir", str(self.run_dir)])

    def test_missing_settings_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = Path(tmp) / "run"
            (bare / "matching").mkdir(parents=True)
            with self.assertRaises(SystemExit):
                run_main(mv, ["--run_dir", str(bare), "--no_map"])


if __name__ == "__main__":
    unittest.main()
