"""Viewer tests on a tiny synthetic run: matcher outputs are produced with
--build_only + fabricated results (no network), then the page is rendered
with and without the map pane."""

import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import build_config
from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield import testing
from experimental.overhead_matching.swag.farfield import llm_lifecycle
from experimental.overhead_matching.swag.farfield.matching import (
    match_landmarks as ml,
    match_landmarks_test as fixtures,
    match_viewer as mv,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

DATASET = fixtures.DATASET


def fabricate_results(match_dir: Path) -> None:
    work_dir = ml.matching_work_dir(match_dir)
    snapshot = llm_lifecycle.load_request_set(
        work_dir / llm_lifecycle.REQUEST_SET_NAME)
    semantic = json.loads(
        (work_dir / ml.WORK_SNAPSHOT_NAME).read_text())
    lighthouse = next(
        signature_id for signature_id, entry in semantic["signatures"].items()
        if entry["display_label"]
        == "man_made=lighthouse; name=Graves Light")
    with open(work_dir / ml.TRANSPORT_RESULTS_NAME, "w") as f:
        for unit in snapshot.units:
            key = unit.key
            record = unit.metadata
            chunk = record["chunk_signature_ids"]
            entries = []
            for i, tid in enumerate(record["batch_keys"]):
                local_id = tid.rsplit("#", 1)[-1]
                set_2 = []
                if local_id == "T1" and lighthouse in chunk:
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


class ViewerTest(unittest.TestCase):
    """Renders the page off a real (tiny) matching artifact."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.dataset_base = cls.root / "datasets" / DATASET
        testing.make_dataset(cls.dataset_base, n_frames=6)
        # The build config first: its identity is what the upstream artifacts
        # must be bound to, so they cannot be written before it exists.
        build_path, orchestration_digest = fixtures.write_build_config(
            cls.root, cls.dataset_base)
        document = build_config.load(build_path.parent)
        (cls.tracks_dir, cls.audit_dir,
         cls.catalog_dir) = fixtures.write_bound_inputs(
             cls.root, document["build_identity"])
        cls.feather = cls.catalog_dir / "catalog.feather"
        cls.match_dir = (cls.root / "landmark_matches" /
                         fixtures.MATCHES_VERSION)
        cls.match_flags = [
            "--dataset", DATASET,
            "--dataset_base", str(cls.dataset_base),
            "--tracks_dir", str(cls.tracks_dir),
            "--audit_dir", str(cls.audit_dir),
            "--catalog_dir", str(cls.catalog_dir),
            "--output_dir", str(cls.match_dir),
            "--build_config", str(build_path),
            "--orchestration_config_digest", orchestration_digest,
            "--online",
        ]
        run_main(ml, cls.match_flags + ["--build_only"])
        fabricate_results(cls.match_dir)
        run_main(ml, [
            "--dataset", DATASET,
            "--output_dir", str(cls.match_dir),
            "--aggregate_only",
        ])

        cls.calibration_path = cls.root / "nominal_forward.json"
        pano_width = 360
        pano_column = 10.0
        cls.calibration_path.write_text(json.dumps({
            "schema": "farfield_nominal_forward/v1",
            "frame": "camera_centre_column_nominal_forward_axis_v1",
            "approved": True,
            "dataset": DATASET,
            "version": "approved-v1",
            "mounting_id": "test-camera",
            "panorama_column": pano_column,
            "panorama_width": pano_width,
            "bearing_camera_cw_deg": float(
                geometry.azimuth_of_pano_column(pano_column, pano_width)),
            "uncertainty_deg": 1.0,
            "evidence_frame_ids": ["frame-0000"],
            "operator": "unit-test",
            "approved_at": "2026-08-23T00:00:00Z",
            "notes": "test calibration",
        }))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _viewer_flags(self, output_name):
        return [
            "--dataset", DATASET,
            "--dataset_base", str(self.dataset_base),
            "--matching_dir", str(self.match_dir),
            "--tracks_dir", str(self.tracks_dir),
            "--audit_dir", str(self.audit_dir),
            "--catalog_dir", str(self.catalog_dir),
            "--output_dir", str(self.root / output_name),
            "--nominal_forward_calibration", str(self.calibration_path),
        ]

    def _page(self, output_name):
        return self.root / output_name / "index.html"

    def test_file_backed_assets_render_one_self_contained_page(self):
        rendered = mv.render_page(["<main>sentinel</main>"])
        self.assertIn("<main>sentinel</main>", rendered)
        self.assertIn(".mapcol{position:sticky", rendered)
        self.assertNotIn("<link ", rendered)
        self.assertNotIn("<script src=", rendered)

    def test_the_page_carries_the_shared_document_guarantees(self):
        """This viewer built its own document and was missing all of these.

        The generated mark is the load-bearing one: `indexes.refresh` refuses
        to overwrite a page without it, so an unmarked viewer page is
        indistinguishable from something hand-written that must be preserved.
        """
        rendered = mv.render_page(["<main>sentinel</main>"])
        self.assertTrue(rendered.startswith(page_lib.GENERATED_MARK))
        self.assertIn("<!DOCTYPE html>", rendered)
        self.assertIn('<meta name="viewport"', rendered)
        self.assertIn(f"<title>{mv.PAGE_TITLE}</title>", rendered)
        self.assertIn(mv.GENERATOR, rendered)
        # Its own stylesheet, not the index pages' -- the skeleton is shared,
        # the design is not.
        self.assertNotIn(page_lib.STYLE, rendered)

    def test_no_map_page_renders(self):
        run_main(
            mv, self._viewer_flags("review-no-map") + ["--no_map"])
        page = self._page("review-no-map").read_text()
        self.assertIn("map skipped", page)
        self.assertIn("data-key='object_tracks:tiny_harbor:tracks-v1", page)
        self.assertIn(">T1</h2>", page)
        # Track links use each track's own range name (all ranges loaded,
        # not next(glob)).
        self.assertIn("track_seg0_T1.html", page)
        self.assertIn("preview/index.html#T1", page)
        self.assertNotIn("../../semantic_audit/", page)
        no_match = next(
            key for key in mv.load_uniqueness(self.match_dir)
            if key.endswith("#T2"))
        self.assertIn(f"<tr id='{no_match}'>", page)
        self.assertIn("aggregate confidence", page)
        self.assertNotIn("<th>probability</th>", page)
        self.assertIn("Human match note", page)
        self.assertIn("MATCH_NOTES_CONTEXT", page)
        self.assertIn("/api/match-notes", page)
        self.assertIn("data-note-select=", page)
        matching_digest = json.loads(
            (self.match_dir / "manifest.json").read_text())["content_digest"]
        self.assertIn(
            f'"content_digest":"{matching_digest}"', page)
        self.assertTrue(
            (self._page("review-no-map").parent /
             "identity_review_draft.json").is_file())

    def test_uniqueness_comes_from_complete_canonical_results(self):
        uniqueness = mv.load_uniqueness(self.match_dir)
        by_local = {key.rsplit("#", 1)[-1]: value
                    for key, value in uniqueness.items()}
        self.assertEqual(by_local, {"T1": [5, 5], "T2": [5, 5]})
        self.assertTrue(all("@sha256:" in key for key in uniqueness))

    def test_raw_result_files_cannot_replace_canonical_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "matching"
            shutil.copytree(self.match_dir, copied)
            (copied / llm_lifecycle.CANONICAL_RESULTS_NAME).unlink()
            # Even apparently usable legacy files are not a fallback.
            (copied / "request_meta.json").write_text("{}")
            (copied / "results.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(SystemExit, "canonical matching"):
                mv.load_uniqueness(copied)

    def test_tampered_canonical_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "matching"
            shutil.copytree(self.match_dir, copied)
            canonical_path = copied / llm_lifecycle.CANONICAL_RESULTS_NAME
            canonical_path.write_text(canonical_path.read_text() + "{}\n")
            with self.assertRaisesRegex(SystemExit, "content digest mismatch"):
                mv.load_uniqueness(copied)

    def test_missing_compatibility_is_not_silently_treated_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, "missing matching"):
                mv.load_log_lrs(Path(tmp))

    def test_map_page_draws_bound_catalog_and_approved_calibration(self):
        flags = self._viewer_flags("review-map") + [
            "--epoch_keyframes", "5",
            "--bearing_sigma_deg", "1.0",
            "--landmark_position_sigma_m", "10.0",
            "--gps_course_min_displacement_m", "1.0",
            "--gps_course_smooth_window_s", "5.0",
        ]
        run_main(mv, flags)
        page = self._page("review-map").read_text()
        self.assertIn("MAP_DATA", page)
        self.assertIn("human-approved nominal-forward calibration", page)
        self.assertNotIn("sun_offset_check", page)
        self.assertNotIn("mount_offset_sweep", page)
        payload = json.loads(
            page.split("MAP_DATA=", 1)[1].split(";</script>", 1)[0]
            .replace("<\\/", "</"))
        self.assertEqual(payload["nominal_forward"]["version"], "approved-v1")
        self.assertEqual(
            payload["nominal_forward"]["authority"],
            "human-approved nominal-forward calibration")
        by_local = {key.rsplit("#", 1)[-1]: value
                    for key, value in payload["tracklets"].items()}
        self.assertIn("T1", by_local)
        self.assertTrue(by_local["T1"]["rays"])
        target_ids = [t[4] for t in by_local["T1"]["targets"]]
        self.assertIn("osm:node:101", target_ids)
        lighthouse = next(
            target for target in by_local["T1"]["targets"]
            if target[4] == "osm:node:101")
        # The polygon's complete closed convex hull is embedded, not sampled.
        self.assertEqual(len(lighthouse[10]), 10)
        hull_e = lighthouse[10][0::2]
        hull_n = lighthouse[10][1::2]
        self.assertLessEqual(payload["bounds"][0], min(hull_e))
        self.assertGreaterEqual(payload["bounds"][1], max(hull_e))
        self.assertLessEqual(payload["bounds"][2], min(hull_n))
        self.assertGreaterEqual(payload["bounds"][3], max(hull_n))
        self.assertIn("const hull = g[10] || []", page)
        # Self-contained page: provenance manifest beside it.
        self.assertTrue(
            (self._page("review-map").parent / "manifest.json").exists())

    def test_map_without_fusion_params_is_refused(self):
        with self.assertRaises(SystemExit):
            run_main(mv, self._viewer_flags("review-no-params"))

    def test_wrong_dataset_calibration_is_refused(self):
        wrong = self.root / "wrong-calibration.json"
        document = json.loads(self.calibration_path.read_text())
        document["dataset"] = "another_dataset"
        wrong.write_text(json.dumps(document))
        flags = self._viewer_flags("review-wrong-calibration")
        flags[flags.index("--nominal_forward_calibration") + 1] = str(wrong)
        with self.assertRaisesRegex(SystemExit, "another_dataset"):
            run_main(mv, flags + ["--no_map"])

    def test_missing_settings_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = Path(tmp) / "matching"
            bare.mkdir()
            with self.assertRaises(SystemExit):
                mv.load_settings(bare)

    def test_sparse_frame_ids_use_real_frame_mapping(self):
        frames = [types.SimpleNamespace(frame_idx=10),
                  types.SimpleNamespace(frame_idx=40)]
        lookup = mv.frame_pose_lookup(
            frames, [1.0, 2.0], [3.0, 4.0], [90.0, 180.0])
        self.assertEqual(lookup[40], (2.0, 4.0, 180.0))
        self.assertNotIn(1, lookup)

    def test_signed_circular_rotation_fit_handles_wrap_and_half_turn(self):
        wrapped = mv.circular_fit_deg([179.0, -179.0, 180.0])
        self.assertGreater(wrapped["resultant_length"], 0.99)
        self.assertAlmostEqual(abs(wrapped["rotation_deg"]), 180.0, places=6)
        self.assertLess(wrapped["median_abs_postfit_deg"], 2.0)
        mixed = mv.circular_fit_deg([-80.0, 10.0, 100.0])
        self.assertLess(mixed["resultant_length"], 0.5)
        # A measured 10° ray toward a due-east (90°) row is 10 - 90 = -80°.
        rays = [[0, 0.0, 0.0, 10.0, 0.0, 1.0]]
        self.assertAlmostEqual(
            mv.bearing_residual_deg(rays, 1.0, 0.0), -80.0)


if __name__ == "__main__":
    unittest.main()
