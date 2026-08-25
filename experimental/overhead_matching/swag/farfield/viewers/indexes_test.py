import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield.viewers import (
    indexes,
    page as pg,
)


def make_tree(root: Path):
    (root / "datasets" / "boston_harbor_leg1" / "panorama").mkdir(
        parents=True)
    (root / "datasets" / "boston_harbor_leg1" / "panorama"
     / "f0000,42.0,-71.0,.jpg").write_bytes(b"")
    (root / "datasets" / "boston_harbor_leg1"
     / "trajectory.png").write_bytes(b"")
    version = root / "artifacts" / "object_tracks" / "boston_harbor_leg1" / "v1"
    version.mkdir(parents=True)
    (version / "manifest.json").write_text(json.dumps(
        {"generator": "farfield.tracking.run_tracking",
         "created": "2026-08-20"}))
    run = root / "runs" / "260820_extent_sigma" / "boston_r001"
    run.mkdir(parents=True)
    viewer = run.with_name(run.name + ".viewer")
    viewer.mkdir()
    (viewer / "viewer.html").write_text("<html></html>")
    plots = run.with_name(run.name + ".plots")
    plots.mkdir()
    (plots / "map.png").write_bytes(b"png")
    satellite = run.with_name(run.name + ".satellite")
    satellite.mkdir()
    (satellite / "satellite.json").write_text("{}")
    (root / "runs" / "260820_extent_sigma" / "experiment.md").write_text(
        "# Extent sigma\nDoes extent-aware sigma fix pohang?\n- yes\n")
    return root


class RefreshTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_tree(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_the_full_chain(self):
        result = indexes.refresh(self.root)
        for rel in ("index.html", "datasets/index.html",
                    "artifacts/index.html",
                    "artifacts/object_tracks/index.html",
                    "runs/index.html",
                    "runs/260820_extent_sigma/index.html"):
            self.assertTrue((self.root / rel).exists(), rel)
            self.assertIn(str(self.root / rel), result["written"])
        self.assertEqual(result["skipped"], [])

    def test_pages_are_relative_and_generated_marked(self):
        indexes.refresh(self.root)
        root_page = (self.root / "index.html").read_text()
        self.assertEqual(root_page.splitlines()[0],
                         indexes.INDEX_GENERATED_MARK)
        self.assertIn(pg.GENERATED_MARK, root_page)
        self.assertNotIn(str(self.root), root_page)  # no absolute paths
        exp = (self.root / "runs" / "260820_extent_sigma"
               / "index.html").read_text()
        self.assertIn("Does extent-aware sigma fix pohang?", exp)
        self.assertIn('href="boston_r001.viewer/viewer.html"', exp)
        self.assertIn('href="boston_r001.plots/map.png"', exp)
        self.assertEqual(
            [path.name for path in indexes._run_dirs(
                self.root / "runs" / "260820_extent_sigma")],
            ["boston_r001"])
        kind = (self.root / "artifacts" / "object_tracks"
                / "index.html").read_text()
        self.assertIn("farfield.tracking.run_tracking", kind)

    def test_track_index_links_the_exact_frame_viewer_sidecar(self):
        frame_source = (self.root / "artifacts" / "frame_landmarks"
                        / "boston_harbor_leg1" / "frames-v1")
        frame_source.mkdir(parents=True)
        (frame_source / "manifest.json").write_text(json.dumps({
            "generator": "frame producer",
            "created": "2026-08-20T12:00:00Z",
        }))
        viewer = (self.root / "artifacts" / "frame_landmarks"
                  / "boston_harbor_leg1" / "viewer-v1")
        (viewer / "tracks").mkdir(parents=True)
        (viewer / "index.html").write_text("frame index")
        (viewer / "tracks" / "index.html").write_text("track index")
        (viewer / "manifest.json").write_text(json.dumps({
            "kind": indexes.FRAME_VIEWER_KIND,
            "created": "2026-08-25T12:00:00Z",
            "config": {"schema": indexes.FRAME_VIEWER_SCHEMA},
            "upstreams": [{
                "kind": "object_tracks",
                "dataset": "boston_harbor_leg1",
                "version": "v1",
            }, {
                "kind": "frame_landmarks",
                "dataset": "boston_harbor_leg1",
                "version": "frames-v1",
            }],
        }))
        indexes.refresh(self.root)
        page = (self.root / "artifacts" / "object_tracks"
                / "index.html").read_text()
        self.assertIn("frames ↔ tracks", page)
        self.assertIn(
            '../frame_landmarks/boston_harbor_leg1/viewer-v1/'
            'tracks/index.html', page)
        self.assertNotIn(
            'href="boston_harbor_leg1/v1/index.html"', page)
        frames_page = (self.root / "artifacts" / "frame_landmarks"
                       / "index.html").read_text()
        self.assertIn("keyframes", frames_page)
        self.assertIn(
            'href="boston_harbor_leg1/viewer-v1/index.html">frames-v1',
            frames_page)
        self.assertNotIn(">viewer-v1</a>", frames_page)

    def test_loose_legacy_artifact_files_are_not_indexed(self):
        dataset_dir = (
            self.root / "artifacts" / "object_tracks"
            / "boston_harbor_leg1")
        (dataset_dir / "legacy.json").write_text("{}")
        (dataset_dir / "legacy.feather").write_bytes(b"legacy")
        indexes.refresh(self.root)
        page = (dataset_dir.parent / "index.html").read_text()
        self.assertNotIn("legacy.json", page)
        self.assertNotIn("legacy.feather", page)

    def test_never_clobbers_a_page_it_did_not_generate(self):
        stage_page = self.root / "runs" / "260820_extent_sigma" / "index.html"
        stage_page.write_text("<html>stage-owned</html>")
        result = indexes.refresh(self.root)
        self.assertEqual(stage_page.read_text(), "<html>stage-owned</html>")
        self.assertIn(str(stage_page), result["skipped"])

    def test_ignores_diagnostics_written_inside_an_immutable_run(self):
        experiment = self.root / "runs" / "260820_extent_sigma"
        run = experiment / "boston_r002"
        run.mkdir()
        (run / "viewer.html").write_text("stale")
        (run / "plots").mkdir()
        (run / "plots" / "map.png").write_bytes(b"stale")
        incomplete = run.with_name(run.name + ".viewer.incomplete")
        incomplete.mkdir()
        (incomplete / "viewer.html").write_text("partial")

        indexes.refresh(self.root)
        page = (experiment / "index.html").read_text()
        self.assertIn("boston_r002", page)
        self.assertNotIn('href="boston_r002/viewer.html"', page)
        self.assertNotIn('href="boston_r002/plots/map.png"', page)
        self.assertNotIn("boston_r002.viewer.incomplete", page)

    def test_refuses_symlinked_sibling_page(self):
        experiment = self.root / "runs" / "260820_extent_sigma"
        viewer = experiment / "boston_r001.viewer"
        (viewer / "viewer.html").unlink()
        target = Path(self.tmp.name) / "outside.html"
        target.write_text("outside")
        (viewer / "viewer.html").symlink_to(target)

        with self.assertRaisesRegex(indexes.IndexRefreshError,
                                    "refusing symlink page"):
            indexes.refresh(self.root)

    def test_generic_viewer_marker_does_not_grant_index_ownership(self):
        stage_page = self.root / "datasets" / "index.html"
        stage_html = f"{pg.GENERATED_MARK}\n<html>stage-owned</html>"
        stage_page.write_text(stage_html)
        result = indexes.refresh(self.root)
        self.assertEqual(stage_page.read_text(), stage_html)
        self.assertIn(str(stage_page), result["skipped"])

    def test_interrupted_replacement_preserves_owned_page(self):
        indexes.refresh(self.root)
        target = self.root / "index.html"
        original = target.read_bytes()
        replacement = (f"{indexes.INDEX_GENERATED_MARK}\n"
                       "<html>replacement</html>")
        with mock.patch.object(indexes.artifact_lib.os, "replace",
                               side_effect=OSError("simulated interruption")):
            with self.assertRaisesRegex(OSError, "simulated interruption"):
                indexes._write_index(self.root, replacement, [])
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".index.html.*.tmp")), [])

    def test_concurrent_refreshes_are_serialized(self):
        original_write = indexes._write_index
        start = threading.Barrier(3)
        state_lock = threading.Lock()
        active_writes = 0
        peak_writes = 0
        errors = []

        def slow_write(*args, **kwargs):
            nonlocal active_writes, peak_writes
            with state_lock:
                active_writes += 1
                peak_writes = max(peak_writes, active_writes)
            try:
                time.sleep(0.02)
                return original_write(*args, **kwargs)
            finally:
                with state_lock:
                    active_writes -= 1

        def worker():
            start.wait()
            try:
                indexes.refresh(self.root)
            except BaseException as exc:  # Preserve worker failures for main.
                errors.append(exc)

        with mock.patch.object(indexes, "_write_index", side_effect=slow_write):
            workers = [threading.Thread(target=worker) for _ in range(2)]
            for worker_thread in workers:
                worker_thread.start()
            start.wait()
            for worker_thread in workers:
                worker_thread.join(timeout=10)

        self.assertFalse(any(worker_thread.is_alive()
                             for worker_thread in workers))
        self.assertEqual(errors, [])
        self.assertEqual(peak_writes, 1)
        self.assertEqual((self.root / "index.html").read_text().splitlines()[0],
                         indexes.INDEX_GENERATED_MARK)

    def test_refuses_symlink_root_and_nested_directories(self):
        root_link = Path(self.tmp.name) / "root-link"
        root_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(indexes.IndexRefreshError,
                                    "refusing symlink directory"):
            indexes.refresh(root_link)

        linked_target = Path(self.tmp.name) / "linked-target"
        linked_target.mkdir()
        (self.root / "datasets" / "linked").symlink_to(
            linked_target, target_is_directory=True)
        with self.assertRaisesRegex(indexes.IndexRefreshError,
                                    "refusing symlink directory"):
            indexes.refresh(self.root)

    def test_accepts_relative_in_dataset_panorama_symlink(self):
        dataset = self.root / "datasets" / "symlinked-panorama"
        frames = dataset / "frames"
        frames.mkdir(parents=True)
        (frames / "f0000,42.0,-71.0,.jpg").write_bytes(b"jpeg")
        (dataset / "panorama").symlink_to(
            "frames", target_is_directory=True)
        indexes.refresh(self.root)
        page = (self.root / "datasets" / "index.html").read_text()
        self.assertIn("symlinked-panorama", page)

    def test_refuses_unsafe_panorama_symlinks(self):
        dataset = self.root / "datasets" / "symlinked-panorama"
        dataset.mkdir()
        outside = Path(self.tmp.name) / "panorama-target"
        outside.mkdir()
        (dataset / "alternate").mkdir()
        panorama = dataset / "panorama"
        for target, message in (
                (outside, "absolute panorama symlink"),
                (Path("frames"), "dangling panorama symlink"),
                (Path("alternate"), "target must be exactly 'frames'"),
                (Path("../../panorama-target"),
                 "target must be exactly 'frames'")):
            with self.subTest(target=target):
                panorama.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(indexes.IndexRefreshError,
                                            message):
                    indexes.refresh(self.root)
                panorama.unlink()

    def test_refuses_frames_that_is_itself_a_symlink(self):
        dataset = self.root / "datasets" / "symlinked-frames"
        actual = dataset / "actual-frames"
        actual.mkdir(parents=True)
        (dataset / "frames").symlink_to("actual-frames", target_is_directory=True)
        (dataset / "panorama").symlink_to("frames", target_is_directory=True)

        with self.assertRaisesRegex(
                indexes.IndexRefreshError,
                "real in-dataset frames directory"):
            indexes.refresh(self.root)

    def test_dataset_collections_are_browsable_but_not_dataset_rows(self):
        collection = self.root / "datasets" / "unvetted"
        frames = collection / "london_thames" / "frames"
        frames.mkdir(parents=True)
        (frames / "f0000,51.0,-0.1,.jpg").write_bytes(b"jpeg")
        (frames.parent / "panorama").symlink_to(
            "frames", target_is_directory=True)
        result = indexes.refresh(self.root)
        collection_page = collection / "index.html"
        self.assertIn(str(collection_page), result["written"])
        datasets_page = (self.root / "datasets" / "index.html").read_text()
        self.assertIn("collections", datasets_page)
        self.assertIn('href="unvetted/index.html"', datasets_page)
        self.assertNotIn("london_thames", datasets_page)
        self.assertIn("london_thames", collection_page.read_text())

    def test_refuses_a_non_directory_root(self):
        not_a_root = Path(self.tmp.name) / "not-a-root"
        not_a_root.write_text("not a directory")
        with self.assertRaisesRegex(indexes.IndexRefreshError,
                                    "expected directory"):
            indexes.refresh(not_a_root)

    def test_refresh_is_idempotent_and_tracks_new_content(self):
        indexes.refresh(self.root)
        # A new experiment appears on the next refresh with no code change.
        new_exp = self.root / "runs" / "260821_near_gate"
        new_exp.mkdir()
        indexes.refresh(self.root)
        runs_page = (self.root / "runs" / "index.html").read_text()
        self.assertIn("260821_near_gate", runs_page)
        # Missing experiment.md is flagged, not fatal.
        exp_page = (new_exp / "index.html").read_text()
        self.assertIn("no experiment.md", exp_page)


class MarkdownLiteTest(unittest.TestCase):
    def test_headers_lists_paragraphs_code(self):
        html = pg.render_markdown_lite(
            "# Title\ntext line\n\n- a\n- b\n```\ncode <x>\n```\n")
        self.assertIn("<h2>Title</h2>", html)
        self.assertIn("<p>text line</p>", html)
        self.assertIn("<li>a</li>", html)
        self.assertIn("code &lt;x&gt;", html)

    def test_escapes_html(self):
        self.assertIn("&lt;script&gt;",
                      pg.render_markdown_lite("<script>alert(1)</script>"))


if __name__ == "__main__":
    unittest.main()
