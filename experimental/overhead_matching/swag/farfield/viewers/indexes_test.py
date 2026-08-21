import json
import tempfile
import unittest
from pathlib import Path

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
    (run / "viewer.html").write_text("<html></html>")
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
        self.assertIn(pg.GENERATED_MARK, root_page)
        self.assertNotIn(str(self.root), root_page)  # no absolute paths
        exp = (self.root / "runs" / "260820_extent_sigma"
               / "index.html").read_text()
        self.assertIn("Does extent-aware sigma fix pohang?", exp)
        self.assertIn('href="boston_r001/viewer.html"', exp)
        kind = (self.root / "artifacts" / "object_tracks"
                / "index.html").read_text()
        self.assertIn("farfield.tracking.run_tracking", kind)

    def test_never_clobbers_a_page_it_did_not_generate(self):
        stage_page = self.root / "runs" / "260820_extent_sigma" / "index.html"
        stage_page.write_text("<html>stage-owned</html>")
        result = indexes.refresh(self.root)
        self.assertEqual(stage_page.read_text(), "<html>stage-owned</html>")
        self.assertIn(str(stage_page), result["skipped"])

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
