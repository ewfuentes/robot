import json
from pathlib import Path
import tempfile
import unittest

from experimental.overhead_matching.swag.farfield.localization import (
    review_assets,
)


class ReviewAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "runs" / "target" / "run-v1"
        self.inputs = self.root / "artifacts" / "localization_inputs" / "li-v1"
        self.matching = self.root / "artifacts" / "landmark_matches" / "m-v1"
        self.tracks = self.root / "artifacts" / "object_tracks" / "t-v1"
        self.audits = self.root / "artifacts" / "semantic_audits" / "a-v1"
        self.catalog = self.root / "artifacts" / "catalogs" / "c-v1"
        for directory in (self.run, self.inputs, self.matching, self.tracks,
                          self.audits, self.catalog):
            directory.mkdir(parents=True)
        self.write_json(self.run / "manifest.json", {
            "kind": "localization_run",
            "dataset": "pohang",
            "upstreams": [{
                "kind": "localization_inputs",
                "path": str(self.inputs),
            }],
        })
        self.write_json(self.inputs / "manifest.json", {
            "kind": "localization_inputs",
            "dataset": "pohang",
            "upstreams": [{
                "kind": "landmark_matches",
                "path": str(self.matching),
            }],
        })

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write_json(path: Path, value: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def make_review(self, name: str, *, matching: Path | None = None,
                    audit_page: Path | None = None) -> Path:
        review = self.root / "runs" / "source" / f"{name}.matcher-review"
        review.mkdir(parents=True)
        (review / "index.html").write_text("<html>matcher</html>")
        self.write_json(review / "manifest.json", {
            "generator": review_assets.MATCHER_GENERATOR,
            "inputs": {
                "matching": str(matching or self.matching),
                "tracks": str(self.tracks),
                "semantic_audits": str(self.audits),
                "catalog": str(self.catalog),
                "semantic_audit_review": (
                    str(audit_page) if audit_page is not None else ""),
            },
        })
        return review

    def discover(self):
        return review_assets.discover(
            self.run, tracks_dir=self.tracks, audit_dir=self.audits,
            catalog_dir=self.catalog)

    def test_reuses_exact_matcher_and_its_companion_audit_page(self):
        audit_page = (self.root / "runs" / "source" / "run.audit-review"
                      / "index.html")
        audit_page.parent.mkdir(parents=True)
        audit_page.write_text("<html>audit</html>")
        review = self.make_review("run", audit_page=audit_page)

        pages = self.discover()

        self.assertEqual(pages.matcher, review / "index.html")
        self.assertEqual(pages.audit, audit_page)

    def test_rejects_review_for_a_different_matching_ancestor(self):
        other = self.root / "artifacts" / "landmark_matches" / "m-v2"
        other.mkdir(parents=True)
        self.make_review("run", matching=other)

        self.assertEqual(self.discover(), review_assets.ReviewPages())

    def test_rejects_any_other_scientific_input(self):
        review = self.make_review("run")
        manifest = json.loads((review / "manifest.json").read_text())
        manifest["inputs"]["tracks"] = str(
            self.root / "artifacts" / "object_tracks" / "different")
        self.write_json(review / "manifest.json", manifest)

        self.assertEqual(self.discover(), review_assets.ReviewPages())

    def test_current_run_sibling_wins_over_other_compatible_pages(self):
        self.make_review("aaa")
        sibling = self.run.with_name(self.run.name + ".matcher-review")
        sibling.mkdir()
        (sibling / "index.html").write_text("<html>sibling</html>")
        self.write_json(sibling / "manifest.json", {
            "generator": review_assets.MATCHER_GENERATOR,
            "inputs": {
                "matching": str(self.matching),
                "tracks": str(self.tracks),
                "semantic_audits": str(self.audits),
                "catalog": str(self.catalog),
                "semantic_audit_review": "",
            },
        })

        self.assertEqual(self.discover().matcher, sibling / "index.html")

    def test_missing_companion_audit_does_not_hide_matcher(self):
        review = self.make_review(
            "run", audit_page=self.root / "missing" / "index.html")

        pages = self.discover()

        self.assertEqual(pages.matcher, review / "index.html")
        self.assertIsNone(pages.audit)


if __name__ == "__main__":
    unittest.main()
