"""The record must stand alone, and the change signal must be about science.

Two properties carry the design. The record travels with its diff, because a
commit hash is a pointer into a mutable store and pointers rot. And the
"did the code change?" signal ignores presentation, because `publication`
refreshes the index on every publish -- which is wanted -- and that puts the
viewer modules in every producing stage's import closure.
"""

import hashlib
import unittest

from experimental.overhead_matching.swag.farfield import (
    code_provenance as cp,
)

DIFF_TRACKER = (
    "diff --git a/experimental/overhead_matching/swag/farfield/tracking/"
    "track_builder.py b/experimental/overhead_matching/swag/farfield/tracking/"
    "track_builder.py\n"
    "@@ -1 +1 @@\n-old\n+new\n")
DIFF_VIEWER_HTML = (
    "diff --git a/experimental/overhead_matching/swag/farfield/matching/"
    "match_viewer_assets/style.css b/experimental/overhead_matching/swag/"
    "farfield/matching/match_viewer_assets/style.css\n"
    "@@ -1 +1 @@\n-.a{color:red}\n+.a{color:blue}\n")
DIFF_VIEWER_PY = (
    "diff --git a/experimental/overhead_matching/swag/farfield/viewers/"
    "indexes.py b/experimental/overhead_matching/swag/farfield/viewers/"
    "indexes.py\n"
    "@@ -1 +1 @@\n-old\n+new\n")


def block(commit, diff):
    computational = cp.computational_diff(diff)
    return {
        "schema": cp.SCHEMA,
        "commit": commit,
        "diff": diff,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "computational_diff_sha256": hashlib.sha256(
            computational.encode()).hexdigest(),
        "dirty": bool(diff),
        "computationally_dirty": bool(computational),
    }


class PresentationTest(unittest.TestCase):
    def test_html_css_and_js_are_presentation(self):
        for path in ("a/page.html", "a/style.css", "a/app.js"):
            self.assertTrue(cp.is_presentation(path), path)

    def test_viewer_modules_are_presentation(self):
        """The case that forced this. `publication` refreshes the index on
        every publish, so `viewers/indexes.py` is in all eight producing
        closures -- and no edit to it can alter `predictions.jsonl`."""
        for path in ("experimental/.../farfield/viewers/indexes.py",
                     "experimental/.../farfield/viewers/page.py",
                     "experimental/.../farfield/tracking/keyframe_viewer.py",
                     "experimental/.../farfield/matching/"
                     "match_viewer_assets/app.js"):
            self.assertTrue(cp.is_presentation(path), path)

    def test_science_modules_are_not(self):
        for path in ("experimental/.../farfield/tracking/track_builder.py",
                     "experimental/.../farfield/localization/filter.py",
                     "experimental/.../farfield/geometry.py"):
            self.assertFalse(cp.is_presentation(path), path)


class DiffSplitTest(unittest.TestCase):
    def test_a_multi_file_diff_splits_by_path(self):
        paths = [path for path, _ in
                 cp.split_diff(DIFF_TRACKER + DIFF_VIEWER_HTML)]
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].endswith("track_builder.py"))
        self.assertTrue(paths[1].endswith("style.css"))

    def test_presentation_is_removed_from_the_computational_diff(self):
        computational = cp.computational_diff(
            DIFF_TRACKER + DIFF_VIEWER_HTML + DIFF_VIEWER_PY)
        self.assertIn("track_builder.py", computational)
        self.assertNotIn("style.css", computational)
        self.assertNotIn("indexes.py", computational)

    def test_a_presentation_only_diff_is_computationally_empty(self):
        self.assertEqual(
            cp.computational_diff(DIFF_VIEWER_HTML + DIFF_VIEWER_PY), "")

    def test_an_empty_diff_is_handled(self):
        self.assertEqual(cp.computational_diff(None), "")
        self.assertEqual(cp.computational_diff(""), "")


class RecordTest(unittest.TestCase):
    def test_a_record_is_always_well_formed(self):
        """Outside `bazel run` there is no stamp. Recording "unknown" is a
        truthful account of not knowing -- which is the point of writing it
        down rather than gating on it."""
        value = cp.validate(cp.record())
        self.assertEqual(value["schema"], cp.SCHEMA)
        self.assertIsInstance(value["commit"], str)
        self.assertIn("diff", value)
        self.assertEqual(len(value["computational_diff_sha256"]), 64)

    def test_the_diff_travels_with_the_record(self):
        """A commit hash is a pointer into a mutable store. The commits
        stamped into the artifacts on disk today survive through one hand-made
        safety branch, because a force-push orphaned them."""
        value = block("c0ffee", DIFF_TRACKER)
        self.assertIn("track_builder.py", value["diff"])

    def test_a_malformed_block_is_refused(self):
        for bad in ({}, {"schema": "other/v1"}, "not a mapping",
                    {"schema": cp.SCHEMA, "commit": "x"}):
            with self.assertRaises(cp.CodeProvenanceError):
                cp.validate(bad)


class DiffersTest(unittest.TestCase):
    def test_the_same_state_does_not_differ(self):
        self.assertFalse(
            cp.differs(block("c0ffee", ""), block("c0ffee", "")))

    def test_a_different_commit_differs(self):
        self.assertTrue(
            cp.differs(block("c0ffee", ""), block("deadbe", "")))

    def test_a_science_edit_differs(self):
        self.assertTrue(
            cp.differs(block("c0ffee", ""), block("c0ffee", DIFF_TRACKER)))

    def test_a_presentation_edit_does_not(self):
        """A restyle is not a change to the tracker, and saying so was the
        whole reason to filter the diff."""
        self.assertFalse(
            cp.differs(block("c0ffee", ""),
                       block("c0ffee", DIFF_VIEWER_HTML + DIFF_VIEWER_PY)))

    def test_an_unknown_commit_always_differs(self):
        """Not knowing is not the same as matching."""
        self.assertTrue(
            cp.differs(block(cp.UNKNOWN, ""), block(cp.UNKNOWN, "")))


class LineageTest(unittest.TestCase):
    def test_one_code_state_reads_as_consistent(self):
        summary = cp.lineage_summary(
            [block("c0ffee", ""), block("c0ffee", "")])
        self.assertFalse(summary["code_differs"])
        self.assertIn("one code state", cp.describe(summary))

    def test_a_mixed_lineage_is_flagged(self):
        """The failure this design accepts: an evaluation table with one leg
        from before a fix and one from after. No gate will stop it now, so it
        has to be on screen."""
        summary = cp.lineage_summary(
            [block("c0ffee", ""), block("deadbe", "")])
        self.assertTrue(summary["code_differs"])
        self.assertIn("CODE DIFFERS", cp.describe(summary))

    def test_a_presentation_only_spread_is_not_flagged(self):
        summary = cp.lineage_summary(
            [block("c0ffee", ""), block("c0ffee", DIFF_VIEWER_HTML)])
        self.assertFalse(summary["code_differs"])

    def test_an_uncommitted_science_change_is_flagged(self):
        summary = cp.lineage_summary(
            [block("c0ffee", ""), block("c0ffee", DIFF_TRACKER)])
        self.assertTrue(summary["code_differs"])
        self.assertTrue(summary["any_dirty"])

    def test_an_unknown_commit_in_the_lineage_is_flagged(self):
        summary = cp.lineage_summary(
            [block("c0ffee", ""), block(cp.UNKNOWN, "")])
        self.assertTrue(summary["code_differs"])
        self.assertIn("recorded no commit", cp.describe(summary))

    def test_an_empty_lineage_is_not_an_error(self):
        summary = cp.lineage_summary([])
        self.assertFalse(summary["code_differs"])
        self.assertIn("no artifacts", cp.describe(summary))


if __name__ == "__main__":
    unittest.main()
