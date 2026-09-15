import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.localization import (
    side_outputs,
)


class SideOutputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run-001"
        self.run_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_is_a_deterministic_sibling(self):
        self.assertEqual(
            side_outputs.default_directory(self.run_dir, ".viewer"),
            self.root / "run-001.viewer")
        self.assertEqual(
            side_outputs.default_directory(self.run_dir, ".plots"),
            self.root / "run-001.plots")

    def test_success_is_published_once_and_never_merged(self):
        with side_outputs.publish_directory(
                self.run_dir, output_dir=None, suffix=".viewer") as output:
            (output.staging_dir / "viewer.html").write_text("complete")
            self.assertFalse(output.destination.exists())
        self.assertEqual(
            (self.root / "run-001.viewer" / "viewer.html").read_text(),
            "complete")
        self.assertFalse((self.root / "run-001.viewer.incomplete").exists())
        with self.assertRaisesRegex(
                side_outputs.SideOutputError, "already exists"):
            with side_outputs.publish_directory(
                    self.run_dir, output_dir=None, suffix=".viewer"):
                pass

    def test_failure_is_never_visible_as_complete(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with side_outputs.publish_directory(
                    self.run_dir, output_dir=None, suffix=".plots") as output:
                (output.staging_dir / "map.png").write_bytes(b"partial")
                raise RuntimeError("interrupted")
        self.assertFalse((self.root / "run-001.plots").exists())
        self.assertTrue((self.root / "run-001.plots.incomplete").is_dir())

    def test_empty_publication_is_rejected(self):
        with self.assertRaisesRegex(
                side_outputs.SideOutputError, "empty side-output"):
            with side_outputs.publish_directory(
                    self.run_dir, output_dir=None, suffix=".viewer"):
                pass
        self.assertFalse((self.root / "run-001.viewer").exists())

    def test_run_and_descendants_are_rejected(self):
        for destination in (
                self.run_dir,
                self.run_dir / "viewer",
                self.run_dir / "nested" / ".." / "viewer"):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(
                        side_outputs.SideOutputError, "immutable run"):
                    with side_outputs.publish_directory(
                            self.run_dir, output_dir=destination,
                            suffix=".viewer"):
                        pass

    def test_symlinked_run_output_and_parent_are_rejected(self):
        run_link = self.root / "run-link"
        run_link.symlink_to(self.run_dir, target_is_directory=True)
        output_link = self.root / "output-link"
        output_link.symlink_to(self.root / "missing", target_is_directory=True)
        parent_link = self.root / "parent-link"
        parent_link.symlink_to(self.root, target_is_directory=True)

        cases = (
            (run_link, None),
            (self.run_dir, output_link),
            (self.run_dir, parent_link / "viewer"),
        )
        for run_dir, output_dir in cases:
            with self.subTest(run_dir=run_dir, output_dir=output_dir):
                with self.assertRaisesRegex(
                        side_outputs.SideOutputError, "symlink"):
                    with side_outputs.publish_directory(
                            run_dir, output_dir=output_dir, suffix=".viewer"):
                        pass

    def test_content_symlink_is_not_publishable(self):
        target = self.root / "target.html"
        target.write_text("outside")
        with self.assertRaisesRegex(
                side_outputs.SideOutputError, "content cannot contain"):
            with side_outputs.publish_directory(
                    self.run_dir, output_dir=None,
                    suffix=".viewer") as output:
                (output.staging_dir / "viewer.html").symlink_to(target)
        self.assertFalse((self.root / "run-001.viewer").exists())


if __name__ == "__main__":
    unittest.main()
