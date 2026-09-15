import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    retire_versions,
)


class RetireVersionsTest(unittest.TestCase):

    def test_moves_everything_but_kept_versions_and_their_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            lane = Path(tmp) / "artifacts" / "catalogs" / "ds"
            for name in ("v1", "v2", "v2.llm-work", "v2--tracks-x--viewer-y",
                         "v3", "index.html"):
                (lane / name).mkdir(parents=True)
            (lane / "index.html").rmdir()
            (lane / "index.html").write_text("x")

            moved = retire_versions.retire(
                lane, keep={"v2"}, replaced_by="v2", reason="test")

            self.assertEqual(moved, ["v1", "v3"])
            self.assertTrue((lane / "v2").is_dir())
            self.assertTrue((lane / "v2.llm-work").is_dir())
            self.assertTrue((lane / "v2--tracks-x--viewer-y").is_dir())
            self.assertTrue((lane / "retired" / "v1").is_dir())
            self.assertTrue((lane / "retired" / "v3").is_dir())
            ledger = (lane / "retired" / "RETIRED.md").read_text()
            self.assertEqual(ledger.count("| `v2` | test |"), 2)
            # Idempotent: a second pass finds nothing live to move.
            self.assertEqual(retire_versions.retire(
                lane, keep={"v2"}, replaced_by="v2", reason="test"), [])
            with self.assertRaises(SystemExit):
                retire_versions.retire(
                    lane, keep={"missing"}, replaced_by="x", reason="y")


if __name__ == "__main__":
    unittest.main()
