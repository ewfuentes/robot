import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    audit_dataset,
    testing,
)


def run_audit(base: Path):
    return audit_dataset.audit(Path(base).resolve())


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_dataset_passes(self):
        base = testing.make_dataset(self.root / "ds")
        a = run_audit(base)
        self.assertFalse(a.failed,
                         msg="\n".join(f"{k}: {m}" for k, m in a.rows))

    def test_missing_tables_fail(self):
        base = testing.make_dataset(self.root / "ds")
        (base / "intrinsics.csv").unlink()
        self.assertTrue(run_audit(base).failed)

    def test_north_aligned_fails(self):
        meta = testing.default_metadata()
        meta["north_aligned"] = True
        base = testing.make_dataset(self.root / "ds", metadata=meta)
        a = run_audit(base)
        self.assertTrue(a.failed)
        self.assertTrue(any("north_aligned" in m for k, m in a.rows
                            if k == "FAIL"))

    def test_unset_heading_preserves_shape_without_claiming_orientation(self):
        meta = testing.default_metadata()
        base = testing.make_dataset(self.root / "ds", metadata=meta)
        path = base / "intrinsics.csv"
        rows = path.read_text().splitlines()
        path.write_text(rows[0] + "\n" + "\n".join(
            row.replace(",45.0,column_0,", ",,,") for row in rows[1:])
            + "\n")
        a = run_audit(base)
        self.assertFalse(a.failed,
                         msg="\n".join(f"{k}: {m}" for k, m in a.rows))
        self.assertTrue(any("no camera/world orientation" in m
                            for _, m in a.rows))

    def test_pano_gap_warns_about_index_divergence(self):
        base = testing.make_dataset(self.root / "ds", n_frames=5,
                                    skip_pano_numbers=(2,))
        a = run_audit(base)
        self.assertTrue(any("frame_index_by_pano_id" in m for k, m in a.rows
                            if k == "warn"))

    def test_legacy_landmarks_dir_warns(self):
        base = testing.make_dataset(self.root / "ds")
        (base / "landmarks").mkdir()
        a = run_audit(base)
        self.assertTrue(any("artifacts/catalogs" in m for k, m in a.rows
                            if k == "warn"))

    def test_main_errors_on_nonexistent_path(self):
        # The old tool silently dropped bad paths and reported "0/0 clean"
        # with exit 0; a typo must be an error.
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, \
                contextlib.redirect_stderr(stderr):
            audit_dataset.main([str(self.root / "does_not_exist")])
        self.assertNotEqual(ctx.exception.code, 0)
        self.assertIn("not a directory", stderr.getvalue())

    def test_main_exit_codes(self):
        clean = testing.make_dataset(self.root / "clean")
        meta = testing.default_metadata()
        meta["north_aligned"] = True
        broken = testing.make_dataset(self.root / "broken", metadata=meta)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_dataset.main([str(clean)]), 0)
            self.assertEqual(audit_dataset.main([str(broken)]), 1)
            self.assertEqual(audit_dataset.main([str(clean), str(broken)]), 1)


if __name__ == "__main__":
    unittest.main()
