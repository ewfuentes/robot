import contextlib
import csv
import io
import json
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
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames
            rows = list(reader)
        for row in rows:
            for field in (
                    "computed_compass_angle_true_deg",
                    "compass_angle_true_deg",
                    "heading_optical_axis_true_deg",
                    "heading_column0_true_deg"):
                row[field] = ""
            row["selected_heading_source"] = ""
        with path.open("w", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
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

    def test_in_dataset_landmarks_dir_fails(self):
        base = testing.make_dataset(self.root / "ds")
        (base / "landmarks").mkdir()
        a = run_audit(base)
        self.assertTrue(a.failed)
        self.assertTrue(any("artifacts/catalogs" in m for k, m in a.rows
                            if k == "FAIL"))

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

    def test_corrupt_fixture_classes_fail_without_tracebacks(self):
        def invalid_json(base):
            (base / "pipeline_metadata.json").write_text("{")

        def non_object_metadata(base):
            (base / "pipeline_metadata.json").write_text("[]")

        def empty_panoramas(base):
            for path in (base / "panorama").glob("*.jpg"):
                path.unlink()

        def empty_tables(base):
            for name in ("frames_gps.csv", "pano_id_mapping.csv",
                         "extraction_log.csv", "intrinsics.csv"):
                path = base / name
                path.write_text(path.read_text().splitlines()[0] + "\n")

        def malformed_filename(base):
            path = next((base / "panorama").glob("*.jpg"))
            path.rename(path.with_name("broken.jpg"))

        def missing_csv_column(base):
            path = base / "frames_gps.csv"
            rows = path.read_text().splitlines()
            rows[0] = rows[0].replace(",dist_m", "")
            path.write_text("\n".join(rows) + "\n")

        def invalid_numeric(base):
            path = base / "frames_gps.csv"
            rows = path.read_text().splitlines()
            fields = rows[1].split(",")
            fields[3] = "not-a-distance"
            rows[1] = ",".join(fields)
            path.write_text("\n".join(rows) + "\n")

        def row_count_disagreement(base):
            path = base / "intrinsics.csv"
            rows = path.read_text().splitlines()
            path.write_text("\n".join(rows[:-1]) + "\n")

        def duplicate_gps_id(base):
            path = base / "frames_gps.csv"
            rows = path.read_text().splitlines()
            second = rows[2].split(",")
            second[0] = "0"
            rows[2] = ",".join(second)
            path.write_text("\n".join(rows) + "\n")

        def invalid_metadata_types(base):
            path = base / "pipeline_metadata.json"
            metadata = json.loads(path.read_text())
            metadata["north_aligned"] = 0
            metadata["azimuth_convention"] = []
            path.write_text(json.dumps(metadata))

        cases = (
            invalid_json, non_object_metadata, empty_panoramas, empty_tables,
            malformed_filename, missing_csv_column, invalid_numeric,
            row_count_disagreement, duplicate_gps_id, invalid_metadata_types,
        )
        for index, corrupt in enumerate(cases):
            with self.subTest(corruption=corrupt.__name__):
                base = testing.make_dataset(self.root / f"corrupt_{index}")
                corrupt(base)
                audit = run_audit(base)
                self.assertTrue(audit.failed)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(audit_dataset.main([str(base)]), 1)


if __name__ == "__main__":
    unittest.main()
