import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.overhead_matching.swag.farfield import dataset as ds_lib
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    checksums,
    ingest_selfcollect as ingest,
)

PANO_W = 512


class FillCourseFromTrackTest(unittest.TestCase):
    def test_course_follows_the_track(self):
        rows = [{"latitude": "42.0", "longitude": "-71.0"},
                {"latitude": "42.0", "longitude": "-70.99"},
                {"latitude": "42.0", "longitude": "-70.98"}]
        ingest.fill_gps_course_from_track(rows)
        for row in rows:
            self.assertAlmostEqual(float(row["gps_course_deg"]), 90.0,
                                   delta=1.0)

    def test_northbound_track(self):
        rows = [{"latitude": "42.00", "longitude": "-71.0"},
                {"latitude": "42.01", "longitude": "-71.0"}]
        ingest.fill_gps_course_from_track(rows)
        self.assertAlmostEqual(float(rows[0]["gps_course_deg"]), 0.0,
                               delta=1.0)


def write_source(root: Path, n=4, with_course=True) -> tuple:
    """A minimal self-collect source tree: frames + a GPS csv."""
    from PIL import Image

    frames = root / "frames"
    frames.mkdir(parents=True)
    rows = []
    for i in range(n):
        name = f"img_{i:04d}.jpg"
        Image.new("RGB", (PANO_W, PANO_W // 2), (10 * i, 60, 90)).save(
            frames / name)
        row = {"frame_file": name,
               "latitude": f"{42.35 + 1e-4 * i:.7f}",
               "longitude": f"{-71.05 + 1e-4 * i:.7f}",
               "video_t_s": f"{2.0 * i:.2f}",
               "dist_m": f"{10.0 * i:.1f}"}
        if with_course:
            # A deliberately blank course on one row: the crash case.
            row["course_deg"] = "" if i == 1 else f"{45.0 + i:.1f}"
        rows.append(row)
    gps = root / "gps.csv"
    with open(gps, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return frames, gps


class MetadataContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames, self.gps = write_source(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_ingest(self, *extra):
        out = self.root / f"ds{len(extra)}"
        argv = ["--source_dir", str(self.root),
                "--gps_csv", str(self.gps),
                "--output", str(out),
                "--dataset_id", "unit_test_leg",
                "--width", str(PANO_W),
                "--height", str(PANO_W // 2),
                "--raw_material", "raw_material/unit_test",
                *extra]
        self.assertEqual(ingest.main(argv), 0)
        return out, json.loads((out / "pipeline_metadata.json").read_text())

    def test_metadata_satisfies_the_dataset_contract(self):
        out, meta = self.run_ingest()
        # north_aligned recorded false: the gate dataset.py enforces.
        self.assertIs(meta["north_aligned"], False)
        self.assertEqual(meta["intrinsics_csv"], "intrinsics.csv")
        ds_lib.require_camera_frame_panoramas(meta, out)
        self.assertEqual(meta["azimuth_convention"]["camera_frame"],
                         geo.CAMERA_FRAME)
        self.assertEqual(meta["gps_course_diagnostic"]["use"],
                         "diagnostic_only")
        self.assertNotIn("heading_reliable", meta)
        self.assertNotIn("heading_source", meta)
        self.assertNotIn("mount_offset", meta)

    def test_mount_offset_option_is_not_accepted(self):
        with self.assertRaises(SystemExit):
            self.run_ingest("--mount_offset_deg", "214.0")

    def test_course_is_diagnostic_and_intrinsics_heading_is_unset(self):
        out, _ = self.run_ingest()
        with open(out / "intrinsics.csv") as f:
            intrinsics = list(csv.DictReader(f))
        with open(out / "extraction_log.csv") as f:
            extlog = list(csv.DictReader(f))
        with open(out / "frames_gps.csv") as f:
            gps_rows = list(csv.DictReader(f))
        self.assertEqual(len(intrinsics), len(extlog))
        for column in ("computed_compass_angle_true_deg",
                       "compass_angle_true_deg",
                       "heading_optical_axis_true_deg",
                       "heading_column0_true_deg",
                       "selected_heading_source"):
            self.assertTrue(all(not row[column] for row in intrinsics), column)
        self.assertTrue(all(row["focal_source"] == "n/a"
                            for row in intrinsics))
        self.assertTrue(all(not row["heading_used"] for row in extlog))
        self.assertEqual(gps_rows[0]["gps_course_deg"], "45.0000")
        self.assertEqual(gps_rows[1]["gps_course_deg"], "")

    def test_extra_metadata_cannot_restore_orientation_authority(self):
        extras = self.root / "extras.json"
        extras.write_text(json.dumps({"mount_offset": {"value": 1.0}}))
        with self.assertRaises(SystemExit):
            self.run_ingest("--extra_metadata", str(extras))

    def test_ingest_output_loads_through_the_dataset_reader(self):
        out, _ = self.run_ingest()
        frames = ds_lib.load_frames(out)
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[0].pano_id, "f0000")
        anchor_lat, _ = ds_lib.fill_enu(frames)
        self.assertAlmostEqual(anchor_lat, 42.35015, places=4)

    def test_provenance_is_recorded(self):
        _, meta = self.run_ingest()
        self.assertTrue(meta["ingest"]["git_commit"])
        # argv is the process's own sys.argv (the tool records how it was
        # invoked; under a test runner that is the runner's argv).
        self.assertIsInstance(meta["ingest"]["argv"], list)
        self.assertTrue(meta["ingest"]["argv"])
        self.assertTrue(meta["ingest"]["created"])
        self.assertEqual(meta["ingest"]["kept"], 4)
        self.assertEqual(
            meta["ingest"]["generator"],
            "farfield/dataset_tools/ingest_selfcollect.py")

    def test_success_never_consumes_source_images(self):
        before = sorted(path.read_bytes() for path in self.frames.glob("*.jpg"))
        self.run_ingest()
        after = sorted(path.read_bytes() for path in self.frames.glob("*.jpg"))
        self.assertEqual(after, before)

    def test_checksum_manifest_is_complete_before_publish(self):
        real_publish = ingest.publish_dataset

        def verify_then_publish(staging, output):
            self.assertFalse(output.exists())
            self.assertGreater(checksums.verify(staging), 0)
            self.assertNotIn(
                checksums.CHECKSUM_FILE,
                (staging / checksums.CHECKSUM_FILE).read_text())
            real_publish(staging, output)

        with mock.patch.object(
                ingest, "publish_dataset", side_effect=verify_then_publish):
            out, _ = self.run_ingest()
        self.assertGreater(checksums.verify(out), 0)

    def test_existing_output_is_rejected_without_clobber(self):
        out = self.root / "existing"
        out.mkdir()
        marker = out / "owner.txt"
        marker.write_text("first writer")
        argv = ["--source_dir", str(self.root), "--gps_csv", str(self.gps),
                "--output", str(out), "--dataset_id", "unit_test_leg",
                "--width", str(PANO_W), "--height", str(PANO_W // 2)]
        with self.assertRaises(SystemExit):
            ingest.main(argv)
        self.assertEqual(marker.read_text(), "first writer")
        self.assertEqual(len(list(self.frames.glob("*.jpg"))), 4)

    def test_bad_row_fails_preflight_without_partial_output(self):
        rows = list(csv.DictReader(self.gps.open()))
        rows[2]["latitude"] = "not-a-number"
        ingest.write_rows(self.gps, rows, list(rows[0]))
        out = self.root / "bad-output"
        argv = ["--source_dir", str(self.root), "--gps_csv", str(self.gps),
                "--output", str(out), "--dataset_id", "unit_test_leg",
                "--width", str(PANO_W), "--height", str(PANO_W // 2)]
        with self.assertRaises(SystemExit):
            ingest.main(argv)
        self.assertFalse(out.exists())
        self.assertFalse(out.with_name(out.name + ".incomplete").exists())
        self.assertEqual(len(list(self.frames.glob("*.jpg"))), 4)


if __name__ == "__main__":
    unittest.main()
