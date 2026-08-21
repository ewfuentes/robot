import csv
import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    dataset as ds_lib,
    geometry as geo,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    ingest_selfcollect as ingest,
)

PANO_W = 512


class HeadingTextTest(unittest.TestCase):
    """The one heading computation. The original carried two copies — a
    guarded one for extraction_log and an unguarded duplicate for
    intrinsics.csv that crashed on a blank course the guarded copy had
    already survived."""

    def test_blank_course_yields_empty_not_a_crash(self):
        for blank in ("", "   ", None):
            self.assertEqual(ingest.heading_deg_text(blank, 214.0, PANO_W), "")
            self.assertEqual(ingest.heading_deg_text(blank, None, PANO_W), "")

    def test_without_offset_heading_is_the_raw_course(self):
        self.assertEqual(ingest.heading_deg_text("37.5", None, PANO_W),
                         "37.5000")
        self.assertEqual(ingest.heading_deg_text("-10", None, PANO_W),
                         "350.0000")

    def test_with_offset_column_zero_bearing_comes_from_geometry(self):
        # Column 0's camera azimuth is 180 deg from centre-column forward, so
        # heading(column 0) = course + (180 - offset). The half turn falls out
        # of the geometry helpers rather than being hand-typed.
        course, offset = 90.0, 214.0
        expected = float(geo.body_to_world_bearing_deg(
            course,
            geo.apply_mount_offset(
                geo.azimuth_of_pano_column(0.0, PANO_W), offset)))
        self.assertAlmostEqual(
            float(ingest.heading_deg_text(str(course), offset, PANO_W)),
            expected, places=4)
        # Sanity: with a direction-of-travel offset of 180 the camera faces
        # backwards, so column 0 points along the course.
        self.assertAlmostEqual(
            float(ingest.heading_deg_text("90", 180.0, PANO_W)), 90.0,
            places=4)

    def test_output_is_wrapped_into_range(self):
        for course in ("350", "10", "180"):
            value = float(ingest.heading_deg_text(course, 214.0, PANO_W))
            self.assertGreaterEqual(value, 0.0)
            self.assertLess(value, 360.0)


class FillCourseFromTrackTest(unittest.TestCase):
    def test_course_follows_the_track(self):
        rows = [{"latitude": "42.0", "longitude": "-71.0"},
                {"latitude": "42.0", "longitude": "-70.99"},
                {"latitude": "42.0", "longitude": "-70.98"}]
        ingest.fill_course_from_track(rows)
        for row in rows:
            self.assertAlmostEqual(float(row["course_deg"]), 90.0, delta=1.0)

    def test_northbound_track(self):
        rows = [{"latitude": "42.00", "longitude": "-71.0"},
                {"latitude": "42.01", "longitude": "-71.0"}]
        ingest.fill_course_from_track(rows)
        self.assertAlmostEqual(float(rows[0]["course_deg"]), 0.0, delta=1.0)


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
        out, meta = self.run_ingest(
            "--mount_offset_deg", "214.0",
            "--mount_offset_source", "surveyed building, 72 keyframes")
        # north_aligned recorded false: the gate dataset.py enforces.
        self.assertIs(meta["north_aligned"], False)
        ds_lib.require_camera_frame_panoramas(meta, out)
        # The azimuth convention carries the mount-offset frame tag whose
        # absence caused the pohang incident.
        self.assertEqual(meta["azimuth_convention"]["mount_offset_frame"],
                         geo.MOUNT_OFFSET_FRAME)
        self.assertEqual(
            meta["azimuth_convention"]["heading_deg_is_bearing_of"],
            "column_0")
        # And the mount_offset block is consumable, truthfully flagged as
        # already folded into heading_deg.
        record = ds_lib.mount_offset_record(meta, out)
        self.assertAlmostEqual(record.offset_deg, 214.0)
        self.assertTrue(record.applied_to_heading_deg)
        self.assertFalse(record.accuracy_validated)
        self.assertEqual(meta["mount_offset"]["convention"],
                         geo.MOUNT_OFFSET_CONVENTION)

    def test_no_offset_means_no_block_not_a_null_one(self):
        out, meta = self.run_ingest()
        self.assertNotIn("mount_offset", meta)
        # Absent is the contract's word for uncalibrated.
        self.assertIsNone(ds_lib.mount_offset_record(meta, out))

    def test_offset_requires_a_source(self):
        with self.assertRaises(SystemExit):
            self.run_ingest("--mount_offset_deg", "214.0")

    def test_blank_course_row_does_not_crash_either_table(self):
        out, _ = self.run_ingest(
            "--mount_offset_deg", "214.0",
            "--mount_offset_source", "prior")
        with open(out / "intrinsics.csv") as f:
            intrinsics = list(csv.DictReader(f))
        with open(out / "extraction_log.csv") as f:
            extlog = list(csv.DictReader(f))
        self.assertEqual(len(intrinsics), len(extlog))
        blanks = [r for r in intrinsics if not r["heading_deg"]]
        self.assertEqual(len(blanks), 1)  # exactly the blank-course row

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


if __name__ == "__main__":
    unittest.main()
