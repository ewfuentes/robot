import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    dataset as ds_lib,
    geometry as geo,
    testing,
)

PARAMS = ds_lib.IngestParams(fov_deg=90.0, seam_gap_norm=25,
                             seam_min_y_iou=0.3)


class FramesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = testing.make_dataset(Path(self.tmp.name) / "ds",
                                         n_frames=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_frames_join_gps_table(self):
        frames = ds_lib.load_frames(self.base)
        self.assertEqual(len(frames), 4)
        self.assertEqual(frames[2].pano_id, "f0002")
        self.assertEqual(frames[2].dist_along_m, 20.0)
        self.assertEqual(frames[2].time_s, 4.0)

    def test_fill_enu_anchors_at_mean(self):
        frames = ds_lib.load_frames(self.base)
        anchor_lat, anchor_lon = ds_lib.fill_enu(frames)
        self.assertAlmostEqual(
            anchor_lat, sum(f.lat for f in frames) / len(frames))
        east, north = geo.enu_from_latlon(frames[0].lat, frames[0].lon,
                                          anchor_lat, anchor_lon)
        self.assertAlmostEqual(frames[0].x_m, east)
        self.assertAlmostEqual(frames[0].y_m, north)

    def test_missing_pano_diverges_id_from_index(self):
        """The trap frame_index_by_pano_id exists for: after a gap,
        int(pano_id[1:]) != frame_idx."""
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        base = testing.make_dataset(Path(tmp2.name) / "ds", n_frames=4,
                                    skip_pano_numbers=(1,))
        frames = ds_lib.load_frames(base)
        index = ds_lib.frame_index_by_pano_id(frames)
        self.assertEqual(index["f0002"], 1)      # positional
        self.assertNotEqual(index["f0002"], 2)   # NOT the parsed number


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / "ds"

    def tearDown(self):
        self.tmp.cleanup()

    def test_north_aligned_true_is_refused(self):
        meta = testing.default_metadata()
        meta["north_aligned"] = True
        testing.make_dataset(self.base, metadata=meta)
        with self.assertRaises(ds_lib.ContractViolation) as ctx:
            ds_lib.require_camera_frame_panoramas(
                ds_lib.load_metadata(self.base), self.base)
        self.assertIn("north-aligned", str(ctx.exception))

    def test_unrecorded_orientation_is_refused(self):
        meta = testing.default_metadata()
        del meta["north_aligned"]
        testing.make_dataset(self.base, metadata=meta)
        with self.assertRaises(ds_lib.ContractViolation):
            ds_lib.require_camera_frame_panoramas(
                ds_lib.load_metadata(self.base), self.base)

    def test_valid_mount_offset_parses(self):
        testing.make_dataset(self.base)
        record = ds_lib.mount_offset_record(
            ds_lib.load_metadata(self.base), self.base)
        self.assertAlmostEqual(record.offset_deg, 214.0)
        self.assertTrue(record.accuracy_validated)
        self.assertFalse(record.applied_to_heading_deg)

    def test_absent_mount_offset_is_none(self):
        meta = testing.default_metadata()
        del meta["mount_offset"]
        testing.make_dataset(self.base, metadata=meta)
        self.assertIsNone(ds_lib.mount_offset_record(
            ds_lib.load_metadata(self.base), self.base))

    def test_unqualified_mount_offset_is_refused(self):
        # The pre-migration shape: a bare number with no frame/applied
        # qualifiers. Consuming it is how pohang shipped 180 deg out.
        meta = testing.default_metadata()
        meta["mount_offset"] = {"mount_offset_deg": 180.0, "status": "manual"}
        testing.make_dataset(self.base, metadata=meta)
        with self.assertRaises(ds_lib.ContractViolation) as ctx:
            ds_lib.mount_offset_record(ds_lib.load_metadata(self.base),
                                       self.base)
        self.assertIn("frame", str(ctx.exception))
        self.assertIn("applied_to_heading_deg", str(ctx.exception))

    def test_wrong_frame_string_is_refused(self):
        meta = testing.default_metadata()
        meta["mount_offset"]["frame"] = "column_0"
        testing.make_dataset(self.base, metadata=meta)
        with self.assertRaises(ds_lib.ContractViolation) as ctx:
            ds_lib.mount_offset_record(ds_lib.load_metadata(self.base),
                                       self.base)
        self.assertIn("180", str(ctx.exception))


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = testing.make_dataset(Path(self.tmp.name) / "ds",
                                         n_frames=3)
        frames = ds_lib.load_frames(self.base)
        self.stems = [f.pano_stem for f in frames]
        self.fl_dir = Path(self.tmp.name) / "frame_landmarks"

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_observation(self):
        testing.make_predictions(self.fl_dir, {
            self.stems[0]: [testing.landmark(
                "Custom House Tower", [(0, 400, 100, 600, 500)])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertEqual(obs.obs_id, "f0000__lm0__box0")
        self.assertFalse(obs.seam_merged)
        # Face 0, centered at x=500: camera azimuth 0.
        self.assertAlmostEqual(obs.bearing_camera_deg,
                               geo.bearing_camera_deg(0, 500.0), places=6)
        # Elevation matches the single definition at the bbox center.
        self.assertAlmostEqual(
            obs.elevation_deg,
            geo.direction_from_face_px(0, 500.0, 300.0)[1], places=6)
        self.assertEqual(result.frames[0].n_observations, 1)

    def test_seam_continuation_merges_into_one_observation(self):
        # Face 0's right edge adjoins face 270's left edge (A - 90 rule):
        # one physical object spanning that seam is one observation.
        testing.make_predictions(self.fl_dir, {
            self.stems[0]: [testing.landmark("Long Wharf", [
                (0, 950, 200, 1000, 400),
                (270, 0, 210, 60, 410),
            ])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 1)
        self.assertTrue(result.observations[0].seam_merged)
        # The merged bearing straddles the 45 deg seam between the faces.
        self.assertAlmostEqual(result.observations[0].bearing_camera_deg,
                               45.0, delta=5.0)

    def test_non_adjacent_boxes_stay_separate(self):
        testing.make_predictions(self.fl_dir, {
            self.stems[0]: [testing.landmark("Two Things", [
                (0, 950, 200, 1000, 400),
                (90, 0, 210, 60, 410),   # 90 is NOT adjacent to 0's right edge
            ])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 2)

    def test_invalid_yaw_boxes_are_counted_and_dropped(self):
        testing.make_predictions(self.fl_dir, {
            self.stems[0]: [testing.landmark(
                "Bad Yaw", [(45, 100, 100, 200, 200)])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 0)
        self.assertEqual(result.stats.n_boxes_invalid_yaw, 1)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 1)

    def test_north_aligned_dataset_is_refused_before_reading_pixels(self):
        meta = testing.default_metadata()
        meta["north_aligned"] = True
        base2 = testing.make_dataset(Path(self.tmp.name) / "ds2",
                                     metadata=meta)
        testing.make_predictions(Path(self.tmp.name) / "fl2", {})
        with self.assertRaises(ds_lib.ContractViolation):
            ds_lib.run_ingest(base2, Path(self.tmp.name) / "fl2", PARAMS)

    def test_missing_predictions_artifact_is_a_pointed_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertIn("extraction stage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
