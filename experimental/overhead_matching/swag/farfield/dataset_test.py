import contextlib
import io
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

    def test_orientation_qualifiers_require_actual_booleans(self):
        for field_path, value in (
                (("is_equirectangular",), 1),
                (("north_aligned",), 0),
                (("azimuth_convention", "images_rotated"), "false")):
            with self.subTest(field_path=field_path):
                metadata = testing.default_metadata()
                target = metadata
                for key in field_path[:-1]:
                    target = target[key]
                target[field_path[-1]] = value
                with self.assertRaisesRegex(ds_lib.ContractViolation,
                                            "actual boolean"):
                    ds_lib.require_camera_frame_panoramas(metadata, self.base)

    def test_non_camera_frame_inputs_are_refused(self):
        cases = (
            ("is_equirectangular", False, "perspective imagery"),
            ("images_rotated", True, "images_rotated must be false"),
            ("camera_frame", "unknown", "canonical camera frame"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                metadata = testing.default_metadata()
                if field == "is_equirectangular":
                    metadata[field] = value
                else:
                    metadata["azimuth_convention"][field] = value
                with self.assertRaisesRegex(ds_lib.ContractViolation, message):
                    ds_lib.require_camera_frame_panoramas(metadata, self.base)


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

    def make_predictions(self, populated=None, *, directory=None,
                         dataset_name="ds"):
        predictions = {stem: [] for stem in self.stems}
        predictions.update(populated or {})
        return testing.make_predictions(
            directory or self.fl_dir, predictions,
            dataset_name=dataset_name)

    def test_end_to_end_observation(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Custom House Tower", [(0, 400, 100, 600, 500)])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 1)
        obs = result.observations[0]
        self.assertEqual(obs.local_obs_id, "f0000__lm0__box0")
        self.assertTrue(obs.obs_id.startswith("obs-"))
        self.assertEqual(obs.obs_id, obs.key.global_id)
        self.assertEqual(obs.key.dataset, "ds")
        self.assertEqual(obs.key.frame_landmarks_version, "v1")
        self.assertEqual(result.dataset_name, "ds")
        self.assertEqual(result.frame_landmarks_ref.kind, "frame_landmarks")
        self.assertFalse(obs.seam_merged)
        # Face 0, centered at x=500: camera azimuth 0.
        self.assertAlmostEqual(obs.bearing_camera_cw_deg,
                               geo.bearing_camera_cw_deg(0, 500.0), places=6)
        # Elevation matches the single definition at the bbox center.
        self.assertAlmostEqual(
            obs.elevation_deg,
            geo.direction_from_face_px(0, 500.0, 300.0)[1], places=6)
        self.assertEqual(result.frames[0].n_observations, 1)

    def test_seam_continuation_merges_into_one_observation(self):
        # Face 0's right edge adjoins face 270's left edge (A - 90 rule):
        # one physical object spanning that seam is one observation.
        self.make_predictions({
            self.stems[0]: [testing.landmark("Long Wharf", [
                (0, 950, 200, 1000, 400),
                (270, 0, 210, 60, 410),
            ])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 1)
        self.assertTrue(result.observations[0].seam_merged)
        # The merged bearing straddles the 45 deg seam between the faces.
        self.assertAlmostEqual(result.observations[0].bearing_camera_cw_deg,
                               45.0, delta=5.0)

    def test_non_adjacent_boxes_stay_separate(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark("Two Things", [
                (0, 950, 200, 1000, 400),
                (90, 0, 210, 60, 410),   # 90 is NOT adjacent to 0's right edge
            ])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 2)

    def test_a_lossy_ingest_says_so_on_stderr(self):
        """Dropped geometry must be visible without reading `.stats`.

        No caller of `run_ingest` reads the counters, so without this line a
        run that ingested 90% of its detections and one that ingested all of
        them are indistinguishable, all the way to a localization result.
        """
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Bad Yaw", [(45, 100, 100, 200, 200)])],
        })
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        reported = stderr.getvalue()
        self.assertIn("WARNING", reported)
        self.assertIn("DISCARDED 1 malformed bounding box", reported)
        self.assertTrue(result.stats.lossy)

    def test_a_clean_ingest_reports_without_warning(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Fine", [(0, 100, 100, 200, 200)])],
        })
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        reported = stderr.getvalue()
        self.assertNotIn("WARNING", reported)
        self.assertIn("no predicted geometry discarded", reported)
        self.assertFalse(result.stats.lossy)

    def test_invalid_yaw_is_dropped_and_counted(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Bad Yaw", [(45, 100, 100, 200, 200)])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(result.observations, [])
        self.assertEqual(result.stats.n_boxes_invalid_geometry, 1)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 1)

    def test_invalid_bbox_is_dropped_without_losing_valid_sibling(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Mixed", [
                    (0, 600, 100, 400, 500),
                    (0, 400.25, 100.5, 600.75, 500.25),
                ])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.observations[0].boxes[0].xmin, 400.25)
        self.assertEqual(result.stats.n_boxes_invalid_geometry, 1)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 0)

    def test_all_invalid_box_shapes_are_dropped_and_counted(self):
        landmark = testing.landmark("Bad boxes", [])
        landmark["bounding_boxes"] = [
            None,
            {"yaw_angle": 0, "xmin": 0, "ymin": 0, "xmax": 1},
            {"yaw_angle": True, "xmin": 0, "ymin": 0,
             "xmax": 1, "ymax": 1},
            {"yaw_angle": 0, "xmin": True, "ymin": 0,
             "xmax": 1, "ymax": 1},
            {"yaw_angle": 0, "xmin": -1, "ymin": 0,
             "xmax": 1, "ymax": 1},
            {"yaw_angle": 0, "xmin": 0, "ymin": 0,
             "xmax": 1, "ymax": 1001},
            {"yaw_angle": 0, "xmin": 0, "ymin": 1,
             "xmax": 1, "ymax": 1},
        ]
        self.make_predictions({self.stems[0]: [landmark]})
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(result.observations, [])
        self.assertEqual(result.stats.n_boxes_invalid_geometry, 7)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 1)

    def test_empty_box_list_is_a_counted_empty_landmark(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark("No boxes", [])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(result.observations, [])
        self.assertEqual(result.stats.n_boxes_invalid_geometry, 0)
        self.assertEqual(result.stats.n_landmarks_without_valid_boxes, 1)

    def test_nonfinite_box_is_dropped_by_geometry_validator(self):
        landmark = testing.landmark("Nonfinite", [])
        landmark["bounding_boxes"] = [{
            "yaw_angle": 0,
            "xmin": 0,
            "ymin": 0,
            "xmax": float("inf"),
            "ymax": 1,
        }]
        _, boxes, n_invalid = ds_lib._validated_landmark(
            landmark, "test landmark")
        self.assertEqual(boxes, [])
        self.assertEqual(n_invalid, 1)

    def test_float_bbox_coordinates_are_not_truncated(self):
        self.make_predictions({
            self.stems[0]: [testing.landmark(
                "Precise", [(0, 400.25, 100.5, 600.75, 500.25)])],
        })
        result = ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertEqual(result.observations[0].boxes[0].xmin, 400.25)

    def test_incomplete_prediction_coverage_is_rejected(self):
        testing.make_predictions(
            self.fl_dir, {self.stems[0]: []}, dataset_name="ds")
        with self.assertRaisesRegex(ds_lib.ContractViolation, "coverage"):
            ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)

    def test_artifact_for_another_dataset_is_rejected(self):
        self.make_predictions(dataset_name="another-dataset")
        with self.assertRaisesRegex(ds_lib.ContractViolation, "dataset mismatch"):
            ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)

    def test_north_aligned_dataset_is_refused_before_reading_pixels(self):
        meta = testing.default_metadata()
        meta["north_aligned"] = True
        base2 = testing.make_dataset(Path(self.tmp.name) / "ds2",
                                     metadata=meta)
        testing.make_predictions(
            Path(self.tmp.name) / "fl2", {}, dataset_name="ds2")
        with self.assertRaises(ds_lib.ContractViolation):
            ds_lib.run_ingest(base2, Path(self.tmp.name) / "fl2", PARAMS)

    def test_bad_orientation_is_refused_before_predictions_are_opened(self):
        for field, value in (("is_equirectangular", False),
                             ("images_rotated", True)):
            with self.subTest(field=field):
                metadata = testing.default_metadata()
                if field == "is_equirectangular":
                    metadata[field] = value
                else:
                    metadata["azimuth_convention"][field] = value
                base = testing.make_dataset(
                    Path(self.tmp.name) / f"ds_{field}", metadata=metadata)
                missing = Path(self.tmp.name) / f"missing_{field}"
                with self.assertRaises(ds_lib.ContractViolation) as ctx:
                    ds_lib.run_ingest(base, missing, PARAMS)
                self.assertNotIn(
                    "frame_landmarks artifact", str(ctx.exception))

    def test_missing_predictions_artifact_is_a_pointed_error(self):
        with self.assertRaises(ds_lib.ContractViolation) as ctx:
            ds_lib.run_ingest(self.base, self.fl_dir, PARAMS)
        self.assertIn("frame_landmarks artifact", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
