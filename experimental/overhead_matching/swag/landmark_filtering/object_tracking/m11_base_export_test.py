import math
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.bearing_only_localization import (
    export_ingest,
    gps_to_odometry,
    structs,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    m11_base_export as mbe,
)


def merged_row(tracklet, keyframe, bearing_camera, kappa=100.0, source=0):
    """One row in m6's `merged/measurements.json` shape."""
    return {"tracklet_id": tracklet, "source_track_id": source,
            "anchor_keyframe_idx": keyframe,
            "bearing_camera_deg": bearing_camera, "kappa": kappa}


class BodyFrameMeasurementTest(unittest.TestCase):

    def test_camera_bearing_rotates_by_the_mount_offset(self):
        rows = [merged_row("LT0", 3, 341.69), merged_row("LT1", 5, 10.0)]
        measurements, fused = mbe.body_frame_measurements(rows, 214.0)

        self.assertEqual(fused, 0)
        self.assertAlmostEqual(measurements[0].bearing_body_deg, 127.69,
                               places=6)
        # Wraps rather than going negative: 10 - 214 = -204 -> 156.
        self.assertAlmostEqual(measurements[1].bearing_body_deg, 156.0,
                               places=6)

    def test_kappa_is_carried_through_untouched(self):
        rows = [merged_row("LT0", 3, 100.0, kappa=2777.436346438264)]
        measurements, _ = mbe.body_frame_measurements(rows, 90.0)
        self.assertEqual(measurements[0].kappa, 2777.436346438264)

    def test_two_source_tracks_on_one_keyframe_fuse_into_one_epoch(self):
        # A merged tracklet whose constituents were both alive at keyframe 264
        # writes two rows for the same epoch; export_ingest rejects duplicates,
        # and physically these are one object seen twice.
        rows = [merged_row("LT227_T249", 264, 0.4875, kappa=264.9, source=227),
                merged_row("LT227_T249", 264, 0.5941, kappa=258.3, source=249)]
        measurements, fused = mbe.body_frame_measurements(rows, 0.0)

        self.assertEqual(len(measurements), 1)
        self.assertEqual(fused, 1)
        self.assertTrue(0.4875 < measurements[0].bearing_body_deg < 0.5941)
        # Two agreeing bearings sharpen: the fused concentration is close to
        # the sum, not to either input.
        self.assertGreater(measurements[0].kappa, 500.0)

    def test_opposed_bearings_cancel_instead_of_averaging(self):
        # The failure this guards against is a confident average of two
        # contradictory bearings. The resultant must be weak, not tight.
        rows = [merged_row("LT9", 1, 0.0, kappa=100.0),
                merged_row("LT9", 1, 180.0, kappa=100.0)]
        measurements, _ = mbe.body_frame_measurements(rows, 0.0)
        self.assertLess(measurements[0].kappa, 1e-9)

    def test_non_positive_kappa_is_refused(self):
        for kappa in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                mbe.body_frame_measurements(
                    [merged_row("LT0", 1, 10.0, kappa=kappa)], 0.0)


class TableTest(unittest.TestCase):

    def test_one_flat_table_per_measured_tracklet(self):
        rows = [merged_row("LT0", 1, 10.0), merged_row("LT0", 2, 12.0),
                merged_row("LT1", 2, 90.0)]
        measurements, _ = mbe.body_frame_measurements(rows, 0.0)
        tables = mbe.uninformative_tables(measurements, 0.0, 4.0)

        self.assertEqual([t.tracklet_id for t in tables], ["LT0", "LT1"])
        for table in tables:
            self.assertEqual(table.entries, [])
            self.assertEqual(table.default_log_lr, 0.0)
            self.assertEqual((table.clip_lo, table.clip_hi), (-4.0, 4.0))


class RoundTripTest(unittest.TestCase):
    """The export has to survive the filter's own loader, whose validate() is
    the boundary that would otherwise fail deep inside a run."""

    def build(self, directory: Path, rows: list, n_keyframes: int = 4):
        east = [0.0, 30.0, 60.0, 90.0][:n_keyframes]
        north = [0.0, 0.0, 10.0, 20.0][:n_keyframes]
        measurements, _ = mbe.body_frame_measurements(rows, 214.0)
        mbe.write_export(
            directory,
            meta={"schema_version": structs.SCHEMA_VERSION,
                  "scenario_name": "unit_test",
                  "anchor_lat_deg": 42.3, "anchor_lon_deg": -71.0,
                  "n_keyframes": n_keyframes,
                  "matcher_version": mbe.UNINFORMATIVE_MATCHER,
                  "mount_offset_deg": 214.0,
                  # Provenance keys the reader's struct does not declare; they
                  # must not break decoding.
                  "mount_offset_source": "unit test",
                  "truth_heading_note": "GPS course"},
            landmarks=[structs.LandmarkEntry(landmark_id="osm:node:1",
                                             lat_deg=42.31, lon_deg=-71.0,
                                             type_key="man_made=tower"),
                       structs.LandmarkEntry(landmark_id="osm:node:2",
                                             lat_deg=42.30, lon_deg=-70.99,
                                             type_key="landmark")],
            tables=mbe.uninformative_tables(measurements, 0.0, 4.0),
            measurements=measurements,
            odometry=gps_to_odometry.derive_increments(east, north),
            truth=mbe.truth_poses(east, north, [0.0] * n_keyframes))

    def test_export_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 1, 341.69),
                             merged_row("LT0", 2, 350.0),
                             merged_row("LT1", 3, 12.0)])
            data = export_ingest.load(out)

        self.assertEqual(data.n_keyframes, 4)
        self.assertEqual(len(data.measurements), 3)
        self.assertEqual(len(data.odometry), 3)
        self.assertEqual(len(data.truth), 4)
        self.assertEqual(sorted(data.tables), ["LT0", "LT1"])
        self.assertEqual(data.meta.mount_offset_deg, 214.0)
        self.assertAlmostEqual(data.measurements[0].bearing_body_deg, 127.69,
                               places=6)

    def test_odometry_indices_are_contiguous_from_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 1, 341.69)])
            data = export_ingest.load(out)
        self.assertEqual([o.keyframe_idx for o in data.odometry], [1, 2, 3])

    def test_measurement_beyond_the_last_keyframe_is_rejected(self):
        # A measurement anchored off the end of the run is exactly the kind of
        # misalignment that a silent renumbering would produce.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            self.build(out, [merged_row("LT0", 9, 341.69)])
            with self.assertRaises(ValueError):
                export_ingest.load(out)


class FuseBearingsTest(unittest.TestCase):

    def test_resultant_is_the_von_mises_product(self):
        bearing, kappa = mbe.fuse_bearings([(10.0, 3.0), (20.0, 4.0)])
        east = 3.0 * math.sin(math.radians(10.0)) + 4.0 * math.sin(
            math.radians(20.0))
        north = 3.0 * math.cos(math.radians(10.0)) + 4.0 * math.cos(
            math.radians(20.0))
        self.assertAlmostEqual(bearing,
                               math.degrees(math.atan2(east, north)) % 360.0)
        self.assertAlmostEqual(kappa, math.hypot(east, north))

    def test_wrap_around_north_is_handled(self):
        # 350 and 10 average to 0, not to 180 -- the trap in naive averaging.
        bearing, _ = mbe.fuse_bearings([(350.0, 5.0), (10.0, 5.0)])
        self.assertAlmostEqual(bearing % 360.0, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
