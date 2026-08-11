import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    scenario,
)


class TruthTrajectoryTest(unittest.TestCase):
    def test_spacing_and_headings(self):
        cfg = scenario.l_turn(speed_mps=10.0, keyframe_period_s=2.0)
        data = scenario.generate(cfg)
        step = cfg.speed_mps * cfg.keyframe_period_s
        for a, b in zip(data.truth[:-1], data.truth[1:]):
            d = math.hypot(b.east_m - a.east_m, b.north_m - a.north_m)
            self.assertLessEqual(d, step + 1e-6)
        # First leg heads due east (090), second leg due north (000).
        self.assertAlmostEqual(data.truth[0].heading_deg, 90.0, places=6)
        self.assertAlmostEqual(data.truth[-1].heading_deg, 0.0, places=6)

    def test_endpoints(self):
        cfg = scenario.straight_leg(speed_mps=20.0, keyframe_period_s=5.0)
        data = scenario.generate(cfg)
        self.assertAlmostEqual(data.truth[0].east_m, -1000.0, places=6)
        self.assertAlmostEqual(data.truth[0].north_m, -500.0, places=6)
        # 2000 m at 100 m/step -> 21 keyframes ending at the far waypoint.
        self.assertEqual(data.n_keyframes, 21)
        self.assertAlmostEqual(data.truth[-1].east_m, 1000.0, places=6)


class GeneratedInputConsistencyTest(unittest.TestCase):
    def test_zero_noise_odometry_sums_to_truth(self):
        cfg = scenario.harbor_loop(odom_sigma_m=0.0, course_sigma_deg=1e-9)
        data = scenario.generate(cfg)
        east = data.truth[0].east_m + sum(o.dx_m for o in data.odometry)
        north = data.truth[0].north_m + sum(o.dy_m for o in data.odometry)
        self.assertAlmostEqual(east, data.truth[-1].east_m, places=6)
        self.assertAlmostEqual(north, data.truth[-1].north_m, places=6)

    def test_bearings_reconstruct_world_direction(self):
        """body bearing + truth heading == compass bearing to the landmark."""
        cfg = scenario.harbor_loop(bearing_sigma_deg=1e-3)
        data = scenario.generate(cfg)
        truth_by_kf = {t.keyframe_idx: t for t in data.truth}
        lm_index = {lm_id: i for i, lm_id in enumerate(data.landmark_ids)}
        self.assertGreater(len(data.measurements), 10)
        for meas in data.measurements:
            pose = truth_by_kf[meas.anchor_keyframe_idx]
            i = lm_index[meas.tracklet_id.removeprefix("trk_")]
            expected_world = math.degrees(math.atan2(
                data.true_east_m[i] - pose.east_m,
                data.true_north_m[i] - pose.north_m))
            reconstructed = meas.bearing_body_deg + pose.heading_deg
            diff = math.degrees(abs(float(geodesy.wrap_rad(
                math.radians(reconstructed - expected_world)))))
            self.assertLess(diff, 0.02)

    def test_epoch_cadence_and_stagger(self):
        cfg = scenario.harbor_loop(epoch_length_keyframes=6)
        data = scenario.generate(cfg)
        anchors_by_tracklet = {}
        for meas in data.measurements:
            anchors_by_tracklet.setdefault(meas.tracklet_id, []).append(
                meas.anchor_keyframe_idx)
        offsets = set()
        for anchors in anchors_by_tracklet.values():
            self.assertEqual(anchors, sorted(anchors))
            spacings = np.diff(anchors)
            self.assertTrue(np.all(spacings == 6))
            offsets.add(anchors[0] % 6)
        # Three tracklets, staggered: distinct phases within the epoch.
        self.assertEqual(len(offsets), 3)

    def test_identity_tables(self):
        cfg = scenario.harbor_loop()
        data = scenario.generate(cfg)
        for lm_id in data.landmark_ids:
            table = data.tables[f"trk_{lm_id}"]
            self.assertEqual(len(table.entries), 1)
            self.assertEqual(table.entries[0].landmark_id, lm_id)
            self.assertEqual(table.entries[0].log_lr, cfg.identity_clip)
            self.assertEqual(table.matcher_version,
                             scenario.MATCHER_VERSION)

    def test_clutter_only_tables_empty(self):
        cfg = scenario.harbor_loop(clutter_only=True)
        data = scenario.generate(cfg)
        for table in data.tables.values():
            self.assertEqual(table.entries, [])

    def test_course_bias_separates_heading_from_course(self):
        """The crab knob must move true heading while leaving the reported
        course over ground alone -- that separation is the whole point."""
        unbiased = scenario.generate(scenario.harbor_loop(
            course_sigma_deg=1e-9, course_bias_deg=0.0))
        biased = scenario.generate(scenario.harbor_loop(
            course_sigma_deg=1e-9, course_bias_deg=7.0))
        for a, b in zip(unbiased.odometry, biased.odometry):
            self.assertAlmostEqual(a.course_deg, b.course_deg, places=6)
        for a, b in zip(unbiased.truth, biased.truth):
            self.assertAlmostEqual((b.heading_deg - a.heading_deg) % 360.0,
                                   7.0, places=6)

    def test_catalog_error_moves_only_the_filters_copy(self):
        data = scenario.generate(scenario.harbor_loop(
            catalog_position_sigma_m=25.0))
        offsets = np.hypot(data.catalog.east_m - data.true_east_m,
                           data.catalog.north_m - data.true_north_m)
        self.assertTrue(np.all(offsets > 0.0))
        self.assertLess(float(np.max(offsets)), 200.0)
        np.testing.assert_allclose(data.catalog.position_sigma_m, 25.0)

    def test_outliers_are_injected_at_the_requested_rate(self):
        clean = scenario.generate(scenario.harbor_loop(bearing_sigma_deg=0.5))
        dirty = scenario.generate(scenario.harbor_loop(bearing_sigma_deg=0.5,
                                                       outlier_frac=0.3))
        clean_bearings = {(m.tracklet_id, m.anchor_keyframe_idx):
                          m.bearing_body_deg for m in clean.measurements}
        differing = sum(
            1 for m in dirty.measurements
            if abs(clean_bearings[(m.tracklet_id, m.anchor_keyframe_idx)]
                   - m.bearing_body_deg) > 5.0)
        self.assertGreater(differing, 0.1 * len(dirty.measurements))

    def test_symmetric_pair_is_actually_symmetric(self):
        data = scenario.generate(scenario.symmetric_pair())
        np.testing.assert_allclose(data.true_east_m, [-2000.0, 2000.0])
        np.testing.assert_allclose(data.true_north_m, [0.0, 0.0])
        for pose in data.truth:
            self.assertAlmostEqual(pose.east_m, 0.0, places=6)
            self.assertAlmostEqual(pose.heading_deg, 0.0, places=6)

    def test_generation_deterministic(self):
        a = scenario.generate(scenario.harbor_loop())
        b = scenario.generate(scenario.harbor_loop())
        self.assertEqual(a.measurements, b.measurements)
        self.assertEqual(a.odometry, b.odometry)


if __name__ == "__main__":
    unittest.main()
