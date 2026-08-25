import math
import pickle
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    scenario,
)


class TruthTrajectoryTest(unittest.TestCase):
    def test_spacing_and_courses(self):
        cfg = scenario.l_turn(max_visible_range_m=10000.0, speed_mps=10.0, keyframe_period_s=2.0)
        data = scenario.generate(cfg)
        step = cfg.speed_mps * cfg.keyframe_period_s
        for a, b in zip(data.truth[:-1], data.truth[1:]):
            d = math.hypot(b.east_m - a.east_m, b.north_m - a.north_m)
            self.assertLessEqual(d, step + 1e-6)
        # First leg courses due east (090), second leg due north (000).
        self.assertAlmostEqual(
            data.truth[0].course_world_cw_deg, 90.0, places=6)
        self.assertAlmostEqual(
            data.truth[-1].course_world_cw_deg, 0.0, places=6)

    def test_endpoints(self):
        cfg = scenario.straight_leg(max_visible_range_m=10000.0, speed_mps=20.0, keyframe_period_s=5.0)
        data = scenario.generate(cfg)
        self.assertAlmostEqual(data.truth[0].east_m, -1000.0, places=6)
        self.assertAlmostEqual(data.truth[0].north_m, -500.0, places=6)
        # 2000 m at 100 m/step -> 21 keyframes ending at the far waypoint.
        self.assertEqual(data.n_keyframes, 21)
        self.assertAlmostEqual(data.truth[-1].east_m, 1000.0, places=6)


class GeneratedInputConsistencyTest(unittest.TestCase):
    def test_zero_noise_odometry_dead_reckons_to_truth(self):
        """Rotate-then-move increments reconstruct the truth trajectory."""
        cfg = scenario.harbor_loop(max_visible_range_m=10000.0, odom_sigma_m=0.0, dyaw_sigma_deg=0.0)
        data = scenario.generate(cfg)
        east, north = data.truth[0].east_m, data.truth[0].north_m
        course = math.radians(data.truth[0].course_world_cw_deg)
        for o in data.odometry:
            course += o.delta_yaw_cw_rad
            east += o.forward_m * math.sin(course) - o.left_m * math.cos(course)
            north += o.forward_m * math.cos(course) + o.left_m * math.sin(course)
        self.assertAlmostEqual(east, data.truth[-1].east_m, places=6)
        self.assertAlmostEqual(north, data.truth[-1].north_m, places=6)

    def test_bearings_reconstruct_world_direction(self):
        """Forward bearing + trajectory course reconstructs world bearing."""
        cfg = scenario.harbor_loop(max_visible_range_m=10000.0, bearing_sigma_deg=1e-3)
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
            reconstructed = (meas.bearing_forward_cw_deg
                             + pose.course_world_cw_deg)
            diff = math.degrees(abs(float(geo.wrap_rad(
                math.radians(reconstructed - expected_world)))))
            self.assertLess(diff, 0.02)

    def test_epoch_cadence_and_stagger(self):
        cfg = scenario.harbor_loop(max_visible_range_m=10000.0, epoch_length_keyframes=6)
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
        cfg = scenario.harbor_loop(max_visible_range_m=10000.0)
        data = scenario.generate(cfg)
        for lm_id in data.landmark_ids:
            table = data.tables[f"trk_{lm_id}"]
            self.assertEqual(len(table.entries), 1)
            self.assertEqual(table.entries[0].landmark_id, lm_id)
            self.assertEqual(table.entries[0].log_lr, cfg.identity_clip)
            self.assertEqual(table.matcher_version,
                             scenario.MATCHER_VERSION)

    def test_clutter_only_tables_empty(self):
        cfg = scenario.harbor_loop(max_visible_range_m=10000.0, clutter_only=True)
        data = scenario.generate(cfg)
        for table in data.tables.values():
            self.assertEqual(table.entries, [])

    def test_crab_moves_forward_bearings_but_not_truth_course_or_increments(self):
        """Crab is an unobserved course-to-forward mismatch, not GPS truth."""
        unbiased = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0, course_bias_deg=0.0))
        biased = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0, course_bias_deg=7.0))
        self.assertEqual(unbiased.odometry, biased.odometry)
        for a, b in zip(unbiased.truth, biased.truth):
            self.assertEqual(a.course_world_cw_deg, b.course_world_cw_deg)
        unbiased_by_key = {
            (item.tracklet_id, item.anchor_keyframe_idx): item
            for item in unbiased.measurements}
        for item in biased.measurements:
            reference = unbiased_by_key[
                (item.tracklet_id, item.anchor_keyframe_idx)]
            self.assertAlmostEqual(
                float(geo.circular_diff_deg(
                    item.bearing_forward_cw_deg,
                    reference.bearing_forward_cw_deg)),
                -7.0, places=6)

    def test_mismatch_knobs_skew_increments_not_declarations(self):
        """gyro_bias_deg_per_hr and odom_scale_error must corrupt the
        emitted increments while the declared sigmas stay unchanged — the
        filter is NOT told (T-F13's contract with the generator)."""
        clean = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, odom_sigma_m=0.0, dyaw_sigma_deg=0.0))
        skewed = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, odom_sigma_m=0.0, dyaw_sigma_deg=0.0,
            gyro_bias_deg_per_hr=3600.0, odom_scale_error=0.1))
        bias_per_step = math.radians(
            3600.0 / 3600.0 * clean.config.keyframe_period_s)
        for a, b in zip(clean.odometry, skewed.odometry):
            self.assertAlmostEqual(b.delta_yaw_cw_rad - a.delta_yaw_cw_rad, bias_per_step,
                                   places=9)
            self.assertAlmostEqual(b.forward_m, 1.1 * a.forward_m, places=9)
            self.assertEqual(b.sigma_m, a.sigma_m)
            self.assertEqual(b.sigma_yaw_rad, a.sigma_yaw_rad)

    def test_filter_catalog_uses_the_manifest_landmark_sigma(self):
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0))
        configured = {landmark.position_sigma_m
                      for landmark in data.config.landmarks}
        self.assertEqual(len(configured), 1)
        np.testing.assert_allclose(
            data.catalog.position_sigma_m, configured.pop())

    def test_catalog_error_moves_only_the_filters_copy(self):
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, catalog_position_sigma_m=25.0))
        offsets = np.hypot(data.catalog.east_m - data.true_east_m,
                           data.catalog.north_m - data.true_north_m)
        self.assertTrue(np.all(offsets > 0.0))
        self.assertLess(float(np.max(offsets)), 200.0)
        np.testing.assert_allclose(data.catalog.position_sigma_m, 25.0)

    def test_outliers_are_injected_at_the_requested_rate(self):
        clean = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0, bearing_sigma_deg=0.5))
        dirty = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0, bearing_sigma_deg=0.5,
                                                       outlier_frac=0.3))
        clean_bearings = {(m.tracklet_id, m.anchor_keyframe_idx):
                          m.bearing_forward_cw_deg for m in clean.measurements}
        differing = sum(
            1 for m in dirty.measurements
            if abs(clean_bearings[(m.tracklet_id, m.anchor_keyframe_idx)]
                   - m.bearing_forward_cw_deg) > 5.0)
        self.assertGreater(differing, 0.1 * len(dirty.measurements))

    def test_symmetric_pair_is_actually_symmetric(self):
        data = scenario.generate(scenario.symmetric_pair(max_visible_range_m=10000.0))
        np.testing.assert_allclose(data.true_east_m, [-2000.0, 2000.0])
        np.testing.assert_allclose(data.true_north_m, [0.0, 0.0])
        for pose in data.truth:
            self.assertAlmostEqual(pose.east_m, 0.0, places=6)
            self.assertAlmostEqual(
                pose.course_world_cw_deg, 0.0, places=6)

    def test_kidnap_preserves_measurement_bytes_and_residuals(self):
        data = scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, bearing_sigma_deg=4.0,
            bearing_bias_deg=2.0, outlier_frac=0.3))
        at_keyframe = data.n_keyframes // 2
        moved = scenario.apply_kidnap(
            data, at_keyframe, east_m=1400.0, north_m=-900.0)

        before_pre = [item for item in data.measurements
                      if item.anchor_keyframe_idx < at_keyframe]
        after_pre = [item for item in moved.measurements
                     if item.anchor_keyframe_idx < at_keyframe]
        self.assertEqual(pickle.dumps(after_pre), pickle.dumps(before_pre))

        def residuals(measurements, truth):
            truth_by_keyframe = {item.keyframe_idx: item for item in truth}
            values = []
            for measurement in measurements:
                if measurement.anchor_keyframe_idx < at_keyframe:
                    continue
                pose = truth_by_keyframe[measurement.anchor_keyframe_idx]
                index = data.catalog.index_of(
                    measurement.tracklet_id.removeprefix("trk_"))
                world = math.atan2(
                    float(data.true_east_m[index]) - pose.east_m,
                    float(data.true_north_m[index]) - pose.north_m)
                forward = math.radians(
                    pose.course_world_cw_deg + data.config.course_bias_deg)
                ideal = float(geo.wrap_rad(world - forward))
                values.append(float(geo.wrap_rad(
                    math.radians(measurement.bearing_forward_cw_deg)
                    - ideal)))
            return np.asarray(values)

        original_residuals = residuals(data.measurements, data.truth)
        moved_residuals = residuals(moved.measurements, moved.truth)
        self.assertGreater(original_residuals.size, 0)
        np.testing.assert_allclose(
            moved_residuals, original_residuals, atol=1e-12)

    def test_generation_deterministic(self):
        a = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0))
        b = scenario.generate(scenario.harbor_loop(max_visible_range_m=10000.0))
        self.assertEqual(a.measurements, b.measurements)
        self.assertEqual(a.odometry, b.odometry)


if __name__ == "__main__":
    unittest.main()
