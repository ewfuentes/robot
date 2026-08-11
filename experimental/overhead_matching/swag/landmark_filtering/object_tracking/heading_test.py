import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    heading,
)


class HeadingModelTest(unittest.TestCase):
    def test_straight_line_heading(self):
        t = np.arange(0.0, 100.0, 1.0)
        east = 2.0 * t   # due-east travel at 2 m/s
        north = np.zeros_like(t)
        model = heading.heading_model_from_positions(east, north, t)
        for q in (10.0, 50.0, 90.0):
            self.assertAlmostEqual(model.at(q) % 360.0, 90.0, places=6)

    def test_circular_arc_heading_tracks_tangent(self):
        # CCW circle, radius 100 m, angular rate 1 deg/s starting due north.
        t = np.arange(0.0, 180.0, 1.0)
        theta = np.radians(t)
        east = 100.0 * np.cos(theta)
        north = 100.0 * np.sin(theta)
        model = heading.heading_model_from_positions(
            east, north, t, min_displacement_m=3.0, smooth_window_s=0.0)
        # Tangent compass heading at angle theta: atan2(-sin, cos).
        for q in (30.0, 90.0, 150.0):
            expected = math.degrees(math.atan2(
                -math.sin(math.radians(q)), math.cos(math.radians(q))))
            diff = (model.at(q) - expected + 180.0) % 360.0 - 180.0
            self.assertLess(abs(diff), 3.0, msg=f"t={q}")

    def test_heading_unwraps_across_north(self):
        # Slow CW turn through north: heading 350 -> 10 must interpolate
        # through 360, not swing backward through 180.
        t = np.arange(0.0, 40.0, 1.0)
        h = np.radians(350.0 + t)  # compass heading 350 -> 30
        east = np.cumsum(np.sin(h)) * 5.0
        north = np.cumsum(np.cos(h)) * 5.0
        model = heading.heading_model_from_positions(
            east, north, t, min_displacement_m=3.0, smooth_window_s=0.0)
        deltas = np.diff(model.at(t))
        self.assertTrue(np.all(deltas > -1.0))
        self.assertGreater(model.delta(35.0, 5.0), 20.0)

    def test_stationary_prefix_holds_first_heading(self):
        t = np.arange(0.0, 60.0, 1.0)
        east = np.where(t < 30.0, 0.0, (t - 30.0) * 3.0)
        north = np.zeros_like(t)
        model = heading.heading_model_from_positions(east, north, t)
        # Before motion begins the model should return the first defined
        # course (due east), not something arbitrary.
        self.assertAlmostEqual(model.at(0.0) % 360.0, 90.0, places=6)

    def test_never_moved_returns_constant(self):
        t = np.arange(0.0, 10.0, 1.0)
        model = heading.heading_model_from_positions(
            np.zeros_like(t), np.zeros_like(t), t)
        self.assertEqual(model.delta(9.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
