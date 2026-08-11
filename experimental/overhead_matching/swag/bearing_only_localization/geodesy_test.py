import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
)
from experimental.overhead_matching.swag.landmark_filtering import (
    bearing_geometry as bg,
)


class GoldenConventionTest(unittest.TestCase):
    """T-U3: hand-derived fixture. Expected values below are computed by hand
    from METERS_PER_DEG_LAT = 111319.4908 m and cos(42 deg) = 0.7431448, NOT
    by the code under test."""

    def setUp(self):
        self.frame = geodesy.RegionFrame(42.0, -71.0)

    def test_point_due_north(self):
        east, north = self.frame.enu_from_latlon(42.01, -71.0)
        self.assertAlmostEqual(float(east), 0.0, places=6)
        self.assertAlmostEqual(float(north), 1113.194908, places=3)
        bearing = math.degrees(geodesy.compass_bearing_rad(east, north))
        self.assertAlmostEqual(bearing % 360.0, 0.0, places=6)

    def test_point_due_east(self):
        east, north = self.frame.enu_from_latlon(42.0, -70.99)
        # 0.01 deg lon * 111319.4908 * cos(42 deg) = 827.265 m (hand-computed)
        self.assertAlmostEqual(float(east), 827.265, delta=0.01)
        self.assertAlmostEqual(float(north), 0.0, places=6)
        bearing = math.degrees(geodesy.compass_bearing_rad(east, north))
        self.assertAlmostEqual(bearing, 90.0, places=6)

    def test_point_southwest(self):
        east, north = self.frame.enu_from_latlon(41.99, -71.01)
        self.assertAlmostEqual(float(east), -827.265, delta=0.01)
        self.assertAlmostEqual(float(north), -1113.195, delta=0.01)
        # atan(827.265 / 1113.195) = 36.615 deg (hand-computed), third
        # quadrant compass: 180 + 36.615 = 216.615 deg.
        bearing = math.degrees(geodesy.compass_bearing_rad(east, north)) % 360.0
        self.assertAlmostEqual(bearing, 216.615, delta=0.05)

    def test_matches_scalar_bearing_geometry(self):
        lats = np.array([42.05, 41.9, 42.2, 41.78])
        lons = np.array([-70.8, -71.2, -71.05, -70.95])
        east_vec, north_vec = self.frame.enu_from_latlon(lats, lons)
        for i in range(len(lats)):
            east_scalar, north_scalar = bg.enu_from_latlon(
                lats[i], lons[i], 42.0, -71.0)
            self.assertAlmostEqual(float(east_vec[i]), east_scalar, places=6)
            self.assertAlmostEqual(float(north_vec[i]), north_scalar, places=6)
            compass = bg.compass_bearing_deg(east_scalar, north_scalar)
            ours = math.degrees(geodesy.compass_bearing_rad(
                east_vec[i], north_vec[i])) % 360.0
            self.assertAlmostEqual(ours, compass, places=6)


class RoundTripTest(unittest.TestCase):
    """T-U4: latlon <-> ENU round trips at 25 km region corners."""

    def test_round_trip_at_corners(self):
        frame = geodesy.RegionFrame(42.335, -70.99)
        corners_e = np.array([-25000.0, 25000.0, 25000.0, -25000.0])
        corners_n = np.array([-25000.0, -25000.0, 25000.0, 25000.0])
        lat, lon = frame.latlon_from_enu(corners_e, corners_n)
        east, north = frame.enu_from_latlon(lat, lon)
        np.testing.assert_allclose(east, corners_e, atol=1e-6)
        np.testing.assert_allclose(north, corners_n, atol=1e-6)


class WrapTest(unittest.TestCase):
    def test_wrap_rad(self):
        self.assertAlmostEqual(float(geodesy.wrap_rad(np.pi)), -np.pi)
        self.assertAlmostEqual(float(geodesy.wrap_rad(-np.pi)), -np.pi)
        self.assertAlmostEqual(float(geodesy.wrap_rad(3 * np.pi / 2)),
                               -np.pi / 2)
        vals = np.linspace(-10.0, 10.0, 101)
        wrapped = geodesy.wrap_rad(vals)
        self.assertTrue(np.all(wrapped >= -np.pi))
        self.assertTrue(np.all(wrapped < np.pi))
        np.testing.assert_allclose(np.sin(wrapped), np.sin(vals), atol=1e-12)
        np.testing.assert_allclose(np.cos(wrapped), np.cos(vals), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
