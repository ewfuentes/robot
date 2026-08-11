import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    bearing_geometry as bg,
)


class BearingGeometryTest(unittest.TestCase):
    def test_face_center_is_face_yaw(self):
        for face_yaw in (0, 90, 180, 270):
            self.assertAlmostEqual(
                bg.bearing_camera_deg(face_yaw, 500.0), float(face_yaw))

    def test_face_edges_are_half_fov(self):
        self.assertAlmostEqual(bg.bearing_camera_deg(90, 0.0), 45.0)
        self.assertAlmostEqual(bg.bearing_camera_deg(90, 1000.0), 135.0)
        # Left edge of face 0 wraps around to 315.
        self.assertAlmostEqual(bg.bearing_camera_deg(0, 0.0), 315.0)

    def test_full_face_box_width_is_fov(self):
        _, _, width = bg.bbox_angles(270, 0, 0, 1000, 1000)
        self.assertAlmostEqual(width, 90.0)

    def test_bbox_center_bearing_wraps(self):
        center, _, _ = bg.bbox_angles(270, 900, 400, 1000, 600)
        self.assertAlmostEqual(
            center, bg.bearing_camera_deg(270, 950.0))
        self.assertGreater(center, 270.0)

    def test_elevation_sign(self):
        # y grows downward: y=0 is the top of the image, elevation positive.
        self.assertAlmostEqual(bg.elevation_deg(0.0), 45.0)
        self.assertAlmostEqual(bg.elevation_deg(1000.0), -45.0)
        self.assertAlmostEqual(bg.elevation_deg(500.0), 0.0)

    def test_wrap_deg(self):
        self.assertAlmostEqual(float(bg.wrap_deg(181.0)), -179.0)
        self.assertAlmostEqual(float(bg.wrap_deg(-181.0)), 179.0)
        self.assertAlmostEqual(float(bg.wrap_deg(360.0)), 0.0)
        np.testing.assert_allclose(
            bg.wrap_deg(np.array([359.0, 1.0])), np.array([-1.0, 1.0]))

    def test_circular_diff_across_wrap(self):
        self.assertAlmostEqual(float(bg.circular_diff_deg(359.0, 1.0)), -2.0)
        self.assertAlmostEqual(float(bg.circular_diff_deg(1.0, 359.0)), 2.0)

    def test_circular_mean_across_wrap(self):
        self.assertAlmostEqual(
            bg.circular_mean_deg([359.0, 1.0]), 0.0, places=6)
        self.assertAlmostEqual(
            bg.circular_mean_deg([89.0, 91.0]), 90.0, places=6)

    def test_enu_roundtrip(self):
        anchor = (42.3544553, -71.0912108)
        lat, lon = 42.3591962, -71.0873143
        east, north = bg.enu_from_latlon(lat, lon, *anchor)
        # Northeast of the anchor along this trajectory.
        self.assertGreater(north, 0)
        self.assertGreater(east, 0)
        lat2, lon2 = bg.latlon_from_enu(east, north, *anchor)
        self.assertAlmostEqual(lat, lat2, places=9)
        self.assertAlmostEqual(lon, lon2, places=9)

    def test_enu_scale(self):
        # One degree of latitude is ~111 km.
        _, north = bg.enu_from_latlon(43.0, -71.0, 42.0, -71.0)
        self.assertAlmostEqual(north, 111319.5, delta=1.0)

    def test_compass_bearing(self):
        self.assertAlmostEqual(bg.compass_bearing_deg(0.0, 1.0), 0.0)
        self.assertAlmostEqual(bg.compass_bearing_deg(1.0, 0.0), 90.0)
        self.assertAlmostEqual(bg.compass_bearing_deg(0.0, -1.0), 180.0)
        self.assertAlmostEqual(bg.compass_bearing_deg(-1.0, 0.0), 270.0)

    def test_bearing_unit_vector_inverts_compass_bearing(self):
        for bearing in (0.0, 45.0, 123.0, 300.0):
            east, north = bg.bearing_unit_vector(bearing)
            self.assertAlmostEqual(
                bg.compass_bearing_deg(east, north), bearing)
            self.assertAlmostEqual(math.hypot(east, north), 1.0)


if __name__ == "__main__":
    unittest.main()
