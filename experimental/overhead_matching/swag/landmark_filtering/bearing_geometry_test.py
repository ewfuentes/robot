import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    bearing_geometry as bg,
)
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)


class BearingGeometryTest(unittest.TestCase):
    """Corrected 2026-08-19: these pinned the pre-delegation camera frame.

    The face-yaw label is a RENDER parameter, not a camera-frame azimuth. In the
    verified convention the panorama runs `180 | 90 | 0 | 270` left to right, so
    the camera-frame azimuth of a face centre is `-face_yaw mod 360`: faces 0 and
    180 are unchanged and faces 90/270 swap. These assertions previously read the
    face label straight back out, which is exactly what hid the error.
    """

    def test_face_center_is_the_negated_face_yaw(self):
        for face_yaw, expected in ((0, 0.0), (90, 270.0),
                                   (180, 180.0), (270, 90.0)):
            self.assertAlmostEqual(
                bg.bearing_camera_deg(face_yaw, 500.0), expected,
                msg=f"face {face_yaw}")

    def test_matches_pano_geometry_exactly(self):
        """The whole point of delegating: there is one definition, not two."""
        for face_yaw in (0, 90, 180, 270):
            for x in (0.0, 250.0, 500.0, 750.0, 1000.0):
                self.assertAlmostEqual(
                    bg.bearing_camera_deg(face_yaw, x),
                    pg.direction_from_face_px(face_yaw, x, 0.0)[0],
                    msg=f"face {face_yaw} x {x}")

    def test_bearing_increases_image_right_on_every_face(self):
        for face_yaw in (0, 90, 180, 270):
            a = bg.bearing_camera_deg(face_yaw, 400.0)
            b = bg.bearing_camera_deg(face_yaw, 600.0)
            self.assertGreater(bg.wrap_deg(b - a), 0.0, msg=f"face {face_yaw}")

    def test_face_edges_are_half_fov(self):
        # Face 90 is centred on camera azimuth 270, so its edges are 225/315.
        self.assertAlmostEqual(bg.bearing_camera_deg(90, 0.0), 225.0)
        self.assertAlmostEqual(bg.bearing_camera_deg(90, 1000.0), 315.0)
        # Face 0 is unchanged by the correction: its left edge still wraps to 315.
        self.assertAlmostEqual(bg.bearing_camera_deg(0, 0.0), 315.0)
        self.assertAlmostEqual(bg.bearing_camera_deg(0, 1000.0), 45.0)

    def test_full_face_box_width_is_fov(self):
        _, _, width = bg.bbox_angles(270, 0, 0, 1000, 1000)
        self.assertAlmostEqual(width, 90.0)

    def test_bbox_center_bearing_wraps(self):
        center, _, _ = bg.bbox_angles(270, 900, 400, 1000, 600)
        self.assertAlmostEqual(
            center, bg.bearing_camera_deg(270, 950.0))
        # Face 270 is centred on 90, so a box at its right edge sits above 90.
        self.assertGreater(center, 90.0)

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
