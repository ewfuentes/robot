import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo

PANO_W, PANO_H = 7680, 3840


def reference_direction(face_yaw_deg, x_norm, y_norm, fov_deg=90.0):
    """Independent reimplementation of the panorama_to_pinhole.py ray math.

    Returns (az_ccw_rad, el_down_rad) exactly as the render computes them.
    """
    fov = math.radians(fov_deg)
    fx = fy = 1.0 / math.tan(fov / 2.0)
    # col_frac = linspace(1, -1) over image x; row_frac = linspace(-1, 1).
    col_frac = 1.0 - 2.0 * x_norm / 1000.0
    row_frac = 2.0 * y_norm / 1000.0 - 1.0
    d = np.array([col_frac, row_frac * (fx / fy), fx])
    d /= np.linalg.norm(d)
    yaw = math.radians(face_yaw_deg)
    ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    d = ry @ d
    return math.atan2(d[0], d[2]), math.asin(d[1])


def reference_pano_px(az_ccw_rad, el_down_rad):
    x = ((math.pi - az_ccw_rad) / (2 * math.pi)) % 1.0 * PANO_W
    y = (el_down_rad / math.pi + 0.5) * PANO_H
    return x, y


class DirectionFromFacePxTest(unittest.TestCase):
    def test_matches_render_reference_on_grid(self):
        for face_yaw in (0, 90, 180, 270):
            for x_norm in (0, 137, 500, 862, 1000):
                for y_norm in (0, 250, 500, 750, 1000):
                    az_cw, el_up = geo.direction_from_face_px(
                        face_yaw, x_norm, y_norm)
                    x, y = geo.pano_px_from_direction(
                        az_cw, el_up, PANO_W, PANO_H)
                    az_ref, el_ref = reference_direction(
                        face_yaw, x_norm, y_norm)
                    x_ref, y_ref = reference_pano_px(az_ref, el_ref)
                    self.assertAlmostEqual(
                        x % PANO_W, x_ref % PANO_W, places=6,
                        msg=f"yaw={face_yaw} x={x_norm} y={y_norm}")
                    self.assertAlmostEqual(y, y_ref, places=6)

    def test_face_centers_land_at_expected_pano_columns(self):
        # Pano layout left-to-right: 180 | 90 | 0 | 270.
        expected = {0: 0.50, 90: 0.25, 180: 0.0, 270: 0.75}
        for face_yaw, frac in expected.items():
            az_cw, el_up = geo.direction_from_face_px(face_yaw, 500, 500)
            x, y = geo.pano_px_from_direction(az_cw, el_up, PANO_W, PANO_H)
            self.assertAlmostEqual(x % PANO_W, frac * PANO_W, places=6)
            self.assertAlmostEqual(y, PANO_H / 2.0, places=6)

    def test_face_center_azimuth_is_the_negated_face_yaw(self):
        # The face-yaw label is a RENDER parameter, not a camera-frame
        # azimuth: the camera-frame azimuth of a face centre is
        # -face_yaw mod 360, so faces 90/270 swap.
        for face_yaw, expected in ((0, 0.0), (90, 270.0),
                                   (180, 180.0), (270, 90.0)):
            self.assertAlmostEqual(
                geo.bearing_camera_deg(face_yaw, 500.0), expected,
                msg=f"face {face_yaw}")

    def test_bearing_increases_image_right_on_every_face(self):
        for face_yaw in (0, 90, 180, 270):
            a = geo.bearing_camera_deg(face_yaw, 400.0)
            b = geo.bearing_camera_deg(face_yaw, 600.0)
            self.assertGreater(
                float(geo.wrap_deg(b - a)), 0.0, msg=f"face {face_yaw}")

    def test_face_edges_are_half_fov(self):
        # Face 90 is centred on camera azimuth 270, so its edges are 225/315.
        self.assertAlmostEqual(geo.bearing_camera_deg(90, 0.0), 225.0)
        self.assertAlmostEqual(geo.bearing_camera_deg(90, 1000.0), 315.0)
        self.assertAlmostEqual(geo.bearing_camera_deg(0, 0.0), 315.0)
        self.assertAlmostEqual(geo.bearing_camera_deg(0, 1000.0), 45.0)

    def test_adjacent_faces_agree_at_shared_seam(self):
        # Face 0's left edge is the same physical direction as face 90's
        # right edge (face 90 looks CCW-left of forward).
        az_a, el_a = geo.direction_from_face_px(0, 0, 300)
        az_b, el_b = geo.direction_from_face_px(90, 1000, 300)
        self.assertAlmostEqual(az_a, az_b, places=9)
        self.assertAlmostEqual(el_a, el_b, places=9)

    def test_elevation_is_off_axis_correct(self):
        # Top of the face at the center column: 45 deg up for 90 deg FOV.
        _, el_up = geo.direction_from_face_px(0, 500, 0)
        self.assertAlmostEqual(el_up, 45.0, places=9)
        # Corner pixel is off-axis: asin(1/sqrt(3)) ~ 35.264 deg, not 45.
        _, el_corner = geo.direction_from_face_px(0, 0, 0)
        self.assertAlmostEqual(
            el_corner, math.degrees(math.asin(1 / math.sqrt(3))), places=6)

    def test_pano_px_direction_round_trip(self):
        for x in (0.0, 1234.5, 3840.0, 7679.0):
            for y in (10.0, 1920.0, 3000.0):
                az, el = geo.direction_from_pano_px(x, y, PANO_W, PANO_H)
                x2, y2 = geo.pano_px_from_direction(az, el, PANO_W, PANO_H)
                self.assertAlmostEqual(x % PANO_W, x2 % PANO_W, places=6)
                self.assertAlmostEqual(y, y2, places=6)

    def test_azimuth_of_pano_column(self):
        self.assertAlmostEqual(
            geo.azimuth_of_pano_column(PANO_W / 2.0, PANO_W), 0.0)
        self.assertAlmostEqual(geo.azimuth_of_pano_column(0.0, PANO_W), 180.0)
        self.assertAlmostEqual(
            geo.azimuth_of_pano_column(0.75 * PANO_W, PANO_W), 90.0)
        # Wraps unwrapped columns.
        self.assertAlmostEqual(
            geo.azimuth_of_pano_column(PANO_W * 1.5, PANO_W), 0.0)


class BBoxAnglesTest(unittest.TestCase):
    def test_full_face_box_width_is_fov(self):
        _, _, width = geo.bbox_angles(270, 0, 0, 1000, 1000)
        self.assertAlmostEqual(width, 90.0)

    def test_bbox_center_bearing_wraps(self):
        center, _, _ = geo.bbox_angles(270, 900, 400, 1000, 600)
        self.assertAlmostEqual(
            center, geo.bearing_camera_deg(270, 950.0))
        # Face 270 is centred on 90, so a box at its right edge sits above 90.
        self.assertGreater(center, 90.0)

    def test_bbox_elevation_matches_the_one_definition(self):
        # There is exactly one elevation formula, in direction_from_face_px;
        # bbox_angles must evaluate it at the bbox center, off-axis included.
        for face_yaw, xmin, ymin, xmax, ymax in (
                (0, 400, 100, 600, 300), (90, 0, 0, 200, 200),
                (270, 800, 700, 1000, 1000)):
            _, el, _ = geo.bbox_angles(face_yaw, xmin, ymin, xmax, ymax)
            _, el_ref = geo.direction_from_face_px(
                face_yaw, (xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
            self.assertAlmostEqual(el, el_ref, places=9)


class _Box:
    def __init__(self, face_yaw_deg, xmin, ymin, xmax, ymax):
        self.face_yaw_deg = face_yaw_deg
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax


class PanoBBoxTest(unittest.TestCase):
    def test_center_face_box_maps_to_center_band(self):
        x_min, y_min, x_max, y_max = geo.pano_bbox_from_face_bbox(
            0, 400, 350, 425, 445, PANO_W, PANO_H)
        self.assertLess(x_max, PANO_W / 2.0)
        self.assertGreater(x_min, 0.25 * PANO_W)
        self.assertLess(y_min, y_max)
        self.assertLess(y_max, PANO_H / 2.0)  # ymax 445 < 500 is above horizon

    def test_bbox_contains_its_corner_directions(self):
        for face_yaw in (0, 90, 270):
            x_min, y_min, x_max, y_max = geo.pano_bbox_from_face_bbox(
                face_yaw, 100, 200, 900, 800, PANO_W, PANO_H)
            for xn, yn in ((100, 200), (900, 200), (100, 800), (900, 800),
                           (500, 200), (500, 800)):
                az, el = geo.direction_from_face_px(face_yaw, xn, yn)
                x, y = geo.pano_px_from_direction(az, el, PANO_W, PANO_H)
                x_off = (x - x_min) % PANO_W
                self.assertLessEqual(x_off, (x_max - x_min) + 1e-6)
                self.assertGreaterEqual(y, y_min - 1e-6)
                self.assertLessEqual(y, y_max + 1e-6)

    def test_wrapping_box_at_pano_seam(self):
        # Face 180 renders at the pano's left/right edges; a box centered on
        # the face center straddles the seam and must come back unwrapped.
        x_min, y_min, x_max, y_max = geo.pano_bbox_from_face_bbox(
            180, 400, 450, 600, 550, PANO_W, PANO_H)
        self.assertLess(x_min, PANO_W)
        self.assertGreater(x_max, PANO_W)

    def test_seam_merged_union_stays_contiguous(self):
        # Two boxes meeting at the face-0/face-90 seam (az_cw = -45 deg,
        # pano x = 0.375 * W): face 0 left edge + face 90 right edge.
        boxes = [_Box(0, 0, 450, 100, 550), _Box(90, 900, 450, 1000, 550)]
        x_min, y_min, x_max, y_max = geo.pano_bbox_for_observation(
            boxes, PANO_W, PANO_H)
        width = x_max - x_min
        self.assertLess(width, 0.1 * PANO_W)  # contiguous, not wrapped-around
        seam_x = 0.375 * PANO_W
        self.assertLess(x_min, seam_x)
        self.assertGreater(x_max, seam_x)

    def test_signed_x_offset(self):
        self.assertAlmostEqual(geo.signed_x_offset(10.0, 0.0, PANO_W), 10.0)
        self.assertAlmostEqual(
            geo.signed_x_offset(PANO_W - 10.0, 0.0, PANO_W), -10.0)


class FrameChainTest(unittest.TestCase):
    def test_apply_mount_offset(self):
        # An object dead ahead of the vehicle: camera bearing equals the
        # mount offset, body bearing is zero.
        self.assertAlmostEqual(geo.apply_mount_offset(214.0, 214.0), 0.0)
        self.assertAlmostEqual(geo.apply_mount_offset(10.0, 214.0), 156.0)
        self.assertAlmostEqual(geo.apply_mount_offset(213.0, 214.0), 359.0)

    def test_column_zero_misuse_is_half_a_turn_out(self):
        # The trap MOUNT_OFFSET_CONVENTION documents: an offset reasoned in
        # the column-0 frame differs from the centre-column frame by exactly
        # 180 deg, for every column.
        for x in (0.0, 1000.0, 3000.0, 7000.0):
            centre_frame = geo.azimuth_of_pano_column(x, PANO_W)
            column0_frame = (x / PANO_W) * 360.0 % 360.0
            self.assertAlmostEqual(
                abs(float(geo.wrap_deg(centre_frame - column0_frame))),
                180.0, places=9)

    def test_body_world_round_trip(self):
        for heading in (0.0, 91.5, 359.0):
            for body in (0.0, 45.0, 300.0):
                world = geo.body_to_world_bearing_deg(heading, body)
                self.assertAlmostEqual(
                    float(geo.world_to_body_bearing_deg(heading, world)),
                    body)

    def test_body_to_world_vectorized(self):
        world = geo.body_to_world_bearing_deg(
            np.array([350.0, 10.0]), np.array([20.0, 355.0]))
        np.testing.assert_allclose(world, np.array([10.0, 5.0]))


class AngleHelpersTest(unittest.TestCase):
    def test_wrap_deg(self):
        self.assertAlmostEqual(float(geo.wrap_deg(181.0)), -179.0)
        self.assertAlmostEqual(float(geo.wrap_deg(-181.0)), 179.0)
        self.assertAlmostEqual(float(geo.wrap_deg(360.0)), 0.0)
        np.testing.assert_allclose(
            geo.wrap_deg(np.array([359.0, 1.0])), np.array([-1.0, 1.0]))

    def test_wrap_rad(self):
        self.assertAlmostEqual(float(geo.wrap_rad(np.pi)), -np.pi)
        self.assertAlmostEqual(float(geo.wrap_rad(-np.pi)), -np.pi)
        self.assertAlmostEqual(float(geo.wrap_rad(3 * np.pi / 2)),
                               -np.pi / 2)
        vals = np.linspace(-10.0, 10.0, 101)
        wrapped = geo.wrap_rad(vals)
        self.assertTrue(np.all(wrapped >= -np.pi))
        self.assertTrue(np.all(wrapped < np.pi))
        np.testing.assert_allclose(np.sin(wrapped), np.sin(vals), atol=1e-12)
        np.testing.assert_allclose(np.cos(wrapped), np.cos(vals), atol=1e-12)

    def test_circular_diff_across_wrap(self):
        self.assertAlmostEqual(float(geo.circular_diff_deg(359.0, 1.0)), -2.0)
        self.assertAlmostEqual(float(geo.circular_diff_deg(1.0, 359.0)), 2.0)

    def test_circular_mean_across_wrap(self):
        self.assertAlmostEqual(
            geo.circular_mean_deg([359.0, 1.0]), 0.0, places=6)
        self.assertAlmostEqual(
            geo.circular_mean_deg([89.0, 91.0]), 90.0, places=6)


class EnuTest(unittest.TestCase):
    def test_enu_roundtrip(self):
        anchor = (42.3544553, -71.0912108)
        lat, lon = 42.3591962, -71.0873143
        east, north = geo.enu_from_latlon(lat, lon, *anchor)
        self.assertGreater(north, 0)
        self.assertGreater(east, 0)
        lat2, lon2 = geo.latlon_from_enu(east, north, *anchor)
        self.assertAlmostEqual(lat, lat2, places=9)
        self.assertAlmostEqual(lon, lon2, places=9)

    def test_enu_scale(self):
        # One degree of latitude is ~111 km.
        _, north = geo.enu_from_latlon(43.0, -71.0, 42.0, -71.0)
        self.assertAlmostEqual(north, 111319.5, delta=1.0)

    def test_region_frame_hand_derived_fixture(self):
        """Expected values computed by hand from METERS_PER_DEG_LAT =
        111319.4908 m and cos(42 deg) = 0.7431448, NOT by the code under
        test."""
        frame = geo.RegionFrame(42.0, -71.0)
        east, north = frame.enu_from_latlon(42.01, -71.0)
        self.assertAlmostEqual(float(east), 0.0, places=6)
        self.assertAlmostEqual(float(north), 1113.194908, places=3)
        self.assertAlmostEqual(
            math.degrees(float(geo.compass_bearing_rad(east, north))) % 360.0,
            0.0, places=6)

        east, north = frame.enu_from_latlon(42.0, -70.99)
        # 0.01 deg lon * 111319.4908 * cos(42 deg) = 827.265 m (hand-computed)
        self.assertAlmostEqual(float(east), 827.265, delta=0.01)
        self.assertAlmostEqual(float(north), 0.0, places=6)

        east, north = frame.enu_from_latlon(41.99, -71.01)
        # atan(827.265 / 1113.195) = 36.615 deg (hand-computed), third
        # quadrant compass: 180 + 36.615 = 216.615 deg.
        bearing = math.degrees(
            float(geo.compass_bearing_rad(east, north))) % 360.0
        self.assertAlmostEqual(bearing, 216.615, delta=0.05)

    def test_region_frame_matches_scalar_helpers(self):
        frame = geo.RegionFrame(42.0, -71.0)
        lats = np.array([42.05, 41.9, 42.2, 41.78])
        lons = np.array([-70.8, -71.2, -71.05, -70.95])
        east_vec, north_vec = frame.enu_from_latlon(lats, lons)
        for i in range(len(lats)):
            east_s, north_s = geo.enu_from_latlon(
                lats[i], lons[i], 42.0, -71.0)
            self.assertAlmostEqual(float(east_vec[i]), east_s, places=6)
            self.assertAlmostEqual(float(north_vec[i]), north_s, places=6)

    def test_region_frame_round_trip_at_corners(self):
        frame = geo.RegionFrame(42.335, -70.99)
        corners_e = np.array([-25000.0, 25000.0, 25000.0, -25000.0])
        corners_n = np.array([-25000.0, -25000.0, 25000.0, 25000.0])
        lat, lon = frame.latlon_from_enu(corners_e, corners_n)
        east, north = frame.enu_from_latlon(lat, lon)
        np.testing.assert_allclose(east, corners_e, atol=1e-6)
        np.testing.assert_allclose(north, corners_n, atol=1e-6)

    def test_compass_bearing(self):
        self.assertAlmostEqual(geo.compass_bearing_deg(0.0, 1.0), 0.0)
        self.assertAlmostEqual(geo.compass_bearing_deg(1.0, 0.0), 90.0)
        self.assertAlmostEqual(geo.compass_bearing_deg(0.0, -1.0), 180.0)
        self.assertAlmostEqual(geo.compass_bearing_deg(-1.0, 0.0), 270.0)

    def test_bearing_unit_vector_inverts_compass_bearing(self):
        for bearing in (0.0, 45.0, 123.0, 300.0):
            east, north = geo.bearing_unit_vector(bearing)
            self.assertAlmostEqual(
                geo.compass_bearing_deg(east, north), bearing)
            self.assertAlmostEqual(math.hypot(east, north), 1.0)


class HaversineTest(unittest.TestCase):
    def test_one_degree_of_latitude(self):
        # 1 deg of latitude on the mean sphere: 2*pi*R/360 = 111194.93 m.
        d = geo.haversine_m(42.0, -71.0, 43.0, -71.0)
        self.assertAlmostEqual(
            d, 2 * math.pi * geo.MEAN_EARTH_RADIUS_M / 360.0, delta=0.01)

    def test_agrees_with_enu_at_small_scale(self):
        anchor = (42.35, -71.05)
        lat, lon = 42.36, -71.03
        east, north = geo.enu_from_latlon(lat, lon, *anchor)
        d_enu = math.hypot(east, north)
        d_hav = geo.haversine_m(*anchor, lat, lon)
        # Different earth models (WGS84 equatorial vs mean sphere) agree to
        # a few permille at 2 km.
        self.assertLess(abs(d_enu - d_hav) / d_hav, 5e-3)

    def test_zero_distance(self):
        self.assertEqual(geo.haversine_m(42.0, -71.0, 42.0, -71.0), 0.0)


if __name__ == "__main__":
    unittest.main()
