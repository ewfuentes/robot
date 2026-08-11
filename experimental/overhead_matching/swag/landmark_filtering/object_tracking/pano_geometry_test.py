import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
)

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
                    az_cw, el_up = pg.direction_from_face_px(
                        face_yaw, x_norm, y_norm)
                    x, y = pg.pano_px_from_direction(
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
            az_cw, el_up = pg.direction_from_face_px(face_yaw, 500, 500)
            x, y = pg.pano_px_from_direction(az_cw, el_up, PANO_W, PANO_H)
            self.assertAlmostEqual(x % PANO_W, frac * PANO_W, places=6)
            self.assertAlmostEqual(y, PANO_H / 2.0, places=6)

    def test_adjacent_faces_agree_at_shared_seam(self):
        # Face 0's left edge is the same physical direction as face 90's
        # right edge (face 90 looks CCW-left of forward).
        az_a, el_a = pg.direction_from_face_px(0, 0, 300)
        az_b, el_b = pg.direction_from_face_px(90, 1000, 300)
        self.assertAlmostEqual(az_a, az_b, places=9)
        self.assertAlmostEqual(el_a, el_b, places=9)

    def test_elevation_at_face_center_column(self):
        # Top of the face at the center column: 45 deg up for 90 deg FOV.
        _, el_up = pg.direction_from_face_px(0, 500, 0)
        self.assertAlmostEqual(el_up, 45.0, places=9)
        # Corner pixel is off-axis: asin(1/sqrt(3)) ~ 35.264 deg, not 45.
        _, el_corner = pg.direction_from_face_px(0, 0, 0)
        self.assertAlmostEqual(
            el_corner, math.degrees(math.asin(1 / math.sqrt(3))), places=6)

    def test_pano_px_direction_round_trip(self):
        for x in (0.0, 1234.5, 3840.0, 7679.0):
            for y in (10.0, 1920.0, 3000.0):
                az, el = pg.direction_from_pano_px(x, y, PANO_W, PANO_H)
                x2, y2 = pg.pano_px_from_direction(az, el, PANO_W, PANO_H)
                self.assertAlmostEqual(x % PANO_W, x2 % PANO_W, places=6)
                self.assertAlmostEqual(y, y2, places=6)


class _Box:
    def __init__(self, face_yaw_deg, xmin, ymin, xmax, ymax):
        self.face_yaw_deg = face_yaw_deg
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax


class PanoBBoxTest(unittest.TestCase):
    def test_center_face_box_maps_to_center_band(self):
        x_min, y_min, x_max, y_max = pg.pano_bbox_from_face_bbox(
            0, 400, 350, 425, 445, PANO_W, PANO_H)
        # Box right of face center (x_norm > 500 would be right; 400-425 is
        # left of center) -> pano x left of pano center.
        self.assertLess(x_max, PANO_W / 2.0)
        self.assertGreater(x_min, 0.25 * PANO_W)
        self.assertLess(y_min, y_max)
        self.assertLess(y_max, PANO_H / 2.0)  # ymax 445 < 500 is above horizon

    def test_bbox_contains_its_corner_directions(self):
        for face_yaw in (0, 90, 270):
            x_min, y_min, x_max, y_max = pg.pano_bbox_from_face_bbox(
                face_yaw, 100, 200, 900, 800, PANO_W, PANO_H)
            for xn, yn in ((100, 200), (900, 200), (100, 800), (900, 800),
                           (500, 200), (500, 800)):
                az, el = pg.direction_from_face_px(face_yaw, xn, yn)
                x, y = pg.pano_px_from_direction(az, el, PANO_W, PANO_H)
                x_off = pg.x_offset_in_window(x, x_min, PANO_W)
                self.assertLessEqual(x_off, (x_max - x_min) + 1e-6)
                self.assertGreaterEqual(y, y_min - 1e-6)
                self.assertLessEqual(y, y_max + 1e-6)

    def test_wrapping_box_at_pano_seam(self):
        # Face 180 renders at the pano's left/right edges; a box centered on
        # the face center straddles the seam and must come back unwrapped.
        x_min, y_min, x_max, y_max = pg.pano_bbox_from_face_bbox(
            180, 400, 450, 600, 550, PANO_W, PANO_H)
        self.assertLess(x_min, PANO_W)
        self.assertGreater(x_max, PANO_W)

    def test_seam_merged_union_stays_contiguous(self):
        # Two boxes meeting at the face-0/face-90 seam (az_cw = -45 deg,
        # pano x = 0.375 * W): face 0 left edge + face 90 right edge.
        boxes = [_Box(0, 0, 450, 100, 550), _Box(90, 900, 450, 1000, 550)]
        x_min, y_min, x_max, y_max = pg.pano_bbox_for_observation(
            boxes, PANO_W, PANO_H)
        width = x_max - x_min
        self.assertLess(width, 0.1 * PANO_W)  # contiguous, not wrapped-around
        seam_x = 0.375 * PANO_W
        self.assertLess(x_min, seam_x)
        self.assertGreater(x_max, seam_x)


if __name__ == "__main__":
    unittest.main()
