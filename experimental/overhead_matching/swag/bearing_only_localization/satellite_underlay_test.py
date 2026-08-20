import math
import unittest

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    satellite_underlay as su,
)


class TileMathTest(unittest.TestCase):
    """Anchored on values the web-mercator definition fixes, not on a snapshot."""

    def test_origin_tile_at_zoom_zero(self):
        self.assertEqual(su.tile_of(0.0, 0.0, 0), (0, 0))

    def test_zoom_one_quadrants(self):
        # (lat>0, lon<0) is the north-west quadrant: x=0, y=0.
        self.assertEqual(su.tile_of(45.0, -90.0, 1), (0, 0))
        self.assertEqual(su.tile_of(45.0, 90.0, 1), (1, 0))
        self.assertEqual(su.tile_of(-45.0, -90.0, 1), (0, 1))
        self.assertEqual(su.tile_of(-45.0, 90.0, 1), (1, 1))

    def test_nw_corner_of_the_world_tile(self):
        lat, lon = su.tile_nw_corner(0, 0, 0)
        self.assertAlmostEqual(lon, -180.0, places=6)
        self.assertAlmostEqual(lat, 85.0511287798, places=6)

    def test_corner_and_index_are_inverse(self):
        for lat, lon, z in ((42.3356, -70.9887, 14), (44.2601, -71.3198, 18),
                            (-33.87, 151.21, 12)):
            x, y = su.tile_of(lat, lon, z)
            nw_lat, nw_lon = su.tile_nw_corner(x, y, z)
            se_lat, se_lon = su.tile_nw_corner(x + 1, y + 1, z)
            self.assertLessEqual(nw_lon, lon)
            self.assertGreaterEqual(se_lon, lon)
            self.assertGreaterEqual(nw_lat, lat)
            self.assertLessEqual(se_lat, lat)

    def test_span_is_ordered_and_inclusive(self):
        x0, y0, x1, y1 = su.tile_span(42.30, 42.36, -71.10, -70.98, 14)
        self.assertLessEqual(x0, x1)
        self.assertLessEqual(y0, y1)
        # every corner of the box must fall inside the span
        for lat in (42.30, 42.36):
            for lon in (-71.10, -70.98):
                x, y = su.tile_of(lat, lon, 14)
                self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1)

    def test_tile_count_quadruples_per_zoom_level(self):
        a = su.layer_plan("t", 42.30, 42.36, -71.10, -70.98, 13)["n_tiles"]
        b = su.layer_plan("t", 42.30, 42.36, -71.10, -70.98, 14)["n_tiles"]
        self.assertGreater(b, 2 * a)      # 4x modulo boundary alignment


class FitZoomTest(unittest.TestCase):
    """The knob that makes one tool serve a 0.4 km track and an 18 km one."""

    BOX = (42.30, 42.42, -71.12, -70.94)      # ~13 x 15 km

    def test_returns_the_requested_zoom_when_it_fits(self):
        plan = su.fit_zoom("f", *self.BOX, 18, 10 ** 6)
        self.assertEqual(plan["zoom"], 18)
        self.assertFalse(plan["capped"])

    def test_lowers_the_zoom_to_fit_and_says_so(self):
        plan = su.fit_zoom("f", *self.BOX, 18, 400)
        self.assertLess(plan["zoom"], 18)
        self.assertLessEqual(plan["n_tiles"], 400)
        self.assertTrue(plan["capped"])

    def test_picks_the_highest_zoom_that_fits(self):
        budget = 400
        plan = su.fit_zoom("f", *self.BOX, 18, budget)
        harder = su.layer_plan("f", *self.BOX, plan["zoom"] + 1)
        self.assertGreater(harder["n_tiles"], budget)


class EnuBoundsTest(unittest.TestCase):

    def test_bounds_contain_the_anchor_and_are_ordered(self):
        frame = geodesy.RegionFrame(44.26017, -71.3198)
        x, y = su.tile_of(44.26017, -71.3198, 16)
        e0, e1, n0, n1 = su.enu_bounds_of_tiles(x, y, x, y, 16, frame)
        self.assertLess(e0, e1)
        self.assertLess(n0, n1)
        self.assertTrue(e0 <= 0.0 <= e1, f"anchor east outside {e0}..{e1}")
        self.assertTrue(n0 <= 0.0 <= n1, f"anchor north outside {n0}..{n1}")

    def test_one_tile_is_about_the_expected_ground_size(self):
        lat = 44.26017
        frame = geodesy.RegionFrame(lat, -71.3198)
        x, y = su.tile_of(lat, -71.3198, 16)
        e0, e1, _, _ = su.enu_bounds_of_tiles(x, y, x, y, 16, frame)
        expected = 156543.03392 * math.cos(math.radians(lat)) / (2 ** 16) * 256
        self.assertAlmostEqual((e1 - e0) / expected, 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
