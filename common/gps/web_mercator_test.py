
import unittest
import math

from common.gps import web_mercator
from common.gps.web_mercator import EARTH_RADIUS_M, get_meters_per_pixel


class TestGetMetersPerPixel(unittest.TestCase):
    """Verify get_meters_per_pixel against known values."""

    def test_equator_zoom_20(self):
        """At equator, meters_per_pixel has an exact formula."""
        lat = 0.0
        zoom = 20
        earth_circumference_m = 2 * math.pi * EARTH_RADIUS_M
        map_size_px = 2 ** (8 + zoom)
        expected = earth_circumference_m / map_size_px

        result = get_meters_per_pixel(lat, zoom)
        self.assertAlmostEqual(result, expected, places=6)

    def test_lat_60_is_half_equator(self):
        """At lat=60°, cos(60°) = 0.5, so meters_per_pixel is half of equator."""
        zoom = 20
        equator_mpp = get_meters_per_pixel(0.0, zoom)
        lat60_mpp = get_meters_per_pixel(60.0, zoom)

        # cos(60°) = 0.5
        expected_ratio = 0.5
        actual_ratio = lat60_mpp / equator_mpp

        self.assertAlmostEqual(actual_ratio, expected_ratio, places=6)

    def test_cosine_scaling(self):
        """Verify cos(lat) scaling at various latitudes."""
        zoom = 18
        equator_mpp = get_meters_per_pixel(0.0, zoom)

        test_lats = [30.0, 45.0, 60.0, 75.0]
        for lat in test_lats:
            expected_ratio = math.cos(math.radians(lat))
            actual_mpp = get_meters_per_pixel(lat, zoom)
            actual_ratio = actual_mpp / equator_mpp

            self.assertAlmostEqual(
                actual_ratio, expected_ratio, places=6,
                msg=f"At lat={lat}°"
            )

    def test_zoom_level_scaling(self):
        """Each zoom level doubles the map size, halving meters_per_pixel."""
        lat = 40.0
        mpp_z18 = get_meters_per_pixel(lat, 18)
        mpp_z19 = get_meters_per_pixel(lat, 19)
        mpp_z20 = get_meters_per_pixel(lat, 20)

        self.assertAlmostEqual(mpp_z18 / mpp_z19, 2.0, places=6)
        self.assertAlmostEqual(mpp_z19 / mpp_z20, 2.0, places=6)


class WebMercatorTest(unittest.TestCase):
    def test_pixel_to_latlon(self):
        # Setup
        # At zoom level 2, the entire world is represented as a 1024x1024 pixel image
        # check that the pixel coordinate (512, 256) gets mapped to (0, -90) in lat, lon
        zoom_level = 2
        pixel_coords = (512, 256)
        expected_lat_deg = 0.0
        expected_lon_deg = -90.0

        # Action
        lat_deg, lon_deg = web_mercator.pixel_coords_to_latlon(*pixel_coords, zoom_level)

        # Verification
        self.assertAlmostEqual(lat_deg, expected_lat_deg)
        self.assertAlmostEqual(lon_deg, expected_lon_deg)

    def test_latlon_to_pixel(self):
        # Setup
        lat_deg = 0.0
        lon_deg = 0.0
        zoom_level = 0

        expected_row = 128
        expected_col = 128

        # Action
        row, col = web_mercator.latlon_to_pixel_coords(lat_deg, lon_deg, zoom_level)

        # Verification
        self.assertAlmostEqual(row, expected_row)
        self.assertAlmostEqual(col, expected_col)

    def test_latlon_round_trip(self):
        # Setup
        lat_deg = 45.123
        lon_deg = -123.456
        zoom_level = 20

        # Action
        pixel_coords = web_mercator.latlon_to_pixel_coords(lat_deg, lon_deg, zoom_level)
        out_lat_deg, out_lon_deg = web_mercator.pixel_coords_to_latlon(*pixel_coords, zoom_level)

        # Verification
        self.assertAlmostEqual(lon_deg, out_lon_deg)
        self.assertAlmostEqual(lat_deg, out_lat_deg)


if __name__ == "__main__":
    unittest.main()
