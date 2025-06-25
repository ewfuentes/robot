
import unittest

from common.gps import web_mercator


class WebMercatorTest(unittest.TestCase):
    def test_pixel_to_latlon(self):
        # Setup
        # At zoom level 1, the entire world is represented as a 512x512 pixel image
        # check that the pixel coordinate (256, 128) gets mapped to (0, -90) in lat, lon
        zoom_level = 1
        pixel_coords = (256, 128)
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


if __name__ == "__main__":
    unittest.main()
