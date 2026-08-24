import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.calibration import (
    sun_offset_check as soc,
)


class SolarPositionTest(unittest.TestCase):
    """Anchored on geometry that does not depend on a table lookup."""

    def test_greenwich_solar_noon_at_the_solstice(self):
        # Due south, and 90 - lat + obliquity above the horizon.
        az, el = soc.solar_position(
            datetime(2026, 6, 21, 12, 2, tzinfo=timezone.utc), 51.4778, -0.0015)
        self.assertAlmostEqual(az, 180.0, delta=0.5)
        self.assertAlmostEqual(el, 90.0 - 51.4778 + 23.44, delta=0.5)

    def test_sun_is_overhead_at_the_equator_at_the_equinox(self):
        _, el = soc.solar_position(
            datetime(2026, 3, 20, 12, 7, tzinfo=timezone.utc), 0.0, 0.0)
        self.assertGreater(el, 87.0)

    def test_azimuth_sweeps_east_to_west_through_the_day(self):
        base = datetime(2026, 6, 21, 8, 0, tzinfo=timezone.utc)
        azimuths = [soc.solar_position(base + timedelta(hours=h), 42.36,
                                       -71.06)[0] for h in range(4, 12, 2)]
        # Boston mid-morning to evening: monotonically increasing through
        # south.
        self.assertTrue(all(b > a for a, b in zip(azimuths, azimuths[1:])),
                        f"not monotonic: {azimuths}")

    def test_southern_hemisphere_sun_is_to_the_north(self):
        az, _ = soc.solar_position(
            datetime(2026, 12, 21, 1, 50, tzinfo=timezone.utc), -33.87, 151.21)
        self.assertLess(min(az, 360.0 - az), 10.0)

    def test_elevation_is_negative_at_local_midnight(self):
        _, el = soc.solar_position(
            datetime(2026, 6, 21, 4, 0, tzinfo=timezone.utc), 42.36, -71.06)
        self.assertLess(el, 0.0)


def pano_with_blob(width, height, az_deg, el_deg, value=250, radius=6,
                   background=40):
    pano = np.full((height, width, 3), background, dtype=np.uint8)
    x, y = geo.pano_px_from_direction(az_deg, el_deg, width, height)
    xs = (np.arange(-radius, radius + 1) + int(x)) % width
    ys = np.clip(np.arange(-radius, radius + 1) + int(y), 0, height - 1)
    pano[np.ix_(ys, xs)] = value
    return pano


class BlobSearchTest(unittest.TestCase):

    def test_finds_a_blob_at_the_expected_elevation(self):
        pano = pano_with_blob(720, 360, az_deg=296.0, el_deg=28.0)
        found = soc.brightest_blob_in_band(pano, 28.0)
        self.assertIsNotNone(found)
        az, el, _ = found
        self.assertAlmostEqual(az, 296.0, delta=2.0)
        self.assertAlmostEqual(el, 28.0, delta=2.0)

    def test_ignores_a_brighter_blob_outside_the_band(self):
        # This is the sun-glitter case: the water below the horizon is at
        # least as bright as the sun, and a naive argmax lands on it.
        pano = pano_with_blob(720, 360, az_deg=296.0, el_deg=28.0, value=230)
        glitter = pano_with_blob(720, 360, az_deg=100.0, el_deg=-20.0,
                                 value=255, radius=20)
        pano = np.maximum(pano, glitter)
        az, _, _ = soc.brightest_blob_in_band(pano, 28.0)
        self.assertAlmostEqual(az, 296.0, delta=2.0)

    def test_blob_straddling_the_seam_does_not_average_to_the_far_side(self):
        pano = pano_with_blob(720, 360, az_deg=180.5, el_deg=30.0)
        az, _, _ = soc.brightest_blob_in_band(pano, 30.0)
        # Azimuth 180 is the seam (column 0 / column W). Anything near 0 deg
        # would mean the circular mean collapsed to the panorama's centre.
        self.assertLess(min(abs(az - 180.5), 360.0 - abs(az - 180.5)), 3.0)

    def test_uniform_sky_is_rejected_outright(self):
        # Overcast: the whole band thresholds bright, so the run spans the
        # panorama. That is not a sun and must not yield an azimuth.
        pano = np.full((360, 720, 3), 200, dtype=np.uint8)
        self.assertIsNone(soc.brightest_blob_in_band(pano, 30.0))

    def test_saturated_flat_topped_blob_is_not_read_off_its_left_edge(self):
        # The regression this module was first wrong about: a saturated sun
        # has a plateau, argmax returns the plateau's left edge, and a window
        # centred there reads about half a blob radius low.
        for radius in (3, 6, 12):
            pano = pano_with_blob(720, 360, az_deg=296.0, el_deg=28.0,
                                  value=255, radius=radius)
            az, _, _ = soc.brightest_blob_in_band(pano, 28.0)
            self.assertAlmostEqual(
                az, 296.0, delta=1.0,
                msg=f"radius {radius}px blob read at {az:.2f} deg")


class CircularStatsTest(unittest.TestCase):

    def test_agreeing_angles_concentrate(self):
        mean, r = soc.circular_stats([276.0, 277.0, 278.0, 276.5])
        self.assertAlmostEqual(mean, 276.875, delta=0.5)
        self.assertGreater(r, 0.99)

    def test_wrap_around_north_is_handled(self):
        mean, r = soc.circular_stats([359.0, 1.0, 0.0])
        self.assertLess(min(mean, 360.0 - mean), 1.0)
        self.assertGreater(r, 0.99)

    def test_scatter_collapses_the_concentration(self):
        _, r = soc.circular_stats([0.0, 90.0, 180.0, 270.0])
        self.assertLess(r, 0.01)

    def test_mean_matches_the_geometry_owner(self):
        angles = [12.0, 44.5, 359.0, 271.25]
        mean, _ = soc.circular_stats(angles)
        self.assertAlmostEqual(mean, geo.circular_mean_deg(angles), places=9)


class OffsetAlgebraTest(unittest.TestCase):
    """The one subtraction the whole check rests on."""

    def test_offset_recovers_a_planted_value(self):
        # Plant a mount offset, synthesise the camera azimuth it would
        # produce, and check the formula inverts it.
        course, sun_world, offset = 248.5, 271.0, 277.0
        az_camera = (sun_world - course + offset) % 360.0
        recovered = (course + az_camera - sun_world) % 360.0
        self.assertAlmostEqual(recovered, offset, places=6)


if __name__ == "__main__":
    unittest.main()
