import unittest

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    truth_strips,
)


def _pano_with_marker(azimuth_cam_deg: float, width: int = 720,
                      height: int = 360) -> Image.Image:
    """Equirectangular photo with a bright column at a camera azimuth."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    col = int(round((azimuth_cam_deg / 360.0 + 0.5) * width)) % width
    array[:, col] = 255
    return Image.fromarray(array)


class PanoStripTest(unittest.TestCase):
    def test_marker_lands_at_its_map_azimuth(self):
        # Travel direction appears 30 deg CW of the pano centre; the platform
        # is heading due east, so the centre column looks at 90 - 30 = 60 deg
        # and a marker 40 deg right of centre is at map azimuth 100 deg.
        pano = _pano_with_marker(40.0)
        n_az = 360
        strip = truth_strips.pano_strip(
            pano, 60.0, elev_min_deg=-10.0, elev_max_deg=10.0,
            n_az=n_az, n_rows=8)
        column = int(np.argmax(strip.mean(axis=(0, 2))))
        self.assertAlmostEqual(column * 360.0 / n_az, 100.0, delta=1.5)

    def test_centre_column_maps_to_centre_yaw(self):
        pano = _pano_with_marker(0.0)
        n_az = 360
        for center_yaw in (0.0, 37.0, 215.0, 359.0):
            strip = truth_strips.pano_strip(
                pano, center_yaw, elev_min_deg=-5.0, elev_max_deg=5.0,
                n_az=n_az, n_rows=4)
            column = int(np.argmax(strip.mean(axis=(0, 2))))
            self.assertAlmostEqual(column * 360.0 / n_az, center_yaw,
                                   delta=1.5, msg=f"yaw {center_yaw}")

    def test_elevation_band_is_the_requested_slice(self):
        # Bright row at exactly elevation 0 (the equirect mid-row).
        array = np.zeros((360, 720, 3), dtype=np.uint8)
        array[180] = 255
        strip = truth_strips.pano_strip(
            Image.fromarray(array), 0.0, elev_min_deg=-20.0,
            elev_max_deg=20.0, n_az=72, n_rows=40)
        row = int(np.argmax(strip.mean(axis=(1, 2))))
        # Row 0 is +20 deg, row 39 is -20 deg, so elevation 0 is mid-strip.
        self.assertAlmostEqual(20.0 - row * 40.0 / 39.0, 0.0, delta=1.5)


class HorizonTest(unittest.TestCase):
    def test_first_terrain_row_per_column(self):
        depth = np.full((5, 4), np.inf, dtype=np.float32)
        depth[2:, 0] = 100.0  # terrain from row 2 down
        depth[0:, 1] = 50.0   # terrain everywhere
        depth[4, 3] = 10.0    # terrain only in the last row
        rows = truth_strips.horizon_rows(depth)
        np.testing.assert_array_equal(rows, [2, 0, 5, 4])

    def test_draw_horizon_marks_only_valid_columns(self):
        image = Image.new("RGB", (4, 5), (0, 0, 0))
        out = np.asarray(truth_strips.draw_horizon(
            image, np.array([2, 0, 5, 4])))
        marked = {(int(r), int(c))
                  for r, c in zip(*np.nonzero(out.sum(axis=2)))}
        self.assertEqual(marked, {(2, 0), (0, 1), (4, 3)})


class ShiftTest(unittest.TestCase):
    def _profile(self, n: int, azimuth_deg: float,
                 width_deg: float = 8.0) -> np.ndarray:
        columns = np.arange(n) * 360.0 / n
        delta = (columns - azimuth_deg + 180.0) % 360.0 - 180.0
        return np.exp(-0.5 * (delta / width_deg) ** 2)

    def test_zero_shift_for_identical_profiles(self):
        profile = self._profile(360, 100.0)
        estimate = truth_strips.estimate_shift(profile, profile)
        self.assertAlmostEqual(estimate.delta_deg, 0.0, places=6)
        self.assertAlmostEqual(estimate.peak, 1.0, places=6)

    def test_photo_clockwise_of_render_gives_negative_correction(self):
        # The photo puts the feature at 110 deg, the surface says 100 deg: the
        # centre yaw used was 10 deg too large, so the correction is -10.
        estimate = truth_strips.estimate_shift(
            self._profile(360, 110.0), self._profile(360, 100.0))
        self.assertAlmostEqual(estimate.delta_deg, -10.0, delta=1.0)
        self.assertGreater(estimate.peak, 0.9)

    def test_photo_counterclockwise_of_render_gives_positive_correction(self):
        estimate = truth_strips.estimate_shift(
            self._profile(720, 20.0), self._profile(720, 35.0))
        self.assertAlmostEqual(estimate.delta_deg, 15.0, delta=1.0)

    def test_wraparound_shift(self):
        estimate = truth_strips.estimate_shift(
            self._profile(360, 5.0), self._profile(360, 350.0))
        self.assertAlmostEqual(estimate.delta_deg, -15.0, delta=1.0)

    def test_flat_profile_abstains(self):
        flat = np.ones(64)
        estimate = truth_strips.estimate_shift(flat, self._profile(64, 30.0))
        self.assertEqual((estimate.delta_deg, estimate.peak), (0.0, 0.0))
        self.assertEqual(estimate.fwhm_deg, 360.0)

    def test_sharp_feature_is_narrow_and_prominent(self):
        estimate = truth_strips.estimate_shift(
            self._profile(720, 100.0, width_deg=3.0),
            self._profile(720, 100.0, width_deg=3.0))
        self.assertLess(estimate.fwhm_deg, 15.0)
        self.assertGreater(estimate.prominence, 0.5)
        self.assertLess(estimate.sigma_deg, 7.0)

    def test_broad_feature_correlates_well_but_is_wide(self):
        # The leg1 failure mode: one smooth hump matches itself at r = 1 while
        # tens of degrees of shift cost almost nothing.
        broad = self._profile(720, 100.0, width_deg=70.0)
        estimate = truth_strips.estimate_shift(broad, broad)
        self.assertAlmostEqual(estimate.peak, 1.0, places=6)
        self.assertGreater(estimate.fwhm_deg, 60.0)

    def test_repeated_structure_is_high_r_but_unprominent(self):
        # Two identical ridges 40 deg apart: the shift is genuinely ambiguous.
        twin = (self._profile(720, 80.0, width_deg=4.0)
                + self._profile(720, 120.0, width_deg=4.0))
        estimate = truth_strips.estimate_shift(twin, twin)
        self.assertAlmostEqual(estimate.peak, 1.0, places=6)
        self.assertLess(estimate.prominence, 0.6)
        self.assertLess(estimate.fwhm_deg, 20.0)  # each ridge is still sharp


    def test_highpass_lets_the_ridge_beat_the_slow_lobe(self):
        # A real sky profile: one 150 deg lobe (open side vs land side) with a
        # small sharp ridge on it. Undetrended, the lobe sets the peak width;
        # detrended, the ridge does. This is why highpass_deg defaults to 45.
        profile = (self._profile(720, 200.0, width_deg=150.0)
                   + 0.15 * self._profile(720, 60.0, width_deg=3.0))
        raw = truth_strips.estimate_shift(profile, profile, highpass_deg=0.0)
        detrended = truth_strips.estimate_shift(profile, profile)
        self.assertGreater(raw.fwhm_deg, 100.0)
        self.assertLess(detrended.fwhm_deg, raw.fwhm_deg - 10.0)
        self.assertAlmostEqual(detrended.delta_deg, 0.0, places=6)
        # Measured on the Mt. Washington legs, where it matters: the median
        # peak width falls from ~110 deg (identical on strong and weak frames,
        # so useless as a gate) to 57 deg on leg2 and 82 deg on leg1, which
        # separates them.

    def test_highpass_of_a_constant_profile_is_zero(self):
        flat = np.full(360, 0.7)
        np.testing.assert_allclose(
            truth_strips.highpass_circular(flat, 45.0), 0.0, atol=1e-9)


class WeightedMeanTest(unittest.TestCase):
    def test_sharp_frames_dominate(self):
        mean = truth_strips.weighted_circular_mean_deg(
            [30.0, 0.0], [1.0, 30.0])
        self.assertAlmostEqual(mean, 30.0, delta=0.2)

    def test_wraparound(self):
        mean = truth_strips.weighted_circular_mean_deg(
            [-179.0, 179.0], [2.0, 2.0])
        self.assertAlmostEqual(abs(mean), 180.0, delta=0.5)

    def test_no_finite_sigma_is_none(self):
        self.assertIsNone(truth_strips.weighted_circular_mean_deg(
            [10.0], [float("inf")]))
        self.assertIsNone(truth_strips.weighted_circular_mean_deg([], []))

    def test_sky_fractions_track_a_synthetic_skyline(self):
        # Photo: bright sky above a dark ridge occupying the right half.
        band = np.zeros((20, 8, 3), dtype=np.uint8)
        band[:, :] = (150, 160, 200)
        band[10:, 4:] = (30, 30, 30)
        photo = truth_strips.photo_sky_fraction(band)
        np.testing.assert_allclose(photo[:4], 1.0)
        np.testing.assert_allclose(photo[4:], 0.5)

        depth = np.full((20, 8), np.inf, dtype=np.float32)
        depth[10:, 4:] = 500.0
        np.testing.assert_allclose(
            truth_strips.render_sky_fraction(depth), photo)


class CircularMedianTest(unittest.TestCase):
    def test_median_across_wraparound(self):
        # Due south, whichever of the two equivalent representatives it takes.
        median = truth_strips.circular_median_deg([-179.0, 179.0, 180.0])
        self.assertAlmostEqual(abs(median), 180.0, delta=1.0)

    def test_plain_median(self):
        self.assertAlmostEqual(
            truth_strips.circular_median_deg([10.0, 12.0, 14.0, 100.0]),
            13.0, delta=1.5)

    def test_empty_is_none(self):
        self.assertIsNone(truth_strips.circular_median_deg([]))


class GridCourseTest(unittest.TestCase):
    def _frames(self, lats, lons, times) -> list[dict]:
        return [{"latitude": f"{lat}", "longitude": f"{lon}",
                 "sensor_elapsed_s": f"{t}",
                 "frame_file": f"f{i:04d},{lat},{lon},.jpg"}
                for i, (lat, lon, t) in enumerate(zip(lats, lons, times))]
    def test_due_north_track_has_course_near_zero(self):
        lats = [42.0 + 0.001 * i for i in range(10)]
        lons = [-71.0] * 10
        course = truth_strips.GridCourse(
            self._frames(lats, lons, list(range(0, 100, 10))), "EPSG:32619")
        # Grid north, not true north: UTM 19N convergence at -71 deg is small
        # but real, so the course is a couple of degrees off zero.
        for i in range(10):
            self.assertAlmostEqual(
                (course.course_deg(i) + 180.0) % 360.0 - 180.0, 0.0, delta=3.0)

    def test_stationary_track_abstains(self):
        frames = self._frames([42.0] * 5, [-71.0] * 5, list(range(5)))
        self.assertIsNone(
            truth_strips.GridCourse(frames, "EPSG:32619").course_deg(0))

    def test_frame_id(self):
        self.assertEqual(
            truth_strips.frame_id({"frame_file": "f0042,44.2,-71.3,.jpg"}),
            "f0042")


if __name__ == "__main__":
    unittest.main()
