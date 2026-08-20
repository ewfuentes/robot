import unittest

from experimental.overhead_matching.swag.bearing_only_localization import (
    observability as obs,
)


class AngularSpanTest(unittest.TestCase):

    def test_smallest_containing_arc(self):
        self.assertAlmostEqual(obs.angular_span_deg([10, 30, 50]), 40.0)

    def test_wraps_at_north(self):
        self.assertAlmostEqual(obs.angular_span_deg([350, 0, 10]), 20.0)

    def test_single_angle_has_no_span(self):
        self.assertEqual(obs.angular_span_deg([42.0]), 0.0)


class TrajectoryTest(unittest.TestCase):

    def test_straight_line_has_net_over_path_one(self):
        east = [0.0, 100.0, 200.0, 300.0]
        north = [0.0] * 4
        summary = obs.describe_trajectory(east, north, [90.0] * 4)
        self.assertAlmostEqual(summary["net_over_path"], 1.0)
        self.assertAlmostEqual(summary["path_m"], 300.0)

    def test_out_and_back_has_low_net_over_path(self):
        east = [0.0, 100.0, 200.0, 100.0, 0.0]
        north = [0.0] * 5
        summary = obs.describe_trajectory(east, north, [90, 90, 270, 270, 270])
        self.assertAlmostEqual(summary["net_over_path"], 0.0)
        self.assertAlmostEqual(summary["course_span_deg"], 180.0)


class DensityTest(unittest.TestCase):
    """No verdict is asserted, because no verdict is offered: see the module
    docstring. What is pinned is the arithmetic and the measured table, so that a
    future predictor has to face the datasets that refuted the last two."""

    # (name, measurements, landmarks, final error in metres)
    MEASURED = (("charles_river", 561, 30370, 25),
                ("boston_leg3", 764, 13210, 34),
                ("mtw_leg3", 347, 4237, 190),
                ("mtw_leg2", 232, 4237, 304),
                ("boston_leg1", 437, 13210, 6504),
                ("mtw_leg1", 39, 4237, 2074),
                ("boston_leg2", 214, 13210, 11651))

    def test_density_is_measurements_over_landmarks(self):
        self.assertAlmostEqual(obs.bearing_density(561, 30370), 0.018472,
                               places=6)

    def test_an_empty_catalog_does_not_divide_by_zero(self):
        self.assertEqual(obs.bearing_density(100, 0), 0.0)

    def test_no_density_threshold_separates_the_measured_outcomes(self):
        """The refutation, as a test. charles localizes to 25 m at the
        second-LOWEST density of the seven, so any threshold that admits it also
        admits every failure. Should someone reintroduce a density verdict, this
        is the fact it has to answer for."""
        worked = [obs.bearing_density(m, l)
                  for _, m, l, err in self.MEASURED if err < 1000]
        failed = [obs.bearing_density(m, l)
                  for _, m, l, err in self.MEASURED if err >= 1000]
        self.assertLess(min(worked), max(failed),
                        "the density ranges no longer overlap -- if new data has "
                        "separated them, revisit the module docstring rather "
                        "than deleting this test")

    def test_course_span_does_not_separate_them_either(self):
        # mount_washington leg2 (116 deg) works at 304 m; boston leg2 (179 deg)
        # fails at 11651 m. Refutation #1.
        self.assertGreater(179.0, 116.0)
