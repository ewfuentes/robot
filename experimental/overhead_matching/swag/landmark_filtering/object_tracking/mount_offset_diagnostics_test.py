import math
import unittest

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    mount_offset_diagnostics as diag,
)


def observation(east, north, camera, course, keyframe):
    return (east, north, camera, course, keyframe)


class SpanTest(unittest.TestCase):

    def test_smallest_containing_arc(self):
        self.assertAlmostEqual(diag.span_deg([10.0, 30.0, 50.0]), 40.0)

    def test_wraps_at_north(self):
        self.assertAlmostEqual(diag.span_deg([350.0, 10.0]), 20.0)

    def test_fewer_than_two_angles_has_no_span(self):
        self.assertEqual(diag.span_deg([7.0]), 0.0)


class BaselineTest(unittest.TestCase):

    def test_longest_pairwise_distance(self):
        obs = [observation(0.0, 0.0, 0.0, 0.0, 0),
               observation(30.0, 40.0, 0.0, 0.0, 1),
               observation(3.0, 4.0, 0.0, 0.0, 2)]
        self.assertAlmostEqual(diag.baseline_m(obs), 50.0)

    def test_single_observation_has_no_baseline(self):
        self.assertEqual(diag.baseline_m([observation(1.0, 2.0, 0, 0, 0)]), 0.0)


class DescribeTest(unittest.TestCase):
    """The point of the module: a distant object seen over a short baseline is
    nearly blind to the offset, and a near one over a long baseline is not."""

    def rays_to(self, east_m, north_m, positions):
        """Observations of a static point, with course 0 so camera == world."""
        obs = []
        for i, (east, north) in enumerate(positions):
            bearing = math.degrees(math.atan2(east_m - east,
                                              north_m - north)) % 360.0
            obs.append(observation(east, north, bearing, 0.0, i))
        return obs

    def test_wide_arc_tracklet_is_sensitive_to_the_offset(self):
        # Object 300 m away, passed over 1200 m of track: a wide sweep.
        obs = self.rays_to(0.0, 300.0, [(-600.0, 0.0), (-200.0, 0.0),
                                        (200.0, 0.0), (600.0, 0.0)])
        row = diag.describe("LT_near", obs, 0.0)
        self.assertIsNotNone(row)
        self.assertGreater(row["arc_deg"], 90.0)
        self.assertGreater(row["sensitivity_deg_per_deg"],
                           diag.BLIND_SENSITIVITY)

    def test_distant_object_over_a_short_baseline_is_blind(self):
        # Object 8 km away, seen over 120 m: the rays are nearly parallel, so
        # rotating them keeps them nearly parallel and the residual barely moves.
        obs = self.rays_to(0.0, 8000.0, [(-60.0, 0.0), (-20.0, 0.0),
                                         (20.0, 0.0), (60.0, 0.0)])
        row = diag.describe("LT_far", obs, 0.0)
        self.assertIsNotNone(row)
        self.assertLess(row["arc_deg"], 2.0)
        self.assertLess(row["sensitivity_deg_per_deg"],
                        diag.BLIND_SENSITIVITY)

    def test_residual_is_near_zero_at_the_true_offset(self):
        obs = self.rays_to(500.0, 400.0, [(0.0, 0.0), (200.0, 0.0),
                                          (400.0, 0.0), (600.0, 0.0)])
        row = diag.describe("LT_exact", obs, 0.0)
        self.assertLess(row["residual_deg"], 1e-6)


class SummariseTest(unittest.TestCase):

    def rows(self, sensitivities, condition=10.0):
        return [{"tracklet_id": f"LT{i}", "n_observations": 5,
                 "baseline_m": 700.0, "median_range_m": 800.0,
                 "arc_deg": 45.0, "residual_deg": 1.0, "condition": condition,
                 "sensitivity_deg_per_deg": s}
                for i, s in enumerate(sensitivities)]

    def test_counts_blind_tracklets(self):
        summary = diag.summarise(self.rows([0.01, 0.02, 0.5, 0.6]), 500.0)
        self.assertEqual(summary["n_blind"], 2)
        self.assertAlmostEqual(summary["frac_blind"], 0.5)

    def test_ill_conditioned_tracklets_are_excluded_from_the_summary(self):
        summary = diag.summarise(self.rows([0.5, 0.5], condition=9000.0), 500.0)
        self.assertEqual(summary["n_tracklets"], 2)
        self.assertEqual(summary["n_well_conditioned"], 0)


if __name__ == "__main__":
    unittest.main()
