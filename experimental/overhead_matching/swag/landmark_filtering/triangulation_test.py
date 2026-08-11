import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    triangulation,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    TriangulationConfig,
)


def bearings_to_point(positions: np.ndarray, point,
                      noise_deg=0.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta = np.asarray(point)[None, :] - positions
    bearings = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0
    return bearings + rng.normal(0.0, noise_deg, len(positions))


def walk_east(n=20, spacing=5.0) -> np.ndarray:
    return np.stack([np.arange(n) * spacing, np.zeros(n)], axis=1)


class TriangulationTest(unittest.TestCase):
    def test_near_landmark_recovered(self):
        positions = walk_east()
        target = (50.0, 40.0)
        bearings = bearings_to_point(positions, target, noise_deg=1.0)
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig())
        self.assertTrue(result.solved)
        self.assertEqual(result.observability, "near")
        self.assertLess(math.hypot(result.x_m - target[0],
                                   result.y_m - target[1]), 5.0)
        self.assertLess(result.residual_rms_deg, 2.0)

    def test_outlier_robustness(self):
        positions = walk_east()
        target = (50.0, 40.0)
        bearings = bearings_to_point(positions, target, noise_deg=0.5)
        bearings[7] += 30.0  # gross outlier
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig())
        self.assertTrue(result.solved)
        self.assertLess(math.hypot(result.x_m - target[0],
                                   result.y_m - target[1]), 5.0)
        self.assertGreaterEqual(result.n_outliers, 1)

    def test_collinear_dead_ahead_degenerate(self):
        # Landmark straight along the walking direction: zero parallax but
        # jitter above the consistency threshold -> degenerate.
        positions = walk_east()
        bearings = bearings_to_point(
            positions, (10000.0, 0.0), noise_deg=8.0)
        config = TriangulationConfig(min_parallax_deg=2.0)
        result = triangulation.triangulate_rays(positions, bearings, config)
        self.assertEqual(result.observability, "degenerate")

    def test_distant_landmark_is_far(self):
        positions = walk_east()
        target = (500.0, 2000.0)  # ~2 km away, broadside
        bearings = bearings_to_point(positions, target, noise_deg=0.5)
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig())
        self.assertEqual(result.observability, "far")

    def test_constant_bearing_low_noise_is_far(self):
        positions = walk_east(n=10)
        bearings = np.full(10, 30.0) + np.linspace(-0.5, 0.5, 10)
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig())
        self.assertEqual(result.observability, "far")
        self.assertFalse(result.solved)

    def test_flipped_bearings_never_classify_near(self):
        # Flipping all bearings by 180 (the yaw-sweep mirror case): the rays
        # diverge, so the best point fit either has huge angular residuals or
        # sits at negative range. Either way it must not come out "near".
        positions = walk_east()
        target = (50.0, 40.0)
        bearings = (bearings_to_point(positions, target, noise_deg=0.5)
                    + 180.0) % 360.0
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig())
        self.assertEqual(result.observability, "degenerate")
        self.assertIn(result.degenerate_reason,
                      ("negative_range", "high_residual"))

    def test_covariance_major_axis_along_range(self):
        positions = walk_east(n=10)
        target = (25.0, 300.0)  # mostly north of a short east-west baseline
        bearings = bearings_to_point(positions, target, noise_deg=1.0)
        result = triangulation.triangulate_rays(
            positions, bearings, TriangulationConfig(far_range_m=1000.0))
        self.assertTrue(result.solved)
        cov = np.array(result.cov_enu)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, np.argmax(eigvals)]
        # Range direction is ~north; the major axis should align with it.
        self.assertGreater(abs(major[1]), abs(major[0]))
        self.assertGreater(result.sigma_major_m, result.sigma_minor_m)


if __name__ == "__main__":
    unittest.main()
