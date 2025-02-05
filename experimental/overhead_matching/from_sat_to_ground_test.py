import unittest

import os
import common.testing.is_test_python as itp


from PIL import Image
from pathlib import Path
import numpy as np

from experimental.overhead_matching.from_sat_to_ground import (
    _calculate_bev_from_projected_points,
    _project_pointcloud_into_bev,
    _calculate_reward_penalty_maps_from_semantic_satellite
)


class TestFromSatToGround(unittest.TestCase):
    def test_calculate_reward_penalty_maps_from_semantic_satellite(self):
        # Create a small semantic satellite array with 3 classes
        semantic_sat = np.array([
            [0, 1, 2],
            [1, 1, 0],
            [2, 0, 1]
        ], dtype=np.uint8)
        sigma = 1.0
        num_classes = 3

        reward, penalty = _calculate_reward_penalty_maps_from_semantic_satellite(
            semantic_sat,
            sigma,
            num_classes
        )

        self.assertEqual(reward.shape, (num_classes, 3, 3))
        self.assertEqual(penalty.shape, (num_classes, 3, 3))
        self.assertTrue(np.all(np.isfinite(reward)))
        self.assertTrue(np.all(np.isfinite(penalty)))
        self.assertEqual(reward[0, 0, 0], 1.0)
        self.assertEqual(reward[2, 2, 0], 1.0)
        self.assertEqual(reward[0, 1, 1], np.exp(- 1 / 2 / sigma))
        self.assertEqual(reward[2, 1, 1], np.exp(- np.sqrt(2) / sigma / 2))

        self.assertEqual(penalty[1, 0, 0], -1)
        self.assertEqual(penalty[0, 0, 0], -1 * np.exp(-1 / 2 / sigma))

    def test_project_pointcloud_into_bev(self):
        # semantic_points: N x 4, format: x, y, z, class
        semantic_points = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [-2.0, 3.5, 0.0, 1.0],
            [-2.0, 3.5, 0.0, 2.0]
        ])
        bev_image_shape = (10, 10)
        meters_per_pixel = (1.5, 1.2)

        

        projected = _project_pointcloud_into_bev(
            semantic_points, bev_image_shape, meters_per_pixel
        )

        self.assertEqual(projected.shape, (5, 3))
        self.assertEqual(projected.shape[0], semantic_points.shape[0])
        self.assertTrue(np.all(np.isfinite(projected)))
        self.assertTrue(np.allclose(projected[:, 0], bev_image_shape[0] / 2 - semantic_points[:, 1] / meters_per_pixel[0]))
        self.assertTrue(np.allclose(projected[:, 1], bev_image_shape[1] / 2 + semantic_points[:, 0] / meters_per_pixel[1]))

    def test_calculate_bev_from_projected_points(self):
        # projected_points: N x 3, format: i, j, class
        projected_points = np.array([
            [1.2, 2.3, 0],
            [1.4, 2.1, 0],
            [1.5, 2.6, 0],
            [1.9, 2.8, 0],
            [7.9, 3.0, 1],
            [2.0, 8.0, 2]
        ])
        num_classes = 3
        bev = _calculate_bev_from_projected_points(projected_points, num_classes)


        # min i is 1.2
        # min j is 2.1

        self.assertEqual(bev.ndim, 3)
        self.assertEqual(bev.shape[0], num_classes)
        self.assertGreaterEqual(bev.shape[1], 8)  # from data
        self.assertGreaterEqual(bev.shape[2], 2)  # from data
        self.assertTrue(np.all(bev >= 0))
        self.assertEqual(bev[0, 0, 0], 3)
        self.assertEqual(bev[0, 1, 1], 1)
        self.assertEqual(bev[1, 0, 3], 0)
        self.assertEqual(bev[1, 7, 1], 1)
        self.assertEqual(bev[2, 1, 6], 1)
        self.assertEqual(bev[2, 0, 3], 0)


if __name__ == "__main__":
    unittest.main()
