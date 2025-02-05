import unittest

import os
import torch
import common.testing.is_test_python as itp
from torchvision.transforms.functional import rotate, InterpolationMode
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np

from experimental.overhead_matching.from_sat_to_ground import (
    _calculate_bev_from_projected_points,
    _project_pointcloud_into_bev,
    _calculate_reward_penalty_maps_from_semantic_satellite,
    _add_orientation_filters,
    _convolve_point_cloud_bev_with_reward_maps,
    from_sat_to_ground
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
        self.assertTrue(np.allclose(
            projected[:, 0], bev_image_shape[0] / 2 - semantic_points[:, 1] / meters_per_pixel[0]))
        self.assertTrue(np.allclose(
            projected[:, 1], bev_image_shape[1] / 2 + semantic_points[:, 0] / meters_per_pixel[1]))

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

    def test_add_orientation_filters(self):
        semantic_bev = torch.zeros((1, 4, 4), dtype=torch.long)
        semantic_bev[0, :, 2] = 1
        semantic_bev[0, 0, 2] = 5
        rot_channels = _add_orientation_filters(semantic_bev, 4)
        self.assertEqual(rot_channels.shape, (4, 1, 4, 4))
        self.assertEqual(rot_channels.dtype, torch.long)
        self.assertEqual(rot_channels[0, 0, 0, 2], 5)
        self.assertEqual(rot_channels[1, 0, 1, 0], 5)
        self.assertEqual(rot_channels[2, 0, 3, 1], 5)
        self.assertEqual(rot_channels[3, 0, 2, 3], 5)

    def test_convolve_point_cloud_bev_with_reward_maps(self):
        semantic_satellite_image = np.zeros((50, 50), dtype=int)
        # place a road in the image
        semantic_satellite_image[10:20, :] = 2
        # rotate the road
        semantic_satellite_image = rotate(torch.from_numpy(semantic_satellite_image).unsqueeze(
            0), angle=45, interpolation=InterpolationMode.NEAREST, expand=False, fill=0).squeeze(0).numpy()

        reward_map, penalty_map = _calculate_reward_penalty_maps_from_semantic_satellite(
            semantic_satellite_image, sigma=1.0, num_classes=3)

        bev_point_cloud = np.zeros((3, 20, 20), dtype=int)
        # add a road and buildings to pointcloud observation
        bev_point_cloud[0, :, :5] = 1
        bev_point_cloud[0, :, -5:] = 1
        bev_point_cloud[2, :, 5:15] = 1

        gamma = 0.5
        num_orientation_bins = 16
        convolved = _convolve_point_cloud_bev_with_reward_maps(
            reward_map,
            penalty_map,
            bev_point_cloud,
            gamma,
            num_orientation_bins
        )

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax[0, 0].imshow(semantic_satellite_image)
        ax[0, 0].set_title("Semantic Satellite Image")
        ax[0, 1].imshow((bev_point_cloud * np.array([1, 2, 3]).reshape(3, 1, 1)).sum(axis=0))
        ax[0, 1].set_title("BEV Point Cloud")
        ax[0, 2].imshow(reward_map[0], vmin=penalty_map.min(), vmax=reward_map.max())
        ax[0, 2].set_title("Reward Map")
        ax[0, 3].imshow(penalty_map[0], vmin=penalty_map.min(), vmax=reward_map.max())
        ax[0, 3].set_title("Penalty Map")
        ax[1, 0].imshow(convolved[0], vmin=convolved.min(), vmax=convolved.max())
        ax[1, 0].set_title("Convolved 0 deg")
        ax[1, 1].imshow(convolved[2], vmin=convolved.min(), vmax=convolved.max())
        ax[1, 1].set_title("Convolved 45 deg")
        ax[1, 2].imshow(convolved[4], vmin=convolved.min(), vmax=convolved.max())
        ax[1, 2].set_title("Convolved 90 deg")
        ax[1, 3].imshow(convolved[6], vmin=convolved.min(), vmax=convolved.max())
        ax[1, 3].set_title("Convolved 135 deg")

        plt.savefig("/tmp/test.png")

        # for i in range(convolved.shape[0]):
        #     fig, ax = plt.subplots(1, 1)
        #     ax.imshow(convolved[i], vmin=convolved.min(), vmax=convolved.max())
        #     plt.savefig(f"/tmp/test_{i:02d}.png")
        # plt.show()

        self.assertEqual(convolved.shape, (num_orientation_bins, 50, 50))
        self.assertTrue(np.all(np.isfinite(convolved)))
        idx = np.unravel_index(np.argmax(convolved), convolved.shape)
        self.assertTrue(idx[0] in [6, 14])  # max should be in the correct angle bins
        self.assertTrue(idx[1]**2 + idx[2]**2 < 25**2) # max should be top left of image

    def test_from_sat_to_ground(self):
        fake_pointcloud = np.random.rand(100, 4)
        fake_pointcloud[:, 3] = np.random.randint(0, 3, 100)
        fake_satellite = np.random.randint(0, 3, (100, 100))
        output = from_sat_to_ground(
            fake_pointcloud,
            fake_satellite, 
            sigma=1.0,
            num_orientation_bins=64,
        )

        self.assertEqual(output.shape, (64, 100, 100))


if __name__ == "__main__":
    unittest.main()
