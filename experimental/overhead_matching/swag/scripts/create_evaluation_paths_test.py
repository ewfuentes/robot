import unittest

import common.torch.load_torch_deps
import torch

from pathlib import Path

from experimental.overhead_matching.swag.data import vigor_dataset
from experimental.overhead_matching.swag.scripts.create_evaluation_paths import (
    compute_city_bounding_box,
    compute_bbox_coverage_ratio,
    generate_goal_directed_path,
)


class CreateEvaluationPathsTest(unittest.TestCase):

    def test_goal_directed_path_generation(self):
        """Test that goal-directed path generation produces valid paths."""
        config = vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )
        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)

        PATH_LENGTH_M = 500
        SEED = 123

        # action
        path = generate_goal_directed_path(
            dataset._panorama_metadata,
            torch.manual_seed(SEED),
            PATH_LENGTH_M,
            goal_reached_threshold_m=100.0,
            min_goal_distance_percentile=0.5,
            min_bbox_coverage_ratio=0.0,  # Disable bbox constraint for basic test
        )

        # Verification
        self.assertGreater(len(path), 0)
        # All items should be pano_id strings
        for item in path:
            self.assertTrue(isinstance(item, str), f"Expected str, got {type(item)}")

    def test_goal_directed_path_is_reproducible(self):
        """Test that goal-directed paths are reproducible with same seed."""
        config = vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )
        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)

        PATH_LENGTH_M = 300
        SEED = 456

        # action
        path1 = generate_goal_directed_path(
            dataset._panorama_metadata,
            torch.manual_seed(SEED),
            PATH_LENGTH_M,
            min_bbox_coverage_ratio=0.0,
        )
        path2 = generate_goal_directed_path(
            dataset._panorama_metadata,
            torch.manual_seed(SEED),
            PATH_LENGTH_M,
            min_bbox_coverage_ratio=0.0,
        )
        path3 = generate_goal_directed_path(
            dataset._panorama_metadata,
            torch.manual_seed(SEED - 10),
            PATH_LENGTH_M,
            min_bbox_coverage_ratio=0.0,
        )

        # Verification
        self.assertListEqual(path1, path2)
        self.assertNotEqual(path1, path3)

    def test_goal_directed_path_bbox_helpers(self):
        """Test bounding box helper methods."""
        config = vigor_dataset.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )
        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)

        # Test city bounding box
        city_bbox = compute_city_bounding_box(dataset._panorama_metadata)
        self.assertEqual(len(city_bbox), 4)
        min_lat, max_lat, min_lon, max_lon = city_bbox
        self.assertLess(min_lat, max_lat)
        self.assertLess(min_lon, max_lon)

        # Test bbox coverage ratio
        # Full city should have ratio 1.0
        coverage = compute_bbox_coverage_ratio(city_bbox, city_bbox)
        self.assertAlmostEqual(coverage, 1.0)

        # Half-size bbox should have ratio 0.25
        half_bbox = (
            min_lat,
            (min_lat + max_lat) / 2,
            min_lon,
            (min_lon + max_lon) / 2,
        )
        coverage = compute_bbox_coverage_ratio(half_bbox, city_bbox)
        self.assertAlmostEqual(coverage, 0.25, places=5)


if __name__ == "__main__":
    unittest.main()
