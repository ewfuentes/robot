
import unittest

import common.torch.load_torch_deps
import torch
import pandas as pd
import geopandas as gpd
import shapely
from common.gps import web_mercator

import experimental.overhead_matching.swag.evaluation.observation_likelihood as lol


def create_square_patch(center_x, center_y, size=10):
    """Create a square polygon centered at (center_x, center_y)."""
    half_size = size / 2
    return shapely.Polygon([
        (center_x - half_size, center_y - half_size),
        (center_x + half_size, center_y - half_size),
        (center_x + half_size, center_y + half_size),
        (center_x - half_size, center_y + half_size),
        (center_x - half_size, center_y - half_size)
    ])


class ObservationLikelihoodTest(unittest.TestCase):
    def test_compute_log_observation_likelihood_valid_inputs(self):
        """Test that valid inputs don't raise exceptions."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0, 1]},
        ])

        pano_sat_similarity = torch.tensor([[0.8, 0.3]])
        pano_osm_similarity = torch.tensor([[0.9, 0.2], [0.1, 0.7]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        query = lol.QueryData(
            pano_ids=['pano_0'],
            particle_locs_deg=torch.tensor([[
                [37.7749, -122.4194],
                [40.7128, -74.0060],
                [51.5074, -0.1278],
            ]])
        )

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            )

        sat_log_likelihood, osm_log_likelihood = lol.compute_log_observation_likelihood(
            prior_data, query, config
        )

        self.assertEqual(sat_log_likelihood.shape, (1, 3))
        self.assertEqual(osm_log_likelihood.shape, (1, 3))

    def test_mismatched_pano_sat_similarity_shape(self):
        """Test that pano_sat_similarity with wrong shape raises error."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
            {'geometry_px': create_square_patch(300, 250, size=20), 'embedding_idx': 2},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        # Should be (1, 3) but is (1, 2)
        pano_sat_similarity = torch.tensor([[0.8, 0.3]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        query = lol.QueryData(
            pano_ids=['pano_0'],
            particle_locs_deg=torch.tensor([[37.7749, -122.4194]])
        )
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            )

        with self.assertRaises(Exception):
            lol.compute_log_observation_likelihood(prior_data, query, config)

    def test_missing_required_column(self):
        """Test that missing osm_embedding_idx column raises error."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50)},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        query = lol.QueryData(
            pano_ids=['pano_0'],
            particle_locs_deg=torch.tensor([[37.7749, -122.4194]])
        )
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            )

        with self.assertRaises((KeyError, ValueError)):
            lol.compute_log_observation_likelihood(prior_data, query, config)

    def test_nonexistent_pano_id(self):
        """Test that requesting non-existent pano_id raises error."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8], [0.5]])
        pano_osm_similarity = torch.tensor([[0.9], [0.7]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        query = lol.QueryData(
            pano_ids=['pano_99'],
            particle_locs_deg=torch.tensor([[37.7749, -122.4194]])
        )
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            )

        with self.assertRaises((AssertionError,)):
            lol.compute_log_observation_likelihood(prior_data, query, config)

    def test_batch_particles_output_shape(self):
        """Test that output shape matches particle shape for batch."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        query = lol.QueryData(
            pano_ids=['pano_0'],
            particle_locs_deg=torch.tensor([[
                [37.7749, -122.4194],
                [37.7750, -122.4195],
                [37.7751, -122.4196],
                [37.7752, -122.4197],
                [37.7753, -122.4198],
            ]])
        )
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            )

        sat_ll, osm_ll = lol.compute_log_observation_likelihood(prior_data, query, config)

        self.assertEqual(sat_ll.shape, (1, 5))
        self.assertEqual(osm_ll.shape, (1, 5))

    def test_get_similarities_single_pano(self):
        """Test extracting similarities for a single panorama."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
            {'geometry_px': create_square_patch(300, 250, size=20), 'embedding_idx': 2},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0, 1]},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [2, 3]},
        ])

        pano_sat_similarity = torch.tensor([
            [0.8, 0.3, 0.5],
            [0.2, 0.9, 0.1],
        ])
        pano_osm_similarity = torch.tensor([
            [0.9, 0.2],
            [0.1, 0.7],
            [0.6, 0.4],
            [0.3, 0.8],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0'])

        expected_sat = torch.tensor([[0.8, 0.3, 0.5]])
        torch.testing.assert_close(similarities.sat_patch, expected_sat)

        self.assertEqual(similarities.landmark.shape[0], 1)
        self.assertEqual(similarities.landmark.shape[2], 2)
        self.assertEqual(similarities.landmark_mask.shape, (1, 2))

    def test_get_similarities_multiple_panos(self):
        """Test extracting similarities for multiple panoramas."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [1]},
            {'pano_id': 'pano_2', 'pano_lm_idxs': [2]},
        ])

        pano_sat_similarity = torch.tensor([
            [0.8, 0.3],
            [0.2, 0.9],
            [0.5, 0.4],
        ])
        pano_osm_similarity = torch.tensor([
            [0.9],
            [0.7],
            [0.6],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0', 'pano_2'])

        expected_sat = torch.tensor([
            [0.8, 0.3],
            [0.5, 0.4],
        ])
        torch.testing.assert_close(similarities.sat_patch, expected_sat)

        self.assertEqual(similarities.landmark.shape, (2, 1, 1))
        self.assertEqual(similarities.landmark_mask.shape, (2, 1))

    def test_get_similarities_preserves_order(self):
        """Test that returned similarities match the order of requested pano_ids."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [1]},
            {'pano_id': 'pano_2', 'pano_lm_idxs': [2]},
        ])

        pano_sat_similarity = torch.tensor([
            [0.1],
            [0.2],
            [0.3],
        ])
        pano_osm_similarity = torch.tensor([
            [0.4],
            [0.5],
            [0.6],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_2', 'pano_1', 'pano_0'])

        expected_sat = torch.tensor([[0.3], [0.2], [0.1]])
        torch.testing.assert_close(similarities.sat_patch, expected_sat)

        self.assertEqual(similarities.landmark.shape, (3, 1, 1))
        self.assertEqual(similarities.landmark_mask.shape, (3, 1))

    def test_get_similarities_nonexistent_pano(self):
        """Test that requesting non-existent pano_id raises assertion error."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        with self.assertRaises(AssertionError):
            lol._get_similarities(prior_data, ['pano_99'])

    def test_get_similarities_empty_list(self):
        """Test with empty pano_ids list."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, [])

        self.assertEqual(similarities.sat_patch.shape, (0, 1))
        self.assertEqual(similarities.landmark.shape, (0, 0, 1))
        self.assertEqual(similarities.landmark_mask.shape, (0, 0))

    def test_get_similarities_pano_with_multiple_landmarks(self):
        """Test panorama with many landmarks."""
        osm_geometry = pd.DataFrame([
            {'osm_id': i, 'geometry_px': shapely.Point(100+i*10, 50+i*10), 'osm_embedding_idx': i}
            for i in range(3)
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0, 1, 2, 3, 4]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([
            [0.9, 0.1, 0.2],
            [0.8, 0.2, 0.3],
            [0.7, 0.3, 0.4],
            [0.6, 0.4, 0.5],
            [0.5, 0.5, 0.6],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0'])

        self.assertEqual(similarities.sat_patch.shape, (1, 1))
        self.assertEqual(similarities.landmark.shape, (1, 5, 3))
        self.assertEqual(similarities.landmark_mask.shape, (1, 5))

    def test_get_similarities_pano_with_no_landmarks(self):
        """Test panorama with empty landmark list."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': []},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0'])

        self.assertEqual(similarities.sat_patch.shape, (1, 1))
        self.assertEqual(similarities.landmark.shape, (1, 0, 1))
        self.assertEqual(similarities.landmark_mask.shape, (1, 0))

    def test_get_similarities_landmark_mask_values(self):
        """Test that landmark_mask correctly marks valid landmarks as True."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0, 1, 2]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([
            [0.9],
            [0.8],
            [0.7],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0'])

        expected_mask = torch.tensor([[True, True, True]])
        torch.testing.assert_close(similarities.landmark_mask, expected_mask)

    def test_get_similarities_landmark_mask_with_padding(self):
        """Test that landmark_mask correctly handles different landmark counts per pano."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0, 1, 2]},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [3]},
            {'pano_id': 'pano_2', 'pano_lm_idxs': [4, 5]},
        ])

        pano_sat_similarity = torch.tensor([[0.8], [0.7], [0.6]])
        pano_osm_similarity = torch.tensor([
            [0.9], [0.8], [0.7],
            [0.6],
            [0.5], [0.4],
        ])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0', 'pano_1', 'pano_2'])

        self.assertEqual(similarities.landmark_mask.shape, (3, 3))

        expected_mask = torch.tensor([
            [True,  True,  True],
            [True,  False, False],
            [True,  True,  False],
        ])
        torch.testing.assert_close(similarities.landmark_mask, expected_mask)

    def test_get_similarities_landmark_mask_empty_pano(self):
        """Test that pano with no landmarks has all-False mask."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': []},
            {'pano_id': 'pano_1', 'pano_lm_idxs': [0, 1]},
        ])

        pano_sat_similarity = torch.tensor([[0.8], [0.7]])
        pano_osm_similarity = torch.tensor([[0.9], [0.8]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0', 'pano_1'])

        expected_mask = torch.tensor([
            [False, False],
            [True,  True],
        ])
        torch.testing.assert_close(similarities.landmark_mask, expected_mask)

    def test_get_similarities_landmark_values_padded_correctly(self):
        """Test that padded positions in landmark tensor are zero."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        pano_metadata = pd.DataFrame([
            {'pano_id': 'pano_0', 'pano_lm_idxs': [0]},
        ])

        pano_sat_similarity = torch.tensor([[0.8]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        similarities = lol._get_similarities(prior_data, ['pano_0'])

        self.assertNotEqual(similarities.landmark[0, 0, 0].item(), 0.0)

        if similarities.landmark.shape[1] > 1:
            torch.testing.assert_close(
                similarities.landmark[0, 1:, :],
                torch.zeros_like(similarities.landmark[0, 1:, :])
            )

    def test_compute_pixel_locs_single_location(self):
        """Test that single lat/lon converts to single pixel location."""
        particle_locs_deg = torch.tensor([[37.7749, -122.4194]])

        pixel_locs = lol._compute_pixel_locs_px(particle_locs_deg)

        self.assertEqual(pixel_locs.shape, (1, 2))
        self.assertIsInstance(pixel_locs, torch.Tensor)

    def test_compute_pixel_locs_batch(self):
        """Test batch conversion preserves shape."""
        particle_locs_deg = torch.tensor([
            [37.7749, -122.4194],
            [40.7128, -74.0060],
            [51.5074, -0.1278],
        ])

        pixel_locs = lol._compute_pixel_locs_px(particle_locs_deg)

        self.assertEqual(pixel_locs.shape, (3, 2))
        self.assertFalse(torch.isnan(pixel_locs).any())
        self.assertFalse(torch.isinf(pixel_locs).any())

    def test_compute_pixel_locs_multidimensional(self):
        """Test that multi-dimensional particle tensors work."""
        particle_locs_deg = torch.tensor([
            [[37.0, -122.0], [37.1, -122.1], [37.2, -122.2]],
            [[38.0, -123.0], [38.1, -123.1], [38.2, -123.2]],
        ])

        pixel_locs = lol._compute_pixel_locs_px(particle_locs_deg)

        self.assertEqual(pixel_locs.shape, (2, 3, 2))

    def test_compute_pixel_locs_output_ordering(self):
        """Test that output is [y, x] not [x, y]."""
        particle_locs_deg = torch.tensor([[0.0, 0.0]])

        pixel_locs = lol._compute_pixel_locs_px(particle_locs_deg, zoom_level=20)

        y_direct, x_direct = web_mercator.latlon_to_pixel_coords(0.0, 0.0, 20)

        torch.testing.assert_close(pixel_locs[0, 0], torch.tensor(y_direct, dtype=torch.float32))
        torch.testing.assert_close(pixel_locs[0, 1], torch.tensor(x_direct, dtype=torch.float32))

    def test_compute_pixel_locs_default_zoom(self):
        """Test that default zoom level is 20."""
        particle_locs_deg = torch.tensor([[37.7749, -122.4194]])

        pixel_locs_default = lol._compute_pixel_locs_px(particle_locs_deg)
        pixel_locs_explicit = lol._compute_pixel_locs_px(particle_locs_deg, zoom_level=20)

        torch.testing.assert_close(pixel_locs_default, pixel_locs_explicit)

    def test_compute_pixel_locs_custom_zoom(self):
        """Test that different zoom levels produce different results."""
        particle_locs_deg = torch.tensor([[37.7749, -122.4194]])

        pixel_locs_10 = lol._compute_pixel_locs_px(particle_locs_deg, zoom_level=10)
        pixel_locs_20 = lol._compute_pixel_locs_px(particle_locs_deg, zoom_level=20)

        self.assertFalse(torch.allclose(pixel_locs_10, pixel_locs_20))

    def test_compute_pixel_locs_torch_compatibility(self):
        """Test that function works with torch tensors and supports autograd."""
        particle_locs_deg = torch.tensor([[37.7749, -122.4194]], requires_grad=True)

        pixel_locs = lol._compute_pixel_locs_px(particle_locs_deg)

        self.assertIsInstance(pixel_locs, torch.Tensor)
        self.assertTrue(pixel_locs.requires_grad)

    def test_compute_sat_log_likelihood_single_particle(self):
        """Test that single particle produces scalar output."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        similarities = torch.tensor([[0.8]])
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (1,))
        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_sat_log_likelihood_batch(self):
        """Test that batch of particles produces correct shape."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])

        similarities = torch.tensor([
            [0.8, 0.3],
            [0.5, 0.6],
            [0.2, 0.9]
        ])
        particle_locs_px = torch.tensor([
            [100.0, 50.0],
            [150.0, 100.0],
            [200.0, 150.0],
        ])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (3,))

    def test_compute_sat_log_likelihood_multidimensional(self):
        """Test with 2D grid of particles."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        similarities = torch.tensor([[0.8], [0.7]])
        particle_locs_px = torch.tensor([
            [[100.0, 50.0], [110.0, 60.0], [120.0, 70.0]],
            [[130.0, 80.0], [140.0, 90.0], [150.0, 100.0]],
        ])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (2, 3))

    def test_compute_sat_log_likelihood_near_high_similarity_patch(self):
        """Test that particles near high-similarity patches have higher likelihood."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])

        similarities = torch.tensor([[0.9, 0.1]])
        particle_near_high = torch.tensor([[100.0, 50.0]])
        particle_near_low = torch.tensor([[200.0, 150.0]])

        ll_near_high = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_near_high
        )
        ll_near_low = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_near_low
        )

        self.assertGreater(ll_near_high.item(), ll_near_low.item())

    def test_compute_sat_log_likelihood_multiple_patches(self):
        """Test with multiple satellite patches."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
            {'geometry_px': create_square_patch(300, 250, size=20), 'embedding_idx': 2},
        ])

        similarities = torch.tensor([[0.8, 0.5, 0.2]])
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (1,))
        self.assertFalse(torch.isnan(log_likelihood).any())
        self.assertFalse(torch.isinf(log_likelihood).any())

    def test_compute_sat_log_likelihood_far_from_patches(self):
        """Test that particles far from all patches get low likelihood."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        similarities = torch.tensor([[0.9]])
        particle_near = torch.tensor([[100.0, 50.0]])
        particle_far = torch.tensor([[1000.0, 1000.0]])

        ll_near = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_near
        )
        ll_far = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_far
        )

        self.assertGreater(ll_near.item(), ll_far.item())

    def test_compute_sat_log_likelihood_no_patches(self):
        """Test with empty satellite geometry."""
        sat_geometry = pd.DataFrame(columns=['geometry_px', 'embedding_idx'])

        similarities = torch.zeros((1, 0))
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (1,))

    def test_compute_sat_log_likelihood_zero_similarity(self):
        """Test with all zero similarities."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        similarities = torch.tensor([[0.0]])
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertEqual(log_likelihood.shape, (1,))

    def test_compute_sat_log_likelihood_returns_tensor(self):
        """Test that function returns a torch.Tensor."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])

        similarities = torch.tensor([[0.8]])
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_locs_px
        )

        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_osm_log_likelihood_single_particle(self):
        """Test that single particle produces scalar output."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_osm_log_likelihood_batch(self):
        """Test that batch of particles produces correct shape."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])

        similarities = torch.tensor([
            [[0.8, 0.2], [0.3, 0.7]],
        ])
        mask = torch.tensor([
            [True, True],
        ])
        particle_locs_px = torch.tensor([[
            [100.0, 50.0],
            [150.0, 100.0],
            [200.0, 150.0],
        ]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 3))

    def test_compute_osm_log_likelihood_multidimensional(self):
        """Test with multiple panos, each with multiple particles."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([
            [[0.8], [0.7]],
            [[0.6], [0.5]]
        ])
        mask = torch.tensor([
            [True, True],
            [True, True]
        ])
        particle_locs_px = torch.tensor([
            [[100.0, 50.0], [110.0, 60.0]],
            [[120.0, 70.0], [130.0, 80.0]],
        ])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (2, 2))

    def test_compute_osm_log_likelihood_near_high_similarity_landmark(self):
        """Test that particles near high-similarity landmarks have higher likelihood."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])

        similarities = torch.tensor([[[0.9, 0.1]]])
        mask = torch.tensor([[True]])

        particle_near_high = torch.tensor([[[100.0, 50.0]]])
        particle_near_low = torch.tensor([[[200.0, 150.0]]])

        ll_near_high = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near_high, point_sigma_px=300
        )
        ll_near_low = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near_low, point_sigma_px=300
        )

        self.assertGreater(ll_near_high.item(), ll_near_low.item())

    def test_compute_osm_log_likelihood_multiple_landmarks(self):
        """Test with multiple OSM landmarks."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
            {'osm_id': 2, 'geometry_px': shapely.Point(300, 250), 'osm_embedding_idx': 2},
        ])

        similarities = torch.tensor([[[0.8, 0.5, 0.2], [0.3, 0.6, 0.9]]])
        mask = torch.tensor([[True, True]])
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertFalse(torch.isnan(log_likelihood).any())
        self.assertFalse(torch.isinf(log_likelihood).any())

    def test_compute_osm_log_likelihood_far_from_landmarks(self):
        """Test that particles far from all landmarks get low likelihood."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[True]])

        particle_near = torch.tensor([[[100.0, 50.0]]])
        particle_far = torch.tensor([[[1000.0, 1000.0]]])

        ll_near = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near, point_sigma_px=300
        )
        ll_far = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_far, point_sigma_px=300
        )

        self.assertGreater(ll_near.item(), ll_far.item())

    def test_compute_osm_log_likelihood_masked_landmarks_ignored(self):
        """Test that masked landmarks (False in mask) don't contribute."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])

        similarities_both = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
        mask_all = torch.tensor([[True, True]])
        mask_first_only = torch.tensor([[True, False]])

        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        ll_all = lol._compute_osm_log_likelihood(
            similarities_both, mask_all, osm_geometry, particle_locs_px, point_sigma_px=300
        )
        ll_masked = lol._compute_osm_log_likelihood(
            similarities_both, mask_first_only, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertNotEqual(ll_all.item(), ll_masked.item())

    def test_compute_osm_log_likelihood_all_masked(self):
        """Test pano with all landmarks masked."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[False]])
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_no_landmarks(self):
        """Test with empty OSM geometry."""
        osm_geometry = pd.DataFrame(columns=['osm_id', 'geometry_px', 'osm_embedding_idx'])

        similarities = torch.zeros((1, 1, 0))
        mask = torch.zeros((1, 1), dtype=torch.bool)
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_zero_similarity(self):
        """Test with all zero similarities."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([[[0.0]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_returns_tensor(self):
        """Test that function returns a torch.Tensor."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])

        similarities = torch.tensor([[[0.8]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300
        )

        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_sat_log_likelihood_overlapping_patches(self):
        """Test behavior with overlapping satellite patches."""
        # Create two overlapping square patches
        # Patch 0: centered at (100, 100), size 40 -> covers [80, 120] x [80, 120]
        # Patch 1: centered at (110, 110), size 40 -> covers [90, 130] x [90, 130]
        # Overlap region: [90, 120] x [90, 120]
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 100, size=40), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(110, 110, size=40), 'embedding_idx': 1},
        ])

        # Give different similarities to the two patches
        # Panorama has high similarity to patch 0, low similarity to patch 1
        similarities = torch.tensor([[0.9, 0.2]])

        # Test 1: Particle in the overlap region at (105, 105)
        # Should be inside both patches
        particle_in_overlap = torch.tensor([[[105.0, 105.0]]])

        ll_overlap = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_in_overlap
        )

        # Should get a finite likelihood (not -inf) since particle is inside at least one patch
        self.assertFalse(torch.isinf(ll_overlap))
        self.assertFalse(torch.isnan(ll_overlap))

        # Test 2: Particle outside all patches at (200, 200)
        particle_outside = torch.tensor([[[200.0, 200.0]]])

        ll_outside = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_outside
        )

        # Should get -inf likelihood since particle is outside all patches
        self.assertTrue(torch.isinf(ll_outside))
        self.assertTrue(ll_outside.item() < 0)

        # Test 3: Particle inside only patch 0 at (85, 85)
        particle_in_patch0_only = torch.tensor([[[85.0, 85.0]]])

        ll_patch0 = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_in_patch0_only
        )

        # Should use patch 0's similarity (0.9), which is higher
        self.assertFalse(torch.isinf(ll_patch0))

        # Test 4: Particle inside only patch 1 at (125, 125)
        particle_in_patch1_only = torch.tensor([[[125.0, 125.0]]])

        ll_patch1 = lol._compute_sat_log_likelihood(
            similarities, sat_geometry, particle_in_patch1_only
        )

        # Should use patch 1's similarity (0.2), which is lower
        self.assertFalse(torch.isinf(ll_patch1))

        # Particle inside high-similarity patch should have higher likelihood
        self.assertGreater(ll_patch0.item(), ll_patch1.item())

    def test_compute_sat_log_likelihood_overlapping_patches_nearest_patch(self):
        """Test that overlapping patches use nearest patch deterministically."""
        # Create two overlapping square patches
        # Patch 0: centered at (100, 100), size 40 -> covers [80, 120] x [80, 120]
        # Patch 1: centered at (110, 110), size 40 -> covers [90, 130] x [90, 130]
        # Overlap region: [90, 120] x [90, 120]
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 100, size=40), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(110, 110, size=40), 'embedding_idx': 1},
        ])

        # Particle at (105, 105) is in the overlap region
        # Distance to patch 0 center (100, 100): sqrt(25 + 25) ≈ 7.07
        # Distance to patch 1 center (110, 110): sqrt(25 + 25) ≈ 7.07
        # They're equidistant! Let's use a point closer to patch 0
        particle_in_overlap = torch.tensor([[[102.0, 102.0]]])
        # Distance to patch 0 center (100, 100): sqrt(4 + 4) ≈ 2.83
        # Distance to patch 1 center (110, 110): sqrt(64 + 64) ≈ 11.31
        # So this should use patch 0

        # Case A: patch 0 has high similarity, patch 1 has low similarity
        similarities_A = torch.tensor([[0.9, 0.2]])
        ll_A = lol._compute_sat_log_likelihood(
            similarities_A, sat_geometry, particle_in_overlap
        )

        # Case B: swap the similarities
        # patch 0 has low similarity, patch 1 has high similarity
        similarities_B = torch.tensor([[0.2, 0.9]])
        ll_B = lol._compute_sat_log_likelihood(
            similarities_B, sat_geometry, particle_in_overlap
        )

        # If the implementation uses nearest patch (patch 0), then:
        # - Case A should use patch 0's similarity (0.9) -> higher likelihood
        # - Case B should use patch 0's similarity (0.2) -> lower likelihood
        # So ll_A should be greater than ll_B
        #
        # If the implementation is non-deterministic or always uses patch 1:
        # - Case A might use patch 1's similarity (0.2)
        # - Case B might use patch 1's similarity (0.9)
        # Then ll_A would be less than ll_B (opposite of expected)

        # With nearest patch, Case A should have higher likelihood
        self.assertGreater(
            ll_A.item(), ll_B.item(),
            "Particle at (102, 102) should use patch 0 (centered at 100, 100) "
            "which is closer than patch 1 (centered at 110, 110). "
            "Case A has patch 0 with similarity 0.9, Case B has patch 0 with similarity 0.2, "
            f"so Case A should have higher likelihood. {ll_A.item()=} {ll_B.item()=}"
        )


if __name__ == "__main__":
    unittest.main()
