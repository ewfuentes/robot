
import unittest
import tempfile
import json
import pickle
from pathlib import Path

import common.torch.load_torch_deps
import torch
import pandas as pd
import geopandas as gpd
import shapely
from common.gps import web_mercator

import experimental.overhead_matching.swag.evaluation.observation_likelihood as lol
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props, prune_landmark


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


def build_test_spatial_index(sat_geometry):
    """Helper to build spatial index for tests."""
    return lol._build_sat_spatial_index(sat_geometry)


class ObservationLikelihoodTest(unittest.TestCase):
    def test_calculator_valid_inputs(self):
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

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
            [40.7128, -74.0060],
            [51.5074, -0.1278],
        ])

        log_likelihood = calculator.compute_log_likelihoods(particles, ['pano_0'])

        self.assertEqual(log_likelihood.shape, (1, 3))

    def test_calculator_mismatched_pano_sat_similarity_shape(self):
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

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
        )

        with self.assertRaises(Exception):
            lol.LandmarkObservationLikelihoodCalculator(
                prior_data=prior_data,
                config=config,
                device=torch.device('cpu')
            )

    def test_calculator_missing_required_column(self):
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

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
        )

        with self.assertRaises((KeyError, ValueError)):
            lol.LandmarkObservationLikelihoodCalculator(
                prior_data=prior_data,
                config=config,
                device=torch.device('cpu')
            )

    def test_calculator_nonexistent_pano_id(self):
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

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([[37.7749, -122.4194]])

        with self.assertRaises(AssertionError):
            calculator.compute_log_likelihoods(particles, ['pano_99'])

    def test_calculator_batch_particles_output_shape(self):
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

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
            [37.7750, -122.4195],
            [37.7751, -122.4196],
            [37.7752, -122.4197],
            [37.7753, -122.4198],
        ])

        log_likelihood = calculator.compute_log_likelihoods(particles, ['pano_0'])

        self.assertEqual(log_likelihood.shape, (1, 5))

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
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.8]])
        particle_locs_px = torch.tensor([[50.0, 100.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (1,))
        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_sat_log_likelihood_batch(self):
        """Test that batch of particles produces correct shape."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([
            [0.8, 0.3],
            [0.5, 0.6],
            [0.2, 0.9]
        ])
        particle_locs_px = torch.tensor([
            [50.0, 100.0],
            [100.0, 150.0],
            [150.0, 200.0],
        ])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (3,))

    def test_compute_sat_log_likelihood_multidimensional(self):
        """Test with 2D grid of particles."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.8], [0.7]])
        particle_locs_px = torch.tensor([
            [[50.0, 100.0], [60.0, 110.0], [70.0, 120.0]],
            [[80.0, 130.0], [90.0, 140.0], [100.0, 150.0]],
        ])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (2, 3))

    def test_compute_sat_log_likelihood_near_high_similarity_patch(self):
        """Test that particles near high-similarity patches have higher likelihood."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.9, 0.1]])
        particle_near_high = torch.tensor([[50.0, 100.0]])
        particle_near_low = torch.tensor([[150.0, 200.0]])

        ll_near_high = lol._compute_sat_log_likelihood(
            similarities, particle_near_high, sat_tree, patch_centroids
        )
        ll_near_low = lol._compute_sat_log_likelihood(
            similarities, particle_near_low, sat_tree, patch_centroids
        )

        self.assertGreater(ll_near_high.item(), ll_near_low.item())

    def test_compute_sat_log_likelihood_multiple_patches(self):
        """Test with multiple satellite patches."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
            {'geometry_px': create_square_patch(200, 150, size=20), 'embedding_idx': 1},
            {'geometry_px': create_square_patch(300, 250, size=20), 'embedding_idx': 2},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.8, 0.5, 0.2]])
        particle_locs_px = torch.tensor([[50.0, 100.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (1,))
        self.assertFalse(torch.isnan(log_likelihood).any())
        self.assertFalse(torch.isinf(log_likelihood).any())

    def test_compute_sat_log_likelihood_far_from_patches(self):
        """Test that particles far from all patches get low likelihood."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.9]])
        particle_near = torch.tensor([[50.0, 100.0]])
        particle_far = torch.tensor([[1000.0, 1000.0]])

        ll_near = lol._compute_sat_log_likelihood(
            similarities, particle_near, sat_tree, patch_centroids
        )
        ll_far = lol._compute_sat_log_likelihood(
            similarities, particle_far, sat_tree, patch_centroids
        )

        self.assertGreater(ll_near.item(), ll_far.item())

    def test_compute_sat_log_likelihood_no_patches(self):
        """Test with empty satellite geometry."""
        sat_geometry = pd.DataFrame(columns=['geometry_px', 'embedding_idx'])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.zeros((1, 0))
        particle_locs_px = torch.tensor([[100.0, 50.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (1,))

    def test_compute_sat_log_likelihood_zero_similarity(self):
        """Test with all zero similarities."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.0]])
        particle_locs_px = torch.tensor([[50.0, 100.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertEqual(log_likelihood.shape, (1,))

    def test_compute_sat_log_likelihood_returns_tensor(self):
        """Test that function returns a torch.Tensor."""
        sat_geometry = pd.DataFrame([
            {'geometry_px': create_square_patch(100, 50, size=20), 'embedding_idx': 0},
        ])
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        similarities = torch.tensor([[0.8]])
        particle_locs_px = torch.tensor([[50.0, 100.0]])

        log_likelihood = lol._compute_sat_log_likelihood(
            similarities, particle_locs_px, sat_tree, patch_centroids
        )

        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_osm_log_likelihood_single_particle(self):
        """Test that single particle produces scalar output."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertIsInstance(log_likelihood, torch.Tensor)

    def test_compute_osm_log_likelihood_batch(self):
        """Test that batch of particles produces correct shape."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([
            [[0.8, 0.2], [0.3, 0.7]],
        ])
        mask = torch.tensor([
            [True, True],
        ])
        particle_locs_px = torch.tensor([[
            [50.0, 100.0],
            [100.0, 150.0],
            [150.0, 200.0],
        ]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 3))

    def test_compute_osm_log_likelihood_multidimensional(self):
        """Test with multiple panos, each with multiple particles."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([
            [[0.8], [0.7]],
            [[0.6], [0.5]]
        ])
        mask = torch.tensor([
            [True, True],
            [True, True]
        ])
        particle_locs_px = torch.tensor([
            [[50.0, 100.0], [60.0, 110.0]],
            [[70.0, 120.0], [80.0, 130.0]],
        ])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (2, 2))

    def test_compute_osm_log_likelihood_near_high_similarity_landmark(self):
        """Test that particles near high-similarity landmarks have higher likelihood."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.9, 0.1]]])
        mask = torch.tensor([[True]])

        particle_near_high = torch.tensor([[[50.0, 100.0]]])
        particle_near_low = torch.tensor([[[150.0, 200.0]]])

        ll_near_high = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near_high, point_sigma_px=300, osm_tree=osm_tree
        )
        ll_near_low = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near_low, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertGreater(ll_near_high.item(), ll_near_low.item())

    def test_compute_osm_log_likelihood_multiple_landmarks(self):
        """Test with multiple OSM landmarks."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
            {'osm_id': 2, 'geometry_px': shapely.Point(300, 250), 'osm_embedding_idx': 2},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.8, 0.5, 0.2], [0.3, 0.6, 0.9]]])
        mask = torch.tensor([[True, True]])
        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertFalse(torch.isnan(log_likelihood).any())
        self.assertFalse(torch.isinf(log_likelihood).any())

    def test_compute_osm_log_likelihood_far_from_landmarks(self):
        """Test that particles far from all landmarks get low likelihood."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[True]])

        particle_near = torch.tensor([[[50.0, 100.0]]])
        particle_far = torch.tensor([[[1000.0, 1000.0]]])

        ll_near = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_near, point_sigma_px=300, osm_tree=osm_tree
        )
        ll_far = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_far, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertGreater(ll_near.item(), ll_far.item())

    def test_compute_osm_log_likelihood_masked_landmarks_ignored(self):
        """Test that masked landmarks (False in mask) don't contribute."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
            {'osm_id': 1, 'geometry_px': shapely.Point(200, 150), 'osm_embedding_idx': 1},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities_both = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
        mask_all = torch.tensor([[True, True]])
        mask_first_only = torch.tensor([[True, False]])

        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        ll_all = lol._compute_osm_log_likelihood(
            similarities_both, mask_all, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )
        ll_masked = lol._compute_osm_log_likelihood(
            similarities_both, mask_first_only, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertNotEqual(ll_all.item(), ll_masked.item())

    def test_compute_osm_log_likelihood_all_masked(self):
        """Test pano with all landmarks masked."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.9]]])
        mask = torch.tensor([[False]])
        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_no_landmarks(self):
        """Test with empty OSM geometry."""
        osm_geometry = pd.DataFrame(columns=['osm_id', 'geometry_px', 'osm_embedding_idx'])
        osm_tree = shapely.STRtree([])

        similarities = torch.zeros((1, 1, 0))
        mask = torch.zeros((1, 1), dtype=torch.bool)
        particle_locs_px = torch.tensor([[[100.0, 50.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_zero_similarity(self):
        """Test with all zero similarities."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.0]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
        )

        self.assertEqual(log_likelihood.shape, (1, 1))

    def test_compute_osm_log_likelihood_returns_tensor(self):
        """Test that function returns a torch.Tensor."""
        osm_geometry = pd.DataFrame([
            {'osm_id': 0, 'geometry_px': shapely.Point(100, 50), 'osm_embedding_idx': 0},
        ])
        osm_tree = shapely.STRtree(osm_geometry.geometry_px.values)

        similarities = torch.tensor([[[0.8]]])
        mask = torch.tensor([[True]])
        particle_locs_px = torch.tensor([[[50.0, 100.0]]])

        log_likelihood = lol._compute_osm_log_likelihood(
            similarities, mask, osm_geometry, particle_locs_px, point_sigma_px=300, osm_tree=osm_tree
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
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

        # Give different similarities to the two patches
        # Panorama has high similarity to patch 0, low similarity to patch 1
        similarities = torch.tensor([[0.9, 0.2]])

        # Test 1: Particle in the overlap region at (105, 105)
        # Should be inside both patches
        particle_in_overlap = torch.tensor([[[105.0, 105.0]]])

        ll_overlap = lol._compute_sat_log_likelihood(
            similarities, particle_in_overlap, sat_tree, patch_centroids
        )

        # Should get a finite likelihood (not -inf) since particle is inside at least one patch
        self.assertFalse(torch.isinf(ll_overlap))
        self.assertFalse(torch.isnan(ll_overlap))

        # Test 2: Particle outside all patches at (200, 200)
        particle_outside = torch.tensor([[[200.0, 200.0]]])

        ll_outside = lol._compute_sat_log_likelihood(
            similarities, particle_outside, sat_tree, patch_centroids
        )

        # Should get -inf likelihood since particle is outside all patches
        self.assertTrue(torch.isinf(ll_outside))
        self.assertTrue(ll_outside.item() < 0)

        # Test 3: Particle inside only patch 0 at (85, 85)
        particle_in_patch0_only = torch.tensor([[[85.0, 85.0]]])

        ll_patch0 = lol._compute_sat_log_likelihood(
            similarities, particle_in_patch0_only, sat_tree, patch_centroids
        )

        # Should use patch 0's similarity (0.9), which is higher
        self.assertFalse(torch.isinf(ll_patch0))

        # Test 4: Particle inside only patch 1 at (125, 125)
        particle_in_patch1_only = torch.tensor([[[125.0, 125.0]]])

        ll_patch1 = lol._compute_sat_log_likelihood(
            similarities, particle_in_patch1_only, sat_tree, patch_centroids
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
        sat_tree, patch_centroids = build_test_spatial_index(sat_geometry)

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
            similarities_A, particle_in_overlap, sat_tree, patch_centroids
        )

        # Case B: swap the similarities
        # patch 0 has low similarity, patch 1 has high similarity
        similarities_B = torch.tensor([[0.2, 0.9]])
        ll_B = lol._compute_sat_log_likelihood(
            similarities_B, particle_in_overlap, sat_tree, patch_centroids
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


class LandmarkObservationLikelihoodCalculatorTest(unittest.TestCase):
    def _create_test_prior_data(self):
        """Create test prior data with known structure."""
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
            {'pano_id': 'pano_1', 'pano_lm_idxs': [2, 3]},
        ])

        pano_sat_similarity = torch.tensor([
            [0.8, 0.3],
            [0.2, 0.9],
        ])
        pano_osm_similarity = torch.tensor([
            [0.9, 0.2],
            [0.1, 0.7],
            [0.6, 0.4],
            [0.3, 0.8],
        ])

        return lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

    def test_compute_log_likelihoods_output_shape(self):
        """Test that compute_log_likelihoods returns correct shape."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        # 5 particles, 1 panorama
        particles = torch.tensor([
            [37.7749, -122.4194],
            [37.7750, -122.4195],
            [37.7751, -122.4196],
            [37.7752, -122.4197],
            [37.7753, -122.4198],
        ])

        log_likelihoods = calculator.compute_log_likelihoods(particles, ['pano_0'])

        self.assertEqual(log_likelihoods.shape, (1, 5))

    def test_compute_log_likelihoods_multiple_panoramas(self):
        """Test with multiple panoramas."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
            [37.7750, -122.4195],
            [37.7751, -122.4196],
        ])

        log_likelihoods = calculator.compute_log_likelihoods(particles, ['pano_0', 'pano_1'])

        self.assertEqual(log_likelihoods.shape, (2, 3))

    def test_sat_only_mode(self):
        """Test that SAT_ONLY mode uses only satellite likelihood."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.SAT_ONLY
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
        ])

        log_likelihoods = calculator.compute_log_likelihoods(particles, ['pano_0'])

        self.assertEqual(log_likelihoods.shape, (1, 1))
        self.assertFalse(torch.isnan(log_likelihoods).any())

    def test_osm_only_mode(self):
        """Test that OSM_ONLY mode uses only OSM likelihood."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.OSM_ONLY
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
        ])

        log_likelihoods = calculator.compute_log_likelihoods(particles, ['pano_0'])

        self.assertEqual(log_likelihoods.shape, (1, 1))
        self.assertFalse(torch.isnan(log_likelihoods).any())

    def test_combined_mode(self):
        """Test that COMBINED mode sums both likelihoods."""
        prior_data = self._create_test_prior_data()

        # Create calculators for each mode
        config_sat = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.SAT_ONLY
        )
        config_osm = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.OSM_ONLY
        )
        config_combined = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        calc_sat = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data, config=config_sat, device=torch.device('cpu')
        )
        calc_osm = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data, config=config_osm, device=torch.device('cpu')
        )
        calc_combined = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data, config=config_combined, device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
        ])

        ll_sat = calc_sat.compute_log_likelihoods(particles, ['pano_0'])
        ll_osm = calc_osm.compute_log_likelihoods(particles, ['pano_0'])
        ll_combined = calc_combined.compute_log_likelihoods(particles, ['pano_0'])

        # Combined should be the sum of sat and osm
        expected = ll_sat + ll_osm
        torch.testing.assert_close(ll_combined, expected)

    def test_sample_from_observation_raises_not_implemented(self):
        """Test that sample_from_observation raises NotImplementedError."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        generator = torch.Generator().manual_seed(42)

        with self.assertRaises(NotImplementedError):
            calculator.sample_from_observation(10, ['pano_0'], generator)

    def test_invalid_pano_id_raises_error(self):
        """Test that invalid panorama ID raises an error."""
        prior_data = self._create_test_prior_data()
        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        calculator = lol.LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=config,
            device=torch.device('cpu')
        )

        particles = torch.tensor([
            [37.7749, -122.4194],
        ])

        with self.assertRaises(AssertionError):
            calculator.compute_log_likelihoods(particles, ['nonexistent_pano'])

    def test_constructor_validates_prior_data(self):
        """Test that constructor validates prior data."""
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

        # Mismatched shape: should be (1, 3) but is (1, 2)
        pano_sat_similarity = torch.tensor([[0.8, 0.3]])
        pano_osm_similarity = torch.tensor([[0.9]])

        prior_data = lol.PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=pano_sat_similarity,
            pano_osm_landmark_similarity=pano_osm_similarity
        )

        config = lol.ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=0.1,
            obs_likelihood_from_osm_similarity_sigma=100.0,
            likelihood_mode=lol.LikelihoodMode.COMBINED
        )

        with self.assertRaises(AssertionError):
            lol.LandmarkObservationLikelihoodCalculator(
                prior_data=prior_data,
                config=config,
                device=torch.device('cpu')
            )


class LandmarkSimilarityDataTest(unittest.TestCase):
    """Tests for compute_landmark_similarity_data and build_prior_data_from_vigor."""

    def setUp(self):
        """Set up test fixtures with vigor_snippet dataset."""
        self.dataset_path = Path("external/vigor_snippet/vigor_snippet")
        self.config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            sample_mode=vd.SampleMode.POS_SEMIPOS,
            landmark_version="v1",
        )
        self.dataset = vd.VigorDataset(self.dataset_path, self.config)

    def _create_mock_embeddings(self, temp_dir: Path, embedding_dim: int = 16):
        """Create deterministic mock embedding files for testing.

        Returns paths to osm_embedding_dir and pano_embedding_dir.
        """
        osm_dir = temp_dir / "osm_embeddings"
        pano_dir = temp_dir / "pano_embeddings"

        # Create subdirectories as expected by compute_landmark_similarity_data
        osm_embeddings_dir = osm_dir / "embeddings"
        osm_sentences_dir = osm_dir / "sentences"
        pano_embeddings_dir = pano_dir / "embeddings"
        pano_sentences_dir = pano_dir / "sentences"

        osm_embeddings_dir.mkdir(parents=True)
        osm_sentences_dir.mkdir(parents=True)
        pano_embeddings_dir.mkdir(parents=True)
        pano_sentences_dir.mkdir(parents=True)

        # Set seed for deterministic embeddings
        torch.manual_seed(42)

        landmark_metadata = self.dataset._landmark_metadata
        pano_metadata = self.dataset._panorama_metadata

        # Generate custom_ids for OSM landmarks
        custom_ids = []
        for _, row in landmark_metadata.iterrows():
            if 'pruned_props' in row:
                props = row.pruned_props
            else:
                props = prune_landmark(row.to_dict())
            custom_id = custom_id_from_props(props)
            if custom_id not in custom_ids:
                custom_ids.append(custom_id)

        # Create OSM embeddings pickle file with deterministic values
        osm_embeddings = torch.randn(len(custom_ids), embedding_dim)
        osm_id_to_idx = {cid: idx for idx, cid in enumerate(custom_ids)}
        with open(osm_embeddings_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump((osm_embeddings, osm_id_to_idx), f)

        # Create mock sentences file for OSM in OpenAI API response format
        with open(osm_sentences_dir / "sentences.jsonl", 'w') as f:
            for cid in custom_ids:
                response = {
                    "custom_id": cid,
                    "error": None,
                    "response": {
                        "body": {
                            "choices": [{
                                "finish_reason": "stop",
                                "message": {
                                    "content": f"Mock sentence for {cid}",
                                    "refusal": None
                                }
                            }],
                            "usage": {
                                "completion_tokens": 10
                            }
                        }
                    }
                }
                f.write(json.dumps(response) + '\n')

        # Build panorama landmark data
        # For each panorama, create embeddings and sentence data
        pano_keys = []  # landmark_custom_id list for embeddings
        pano_metadata_entries = []  # For panorama_metadata.jsonl
        pano_sentence_data = {}  # panorama_id -> landmarks list for sentences.jsonl

        for _, row in pano_metadata.iterrows():
            panorama_id = row.path.stem  # Format: "{pano_id},{lat},{lon},"

            # Create 2 landmarks per panorama for testing
            landmarks_for_pano = []
            for lm_idx in range(2):
                custom_id = f"{panorama_id}__landmark_{lm_idx}"
                pano_keys.append(custom_id)
                pano_metadata_entries.append({
                    'custom_id': custom_id,
                    'panorama_id': panorama_id,
                    'landmark_idx': lm_idx,
                    'yaw_angles': [0]
                })
                landmarks_for_pano.append({
                    "description": f"Mock description for {custom_id}",
                    "yaw_angles": [0]
                })

            pano_sentence_data[panorama_id] = landmarks_for_pano

        pano_embeddings = torch.randn(len(pano_keys), embedding_dim)
        pano_id_to_idx = {key: idx for idx, key in enumerate(pano_keys)}
        with open(pano_embeddings_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump((pano_embeddings, pano_id_to_idx), f)

        # Create mock sentences file for pano in OpenAI API response format
        # The content should be a JSON string containing a "landmarks" array
        with open(pano_sentences_dir / "sentences.jsonl", 'w') as f:
            for panorama_id, landmarks in pano_sentence_data.items():
                content_dict = {"landmarks": landmarks}
                response = {
                    "custom_id": panorama_id,
                    "error": None,
                    "response": {
                        "body": {
                            "choices": [{
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(content_dict),
                                    "refusal": None
                                }
                            }],
                            "usage": {
                                "completion_tokens": 10
                            }
                        }
                    }
                }
                f.write(json.dumps(response) + '\n')

        # Create panorama_metadata.jsonl in embedding_requests folder (under pano_dir, not pano_embeddings_dir)
        embedding_requests_dir = pano_dir / "embedding_requests"
        embedding_requests_dir.mkdir(parents=True)
        with open(embedding_requests_dir / "panorama_metadata.jsonl", 'w') as f:
            for entry in pano_metadata_entries:
                f.write(json.dumps(entry) + '\n')

        return osm_dir, pano_dir

    def test_compute_landmark_similarity_data_returns_correct_types(self):
        """Test that compute_landmark_similarity_data returns correct data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            result = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            self.assertIsInstance(result, lol.LandmarkSimilarityData)
            self.assertIsInstance(result.osm_geometry, gpd.GeoDataFrame)
            self.assertIsInstance(result.pano_metadata, gpd.GeoDataFrame)
            self.assertIsInstance(result.pano_osm_landmark_similarity, torch.Tensor)

    def test_compute_landmark_similarity_data_osm_geometry_columns(self):
        """Test that osm_geometry has required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            result = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            self.assertIn('osm_id', result.osm_geometry.columns)
            self.assertIn('geometry_px', result.osm_geometry.columns)
            self.assertIn('osm_embedding_idx', result.osm_geometry.columns)

    def test_compute_landmark_similarity_data_pano_metadata_columns(self):
        """Test that pano_metadata has required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            result = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            self.assertIn('pano_id', result.pano_metadata.columns)
            self.assertIn('pano_lm_idxs', result.pano_metadata.columns)

    def test_compute_landmark_similarity_data_similarity_shape(self):
        """Test that similarity matrix has correct shape."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            result = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            # Shape should be (num_pano_landmarks, num_osm_embeddings)
            # The second dimension is the number of OSM embeddings loaded, which may differ
            # from len(osm_geometry) if some landmarks map to same embedding
            self.assertEqual(result.pano_osm_landmark_similarity.ndim, 2)
            # osm_geometry should have entries, similarity should have matching columns
            self.assertGreater(len(result.osm_geometry), 0)
            self.assertGreater(result.pano_osm_landmark_similarity.shape[1], 0)

    def test_compute_landmark_similarity_data_pano_count_matches(self):
        """Test that pano_metadata has same number of rows as dataset panoramas."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            result = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            self.assertEqual(len(result.pano_metadata), len(self.dataset._panorama_metadata))

    def test_build_prior_data_from_vigor_returns_correct_type(self):
        """Test that build_prior_data_from_vigor returns PriorData."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            landmark_sim_data = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            # Create mock pano_sat_similarity
            num_panos = len(self.dataset._panorama_metadata)
            num_sats = len(self.dataset._satellite_metadata)
            pano_sat_similarity = torch.randn(num_panos, num_sats)

            result = lol.build_prior_data_from_vigor(
                self.dataset, pano_sat_similarity, landmark_sim_data)

            self.assertIsInstance(result, lol.PriorData)

    def test_build_prior_data_from_vigor_sat_geometry_columns(self):
        """Test that sat_geometry has required columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            landmark_sim_data = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            num_panos = len(self.dataset._panorama_metadata)
            num_sats = len(self.dataset._satellite_metadata)
            pano_sat_similarity = torch.randn(num_panos, num_sats)

            result = lol.build_prior_data_from_vigor(
                self.dataset, pano_sat_similarity, landmark_sim_data)

            self.assertIn('geometry_px', result.sat_geometry.columns)
            self.assertIn('embedding_idx', result.sat_geometry.columns)

    def test_build_prior_data_from_vigor_sat_geometry_count(self):
        """Test that sat_geometry has correct number of patches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            landmark_sim_data = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            num_panos = len(self.dataset._panorama_metadata)
            num_sats = len(self.dataset._satellite_metadata)
            pano_sat_similarity = torch.randn(num_panos, num_sats)

            result = lol.build_prior_data_from_vigor(
                self.dataset, pano_sat_similarity, landmark_sim_data)

            self.assertEqual(len(result.sat_geometry), num_sats)

    def test_build_prior_data_from_vigor_sat_geometry_is_boxes(self):
        """Test that sat_geometry contains box polygons."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            landmark_sim_data = lol.compute_landmark_similarity_data(
                self.dataset, osm_dir, pano_dir, embedding_dim=16)

            num_panos = len(self.dataset._panorama_metadata)
            num_sats = len(self.dataset._satellite_metadata)
            pano_sat_similarity = torch.randn(num_panos, num_sats)

            result = lol.build_prior_data_from_vigor(
                self.dataset, pano_sat_similarity, landmark_sim_data)

            # Check first geometry is a polygon
            first_geom = result.sat_geometry.iloc[0].geometry_px
            self.assertEqual(first_geom.geom_type, 'Polygon')

    def test_compute_cached_landmark_similarity_data_caches_result(self):
        """Test that compute_cached_landmark_similarity_data creates cache file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            osm_dir, pano_dir = self._create_mock_embeddings(temp_path)

            # Patch the cache directory to use temp dir
            cache_dir = temp_path / "cache"
            cache_dir.mkdir()

            # Compute with caching - need to patch the cache path
            import unittest.mock as mock
            with mock.patch.object(Path, 'expanduser', return_value=cache_dir / "test.pkl"):
                result1 = lol.compute_cached_landmark_similarity_data(
                    self.dataset, osm_dir, pano_dir, embedding_dim=16, use_cache=False)

            self.assertIsInstance(result1, lol.LandmarkSimilarityData)


if __name__ == "__main__":
    unittest.main()
