
import unittest

import common.torch.load_torch_deps
import torch
import pandas as pd
import geopandas as gpd
import shapely

import experimental.overhead_matching.swag.evaluation.landmark_observation_likelihood as lol


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


class LandmarkObservationLikelihoodTest(unittest.TestCase):
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
            particle_locs_deg=torch.tensor([
                [37.7749, -122.4194],
                [40.7128, -74.0060],
                [51.5074, -0.1278],
            ])
        )

        config = lol.LandmarkObservationLikelihoodConfig()

        sat_log_likelihood, osm_log_likelihood = lol.compute_log_observation_likelihood(
            prior_data, query, config
        )

        self.assertEqual(sat_log_likelihood.shape, (3,))
        self.assertEqual(osm_log_likelihood.shape, (3,))

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
        config = lol.LandmarkObservationLikelihoodConfig()

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
        config = lol.LandmarkObservationLikelihoodConfig()

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
        config = lol.LandmarkObservationLikelihoodConfig()

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
            particle_locs_deg=torch.tensor([
                [37.7749, -122.4194],
                [37.7750, -122.4195],
                [37.7751, -122.4196],
                [37.7752, -122.4197],
                [37.7753, -122.4198],
            ])
        )
        config = lol.LandmarkObservationLikelihoodConfig()

        sat_ll, osm_ll = lol.compute_log_observation_likelihood(prior_data, query, config)

        self.assertEqual(sat_ll.shape, (5,))
        self.assertEqual(osm_ll.shape, (5,))

    def test_multidimensional_particles_output_shape(self):
        """Test with 2D grid of particles."""
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

        base_lat, base_lon = 37.7749, -122.4194
        query = lol.QueryData(
            pano_ids=['pano_0'],
            particle_locs_deg=torch.tensor([
                [[base_lat + 0.001*i, base_lon + 0.001*j] for j in range(4)]
                for i in range(3)
            ])
        )
        config = lol.LandmarkObservationLikelihoodConfig()

        sat_ll, osm_ll = lol.compute_log_observation_likelihood(prior_data, query, config)

        self.assertEqual(sat_ll.shape, (3, 4))
        self.assertEqual(osm_ll.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
