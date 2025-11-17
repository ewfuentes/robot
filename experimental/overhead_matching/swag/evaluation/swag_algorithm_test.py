import unittest

import common.torch.load_torch_deps
import torch
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig


class WagObservationLikelihoodCalculatorTest(unittest.TestCase):
    def test_batch_computation_single_vs_multiple(self):
        """Test that batching multiple panoramas gives expected results."""
        # Setup
        num_patches = 100
        num_particles = 50
        num_panoramas = 3
        state_dim = 2

        # Create mock data
        similarity_matrix = torch.rand(num_panoramas, num_patches)
        panorama_ids = [f"pano_{i}" for i in range(num_panoramas)]
        satellite_patch_locations = torch.rand(num_patches, state_dim)
        particles = torch.rand(num_particles, state_dim)

        # Create a simple patch_index_from_particle function
        def patch_index_from_particle(particles):
            # Map particles to patch indices (simple modulo for testing)
            # Ensure indices are valid by clipping to [0, num_patches)
            indices = (particles[:, 0] * num_patches).long()
            return torch.clamp(indices, 0, num_patches - 1)

        wag_config = WagConfig()
        wag_config.sigma_obs_prob_from_sim = 0.1

        # Create calculator
        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            wag_config=wag_config,
            device=torch.device("cpu")
        )

        # Test 1: Compute likelihoods for single panorama
        single_likelihoods = []
        for pano_id in panorama_ids:
            likelihood = calculator.compute_log_likelihoods(particles, [pano_id])
            single_likelihoods.append(likelihood)
        single_likelihoods = torch.cat(single_likelihoods, dim=0)

        # Test 2: Compute likelihoods for all panoramas at once
        batch_likelihoods = calculator.compute_log_likelihoods(particles, panorama_ids)

        # Verify shapes
        self.assertEqual(single_likelihoods.shape, (num_panoramas, num_particles))
        self.assertEqual(batch_likelihoods.shape, (num_panoramas, num_particles))

        # Verify that batch computation gives same results as sequential
        if not torch.allclose(single_likelihoods, batch_likelihoods):
            print(f"single_likelihoods:\n{single_likelihoods}")
            print(f"batch_likelihoods:\n{batch_likelihoods}")
            print(f"difference:\n{single_likelihoods - batch_likelihoods}")
            print(f"max difference: {(single_likelihoods - batch_likelihoods).abs().max()}")
        self.assertTrue(torch.allclose(single_likelihoods, batch_likelihoods))

    def test_sample_from_observation_batch(self):
        """Test that sampling from multiple observations works."""
        # Setup
        num_patches = 100
        num_particles = 30
        num_panoramas = 2
        state_dim = 2

        # Create mock data
        similarity_matrix = torch.rand(num_panoramas, num_patches)
        panorama_ids = [f"pano_{i}" for i in range(num_panoramas)]
        satellite_patch_locations = torch.rand(num_patches, state_dim)

        def patch_index_from_particle(particles):
            indices = (particles[:, 0] * num_patches).long()
            return torch.clamp(indices, 0, num_patches - 1)

        wag_config = WagConfig()
        wag_config.sigma_obs_prob_from_sim = 0.1

        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            wag_config=wag_config,
            device=torch.device("cpu")
        )

        generator = torch.Generator().manual_seed(42)

        # Sample from multiple observations
        sampled_particles = calculator.sample_from_observation(
            num_particles, panorama_ids, generator
        )

        # Verify shape
        self.assertEqual(sampled_particles.shape, (num_panoramas, num_particles, state_dim))

    def test_measurement_wag_backward_compatibility(self):
        """Test that measurement_wag works with both single string and list of panorama IDs."""
        # Setup
        num_patches = 100
        num_particles = 50
        state_dim = 2

        # Use satellite_patch_locations on a regular grid to ensure patch_index_from_particle is consistent
        # Create a grid of 10x10 patches
        grid_size = 10
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        satellite_patch_locations = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        particles = torch.rand(num_particles, state_dim)
        similarity_matrix = torch.rand(2, num_patches)
        panorama_ids = ["pano_0", "pano_1"]

        def patch_index_from_particle(particles):
            # Map to grid indices
            x_idx = (particles[:, 0] * grid_size).long()
            y_idx = (particles[:, 1] * grid_size).long()
            x_idx = torch.clamp(x_idx, 0, grid_size - 1)
            y_idx = torch.clamp(y_idx, 0, grid_size - 1)
            return x_idx * grid_size + y_idx

        wag_config = WagConfig()
        wag_config.sigma_obs_prob_from_sim = 0.1
        wag_config.dual_mcl_frac = 0.0  # Disable dual MCL to avoid index issues in test
        wag_config.dual_mcl_belief_phantom_counts_frac = 0.01

        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            wag_config=wag_config,
            device=torch.device("cpu")
        )

        belief_weighting = sa.BeliefWeighting(
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            phantom_counts_frac=wag_config.dual_mcl_belief_phantom_counts_frac
        )

        generator = torch.Generator().manual_seed(42)

        # Test with single panorama ID (string)
        result_single = sa.measurement_wag(
            particles,
            calculator,
            belief_weighting,
            "pano_0",
            wag_config,
            generator
        )

        # Reset generator for fair comparison
        generator = torch.Generator().manual_seed(42)

        # Test with single panorama ID (list)
        result_list = sa.measurement_wag(
            particles,
            calculator,
            belief_weighting,
            ["pano_0"],
            wag_config,
            generator
        )

        # Verify both produce same result
        self.assertTrue(torch.allclose(result_single.resampled_particles, result_list.resampled_particles))

    def test_measurement_wag_multiple_panoramas(self):
        """Test that measurement_wag can handle multiple panorama IDs."""
        # Setup
        num_patches = 100
        num_particles = 50
        state_dim = 2

        # Use satellite_patch_locations on a regular grid
        grid_size = 10
        x = torch.linspace(0, 1, grid_size)
        y = torch.linspace(0, 1, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        satellite_patch_locations = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        particles = torch.rand(num_particles, state_dim)
        similarity_matrix = torch.rand(3, num_patches)
        panorama_ids = ["pano_0", "pano_1", "pano_2"]

        def patch_index_from_particle(particles):
            x_idx = (particles[:, 0] * grid_size).long()
            y_idx = (particles[:, 1] * grid_size).long()
            x_idx = torch.clamp(x_idx, 0, grid_size - 1)
            y_idx = torch.clamp(y_idx, 0, grid_size - 1)
            return x_idx * grid_size + y_idx

        wag_config = WagConfig()
        wag_config.sigma_obs_prob_from_sim = 0.1
        wag_config.dual_mcl_frac = 0.0  # Disable dual MCL to avoid index issues in test
        wag_config.dual_mcl_belief_phantom_counts_frac = 0.01

        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            wag_config=wag_config,
            device=torch.device("cpu")
        )

        belief_weighting = sa.BeliefWeighting(
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            phantom_counts_frac=wag_config.dual_mcl_belief_phantom_counts_frac
        )

        generator = torch.Generator().manual_seed(42)

        # Test with multiple panorama IDs
        result = sa.measurement_wag(
            particles,
            calculator,
            belief_weighting,
            ["pano_0", "pano_1", "pano_2"],
            wag_config,
            generator
        )

        # Verify result has expected shape
        self.assertEqual(result.resampled_particles.shape[0], num_particles)
        self.assertEqual(result.resampled_particles.shape[1], state_dim)


if __name__ == "__main__":
    unittest.main()
