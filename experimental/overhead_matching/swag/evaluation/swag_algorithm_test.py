import unittest

import common.torch.load_torch_deps
import torch
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig


class WagObservationLikelihoodCalculatorTest(unittest.TestCase):
    def test_3d_particles_input(self):
        """Test that 3D particles input works correctly."""
        # Setup
        num_patches = 100
        num_particles = 50
        num_panoramas = 3
        state_dim = 2

        # Create mock data
        similarity_matrix = torch.rand(num_panoramas, num_patches)
        panorama_ids = [f"pano_{i}" for i in range(num_panoramas)]
        satellite_patch_locations = torch.rand(num_patches, state_dim)
        # 3D particles: each panorama has its own set of particles
        particles = torch.rand(num_panoramas, num_particles, state_dim)

        # Create a simple patch_index_from_particle function
        def patch_index_from_particle(particles):
            # Map particles to patch indices (simple modulo for testing)
            # Ensure indices are valid by clipping to [0, num_patches)
            indices = (particles[:, 0] * num_patches).long()
            return torch.clamp(indices, 0, num_patches - 1)

        # Create calculator
        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            sigma=0.1,
            device=torch.device("cpu")
        )

        # Compute likelihoods
        batch_likelihoods = calculator.compute_log_likelihoods(particles, panorama_ids)

        # Verify shape
        self.assertEqual(batch_likelihoods.shape, (num_panoramas, num_particles))

    def test_each_trajectory_uses_own_particles(self):
        """Test that each trajectory's particles are evaluated against that trajectory's observation."""
        # Setup
        num_patches = 100
        num_particles = 50
        num_panoramas = 3
        state_dim = 2

        # Create mock data
        similarity_matrix = torch.rand(num_panoramas, num_patches)
        panorama_ids = [f"pano_{i}" for i in range(num_panoramas)]
        satellite_patch_locations = torch.rand(num_patches, state_dim)
        # 3D particles: each panorama has its own set of particles
        particles = torch.rand(num_panoramas, num_particles, state_dim)

        def patch_index_from_particle(particles):
            indices = (particles[:, 0] * num_patches).long()
            return torch.clamp(indices, 0, num_patches - 1)

        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            sigma=0.1,
            device=torch.device("cpu")
        )

        # Compute likelihoods for all trajectories
        batch_likelihoods = calculator.compute_log_likelihoods(particles, panorama_ids)

        # Compute likelihoods one at a time and verify they match
        for i, pano_id in enumerate(panorama_ids):
            # Extract this trajectory's particles as 3D with batch size 1
            single_particles = particles[i:i+1]
            single_likelihoods = calculator.compute_log_likelihoods(single_particles, [pano_id])

            # Should match the corresponding row of batch_likelihoods
            self.assertTrue(
                torch.allclose(single_likelihoods[0], batch_likelihoods[i]),
                f"Trajectory {i} likelihoods don't match"
            )

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

        calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=similarity_matrix,
            panorama_ids=panorama_ids,
            satellite_patch_locations=satellite_patch_locations,
            patch_index_from_particle=patch_index_from_particle,
            sigma=0.1,
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
            sigma=wag_config.sigma_obs_prob_from_sim,
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
            sigma=wag_config.sigma_obs_prob_from_sim,
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
