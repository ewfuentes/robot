import unittest

import common.torch.load_torch_deps
import torch
import experimental.overhead_matching.swag.evaluation.combined_observation_likelihood as col
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa


class CombinedObservationLikelihoodCalculatorTest(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.num_patches = 100
        self.num_particles = 50
        self.num_panoramas = 3
        self.state_dim = 2

        # Create mock similarity matrices
        torch.manual_seed(42)
        self.sat_similarity_matrix = torch.rand(self.num_panoramas, self.num_patches)
        self.osm_similarity_matrix = torch.rand(self.num_panoramas, self.num_patches)
        self.panorama_ids = [f"pano_{i}" for i in range(self.num_panoramas)]
        self.satellite_patch_locations = torch.rand(self.num_patches, self.state_dim)
        self.particles = torch.rand(self.num_particles, self.state_dim)

        # Simple patch_index_from_particle function
        def patch_index_from_particle(particles):
            indices = (particles[:, 0] * self.num_patches).long()
            return torch.clamp(indices, 0, self.num_patches - 1)

        self.patch_index_from_particle = patch_index_from_particle
        self.device = torch.device("cpu")

    def test_sat_only_mode_shape(self):
        """Test that SAT_ONLY mode produces correct output shape."""
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.SAT_ONLY,
            sat_sigma=0.1,
        )
        calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        likelihoods = calculator.compute_log_likelihoods(self.particles, self.panorama_ids)

        self.assertEqual(likelihoods.shape, (self.num_panoramas, self.num_particles))

    def test_osm_only_mode_shape(self):
        """Test that OSM_ONLY mode produces correct output shape."""
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.OSM_ONLY,
            osm_sigma=0.1,
        )
        calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        likelihoods = calculator.compute_log_likelihoods(self.particles, self.panorama_ids)

        self.assertEqual(likelihoods.shape, (self.num_panoramas, self.num_particles))

    def test_combined_mode_shape(self):
        """Test that COMBINED mode produces correct output shape."""
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.COMBINED,
            sat_sigma=0.1,
            osm_sigma=0.1,
            sat_weight=1.0,
            osm_weight=1.0,
        )
        calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        likelihoods = calculator.compute_log_likelihoods(self.particles, self.panorama_ids)

        self.assertEqual(likelihoods.shape, (self.num_panoramas, self.num_particles))

    def test_sat_only_matches_wag_calculator(self):
        """Test that SAT_ONLY mode produces identical results to WagObservationLikelihoodCalculator."""
        sigma = 0.1

        # Create combined calculator in SAT_ONLY mode
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.SAT_ONLY,
            sat_sigma=sigma,
        )
        combined_calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        # Create WagObservationLikelihoodCalculator for comparison
        wag_calculator = sa.WagObservationLikelihoodCalculator(
            similarity_matrix=self.sat_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            sigma=sigma,
            device=self.device,
        )

        # Compute likelihoods from both
        combined_likelihoods = combined_calculator.compute_log_likelihoods(
            self.particles, self.panorama_ids
        )
        wag_likelihoods = wag_calculator.compute_log_likelihoods(
            self.particles, self.panorama_ids
        )

        # Verify they match
        self.assertTrue(
            torch.allclose(combined_likelihoods, wag_likelihoods),
            f"Combined SAT_ONLY differs from WagObservationLikelihoodCalculator.\n"
            f"Max diff: {(combined_likelihoods - wag_likelihoods).abs().max()}"
        )

    def test_combined_mode_is_weighted_sum(self):
        """Test that COMBINED mode correctly computes weighted sum of log-likelihoods."""
        sat_sigma = 0.1
        osm_sigma = 0.2
        sat_weight = 0.7
        osm_weight = 0.3

        # Create combined calculator
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.COMBINED,
            sat_sigma=sat_sigma,
            osm_sigma=osm_sigma,
            sat_weight=sat_weight,
            osm_weight=osm_weight,
        )
        combined_calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        # Create separate calculators for SAT and OSM
        sat_config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.SAT_ONLY,
            sat_sigma=sat_sigma,
        )
        sat_calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=sat_config,
            device=self.device,
        )

        osm_config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.OSM_ONLY,
            osm_sigma=osm_sigma,
        )
        osm_calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=osm_config,
            device=self.device,
        )

        # Compute likelihoods
        combined_ll = combined_calculator.compute_log_likelihoods(
            self.particles, self.panorama_ids
        )
        sat_ll = sat_calculator.compute_log_likelihoods(
            self.particles, self.panorama_ids
        )
        osm_ll = osm_calculator.compute_log_likelihoods(
            self.particles, self.panorama_ids
        )

        # Expected: sat_weight * sat_ll + osm_weight * osm_ll
        expected_ll = sat_weight * sat_ll + osm_weight * osm_ll

        # Verify
        self.assertTrue(
            torch.allclose(combined_ll, expected_ll),
            f"Combined log-likelihood doesn't match weighted sum.\n"
            f"Max diff: {(combined_ll - expected_ll).abs().max()}"
        )

    def test_different_sigma_values(self):
        """Test that different sigma values produce different results."""
        config_low_sigma = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.SAT_ONLY,
            sat_sigma=0.05,
        )
        config_high_sigma = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.SAT_ONLY,
            sat_sigma=0.5,
        )

        calc_low = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config_low_sigma,
            device=self.device,
        )

        calc_high = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config_high_sigma,
            device=self.device,
        )

        ll_low = calc_low.compute_log_likelihoods(self.particles, self.panorama_ids)
        ll_high = calc_high.compute_log_likelihoods(self.particles, self.panorama_ids)

        # They should NOT be equal
        self.assertFalse(
            torch.allclose(ll_low, ll_high),
            "Different sigma values should produce different results"
        )

    def test_sample_from_observation_raises_not_implemented(self):
        """Test that sample_from_observation raises NotImplementedError."""
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.COMBINED,
        )
        calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        generator = torch.Generator()
        with self.assertRaises(NotImplementedError):
            calculator.sample_from_observation(10, self.panorama_ids, generator)

    def test_mismatched_similarity_matrices_raises_error(self):
        """Test that mismatched similarity matrix shapes raise an error."""
        # Create mismatched OSM similarity matrix
        osm_similarity_wrong_shape = torch.rand(self.num_panoramas, self.num_patches + 10)

        config = col.CombinedObservationLikelihoodConfig()

        with self.assertRaises(AssertionError):
            col.CombinedObservationLikelihoodCalculator(
                sat_similarity_matrix=self.sat_similarity_matrix,
                osm_similarity_matrix=osm_similarity_wrong_shape,
                panorama_ids=self.panorama_ids,
                satellite_patch_locations=self.satellite_patch_locations,
                patch_index_from_particle=self.patch_index_from_particle,
                config=config,
                device=self.device,
            )

    def test_single_panorama_computation(self):
        """Test computation with a single panorama."""
        config = col.CombinedObservationLikelihoodConfig(
            mode=col.CombinedLikelihoodMode.COMBINED,
        )
        calculator = col.CombinedObservationLikelihoodCalculator(
            sat_similarity_matrix=self.sat_similarity_matrix,
            osm_similarity_matrix=self.osm_similarity_matrix,
            panorama_ids=self.panorama_ids,
            satellite_patch_locations=self.satellite_patch_locations,
            patch_index_from_particle=self.patch_index_from_particle,
            config=config,
            device=self.device,
        )

        # Compute for single panorama
        likelihoods = calculator.compute_log_likelihoods(self.particles, ["pano_0"])

        self.assertEqual(likelihoods.shape, (1, self.num_particles))


if __name__ == "__main__":
    unittest.main()
