import unittest
import common.torch.load_torch_deps
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.evaluate_swag import evaluate_prediction_top_k, get_distance_error_between_pano_and_particles_meters, PathInferenceResult

from common.math.haversine import find_d_on_unit_circle
import torch.nn as nn
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
import enum


class ModelType(enum.Enum):
    SATELLITE = enum.auto()
    PANORAMA = enum.auto()


class MockEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim, model_type):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_network = nn.LazyLinear(self.embedding_dim)
        self.model_type = model_type

    def model_input_from_batch(self, x):
        if self.model_type == ModelType.SATELLITE:
            return x.satellite
        else:
            return x.panorama

    def forward(self, data: torch.Tensor):
        batch_size = data.shape[0]
        out = self.embedding_network(data[:, :, :100, :100].float().reshape(batch_size, -1))
        out = nn.functional.normalize(out, dim=1)
        return out


class EvaluateSwagTest(unittest.TestCase):

    def test_evaluate_prediction_top_k(self):
        # Setup
        EMBEDDING_DIM = 16
        SEED = 42
        config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            sample_mode=vd.SampleMode.POS_SEMIPOS)

        # Use same random seed for reproducibility
        torch.manual_seed(SEED)

        # Create dataset, model, and embedding database
        dataset = vd.VigorDataset(Path("external/vigor_snippet/vigor_snippet"), config)

        sat_model = MockEmbeddingModel(EMBEDDING_DIM, ModelType.SATELLITE)
        ego_model = MockEmbeddingModel(EMBEDDING_DIM, ModelType.PANORAMA)

        # Call the function to test
        result_df, all_similarities = evaluate_prediction_top_k(
                sat_model, ego_model, dataset, device='cpu', use_cached_similarity=False)
        self.assertEqual(len(result_df), len(dataset))

        # Create satellite embedding database
        BATCH_SIZE = 4
        overhead_view = dataset.get_sat_patch_view()
        sat_dataloader = vd.get_dataloader(overhead_view, batch_size=BATCH_SIZE, shuffle=False)
        satellite_embedding_database = sed.build_satellite_db(sat_model, sat_dataloader, device="cpu")

        # Create panorama dataloader for testing
        pano_dataloader = vd.get_dataloader(dataset.get_pano_view(), batch_size=BATCH_SIZE, shuffle=False)

        to_compare = []
        expected_similarities = []
        with torch.no_grad():
            for batch in pano_dataloader:
                # Get embeddings for panorama images
                batch_embedding = ego_model(batch.panorama)

                # Calculate cosine similarity with all satellite patches
                patch_cosine_similarity = sed.calculate_cos_similarity_against_database(
                    batch_embedding, satellite_embedding_database
                )
                expected_similarities.append(patch_cosine_similarity)
                rankings = torch.argsort(patch_cosine_similarity, dim=1, descending=True)

                for batch_idx, pano_metadata in enumerate(batch.panorama_metadata):
                    pano_idx = pano_metadata["index"]
                    for sat_idx in pano_metadata["positive_satellite_idxs"]:
                        k_value = torch.where(rankings[batch_idx] == sat_idx)[0].item()
                        to_compare.append((pano_idx, sat_idx, k_value))

                    for sat_idx in pano_metadata["semipositive_satellite_idxs"]:
                        k_value = torch.where(rankings[batch_idx] == sat_idx)[0].item()
                        to_compare.append((pano_idx, sat_idx, k_value))

        expected_similarities = torch.cat(expected_similarities)

        # Verify results
        self.assertEqual(len(result_df), len(to_compare))

        # Check that patch_cosine_similarity values match
        for pano_idx, sat_idx, expected_k_value in to_compare:
            mask = np.logical_and(
                result_df["pano_idx"] == pano_idx,
                result_df["sat_idx"] == sat_idx)
            item = result_df[mask]
            self.assertEqual(item.iloc[0]["k_value"], expected_k_value)

    def test_get_distance_error_meters(self):
        # Setup
        config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None, panorama_tensor_cache_info=None)
        dataset = vd.VigorDataset(Path("external/vigor_snippet/vigor_snippet"), config)


        true_latlong_multiple = torch.stack([
            torch.tensor([37.7739, -122.4312]),  # San Francisco
            torch.tensor([34.0522, -118.2437])   # Los Angeles
        ])
        original_get_positions = dataset.get_panorama_positions

        try:
            # Create a mock method for a single panorama
            def mock_get_panorama_positions(indices):
                return true_latlong_multiple

            dataset.get_panorama_positions = mock_get_panorama_positions

            # 2. Test multiple panorama indices case
            panorama_indices = [0, 1]
            particle_means = torch.tensor([
                [37.7749, -122.4194],  # SF estimate
                [34.0522, -118.2437]   # LA estimate
            ])
            multi_particles = particle_means.unsqueeze(1) + torch.zeros((2, 100, 2))
            # to avoid floating point errors (mean of particles does not exactly match, leading to differences)
            particle_means = multi_particles.mean(dim=1)
            expected_distances = torch.tensor(
                [vd.EARTH_RADIUS_M * find_d_on_unit_circle(particle_means[0], true_latlong_multiple[0]),
                 vd.EARTH_RADIUS_M * find_d_on_unit_circle(particle_means[1], true_latlong_multiple[1])]
            )
            distances, _ = get_distance_error_between_pano_and_particles_meters(
                    dataset, panorama_indices, multi_particles)
            self.assertTrue(torch.allclose(distances, expected_distances))

        finally:
            # Restore the original method
            dataset.get_panorama_positions = original_get_positions

    def test_path_inference_result(self):
        # setup
        PATH_LENGTH = 101
        STATE_DIM = 3
        NUM_PARTICLES = 75
        inf_result_none = PathInferenceResult(
            particle_history=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            particle_history_pre_move=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            log_particle_weights=torch.rand(PATH_LENGTH, NUM_PARTICLES),
            num_dual_particles=None
        )
        inf_result_0 = PathInferenceResult(
            particle_history=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            particle_history_pre_move=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            log_particle_weights=torch.rand(PATH_LENGTH, NUM_PARTICLES),
            num_dual_particles=0
        )
        inf_result_8 = PathInferenceResult(
            particle_history=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            particle_history_pre_move=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            log_particle_weights=torch.rand(PATH_LENGTH, NUM_PARTICLES),
            num_dual_particles=8
        )
        inf_result_way_too_many = PathInferenceResult(
            particle_history=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            particle_history_pre_move=torch.rand(PATH_LENGTH, NUM_PARTICLES, STATE_DIM),
            log_particle_weights=torch.rand(PATH_LENGTH, NUM_PARTICLES),
            num_dual_particles=10 * NUM_PARTICLES
        )

        # action + verification
        self.assertEqual(inf_result_0.get_dual_particle_history(), None)
        self.assertEqual(inf_result_none.get_dual_particle_history(), None)
        self.assertEqual(inf_result_0.get_dual_log_particle_weights(), None)
        self.assertEqual(inf_result_none.get_dual_log_particle_weights(), None)
        self.assertEqual(inf_result_0.get_dual_particle_history_pre_move(), None)
        self.assertEqual(inf_result_none.get_dual_particle_history_pre_move(), None)

        self.assertTrue(torch.equal(inf_result_8.get_dual_particle_history(), inf_result_8.particle_history[:, -8:]))
        self.assertTrue(torch.equal(inf_result_8.get_dual_particle_history_pre_move(), inf_result_8.particle_history_pre_move[:, -8:]))
        self.assertTrue(torch.equal(inf_result_8.get_dual_log_particle_weights(), inf_result_8.log_particle_weights[:, -8:]))

        self.assertRaises(AssertionError, inf_result_way_too_many.get_dual_log_particle_weights)
        self.assertRaises(AssertionError, inf_result_way_too_many.get_dual_particle_history)
        self.assertRaises(AssertionError, inf_result_way_too_many.get_dual_particle_history_pre_move)
if __name__ == "__main__":
    unittest.main()
