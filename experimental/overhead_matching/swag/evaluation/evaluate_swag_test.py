import unittest
import common.torch.load_torch_deps
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset, get_dataloader
from experimental.overhead_matching.swag.evaluation.evaluate_swag import evaluate_prediction_top_k

import torch.nn as nn
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed


class MockEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_network = nn.LazyLinear(self.embedding_dim)
    
    def forward(self, data: torch.Tensor):
        batch_size = data.shape[0]
        out = self.embedding_network(data[:, :, :100, :100].float().view(batch_size, -1))
        out[1,:] = 0.5  # make sure some vectors are identical/linearly dependent 
        return out


class EvaluateSwagTest(unittest.TestCase):
    
    def test_evaluate_prediction_top_k(self):
        # Setup
        EMBEDDING_DIM = 16
        BATCH_SIZE = 4
        SEED = 42
        PANO_NEIGHBOR_RADIUS = 0.2
        
        # Use same random seed for reproducibility
        torch.manual_seed(SEED)
        
        # Create dataset, model, and embedding database
        dataset = VigorDataset(Path("external/vigor_snippet/vigor_snippet"), PANO_NEIGHBOR_RADIUS)
        overhead_view = dataset.get_sat_patch_view()
        
        sat_model = MockEmbeddingModel(EMBEDDING_DIM)
        ego_model = MockEmbeddingModel(EMBEDDING_DIM)
        
        # Create satellite embedding database
        sat_dataloader = get_dataloader(overhead_view, batch_size=BATCH_SIZE, shuffle=False)
        satellite_embedding_database = sed.build_satellite_embedding_database(sat_model, sat_dataloader, device="cpu")
        
        # Create panorama dataloader for testing
        pano_dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Call the function to test
        result_df = evaluate_prediction_top_k(
            satellite_embedding_database=satellite_embedding_database,
            dataloader=pano_dataloader,
            model=ego_model,
            device="cpu"
        )
        self.assertEqual(len(result_df), len(dataset))
        
        # Manually calculate expected results for verification
        expected_panorama_indices = []
        expected_patch_similarities = []
        expected_k_values = []
        
        with torch.no_grad():
            for batch in pano_dataloader:
                # Get embeddings for panorama images
                batch_embedding = ego_model(batch.panorama)
                
                # Calculate cosine similarity with all satellite patches
                patch_cosine_similarity = sed.calculate_cos_similarity_against_database(
                    batch_embedding, satellite_embedding_database
                )
                
                # Get indices and correct patches
                panorama_indices = [x['index'] for x in batch.panorama_metadata]
                correct_overhead_patch_indices = [x['satellite_idx'] for x in batch.panorama_metadata]
                
                # Calculate rankings
                rankings = torch.argsort(patch_cosine_similarity, dim=1, descending=True)
                
                # Find where the correct patches are in the rankings
                for i, correct_idx in enumerate(correct_overhead_patch_indices):
                    # Find position (k value) of correct patch in rankings
                    expected_k_values.append(rankings[i, correct_idx].item())
                    
                expected_panorama_indices.extend(panorama_indices)
                expected_patch_similarities.extend(patch_cosine_similarity.tolist())
        
        # Verify results
        self.assertEqual(len(result_df), len(expected_panorama_indices))
        self.assertListEqual(result_df.index.tolist(), expected_panorama_indices)
        
        # Check that patch_cosine_similarity values match
        for i, (expected, actual) in enumerate(zip(expected_patch_similarities, result_df['patch_cosine_similarity'])):
            self.assertTrue(np.allclose(expected, actual), f"Mismatch at index {i}")
        
        # Check k_values - the position of the correct patch in rankings
        self.assertListEqual(result_df['k_value'].tolist(), expected_k_values)


if __name__ == "__main__":
    unittest.main()