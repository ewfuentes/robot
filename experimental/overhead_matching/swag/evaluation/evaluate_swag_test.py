import unittest
import common.torch.load_torch_deps
import torch
from pathlib import Path
import pandas as pd
import numpy as np
from experimental.overhead_matching.swag.data import vigor_dataset as vd
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
        out = self.embedding_network(data[:, :, :100, :100].float().reshape(batch_size, -1))
        out[1,:] = 0.5  # make sure some vectors are identical/linearly dependent 
        out = nn.functional.normalize(out, dim=1)
        return out


class EvaluateSwagTest(unittest.TestCase):

    def test_evaluate_prediction_top_k_manual(self):
        embedding_database = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.5, 0.5, 0.5],
            [-0.1, -0.2, -0.3],
            [0.77, 0.63, 0.99], # correct match
        ])
        # normalize 
        embedding_database = nn.functional.normalize(embedding_database, dim=1)
        mock_dataloader = [vd.VigorDatasetItem(
            panorama=torch.tensor([[1,2,3]]),
            panorama_metadata = [{"satellite_idx": 3, "index": 0}],
            satellite=None,
            satellite_metadata=None
        )]
        class MockModule(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return nn.functional.normalize(torch.tensor([[0.8, 0.8, 0.8]]))
        m = MockModule()
        
        # action 
        result_df = evaluate_prediction_top_k(embedding_database, mock_dataloader, m, device='cpu')

        # verification
        self.assertEqual(result_df.loc[0, 'k_value'], 1)
        self.assertTrue(np.allclose(result_df.loc[0, 'patch_cosine_similarity'], [0.9258200997725513, 1.0, -0.9258200997725513, 0.9831395864967746]))

    
    def test_evaluate_prediction_top_k(self):
        # Setup
        EMBEDDING_DIM = 16
        BATCH_SIZE = 4
        SEED = 42
        config = vd.VigorDatasetConfig(panorama_neighbor_radius=0.2)
        
        # Use same random seed for reproducibility
        torch.manual_seed(SEED)
        
        # Create dataset, model, and embedding database
        dataset = vd.VigorDataset(Path("external/vigor_snippet/vigor_snippet"), config)
        overhead_view = dataset.get_sat_patch_view()
        
        sat_model = MockEmbeddingModel(EMBEDDING_DIM)
        ego_model = MockEmbeddingModel(EMBEDDING_DIM)
        
        # Create satellite embedding database
        sat_dataloader = vd.get_dataloader(overhead_view, batch_size=BATCH_SIZE, shuffle=False)
        satellite_embedding_database = sed.build_satellite_db(sat_model, sat_dataloader, device="cpu")
        
        # Create panorama dataloader for testing
        pano_dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
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
                    expected_k_values.append(torch.argwhere(rankings[i] == correct_idx).item())
                    
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
