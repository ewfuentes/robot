import common.torch.load_torch_deps
import torch
import torch.nn as nn
import unittest
from pathlib import Path
from experimental.overhead_matching.swag.data import vigor_dataset
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed

class MockEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def model_input_from_batch(self, x):
        return x.satellite

    def forward(self, data: torch.Tensor):
        batch_size = data.shape[0]
        out = torch.rand((batch_size, self.embedding_dim))
        out[:, 0] = data[:, 0, 0, 0]
        return out 


class SatelliteEmbeddingDatabaseTest(unittest.TestCase):

    def test_build_satellite_embedding_database(self):
        EMBEDDING_DIM = 16
        BATCH_SIZE = 32
        SEED = 32
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.2,
        )
        dataset = vigor_dataset.VigorDataset(Path("external/vigor_snippet/vigor_snippet"), config)
        overhead_view = dataset.get_sat_patch_view()
        dataloader = vigor_dataset.get_dataloader(overhead_view, batch_size=BATCH_SIZE, shuffle=False, generator=torch.manual_seed(SEED))
        model = MockEmbeddingModel(EMBEDDING_DIM)

        # action
        database = sed.build_satellite_db(model, dataloader, device="cpu")

        pixels_in_order = []
        for item in overhead_view:
            pixels_in_order.append(item.satellite[0, 0, 0])
        pixels_in_order = torch.tensor(pixels_in_order, dtype=database.dtype)

        # verification
        self.assertEqual(database.shape, (dataset.num_satellite_patches, EMBEDDING_DIM))
        self.assertTrue(torch.allclose(database[:, 0], pixels_in_order))

    def test_cosine_similarity(self):
        EMBEDDING_DIM = 16
        DATABASE_SIZE = 10
        torch.manual_seed(0)

        # Create a mock embedding database
        embedding_database = torch.rand((DATABASE_SIZE, EMBEDDING_DIM))
        embedding_database = embedding_database / torch.norm(embedding_database, dim=1, keepdim=True)

        # Create a mock embedding vector
        embedding = torch.rand((1, EMBEDDING_DIM))
        embedding = embedding / torch.norm(embedding)

        # Calculate cosine similarity using the function
        similarity = sed.calculate_cos_similarity_against_database(embedding, embedding_database)

        # Verification
        self.assertEqual(similarity.shape, (1,DATABASE_SIZE))
        similarity = similarity.squeeze(0)
        for i in range(DATABASE_SIZE):
            expected_similarity = torch.dot(embedding.squeeze(0), embedding_database[i]) / (
            torch.norm(embedding) * torch.norm(embedding_database[i])
            )
            self.assertAlmostEqual(similarity[i].item(), expected_similarity.item(), places=6)

if __name__ == "__main__": 
    unittest.main()
