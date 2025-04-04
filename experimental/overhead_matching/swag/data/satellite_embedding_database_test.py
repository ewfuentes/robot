import common.torch.load_torch_deps
import torch
import torch.nn as nn
import unittest
from pathlib import Path
from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed

class MockEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    def forward(self, data: torch.Tensor):
        batch_size = data.shape[0]
        return torch.rand((batch_size, self.embedding_dim))


class SatelliteEmbeddingDatabaseTest(unittest.TestCase):

    def test_build_satellite_embedding_database(self):
        PANO_NEIGHBOR_RADIUS = 0.2
        EMBEDDING_DIM = 16
        dataset = VigorDataset(Path("external/vigor_snippet/vigor_snippet"), PANO_NEIGHBOR_RADIUS)
        model = MockEmbeddingModel(EMBEDDING_DIM)
        database = sed.build_satellite_embedding_database(model, dataset, device="cpu")

        self.assertEqual(database.shape, (dataset.num_satellite_patches, EMBEDDING_DIM))

if __name__ == "__main__": 
    unittest.main()