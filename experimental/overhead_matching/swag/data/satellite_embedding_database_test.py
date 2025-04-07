import common.torch.load_torch_deps
import torch
import torch.nn as nn
import unittest
from pathlib import Path
from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset, get_dataloader
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed

class MockEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
    def forward(self, data: torch.Tensor):
        batch_size = data.shape[0]
        out = torch.rand((batch_size, self.embedding_dim))
        out[:, 0] = data[:, 0, 0, 0]
        return out 


class SatelliteEmbeddingDatabaseTest(unittest.TestCase):

    def test_build_satellite_embedding_database(self):
        PANO_NEIGHBOR_RADIUS = 0.2
        EMBEDDING_DIM = 16
        BATCH_SIZE = 32
        SEED = 32
        dataset = VigorDataset(Path("external/vigor_snippet/vigor_snippet"), PANO_NEIGHBOR_RADIUS)
        overhead_view = dataset.get_sat_patch_view()
        shuffle_dataloader = get_dataloader(overhead_view, batch_size=BATCH_SIZE, shuffle=True, generator=torch.manual_seed(SEED))
        model = MockEmbeddingModel(EMBEDDING_DIM)

        # action
        database = sed.build_satellite_embedding_database(model, shuffle_dataloader, device="cpu")

        pixels_in_order = []
        for item in overhead_view:
            pixels_in_order.append(item.satellite[0, 0, 0])
        pixels_in_order = torch.tensor(pixels_in_order, dtype=database.dtype)

        # verification
        self.assertEqual(database.shape, (dataset.num_satellite_patches, EMBEDDING_DIM))
        self.assertTrue(torch.allclose(database[:, 0], pixels_in_order))

if __name__ == "__main__": 
    unittest.main()