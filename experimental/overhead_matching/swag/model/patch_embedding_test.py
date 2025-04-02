
import unittest

import common.torch.load_torch_deps
import torch
from experimental.overhead_matching.swag.model import patch_embedding as pe


class PatchEmbeddingTest(unittest.TestCase):
    def test_cnn_patch_embedding(self):
        # Setup
        PATCH_DIMS = (112, 616)
        NUM_HEADS = 4
        # VGG19 produces feature maps with 512 dimensions
        CHANNEL_DIM = 512
        BATCH = 25
        CHANNELS = 3
        config = pe.WagPatchEmbeddingConfig(patch_dims=PATCH_DIMS, num_aggregation_heads=NUM_HEADS)
        model = pe.WagPatchEmbedding(config)
        input = torch.rand((BATCH, CHANNELS, *PATCH_DIMS))

        # Action
        embedding = model(input)

        # Verification
        self.assertEqual(embedding.shape[0], BATCH)
        self.assertEqual(embedding.shape[1], NUM_HEADS * CHANNEL_DIM)


if __name__ == "__main__":
    unittest.main()
