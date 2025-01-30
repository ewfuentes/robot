import unittest
import itertools
from collections import defaultdict
import numpy as np
from pathlib import Path

from experimental.overhead_matching.learned.model import clevr_tokenizer
from experimental.overhead_matching.learned.data import clevr_dataset

vocabulary = {
    "color": ["red", "green", "blue"],
    "shape": ["cube", "disk", "prism", "sphere"],
    "size": ["tiny", "small", "medium", "large", "enormous"],
}


class ClevrTokenizerTest(unittest.TestCase):

    def test_uniqueness(self):
        # Setup
        items = itertools.product(*vocabulary.values())
        batch = [[{"color": c, "shape": sh, "size": si} for c, sh, si in items]]

        # Action
        result = clevr_tokenizer.create_tokens(batch, vocabulary)

        # Verification
        self.assertEqual(result["tokens"].unique().numel(), result["tokens"].numel())

    def test_tokenizer(self):
        # Setup
        batch = [
            [("small", "red", "disk"), ("medium", "blue", "cube")],
            [("large", "green", "sphere"), ("medium", "red", "prism")],
            [
                ("small", "red", "disk"),
                ("medium", "blue", "cube"),
                ("large", "green", "sphere"),
            ],
        ]
        batch = [
            [{"color": c, "shape": sh, "size": si} for si, c, sh in scene]
            for scene in batch
        ]

        # Action
        result = clevr_tokenizer.create_tokens(batch, vocabulary)

        tokens = result["tokens"]
        mask = result["mask"]

        # Verification
        self.assertEqual(tokens.shape, (3, 3))
        self.assertEqual(tokens.shape, mask.shape)

        # Check that the entries that are beyond the number of items in a given
        # scene are marked invalid
        for i, j in itertools.product(range(3), range(3)):
            self.assertEqual(mask[i, j], j >= len(batch[i]))

        # Check that the same item in different scenes have the same token id
        self.assertEqual(tokens[2, 0], tokens[0, 0])
        self.assertEqual(tokens[2, 1], tokens[0, 1])
        self.assertEqual(tokens[2, 2], tokens[1, 0])

    def test_position_encoder(self):
        # Setup
        batch = [
            [(0.0, 0.0)],
            [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)],
        ]
        batch = [[{"3d_coords": v} for v in scene] for scene in batch]

        # Action
        result = clevr_tokenizer.create_position_embeddings(batch, embedding_size=32)

        # Verification
        self.assertEqual(result.shape, (2, 9, 32))
        self.assertAlmostEqual(result[0, 0, 0::2].abs().sum().item(), 0.0, places=3)
        self.assertAlmostEqual(result[0, 0, 1::2].abs().sum().item(), 16.0, places=3)

    def test_with_dataset(self):
        # Setup
        BATCH_SIZE = 4
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set")
        )
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)
        batch = next(iter(loader))
        vocabulary = dataset.vocabulary()

        token_result = clevr_tokenizer.create_tokens(batch["objects"], vocabulary)
        position_embedding = clevr_tokenizer.create_position_embeddings(
            batch["objects"], embedding_size=32
        )

        self.assertEqual(token_result["tokens"].shape, token_result["mask"].shape)
        self.assertEqual(token_result["tokens"].shape, position_embedding.shape[:-1])


if __name__ == "__main__":
    unittest.main()
