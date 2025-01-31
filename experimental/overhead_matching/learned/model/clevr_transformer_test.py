import unittest

from pathlib import Path

from functools import reduce
import operator

import common.torch as torch
from experimental.overhead_matching.learned.model import clevr_transformer, clevr_tokenizer
from experimental.overhead_matching.learned.data import clevr_dataset

def project_to_ego(batch, world_from_ego):
    ...
    return batch

class ClevrTransformerTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup

        dataset = clevr_dataset.ClevrDataset(Path('external/clevr_test_set/clevr_test_set'))
        loader = clevr_dataset.get_dataloader(dataset, batch_size=4)

        vocab = dataset.vocabulary()
        vocab_size = reduce(operator.mul, [len(v) for v in vocab.values()])

        MODEL_SIZE = 128
        config = clevr_transformer.ClevrTransformerConfig(
            token_dim=MODEL_SIZE,
            vocabulary_size=vocab_size,
            num_encoder_heads=8,
            num_encoder_layers=4,
            num_decoder_heads=8,
            num_decoder_layers=4)

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        batch = batch["objects"]

        overhead_result = clevr_tokenizer.create_tokens(batch, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(batch, embedding_size=MODEL_SIZE)

        world_from_ego = None
        ego_batch = project_to_ego(batch, world_from_ego)
        ego_result = clevr_tokenizer.create_tokens(batch, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(ego_batch, embedding_size=MODEL_SIZE)

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=ego_position,
            ego_mask=ego_result["mask"])

        # Action
        NUM_QUERY_TOKENS = 100
        query_tokens = torch.randn((len(batch), NUM_QUERY_TOKENS, MODEL_SIZE))
        query_mask = torch.zeros((len(batch), NUM_QUERY_TOKENS), dtype=torch.bool)
        model(input, query_tokens, query_mask)

        # Verification
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
