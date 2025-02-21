import unittest

from pathlib import Path

from functools import reduce
import operator

import common.torch as torch
from experimental.overhead_matching.learned.model import (
    clevr_transformer,
    clevr_tokenizer,
)
from experimental.overhead_matching.learned.data import clevr_dataset


def project_to_ego(batch):
    return batch


def positions_from_batch(batch):
    batch_size = len(batch)
    max_num_objects = max([len(x) for x in batch])
    out = torch.zeros((batch_size, max_num_objects, 2))
    for scene_idx, scene in enumerate(batch):
        for obj_idx, obj in enumerate(scene):
            out[scene_idx, obj_idx, :] = torch.tensor(obj["3d_coords"][:2])
    return out


class ClevrTransformerTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup

        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set")
        )
        BATCH_SIZE = 4
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        vocab = dataset.vocabulary()
        vocab_size = reduce(operator.mul, [len(v) for v in vocab.values()])

        MODEL_SIZE = 256
        OUTPUT_DIM = 384
        config = clevr_transformer.ClevrTransformerConfig(
            token_dim=MODEL_SIZE,
            vocabulary_size=vocab_size,
            num_encoder_heads=4,
            num_encoder_layers=4,
            num_decoder_heads=4,
            num_decoder_layers=4,
            output_dim=OUTPUT_DIM,
            predict_gaussian=False,
        )

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        batch = batch["objects"]

        overhead_result = clevr_tokenizer.create_tokens(batch, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(
            batch, embedding_size=MODEL_SIZE
        )

        ego_batch = project_to_ego(batch)
        ego_result = clevr_tokenizer.create_tokens(batch, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(
            ego_batch, embedding_size=MODEL_SIZE
        )
        torch.set_printoptions(linewidth=200, precision=3)

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=positions_from_batch(batch),
            overhead_position_embeddings=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=positions_from_batch(ego_batch),
            ego_position_embeddings=ego_position,
            ego_mask=ego_result["mask"],
        )

        # Action
        NUM_QUERY_TOKENS = 100
        query_tokens = torch.randn((len(batch), NUM_QUERY_TOKENS, MODEL_SIZE))
        query_mask = torch.zeros((len(batch), NUM_QUERY_TOKENS), dtype=torch.bool)
        output = model(input, query_tokens, query_mask)

        # Verification
        output_tokens = output["decoder_output"]
        correspondences = output["learned_correspondence"]
        self.assertEqual(output_tokens.shape, (BATCH_SIZE, NUM_QUERY_TOKENS, OUTPUT_DIM))

        NUM_OVERHEAD_TOKENS = input.overhead_tokens.shape[1]
        NUM_EGO_TOKENS = input.ego_tokens.shape[1]
        self.assertEqual(correspondences.shape,
                         (BATCH_SIZE, NUM_OVERHEAD_TOKENS, NUM_EGO_TOKENS + 1))

        # TODO Check that the no match correspondence is appropriately handled.


if __name__ == "__main__":
    unittest.main()
