import unittest

from pathlib import Path

from functools import reduce
import operator

import common.torch.load_torch_deps
import torch
from experimental.overhead_matching.learned.model import (
    clevr_transformer,
    clevr_tokenizer,
)
from experimental.overhead_matching.learned.data import clevr_dataset


def project_to_ego(batch):
    return batch


def positions_from_scene_objects(scene_objects):
    batch_size = len(scene_objects)
    max_num_objects = max([len(x) for x in scene_objects])
    out = torch.zeros((batch_size, max_num_objects, 2))
    for scene_idx, scene in enumerate(scene_objects):
        for obj_idx, obj in enumerate(scene):
            out[scene_idx, obj_idx, :] = torch.tensor(obj["3d_coords"][:2])
    return out


class ClevrTransformerTest(unittest.TestCase):
    def test_predict_gaussian(self):
        # Setup
        dataset = clevr_dataset.ClevrDataset(
            Path("external/clevr_test_set/clevr_test_set")
        )
        BATCH_SIZE = 4
        loader = clevr_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        vocab = dataset.vocabulary()
        vocab_size = reduce(operator.mul, [len(v) for v in vocab.values()])

        MODEL_SIZE = 256
        OUTPUT_DIM = 4
        config = clevr_transformer.ClevrTransformerConfig(
            token_dim=MODEL_SIZE,
            vocabulary_size=vocab_size,
            num_encoder_heads=4,
            num_encoder_layers=4,
            num_decoder_heads=4,
            num_decoder_layers=4,
            output_dim=OUTPUT_DIM,
            inference_method=clevr_transformer.InferenceMethod.MEAN,
        )

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        scene_objects = batch.scene_description["objects"]

        overhead_result = clevr_tokenizer.create_scene_tokens(scene_objects, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(
            scene_objects, embedding_size=MODEL_SIZE
        )

        ego_batch = project_to_ego(scene_objects)
        ego_result = clevr_tokenizer.create_scene_tokens(scene_objects, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(
            ego_batch, embedding_size=MODEL_SIZE
        )

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=positions_from_scene_objects(scene_objects),
            overhead_position_embeddings=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=positions_from_scene_objects(ego_batch),
            ego_position_embeddings=ego_position,
            ego_mask=ego_result["mask"],
        )

        # Action
        output = model(input, None, None)

        self.assertIn('prediction', output)
        self.assertIn('mean', output)

        self.assertEqual(output["mean"].shape, (BATCH_SIZE, OUTPUT_DIM))

    def test_predict_histogram(self):
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
            inference_method=clevr_transformer.InferenceMethod.HISTOGRAM,
        )

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        scene_objects = batch.scene_description["objects"]

        overhead_result = clevr_tokenizer.create_scene_tokens(scene_objects, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(
            scene_objects, embedding_size=MODEL_SIZE
        )

        ego_scene_objects = project_to_ego(scene_objects)
        ego_result = clevr_tokenizer.create_scene_tokens(ego_scene_objects, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(
            ego_scene_objects, embedding_size=MODEL_SIZE
        )

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=positions_from_scene_objects(scene_objects),
            overhead_position_embeddings=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=positions_from_scene_objects(ego_scene_objects),
            ego_position_embeddings=ego_position,
            ego_mask=ego_result["mask"],
        )

        # Action
        NUM_QUERY_TOKENS = 100
        query_tokens = torch.randn((len(batch), NUM_QUERY_TOKENS, MODEL_SIZE))
        query_mask = torch.zeros((len(batch), NUM_QUERY_TOKENS), dtype=torch.bool)
        output = model(input, query_tokens, query_mask)

        self.assertIn('histogram', output)
        self.assertEqual(output["histogram"].shape, (BATCH_SIZE, NUM_QUERY_TOKENS, OUTPUT_DIM))

    def test_predict_correspondence(self):
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
            inference_method=clevr_transformer.InferenceMethod.LEARNED_CORRESPONDENCE,
        )

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        scene_objects = batch.scene_description["objects"]

        overhead_result = clevr_tokenizer.create_scene_tokens(scene_objects, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(
            scene_objects, embedding_size=MODEL_SIZE
        )

        ego_scene_objects = project_to_ego(scene_objects)
        ego_result = clevr_tokenizer.create_scene_tokens(ego_scene_objects, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(
            ego_scene_objects, embedding_size=MODEL_SIZE
        )

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=positions_from_scene_objects(scene_objects),
            overhead_position_embeddings=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=positions_from_scene_objects(ego_scene_objects),
            ego_position_embeddings=ego_position,
            ego_mask=ego_result["mask"],
        )

        # Action
        model.train()
        train_output = model(input, None, None)
        model.eval()
        eval_output = model(input, None, None)

        self.assertIn('learned_correspondence', train_output)
        self.assertIn('learned_correspondence', eval_output)
        self.assertNotIn('prediction', train_output)
        self.assertIn('prediction', eval_output)

    def test_optimized_pose(self):
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
            inference_method=clevr_transformer.InferenceMethod.OPTIMIZED_POSE,
        )

        model = clevr_transformer.ClevrTransformer(config)
        batch = next(iter(loader))
        scene_objects = batch.scene_description["objects"]

        overhead_result = clevr_tokenizer.create_scene_tokens(scene_objects, vocab)
        overhead_position = clevr_tokenizer.create_position_embeddings(
            scene_objects, embedding_size=MODEL_SIZE
        )

        ego_scene_objects = project_to_ego(scene_objects)
        ego_result = clevr_tokenizer.create_scene_tokens(ego_scene_objects, vocab)
        ego_position = clevr_tokenizer.create_position_embeddings(
            ego_scene_objects, embedding_size=MODEL_SIZE
        )

        input = clevr_transformer.ClevrInputTokens(
            overhead_tokens=overhead_result["tokens"],
            overhead_position=positions_from_scene_objects(scene_objects),
            overhead_position_embeddings=overhead_position,
            overhead_mask=overhead_result["mask"],
            ego_tokens=ego_result["tokens"],
            ego_position=positions_from_scene_objects(ego_scene_objects),
            ego_position_embeddings=ego_position,
            ego_mask=ego_result["mask"],
        )

        # Action
        output = model(input, None, None)

        self.assertIn('learned_correspondence', output)
        self.assertIn('prediction', output)

if __name__ == "__main__":
    unittest.main()
