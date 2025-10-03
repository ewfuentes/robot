
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.model import swag_patch_embedding as spe


class SwagPatchEmbeddingTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                    {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result = model(model_input)

        # Verification
        self.assertEqual(result.shape[0], BATCH_DIM)
        self.assertEqual(result.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result.shape[2], config.output_dim)

    def test_input_token_generation(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                    {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        input_tokens, input_mask = model._get_input_tokens(model_input)

        # Verification
        self.assertEqual(input_tokens.shape[0], BATCH_DIM)
        self.assertEqual(input_tokens.shape[2], config.output_dim)
        self.assertTrue(torch.allclose(torch.linalg.norm(input_tokens, dim=2), torch.ones((1,))))  # normalized over embedding dimension
        self.assertEqual(input_mask.dtype, torch.bool)
        self.assertEqual(input_mask.shape, input_tokens.shape[:2])
    def test_extractor_config(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=None,
            semantic_token_extractor_config=spe.SemanticNullExtractorConfig(),
            extractor_config_by_name={
                "embedding_mat_1": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
                "embedding_mat_2": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=16),
            },
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                    {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result = model(model_input)

        # Verification
        self.assertEqual(result.shape[0], BATCH_DIM)
        self.assertEqual(result.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result.shape[2], config.output_dim)

    def test_none_for_legacy_configs(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=None,
            semantic_token_extractor_config=None,
            extractor_config_by_name={
                "dino_feature_map": spe.DinoFeatureMapExtractorConfig(),
                "embedding_mat_1": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
                "embedding_mat_2": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=16),
            },
            position_embedding_config=spe.SphericalPositionEmbeddingConfig(
                scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                    {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result = model(model_input)

        # Verification
        self.assertEqual(result.shape[0], BATCH_DIM)
        self.assertEqual(result.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result.shape[2], config.output_dim)

    def test_float_mask(self):
        # Setup
        BATCH_DIM = 23
        NUM_TOKENS = 124

        input_bool_mask = torch.rand((BATCH_DIM, NUM_TOKENS), generator=torch.manual_seed(42)) > 0.5

        # Action
        float_mask = spe.make_float_mask_from_bool_mask(input_bool_mask)

        # Verification
        self.assertEqual(float_mask.shape, input_bool_mask.shape)
        self.assertEqual(float_mask.dtype, torch.float32)
        self.assertTrue(torch.all(float_mask[input_bool_mask] == -torch.inf))
        self.assertTrue(torch.all(float_mask[~input_bool_mask] == 0.0))

    def test_multiple_embeddings(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 3  # Test with multiple embeddings
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            normalize_embeddings=True,
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                    {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result = model(model_input)

        # Verification
        self.assertEqual(result.shape[0], BATCH_DIM)
        self.assertEqual(result.shape[1], NUM_EMBEDDINGS)  # Should have multiple embeddings per sample
        self.assertEqual(result.shape[2], config.output_dim)

        # Test that each embedding vector is properly normalized
        norms = torch.linalg.norm(result, dim=2)
        expected_norms = torch.ones((BATCH_DIM, NUM_EMBEDDINGS))
        self.assertTrue(torch.allclose(norms, expected_norms, atol=1e-6))

        # Test that the model's num_embeddings property matches what we expect
        self.assertEqual(model.num_embeddings, NUM_EMBEDDINGS)

    def test_swag_with_compile(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config).cuda()
        compiled_model = torch.compile(model, mode="default", fullgraph=False)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS)).cuda()
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata, cached_tensors={})
        with torch.no_grad():
            result_compiled = compiled_model(model_input)

        # Verification
        self.assertEqual(result_compiled.shape[0], BATCH_DIM)
        self.assertEqual(result_compiled.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result_compiled.shape[2], config.output_dim)

    def test_swag_with_compile_multiple_extractors(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 1
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=None,
            semantic_token_extractor_config=spe.SemanticNullExtractorConfig(),
            extractor_config_by_name={
                "dino_feature_map": spe.DinoFeatureMapExtractorConfig(),
                "embedding_mat_1": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
            },
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config).cuda()
        compiled_model = torch.compile(model, mode="default", fullgraph=False)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS)).cuda()
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata, cached_tensors={})
        with torch.no_grad():
            result_compiled = compiled_model(model_input)

        # Verification
        self.assertEqual(result_compiled.shape[0], BATCH_DIM)
        self.assertEqual(result_compiled.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result_compiled.shape[2], config.output_dim)

    def test_swag_with_compile_multiple_embeddings(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
        NUM_EMBEDDINGS = 3  # Test with multiple embeddings
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=4,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            normalize_embeddings=True,
            output_dim=16,
            num_embeddings=NUM_EMBEDDINGS)

        model = spe.SwagPatchEmbedding(config).cuda()
        compiled_model = torch.compile(model, mode="default", fullgraph=False)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS)).cuda()
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_shape": (NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata, cached_tensors={})
        with torch.no_grad():
            result_compiled = compiled_model(model_input)

        # Verification
        self.assertEqual(result_compiled.shape[0], BATCH_DIM)
        self.assertEqual(result_compiled.shape[1], NUM_EMBEDDINGS)
        self.assertEqual(result_compiled.shape[2], config.output_dim)
        # Verify embeddings are normalized
        norms = torch.linalg.norm(result_compiled, dim=2)
        expected_norms = torch.ones((BATCH_DIM, NUM_EMBEDDINGS)).cuda()
        self.assertTrue(torch.allclose(norms, expected_norms, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
