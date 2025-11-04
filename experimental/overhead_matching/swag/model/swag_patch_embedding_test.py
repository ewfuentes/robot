
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.model import swag_patch_embedding as spe
from experimental.overhead_matching.swag.model.landmark_scheduler import LandmarkDropoutScheduler
from experimental.overhead_matching.swag.model.swag_config_types import LandmarkDropoutSchedule


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
        result, _ = model(model_input)

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
        input_tokens, input_mask, _ = model._get_input_tokens(model_input)

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
        result, _ = model(model_input)

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
        result, _ = model(model_input)

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
        result, _ = model(model_input)

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

    def test_null_position_embedding(self):
        # Setup
        config = spe.NullPositionEmbeddingConfig()
        model = spe.NullPositionEmbedding(config)

        batch_size = 2
        num_tokens = 5
        num_positions = 1
        image = torch.zeros((batch_size, 3, 28, 28))
        metadata = [
            {"web_mercator_y": 100.0, "web_mercator_x": 200.0, "landmarks": []},
            {"web_mercator_y": 300.0, "web_mercator_x": 400.0, "landmarks": []}
        ]
        model_input = spe.ModelInput(image=image, metadata=metadata)
        relative_positions = torch.randn((batch_size, num_tokens, num_positions, 2))

        # Action
        result = model(model_input=model_input, relative_positions=relative_positions)

        # Verification
        self.assertEqual(result.shape, (batch_size, num_tokens, 0))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(model.output_dim, 0)
        # Verify all values are zero (even though dimension is 0)
        self.assertEqual(result.numel(), 0)

    def test_panorama_landmark_dropout_deterministic(self):
        """Test that dropout is deterministic based on pano_id."""
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 320
        NUM_IMAGE_COLS = 640
        config = spe.SwagPatchEmbeddingConfig(
            extractor_config_by_name={
                "test_extractor": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
            },
            position_embedding_config=spe.NullPositionEmbeddingConfig(),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=2,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=1)

        # Create scheduler with dropout configuration
        dropout_scheduler = LandmarkDropoutScheduler(
            schedules=[
                LandmarkDropoutSchedule(
                    start_progress=0.0,
                    end_progress=1.0,
                    initial_dropout_rate=0.5,
                    final_dropout_rate=0.5,
                    extractor_names=["test_extractor"],
                    min_landmarks=2
                )
            ],
            total_epochs=10)
        dropout_scheduler.set_epoch(0)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))

        # Create panorama metadata (with pano_id)
        metadata = [
            {"pano_id": "pano_001",
             "web_mercator_y": 100.0,
             "web_mercator_x": 200.0,
             "landmarks": [
                {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "c"},
                {"web_mercator_y": 110.0, "web_mercator_x": 205.0, "landmark_type": "a"}]},
            {"pano_id": "pano_002",
             "web_mercator_y": 300.0,
             "web_mercator_x": 400.0,
             "landmarks": [
                {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}
        ]

        # Action - run twice with same input
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result1, outputs1 = model(model_input, dropout_scheduler)

        result2, outputs2 = model(model_input, dropout_scheduler)

        # Verification - masks should be identical across runs
        mask1 = outputs1["test_extractor"].mask
        mask2 = outputs2["test_extractor"].mask
        self.assertTrue(torch.equal(mask1, mask2), "Dropout should be deterministic")

        # Verify dropout happened (at least some landmarks masked)
        self.assertTrue(mask1[0].any() or mask1[1].any(),
                       "At least some landmarks should be dropped")

    def test_panorama_landmark_dropout_respects_minimum(self):
        """Test that dropout respects min_panorama_landmarks."""
        # Setup
        BATCH_DIM = 1
        NUM_IMAGE_ROWS = 320
        NUM_IMAGE_COLS = 640
        MIN_LANDMARKS = 3
        config = spe.SwagPatchEmbeddingConfig(
            extractor_config_by_name={
                "test_extractor": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
            },
            position_embedding_config=spe.NullPositionEmbeddingConfig(),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=2,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=1)

        # Create scheduler with dropout configuration
        dropout_scheduler = LandmarkDropoutScheduler(
            schedules=[
                LandmarkDropoutSchedule(
                    start_progress=0.0,
                    end_progress=1.0,
                    initial_dropout_rate=0.9,  # High dropout rate
                    final_dropout_rate=0.9,
                    extractor_names=["test_extractor"],
                    min_landmarks=MIN_LANDMARKS
                )
            ],
            total_epochs=10)
        dropout_scheduler.set_epoch(0)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))

        # Create metadata with more landmarks than minimum
        metadata = [
            {"pano_id": "pano_001",
             "web_mercator_y": 100.0,
             "web_mercator_x": 200.0,
             "landmarks": [
                {"web_mercator_y": float(95 + i), "web_mercator_x": float(210 + i), "landmark_type": "a"}
                for i in range(10)
             ]}
        ]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result, outputs = model(model_input, dropout_scheduler)

        # Verification - should keep at least MIN_LANDMARKS
        mask = outputs["test_extractor"].mask[0]
        num_kept = (~mask).sum().item()
        self.assertGreaterEqual(num_kept, MIN_LANDMARKS,
                               f"Should keep at least {MIN_LANDMARKS} landmarks")

    def test_panorama_landmark_dropout_satellite_unaffected(self):
        """Test that when no dropout scheduler is passed, landmarks are not affected.

        Note: With the new design, dropout schedules are passed to forward() calls.
        When no scheduler is passed, no dropout occurs.
        This test verifies that a model without a dropout scheduler works correctly.
        """
        # Setup - satellite uses square patch, NO dropout configured
        BATCH_DIM = 1
        NUM_IMAGE_SIZE = 320
        config = spe.SwagPatchEmbeddingConfig(
            extractor_config_by_name={
                "test_extractor": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a", "b", "c"],
                    embedding_dim=8),
            },
            position_embedding_config=spe.NullPositionEmbeddingConfig(),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=2,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_SIZE, NUM_IMAGE_SIZE),  # Square = satellite
            output_dim=16,
            num_embeddings=1)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_SIZE, NUM_IMAGE_SIZE))

        # Create satellite metadata (no pano_id)
        metadata = [
            {"web_mercator_y": 100.0,
             "web_mercator_x": 200.0,
             "landmarks": [
                {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"},
                {"web_mercator_y": 105.0, "web_mercator_x": 195.0, "landmark_type": "c"},
                {"web_mercator_y": 110.0, "web_mercator_x": 205.0, "landmark_type": "a"}]}
        ]

        # Action - no scheduler passed
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result, outputs = model(model_input)

        # Verification - no dropout should occur when no scheduler passed
        mask = outputs["test_extractor"].mask[0]
        num_kept = (~mask).sum().item()
        self.assertEqual(num_kept, 4, "All landmarks should be kept when no dropout scheduler passed")

    def test_panorama_landmark_dropout_percentage(self):
        """Test that dropout percentage is approximately correct."""
        # Setup
        BATCH_DIM = 1
        NUM_IMAGE_ROWS = 320
        NUM_IMAGE_COLS = 640
        DROPOUT_RATE = 0.3
        NUM_LANDMARKS = 100  # Large number for statistical testing

        config = spe.SwagPatchEmbeddingConfig(
            extractor_config_by_name={
                "test_extractor": spe.SemanticEmbeddingMatrixConfig(
                    vocabulary=["a"],
                    embedding_dim=8),
            },
            position_embedding_config=spe.NullPositionEmbeddingConfig(),
            aggregation_config=spe.TransformerAggregatorConfig(
                num_transformer_layers=2,
                num_attention_heads=4,
                hidden_dim=64,
                dropout_frac=0.1),
            patch_dims=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16,
            num_embeddings=1)

        # Create scheduler with dropout configuration
        dropout_scheduler = LandmarkDropoutScheduler(
            schedules=[
                LandmarkDropoutSchedule(
                    start_progress=0.0,
                    end_progress=1.0,
                    initial_dropout_rate=DROPOUT_RATE,
                    final_dropout_rate=DROPOUT_RATE,
                    extractor_names=["test_extractor"],
                    min_landmarks=1
                )
            ],
            total_epochs=10)
        dropout_scheduler.set_epoch(0)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((BATCH_DIM, 3, NUM_IMAGE_ROWS, NUM_IMAGE_COLS))

        # Create metadata with many landmarks
        metadata = [
            {"pano_id": "pano_001",
             "web_mercator_y": 100.0,
             "web_mercator_x": 200.0,
             "landmarks": [
                {"web_mercator_y": float(95 + i), "web_mercator_x": float(210 + i), "landmark_type": "a"}
                for i in range(NUM_LANDMARKS)
             ]}
        ]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result, outputs = model(model_input, dropout_scheduler)

        # Verification
        mask = outputs["test_extractor"].mask[0]
        num_kept = (~mask).sum().item()
        expected_kept = int(NUM_LANDMARKS * (1 - DROPOUT_RATE))

        # Allow some tolerance (within 5 landmarks)
        self.assertAlmostEqual(num_kept, expected_kept, delta=5,
                              msg=f"Expected ~{expected_kept} landmarks, got {num_kept}")


if __name__ == "__main__":
    unittest.main()
