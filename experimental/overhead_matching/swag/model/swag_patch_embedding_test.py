
import unittest

import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.model import swag_patch_embedding as spe


class SwagPatchEmbeddingTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        BATCH_DIM = 2
        NUM_IMAGE_ROWS = 28
        NUM_IMAGE_COLS = 42
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
            image_input_dim=(NUM_IMAGE_ROWS, NUM_IMAGE_COLS),
            output_dim=16)

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
        self.assertEqual(result.shape[1], config.output_dim)


if __name__ == "__main__":
    unittest.main()
