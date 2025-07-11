
import unittest

import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.model import swag_patch_embedding as spe


class SwagPatchEmbeddingTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        config = spe.SwagPatchEmbeddingConfig(
            feature_map_extractor_config=spe.DinoFeatureMapExtractorConfig(),
            semantic_token_extractor_config=spe.SemanticEmbeddingMatrixConfig(
                vocabulary=["a", "b", "c"],
                embedding_dim=8),
            position_embedding_config=spe.PlanarPositionEmbeddingConfig(
                min_scale=0.1, scale_step=2.0, embedding_dim=64),
            aggregation_config=spe.TransformerAggregatorConfig(),
            image_input_dim=(128, 256),
            output_dim=1024)

        model = spe.SwagPatchEmbedding(config)
        input_image = torch.zeros((2, 3, 140, 252))
        metadata = [
                {"web_mercator_y": 100.0,
                 "web_mercator_x": 200.0,
                 "original_size": (140, 252),
                 "landmarks": [
                    {"web_mercator_y": 95.0, "web_mercator_x": 210.0, "landmark_type": "a"},
                    {"web_mercator_y": 90.0, "web_mercator_x": 190.0, "landmark_type": "b"}]},
                {"web_mercator_y": 300.0,
                 "web_mercator_x": 400.0,
                 "original_size": (140, 252),
                 "landmarks": [
                    {"web_mercator_y": 275.0, "web_mercator_x": 425.0, "landmark_type": "c"},
                    {"web_mercator_y": 350.0, "web_mercator_x": 390.0, "landmark_type": "b"}]}]

        # Action
        model_input = spe.ModelInput(image=input_image, metadata=metadata)
        result = model(model_input)

        # Verification
        print(result)


if __name__ == "__main__":
    unittest.main()
