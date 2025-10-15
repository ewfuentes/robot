
import unittest
import tempfile
import json
import math

import common.torch.load_torch_deps
import torch
from pathlib import Path

import experimental.overhead_matching.swag.model.panorama_semantic_landmark_extractor as psle
from experimental.overhead_matching.swag.model.swag_config_types import PanoramaSemanticLandmarkExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput


def create_embedding_response(custom_id, embedding):
    """Create an OpenAI batch API response in the expected format"""
    return {
        "id": f"batch_req_{custom_id[:16]}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": "test_request",
            "body": {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding
                }],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }
        },
        "error": None
    }


def create_sentence_response(pano_id, landmarks):
    """Create a sentence response with multiple landmarks for a panorama"""
    return {
        "id": f"batch_req_{pano_id[:16]}",
        "custom_id": pano_id,
        "response": {
            "status_code": 200,
            "request_id": "test_request",
            "body": {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-5-2025-08-07",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"landmarks": landmarks}),
                        "refusal": None,
                        "annotations": []
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
            }
        },
        "error": None
    }


class PanoramaSemanticLandmarkExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory structure
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)

        # Create test data for two cities
        version = "test_v1"
        cities = ["Chicago", "Seattle"]

        cls.test_panoramas = {}

        for city in cities:
            city_dir = base_path / version / city
            embedding_dir = city_dir / "embeddings"
            sentence_dir = city_dir / "sentences"
            metadata_dir = city_dir / "embedding_requests"

            embedding_dir.mkdir(parents=True, exist_ok=True)
            sentence_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Create test panoramas
            for pano_idx in range(2):
                pano_id = f"pano_{city}_{pano_idx},41.85,-87.65,"

                # Create 3 landmarks per panorama
                landmarks = []
                embeddings = []
                metadata_entries = []

                for lm_idx in range(3):
                    custom_id = f"{pano_id}__landmark_{lm_idx}"
                    # Create unique embedding for each landmark
                    embedding = [0.1 * (pano_idx + lm_idx + 1)] * 1536
                    yaw_angles = [0, 90] if lm_idx == 0 else ([180] if lm_idx == 1 else [270])

                    landmarks.append({
                        "description": f"{city} landmark {lm_idx}",
                        "yaw_angles": yaw_angles
                    })
                    embeddings.append(create_embedding_response(custom_id, embedding))
                    metadata_entries.append({
                        "custom_id": custom_id,
                        "panorama_id": pano_id,
                        "landmark_idx": lm_idx,
                        "yaw_angles": yaw_angles
                    })

                # Write embeddings
                with open(embedding_dir / f"pano_{pano_idx}.jsonl", "w") as f:
                    for emb in embeddings:
                        f.write(json.dumps(emb) + "\n")

                # Write sentences
                with open(sentence_dir / f"pano_{pano_idx}.jsonl", "w") as f:
                    f.write(json.dumps(create_sentence_response(pano_id, landmarks)) + "\n")

                # Write metadata
                with open(metadata_dir / "panorama_metadata.jsonl", "a") as f:
                    for meta in metadata_entries:
                        f.write(json.dumps(meta) + "\n")

                # Store for testing
                cls.test_panoramas[pano_id] = {
                    "city": city,
                    "landmarks": landmarks,
                    "custom_ids": [f"{pano_id}__landmark_{i}" for i in range(3)]
                }

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_file_loading(self):
        """Test that files are loaded correctly from multi-city structure"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.load_files()

        # Verify embeddings loaded
        self.assertGreater(len(model.all_embeddings), 0)
        # Each panorama has 3 landmarks, 2 panoramas per city, 2 cities = 12 total
        self.assertEqual(len(model.all_embeddings), 12)

        # Verify sentences loaded
        self.assertGreater(len(model.all_sentences), 0)
        self.assertEqual(len(model.all_sentences), 12)

        # Verify metadata loaded
        self.assertGreater(len(model.panorama_metadata), 0)

    def test_forward_basic(self):
        """Test basic forward pass with panorama data"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create test input
        batch_size = 2
        pano_ids = list(self.test_panoramas.keys())[:batch_size]

        model_input = ModelInput(
            image=torch.zeros((batch_size, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in pano_ids]
        )

        # Forward pass
        output = model(model_input)

        # Verify output shape
        self.assertEqual(output.features.shape[0], batch_size)
        self.assertEqual(output.features.shape[2], 1536)
        self.assertEqual(output.mask.shape[0], batch_size)
        self.assertEqual(output.positions.shape[0], batch_size)
        self.assertEqual(output.positions.shape[2], 2)  # min/max bounds
        self.assertEqual(output.positions.shape[3], 2)  # [vertical, horizontal]

        # Verify at least some landmarks are not masked
        self.assertFalse(output.mask.all())

    def test_angular_position_computation(self):
        """Test that yaw angles are correctly converted to angular positions"""
        # Test single angle
        angle1, angle2 = psle.yaw_angles_to_radians([0])
        self.assertAlmostEqual(angle1, 0.0, places=5)
        self.assertAlmostEqual(angle2, 0.0, places=5)

        # Test continuous range: [0, 90] (adjacent, continuous)
        angle1, angle2 = psle.yaw_angles_to_radians([0, 90])
        self.assertAlmostEqual(angle1, 0.0, places=5)
        self.assertAlmostEqual(angle2, math.pi / 2, places=5)

        # Test continuous wrap-around: [270, 0] = [-pi/2, 0] (adjacent across boundary)
        angle1, angle2 = psle.yaw_angles_to_radians([270, 0])
        self.assertAlmostEqual(angle1, -math.pi / 2, places=5)
        self.assertAlmostEqual(angle2, 0.0, places=5)

        # Test discontinuous range: [0, 180] (opposite sides, not adjacent)
        # Should return only the first angle
        angle1, angle2 = psle.yaw_angles_to_radians([0, 180])
        self.assertAlmostEqual(angle1, 0.0, places=5)
        self.assertAlmostEqual(angle2, 0.0, places=5)  # Same as first

        # Test discontinuous range: [90, 270] (opposite sides)
        # Should return only the first angle
        angle1, angle2 = psle.yaw_angles_to_radians([90, 270])
        self.assertAlmostEqual(angle1, math.pi / 2, places=5)
        self.assertAlmostEqual(angle2, math.pi / 2, places=5)  # Same as first

    def test_error_on_satellite_data(self):
        """Test that extractor raises error when given satellite data"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create satellite input (no pano_id)
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 256)),
            metadata=[{"lat": 41.85, "lon": -87.65}]  # Missing pano_id
        )

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            model(model_input)

        self.assertIn("pano_id", str(context.exception))
        self.assertIn("panorama", str(context.exception).lower())

    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        pano_id = list(self.test_panoramas.keys())[0]
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}]
        )

        output = model(model_input)

        # Check that unmasked embeddings have unit norm
        for i in range(output.features.shape[1]):
            if not output.mask[0, i]:
                norm = torch.norm(output.features[0, i])
                self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_embedding_size_cropping(self):
        """Test that embeddings can be cropped to smaller size"""
        smaller_size = 512
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=smaller_size,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        pano_id = list(self.test_panoramas.keys())[0]
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}]
        )

        output = model(model_input)

        # Verify output dimension
        self.assertEqual(output.features.shape[2], smaller_size)
        self.assertEqual(model.output_dim, smaller_size)

        # Verify still normalized
        for i in range(output.features.shape[1]):
            if not output.mask[0, i]:
                norm = torch.norm(output.features[0, i])
                self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_data_requirements(self):
        """Test that data_requirements is empty (no landmark data needed from dataset)"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Should be empty since panorama landmarks are stored separately
        self.assertEqual(model.data_requirements, [])

    def test_position_output_format(self):
        """Test that positions are output in angular format suitable for SphericalPositionEmbedding"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        pano_id = list(self.test_panoramas.keys())[0]
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}]
        )

        output = model(model_input)

        # Check first unmasked landmark
        for i in range(output.features.shape[1]):
            if not output.mask[0, i]:
                # Positions should be [batch, num_landmarks, 2, 2]
                # dim 2: min/max bounds
                # dim 3: [horizontal_angle, horizontal_angle] (no vertical component)
                pos_bound1 = output.positions[0, i, 0, :]
                pos_bound2 = output.positions[0, i, 1, :]

                # Both elements in dim 3 should be the same (horizontal angle only)
                self.assertAlmostEqual(pos_bound1[0].item(), pos_bound1[1].item(), places=5)
                self.assertAlmostEqual(pos_bound2[0].item(), pos_bound2[1].item(), places=5)

                # Horizontal angles should be in [-π, π]
                self.assertGreaterEqual(pos_bound1[0].item(), -math.pi - 0.01)
                self.assertLessEqual(pos_bound1[0].item(), math.pi + 0.01)
                self.assertGreaterEqual(pos_bound2[0].item(), -math.pi - 0.01)
                self.assertLessEqual(pos_bound2[0].item(), math.pi + 0.01)

                break


if __name__ == "__main__":
    unittest.main()
