
import unittest
import tempfile
import json
import math
import hashlib
import base64

import common.torch.load_torch_deps
import torch
from pathlib import Path
import numpy as np

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


def create_random_embedding_vector(category: str):
    """Create a deterministic random vector from a category string (returns raw numpy array)"""
    bits = hashlib.sha512(category.encode('utf-8')).digest() * 3
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))
    scaled_bits = (2.0 * bits - 1.0).astype(dtype=np.float32)
    scaled_bits = scaled_bits / np.linalg.norm(scaled_bits)
    return scaled_bits


def random_vector(category: str):
    """Create a random(ish) vector with 1536 elements (returns base64 encoded string)"""
    scaled_bits = create_random_embedding_vector(category)
    return base64.b64encode(scaled_bits.data).decode('utf-8')


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

        # Define class assignments: city -> pano_idx -> [3 landmark classes]
        class_assignments = {
            "Chicago": {
                0: ['street', 'cat', 'car'],
                1: ['avenue', 'dog', 'truck']
            },
            "Seattle": {
                0: ['boulevard', 'deer', 'scooter'],
                1: ['street', 'cat', 'car']
            }
        }

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

                # Get the 3 classes for this panorama
                landmark_classes = class_assignments[city][pano_idx]

                # Create 3 landmarks per panorama
                landmarks = []
                embeddings = []
                metadata_entries = []

                for lm_idx in range(3):
                    custom_id = f"{pano_id}__landmark_{lm_idx}"

                    # Get assigned class for this landmark and generate matching embedding
                    assigned_class = landmark_classes[lm_idx]
                    embedding = create_random_embedding_vector(assigned_class).tolist()

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
                pano_id_no_coords = pano_id.split(",")[0]
                cls.test_panoramas[pano_id_no_coords] = {
                    "city": city,
                    "landmarks": landmarks,
                    "custom_ids": [f"{pano_id}__landmark_{i}" for i in range(3)],
                    "expected_classes": landmark_classes
                }

        # Create a semantic class grouping
        semantic_class_grouping_path = base_path / version / "semantic_class_grouping.json"

        semantic_class_grouping = {
            "semantic_groups": {
                "road": ['street', 'avenue', 'boulevard'],
                "animal": ["cat", "dog", "deer"],
                "vehicles": ["car", "truck", "scooter"],
                },
            "class_details": {
                "street": {
                    "osm_tags": {"paved": "yes", "direction": "east-west"},
                    "embedding": {'model': "random", "vector": random_vector("street")},
                    },
                "avenue": {
                    "osm_tags": {"paved": "yes", "direction": "north-south"},
                    "embedding": {'model': "random", "vector": random_vector("avenue")},
                    },
                "boulevard": {
                    "osm_tags": {"paved": "yes", "direction": "one"},
                    "embedding": {'model': "random", "vector": random_vector("boulevard")},
                    },
                "cat": {
                    "osm_tags": {"fuzzy": "yes", "friendly": "no"},
                    "embedding": {'model': "random", "vector": random_vector("cat")},
                    },
                "dog": {
                    "osm_tags": {"fuzzy": "yes", "friendly": "yes"},
                    "embedding": {'model': "random", "vector": random_vector("dog")},
                    },
                "deer": {
                    "osm_tags": {"fuzzy": "no", "friendly": "maybe"},
                    "embedding": {'model': "random", "vector": random_vector("deer")},
                    },
                "car": {
                    "osm_tags": {"wheels": "four", "size": "medium"},
                    "embedding": {'model': "random", "vector": random_vector("car")},
                    },
                "truck": {
                    "osm_tags": {"wheels": "eighteen", "size": "large"},
                    "embedding": {'model': "random", "vector": random_vector("truck")},
                    },
                "scooter": {
                    "osm_tags": {"wheels": "two", "size": "small"},
                    "embedding": {'model': "random", "vector": random_vector("scooter")},
                    },
                }
        }
        semantic_class_grouping_path.write_text(json.dumps(semantic_class_grouping))

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_file_loading(self):
        """Test that files are loaded correctly from multi-city structure"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key",)
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.load_files()

        # Each panorama has 3 landmarks, 2 panoramas per city, 2 cities = 12 total
        self.assertEqual(len(model.all_embeddings), 12)

        self.assertEqual(len(model.all_sentences), 12)

        # Verify metadata loaded
        self.assertGreater(len(model.panorama_metadata), 0)

        # Verify that the semantic groupings have been loaded
        self.assertIn('semantic_groups', model.semantic_groupings)
        self.assertIn('class_details', model.semantic_groupings)
        class_details = model.semantic_groupings["class_details"]
        for items in model.semantic_groupings["semantic_groups"].values():
            for item in items:
                self.assertIn(item, class_details)
                embedding_vector = class_details[item]["embedding"]["vector"]
                self.assertAlmostEqual(torch.linalg.norm(embedding_vector).item(), 1.0, places=5)

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
        self.assertEqual(model.output_dim, config.openai_embedding_size)
        self.assertEqual(output.features.shape[0], batch_size)
        self.assertEqual(output.features.shape[2], model.output_dim)
        self.assertEqual(output.mask.shape[0], batch_size)
        self.assertEqual(output.positions.shape[0], batch_size)
        self.assertEqual(output.positions.shape[2], 2)  # min/max bounds
        self.assertEqual(output.positions.shape[3], 2)  # [vertical, horizontal]

        # Verify at least some landmarks are not masked
        self.assertFalse(output.mask.all())

    def test_semantic_grouping(self):
        """Test basic forward pass with panorama data"""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v1",
            auxiliary_info_key="test_key",
            should_classify_against_grouping=True)
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
        self.assertEqual(model.output_dim, len(model.semantic_groupings["semantic_groups"]))
        self.assertEqual(output.features.shape[0], batch_size)
        self.assertEqual(output.features.shape[2], model.output_dim)
        self.assertEqual(output.mask.shape[0], batch_size)
        self.assertEqual(output.positions.shape[0], batch_size)
        self.assertEqual(output.positions.shape[2], 2)  # 2 position embeddings
        self.assertEqual(output.positions.shape[3], 2)  # 2 values per position

        # Verify at least some landmarks are not masked
        self.assertFalse(output.mask.all())

        # Verify classifications match expected classes
        group_names = list(model.semantic_groupings["semantic_groups"].keys())

        for batch_idx, pano_id in enumerate(pano_ids):
            expected_classes = self.test_panoramas[pano_id]["expected_classes"]

            for lm_idx in range(len(expected_classes)):
                # Get predicted group (argmax of feature vector)
                predicted_group_idx = torch.argmax(output.features[batch_idx, lm_idx]).item()
                predicted_group_name = group_names[predicted_group_idx]

                # Determine expected group from class assignment
                expected_class = expected_classes[lm_idx]
                expected_group_name = None
                for group_name, classes in model.semantic_groupings["semantic_groups"].items():
                    if expected_class in classes:
                        expected_group_name = group_name
                        break

                self.assertEqual(predicted_group_name, expected_group_name,
                                f"Landmark {lm_idx} (class '{expected_class}') should map to "
                                f"group '{expected_group_name}', but got '{predicted_group_name}'")

    def test_yaw_binary_vector_computation(self):
        """Test that yaw angles are correctly converted to binary presence vectors"""
        # Test single yaw angles
        vec = psle.yaw_angles_to_binary_vector([0])
        self.assertEqual(vec, [1.0, 0.0, 0.0, 0.0])

        vec = psle.yaw_angles_to_binary_vector([90])
        self.assertEqual(vec, [0.0, 1.0, 0.0, 0.0])

        vec = psle.yaw_angles_to_binary_vector([180])
        self.assertEqual(vec, [0.0, 0.0, 1.0, 0.0])

        vec = psle.yaw_angles_to_binary_vector([270])
        self.assertEqual(vec, [0.0, 0.0, 0.0, 1.0])

        # Test multiple yaw angles
        vec = psle.yaw_angles_to_binary_vector([0, 90])
        self.assertEqual(vec, [1.0, 1.0, 0.0, 0.0])

        vec = psle.yaw_angles_to_binary_vector([90, 270])
        self.assertEqual(vec, [0.0, 1.0, 0.0, 1.0])

        vec = psle.yaw_angles_to_binary_vector([0, 180])
        self.assertEqual(vec, [1.0, 0.0, 1.0, 0.0])

        # Test all yaws present
        vec = psle.yaw_angles_to_binary_vector([0, 90, 180, 270])
        self.assertEqual(vec, [1.0, 1.0, 1.0, 1.0])

        # Test empty list
        vec = psle.yaw_angles_to_binary_vector([])
        self.assertEqual(vec, [0.0, 0.0, 0.0, 0.0])

        # Test that invalid yaw raises error
        with self.assertRaises(ValueError):
            psle.yaw_angles_to_binary_vector([45])

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
        """Test that positions are output as binary vectors indicating yaw presence"""
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
                # Split 4D binary vector across 2 positions:
                # position 0: [yaw_0_present, yaw_90_present]
                # position 1: [yaw_180_present, yaw_270_present]
                pos0 = output.positions[0, i, 0, :]
                pos1 = output.positions[0, i, 1, :]

                # Each element should be either 0.0 or 1.0
                for val in [pos0[0].item(), pos0[1].item(), pos1[0].item(), pos1[1].item()]:
                    self.assertTrue(val == 0.0 or val == 1.0,
                                    f"Yaw vector element should be 0.0 or 1.0, got {val}")

                # At least one yaw should be present (not all zeros)
                total = pos0.sum().item() + pos1.sum().item()
                self.assertTrue(total > 0,
                                "At least one yaw should be present in the vector")

                break


if __name__ == "__main__":
    unittest.main()
