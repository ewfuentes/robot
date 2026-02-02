
import unittest
import tempfile
import json
import math
import hashlib
import pickle

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
                }

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

        # Verify at least one unmasked entry exists
        unmasked_count = (~output.mask[0]).sum().item()
        self.assertGreater(unmasked_count, 0, "Should have at least one unmasked landmark")

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

        # Verify at least one unmasked entry exists
        unmasked_count = (~output.mask[0]).sum().item()
        self.assertGreater(unmasked_count, 0, "Should have at least one unmasked landmark")

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

        # Verify all 3 landmarks are unmasked
        self.assertEqual(output.features.shape[1], 3, "Should have 3 landmarks")
        for i in range(3):
            self.assertFalse(output.mask[0, i].item(), f"Landmark {i} should not be masked")

        # Positions should be [batch, num_landmarks, 2, 2]
        # Split 4D binary vector across 2 positions:
        # position 0: [yaw_0_present, yaw_90_present]
        # position 1: [yaw_180_present, yaw_270_present]
        #
        # Test fixture defines:
        # - landmark 0: yaws [0, 90] -> [1.0, 1.0, 0.0, 0.0]
        # - landmark 1: yaws [180]   -> [0.0, 0.0, 1.0, 0.0]
        # - landmark 2: yaws [270]   -> [0.0, 0.0, 0.0, 1.0]

        # Check landmark 0: yaws [0, 90]
        pos = output.positions[0, 0]
        self.assertEqual(pos[0, 0].item(), 1.0, "Landmark 0 should have yaw 0")
        self.assertEqual(pos[0, 1].item(), 1.0, "Landmark 0 should have yaw 90")
        self.assertEqual(pos[1, 0].item(), 0.0, "Landmark 0 should not have yaw 180")
        self.assertEqual(pos[1, 1].item(), 0.0, "Landmark 0 should not have yaw 270")

        # Check landmark 1: yaws [180]
        pos = output.positions[0, 1]
        self.assertEqual(pos[0, 0].item(), 0.0, "Landmark 1 should not have yaw 0")
        self.assertEqual(pos[0, 1].item(), 0.0, "Landmark 1 should not have yaw 90")
        self.assertEqual(pos[1, 0].item(), 1.0, "Landmark 1 should have yaw 180")
        self.assertEqual(pos[1, 1].item(), 0.0, "Landmark 1 should not have yaw 270")

        # Check landmark 2: yaws [270]
        pos = output.positions[0, 2]
        self.assertEqual(pos[0, 0].item(), 0.0, "Landmark 2 should not have yaw 0")
        self.assertEqual(pos[0, 1].item(), 0.0, "Landmark 2 should not have yaw 90")
        self.assertEqual(pos[1, 0].item(), 0.0, "Landmark 2 should not have yaw 180")
        self.assertEqual(pos[1, 1].item(), 1.0, "Landmark 2 should have yaw 270")


def decode_sentence_tensor(sentence_bytes: torch.Tensor) -> str:
    """Decode a UTF-8 byte tensor back to a string."""
    byte_list = sentence_bytes.tolist()
    while byte_list and byte_list[-1] == 0:
        byte_list.pop()
    return bytes(byte_list).decode('utf-8', errors='replace')


class PanoramaSemanticLandmarkExtractorV2Test(unittest.TestCase):
    """Tests for v2.0 pickle format loading.

    These tests verify that panorama_semantic_landmark_extractor correctly:
    1. Loads v2.0 format pickles
    2. Returns the correct description/sentence for each landmark
    3. Returns the correct embedding for each description
    4. Handles multi-city loading without offset issues

    Note: panorama_semantic_landmark_extractor uses a dict-based approach
    (not tensor+offset), so it's NOT vulnerable to the offset bug that
    affected additional_panorama_extractors. But we test this explicitly.
    """

    @classmethod
    def setUpClass(cls):
        """Create temp directory with v2.0 pickle format for multiple cities.

        Test fixtures include:
        - Different numbers of landmarks per panorama (1-4)
        - Duplicate descriptions within a city (same text, different custom_ids)
        - Duplicate descriptions across cities (same text in Chicago and Seattle)
        - Varied yaw angle combinations
        - Unique descriptions for specific test assertions
        """
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)
        version = "test_v2"

        # Store expected data for verification
        cls.expected_embeddings = {}
        cls.expected_sentences = {}

        # Shared descriptions that appear in BOTH cities (cross-city duplicates)
        SHARED_DESCRIPTIONS = [
            "A red brick building with white trim",
            "A tall glass skyscraper reflecting the sky",
            "A small coffee shop with outdoor seating",
        ]

        # === City 1: Chicago ===
        # Landmarks with varied descriptions and some duplicates
        chicago_landmark_data = {
            # pano_chi_0: 3 landmarks, includes shared description
            "pano_chi_0,41.85,-87.65,": [
                {"desc": SHARED_DESCRIPTIONS[0], "yaws": [0, 90]},  # Shared with Seattle
                {"desc": "Chicago-only: The Willis Tower dominates the skyline", "yaws": [180]},
                {"desc": "A yellow taxi cab parked on the street", "yaws": [270]},
            ],
            # pano_chi_1: 4 landmarks, includes within-city duplicate
            "pano_chi_1,41.86,-87.64,": [
                {"desc": "A yellow taxi cab parked on the street", "yaws": [0]},  # Duplicate within Chicago
                {"desc": SHARED_DESCRIPTIONS[1], "yaws": [90, 180]},  # Shared with Seattle
                {"desc": "Chicago-only: Deep dish pizza restaurant with neon sign", "yaws": [270]},
                {"desc": "A fire hydrant next to a lamp post", "yaws": [0, 90, 180, 270]},  # All yaws
            ],
            # pano_chi_2: 2 landmarks only
            "pano_chi_2,41.87,-87.63,": [
                {"desc": SHARED_DESCRIPTIONS[2], "yaws": [90]},  # Shared with Seattle
                {"desc": "Chicago-only: The Bean sculpture in Millennium Park", "yaws": [180, 270]},
            ],
        }

        chicago_panoramas = {}
        chicago_descriptions = []
        for pano_id, landmark_list in chicago_landmark_data.items():
            landmarks = []
            for lm_idx, lm_data in enumerate(landmark_list):
                description = lm_data["desc"]
                custom_id = f"{pano_id}__landmark_{lm_idx}"
                chicago_descriptions.append((custom_id, description))
                cls.expected_sentences[custom_id] = description
                cls.expected_embeddings[custom_id] = create_random_embedding_vector(description)

                landmarks.append({
                    "description": description,
                    "bounding_boxes": [
                        {"yaw_angle": str(yaw), "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        for yaw in lm_data["yaws"]
                    ],
                    "proper_nouns": [],
                    "landmark_idx": lm_idx,
                })
            chicago_panoramas[pano_id] = {
                "location_type": "Urban commercial district",
                "landmarks": landmarks,
            }

        chicago_dir = base_path / version / "1_Chicago" / "embeddings"
        chicago_dir.mkdir(parents=True, exist_ok=True)
        chicago_data = cls._create_v2_pickle(chicago_panoramas, chicago_descriptions)
        with open(chicago_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(chicago_data, f)

        # === City 2: Seattle ===
        seattle_landmark_data = {
            # pano_sea_0: 3 landmarks
            "pano_sea_0,47.60,-122.33,": [
                {"desc": "Seattle-only: The Space Needle against cloudy sky", "yaws": [0]},
                {"desc": SHARED_DESCRIPTIONS[0], "yaws": [90, 180]},  # Shared with Chicago
                {"desc": "A green street sign reading Pike Place", "yaws": [270]},
            ],
            # pano_sea_1: 4 landmarks, includes within-city duplicate
            "pano_sea_1,47.61,-122.32,": [
                {"desc": SHARED_DESCRIPTIONS[1], "yaws": [0, 90]},  # Shared with Chicago
                {"desc": "A green street sign reading Pike Place", "yaws": [180]},  # Duplicate within Seattle
                {"desc": "Seattle-only: Ferry boat crossing Puget Sound", "yaws": [270]},
                {"desc": "Seattle-only: Mount Rainier visible in distance", "yaws": [0, 180]},
            ],
            # pano_sea_2: 1 landmark only (minimal case)
            "pano_sea_2,47.62,-122.31,": [
                {"desc": SHARED_DESCRIPTIONS[2], "yaws": [90, 270]},  # Shared with Chicago
            ],
        }

        seattle_panoramas = {}
        seattle_descriptions = []
        for pano_id, landmark_list in seattle_landmark_data.items():
            landmarks = []
            for lm_idx, lm_data in enumerate(landmark_list):
                description = lm_data["desc"]
                custom_id = f"{pano_id}__landmark_{lm_idx}"
                seattle_descriptions.append((custom_id, description))
                cls.expected_sentences[custom_id] = description
                cls.expected_embeddings[custom_id] = create_random_embedding_vector(description)

                landmarks.append({
                    "description": description,
                    "bounding_boxes": [
                        {"yaw_angle": str(yaw), "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        for yaw in lm_data["yaws"]
                    ],
                    "proper_nouns": [],
                    "landmark_idx": lm_idx,
                })
            seattle_panoramas[pano_id] = {
                "location_type": "Urban waterfront",
                "landmarks": landmarks,
            }

        seattle_dir = base_path / version / "2_Seattle" / "embeddings"
        seattle_dir.mkdir(parents=True, exist_ok=True)
        seattle_data = cls._create_v2_pickle(seattle_panoramas, seattle_descriptions)
        with open(seattle_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(seattle_data, f)

        cls.chicago_pano_ids = ["pano_chi_0", "pano_chi_1", "pano_chi_2"]
        cls.seattle_pano_ids = ["pano_sea_0", "pano_sea_1", "pano_sea_2"]
        cls.all_pano_ids = cls.chicago_pano_ids + cls.seattle_pano_ids

        # Store shared descriptions for tests
        cls.shared_descriptions = SHARED_DESCRIPTIONS

        # Store expected landmark counts per panorama
        cls.expected_landmark_counts = {
            "pano_chi_0": 3,
            "pano_chi_1": 4,
            "pano_chi_2": 2,
            "pano_sea_0": 3,
            "pano_sea_1": 4,
            "pano_sea_2": 1,
        }

    @classmethod
    def _create_v2_pickle(cls, panoramas: dict, descriptions: list) -> dict:
        """Helper to create v2.0 pickle data."""
        description_embeddings = torch.stack([
            torch.tensor(create_random_embedding_vector(desc)) for _, desc in descriptions
        ])
        description_id_to_idx = {cid: i for i, (cid, _) in enumerate(descriptions)}

        return {
            "version": "2.0",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "task_type": "SEMANTIC_SIMILARITY",
            "panoramas": panoramas,
            "description_embeddings": description_embeddings,
            "description_id_to_idx": description_id_to_idx,
            "location_type_embeddings": torch.zeros((0, 1536)),
            "location_type_to_idx": {},
            "proper_noun_embeddings": torch.zeros((0, 1536)),
            "proper_noun_to_idx": {},
        }

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_v2_pickle_loading(self):
        """Test that v2.0 pickle format is loaded correctly from multiple cities."""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.load_files()

        # Chicago: 3 + 4 + 2 = 9 landmarks, Seattle: 3 + 4 + 1 = 8 landmarks
        # Total = 17 unique custom_ids (descriptions can be duplicated but custom_ids are unique)
        self.assertEqual(len(model.all_embeddings), 17)
        self.assertEqual(len(model.all_sentences), 17)

        # Should have loaded metadata for all 6 panoramas
        self.assertEqual(len(model.panorama_metadata), 6)

        # Verify correct number of landmarks per panorama
        for pano_id, expected_count in self.expected_landmark_counts.items():
            self.assertIn(pano_id, model.panorama_metadata,
                         f"Missing panorama {pano_id}")
            actual_count = len(model.panorama_metadata[pano_id])
            self.assertEqual(actual_count, expected_count,
                           f"Panorama {pano_id}: expected {expected_count} landmarks, got {actual_count}")

    def test_v2_returns_correct_sentence(self):
        """Test that the correct description/sentence is returned for each landmark."""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Test Chicago panorama with 4 landmarks (pano_chi_1)
        pano_id = "pano_chi_1"
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}]
        )
        output = model(model_input)

        # Should have exactly 4 landmarks
        unmasked_count = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked_count, 4, "pano_chi_1 should have 4 landmarks")

        # Collect all returned sentences
        returned_sentences = []
        for j in range(output.mask.shape[1]):
            if not output.mask[0, j].item():
                sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                returned_sentences.append(sentence)

        # Verify expected sentences are present
        self.assertIn("A yellow taxi cab parked on the street", returned_sentences)
        self.assertIn("A tall glass skyscraper reflecting the sky", returned_sentences)
        self.assertIn("Chicago-only: Deep dish pizza restaurant with neon sign", returned_sentences)
        self.assertIn("A fire hydrant next to a lamp post", returned_sentences)

        # Test Seattle panorama with only 1 landmark (pano_sea_2) - minimal case
        pano_id = "pano_sea_2"
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}]
        )
        output = model(model_input)

        # Should have exactly 1 landmark
        unmasked_count = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked_count, 1, "pano_sea_2 should have 1 landmark")

        sentence = decode_sentence_tensor(output.debug["sentences"][0, 0])
        self.assertEqual(sentence, "A small coffee shop with outdoor seating")

    def test_v2_returns_correct_embedding(self):
        """Test that returned embeddings match expected values for each description.

        Key test: embeddings are generated from the description text, so duplicate
        descriptions should produce the same embedding regardless of which panorama
        or city they appear in.
        """
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Test all panoramas to verify embeddings match descriptions
        for pano_id in self.all_pano_ids:
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}]
            )
            output = model(model_input)

            expected_count = self.expected_landmark_counts[pano_id]
            unmasked_count = (~output.mask[0]).sum().item()
            self.assertEqual(unmasked_count, expected_count,
                           f"{pano_id} should have {expected_count} landmarks")

            for j in range(output.mask.shape[1]):
                if not output.mask[0, j].item():
                    sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                    returned_embedding = output.features[0, j]

                    # Embedding is generated from description text
                    expected_embedding = torch.tensor(create_random_embedding_vector(sentence))

                    # Normalize for comparison (model normalizes after cropping)
                    expected_normalized = expected_embedding / torch.norm(expected_embedding)
                    returned_normalized = returned_embedding / torch.norm(returned_embedding)

                    self.assertTrue(
                        torch.allclose(returned_normalized, expected_normalized, atol=1e-5),
                        f"Embedding mismatch for '{sentence[:50]}...' in {pano_id}")

    def test_v2_duplicate_descriptions_same_embedding(self):
        """Test that duplicate descriptions (within or across cities) produce same embedding.

        This verifies that the embedding is based on the description text, not the
        custom_id or panorama location.
        """
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # "A yellow taxi cab parked on the street" appears in both pano_chi_0 and pano_chi_1
        # Get embeddings from both panoramas
        embeddings_for_taxi = []

        for pano_id in ["pano_chi_0", "pano_chi_1"]:
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}]
            )
            output = model(model_input)

            for j in range(output.mask.shape[1]):
                if not output.mask[0, j].item():
                    sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                    if sentence == "A yellow taxi cab parked on the street":
                        embeddings_for_taxi.append(output.features[0, j].clone())

        # Should have found 2 occurrences
        self.assertEqual(len(embeddings_for_taxi), 2,
                        "Should find 'taxi' description in 2 panoramas")

        # Both embeddings should be identical
        self.assertTrue(
            torch.allclose(embeddings_for_taxi[0], embeddings_for_taxi[1], atol=1e-5),
            "Same description should produce same embedding across panoramas")

    def test_v2_multi_city_no_cross_contamination(self):
        """Verify that multi-city loading returns correct landmarks for each panorama.

        This tests that when querying panoramas from different cities in the same batch,
        each panorama gets its own landmarks (not mixed up with other panoramas).
        """
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Query panoramas with city-specific descriptions
        # pano_chi_2 has: shared coffee shop + Chicago-only Bean sculpture
        # pano_sea_0 has: Seattle-only Space Needle + shared brick building + Pike Place sign
        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": "pano_chi_2"}, {"pano_id": "pano_sea_0"}]
        )
        output = model(model_input)

        # Verify correct landmark counts
        chicago_unmasked = (~output.mask[0]).sum().item()
        seattle_unmasked = (~output.mask[1]).sum().item()
        self.assertEqual(chicago_unmasked, 2, "pano_chi_2 should have 2 landmarks")
        self.assertEqual(seattle_unmasked, 3, "pano_sea_0 should have 3 landmarks")

        # Collect sentences from each panorama
        chicago_sentences = []
        seattle_sentences = []

        for j in range(output.mask.shape[1]):
            if not output.mask[0, j].item():
                chicago_sentences.append(decode_sentence_tensor(output.debug["sentences"][0, j]))
            if not output.mask[1, j].item():
                seattle_sentences.append(decode_sentence_tensor(output.debug["sentences"][1, j]))

        # Verify Chicago has its city-only description
        self.assertTrue(
            any("Chicago-only" in s for s in chicago_sentences),
            f"Chicago should have Chicago-only landmark, got: {chicago_sentences}")

        # Verify Seattle has its city-only description
        self.assertTrue(
            any("Seattle-only" in s for s in seattle_sentences),
            f"Seattle should have Seattle-only landmark, got: {seattle_sentences}")

        # Verify no cross-contamination of city-only descriptions
        self.assertFalse(
            any("Seattle-only" in s for s in chicago_sentences),
            "Chicago should not have Seattle-only descriptions")
        self.assertFalse(
            any("Chicago-only" in s for s in seattle_sentences),
            "Seattle should not have Chicago-only descriptions")

    def test_v2_shared_descriptions_across_cities(self):
        """Test that shared descriptions work correctly across cities.

        Some descriptions appear in both Chicago and Seattle panoramas.
        Each should return the correct embedding (which is the same since
        embeddings are generated from description text).
        """
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # "A red brick building with white trim" appears in pano_chi_0 and pano_sea_0
        shared_desc = self.shared_descriptions[0]
        embeddings_found = []

        for pano_id in ["pano_chi_0", "pano_sea_0"]:
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}]
            )
            output = model(model_input)

            for j in range(output.mask.shape[1]):
                if not output.mask[0, j].item():
                    sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                    if sentence == shared_desc:
                        embeddings_found.append((pano_id, output.features[0, j].clone()))

        # Should find the shared description in both cities
        self.assertEqual(len(embeddings_found), 2,
                        f"Should find '{shared_desc}' in both cities")

        # Both should have the same embedding
        self.assertTrue(
            torch.allclose(embeddings_found[0][1], embeddings_found[1][1], atol=1e-5),
            "Same description should produce same embedding across cities")

    def test_v2_forward_pass(self):
        """Test forward pass with v2.0 format data."""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in self.chicago_pano_ids[:2]]
        )

        output = model(model_input)

        # Verify output shape
        self.assertEqual(output.features.shape[0], 2)
        self.assertEqual(output.features.shape[2], config.openai_embedding_size)

        # Verify at least some landmarks are not masked
        self.assertFalse(output.mask.all())

    def test_v2_yaw_extraction(self):
        """Test that yaw angles are correctly extracted from v2.0 bounding boxes."""
        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        model = psle.PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.load_files()

        # Check that metadata has yaw angles
        for pano_id in self.all_pano_ids:
            self.assertIn(pano_id, model.panorama_metadata)
            for landmark in model.panorama_metadata[pano_id]:
                self.assertIn("yaw_angles", landmark)
                self.assertIsInstance(landmark["yaw_angles"], list)
                # Verify yaw angles are valid values
                for yaw in landmark["yaw_angles"]:
                    self.assertIn(yaw, [0, 90, 180, 270])


if __name__ == "__main__":
    unittest.main()
