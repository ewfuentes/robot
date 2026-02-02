
import unittest
import tempfile
import pickle
import hashlib

import common.torch.load_torch_deps
import torch
from pathlib import Path
import numpy as np

import experimental.overhead_matching.swag.model.additional_panorama_extractors as ape
from experimental.overhead_matching.swag.model.swag_config_types import (
    PanoramaProperNounExtractorConfig,
    PanoramaLocationTypeExtractorConfig,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput


def create_random_embedding_vector(category: str, dim: int = 1536) -> torch.Tensor:
    """Create a deterministic random vector from a category string."""
    bits = hashlib.sha512(category.encode('utf-8')).digest() * 3
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))[:dim]
    scaled_bits = (2.0 * bits - 1.0).astype(dtype=np.float32)
    scaled_bits = scaled_bits / np.linalg.norm(scaled_bits)
    return torch.tensor(scaled_bits)


def decode_sentence_tensor(sentence_bytes: torch.Tensor) -> str:
    """Decode a UTF-8 byte tensor back to a string."""
    byte_list = sentence_bytes.tolist()
    while byte_list and byte_list[-1] == 0:
        byte_list.pop()
    return bytes(byte_list).decode('utf-8', errors='replace')


def create_v2_pickle_data(
    panoramas: dict,
    proper_nouns: list[str],
    location_types: list[str],
    descriptions: list[tuple[str, str]],  # list of (custom_id, description)
    embedding_dim: int = 1536,
) -> dict:
    """Create a v2.0 format pickle data structure."""
    # Create proper noun embeddings
    proper_noun_embeddings = torch.stack([
        create_random_embedding_vector(noun, embedding_dim) for noun in proper_nouns
    ]) if proper_nouns else torch.zeros((0, embedding_dim))
    proper_noun_to_idx = {noun: i for i, noun in enumerate(proper_nouns)}

    # Create location type embeddings
    location_type_embeddings = torch.stack([
        create_random_embedding_vector(loc, embedding_dim) for loc in location_types
    ]) if location_types else torch.zeros((0, embedding_dim))
    location_type_to_idx = {loc: i for i, loc in enumerate(location_types)}

    # Create description embeddings
    description_embeddings = torch.stack([
        create_random_embedding_vector(desc, embedding_dim) for _, desc in descriptions
    ]) if descriptions else torch.zeros((0, embedding_dim))
    description_id_to_idx = {cid: i for i, (cid, _) in enumerate(descriptions)}

    return {
        "version": "2.0",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": embedding_dim,
        "task_type": "SEMANTIC_SIMILARITY",
        "panoramas": panoramas,
        "description_embeddings": description_embeddings,
        "description_id_to_idx": description_id_to_idx,
        "location_type_embeddings": location_type_embeddings,
        "location_type_to_idx": location_type_to_idx,
        "proper_noun_embeddings": proper_noun_embeddings,
        "proper_noun_to_idx": proper_noun_to_idx,
    }


class AdditionalPanoramaExtractorsTest(unittest.TestCase):
    """Tests for PanoramaProperNounExtractor and PanoramaLocationTypeExtractor.

    These tests verify correct behavior when loading embeddings from multiple cities.
    The key bug being tested: with dedupe_keys=True (for location types), the old code
    calculated offset as len(dict) instead of len(tensor_rows). This caused indices
    to point to wrong rows when there were duplicate keys being skipped.

    Test fixtures include:
    - Different numbers of landmarks per panorama (1-4)
    - Shared proper nouns across cities (e.g., "Starbucks" in Chicago and Seattle)
    - Duplicate proper nouns within cities (same noun on multiple landmarks)
    - Varied yaw angle combinations
    - Location type duplicates across cities to test offset bug
    """

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)
        version = "test_v2"

        # Store expected embeddings
        cls.expected_proper_noun_embeddings = {}
        cls.expected_location_type_embeddings = {}

        # Shared proper nouns that appear in BOTH cities
        SHARED_PROPER_NOUNS = ["Starbucks", "McDonald's", "Subway"]

        # === City 1: Chicago ===
        chicago_location_types = [
            "Urban commercial district",
            "Residential neighborhood",
            "Industrial zone",
            "Downtown area",
            "Suburban street",
        ]
        # Chicago proper nouns: 2 shared + 2 Chicago-only
        chicago_proper_nouns = [
            "Starbucks",      # Shared with Seattle
            "McDonald's",     # Shared with Seattle
            "Chicago Pizza",  # Chicago-only
            "Willis Tower",   # Chicago-only
        ]

        # Store expected embeddings (Chicago loaded first)
        for loc in chicago_location_types:
            cls.expected_location_type_embeddings[loc] = create_random_embedding_vector(loc, 1536)
        for noun in chicago_proper_nouns:
            cls.expected_proper_noun_embeddings[noun] = create_random_embedding_vector(noun, 1536)

        # Define Chicago landmarks with varied structure
        # pano_chi_0: 3 landmarks, includes duplicate proper noun within panorama
        # pano_chi_1: 4 landmarks, varied yaws
        # pano_chi_2: 2 landmarks only
        chicago_landmark_data = {
            "pano_chi_0,41.85,-87.65,": {
                "location_type": "Urban commercial district",
                "landmarks": [
                    {"proper_nouns": ["Starbucks"], "yaws": [0, 90]},
                    {"proper_nouns": ["McDonald's"], "yaws": [180]},
                    {"proper_nouns": ["Starbucks"], "yaws": [270]},  # Duplicate within pano
                ],
            },
            "pano_chi_1,41.86,-87.64,": {
                "location_type": "Residential neighborhood",
                "landmarks": [
                    {"proper_nouns": ["Chicago Pizza"], "yaws": [0]},
                    {"proper_nouns": ["Starbucks", "McDonald's"], "yaws": [90, 180]},  # Multiple nouns
                    {"proper_nouns": ["Willis Tower"], "yaws": [270]},
                    {"proper_nouns": ["McDonald's"], "yaws": [0, 90, 180, 270]},  # All yaws
                ],
            },
            "pano_chi_2,41.87,-87.63,": {
                "location_type": "Industrial zone",
                "landmarks": [
                    {"proper_nouns": ["Chicago Pizza"], "yaws": [90]},
                    {"proper_nouns": ["Willis Tower"], "yaws": [180, 270]},
                ],
            },
        }

        chicago_panoramas = {}
        chicago_descriptions = []
        cls.chicago_expected_landmarks = {}  # Track expected landmarks per pano
        for pano_id, pano_data in chicago_landmark_data.items():
            landmarks = []
            pano_id_clean = pano_id.split(",")[0]
            cls.chicago_expected_landmarks[pano_id_clean] = []
            for lm_idx, lm_data in enumerate(pano_data["landmarks"]):
                custom_id = f"{pano_id}__landmark_{lm_idx}"
                description = f"Chicago landmark at {pano_id_clean} index {lm_idx}"
                chicago_descriptions.append((custom_id, description))
                landmarks.append({
                    "description": description,
                    "bounding_boxes": [
                        {"yaw_angle": str(yaw), "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        for yaw in lm_data["yaws"]
                    ],
                    "proper_nouns": lm_data["proper_nouns"],
                    "landmark_idx": lm_idx,
                })
                # Track one entry per proper noun (extractor creates one output per proper noun)
                for noun in lm_data["proper_nouns"]:
                    cls.chicago_expected_landmarks[pano_id_clean].append({
                        "proper_noun": noun,
                        "yaws": lm_data["yaws"],
                    })
            chicago_panoramas[pano_id] = {
                "location_type": pano_data["location_type"],
                "landmarks": landmarks,
            }

        # Use numeric prefixes to ensure consistent directory iteration order
        chicago_dir = base_path / version / "1_Chicago" / "embeddings"
        chicago_dir.mkdir(parents=True, exist_ok=True)
        chicago_data = create_v2_pickle_data(
            chicago_panoramas, chicago_proper_nouns, chicago_location_types, chicago_descriptions)
        with open(chicago_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(chicago_data, f)

        # === City 2: NewYork ===
        # NewYork has 5 location types, 3 overlap with Chicago (to create offset divergence)
        newyork_location_types = [
            "Urban commercial district",  # Overlaps with Chicago
            "Residential neighborhood",   # Overlaps with Chicago
            "Downtown area",              # Overlaps with Chicago
            "Times Square",               # NewYork-only
            "Brooklyn Bridge",            # NewYork-only
        ]
        newyork_proper_nouns = ["Shake Shack", "Yellow Cab", "Broadway", "Subway"]  # Subway shared

        for loc in newyork_location_types:
            if loc not in cls.expected_location_type_embeddings:
                cls.expected_location_type_embeddings[loc] = create_random_embedding_vector(loc, 1536)
        for noun in newyork_proper_nouns:
            if noun not in cls.expected_proper_noun_embeddings:
                cls.expected_proper_noun_embeddings[noun] = create_random_embedding_vector(noun, 1536)

        newyork_landmark_data = {
            "pano_nyc_0,40.71,-74.01,": {
                "location_type": "Times Square",
                "landmarks": [
                    {"proper_nouns": ["Shake Shack"], "yaws": [0]},
                    {"proper_nouns": ["Broadway"], "yaws": [90, 180]},
                    {"proper_nouns": ["Yellow Cab"], "yaws": [270]},
                ],
            },
            "pano_nyc_1,40.72,-74.00,": {
                "location_type": "Brooklyn Bridge",
                "landmarks": [
                    {"proper_nouns": ["Subway"], "yaws": [0, 90]},  # Shared noun
                    {"proper_nouns": ["Yellow Cab"], "yaws": [180, 270]},
                ],
            },
            "pano_nyc_2,40.73,-73.99,": {
                "location_type": "Urban commercial district",  # Shared with Chicago
                "landmarks": [
                    {"proper_nouns": ["Shake Shack", "Subway"], "yaws": [90]},
                ],
            },
        }

        newyork_panoramas = {}
        newyork_descriptions = []
        for pano_id, pano_data in newyork_landmark_data.items():
            landmarks = []
            for lm_idx, lm_data in enumerate(pano_data["landmarks"]):
                custom_id = f"{pano_id}__landmark_{lm_idx}"
                description = f"NewYork landmark {lm_idx}"
                newyork_descriptions.append((custom_id, description))
                landmarks.append({
                    "description": description,
                    "bounding_boxes": [
                        {"yaw_angle": str(yaw), "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        for yaw in lm_data["yaws"]
                    ],
                    "proper_nouns": lm_data["proper_nouns"],
                    "landmark_idx": lm_idx,
                })
            newyork_panoramas[pano_id] = {
                "location_type": pano_data["location_type"],
                "landmarks": landmarks,
            }

        newyork_dir = base_path / version / "2_NewYork" / "embeddings"
        newyork_dir.mkdir(parents=True, exist_ok=True)
        newyork_data = create_v2_pickle_data(
            newyork_panoramas, newyork_proper_nouns, newyork_location_types, newyork_descriptions)
        with open(newyork_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(newyork_data, f)

        # === City 3: Seattle ===
        # 2 location types overlap with Chicago, 3 are Seattle-only
        seattle_location_types = [
            "Urban commercial district",  # Overlaps with Chicago (index 0)
            "Residential neighborhood",   # Overlaps with Chicago (index 1)
            "Urban waterfront",           # Seattle-only (index 2) - KEY TEST CASE
            "Tech campus",                # Seattle-only (index 3)
            "Ferry terminal",             # Seattle-only (index 4)
        ]
        # Seattle proper nouns: 2 shared + 2 Seattle-only
        seattle_proper_nouns = [
            "Space Needle",   # Seattle-only
            "Starbucks",      # Shared with Chicago - DIFFERENT INDEX than Chicago!
            "Pike Place",     # Seattle-only
            "McDonald's",     # Shared with Chicago
        ]

        # Store expected embeddings for Seattle-only items
        for loc in seattle_location_types:
            if loc not in cls.expected_location_type_embeddings:
                cls.expected_location_type_embeddings[loc] = create_random_embedding_vector(loc, 1536)
        for noun in seattle_proper_nouns:
            if noun not in cls.expected_proper_noun_embeddings:
                cls.expected_proper_noun_embeddings[noun] = create_random_embedding_vector(noun, 1536)

        # Define Seattle landmarks with varied structure
        seattle_landmark_data = {
            "pano_sea_0,47.60,-122.33,": {
                "location_type": "Urban waterfront",
                "landmarks": [
                    {"proper_nouns": ["Space Needle"], "yaws": [90]},
                    {"proper_nouns": ["Starbucks"], "yaws": [180, 270]},  # Shared noun
                    {"proper_nouns": ["Pike Place"], "yaws": [0, 90, 180]},
                ],
            },
            "pano_sea_1,47.61,-122.32,": {
                "location_type": "Tech campus",
                "landmarks": [
                    {"proper_nouns": ["Starbucks"], "yaws": [0]},  # Duplicate within Seattle
                    {"proper_nouns": ["McDonald's"], "yaws": [90]},  # Shared with Chicago
                    {"proper_nouns": ["Space Needle"], "yaws": [180]},
                    {"proper_nouns": ["Pike Place"], "yaws": [270]},
                ],
            },
            "pano_sea_2,47.62,-122.31,": {
                "location_type": "Ferry terminal",
                "landmarks": [
                    {"proper_nouns": ["Pike Place", "Starbucks"], "yaws": [0, 180]},  # Multiple nouns
                ],
            },
        }

        seattle_panoramas = {}
        seattle_descriptions = []
        cls.seattle_expected_landmarks = {}
        for pano_id, pano_data in seattle_landmark_data.items():
            landmarks = []
            pano_id_clean = pano_id.split(",")[0]
            cls.seattle_expected_landmarks[pano_id_clean] = []
            for lm_idx, lm_data in enumerate(pano_data["landmarks"]):
                custom_id = f"{pano_id}__landmark_{lm_idx}"
                description = f"Seattle landmark at {pano_id_clean} index {lm_idx}"
                seattle_descriptions.append((custom_id, description))
                landmarks.append({
                    "description": description,
                    "bounding_boxes": [
                        {"yaw_angle": str(yaw), "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        for yaw in lm_data["yaws"]
                    ],
                    "proper_nouns": lm_data["proper_nouns"],
                    "landmark_idx": lm_idx,
                })
                # Track one entry per proper noun (extractor creates one output per proper noun)
                for noun in lm_data["proper_nouns"]:
                    cls.seattle_expected_landmarks[pano_id_clean].append({
                        "proper_noun": noun,
                        "yaws": lm_data["yaws"],
                    })
            seattle_panoramas[pano_id] = {
                "location_type": pano_data["location_type"],
                "landmarks": landmarks,
            }

        seattle_dir = base_path / version / "3_Seattle" / "embeddings"
        seattle_dir.mkdir(parents=True, exist_ok=True)
        seattle_data = create_v2_pickle_data(
            seattle_panoramas, seattle_proper_nouns, seattle_location_types, seattle_descriptions)
        with open(seattle_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(seattle_data, f)

        # Store pano IDs
        cls.chicago_pano_ids = ["pano_chi_0", "pano_chi_1", "pano_chi_2"]
        cls.seattle_pano_ids = ["pano_sea_0", "pano_sea_1", "pano_sea_2"]

        # Store expected mappings for location types
        cls.seattle_pano_location_types = {
            "pano_sea_0": "Urban waterfront",
            "pano_sea_1": "Tech campus",
            "pano_sea_2": "Ferry terminal",
        }
        cls.chicago_pano_location_types = {
            "pano_chi_0": "Urban commercial district",
            "pano_chi_1": "Residential neighborhood",
            "pano_chi_2": "Industrial zone",
        }

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    # ==================== Location Type Tests ====================

    def test_location_type_seattle_only_returns_correct_embedding(self):
        """Regression test: Seattle-only location types must return correct embeddings.

        Tests the offset bug in the old tensor+offset code where dedupe_keys=True
        caused indices to point to wrong rows in the concatenated tensor.

        The bug: offset was calculated as len(dict) instead of tensor_rows.
        With 3 cities where earlier cities have duplicate keys being skipped:
        - After Chicago (5 types): dict=5, tensor=5
        - After NewYork (5 types, 3 duplicates): dict=7, tensor=10
        - For Seattle (5 types, 2 duplicates): offset should be 10 (tensor), not 7 (dict)

        Seattle's "Urban waterfront" at local index 2:
        - Buggy: 2 + 7 = 9 (points to NewYork's row)
        - Correct: 2 + 10 = 12 (points to Seattle's row)
        """
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))

        for pano_id, expected_loc in self.seattle_pano_location_types.items():
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}])

            output = extractor(model_input)

            self.assertFalse(output.mask[0, 0].item(),
                           f"Entry should not be masked for {pano_id}")

            returned_loc = decode_sentence_tensor(output.debug["sentences"][0, 0])
            self.assertEqual(returned_loc, expected_loc,
                           f"Location type mismatch for {pano_id}")

            returned_embedding = output.features[0, 0]
            expected_embedding = self.expected_location_type_embeddings[expected_loc]
            self.assertTrue(
                torch.allclose(returned_embedding, expected_embedding, atol=1e-5),
                f"Embedding mismatch for '{expected_loc}' in {pano_id}")

    def test_location_type_file_loading(self):
        """Test loading location type embeddings from multi-city v2.0 pickle structure."""
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))
        extractor.load_files()
        self.assertTrue(extractor.files_loaded)

    def test_location_type_forward_basic(self):
        """Test forward pass produces exactly one token per panorama."""
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))

        batch_pano_ids = self.chicago_pano_ids[:2] + self.seattle_pano_ids[:1]
        batch_size = len(batch_pano_ids)

        model_input = ModelInput(
            image=torch.zeros((batch_size, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in batch_pano_ids])

        output = extractor(model_input)

        self.assertEqual(output.features.shape[1], 1)
        self.assertEqual(output.features.shape[0], batch_size)
        self.assertEqual(output.features.shape[2], config.openai_embedding_size)
        self.assertFalse(output.mask.all())

    def test_location_type_positions_are_zero(self):
        """Test that location type positions are zero (global scene descriptor)."""
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in self.chicago_pano_ids[:2]])

        output = extractor(model_input)
        self.assertTrue(torch.all(output.positions == 0))

    def test_location_type_shared_across_cities(self):
        """Test that shared location types return same embedding across cities.

        "Urban commercial district" appears in both Chicago and NewYork.
        The dict-based approach should return the same embedding for both.
        """
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))

        # pano_chi_0 and pano_nyc_2 both have "Urban commercial district"
        chicago_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_chi_0"}])
        chicago_output = extractor(chicago_input)

        newyork_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_nyc_2"}])
        newyork_output = extractor(newyork_input)

        # Verify both return the same location type
        chicago_loc = decode_sentence_tensor(chicago_output.debug["sentences"][0, 0])
        newyork_loc = decode_sentence_tensor(newyork_output.debug["sentences"][0, 0])
        self.assertEqual(chicago_loc, "Urban commercial district")
        self.assertEqual(newyork_loc, "Urban commercial district")

        # Verify embeddings match
        self.assertTrue(
            torch.allclose(chicago_output.features[0, 0], newyork_output.features[0, 0], atol=1e-5),
            "Shared location type should have identical embeddings")

        # Verify matches expected
        expected_embedding = self.expected_location_type_embeddings["Urban commercial district"]
        self.assertTrue(
            torch.allclose(chicago_output.features[0, 0], expected_embedding, atol=1e-5),
            "Location type embedding should match expected")

    def test_location_type_chicago_values(self):
        """Test that Chicago panoramas return correct location types."""
        config = PanoramaLocationTypeExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaLocationTypeExtractor(config, Path(self._temp_dir.name))

        for pano_id, expected_loc in self.chicago_pano_location_types.items():
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}])
            output = extractor(model_input)

            self.assertFalse(output.mask[0, 0].item(),
                           f"Entry should not be masked for {pano_id}")

            returned_loc = decode_sentence_tensor(output.debug["sentences"][0, 0])
            self.assertEqual(returned_loc, expected_loc,
                           f"Location type mismatch for {pano_id}")

            returned_embedding = output.features[0, 0]
            expected_embedding = self.expected_location_type_embeddings[expected_loc]
            self.assertTrue(
                torch.allclose(returned_embedding, expected_embedding, atol=1e-5),
                f"Embedding mismatch for '{expected_loc}' in {pano_id}")

    # ==================== Proper Noun Tests ====================

    def test_proper_noun_file_loading(self):
        """Test loading proper noun embeddings from multi-city v2.0 pickle structure."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))
        extractor.load_files()
        self.assertTrue(extractor.files_loaded)

    def test_proper_noun_forward_basic(self):
        """Test forward pass with panoramas from multiple cities."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        batch_pano_ids = self.chicago_pano_ids[:2] + self.seattle_pano_ids[:1]
        batch_size = len(batch_pano_ids)

        model_input = ModelInput(
            image=torch.zeros((batch_size, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in batch_pano_ids])

        output = extractor(model_input)

        self.assertEqual(output.features.shape[0], batch_size)
        self.assertEqual(output.features.shape[2], config.openai_embedding_size)
        self.assertFalse(output.mask.all())

    def test_proper_noun_returns_correct_embedding(self):
        """Test that returned embeddings match expected values."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        # Test Seattle panorama with Seattle-only proper nouns
        pano_id = "pano_sea_0"
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": pano_id}])

        output = extractor(model_input)

        # Verify we have at least one unmasked entry
        self.assertTrue(output.mask.shape[1] > 0, "Should have at least one proper noun")
        unmasked_count = (~output.mask[0]).sum().item()
        self.assertGreater(unmasked_count, 0, "Should have at least one unmasked proper noun")

        for j in range(output.mask.shape[1]):
            if not output.mask[0, j].item():
                sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                returned_embedding = output.features[0, j]
                expected_embedding = self.expected_proper_noun_embeddings[sentence]

                self.assertTrue(
                    torch.allclose(returned_embedding, expected_embedding, atol=1e-5),
                    f"Embedding mismatch for '{sentence}'")

    def test_proper_noun_yaw_extraction(self):
        """Test that yaw angles are correctly extracted from bounding boxes.

        Verifies exact yaw values, not just that values are 0 or 1.
        Test fixture for pano_chi_0 has 3 proper noun entries:
        - "Starbucks" with yaws [0, 90]
        - "McDonald's" with yaws [180]
        - "Starbucks" with yaws [270]
        """
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_chi_0"}])

        output = extractor(model_input)

        # pano_chi_0 has 3 proper noun entries (one per proper noun, not per landmark)
        expected_entries = self.chicago_expected_landmarks["pano_chi_0"]
        self.assertEqual(output.features.shape[1], len(expected_entries),
                        f"Should have {len(expected_entries)} proper noun entries")

        for j in range(len(expected_entries)):
            self.assertFalse(output.mask[0, j].item(), f"Entry {j} should not be masked")

            # Get the actual proper noun from debug to match with expected
            actual_noun = decode_sentence_tensor(output.debug["sentences"][0, j])
            expected_noun = expected_entries[j]["proper_noun"]
            self.assertEqual(actual_noun, expected_noun,
                           f"Entry {j} should be '{expected_noun}', got '{actual_noun}'")

            expected_yaws = expected_entries[j]["yaws"]
            pos = output.positions[0, j]

            # Position format: [2, 2] where:
            # pos[0, 0] = yaw 0 present, pos[0, 1] = yaw 90 present
            # pos[1, 0] = yaw 180 present, pos[1, 1] = yaw 270 present
            self.assertEqual(pos[0, 0].item(), 1.0 if 0 in expected_yaws else 0.0,
                           f"Entry {j} ({expected_noun}) yaw 0 mismatch")
            self.assertEqual(pos[0, 1].item(), 1.0 if 90 in expected_yaws else 0.0,
                           f"Entry {j} ({expected_noun}) yaw 90 mismatch")
            self.assertEqual(pos[1, 0].item(), 1.0 if 180 in expected_yaws else 0.0,
                           f"Entry {j} ({expected_noun}) yaw 180 mismatch")
            self.assertEqual(pos[1, 1].item(), 1.0 if 270 in expected_yaws else 0.0,
                           f"Entry {j} ({expected_noun}) yaw 270 mismatch")

    def test_proper_noun_shared_across_cities(self):
        """Regression test: same proper noun in different cities returns same embedding.

        "Starbucks" appears in both Chicago and Seattle with different indices in each
        city's tensor. The dict-based approach should return the same embedding.
        """
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        # Get Starbucks embedding from Chicago (pano_chi_0, landmark 0)
        chicago_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_chi_0"}])
        chicago_output = extractor(chicago_input)

        # Find "Starbucks" in Chicago output
        chicago_starbucks_embedding = None
        for j in range(chicago_output.mask.shape[1]):
            if not chicago_output.mask[0, j].item():
                sentence = decode_sentence_tensor(chicago_output.debug["sentences"][0, j])
                if sentence == "Starbucks":
                    chicago_starbucks_embedding = chicago_output.features[0, j]
                    break
        self.assertIsNotNone(chicago_starbucks_embedding, "Should find Starbucks in Chicago")

        # Get Starbucks embedding from Seattle (pano_sea_0, landmark 1)
        seattle_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_sea_0"}])
        seattle_output = extractor(seattle_input)

        # Find "Starbucks" in Seattle output
        seattle_starbucks_embedding = None
        for j in range(seattle_output.mask.shape[1]):
            if not seattle_output.mask[0, j].item():
                sentence = decode_sentence_tensor(seattle_output.debug["sentences"][0, j])
                if sentence == "Starbucks":
                    seattle_starbucks_embedding = seattle_output.features[0, j]
                    break
        self.assertIsNotNone(seattle_starbucks_embedding, "Should find Starbucks in Seattle")

        # Both should return the same embedding
        self.assertTrue(
            torch.allclose(chicago_starbucks_embedding, seattle_starbucks_embedding, atol=1e-5),
            "Starbucks embedding should be identical across cities")

        # And it should match expected
        expected_embedding = self.expected_proper_noun_embeddings["Starbucks"]
        self.assertTrue(
            torch.allclose(chicago_starbucks_embedding, expected_embedding, atol=1e-5),
            "Starbucks embedding should match expected")

    def test_proper_noun_duplicate_within_city(self):
        """Test that duplicate proper noun within a panorama returns same embedding.

        pano_chi_0 has "Starbucks" at landmarks 0 and 2. Both should have same embedding.
        """
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_chi_0"}])
        output = extractor(model_input)

        # Find all Starbucks entries
        starbucks_embeddings = []
        for j in range(output.mask.shape[1]):
            if not output.mask[0, j].item():
                sentence = decode_sentence_tensor(output.debug["sentences"][0, j])
                if sentence == "Starbucks":
                    starbucks_embeddings.append(output.features[0, j])

        # pano_chi_0 has Starbucks at landmarks 0 and 2
        self.assertEqual(len(starbucks_embeddings), 2,
                        "Should have 2 Starbucks landmarks in pano_chi_0")

        # Both should be identical
        self.assertTrue(
            torch.allclose(starbucks_embeddings[0], starbucks_embeddings[1], atol=1e-5),
            "Duplicate Starbucks should have identical embeddings")

    def test_proper_noun_varied_landmark_counts(self):
        """Test that panoramas with different landmark counts work correctly.

        Test fixtures have:
        - pano_chi_0: 3 landmarks
        - pano_chi_1: 4 landmarks
        - pano_chi_2: 2 landmarks
        """
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        for pano_id in self.chicago_pano_ids:
            model_input = ModelInput(
                image=torch.zeros((1, 3, 256, 512)),
                metadata=[{"pano_id": pano_id}])
            output = extractor(model_input)

            expected_count = len(self.chicago_expected_landmarks[pano_id])
            unmasked_count = (~output.mask[0]).sum().item()
            self.assertEqual(unmasked_count, expected_count,
                           f"{pano_id} should have {expected_count} proper noun landmarks")

    # ==================== Common Tests ====================

    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized after cropping."""
        smaller_size = 512
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=smaller_size,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": pid} for pid in self.chicago_pano_ids[:2]])

        output = extractor(model_input)

        self.assertEqual(output.features.shape[2], smaller_size)

        # Verify at least one entry is checked
        checked_count = 0
        for b in range(output.features.shape[0]):
            for i in range(output.features.shape[1]):
                if not output.mask[b, i]:
                    norm = torch.norm(output.features[b, i])
                    self.assertAlmostEqual(norm.item(), 1.0, places=5)
                    checked_count += 1
        self.assertGreater(checked_count, 0, "Should have checked at least one embedding")

    def test_debug_sentences_encoding(self):
        """Test that debug sentences are properly encoded as UTF-8."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": self.chicago_pano_ids[0]}])

        output = extractor(model_input)

        self.assertIn("sentences", output.debug)

        # Verify at least one unmasked entry exists
        self.assertTrue(output.features.shape[1] > 0, "Should have at least one proper noun")
        self.assertFalse(output.mask[0, 0].item(), "First entry should not be masked")

        sentence_bytes = output.debug["sentences"][0, 0].numpy()
        sentence_bytes = sentence_bytes[sentence_bytes > 0]
        decoded = bytes(sentence_bytes).decode('utf-8')
        self.assertGreater(len(decoded), 0)

    def test_error_on_missing_pano_id(self):
        """Test that extractor raises error when pano_id is missing."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"lat": 41.85, "lon": -87.65}])

        with self.assertRaises(ValueError) as context:
            extractor(model_input)
        self.assertIn("pano_id", str(context.exception))

    def test_empty_batch_handling(self):
        """Test that extractor handles panoramas with no proper nouns."""
        config = PanoramaProperNounExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_v2",
            auxiliary_info_key="test_key")
        extractor = ape.PanoramaProperNounExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "nonexistent_pano"}])

        output = extractor(model_input)

        self.assertEqual(output.features.shape[1], 0)
        self.assertTrue(output.mask.shape[1] == 0 or output.mask.all())


if __name__ == "__main__":
    unittest.main()
