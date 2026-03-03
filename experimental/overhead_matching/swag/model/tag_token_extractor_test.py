
import unittest
import tempfile
import pickle
import hashlib

import common.torch.load_torch_deps
import torch
from pathlib import Path
import numpy as np

from experimental.overhead_matching.swag.model.tag_token_extractor import (
    CharNgramHasher,
    OSMTagTokenExtractor,
    PanoTagTokenExtractor,
    _load_key_vocabulary,
)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    load_v2_tags_pickle,
)
from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMTagTokenExtractorConfig,
    PanoTagTokenExtractorConfig,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props


def create_random_embedding_vector(category: str, dim: int = 1536) -> torch.Tensor:
    """Create a deterministic random vector from a category string."""
    bits = hashlib.sha512(category.encode('utf-8')).digest() * 3
    bits = np.unpackbits(np.frombuffer(bits, dtype=np.uint8))[:dim]
    scaled_bits = (2.0 * bits - 1.0).astype(dtype=np.float32)
    scaled_bits = scaled_bits / np.linalg.norm(scaled_bits)
    return torch.tensor(scaled_bits)


def create_v2_tags_pickle_data(
    panoramas: dict,
    descriptions: list[tuple[str, str]],  # list of (custom_id, description)
    embedding_dim: int = 1536,
) -> dict:
    """Create a v2.0_tags format pickle data structure."""
    description_embeddings = torch.stack([
        create_random_embedding_vector(desc, embedding_dim) for _, desc in descriptions
    ]) if descriptions else torch.zeros((0, embedding_dim))
    description_id_to_idx = {cid: i for i, (cid, _) in enumerate(descriptions)}

    return {
        "version": "2.0_tags",
        "embedding_model": "gemini-embedding-001",
        "embedding_dim": embedding_dim,
        "task_type": "SEMANTIC_SIMILARITY",
        "panoramas": panoramas,
        "description_embeddings": description_embeddings,
        "description_id_to_idx": description_id_to_idx,
    }


def create_flat_osm_embeddings(
    custom_ids: list[str],
    embedding_dim: int = 1536,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Create flat tuple pickle format for OSM embeddings."""
    embeddings = torch.stack([
        create_random_embedding_vector(cid, embedding_dim) for cid in custom_ids
    ]) if custom_ids else torch.zeros((0, embedding_dim))
    id_to_idx = {cid: i for i, cid in enumerate(custom_ids)}
    return embeddings, id_to_idx


# ======== Key vocabulary for tests ========
TEST_KEY_VOCABULARY = [
    "amenity", "shop", "building", "tourism", "leisure",
    "highway", "man_made", "historic", "natural", "office",
    "craft", "railway", "power", "landuse", "emergency",
    "public_transport", "name", "brand", "cuisine", "building:levels",
]


class CharNgramHasherTest(unittest.TestCase):

    def test_basic_output_shape(self):
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        result = hasher(["hello", "world", "cafe"])
        self.assertEqual(result.shape, (3, 32))

    def test_empty_input(self):
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        result = hasher([])
        self.assertEqual(result.shape, (0, 32))

    def test_single_char(self):
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        result = hasher(["a"])
        self.assertEqual(result.shape, (1, 32))

    def test_deterministic(self):
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        r1 = hasher(["cafe"])
        r2 = hasher(["cafe"])
        self.assertTrue(torch.allclose(r1, r2))

    def test_different_strings_differ(self):
        hasher = CharNgramHasher(num_buckets=10000, embedding_dim=64)
        r1 = hasher(["starbucks"])
        r2 = hasher(["mcdonalds"])
        self.assertFalse(torch.allclose(r1, r2))

    def test_case_insensitivity(self):
        """'CAFE' and 'cafe' should produce identical embeddings."""
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        upper = hasher(["CAFE"])
        lower = hasher(["cafe"])
        self.assertTrue(torch.allclose(upper, lower))

    def test_empty_string(self):
        """hasher(['']) returns shape (1, D) without error."""
        hasher = CharNgramHasher(num_buckets=1000, embedding_dim=32)
        result = hasher([""])
        self.assertEqual(result.shape, (1, 32))


class LoadV2TagsPickleTest(unittest.TestCase):

    def test_loads_valid_pickle(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            data = {"version": "2.0_tags", "panoramas": {}}
            pickle.dump(data, f)
            f.flush()
            result = load_v2_tags_pickle(Path(f.name))
            self.assertIsNotNone(result)
            self.assertEqual(result["version"], "2.0_tags")

    def test_returns_none_for_missing_file(self):
        result = load_v2_tags_pickle(Path("/tmp/nonexistent_test_file.pkl"))
        self.assertIsNone(result)

    def test_raises_on_wrong_version(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            data = {"version": "2.0", "panoramas": {}}
            pickle.dump(data, f)
            f.flush()
            with self.assertRaises(RuntimeError):
                load_v2_tags_pickle(Path(f.name))


class TagPickleEmbeddingAssociationTest(unittest.TestCase):
    """Test correct embedding-to-landmark association in 2.0_tags pickle format.

    Creates multi-city synthetic pickles and verifies correct ID associations.
    """

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)
        version = "test_tags_v2"

        # === City 1: Chicago ===
        chicago_panoramas = {
            "pano_chi_0,41.85,-87.65,": {
                "location_type": "urban_commercial",
                "landmarks": [
                    {
                        "primary_tag": {"key": "amenity", "value": "cafe"},
                        "additional_tags": [
                            {"key": "name", "value": "Starbucks"},
                            {"key": "brand", "value": "Starbucks"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "0", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        ],
                        "description": "A Starbucks cafe on the corner",
                        "landmark_idx": 0,
                    },
                    {
                        "primary_tag": {"key": "shop", "value": "pharmacy"},
                        "additional_tags": [
                            {"key": "name", "value": "CVS"},
                        ],
                        "confidence": "medium",
                        "bounding_boxes": [
                            {"yaw_angle": "90", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        ],
                        "description": "A CVS pharmacy",
                        "landmark_idx": 1,
                    },
                ],
            },
            "pano_chi_1,41.86,-87.64,": {
                "location_type": "suburban",
                "landmarks": [
                    {
                        "primary_tag": {"key": "building", "value": "apartments"},
                        "additional_tags": [
                            {"key": "building:levels", "value": "4"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "180", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                            {"yaw_angle": "270", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                        ],
                        "description": "A four-story apartment building",
                        "landmark_idx": 0,
                    },
                ],
            },
        }

        chicago_descriptions = []
        for pano_id, pano_data in chicago_panoramas.items():
            for lm in pano_data["landmarks"]:
                custom_id = f"{pano_id}__landmark_{lm['landmark_idx']}"
                chicago_descriptions.append((custom_id, lm["description"]))

        chicago_dir = base_path / version / "1_Chicago" / "embeddings"
        chicago_dir.mkdir(parents=True, exist_ok=True)
        chicago_data = create_v2_tags_pickle_data(chicago_panoramas, chicago_descriptions)
        with open(chicago_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(chicago_data, f)

        cls.chicago_descriptions = chicago_descriptions
        cls.chicago_panoramas = chicago_panoramas

        # === City 2: Seattle ===
        seattle_panoramas = {
            "pano_sea_0,47.60,-122.33,": {
                "location_type": "urban_waterfront",
                "landmarks": [
                    {
                        "primary_tag": {"key": "amenity", "value": "restaurant"},
                        "additional_tags": [
                            {"key": "name", "value": "Pike Place Chowder"},
                            {"key": "cuisine", "value": "seafood"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "0", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                            {"yaw_angle": "90", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                        ],
                        "description": "Pike Place Chowder seafood restaurant",
                        "landmark_idx": 0,
                    },
                    {
                        "primary_tag": {"key": "man_made", "value": "tower"},
                        "additional_tags": [
                            {"key": "name", "value": "Space Needle"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "180", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                        ],
                        "description": "The Space Needle observation tower",
                        "landmark_idx": 1,
                    },
                ],
            },
        }

        seattle_descriptions = []
        for pano_id, pano_data in seattle_panoramas.items():
            for lm in pano_data["landmarks"]:
                custom_id = f"{pano_id}__landmark_{lm['landmark_idx']}"
                seattle_descriptions.append((custom_id, lm["description"]))

        seattle_dir = base_path / version / "2_Seattle" / "embeddings"
        seattle_dir.mkdir(parents=True, exist_ok=True)
        seattle_data = create_v2_tags_pickle_data(seattle_panoramas, seattle_descriptions)
        with open(seattle_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(seattle_data, f)

        cls.seattle_descriptions = seattle_descriptions
        cls.seattle_panoramas = seattle_panoramas
        cls.version = version

        # Store all expected embeddings
        cls.expected_desc_embeddings = {}
        for cid, desc in chicago_descriptions + seattle_descriptions:
            cls.expected_desc_embeddings[cid] = create_random_embedding_vector(desc, 1536)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_description_id_to_idx_correct(self):
        """Verify description_embeddings[description_id_to_idx[custom_id]] returns correct embedding."""
        base_path = Path(self._temp_dir.name) / self.version

        for city_dir in sorted(base_path.iterdir()):
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            data = load_v2_tags_pickle(pickle_path)
            self.assertIsNotNone(data)

            tensor = data["description_embeddings"]
            id_to_idx = data["description_id_to_idx"]

            for custom_id, idx in id_to_idx.items():
                embedding = tensor[idx]
                expected = self.expected_desc_embeddings[custom_id]
                self.assertTrue(
                    torch.allclose(embedding, expected, atol=1e-5),
                    f"Embedding mismatch for {custom_id}")

    def test_panorama_tag_structure_preserved(self):
        """Verify panoramas dict preserves full tag structure."""
        base_path = Path(self._temp_dir.name) / self.version
        chicago_pickle = base_path / "1_Chicago" / "embeddings" / "embeddings.pkl"
        data = load_v2_tags_pickle(chicago_pickle)

        pano_data = data["panoramas"]["pano_chi_0,41.85,-87.65,"]
        lm0 = pano_data["landmarks"][0]

        self.assertEqual(lm0["primary_tag"]["key"], "amenity")
        self.assertEqual(lm0["primary_tag"]["value"], "cafe")
        self.assertEqual(len(lm0["additional_tags"]), 2)
        self.assertEqual(lm0["additional_tags"][0]["key"], "name")
        self.assertEqual(lm0["additional_tags"][0]["value"], "Starbucks")
        self.assertEqual(lm0["confidence"], "high")

    def test_version_tag(self):
        """Verify version is 2.0_tags."""
        base_path = Path(self._temp_dir.name) / self.version
        for city_dir in base_path.iterdir():
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            data = load_v2_tags_pickle(pickle_path)
            self.assertEqual(data["version"], "2.0_tags")


class OSMFlatPickleAssociationTest(unittest.TestCase):
    """Test correct embedding lookup for OSM flat tuple pickle format."""

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)

        # Create fake pruned_props and their custom_ids
        cls.test_props = [
            frozenset([("amenity", "cafe"), ("name", "Starbucks")]),
            frozenset([("shop", "pharmacy"), ("name", "CVS")]),
            frozenset([("building", "yes")]),
        ]
        cls.custom_ids = [custom_id_from_props(p) for p in cls.test_props]

        # Create flat tuple pickle
        embeddings, id_to_idx = create_flat_osm_embeddings(cls.custom_ids)
        emb_dir = base_path / "test_osm" / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        with open(emb_dir / "embeddings.pkl", "wb") as f:
            pickle.dump((embeddings, id_to_idx), f)

        cls.expected_embeddings = {
            cid: create_random_embedding_vector(cid, 1536)
            for cid in cls.custom_ids
        }

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_custom_id_lookup_returns_correct_embedding(self):
        """Verify custom_id_from_props lookup returns correct embedding."""
        emb_path = Path(self._temp_dir.name) / "test_osm" / "embeddings" / "embeddings.pkl"
        with open(emb_path, "rb") as f:
            tensor, id_to_idx = pickle.load(f)

        for props, expected_cid in zip(self.test_props, self.custom_ids):
            cid = custom_id_from_props(props)
            self.assertEqual(cid, expected_cid)
            self.assertIn(cid, id_to_idx)

            embedding = tensor[id_to_idx[cid]]
            expected = self.expected_embeddings[cid]
            self.assertTrue(
                torch.allclose(embedding, expected, atol=1e-5),
                f"Embedding mismatch for props {dict(props)}")


class PanoTagTokenExtractorTest(unittest.TestCase):
    """Smoke tests for PanoTagTokenExtractor output shape contract."""

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)
        cls.version = "test_pano_tags"

        # Write key vocabulary file
        cls.vocab_file = base_path / "tag_key_vocabulary.txt"
        cls.vocab_file.write_text("\n".join(TEST_KEY_VOCABULARY))

        # Create v2.0_tags pickle with synthetic data
        panoramas = {
            "pano_a,41.85,-87.65,": {
                "location_type": "urban_commercial",
                "landmarks": [
                    {
                        "primary_tag": {"key": "amenity", "value": "cafe"},
                        "additional_tags": [
                            {"key": "name", "value": "Starbucks"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "0", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        ],
                        "description": "A Starbucks cafe",
                        "landmark_idx": 0,
                    },
                    {
                        "primary_tag": {"key": "shop", "value": "pharmacy"},
                        "additional_tags": [],
                        "confidence": "medium",
                        "bounding_boxes": [
                            {"yaw_angle": "90", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
                        ],
                        "description": "A pharmacy",
                        "landmark_idx": 1,
                    },
                ],
            },
            "pano_b,41.86,-87.64,": {
                "location_type": "suburban",
                "landmarks": [
                    {
                        "primary_tag": {"key": "building", "value": "apartments"},
                        "additional_tags": [
                            {"key": "building:levels", "value": "4"},
                        ],
                        "confidence": "high",
                        "bounding_boxes": [
                            {"yaw_angle": "180", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                            {"yaw_angle": "270", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100},
                        ],
                        "description": "A four-story building",
                        "landmark_idx": 0,
                    },
                ],
            },
        }

        descriptions = []
        for pano_id, pano_data in panoramas.items():
            for lm in pano_data["landmarks"]:
                custom_id = f"{pano_id}__landmark_{lm['landmark_idx']}"
                descriptions.append((custom_id, lm["description"]))

        city_dir = base_path / cls.version / "1_TestCity" / "embeddings"
        city_dir.mkdir(parents=True, exist_ok=True)
        pickle_data = create_v2_tags_pickle_data(panoramas, descriptions)
        with open(city_dir / "embeddings.pkl", "wb") as f:
            pickle.dump(pickle_data, f)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def _make_config(self, include_descs=True):
        return PanoTagTokenExtractorConfig(
            token_dim=64,
            key_embedding_dim=32,
            value_embedding_dim=32,
            ngram_bucket_size=1000,
            key_vocabulary_file=str(self.vocab_file),
            auxiliary_info_key="test",
            embedding_version=self.version,
            include_description_embeddings=include_descs,
            description_embedding_dim=1536,
            max_landmarks=30,
        )

    def test_forward_output_shape(self):
        """Test output follows ExtractorOutput contract: (B, T, D), (B, T, 2, 2), (B, T)."""
        config = self._make_config()
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": "pano_a"}, {"pano_id": "pano_b"}],
        )
        output = extractor(model_input)

        B = 2
        T = output.features.shape[1]
        D = config.token_dim

        self.assertEqual(output.features.shape[0], B)
        self.assertEqual(output.features.shape[2], D)
        self.assertEqual(output.positions.shape, (B, T, 2, 2))
        self.assertEqual(output.mask.shape, (B, T))
        self.assertTrue(T > 0, "Should produce at least one token")

    def test_forward_without_description_embeddings(self):
        """Test that disabling description embeddings still works."""
        config = self._make_config(include_descs=False)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_a"}],
        )
        output = extractor(model_input)

        self.assertEqual(output.features.shape[0], 1)
        self.assertEqual(output.features.shape[2], config.token_dim)
        # pano_a has 2 landmarks: first has amenity+name=2 tags, second has shop=1 tag = 3 tokens
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 3, "Should have 3 tag tokens (no desc tokens)")

    def test_missing_pano_id_graceful(self):
        """Test that missing pano_id produces all-masked output."""
        config = self._make_config()
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "nonexistent_pano"}],
        )
        output = extractor(model_input)
        # Should produce zero tokens
        self.assertEqual(output.features.shape[1], 0)

    def test_unknown_keys_skipped(self):
        """Test that tags with unknown keys are skipped."""
        config = self._make_config(include_descs=False)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        # Hack: add a landmark with an unknown key to the loaded data
        extractor.load_files()
        extractor.panorama_data["pano_unknown"] = {
            "location_type": "test",
            "landmarks": [{
                "primary_tag": {"key": "unknown_key_not_in_vocab", "value": "blah"},
                "additional_tags": [],
                "confidence": "low",
                "bounding_boxes": [],
                "description": "",
                "landmark_idx": 0,
            }],
        }

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_unknown"}],
        )
        output = extractor(model_input)
        # Unknown key should be skipped, no tokens produced
        self.assertEqual(output.features.shape[1], 0)

    def test_raises_without_pano_id(self):
        """Test that extractor raises error when pano_id is missing."""
        config = self._make_config()
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"lat": 41.85}],
        )
        with self.assertRaises(ValueError):
            extractor(model_input)

    def test_properties(self):
        """Test output_dim, num_position_outputs, data_requirements properties."""
        config = self._make_config()
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))
        self.assertEqual(extractor.output_dim, 64)
        self.assertEqual(extractor.num_position_outputs, 2)
        self.assertEqual(extractor.data_requirements, [])

    def test_gradient_flow(self):
        """Forward + backward produces non-zero grad on _tag_projection.weight."""
        config = self._make_config(include_descs=True)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_a"}],
        )
        output = extractor(model_input)
        loss = output.features.sum()
        loss.backward()

        self.assertIsNotNone(extractor._tag_projection.weight.grad)
        self.assertTrue(extractor._tag_projection.weight.grad.abs().sum() > 0)

    def test_batch_padding_correctness(self):
        """Batch of 2 items with different token counts; verify masks and zero-padding."""
        config = self._make_config(include_descs=False)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        # pano_a: 3 tags (amenity + name for lm0, shop for lm1)
        # pano_b: 2 tags (building + building:levels for lm0)
        model_input = ModelInput(
            image=torch.zeros((2, 3, 256, 512)),
            metadata=[{"pano_id": "pano_a"}, {"pano_id": "pano_b"}],
        )
        output = extractor(model_input)

        unmasked_a = (~output.mask[0]).sum().item()
        unmasked_b = (~output.mask[1]).sum().item()
        self.assertEqual(unmasked_a, 3)
        self.assertEqual(unmasked_b, 2)

        # Padded slots in the shorter batch item should be zeros
        T = output.features.shape[1]
        if T > unmasked_b:
            padded_features = output.features[1, unmasked_b:]
            self.assertTrue(torch.all(padded_features == 0))

    def test_pano_id_prefix_collision(self):
        """Two pano IDs where one is prefix of other get correct desc embeddings."""
        config = self._make_config(include_descs=True)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))
        extractor.load_files()

        # Inject two panoramas with identical tag structure
        landmark_template = {
            "primary_tag": {"key": "amenity", "value": "cafe"},
            "additional_tags": [],
            "confidence": "high",
            "bounding_boxes": [
                {"yaw_angle": "0", "ymin": 0, "xmin": 0, "ymax": 100, "xmax": 100}
            ],
            "landmark_idx": 0,
        }
        extractor.panorama_data["pano_1"] = {"landmarks": [dict(landmark_template)]}
        extractor.panorama_data["pano_10"] = {"landmarks": [dict(landmark_template)]}

        # Inject desc embeddings with very different values
        desc_emb_1 = torch.ones(1536)
        desc_emb_10 = torch.ones(1536) * 100
        extractor._desc_embeddings_dict["pano_1,41.0,-87.0,__landmark_0"] = desc_emb_1
        extractor._desc_embeddings_dict["pano_10,42.0,-88.0,__landmark_0"] = desc_emb_10
        extractor._pano_id_to_full_prefix["pano_1"] = "pano_1,41.0,-87.0,"
        extractor._pano_id_to_full_prefix["pano_10"] = "pano_10,42.0,-88.0,"

        out_1 = extractor(ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_1"}],
        ))
        out_10 = extractor(ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_10"}],
        ))

        # Features must differ because desc embeddings differ
        self.assertFalse(
            torch.allclose(out_1.features, out_10.features),
            "pano_1 and pano_10 should have different features due to different descriptions")

    def test_description_folded_into_tags(self):
        """With descriptions enabled, token count equals tag count only (no extra desc tokens)."""
        config = self._make_config(include_descs=True)
        extractor = PanoTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{"pano_id": "pano_a"}],
        )
        output = extractor(model_input)

        # pano_a: 3 tag tokens only (desc folded in, not separate tokens)
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 3, "Should have 3 tags (desc folded in, not separate tokens)")

        # Features should be non-zero (desc embeddings contribute to each tag)
        self.assertTrue(output.features[0, :unmasked].abs().sum() > 0)


class OSMTagTokenExtractorTest(unittest.TestCase):
    """Smoke tests for OSMTagTokenExtractor output shape contract."""

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_path = Path(cls._temp_dir.name)
        cls.version = "test_osm_tags"

        # Write key vocabulary file
        cls.vocab_file = base_path / "tag_key_vocabulary.txt"
        cls.vocab_file.write_text("\n".join(TEST_KEY_VOCABULARY))

        # Create fake pruned_props for landmarks
        cls.landmark_props = [
            frozenset([("amenity", "cafe"), ("name", "Starbucks")]),
            frozenset([("shop", "pharmacy"), ("name", "CVS")]),
        ]
        cls.custom_ids = [custom_id_from_props(p) for p in cls.landmark_props]

        # Create flat tuple pickle for description embeddings
        embeddings, id_to_idx = create_flat_osm_embeddings(cls.custom_ids)
        emb_dir = base_path / cls.version / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        with open(emb_dir / "embeddings.pkl", "wb") as f:
            pickle.dump((embeddings, id_to_idx), f)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def _make_config(self, include_descs=True):
        return OSMTagTokenExtractorConfig(
            token_dim=64,
            key_embedding_dim=32,
            value_embedding_dim=32,
            ngram_bucket_size=1000,
            key_vocabulary_file=str(self.vocab_file),
            auxiliary_info_key="test",
            embedding_version=self.version,
            include_description_embeddings=include_descs,
            description_embedding_dim=1536,
            max_landmarks=30,
        )

    def _make_metadata_with_landmarks(self):
        """Create metadata dicts with pruned_props, mimicking vigor_dataset output."""
        return [
            {
                "landmarks": [
                    {
                        "pruned_props": dict(self.landmark_props[0]),
                        "geometry_px": _FakePoint(41.85, -87.65),
                        "geometry": _FakePoint(41.85, -87.65),
                    },
                    {
                        "pruned_props": dict(self.landmark_props[1]),
                        "geometry_px": _FakePoint(41.86, -87.64),
                        "geometry": _FakePoint(41.86, -87.64),
                    },
                ],
                "web_mercator_y": 41.855,
                "web_mercator_x": -87.645,
            },
        ]

    def test_forward_output_shape(self):
        """Test output follows ExtractorOutput contract."""
        config = self._make_config(include_descs=False)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = self._make_metadata_with_landmarks()
        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)

        B = 1
        T = output.features.shape[1]
        D = config.token_dim

        self.assertEqual(output.features.shape[0], B)
        self.assertEqual(output.features.shape[2], D)
        self.assertEqual(output.positions.shape, (B, T, 2, 2))
        self.assertEqual(output.mask.shape, (B, T))
        self.assertTrue(T > 0, "Should produce at least one token")

    def test_token_count_tags_only(self):
        """Test correct token count without description embeddings."""
        config = self._make_config(include_descs=False)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = self._make_metadata_with_landmarks()
        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)

        # Landmark 0: amenity + name = 2 tags
        # Landmark 1: shop + name = 2 tags
        # Total: 4 tokens
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 4)

    def test_token_count_with_descriptions(self):
        """Test correct token count with description embeddings."""
        config = self._make_config(include_descs=True)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = self._make_metadata_with_landmarks()
        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)

        # 4 tag tokens (descriptions folded into each tag, not separate tokens)
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 4)

    def test_empty_landmarks(self):
        """Test handling of metadata with no landmarks."""
        config = self._make_config(include_descs=False)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=[{"landmarks": [], "web_mercator_y": 0, "web_mercator_x": 0}],
        )
        output = extractor(model_input)
        self.assertEqual(output.features.shape[1], 0)

    def test_properties(self):
        """Test output_dim, num_position_outputs, data_requirements properties."""
        config = self._make_config()
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))
        self.assertEqual(extractor.output_dim, 64)
        self.assertEqual(extractor.num_position_outputs, 2)
        self.assertIn("landmarks", [str(r) for r in extractor.data_requirements])

    def test_gradient_flow(self):
        """Forward + backward produces non-zero grad on _tag_projection.weight."""
        config = self._make_config(include_descs=True)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = self._make_metadata_with_landmarks()
        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)
        loss = output.features.sum()
        loss.backward()

        self.assertIsNotNone(extractor._tag_projection.weight.grad)
        self.assertTrue(extractor._tag_projection.weight.grad.abs().sum() > 0)

    def test_frozenset_props_forward(self):
        """Pass frozenset props through forward, verify same token count as dict props."""
        config = self._make_config(include_descs=False)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        # Run with dict props
        metadata_dict = [{
            "landmarks": [{
                "pruned_props": {"amenity": "cafe", "name": "Starbucks"},
                "geometry_px": _FakePoint(41.85, -87.65),
                "geometry": _FakePoint(41.85, -87.65),
            }],
            "web_mercator_y": 41.855,
            "web_mercator_x": -87.645,
        }]
        out_dict = extractor(ModelInput(
            image=torch.zeros((1, 3, 320, 320)), metadata=metadata_dict))

        # Run with frozenset props (same key-value pairs)
        metadata_frozen = [{
            "landmarks": [{
                "pruned_props": frozenset([("amenity", "cafe"), ("name", "Starbucks")]),
                "geometry_px": _FakePoint(41.85, -87.65),
                "geometry": _FakePoint(41.85, -87.65),
            }],
            "web_mercator_y": 41.855,
            "web_mercator_x": -87.645,
        }]
        out_frozen = extractor(ModelInput(
            image=torch.zeros((1, 3, 320, 320)), metadata=metadata_frozen))

        # Same number of tokens produced
        self.assertEqual(out_dict.features.shape, out_frozen.features.shape)
        self.assertEqual(
            (~out_dict.mask).sum().item(),
            (~out_frozen.mask).sum().item())

    def test_max_landmarks_enforcement(self):
        """Set max_landmarks=1, provide 3 landmarks, verify only first landmark's tags appear."""
        config = OSMTagTokenExtractorConfig(
            token_dim=64,
            key_embedding_dim=32,
            value_embedding_dim=32,
            ngram_bucket_size=1000,
            key_vocabulary_file=str(self.vocab_file),
            auxiliary_info_key="test",
            embedding_version=self.version,
            include_description_embeddings=False,
            description_embedding_dim=1536,
            max_landmarks=1,
        )
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = [{
            "landmarks": [
                {
                    "pruned_props": {"amenity": "cafe", "name": "Starbucks"},
                    "geometry_px": _FakePoint(41.85, -87.65),
                    "geometry": _FakePoint(41.85, -87.65),
                },
                {
                    "pruned_props": {"shop": "pharmacy", "name": "CVS"},
                    "geometry_px": _FakePoint(41.86, -87.64),
                    "geometry": _FakePoint(41.86, -87.64),
                },
                {
                    "pruned_props": {"building": "apartments"},
                    "geometry_px": _FakePoint(41.87, -87.63),
                    "geometry": _FakePoint(41.87, -87.63),
                },
            ],
            "web_mercator_y": 41.855,
            "web_mercator_x": -87.645,
        }]

        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)

        # Only first landmark's tags: amenity + name = 2 tokens
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 2)

    def test_description_folded_into_tags(self):
        """With descriptions enabled, token count equals tag count (no extra desc tokens)."""
        config = self._make_config(include_descs=True)
        extractor = OSMTagTokenExtractor(config, Path(self._temp_dir.name))

        metadata = self._make_metadata_with_landmarks()
        model_input = ModelInput(
            image=torch.zeros((1, 3, 320, 320)),
            metadata=metadata,
        )
        output = extractor(model_input)

        # 4 tag tokens only (desc folded in, not separate tokens)
        unmasked = (~output.mask[0]).sum().item()
        self.assertEqual(unmasked, 4, "Should have 4 tags (desc folded in, not separate tokens)")

        # Features should be non-zero (desc embeddings contribute to each tag)
        self.assertTrue(output.features[0, :unmasked].abs().sum() > 0)


class _FakePoint:
    """Minimal fake geometry for testing OSM extractor positions."""
    geom_type = "Point"

    def __init__(self, y, x):
        self.x = x
        self.y = y


if __name__ == "__main__":
    unittest.main()
