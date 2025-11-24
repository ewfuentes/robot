"""Tests for trainable sentence embedding functionality."""

import unittest
import tempfile
import json
from pathlib import Path

import common.torch.load_torch_deps
import torch
import shapely

from experimental.overhead_matching.swag.model.trainable_sentence_embedder import (
    TrainableSentenceEmbedder
)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    SemanticLandmarkExtractor, ModelInput, prune_landmark, custom_id_from_props
)
from experimental.overhead_matching.swag.model.panorama_semantic_landmark_extractor import (
    PanoramaSemanticLandmarkExtractor
)
from experimental.overhead_matching.swag.model.swag_config_types import (
    SemanticLandmarkExtractorConfig,
    PanoramaSemanticLandmarkExtractorConfig,
    TrainableSentenceEmbedderConfig,
    LandmarkType
)


class TrainableSentenceEmbedderTest(unittest.TestCase):
    """Tests for the TrainableSentenceEmbedder model."""

    def test_model_creation(self):
        """Test that the model can be created with a HuggingFace model."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",  # Small model for testing
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        self.assertEqual(embedder.output_dim, 128)
        self.assertEqual(embedder.max_sequence_length, 64)

        # Check that transformer weights are frozen
        for param in embedder.transformer.parameters():
            self.assertFalse(param.requires_grad)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=256,
            freeze_weights=True,
            max_sequence_length=64,
        )

        texts = ["This is a test.", "Another test sentence.", "Third one."]
        embeddings = embedder(texts)

        self.assertEqual(embeddings.shape, (3, 256))

    def test_embeddings_are_normalized(self):
        """Test that output embeddings are normalized to unit length."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        texts = ["Test sentence for normalization check."]
        embeddings = embedder(texts)

        # Check that each embedding has unit norm
        norms = torch.norm(embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_empty_input(self):
        """Test handling of empty input list."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        texts = []
        embeddings = embedder(texts)

        self.assertEqual(embeddings.shape, (0, 128))

    def test_freeze_unfreeze(self):
        """Test freeze and unfreeze functionality."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        # Initially frozen
        for param in embedder.transformer.parameters():
            self.assertFalse(param.requires_grad)

        # Unfreeze
        embedder.unfreeze()
        for param in embedder.transformer.parameters():
            self.assertTrue(param.requires_grad)

        # Freeze again
        embedder.freeze()
        for param in embedder.transformer.parameters():
            self.assertFalse(param.requires_grad)

    def test_batch_processing(self):
        """Test that batching produces consistent results."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )
        embedder.eval()  # Ensure deterministic behavior

        text = "Test sentence for batch consistency."

        # Embed individually
        single_embedding = embedder([text])

        # Embed in batch
        batch_embeddings = embedder([text, text, text])

        # All should be identical
        self.assertTrue(torch.allclose(single_embedding, batch_embeddings[0], atol=1e-5))
        self.assertTrue(torch.allclose(single_embedding, batch_embeddings[1], atol=1e-5))
        self.assertTrue(torch.allclose(single_embedding, batch_embeddings[2], atol=1e-5))

    def test_different_lengths(self):
        """Test handling of sentences with different lengths."""
        embedder = TrainableSentenceEmbedder(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=128,
        )

        texts = [
            "Short.",
            "A medium length sentence with some more words.",
            "A very long sentence that contains many words and should test the tokenizer's ability to handle longer sequences properly."
        ]
        embeddings = embedder(texts)

        # All should have the same output dimension
        self.assertEqual(embeddings.shape, (3, 128))

        # All should be normalized
        norms = torch.norm(embeddings, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class SemanticLandmarkExtractorWithTrainableEmbedderTest(unittest.TestCase):
    """Tests for SemanticLandmarkExtractor with trainable sentence embedder."""

    @classmethod
    def setUpClass(cls):
        """Create test data directories with sentences."""
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_dir = Path(cls._temp_dir.name) / "test_version"

        # Create sentence directory with test data
        sentence_dir = base_dir / "sentences"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        # Define test landmarks
        cls.test_landmarks = {
            'restaurant': {'name': 'Pizza Palace', 'amenity': 'restaurant', 'cuisine': 'italian'},
            'bus_stop': {'name': 'Main St & 5th Ave', 'highway': 'bus_stop'},
            'bank': {'name': 'First National Bank', 'amenity': 'bank', 'building': 'yes'},
        }

        # Create sentences for each landmark
        cls.landmark_sentences = {}
        for key, props in cls.test_landmarks.items():
            pruned_props = prune_landmark(props)
            custom_id = custom_id_from_props(pruned_props)

            # Create natural language sentence
            if key == 'restaurant':
                sentence = "A pizza restaurant serving Italian cuisine."
            elif key == 'bus_stop':
                sentence = "A public bus stop on Main Street."
            elif key == 'bank':
                sentence = "A bank building with financial services."

            cls.landmark_sentences[key] = {
                'custom_id': custom_id,
                'pruned_props': pruned_props,
                'sentence': sentence
            }

        # Write sentences to JSONL with correct OpenAI batch API format
        with open(sentence_dir / "test.jsonl", "w") as f:
            for info in cls.landmark_sentences.values():
                entry = {
                    "custom_id": info['custom_id'],
                    "error": None,
                    "response": {
                        "body": {
                            "choices": [{
                                "finish_reason": "stop",
                                "message": {
                                    "content": info['sentence'],
                                    "refusal": None
                                }
                            }],
                            "usage": {
                                "completion_tokens": 10,
                                "prompt_tokens": 10,
                                "total_tokens": 20
                            }
                        }
                    }
                }
                f.write(json.dumps(entry) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_model_creation_with_trainable_embedder(self):
        """Test that model is created correctly with trainable embedder config."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        config = SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,  # Not used with trainable embedder
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            trainable_embedder_config=embedder_config,
            osm_input_mode="natural_language",
        )

        model = SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Check that trainable embedder was created
        self.assertIsNotNone(model.trainable_embedder)
        self.assertEqual(model.output_dim, 128)

    def test_forward_pass_natural_language_mode(self):
        """Test forward pass with natural language mode."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        config = SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            trainable_embedder_config=embedder_config,
            osm_input_mode="natural_language",
        )

        model = SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create test input with landmarks
        restaurant_props = self.test_landmarks['restaurant'].copy()
        geom = shapely.Point(100.0, 200.0)
        restaurant_props['geometry'] = geom
        restaurant_props['geometry_px'] = geom
        restaurant_props['pruned_props'] = prune_landmark(restaurant_props)

        bank_props = self.test_landmarks['bank'].copy()
        geom2 = shapely.Point(150.0, 250.0)
        bank_props['geometry'] = geom2
        bank_props['geometry_px'] = geom2
        bank_props['pruned_props'] = prune_landmark(bank_props)

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 256)),
            metadata=[{
                'web_mercator_y': 0.0,
                'web_mercator_x': 0.0,
                'landmarks': [restaurant_props, bank_props]
            }],
        )

        # Forward pass
        output = model(model_input)

        # Check output shapes
        self.assertEqual(output.features.shape, (1, 2, 128))
        self.assertEqual(output.mask.shape, (1, 2))
        self.assertEqual(output.positions.shape, (1, 2, 2, 2))

        # Check that landmarks are not masked
        self.assertFalse(output.mask[0, 0].item())
        self.assertFalse(output.mask[0, 1].item())

        # Check that embeddings are normalized
        norms = torch.norm(output.features[0], dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_forward_pass_osm_text_mode(self):
        """Test forward pass with OSM text mode."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=64,
        )

        config = SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            trainable_embedder_config=embedder_config,
            osm_input_mode="osm_text",  # Use raw OSM properties
        )

        model = SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create test input
        restaurant_props = self.test_landmarks['restaurant'].copy()
        geom = shapely.Point(100.0, 200.0)
        restaurant_props['geometry'] = geom
        restaurant_props['geometry_px'] = geom
        restaurant_props['pruned_props'] = prune_landmark(restaurant_props)

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 256)),
            metadata=[{
                'web_mercator_y': 0.0,
                'web_mercator_x': 0.0,
                'landmarks': [restaurant_props]
            }],
        )

        # Forward pass
        output = model(model_input)

        # Check output shapes
        self.assertEqual(output.features.shape, (1, 1, 128))
        self.assertFalse(output.mask[0, 0].item())

    def test_gradient_flow_with_unfrozen_embedder(self):
        """Test that gradients flow when embedder is unfrozen."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=False,  # Unfrozen!
            max_sequence_length=64,
        )

        config = SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            trainable_embedder_config=embedder_config,
            osm_input_mode="natural_language",
        )

        model = SemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.train()  # Set to training mode

        # Create test input
        restaurant_props = self.test_landmarks['restaurant'].copy()
        geom = shapely.Point(100.0, 200.0)
        restaurant_props['geometry'] = geom
        restaurant_props['geometry_px'] = geom
        restaurant_props['pruned_props'] = prune_landmark(restaurant_props)

        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 256)),
            metadata=[{
                'web_mercator_y': 0.0,
                'web_mercator_x': 0.0,
                'landmarks': [restaurant_props]
            }],
        )

        # Forward pass
        output = model(model_input)

        # Compute a simple loss
        loss = output.features.sum()
        loss.backward()

        # Check that gradients exist in the embedder
        has_grad = False
        for param in model.trainable_embedder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients found in unfrozen embedder")

    def test_output_dim_property(self):
        """Test that output_dim property returns correct value."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=256,
            freeze_weights=True,
            max_sequence_length=64,
        )

        config = SemanticLandmarkExtractorConfig(
            landmark_type=LandmarkType.POINT,
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            trainable_embedder_config=embedder_config,
            osm_input_mode="natural_language",
        )

        model = SemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Should return embedder's output_dim, not openai_embedding_size
        self.assertEqual(model.output_dim, 256)


class PanoramaSemanticLandmarkExtractorWithTrainableEmbedderTest(unittest.TestCase):
    """Tests for PanoramaSemanticLandmarkExtractor with trainable sentence embedder."""

    @classmethod
    def setUpClass(cls):
        """Create test data directories with panorama sentences."""
        cls._temp_dir = tempfile.TemporaryDirectory()
        base_dir = Path(cls._temp_dir.name) / "test_version"

        # Create city directory structure
        city_dir = base_dir / "Chicago"
        sentence_dir = city_dir / "sentences"
        sentence_dir.mkdir(parents=True, exist_ok=True)

        embedding_requests_dir = city_dir / "embedding_requests"
        embedding_requests_dir.mkdir(parents=True, exist_ok=True)

        # Create semantic class grouping file
        grouping_data = {
            "semantic_groups": {
                "buildings": ["building", "house"],
            },
            "class_details": {
                "building": {
                    "embedding": {
                        "vector": "AAAAAAAAAAAAAAAAAAAAAA=="  # Base64 dummy
                    }
                },
                "house": {
                    "embedding": {
                        "vector": "AAAAAAAAAAAAAAAAAAAAAA=="
                    }
                }
            }
        }
        with open(base_dir / "semantic_class_grouping.json", "w") as f:
            json.dump(grouping_data, f)

        # Create test panorama landmarks
        cls.test_pano_id = "test_pano_001"
        cls.test_landmarks = [
            {
                'description': 'A tall office building with glass windows.',
                'yaw_angles': [0, 90],
            },
            {
                'description': 'A red brick church with a steeple.',
                'yaw_angles': [180],
            },
        ]

        # Write sentences with correct OpenAI batch API format
        # The content needs to be a JSON string with a "landmarks" array
        with open(sentence_dir / "test.jsonl", "w") as f:
            content_json = json.dumps({"landmarks": cls.test_landmarks})
            entry = {
                "custom_id": cls.test_pano_id,
                "error": None,
                "response": {
                    "body": {
                        "choices": [{
                            "finish_reason": "stop",
                            "message": {
                                "content": content_json,
                                "refusal": None
                            }
                        }],
                        "usage": {
                            "completion_tokens": 10,
                            "prompt_tokens": 10,
                            "total_tokens": 20
                        }
                    }
                }
            }
            f.write(json.dumps(entry) + "\n")

        # Write metadata
        # The custom_id format is generated by make_sentence_dict_from_pano_jsons as {pano_id}__landmark_{idx}
        with open(embedding_requests_dir / "panorama_metadata.jsonl", "w") as f:
            for idx, landmark in enumerate(cls.test_landmarks):
                custom_id = f"{cls.test_pano_id}__landmark_{idx}"
                meta = {
                    "panorama_id": cls.test_pano_id,
                    "landmark_idx": idx,
                    "custom_id": custom_id,
                    "yaw_angles": landmark['yaw_angles'],
                }
                f.write(json.dumps(meta) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()

    def test_model_creation_with_trainable_embedder(self):
        """Test that panorama model is created correctly with trainable embedder."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=128,
        )

        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            should_classify_against_grouping=False,
            trainable_embedder_config=embedder_config,
        )

        model = PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Check that trainable embedder was created
        self.assertIsNotNone(model.trainable_embedder)
        self.assertEqual(model.output_dim, 128)

    def test_forward_pass_with_trainable_embedder(self):
        """Test forward pass with trainable embedder."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=True,
            max_sequence_length=128,
        )

        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            should_classify_against_grouping=False,
            trainable_embedder_config=embedder_config,
        )

        model = PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Create test input
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{
                'pano_id': self.test_pano_id,
            }],
        )

        # Forward pass
        output = model(model_input)

        # Check output shapes
        num_landmarks = len(self.test_landmarks)
        self.assertEqual(output.features.shape, (1, num_landmarks, 128))
        self.assertEqual(output.mask.shape, (1, num_landmarks))
        self.assertEqual(output.positions.shape, (1, num_landmarks, 2, 2))

        # Check that landmarks are not masked
        for i in range(num_landmarks):
            self.assertFalse(output.mask[0, i].item())

        # Check that embeddings are normalized
        norms = torch.norm(output.features[0], dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_output_dim_with_trainable_embedder(self):
        """Test that output_dim is correct with trainable embedder."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=256,
            freeze_weights=True,
            max_sequence_length=128,
        )

        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            should_classify_against_grouping=False,
            trainable_embedder_config=embedder_config,
        )

        model = PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))

        # Should return embedder's output_dim
        self.assertEqual(model.output_dim, 256)

    def test_gradient_flow_unfrozen(self):
        """Test gradient flow with unfrozen embedder."""
        embedder_config = TrainableSentenceEmbedderConfig(
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            output_dim=128,
            freeze_weights=False,  # Unfrozen!
            max_sequence_length=128,
        )

        config = PanoramaSemanticLandmarkExtractorConfig(
            openai_embedding_size=1536,
            embedding_version="test_version",
            auxiliary_info_key="test_key",
            should_classify_against_grouping=False,
            trainable_embedder_config=embedder_config,
        )

        model = PanoramaSemanticLandmarkExtractor(config, Path(self._temp_dir.name))
        model.train()

        # Create test input
        model_input = ModelInput(
            image=torch.zeros((1, 3, 256, 512)),
            metadata=[{
                'pano_id': self.test_pano_id,
            }],
        )

        # Forward pass
        output = model(model_input)

        # Compute loss and backprop
        loss = output.features.sum()
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.trainable_embedder.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients found in unfrozen embedder")


if __name__ == "__main__":
    unittest.main()
