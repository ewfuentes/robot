#!/usr/bin/env python3
"""Tests for OSMSemanticClassExtractor."""
from common.torch import load_torch_deps
import unittest
import torch
import tempfile
import shutil
import json
from pathlib import Path
from shapely.geometry import Point

from experimental.overhead_matching.swag.model.osm_semantic_class_extractor import OSMSemanticClassExtractor
from experimental.overhead_matching.swag.model.swag_config_types import OSMSemanticClassExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput


class TestOSMSemanticClassExtractor(unittest.TestCase):
    """Test cases for OSMSemanticClassExtractor."""

    @classmethod
    def setUpClass(cls):
        """Create a test JSON file and extractor."""
        # Create a minimal semantic class grouping JSON in new format
        test_data = {
            'semantic_groups': {
                'roads': ['footway'],
                'buildings': ['building', 'restaurant_building'],
                'vegetation': ['tree'],
                'transit': ['bus_stop']
            },
            'class_details': {
                'footway': {
                    'osm_tags': {'highway': 'footway'},
                    'embedding': {'model': 'test', 'vector': 'dummy'}
                },
                'building': {
                    'osm_tags': {'building': 'yes'},
                    'embedding': {'model': 'test', 'vector': 'dummy'}
                },
                'tree': {
                    'osm_tags': {'natural': 'tree'},
                    'embedding': {'model': 'test', 'vector': 'dummy'}
                },
                'bus_stop': {
                    'osm_tags': {'highway': 'bus_stop', 'public_transport': 'platform'},
                    'embedding': {'model': 'test', 'vector': 'dummy'}
                },
                'restaurant_building': {
                    'osm_tags': {'building': 'yes', 'amenity': 'restaurant'},
                    'embedding': {'model': 'test', 'vector': 'dummy'}
                }
            }
        }

        # Write to temp file
        cls.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_data, cls.temp_file)
        cls.temp_file.close()

        # Create config and extractor
        cls.config = OSMSemanticClassExtractorConfig(
            auxiliary_info_key='semantic_class_base_path',
            embedding_version='test_semantic'
        )

        # Save temp file with the expected name in temp directory
        cls.temp_dir = tempfile.mkdtemp()
        # Create subdirectory for embedding_version
        semantic_dir = Path(cls.temp_dir) / 'test_semantic'
        semantic_dir.mkdir(parents=True, exist_ok=True)
        cls.json_path = semantic_dir / 'semantic_class_grouping.json'
        shutil.move(cls.temp_file.name, cls.json_path)

        cls.extractor = OSMSemanticClassExtractor(cls.config, cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp file."""
        shutil.rmtree(cls.temp_dir)

    def test_initialization(self):
        """Test that extractor initializes correctly."""
        self.assertEqual(self.extractor.num_classes, 4)
        self.assertEqual(self.extractor.output_dim, 4)
        self.assertEqual(len(self.extractor.mappings), 5)
        print(f"✓ Initialized with {self.extractor.num_classes} classes")

    def test_map_tags_to_class_id(self):
        """Test mapping tags to class IDs."""
        # Note: ontology order matches insertion order: ['roads', 'buildings', 'vegetation', 'transit']

        # Test simple road
        tags = frozenset([('highway', 'footway')])
        class_id = self.extractor.map_tags_to_class_id(tags)
        self.assertIsNotNone(class_id)
        self.assertEqual(class_id, 0)  # 'roads' is first in ontology
        print(f"✓ Footway → class {class_id} (roads)")

        # Test building
        tags = frozenset([('building', 'yes')])
        class_id = self.extractor.map_tags_to_class_id(tags)
        self.assertIsNotNone(class_id)
        self.assertEqual(class_id, 1)  # 'buildings' is second in ontology
        print(f"✓ Building → class {class_id} (buildings)")

        # Test vegetation
        tags = frozenset([('natural', 'tree')])
        class_id = self.extractor.map_tags_to_class_id(tags)
        self.assertIsNotNone(class_id)
        self.assertEqual(class_id, 2)  # 'vegetation' is third in ontology
        print(f"✓ Tree → class {class_id} (vegetation)")

    def test_most_specific_match(self):
        """Test that most specific match is chosen."""
        # Building + restaurant should match the more specific 'restaurant_building'
        tags = frozenset([('building', 'yes'), ('amenity', 'restaurant')])
        class_id = self.extractor.map_tags_to_class_id(tags)
        self.assertIsNotNone(class_id)
        self.assertEqual(class_id, 1)  # 'buildings' is second in ontology
        print(f"✓ Restaurant building → class {class_id} (most specific)")

        # With extra tags
        tags_extra = frozenset([
            ('building', 'yes'),
            ('amenity', 'restaurant'),
            ('cuisine', 'italian')
        ])
        class_id_extra = self.extractor.map_tags_to_class_id(tags_extra)
        self.assertEqual(class_id, class_id_extra)
        print(f"✓ Restaurant + extras → same class {class_id_extra}")

    def test_no_match_returns_none(self):
        """Test that unmatched tags return None."""
        tags = frozenset([('unknown', 'tag')])
        class_id = self.extractor.map_tags_to_class_id(tags)
        self.assertIsNone(class_id)
        print(f"✓ Unknown tags → None")

    def test_forward_single_landmark(self):
        """Test forward pass with a single landmark."""
        # Create model input with one landmark
        metadata = [{
            'web_mercator_x': 100.0,
            'web_mercator_y': 200.0,
            'landmarks': [
                {
                    'highway': 'footway',
                    'geometry_px': Point(105.0, 205.0)
                }
            ]
        }]

        model_input = ModelInput(
            image=torch.zeros(1, 3, 256, 256),
            metadata=metadata
        )

        output = self.extractor.forward(model_input)

        # Check output shapes
        self.assertEqual(output.mask.shape, (1, 1))
        self.assertEqual(output.features.shape, (1, 1, 4))
        self.assertEqual(output.positions.shape, (1, 1, 2, 2))

        # Check mask (should be False for valid landmark)
        self.assertFalse(output.mask[0, 0].item())

        # Check one-hot encoding (should be [1, 0, 0, 0] for roads at index 0)
        expected_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(output.features[0, 0], expected_embedding))
        print(f"✓ Forward pass produces correct one-hot: {output.features[0, 0].tolist()}")

    def test_forward_multiple_landmarks(self):
        """Test forward pass with multiple landmarks."""
        # Ontology order: ['roads', 'buildings', 'vegetation', 'transit']
        metadata = [{
            'web_mercator_x': 100.0,
            'web_mercator_y': 200.0,
            'landmarks': [
                {'highway': 'footway', 'geometry_px': Point(105.0, 205.0)},  # roads (idx 0): [1,0,0,0]
                {'building': 'yes', 'geometry_px': Point(110.0, 210.0)},      # buildings (idx 1): [0,1,0,0]
                {'natural': 'tree', 'geometry_px': Point(115.0, 215.0)},      # vegetation (idx 2): [0,0,1,0]
            ]
        }]

        model_input = ModelInput(
            image=torch.zeros(1, 3, 256, 256),
            metadata=metadata
        )

        output = self.extractor.forward(model_input)

        # Check shapes
        self.assertEqual(output.mask.shape, (1, 3))
        self.assertEqual(output.features.shape, (1, 3, 4))

        # Check all landmarks are unmasked
        self.assertFalse(output.mask[0, 0].item())
        self.assertFalse(output.mask[0, 1].item())
        self.assertFalse(output.mask[0, 2].item())

        # Check embeddings with updated indices
        self.assertTrue(torch.allclose(output.features[0, 0], torch.tensor([1., 0., 0., 0.])))  # roads
        self.assertTrue(torch.allclose(output.features[0, 1], torch.tensor([0., 1., 0., 0.])))  # buildings
        self.assertTrue(torch.allclose(output.features[0, 2], torch.tensor([0., 0., 1., 0.])))  # vegetation
        print(f"✓ Multiple landmarks produce correct one-hot encodings")

    def test_forward_batch(self):
        """Test forward pass with batch size > 1."""
        metadata = [
            {
                'web_mercator_x': 100.0,
                'web_mercator_y': 200.0,
                'landmarks': [
                    {'highway': 'footway', 'geometry_px': Point(105.0, 205.0)},
                ]
            },
            {
                'web_mercator_x': 150.0,
                'web_mercator_y': 250.0,
                'landmarks': [
                    {'building': 'yes', 'geometry_px': Point(155.0, 255.0)},
                    {'natural': 'tree', 'geometry_px': Point(160.0, 260.0)},
                ]
            }
        ]

        model_input = ModelInput(
            image=torch.zeros(2, 3, 256, 256),
            metadata=metadata
        )

        output = self.extractor.forward(model_input)

        # Check shapes (batch=2, max_landmarks=2)
        self.assertEqual(output.mask.shape, (2, 2))
        self.assertEqual(output.features.shape, (2, 2, 4))

        # First batch item: 1 landmark, second is masked
        self.assertFalse(output.mask[0, 0].item())
        self.assertTrue(output.mask[0, 1].item())

        # Second batch item: 2 landmarks, both valid
        self.assertFalse(output.mask[1, 0].item())
        self.assertFalse(output.mask[1, 1].item())

        print(f"✓ Batch processing works correctly")

    def test_forward_no_match_landmark(self):
        """Test forward pass with landmark that doesn't match."""
        metadata = [{
            'web_mercator_x': 100.0,
            'web_mercator_y': 200.0,
            'landmarks': [
                {'unknown': 'tag', 'geometry_px': Point(105.0, 205.0)},
            ]
        }]

        model_input = ModelInput(
            image=torch.zeros(1, 3, 256, 256),
            metadata=metadata
        )

        output = self.extractor.forward(model_input)

        # Landmark should be unmasked and have all zeros embedding (no match)
        self.assertFalse(output.mask[0, 0].item())
        self.assertTrue(torch.allclose(output.features[0, 0], torch.zeros(4)))
        print(f"✓ Unmatched landmark produces all-zeros vector")

    def test_data_requirements(self):
        """Test that data requirements are correct."""
        from experimental.overhead_matching.swag.model.swag_config_types import ExtractorDataRequirement

        reqs = self.extractor.data_requirements
        self.assertIn(ExtractorDataRequirement.LANDMARKS, reqs)
        print(f"✓ Data requirements: {reqs}")

    def test_properties_methods(self):
        """Test output_dim and num_position_outputs properties."""
        self.assertEqual(self.extractor.output_dim, 4)
        self.assertEqual(self.extractor.num_position_outputs, 2)
        print(f"✓ output_dim={self.extractor.output_dim}, "
              f"num_position_outputs={self.extractor.num_position_outputs}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
