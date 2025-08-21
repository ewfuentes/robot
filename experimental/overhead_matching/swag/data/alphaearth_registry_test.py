import unittest

from pathlib import Path
import experimental.overhead_matching.swag.data.alphaearth_registry as ae_reg
import numpy as np


class AlphaEarthRegistryTest(unittest.TestCase):
    def test_valid_queries(self):
        # Setup
        base_path = Path('external/alphaearth_snippet')
        dataset_version = "v1"
        registry = ae_reg.AlphaEarthRegistry(base_path=base_path, version=dataset_version)

        # Action
        # Dana square park in cambridgeport
        cambridge_features, cambridge_position = registry.query(42.3614103, -71.1095213, (6, 6), zoom_level=20)
        # Mercurio in Shadyside
        shadyside_features, shadyside_position = registry.query(40.4503414, -79.9357473, (6, 8), zoom_level=20)

        # Verification
        self.assertEqual(cambridge_features.shape, (6, 6, 64))
        self.assertEqual(shadyside_features.shape, (6, 8, 64))
        self.assertEqual(cambridge_position.shape, (6, 6, 2))
        self.assertEqual(shadyside_position.shape, (6, 8, 2))
        self.assertTrue(np.all(np.abs(np.linalg.norm(cambridge_features, axis=-1) - 1.0) < 1e-2))
        self.assertTrue(np.all(np.abs(np.linalg.norm(shadyside_features, axis=-1) - 1.0) < 1e-2))

    def test_partially_invalid(self):
        # Setup
        base_path = Path('external/alphaearth_snippet')
        dataset_version = "v1"
        registry = ae_reg.AlphaEarthRegistry(base_path=base_path, version=dataset_version)

        # Action
        # Briggs Field, just outside of the test region
        briggs_features, briggs_position = registry.query(42.356887, -71.101290, (10, 12), zoom_level=20)

        # Verification
        self.assertEqual(briggs_features.shape, (10, 12, 64))
        self.assertEqual(briggs_position.shape, (10, 12, 2))
        self.assertTrue(np.all(np.isnan(briggs_features[-1, -1, :])))

    def test_invalid_query(self):
        # Setup
        base_path = Path('external/alphaearth_snippet')
        dataset_version = "v1"
        registry = ae_reg.AlphaEarthRegistry(base_path=base_path, version=dataset_version)

        # Action + Verification
        # Times Square in New York
        self.assertRaises(ValueError, registry.query, 40.758054, -73.9854918, (128, 128), 20)


if __name__ == "__main__":
    unittest.main()
