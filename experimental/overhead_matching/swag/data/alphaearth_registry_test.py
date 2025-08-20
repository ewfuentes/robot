import unittest

from pathlib import Path
import experimental.overhead_matching.swag.data.alphaearth_registry as ae_reg


class AlphaEarthRegistryTest(unittest.TestCase):
    def test_create_registry(self):
        # Setup
        base_path = Path('external/alphaearth_snippet')
        dataset_version = "v1"
        registry = ae_reg.AlphaEarthRegistry(base_path=base_path, version=dataset_version)
        
        # Action + Verification
        # Dana square park in cambridgeport
        registry.query(42.3614103, -71.1095213, (5, 6))
        # Mercurio in Shadyside
        registry.query(40.4503414, -79.9357473, (5, 7))
        # Times Square in New York
        self.assertRaises(ValueError, registry.query, 40.758054, -73.9854918, (128, 128))


if __name__ == "__main__":
    unittest.main()
