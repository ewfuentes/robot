
import unittest

from pathlib import Path
import tempfile
import numpy as np
import itertools
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import vigor_dataset

class VigorDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._temp_dir = tempfile.TemporaryDirectory()
        temp_dir = Path(cls._temp_dir.name)
        # Create satellite overhead
        sat_dir = temp_dir / "satellite"
        sat_dir.mkdir()
        for (lat, lon) in itertools.product(np.arange(-10, 10.1, 0.5), np.arange(20.0, 40.1, 0.5)):
            file_path = sat_dir / f"satellite_{lat}_{lon}.png"
            image = Image.new("RGB", size=(200, 200))
            draw = ImageDraw.Draw(image)
            draw.text((5, 5), f"({lat}, {lon})")
            image.putpixel((0, 0), (0 if lat >= 0 else 1, int(lat), int((lat % 1) * 100)))
            image.putpixel((1, 0), (0 if lon >= 0 else 1, int(lon), int((lon % 1) * 100)))
            image.save(file_path)

        # Create panoramas
        sat_dir = temp_dir / "panorama"
        sat_dir.mkdir()
        for (lat, lon) in itertools.product(np.arange(-10, 10.1, 0.5), np.arange(20.0, 40.1, 0.5)):
            file_path = sat_dir / f"pano_id_{lat}_{lon},{lat},{lon},.png"
            image = Image.new("RGB", size=(200, 200), color=(128, 128, 128))
            draw = ImageDraw.Draw(image)
            draw.text((5, 5), f"({lat}, {lon})")
            image.putpixel((0, 0), (0 if lat >= 0 else 1, int(lat), int((lat % 1) * 100)))
            image.putpixel((1, 0), (0 if lon >= 0 else 1, int(lon), int((lon % 1) * 100)))
            image.save(file_path)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        del cls._temp_dir

    def test_happy_case(self):
        # Setup
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name))

        # Action
        print(dataset)

        # Verification
        

if __name__ == "__main__":
    unittest.main()
