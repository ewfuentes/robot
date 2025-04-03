import unittest

import common.torch.load_torch_deps
import torch

from pathlib import Path
import tempfile
import numpy as np
import itertools
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.data import vigor_dataset


class VigorDatasetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We create a dataset where the satellite patches are every half degree
        # The road network has roads with 1 degree spacing and the panoramas are
        # every 0.5 / 3 degrees. The roads are also offset from the satellite patch
        # grid by 0.5 / 4 degrees. This makes it so it's unambiguous which patch
        # a panorama belongs to.

        cls._temp_dir = tempfile.TemporaryDirectory()
        temp_dir = Path(cls._temp_dir.name)
        # Create satellite overhead
        MIN_LAT = -10.0
        MAX_LAT = 10.0
        LAT_STEP = 0.5

        MIN_LON = 20.0
        MAX_LON = 40.0
        LON_STEP = 0.5

        lats = np.arange(MIN_LAT, MAX_LAT + LAT_STEP / 2.0, LAT_STEP)
        lons = np.arange(MIN_LON, MAX_LON + LON_STEP / 2.0, LON_STEP)

        sat_dir = temp_dir / "satellite"
        sat_dir.mkdir()
        for lat, lon in itertools.product(lats, lons):
            file_path = sat_dir / f"satellite_{lat}_{lon}.png"
            image = Image.new("RGB", size=(200, 200))
            draw = ImageDraw.Draw(image)
            draw.text((5, 5), f"({lat}, {lon})")
            image.putpixel(
                (0, 0), (0 if lat >= 0 else 1, int(lat), int((lat % 1) * 100))
            )
            image.putpixel(
                (1, 0), (0 if lon >= 0 else 1, int(lon), int((lon % 1) * 100))
            )
            image.save(file_path)

        # Create panoramas
        sat_dir = temp_dir / "panorama"
        sat_dir.mkdir()
        for road_lat in lats[::2] + LAT_STEP / 4.0:
            road_lons = np.arange(MIN_LON, MAX_LON + LON_STEP / 4, LON_STEP / 3.0)
            for road_lon in road_lons:
                file_path = (
                    sat_dir
                    / f"pano_id_{road_lat}_{road_lon},{road_lat},{road_lon},.png"
                )
                image = Image.new("RGB", size=(200, 200), color=(128, 128, 128))
                draw = ImageDraw.Draw(image)
                draw.text((5, 5), f"({road_lat}, {road_lon})")
                image.putpixel(
                    (0, 0),
                    (
                        0 if road_lat >= 0 else 1,
                        int(road_lat),
                        int((road_lat % 1) * 100),
                    ),
                )
                image.putpixel(
                    (1, 0),
                    (
                        0 if road_lon >= 0 else 1,
                        int(road_lon),
                        int((road_lon % 1) * 100),
                    ),
                )
                image.save(file_path)

        for road_lon in lons[::2] + LAT_STEP / 4.0:
            road_lats = np.arange(MIN_LAT, MAX_LAT + LAT_STEP / 4, LAT_STEP / 3.0)
            for road_lat in road_lats:
                file_path = (
                    sat_dir
                    / f"pano_id_{road_lat}_{road_lon},{road_lat},{road_lon},.png"
                )
                image = Image.new("RGB", size=(200, 200), color=(128, 128, 128))
                draw = ImageDraw.Draw(image)
                draw.text((5, 5), f"({road_lat}, {road_lon})")
                image.putpixel(
                    (0, 0),
                    (
                        0 if road_lat >= 0 else 1,
                        int(road_lat),
                        int((road_lat % 1) * 100),
                    ),
                )
                image.putpixel(
                    (1, 0),
                    (
                        0 if road_lon >= 0 else 1,
                        int(road_lon),
                        int((road_lon % 1) * 100),
                    ),
                )
                image.save(file_path)

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        del cls._temp_dir

    def test_get_single_item(self):
        # Setup
        NEIGHBOR_PANO_RADIUS = 0.2
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), NEIGHBOR_PANO_RADIUS)

        # Action
        item = dataset[100]

        # Verification
        # Check that the panorama has neighbors and the associated satellite patch has at least 1 child panorama
        self.assertGreater(len(item.panorama_metadata["neighbor_panorama_idxs"]), 0)
        self.assertGreater(len(item.satellite_metadata["panorama_idxs"]), 0)

        # Check that the location embedded in the images matches the metadata
        pano_lat_sign = -1 if item.panorama[0, 0, 0] == 1 else 1
        pano_embedded_lat = pano_lat_sign * (item.panorama[1, 0, 0] + 0.01 * item.panorama[2, 0, 0]).item()

        pano_lon_sign = -1 if item.panorama[0, 0, 1] == 1 else 1
        pano_embedded_lon = pano_lon_sign * (item.panorama[1, 0, 1] + 0.01 * item.panorama[2, 0, 1]).item()

        print(item.panorama_metadata)
        print(item.panorama[:, 0, 0])
        print(item.panorama[:, 0, 1])
        self.assertAlmostEqual(item.panorama_metadata["lat"], pano_embedded_lat, places=1)
        self.assertAlmostEqual(item.panorama_metadata["lon"], pano_embedded_lon, places=1)

        sat_lat_sign = -1 if item.satellite[0, 0, 0] == 1 else 1
        sat_embedded_lat = sat_lat_sign * (item.satellite[1, 0, 0] + 0.01 * item.satellite[2, 0, 0]).item()

        sat_lon_sign = -1 if item.satellite[0, 0, 1] == 1 else 1
        sat_embedded_lon = sat_lon_sign * (item.satellite[1, 0, 1] + 0.01 * item.satellite[2, 0, 1]).item()

        print(item.satellite_metadata)
        print(item.satellite[:, 0, 0])
        print(item.satellite[:, 0, 1])
        self.assertAlmostEqual(item.satellite_metadata["lat"], sat_embedded_lat, places=1)
        self.assertAlmostEqual(item.satellite_metadata["lon"], sat_embedded_lon, places=1)

    def test_get_batch(self):
        # Setup
        NEIGHBOR_PANO_RADIUS = 0.2
        BATCH_SIZE = 32
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), NEIGHBOR_PANO_RADIUS)
        dataloader = vigor_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(dataloader))

        # Verification
        self.assertEqual(len(batch.panorama_metadata), BATCH_SIZE)
        self.assertEqual(len(batch.satellite_metadata), BATCH_SIZE)
        self.assertEqual(batch.panorama.shape[0], BATCH_SIZE)
        self.assertEqual(batch.satellite.shape[0], BATCH_SIZE)


if __name__ == "__main__":
    unittest.main()
