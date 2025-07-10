import unittest

import common.torch.load_torch_deps
import torch
import math
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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
        LAT_STEP = 1.0

        MIN_LON = 20.0
        MAX_LON = 40.0
        LON_STEP = 1.0

        lats = np.arange(MIN_LAT, MAX_LAT + LAT_STEP / 2.0, LAT_STEP)
        lons = np.arange(MIN_LON, MAX_LON + LON_STEP / 2.0, LON_STEP)

        LANDMARK_LON_OFFSET_DEG = 0.025
        LANDMARK_LAT_OFFSET_DEG = 0.05
        landmark_info = []
        sat_dir = temp_dir / "satellite"
        sat_dir.mkdir()
        for (lat_idx, lat), (lon_idx, lon) in itertools.product(enumerate(lats), enumerate(lons)):
            file_path = sat_dir / f"satellite_{lat}_{lon}.png"
            # We place landmarks at a fixed offset relative to each satellite patch.
            # The landmarks are placed such that they are present in multiple satellite patches
            # There is no structure enforced on a landmark, other than it must have a location
            landmark_info.append({
                "lat": lat + LANDMARK_LAT_OFFSET_DEG,
                "lon": lon + LANDMARK_LON_OFFSET_DEG,
                "lat_idx": lat_idx,
                "lon_idx": lon_idx,
                "landmark_type": f"LAT{lat_idx}LON{lon_idx}"})
            image = Image.new("RGB", size=(200, 200))
            draw = ImageDraw.Draw(image)
            draw.text((5, 5), f"({lat}, {lon})")
            image.putpixel(
                (0, 0), (0 if lat >= 0 else 1, int(abs(lat)), int((abs(lat) % 1) * 100))
            )
            image.putpixel(
                (1, 0), (0 if lon >= 0 else 1, int(abs(lon)), int((abs(lon) % 1) * 100))
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
                        int(abs(road_lat)),
                        int((abs(road_lat) % 1) * 100),
                    ),
                )
                image.putpixel(
                    (1, 0),
                    (
                        0 if road_lon >= 0 else 1,
                        int(abs(road_lon)),
                        int((abs(road_lon) % 1) * 100),
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
                        int(abs(road_lat)),
                        int((abs(road_lat) % 1) * 100),
                    ),
                )
                image.putpixel(
                    (1, 0),
                    (
                        0 if road_lon >= 0 else 1,
                        int(abs(road_lon)),
                        int((abs(road_lon) % 1) * 100),
                    ),
                )
                image.save(file_path)

        landmark_info_df = pd.DataFrame.from_records(landmark_info)
        landmark_geometry = gpd.points_from_xy(landmark_info_df["lon"], landmark_info_df["lat"])
        landmark_info_df = gpd.GeoDataFrame(landmark_info_df, geometry=landmark_geometry)
        landmark_info_df.to_file(temp_dir / "test.geojson", driver="GeoJSON")

    @classmethod
    def tearDownClass(cls):
        cls._temp_dir.cleanup()
        del cls._temp_dir

    def test_get_single_item(self):
        # Setup
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.4,
            satellite_patch_size=None,
            panorama_size=None,
            satellite_zoom_level=7,
        )
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)

        # Action
        item = dataset[100]

        # Verification
        # Check that the panorama has neighbors and the associated satellite patch has at least 1 child panorama
        self.assertGreater(len(item.panorama_metadata["neighbor_panorama_idxs"]), 0)
        self.assertGreater(len(item.satellite_metadata["positive_panorama_idxs"]), 0)

        # Check that the location embedded in the images matches the metadata
        pano_lat_sign = -1 if item.panorama[0, 0, 0] > 0 else 1
        pano_embedded_lat = pano_lat_sign * 255 * (item.panorama[1, 0, 0] + 0.01 * item.panorama[2, 0, 0]).item()

        pano_lon_sign = -1 if item.panorama[0, 0, 1] > 0 else 1
        pano_embedded_lon = pano_lon_sign * 255 * (item.panorama[1, 0, 1] + 0.01 * item.panorama[2, 0, 1]).item()

        self.assertAlmostEqual(item.panorama_metadata["lat"], pano_embedded_lat, places=1)
        self.assertAlmostEqual(item.panorama_metadata["lon"], pano_embedded_lon, places=1)

        sat_lat_sign = -1 if item.satellite[0, 0, 0] > 0 else 1
        sat_embedded_lat = sat_lat_sign * 255 * (item.satellite[1, 0, 0] + 0.01 * item.satellite[2, 0, 0]).item()

        sat_lon_sign = -1 if item.satellite[0, 0, 1] > 0 else 1
        sat_embedded_lon = sat_lon_sign * 255 * (item.satellite[1, 0, 1] + 0.01 * item.satellite[2, 0, 1]).item()

        self.assertAlmostEqual(item.satellite_metadata["lat"], sat_embedded_lat, places=1)
        self.assertAlmostEqual(item.satellite_metadata["lon"], sat_embedded_lon, places=1)

        dataset.visualize(include_text_labels=True)
        plt.show(block=True)

    def test_landmarks_are_correct(self):
        # Setup
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.4,
            satellite_patch_size=None,
            panorama_size=None,
            satellite_zoom_level=7,
        )

        # Action
        dataset = vigor_dataset.VigorDataset(
                Path(self._temp_dir.name),
                config,
                landmark_path=[Path(self._temp_dir.name) / "test.geojson"])

        # Verification
        max_lat_idx = dataset._landmark_metadata["lat_idx"].max()
        max_lon_idx = dataset._landmark_metadata["lon_idx"].max()
        for _, landmark_meta in dataset._landmark_metadata.iterrows():
            num_expected_neighbors = 9
            lat_idx = landmark_meta["lat_idx"]
            lon_idx = landmark_meta["lon_idx"]
            if lat_idx == 0 or lat_idx == max_lat_idx:
                num_expected_neighbors -= 3
            if lon_idx == 0 or lon_idx == max_lon_idx:
                num_expected_neighbors -= 3
            num_expected_neighbors = max(num_expected_neighbors, 4)
            self.assertEqual(len(landmark_meta["satellite_idxs"]), num_expected_neighbors)

    def test_get_batch(self):
        # Setup
        BATCH_SIZE = 32
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius = 0.4,
            satellite_patch_size = (50, 50),
            panorama_size = (100, 100),
            satellite_zoom_level=7,
        )
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)
        dataloader = vigor_dataset.get_dataloader(dataset, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(dataloader))

        # Verification
        self.assertEqual(len(batch.panorama_metadata), BATCH_SIZE)
        self.assertEqual(len(batch.satellite_metadata), BATCH_SIZE)
        self.assertEqual(batch.panorama.shape, (BATCH_SIZE, 3, *config.panorama_size))
        self.assertEqual(batch.satellite.shape, (BATCH_SIZE, 3, *config.satellite_patch_size))

    def test_iterate_overhead_dataset(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius = 0.2,
            satellite_patch_size = (50, 50),
            panorama_size = (100, 100),
            satellite_zoom_level=7,
        )
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)
        overhead_view = dataset.get_sat_patch_view()
        # Action and verification
        for item in overhead_view:
            pass

    def test_get_overhead_batch(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius = 0.2,
            satellite_patch_size = (50, 50),
            panorama_size = (100, 100),
            satellite_zoom_level=7,
        )
        BATCH_SIZE = 32
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)
        overhead_view = dataset.get_sat_patch_view()
        dataloader = vigor_dataset.get_dataloader(overhead_view, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(dataloader))

        # Verification
        self.assertIsNone(batch.panorama_metadata)
        self.assertEqual(len(batch.satellite_metadata), BATCH_SIZE)
        self.assertIsNone(batch.panorama)
        self.assertEqual(batch.satellite.shape[0], BATCH_SIZE)

    def test_get_panorama_batch(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius = 0.2,
            satellite_patch_size = (50, 50),
            panorama_size = (100, 100),
            satellite_zoom_level=7,
        )
        BATCH_SIZE = 32
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)
        panorama_view = dataset.get_pano_view()
        dataloader = vigor_dataset.get_dataloader(panorama_view, batch_size=BATCH_SIZE)

        # Action
        batch = next(iter(dataloader))

        # Verification
        self.assertIsNone(batch.satellite_metadata)
        self.assertEqual(len(batch.panorama_metadata), BATCH_SIZE)
        self.assertEqual(batch.panorama.shape[0], BATCH_SIZE)
        self.assertIsNone(batch.satellite)

    def test_overhead_and_main_dataset_are_consistient(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius = 0.2,
            satellite_patch_size = (50, 50),
            panorama_size = (100, 100),
            satellite_zoom_level=7,
        )
        CHECK_INDEX = 25
        dataset = vigor_dataset.VigorDataset(Path(self._temp_dir.name), config)
        overhead_view = dataset.get_sat_patch_view()

        # Action
        dataset_item = dataset[CHECK_INDEX]
        sat_index = dataset_item.satellite_metadata['index']
        overhead_view_item = overhead_view[sat_index]

        # Verification
        self.assertIsNone(overhead_view_item.panorama_metadata)
        self.assertIsNone(overhead_view_item.panorama)
        self.assertTrue(torch.allclose(overhead_view_item.satellite, dataset_item.satellite))
        self.assertEqual(dataset_item.satellite_metadata, overhead_view_item.satellite_metadata)

    def test_index_by_tensor(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )

        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)

        # action and verification
        item = dataset[torch.tensor([7])]

    def test_path_generation_is_reproducible(self):
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )
        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)

        PATH_LENGTH_M = 100
        SEED = 532
        # action 
        path1 = dataset.generate_random_path(torch.manual_seed(SEED), PATH_LENGTH_M, 1.0)
        path2 = dataset.generate_random_path(torch.manual_seed(SEED), PATH_LENGTH_M, 1.0)
        path3 = dataset.generate_random_path(torch.manual_seed(SEED-3), PATH_LENGTH_M, 1.0)
    
        for item in path1:
            self.assertTrue(type(item) == int)
        self.assertListEqual(path1, path2)
        self.assertNotEqual(path1, path3)

    @unittest.skip("Visualization only")
    def test_visualize_path(self):
        import matplotlib.pyplot as plt
        PATH_LENGTH_M = 10000
        SEED = 532
        config = vigor_dataset.VigorDatasetConfig(
            panorama_neighbor_radius=0.0005,
            satellite_patch_size=None,
            panorama_size=None,
        )
        dataset = vigor_dataset.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet"), config)
        # action 
        path1 = dataset.generate_random_path(torch.manual_seed(SEED), PATH_LENGTH_M, 0.1)

        fig, ax = dataset.visualize(path=path1)
        #plt.show()
        plt.savefig("/tmp/path_visual.png")
        plt.close(fig)


class HardNegativeMinerTest(unittest.TestCase):
    def test_negative_miner_returns_hard_negatives_in_hard_negative_mode(self):
        '''
        When the hard negative miner is configured to produce hard negatives, we expect that
        hard negatives are sampled first and then any remaining items in the batch are sampled
        randomly. In this test, one satellite patch is closely aligned with the panorama embedding,
        and we have a batch size of 1. As a result, this closely aligned satellite embedding shoud
        be the only sample produced.
        '''
        # Setup
        EMBEDDING_DIMENSION = 4
        BATCH_SIZE = 1
        panorama_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]])
        satellite_embeddings = torch.tensor([
             [0.0, 1.0, 0.0, 0.0],
             [math.sqrt(2)/2, math.sqrt(2), 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0]])
        generator = torch.Generator().manual_seed(42)

        miner = vigor_dataset.HardNegativeMiner(
            batch_size=BATCH_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            random_sample_type=vigor_dataset.HardNegativeMiner.RandomSampleType.POS_SEMIPOS,
            num_panoramas=len(panorama_embeddings),
            num_satellite_patches=len(satellite_embeddings),
            hard_negative_pool_size=1,
            panorama_info_from_pano_idx={
                0: vigor_dataset.PanoramaIndexInfo(
                    panorama_idx=0,
                    nearest_satellite_idx=2,
                    positive_satellite_idxs=[0],
                    semipositive_satellite_idxs=[1])},
            device='cpu',
            generator=generator)

        # Action
        miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE)
        miner.consume(
            panorama_embeddings=panorama_embeddings,
            satellite_embeddings=satellite_embeddings,
            panorama_idxs=[0],
            satellite_patch_idxs=[0, 1, 2])

        # Verification
        for i in range(100):
            batch_idxs = next(iter(miner))

            # Verification
            self.assertEqual(len(batch_idxs), BATCH_SIZE)
            self.assertEqual(batch_idxs[0].panorama_idx, 0)
            self.assertEqual(batch_idxs[0].satellite_idx, 1)

    def test_negative_miner_returns_random_pos_semipos_samples_in_pos_semipos_mode(self):
        '''
        When the hard negative miner is configured to produce random samples, we expect that
        the two satellite patches are sampled about evenly.
        '''
        # Setup
        EMBEDDING_DIMENSION = 4
        BATCH_SIZE = 1
        panorama_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]])
        satellite_embeddings = torch.tensor([
             [0.0, 1.0, 0.0, 0.0],
             [math.sqrt(2)/2, math.sqrt(2), 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0]])
        generator = torch.Generator().manual_seed(42)

        miner = vigor_dataset.HardNegativeMiner(
            batch_size=BATCH_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            random_sample_type=vigor_dataset.HardNegativeMiner.RandomSampleType.POS_SEMIPOS,
            num_panoramas=len(panorama_embeddings),
            num_satellite_patches=len(satellite_embeddings),
            panorama_info_from_pano_idx={
                0: vigor_dataset.PanoramaIndexInfo(
                    panorama_idx=0,
                    nearest_satellite_idx=2,
                    positive_satellite_idxs=[0],
                    semipositive_satellite_idxs=[1])},
            device='cpu',
            generator=generator)

        # Action
        miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.RANDOM)
        miner.consume(
            panorama_embeddings=panorama_embeddings,
            satellite_embeddings=satellite_embeddings,
            panorama_idxs=[0],
            satellite_patch_idxs=[0, 1, 2])

        # Verification
        satellite_patch_count = [0, 0, 0]
        for i in range(100):
            batch_idxs = next(iter(miner))
            self.assertEqual(len(batch_idxs), BATCH_SIZE)
            self.assertEqual(batch_idxs[0].panorama_idx, 0)
            satellite_patch_count[batch_idxs[0].satellite_idx] += 1

        self.assertGreater(satellite_patch_count[0], 25)
        self.assertGreater(satellite_patch_count[1], 25)
        self.assertEqual(satellite_patch_count[2], 0)

    def test_negative_miner_returns_random_nearest_samples_in_nearest_mode(self):
        '''
        When the hard negative miner is configured to produce random samples, we expect that
        the two satellite patches are sampled about evenly.
        '''
        # Setup
        EMBEDDING_DIMENSION = 4
        BATCH_SIZE = 1
        panorama_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]])
        satellite_embeddings = torch.tensor([
             [0.0, 1.0, 0.0, 0.0],
             [math.sqrt(2)/2, math.sqrt(2), 0.0, 0.0],
             [-1.0, 0.0, 0.0, 0.0]])
        generator = torch.Generator().manual_seed(42)

        miner = vigor_dataset.HardNegativeMiner(
            batch_size=BATCH_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            random_sample_type=vigor_dataset.HardNegativeMiner.RandomSampleType.NEAREST,
            num_panoramas=len(panorama_embeddings),
            num_satellite_patches=len(satellite_embeddings),
            panorama_info_from_pano_idx={
                0: vigor_dataset.PanoramaIndexInfo(
                    panorama_idx=0,
                    nearest_satellite_idx=2,
                    positive_satellite_idxs=[0],
                    semipositive_satellite_idxs=[1])},
            device='cpu',
            generator=generator)

        # Action
        miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.RANDOM)
        miner.consume(
            panorama_embeddings=panorama_embeddings,
            satellite_embeddings=satellite_embeddings,
            panorama_idxs=[0],
            satellite_patch_idxs=[0, 1, 2])

        # Verification
        for i in range(100):
            batch_idxs = next(iter(miner))
            self.assertEqual(len(batch_idxs), BATCH_SIZE)
            self.assertEqual(batch_idxs[0].panorama_idx, 0)
            self.assertEqual(batch_idxs[0].satellite_idx, 2)


if __name__ == "__main__":
    unittest.main()
