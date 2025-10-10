
import unittest

import common.torch.load_torch_deps
import torch
from pathlib import Path
import math

import experimental.overhead_matching.swag.model.synthetic_landmark_extractor as sle
import experimental.overhead_matching.swag.data.vigor_dataset as vd


class SyntheticLandmarkExtractorTest(unittest.TestCase):
    def test_panorama_landmark_extractor_with_dataset_planar(self):
        # Setup
        BATCH_SIZE = 7
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SyntheticLandmarkExtractorConfig(
            log_grid_spacing=5,
            grid_bounds_px=128,
            should_produce_bearing_position_for_pano=False,
            embedding_dim=16)
        model = sle.SyntheticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.panorama,
            metadata=batch.panorama_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        max_num_landmarks = extractor_output.features.shape[1]
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, max_num_landmarks))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, max_num_landmarks, model.output_dim))
        self.assertEqual(extractor_output.positions.shape,
                         (BATCH_SIZE, max_num_landmarks, model.num_position_outputs, 2))

    def test_satellite_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 13
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SyntheticLandmarkExtractorConfig(
            log_grid_spacing=5,
            grid_bounds_px=128,
            should_produce_bearing_position_for_pano=False,
            embedding_dim=16)
        model = sle.SyntheticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.satellite,
            metadata=batch.satellite_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        max_num_landmarks = extractor_output.features.shape[1]
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, max_num_landmarks))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, max_num_landmarks, model.output_dim))
        self.assertEqual(extractor_output.positions.shape,
                         (BATCH_SIZE, max_num_landmarks, model.num_position_outputs, 2))

    def test_satellite_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 25
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1]),
            ],
        )
        config = sle.SyntheticLandmarkExtractorConfig(
            log_grid_spacing=5,
            grid_bounds_px=128,
            should_produce_bearing_position_for_pano=False,
            embedding_dim=16)
        model = sle.SyntheticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.num_position_outputs, 2))

    def test_panorama_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 25
        PANO_WIDTH = 512
        loc_1 = (128.0, 256.0)
        loc_2 = (1024.0, 2048.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    pano_id="pano_id"),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    pano_id="pano_id2"),
            ],
        )
        config = sle.SyntheticLandmarkExtractorConfig(
            log_grid_spacing=5,
            grid_bounds_px=128,
            should_produce_bearing_position_for_pano=True,
            embedding_dim=256)
        model = sle.SyntheticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.num_position_outputs, 2))

if __name__ == "__main__":
    unittest.main()
