
import unittest

import common.torch.load_torch_deps
import torch
from pathlib import Path
import math

import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
import experimental.overhead_matching.swag.data.vigor_dataset as vd


class SemanticLandmarkExtractorTest(unittest.TestCase):
    def test_panorama_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 7
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

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
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, max_num_landmarks, 2))

    def test_satellite_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 13
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

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
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, max_num_landmarks, 2))

    def test_satellite_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 2
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    landmarks=[
                        dict(
                            web_mercator_y=landmark_1[0],
                            web_mercator_x=landmark_1[1],
                            landmark_type="bus_stop"),
                        dict(
                            web_mercator_y=landmark_2[0],
                            web_mercator_x=landmark_2[1],
                            landmark_type="restaurants")]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            web_mercator_y=landmark_1[0],
                            web_mercator_x=landmark_1[1],
                            landmark_type="places_of_worship")]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2))

        self.assertTrue((extractor_output.mask ==
                         torch.tensor([[False, False], [False, True]])).all())
        self.assertTrue((extractor_output.positions[0, 0] ==
                         torch.tensor([landmark_1[0] - loc_1[0], landmark_1[1] - loc_1[1]])).all())
        self.assertTrue((extractor_output.positions[0, 1] ==
                         torch.tensor([landmark_2[0] - loc_1[0], landmark_2[1] - loc_1[1]])).all())
        self.assertTrue((extractor_output.positions[1, 0] ==
                         torch.tensor([landmark_1[0] - loc_2[0], landmark_1[1] - loc_2[1]])).all())

    def test_panorama_landmark_extractor(self):
        # Setup
        BATCH_SIZE = 2
        MAX_NUM_LANDMARKS = 2
        PANO_WIDTH = 512
        loc_1 = (123.0, 456.0)
        loc_2 = (987.0, 654.0)
        landmark_1 = (1000.0, 2000.0)
        landmark_2 = (125.0, 458.0)
        model_input = sle.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, PANO_WIDTH)),
            metadata=[
                dict(
                    web_mercator_y=loc_1[0],
                    web_mercator_x=loc_1[1],
                    pano_id="pano_id",
                    landmarks=[
                        dict(
                            web_mercator_y=landmark_1[0],
                            web_mercator_x=landmark_1[1],
                            landmark_type="bus_stop"),
                        dict(
                            web_mercator_y=landmark_2[0],
                            web_mercator_x=landmark_2[1],
                            landmark_type="restaurants")]),
                dict(
                    web_mercator_y=loc_2[0],
                    web_mercator_x=loc_2[1],
                    landmarks=[
                        dict(
                            web_mercator_y=landmark_1[0],
                            web_mercator_x=landmark_1[1],
                            landmark_type="places_of_worship")]),
            ],
        )
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        # Action
        extractor_output = model(model_input)

        # Verification
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS))
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, MAX_NUM_LANDMARKS, model.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, MAX_NUM_LANDMARKS, 2))

        self.assertTrue((extractor_output.mask ==
                         torch.tensor([[False, False], [False, True]])).all())

        def compute_column_from_pano_landmark(pano_loc, landmark_loc, pano_width):
            dx = landmark_loc[1] - pano_loc[1]
            dy = landmark_loc[0] - pano_loc[0]
            theta = math.atan2(dx, dy)
            frac = (theta + math.pi) / (2 * math.pi)
            return frac * pano_width

        self.assertEqual(extractor_output.positions[0, 0, 1],
                         compute_column_from_pano_landmark(loc_1, landmark_1, PANO_WIDTH))
        self.assertEqual(extractor_output.positions[0, 1, 1],
                         compute_column_from_pano_landmark(loc_1, landmark_2, PANO_WIDTH))
        self.assertEqual(extractor_output.positions[1, 0, 1],
                         compute_column_from_pano_landmark(loc_2, landmark_1, PANO_WIDTH))

if __name__ == "__main__":
    unittest.main()
