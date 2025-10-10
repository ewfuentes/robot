
import unittest

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import math

import experimental.overhead_matching.swag.model.absolute_position_extractor as ape
import experimental.overhead_matching.swag.data.vigor_dataset as vd


class AbsolutePositionExtractorTest(unittest.TestCase):
    def test_panorama_landmark_extractor_with_dataset(self):
        # Setup
        BATCH_SIZE = 7
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = ape.AbsolutePositionExtractorConfig()
        model = ape.AbsolutePositionExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = ape.ModelInput(
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

        config = ape.AbsolutePositionExtractorConfig()
        model = ape.AbsolutePositionExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=BATCH_SIZE)
        batch = next(iter(dataloader))
        model_input = ape.ModelInput(
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

    def test_simple_inputs(self):
        BATCH_SIZE = 2
        loc1_deg = (0,0)
        loc2_deg = (0, -180)
        model_input = ape.ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 256, 256)),
            metadata=[
                dict(
                    lat=loc1_deg[0],
                    lon=loc1_deg[1],
                ),
                dict(
                    lat=loc2_deg[0],
                    lon=loc2_deg[1],
                )
            ],
        )
        config = ape.AbsolutePositionExtractorConfig()
        model = ape.AbsolutePositionExtractor(config)

        # Action
        extractor_output = model(model_input)
        # Verification
        self.assertAlmostEqual(extractor_output.features[0, 0, 0].item(), math.sin(np.deg2rad(loc1_deg[0])), places=5) 
        self.assertAlmostEqual(extractor_output.features[0, 0, 1].item(), math.cos(np.deg2rad(loc1_deg[0])), places=5)
        self.assertAlmostEqual(extractor_output.features[0, 0, 4].item(), math.sin(np.deg2rad(loc1_deg[0])), places=5)

        self.assertAlmostEqual(extractor_output.features[1, 0, 2].item(), math.sin(np.deg2rad(loc2_deg[1])), places=5)
        self.assertAlmostEqual(extractor_output.features[1, 0, 3].item(), math.cos(np.deg2rad(loc2_deg[1])), places=5)
        self.assertAlmostEqual(extractor_output.features[1, 0, 6].item(), math.sin(np.deg2rad(loc2_deg[1]) * 2), places=5)


if __name__ == "__main__":
    unittest.main()
