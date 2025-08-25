
import unittest

import common.torch.load_torch_deps
import torch
from pathlib import Path

import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
import experimental.overhead_matching.swag.data.vigor_dataset as vd


class SemanticLandmarkExtractorTest(unittest.TestCase):
    def test_panorama_landmark_extractor(self):
        # Setup
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=10)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.panorama,
            metadata=batch.panorama_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        # print(extractor_output)

    def test_satellite_landmark_extractor(self):
        # Setup
        dataset = vd.VigorDataset(
            Path("external/vigor_snippet/vigor_snippet/"),
            vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None, panorama_tensor_cache_info=None))
        config = sle.SemanticLandmarkExtractorConfig()
        model = sle.SemanticLandmarkExtractor(config)

        dataloader = vd.get_dataloader(dataset, batch_size=1)
        batch = next(iter(dataloader))
        model_input = sle.ModelInput(
            image=batch.satellite,
            metadata=batch.satellite_metadata,
        )

        # Action
        extractor_output = model(model_input)

        # Verification
        # print(extractor_output)


if __name__ == "__main__":
    unittest.main()
