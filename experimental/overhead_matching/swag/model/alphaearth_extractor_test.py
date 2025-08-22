
import unittest

from pathlib import Path

import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.model.alphaearth_extractor import AlphaEarthExtractor
from experimental.overhead_matching.swag.model.swag_config_types import AlphaEarthExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput
from common.gps.web_mercator import latlon_to_pixel_coords


class AlphaEarthExtractorTest(unittest.TestCase):
    def test_happy_case(self):
        # Setup
        PATCH_SIZE = (10, 12)
        config = AlphaEarthExtractorConfig(
            version="v1",
            patch_size=PATCH_SIZE,
            auxiliary_info_key='')
        extractor = AlphaEarthExtractor(config, Path("external/alphaearth_snippet"))

        ZOOM_LEVEL = 20
        locations = [
            # Mercurio, Shadyside
            (40.4503414, -79.9357473),
            # Dana Square Park, Cambridge
            (42.3614103, -71.1095213),
            # Briggs Field
            (42.356887, -71.101290),
        ]
        BATCH_SIZE = len(locations)

        model_input = ModelInput(
            image=torch.zeros((BATCH_SIZE, 0, 0, 0)),
            metadata=[
                {"lat": latlon[0],
                 "lon": latlon[1],
                 "zoom_level": ZOOM_LEVEL,
                 "web_mercator_y": webm_yx[0],
                 "web_mercator_x": webm_yx[1]}
                for latlon in locations
                if (webm_yx := latlon_to_pixel_coords(*latlon, ZOOM_LEVEL))])

        # Action
        extractor_output = extractor(model_input)

        # Verification
        NUM_TOKENS = PATCH_SIZE[0] * PATCH_SIZE[1]
        self.assertEqual(extractor_output.features.shape,
                         (BATCH_SIZE, NUM_TOKENS, extractor.output_dim))
        self.assertEqual(extractor_output.positions.shape, (BATCH_SIZE, NUM_TOKENS, 2))
        self.assertEqual(extractor_output.mask.shape, (BATCH_SIZE, NUM_TOKENS))

        feature_norms = torch.linalg.norm(extractor_output.features, dim=-1)
        abs_feature_deltas = torch.abs(feature_norms - 1.0)
        self.assertTrue(torch.logical_or(
            abs_feature_deltas < 1e-2,
            extractor_output.mask).all())

        # All tokens from the first two batch items should be valid
        self.assertFalse(extractor_output.mask[:2].any())

        # Some tokens from the last batch items should be invalid
        self.assertTrue(extractor_output.mask[-1].any())
        # Some tokens from the last batch items should be valid
        self.assertTrue((~extractor_output.mask[-1]).any())


if __name__ == "__main__":
    unittest.main()
