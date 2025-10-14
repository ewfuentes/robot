"""
Unit tests for SpectralLandmarkExtractor.
"""

import common.torch.load_torch_deps
import torch
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.model.spectral_landmark_extractor import SpectralLandmarkExtractor
from experimental.overhead_matching.swag.model.swag_config_types import SpectralLandmarkExtractorConfig
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput


class TestSpectralLandmarkExtractor(unittest.TestCase):
    """Test suite for SpectralLandmarkExtractor."""

    @classmethod
    def setUpClass(cls):
        """Set up device for all tests."""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_single_image(self):
        """Test with a single image."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov3_vitb16",
            feature_source="attention_keys",
            lambda_knn=10.0,
            max_landmarks_per_image=5,
            min_bbox_size=10
        )

        extractor = SpectralLandmarkExtractor(config).to(self.device)
        extractor.eval()

        # Create dummy input
        batch_size = 1
        image = torch.rand(batch_size, 3, 224, 224).to(self.device)

        model_input = ModelInput(
            image=image,
            metadata=[{"path": Path("test_image_1.jpg")}]
        )

        # Forward pass
        with torch.no_grad():
            output = extractor(model_input)

        # Assertions
        self.assertEqual(output.features.shape, (1, 5, 768))
        self.assertEqual(output.positions.shape, (1, 5, 2))
        self.assertEqual(output.mask.shape, (1, 5))

        # Check that features and positions are finite
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(output.positions).all())

        # Check debug info exists
        self.assertIn('bboxes', output.debug)
        self.assertIn('eigenvector_indices', output.debug)
        self.assertIn('eigenvalues', output.debug)
        self.assertIn('eigenvectors', output.debug)
        self.assertIn('object_masks', output.debug)

    def test_batch(self):
        """Test with a batch of images."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov3_vitb16",
            feature_source="attention_keys",
            lambda_knn=0.0,  # No color information for faster test
            max_landmarks_per_image=3,
            min_bbox_size=10
        )

        extractor = SpectralLandmarkExtractor(config).to(self.device)
        extractor.eval()

        # Create batch input
        batch_size = 4
        images = torch.rand(batch_size, 3, 224, 224).to(self.device)

        model_input = ModelInput(
            image=images,
            metadata=[{"path": Path(f"test_image_{i}.jpg")} for i in range(batch_size)]
        )

        # Forward pass
        with torch.no_grad():
            output = extractor(model_input)

        # Assertions
        self.assertEqual(output.features.shape, (4, 3, 768))
        self.assertEqual(output.positions.shape, (4, 3, 2))
        self.assertEqual(output.mask.shape, (4, 3))

        # Check that features and positions are finite
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(output.positions).all())

        # Check that each image has at least one valid landmark or all masked
        for i in range(batch_size):
            num_detected = (~output.mask[i]).sum().item()
            self.assertGreaterEqual(num_detected, 0)
            self.assertLessEqual(num_detected, 3)

    def test_feature_source_attention_keys(self):
        """Test with attention_keys feature source."""
        self._test_feature_source("attention_keys")

    def test_feature_source_attention_input(self):
        """Test with attention_input feature source."""
        self._test_feature_source("attention_input")

    def test_feature_source_model_output(self):
        """Test with model_output feature source."""
        self._test_feature_source("model_output")

    def _test_feature_source(self, feature_source: str):
        """Helper method to test a specific feature source."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov3_vitb16",
            feature_source=feature_source,
            lambda_knn=0.0,
            max_landmarks_per_image=3
        )

        extractor = SpectralLandmarkExtractor(config).to(self.device)
        extractor.eval()

        image = torch.rand(1, 3, 224, 224).to(self.device)
        model_input = ModelInput(
            image=image,
            metadata=[{"path": Path("test.jpg")}]
        )

        with torch.no_grad():
            output = extractor(model_input)

        # Assertions
        self.assertEqual(output.features.shape, (1, 3, 768))
        self.assertEqual(output.positions.shape, (1, 3, 2))
        self.assertEqual(output.mask.shape, (1, 3))
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(output.positions).all())

    def test_no_color_info(self):
        """Test with no color information (lambda_knn=0)."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov3_vitb16",
            feature_source="attention_keys",
            lambda_knn=0.0,  # No color
            max_landmarks_per_image=5
        )

        extractor = SpectralLandmarkExtractor(config).to(self.device)
        extractor.eval()

        image = torch.rand(1, 3, 224, 224).to(self.device)
        model_input = ModelInput(
            image=image,
            metadata=[{"path": Path("test.jpg")}]
        )

        with torch.no_grad():
            output = extractor(model_input)

        # Assertions
        self.assertEqual(output.features.shape, (1, 5, 768))
        self.assertEqual(output.positions.shape, (1, 5, 2))
        self.assertEqual(output.mask.shape, (1, 5))
        self.assertTrue(torch.isfinite(output.features).all())
        self.assertTrue(torch.isfinite(output.positions).all())

        # Should work with pure feature-based decomposition
        num_detected = (~output.mask[0]).sum().item()
        self.assertGreaterEqual(num_detected, 0)
        self.assertLessEqual(num_detected, 5)

    def test_output_properties(self):
        """Test that output properties are correct."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov3_vitb16",
            feature_source="attention_keys",
            lambda_knn=0.0,
            max_landmarks_per_image=3
        )

        extractor = SpectralLandmarkExtractor(config).to(self.device)

        # Test output_dim property (should be DINO's native 768)
        self.assertEqual(extractor.output_dim, 768)

        # Test patch_size property
        self.assertEqual(extractor.patch_size, 16)

    def test_invalid_model_raises_error(self):
        """Test that invalid model name raises ValueError."""
        config = SpectralLandmarkExtractorConfig(
            dino_model="dinov2_vitb16",  # Invalid - should be dinov3
            feature_source="attention_keys",
            lambda_knn=0.0,
            max_landmarks_per_image=3
        )

        with self.assertRaises(ValueError) as context:
            extractor = SpectralLandmarkExtractor(config)

        self.assertIn("DINOv3", str(context.exception))


if __name__ == "__main__":
    unittest.main()
