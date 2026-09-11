import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    query_crops,
)


def _azimuth_coded_pano(pano_w: int = 720, pano_h: int = 360) -> np.ndarray:
    """Pano whose red channel encodes column azimuth, green encodes row."""
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
    pano[:, :, 0] = np.arange(pano_w)[None, :] / pano_w
    pano[:, :, 1] = np.arange(pano_h)[:, None] / pano_h
    return pano


class CropConventionTest(unittest.TestCase):
    """The yaw round trip demanded by the plan before any evaluation."""

    def test_center_pixel_looks_along_crop_azimuth(self):
        pano = _azimuth_coded_pano()
        config = query_crops.CropRingConfig(n_crops=12, width=100, height=100)
        for az in (0.0, 30.0, 90.0, 210.0, 330.0):
            crop = query_crops.extract_crop(pano, config, az)
            center = crop[config.height // 2, config.width // 2]
            x_expected, y_expected = geometry.pano_px_from_direction(
                az, 0.0, pano.shape[1], pano.shape[0])
            self.assertAlmostEqual(center[0],
                                   (x_expected / pano.shape[1]) % 1.0,
                                   delta=0.01, msg=f"azimuth {az}")
            self.assertAlmostEqual(center[1], y_expected / pano.shape[0],
                                   delta=0.01, msg=f"azimuth {az}")

    def test_right_of_center_is_clockwise(self):
        pano = _azimuth_coded_pano()
        config = query_crops.CropRingConfig(width=100, height=100)
        crop = query_crops.extract_crop(pano, config, 90.0)
        row = crop[config.height // 2]
        # Larger azimuth (CW) means larger pano x means larger red value.
        self.assertGreater(row[-1][0], row[0][0])

    def test_top_of_crop_looks_up(self):
        pano = _azimuth_coded_pano()
        config = query_crops.CropRingConfig(width=100, height=100)
        crop = query_crops.extract_crop(pano, config, 0.0)
        # Up-positive elevation is smaller pano y (green).
        self.assertLess(crop[0, config.width // 2][1],
                        crop[-1, config.width // 2][1])

    def test_wraps_across_pano_seam(self):
        pano = _azimuth_coded_pano()
        config = query_crops.CropRingConfig(width=100, height=100)
        # Azimuth 180 is the pano seam (columns 0 / W-1 meet).
        crop = query_crops.extract_crop(pano, config, 180.0)
        self.assertTrue(np.isfinite(crop).all())
        row = crop[config.height // 2, :, 0]
        # Red wraps from ~1.0 down to ~0.0 across the seam: the left half of
        # the crop is near 1, the right half near 0.
        self.assertGreater(row[10], 0.85)
        self.assertLess(row[-10], 0.15)

    def test_ring_shape_and_uint8_round_trip(self):
        pano = (255 * _azimuth_coded_pano()).astype(np.uint8)
        config = query_crops.CropRingConfig(n_crops=12, width=64, height=64)
        ring = query_crops.extract_crop_ring(pano, config)
        self.assertEqual(ring.shape, (12, 64, 64, 3))
        self.assertEqual(ring.dtype, np.uint8)


class ImpliedHeadingTest(unittest.TestCase):
    def test_wraps(self):
        self.assertEqual(query_crops.implied_heading_cw_deg(90.0, 30.0), 60.0)
        self.assertEqual(query_crops.implied_heading_cw_deg(0.0, 30.0), 330.0)


if __name__ == "__main__":
    unittest.main()
