import math
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch
import pandas as pd

from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    SafaPlusNormalizedLandmarkAggregator,
    SafaPlusNormalizedLandmarkAggregatorConfig,
)
from experimental.overhead_matching.swag.filter.particle_filter import (
    wag_observation_log_likelihood_from_similarity_matrix,
)


def _make_metadata(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"pano_id": [f"p{i}" for i in range(n)]})


class TestSafaPlusNormalizedLandmarkAggregator(unittest.TestCase):
    """The new aggregator's contract is: per-patch ``log p(z|patch_j)`` =
    SAFA Gaussian-on-residuals (image) + Gaussian-on-r_norm (landmark),
    with image-only fall-through on all-zero / constant landmark rows."""

    def _make(self, img_sim, lm_sim, image_sigma=0.187, landmark_sigma=0.72,
              landmark_use_raw_residual=False):
        meta = _make_metadata(img_sim.shape[0])
        return SafaPlusNormalizedLandmarkAggregator(
            image_similarity_matrix=img_sim,
            landmark_similarity_matrix=lm_sim,
            panorama_metadata=meta,
            image_sigma=image_sigma,
            landmark_sigma=landmark_sigma,
            landmark_use_raw_residual=landmark_use_raw_residual,
            device=torch.device("cpu"),
        )

    def test_all_zero_landmark_row_falls_through_to_image_only(self):
        torch.manual_seed(0)
        img_sim = torch.randn(2, 50) * 0.3 + 0.4
        lm_sim = torch.zeros(2, 50)        # row 0 is all-zero
        lm_sim[1] = torch.rand(50)         # row 1 has signal
        agg = self._make(img_sim, lm_sim)

        out_zero = agg("p0")
        expected = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim[0], 0.187
        )
        self.assertTrue(torch.allclose(out_zero, expected, atol=1e-6),
                        "all-zero landmark row must equal image-only log_p")

    def test_constant_landmark_row_falls_through_to_image_only(self):
        """Even a non-zero constant row carries no information after / row_max
        normalization (every entry has r_norm = 0), so we route to image-only
        rather than divide-by-anything fragile."""
        img_sim = torch.randn(1, 30) * 0.3 + 0.4
        lm_sim = torch.full((1, 30), 0.42)
        agg = self._make(img_sim, lm_sim)

        out = agg("p0")
        expected = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim[0], 0.187
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_patch_at_row_max_gets_zero_landmark_residual(self):
        """If sim_lm[j*] == row_max, then r_norm[j*] = 0 and log_p_lm[j*] =
        -log(σ_lm √2π). The fused log_p[j*] = log_p_img[j*] + that constant.
        """
        img_sim = torch.zeros(1, 5)
        lm_sim = torch.tensor([[0.0, 0.3, 1.0, 0.7, 0.0]])  # row_max = 1.0 at idx 2
        sigma_lm = 0.5
        agg = self._make(img_sim, lm_sim, landmark_sigma=sigma_lm)

        out = agg("p0")
        log_p_img = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim[0], 0.187
        )
        expected_lm_at_max = -math.log(sigma_lm * math.sqrt(2 * math.pi))
        self.assertAlmostEqual(
            (out[2] - log_p_img[2]).item(), expected_lm_at_max, places=4
        )

    def test_landmark_residual_decreases_likelihood_quadratically(self):
        """log_p_lm[j] − log_p_lm[j*] should be exactly −0.5 (r_norm / σ_lm)^2."""
        img_sim = torch.zeros(1, 4)
        lm_sim = torch.tensor([[0.0, 0.5, 1.0, 0.25]])  # row_max = 1.0 at idx 2
        sigma_lm = 0.3
        agg = self._make(img_sim, lm_sim, landmark_sigma=sigma_lm)
        out = agg("p0")
        log_p_img = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim[0], 0.187
        )
        log_p_lm = out - log_p_img

        # Reference at idx 2 (row_max → r_norm=0):
        ref = log_p_lm[2]
        for j, expected_r_norm in [(0, 1.0), (1, 0.5), (3, 0.75)]:
            expected_drop = -0.5 * (expected_r_norm / sigma_lm) ** 2
            self.assertAlmostEqual(
                (log_p_lm[j] - ref).item(), expected_drop, places=4,
                msg=f"residual at idx {j} should be −0.5·(r/σ)^2",
            )

    def test_raw_residual_mode_equals_sum_of_safa_streams(self):
        """With landmark_use_raw_residual=True both streams use the SAFA
        helper, so the output must equal the elementwise sum of two
        wag_observation_log_likelihood_from_similarity_matrix calls."""
        torch.manual_seed(0)
        img_sim = torch.randn(1, 12) * 0.3 + 0.4
        lm_sim = torch.randn(1, 12) * 0.3 + 0.4
        sigma_img, sigma_lm = 0.2, 0.25
        agg = self._make(img_sim, lm_sim, image_sigma=sigma_img,
                         landmark_sigma=sigma_lm,
                         landmark_use_raw_residual=True)
        out = agg("p0")

        expected = (
            wag_observation_log_likelihood_from_similarity_matrix(img_sim[0], sigma_img)
            + wag_observation_log_likelihood_from_similarity_matrix(lm_sim[0], sigma_lm)
        )
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_raw_residual_mode_zeroes_nan_landmark_entries(self):
        """NaN landmark entries must contribute 0 to the fused log-likelihood
        (image-only at those indices) without poisoning the helper's .max()."""
        img_sim = torch.zeros(1, 5)
        lm_sim = torch.tensor([[0.1, float("nan"), 0.7, 0.3, float("nan")]])
        sigma_img, sigma_lm = 0.187, 0.2
        agg = self._make(img_sim, lm_sim, image_sigma=sigma_img,
                         landmark_sigma=sigma_lm,
                         landmark_use_raw_residual=True)
        out = agg("p0")

        log_p_img = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim[0], sigma_img
        )
        # At NaN positions: out should equal image-only.
        self.assertTrue(torch.allclose(out[1], log_p_img[1], atol=1e-6))
        self.assertTrue(torch.allclose(out[4], log_p_img[4], atol=1e-6))
        self.assertTrue(torch.isfinite(out).all())

    def test_raises_on_nan_image_sim(self):
        """Non-finite image similarity must raise rather than silently zero."""
        img_sim = torch.zeros(1, 5)
        img_sim[0, 2] = float("nan")
        lm_sim = torch.rand(1, 5)
        agg = self._make(img_sim, lm_sim)
        with self.assertRaises(RuntimeError):
            agg("p0")

    def test_config_rejects_nonpositive_sigma(self):
        with self.assertRaises(ValueError):
            SafaPlusNormalizedLandmarkAggregatorConfig(
                image_similarity_matrix_path="/tmp/img.pt",
                landmark_similarity_matrix_path="/tmp/lm.pt",
                image_sigma=0.0,
                landmark_sigma=0.5,
            )
        with self.assertRaises(ValueError):
            SafaPlusNormalizedLandmarkAggregatorConfig(
                image_similarity_matrix_path="/tmp/img.pt",
                landmark_similarity_matrix_path="/tmp/lm.pt",
                image_sigma=0.5,
                landmark_sigma=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
