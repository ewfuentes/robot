import math
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch
import pandas as pd

from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    SafaPlusNormalizedLandmarkAggregator,
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

    def _make(self, img_sim, lm_sim, image_sigma=0.187, landmark_sigma=0.72):
        meta = _make_metadata(img_sim.shape[0])
        return SafaPlusNormalizedLandmarkAggregator(
            image_similarity_matrix=img_sim,
            landmark_similarity_matrix=lm_sim,
            panorama_metadata=meta,
            image_sigma=image_sigma,
            landmark_sigma=landmark_sigma,
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
        # The image-only fall-through path goes through ``_replace_nan_with_zero``,
        # which mutates the input in-place. Re-derive the expected value the
        # same way to avoid a stale comparison.
        expected = torch.where(torch.isnan(expected), torch.zeros_like(expected), expected)
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
        expected = torch.where(torch.isnan(expected), torch.zeros_like(expected), expected)
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
        log_p_img = torch.where(
            torch.isnan(log_p_img), torch.zeros_like(log_p_img), log_p_img
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
        log_p_img = torch.where(
            torch.isnan(log_p_img), torch.zeros_like(log_p_img), log_p_img
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


if __name__ == "__main__":
    unittest.main()
