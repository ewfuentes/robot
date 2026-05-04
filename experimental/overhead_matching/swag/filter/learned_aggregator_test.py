import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch
import pandas as pd

from experimental.overhead_matching.swag.filter.learned_aggregator import (
    SigmaPolicy,
    LearnedAggregator,
    SIGMA_MIN,
    StreamRowStats,
    StepContext,
    HistoryEntry,
    HISTORY_DEPTH,
    NUM_HISTORY_ENTRY_FEATURES,
    compute_row_stats,
    extract_belief_features,
    history_to_tensor,
    FEATURE_DIM,
)


def _make_metadata(num_panos: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"pano_id": [f"p{i}" for i in range(num_panos)]})


class TestRowStats(unittest.TestCase):
    def test_features_match_expected_count(self):
        row = torch.linspace(-0.5, 0.9, 100)
        stats = compute_row_stats(row).to_tensor()
        self.assertEqual(stats.numel(), StreamRowStats.NUM_FEATURES)
        self.assertTrue(torch.isfinite(stats).all())

    def test_constant_row_yields_zero_top1_minus_top2(self):
        row = torch.zeros(50)
        stats = compute_row_stats(row)
        self.assertEqual(stats.top1_minus_top2.item(), 0.0)
        self.assertEqual(stats.top1_minus_top5.item(), 0.0)
        # softmax over a constant row is uniform → zero peak sharpness
        self.assertAlmostEqual(stats.softmax_peak_sharp.item(), 0.0, places=5)

    def test_handles_inf_entries(self):
        row = torch.tensor([0.1, 0.5, float("-inf"), 0.7, float("nan")])
        stats = compute_row_stats(row).to_tensor()
        self.assertTrue(torch.isfinite(stats).all())


class TestLearnedAggregator(unittest.TestCase):
    def _make_aggregator(self, num_panos: int = 3, num_patches: int = 100):
        torch.manual_seed(42)
        img_sim = torch.randn(num_panos, num_patches) * 0.3 + 0.2
        # First pano has constant landmark row (the "no-signal" fall-through case)
        lm_sim = torch.randn(num_panos, num_patches).abs()
        lm_sim[0] = 0.0
        meta = _make_metadata(num_panos)
        policy = SigmaPolicy()
        return LearnedAggregator(
            image_similarity_matrix=img_sim,
            landmark_similarity_matrix=lm_sim,
            panorama_metadata=meta,
            policy=policy,
            device=torch.device("cpu"),
        )

    def test_feature_dim_consistent(self):
        agg = self._make_aggregator()
        features = agg._compute_features(
            agg.image_similarity_matrix[1].to(agg.device),
            agg.landmark_similarity_matrix[1].to(agg.device),
        )
        self.assertEqual(features.numel(), FEATURE_DIM)

    def test_call_returns_correct_shape(self):
        agg = self._make_aggregator(num_panos=3, num_patches=100)
        out = agg("p1")
        self.assertEqual(out.shape, (100,))
        self.assertTrue(torch.isfinite(out).all())

    def test_constant_landmark_row_falls_back_to_image_only(self):
        """If the landmark row is constant, output should equal image-only
        log-softmax — no contribution from the landmark stream regardless of α.
        """
        agg = self._make_aggregator()
        img_row = agg.image_similarity_matrix[0].to(agg.device)
        lm_row = agg.landmark_similarity_matrix[0].to(agg.device)
        # Sanity: this row should be the all-zero fall-through case.
        self.assertEqual(lm_row.max().item(), lm_row.min().item())
        log_ll, sigma_img, _, _ = agg.fused_log_likelihood(img_row, lm_row)
        expected = torch.log_softmax(img_row / sigma_img, dim=0)
        self.assertTrue(torch.allclose(log_ll, expected, atol=1e-6))

    def test_gradient_flows_to_policy(self):
        """Gradients on a downstream loss must reach the policy parameters."""
        agg = self._make_aggregator()
        img_row = agg.image_similarity_matrix[1].to(agg.device)
        lm_row = agg.landmark_similarity_matrix[1].to(agg.device)
        log_ll, _, _, _ = agg.fused_log_likelihood(img_row, lm_row)
        loss = -log_ll[0]   # arbitrary scalar functional
        loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agg.policy.parameters()
        )
        self.assertTrue(any_grad, "no policy parameter received gradient")

    def test_sigma_outputs_respect_min(self):
        agg = self._make_aggregator()
        img_row = agg.image_similarity_matrix[1].to(agg.device)
        lm_row = agg.landmark_similarity_matrix[1].to(agg.device)
        _, sigma_img, sigma_lm, alpha = agg.fused_log_likelihood(img_row, lm_row)
        self.assertGreaterEqual(sigma_img.item(), SIGMA_MIN)
        self.assertGreaterEqual(sigma_lm.item(), SIGMA_MIN)
        self.assertGreaterEqual(alpha.item(), 0.0)
        self.assertLessEqual(alpha.item(), 1.0)

    def test_step_context_changes_features(self):
        agg = self._make_aggregator()
        img_row = agg.image_similarity_matrix[1].to(agg.device)
        lm_row = agg.landmark_similarity_matrix[1].to(agg.device)
        feats_default = agg._compute_features(img_row, lm_row)
        agg.set_step_context(StepContext(belief_entropy=2.0, norm_cum_distance=0.5))
        feats_with_ctx = agg._compute_features(img_row, lm_row)
        # Same row stats (first 2*NUM_ROW_FEATURES dims) but trailing context
        # features should change.
        n_row = 2 * StreamRowStats.NUM_FEATURES
        self.assertTrue(torch.equal(feats_default[:n_row], feats_with_ctx[:n_row]))
        self.assertFalse(torch.equal(feats_default[n_row:], feats_with_ctx[n_row:]))


class TestHistory(unittest.TestCase):
    def test_history_tensor_zero_padded_when_empty(self):
        out = history_to_tensor(None, torch.device("cpu"), torch.float32)
        self.assertEqual(out.numel(), HISTORY_DEPTH * NUM_HISTORY_ENTRY_FEATURES)
        self.assertEqual(out.abs().sum().item(), 0.0)

    def test_history_tensor_left_pads_partial_history(self):
        h = [HistoryEntry(belief_entropy=1.0, last_obs_max_log_p=0.5)]
        out = history_to_tensor(h, torch.device("cpu"), torch.float32)
        self.assertEqual(out.numel(), HISTORY_DEPTH * NUM_HISTORY_ENTRY_FEATURES)
        # Newest entry sits at the *end* of the flattened tensor.
        last_entry_start = (HISTORY_DEPTH - 1) * NUM_HISTORY_ENTRY_FEATURES
        self.assertAlmostEqual(out[last_entry_start].item(), 1.0, places=5)
        self.assertAlmostEqual(
            out[last_entry_start + NUM_HISTORY_ENTRY_FEATURES - 1].item(), 0.5, places=5
        )
        # Everything before the last entry is zero (left-padding).
        self.assertEqual(out[:last_entry_start].abs().sum().item(), 0.0)

    def test_history_tensor_truncates_long_history(self):
        h = [HistoryEntry(belief_entropy=float(i)) for i in range(HISTORY_DEPTH + 5)]
        out = history_to_tensor(h, torch.device("cpu"), torch.float32)
        # Only the last HISTORY_DEPTH entries kept; entropy of newest = HISTORY_DEPTH+4.
        last_entry_start = (HISTORY_DEPTH - 1) * NUM_HISTORY_ENTRY_FEATURES
        self.assertAlmostEqual(
            out[last_entry_start].item(), float(HISTORY_DEPTH + 4), places=4
        )
        # Oldest kept entry corresponds to index `5` (truncation drops 0..4).
        self.assertAlmostEqual(out[0].item(), 5.0, places=4)

    def test_feature_dim_includes_history(self):
        # Net input dim must equal current scalars + history window.
        n_row = 2 * StreamRowStats.NUM_FEATURES
        n_ctx = 4 + 3  # NUM_BELIEF_FEATURES + NUM_STEP_FEATURES
        expected = n_row + n_ctx + HISTORY_DEPTH * NUM_HISTORY_ENTRY_FEATURES
        self.assertEqual(FEATURE_DIM, expected)


if __name__ == "__main__":
    unittest.main()
