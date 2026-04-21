"""Tests for correspondence_matching.py.

Covers the shape/index contract of `precompute_raw_cost_data` and the
`compute_pairs_cost_matrix` helper, using a hand-built fake VigorDataset
stand-in.
"""

import unittest
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import pandas as pd
import torch

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    NUM_CROSS_FEATURES,
)
from experimental.overhead_matching.swag.evaluation import correspondence_matching as cm
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TAG_KEY_TO_IDX,
    TagBundleEncoderConfig,
)


_SOME_KEY = next(iter(TAG_KEY_TO_IDX))


def _build_fake_dataset(pano_ids, osm_landmarks, sat_to_landmark_idxs):
    """Build a minimal VigorDataset-like SimpleNamespace for the precompute path.

    Args:
        pano_ids: list[str] of panorama IDs.
        osm_landmarks: list[dict[str, str]] — each maps tag key to value;
            indexed by landmark_idx in the landmark metadata.
        sat_to_landmark_idxs: list[list[int]] — one entry per satellite,
            listing the landmark indices it covers.
    """
    pano_df = pd.DataFrame({"pano_id": pano_ids})
    landmark_df = pd.DataFrame({
        "pruned_props": [frozenset(tags.items()) for tags in osm_landmarks],
    })
    sat_df = pd.DataFrame({"landmark_idxs": [list(x) for x in sat_to_landmark_idxs]})
    return SimpleNamespace(
        _panorama_metadata=pano_df,
        _landmark_metadata=landmark_df,
        _satellite_metadata=sat_df,
    )


def _make_text_embeddings(values, dim):
    torch.manual_seed(0)
    return {v: torch.randn(dim) for v in values}


class TestPrecomputeRawCostData(unittest.TestCase):
    def setUp(self):
        self.text_input_dim = 8
        self.enc_cfg = TagBundleEncoderConfig(
            text_input_dim=self.text_input_dim, text_proj_dim=4, per_tag_dim=4,
        )
        self.cfg = CorrespondenceClassifierConfig(
            encoder=self.enc_cfg, mlp_hidden_dim=4, dropout=0.0,
        )
        self.device = torch.device("cpu")
        self.model = CorrespondenceClassifier(self.cfg).to(self.device).eval()

        self.osm_landmarks = [
            {_SOME_KEY: "cafe"},
            {_SOME_KEY: "bar"},
            {_SOME_KEY: "library"},
        ]
        # Three panoramas, each with 2 landmarks.
        self.pano_ids = ["p0", "p1", "p2"]
        self.pano_tags_from_pano_id = {
            "p0": [{"tags": [(_SOME_KEY, "cafe")]}, {"tags": [(_SOME_KEY, "bar")]}],
            "p1": [{"tags": [(_SOME_KEY, "cafe")]}, {"tags": [(_SOME_KEY, "library")]}],
            "p2": [{"tags": [(_SOME_KEY, "bar")]}, {"tags": [(_SOME_KEY, "cafe")]}],
        }
        self.sat_to_landmark_idxs = [[0, 1], [1, 2]]

        self.text_embeddings = _make_text_embeddings(
            ["cafe", "bar", "library"], self.text_input_dim,
        )

        self.dataset = _build_fake_dataset(
            self.pano_ids, self.osm_landmarks, self.sat_to_landmark_idxs,
        )

    def test_shape_and_index_contract(self):
        raw = cm.precompute_raw_cost_data(
            model=self.model,
            text_embeddings=self.text_embeddings,
            text_input_dim=self.text_input_dim,
            dataset=self.dataset,
            pano_tags_from_pano_id=self.pano_tags_from_pano_id,
            device=self.device,
        )

        total_pano_lm = sum(
            len(self.pano_tags_from_pano_id[p]) for p in self.pano_ids
        )
        n_osm = len(self.osm_landmarks)

        self.assertEqual(raw.cost_matrix.shape, (total_pano_lm, n_osm))
        self.assertEqual(len(raw.osm_lm_tags), n_osm)
        self.assertEqual(len(raw.osm_lm_indices), n_osm)
        self.assertEqual(len(raw.pano_lm_tags), total_pano_lm)

        all_rows = [r for rows in raw.pano_id_to_lm_rows.values() for r in rows]
        self.assertEqual(len(all_rows), total_pano_lm)
        self.assertTrue(all(0 <= r < total_pano_lm for r in all_rows))

        # All cost matrix values are valid probabilities.
        self.assertTrue(np.all(raw.cost_matrix >= 0.0))
        self.assertTrue(np.all(raw.cost_matrix <= 1.0))
        self.assertTrue(np.isfinite(raw.cost_matrix).all())

    def test_similarity_from_raw_data_shape(self):
        raw = cm.precompute_raw_cost_data(
            model=self.model,
            text_embeddings=self.text_embeddings,
            text_input_dim=self.text_input_dim,
            dataset=self.dataset,
            pano_tags_from_pano_id=self.pano_tags_from_pano_id,
            device=self.device,
        )
        similarity = cm.similarity_from_raw_data(
            raw, self.dataset,
            cm.MatchingMethod.HUNGARIAN,
            cm.AggregationMode.SUM,
            prob_threshold=0.0,
        )
        self.assertEqual(
            similarity.shape,
            (len(self.pano_ids), len(self.sat_to_landmark_idxs)),
        )
        self.assertTrue(torch.isfinite(similarity).all())


class TestComputePairsCostMatrix(unittest.TestCase):
    def test_small_pairs(self):
        text_input_dim = 8
        enc_cfg = TagBundleEncoderConfig(
            text_input_dim=text_input_dim, text_proj_dim=4, per_tag_dim=4,
        )
        cfg = CorrespondenceClassifierConfig(
            encoder=enc_cfg, mlp_hidden_dim=4, dropout=0.0,
        )
        model = CorrespondenceClassifier(cfg).eval()
        text_embeddings = _make_text_embeddings(
            ["cafe", "bar"], text_input_dim,
        )

        pano_tags = [{_SOME_KEY: "cafe"}, {_SOME_KEY: "bar"}]
        osm_tags = [{_SOME_KEY: "cafe"}, {_SOME_KEY: "bar"}, {_SOME_KEY: "cafe"}]

        cost = cm.compute_pairs_cost_matrix(
            pano_tags, osm_tags, model, text_embeddings, text_input_dim,
            device=torch.device("cpu"),
        )
        self.assertEqual(cost.shape, (2, 3))
        self.assertTrue(np.all((cost >= 0.0) & (cost <= 1.0)))


class TestCrossFeatureVectorization(unittest.TestCase):
    def test_gpu_matches_python_loop(self):
        torch.manual_seed(42)
        text_input_dim = 16
        # A handful of known values; also exercise missing-embedding case via
        # tag values not present in the dict.
        text_embeddings = {
            "cafe": torch.randn(text_input_dim),
            "bar": torch.randn(text_input_dim),
            "library": torch.randn(text_input_dim),
            "park": torch.randn(text_input_dim),
        }
        some_key = _SOME_KEY
        name_key = "name" if "name" in TAG_KEY_TO_IDX else some_key

        pano_tags_list = [
            {some_key: "cafe", name_key: "cafe"},
            {some_key: "bar"},
            {"addr:housenumber": "100-110"},
            {},
        ]
        osm_tags_list = [
            {some_key: "cafe", name_key: "cafe"},
            {some_key: "library"},
            {"addr:housenumber": "105"},
            {some_key: "not_in_embeddings"},
            {name_key: "park"},
        ]

        device = torch.device("cpu")
        cpu = cm._compute_cross_features_for_pairs(
            pano_tags_list, osm_tags_list, text_embeddings,
        )
        gpu = cm._compute_cross_features_gpu(
            pano_tags_list, osm_tags_list, text_embeddings,
            text_input_dim, device,
        )
        np.testing.assert_allclose(gpu, cpu, atol=1e-5, rtol=1e-5)


class TestMatchAndAggregate(unittest.TestCase):
    def test_empty_cost_matrix_returns_zero_score(self):
        result = cm.match_and_aggregate(
            np.zeros((0, 5)), cm.MatchingMethod.HUNGARIAN,
            cm.AggregationMode.SUM,
        )
        self.assertEqual(result.similarity_score, 0.0)

    def test_threshold_excludes_low_prob_pairs(self):
        cost = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        result = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.5,
        )
        self.assertEqual(len(result.match_probs), 2)
        self.assertAlmostEqual(result.similarity_score, 1.8, places=5)


class TestDustbinHungarian(unittest.TestCase):
    """Dustbin-augmented Hungarian should beat post-hoc thresholding when
    an unforced 1-to-1 assignment saddles a row with a sub-threshold match
    to free up a globally-optimal pairing.
    """

    def test_dustbin_routes_weak_rows_to_sink(self):
        # pano0 has one strong candidate (0.90) and one weak (0.50); pano1
        # only has weak options (0.70 and 0.75). With threshold 0.80 only
        # pano0→osm0 (0.90) is acceptable.
        #
        # Post-hoc threshold (use_dustbin=False) still lets Hungarian force
        # pano1→osm1 (0.75) temporarily, then drops it. pano0→osm0 survives.
        # Dustbin should produce the same set of kept matches.
        cost = np.array([[0.90, 0.50], [0.70, 0.75]], dtype=np.float32)

        without = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.80, use_dustbin=False,
        )
        with_dustbin = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.80, use_dustbin=True,
        )
        np.testing.assert_allclose(without.match_probs, [0.9], rtol=1e-5)
        np.testing.assert_allclose(with_dustbin.match_probs, [0.9], rtol=1e-5)

    def test_dustbin_prevents_saddling(self):
        # Classic failure case: the forced 1-to-1 assignment that maximizes
        # total probability pairs pano0 with its 2nd choice so pano1 can
        # take pano0's 1st choice, even though both end up sub-threshold.
        # Post-hoc threshold drops both; dustbin keeps pano0's 1st choice
        # and sends pano1 to the sink.
        cost = np.array([
            [0.85, 0.82],  # pano0 prefers osm0 (0.85), osm1 also ok (0.82)
            [0.84, 0.10],  # pano1 needs osm0 but osm1 is way below
        ], dtype=np.float32)
        # Hungarian (maximize total) picks 0.82 + 0.84 = 1.66 over 0.85 + 0.10 = 0.95.
        without = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.83, use_dustbin=False,
        )
        with_dustbin = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.83, use_dustbin=True,
        )
        # Without dustbin: forced assignment gives (pano0→osm1=0.82, pano1→osm0=0.84);
        # 0.82 < 0.83 → dropped; 0.84 ≥ 0.83 → kept. Only pano1 survives.
        np.testing.assert_allclose(sorted(without.match_probs), [0.84], rtol=1e-5)
        # With dustbin: should pick the >threshold match (pano0→osm0=0.85)
        # and sink pano1.
        np.testing.assert_allclose(
            sorted(with_dustbin.match_probs), [0.85], rtol=1e-5,
        )

    def test_dustbin_doesnt_fabricate_matches(self):
        # All entries below threshold → all rows go to dustbin, zero matches.
        cost = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = cm.match_and_aggregate(
            cost, cm.MatchingMethod.HUNGARIAN, cm.AggregationMode.SUM,
            prob_threshold=0.5, use_dustbin=True,
        )
        self.assertEqual(result.match_probs, [])
        self.assertEqual(result.similarity_score, 0.0)


if __name__ == "__main__":
    unittest.main()
