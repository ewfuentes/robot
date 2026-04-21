"""Tests for the landmark correspondence dataset (text-only variant)."""

import functools
import math
import unittest

import common.torch.load_torch_deps  # noqa: F401

import torch

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    CorrespondenceBatch,
    CorrespondencePair,
    HOUSENUMBER_KEY,
    LandmarkCorrespondenceDataset,
    _pad_side,
    collate_correspondence,
    compute_cross_features,
    encode_tag_bundle,
    parse_housenumber,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    NUM_CROSS_FEATURES,
    TAG_KEY_TO_IDX,
)


# The smallest key that actually exists in TAG_KEY_TO_IDX, used for tests.
_SOME_KEY = next(iter(TAG_KEY_TO_IDX))
_NAME_KEY = "name" if "name" in TAG_KEY_TO_IDX else _SOME_KEY


class TestParseHousenumber(unittest.TestCase):
    def test_range(self):
        self.assertEqual(parse_housenumber("665-667"), (665.0, 667.0))

    def test_single(self):
        self.assertEqual(parse_housenumber("1858"), (1858.0, 1858.0))

    def test_alpha_suffix_returns_nan(self):
        lo, hi = parse_housenumber("512D")
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_none_returns_nan(self):
        lo, hi = parse_housenumber(None)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_int_returns_nan(self):
        lo, hi = parse_housenumber(123)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))


class TestEncodeTagBundle(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.emb = {"cafe": torch.ones(self.dim), "bar": torch.zeros(self.dim)}

    def test_encodes_known_values(self):
        encoded = encode_tag_bundle(
            {_SOME_KEY: "cafe"}, self.emb, self.dim,
        )
        self.assertEqual(encoded["key_indices"], [TAG_KEY_TO_IDX[_SOME_KEY]])
        self.assertEqual(len(encoded["text_embeddings"]), 1)

    def test_skips_unknown_keys(self):
        encoded = encode_tag_bundle(
            {"bogus_key_not_in_whitelist": "cafe"}, self.emb, self.dim,
        )
        self.assertEqual(encoded["key_indices"], [])
        self.assertEqual(encoded["text_embeddings"], [])

    def test_raises_on_missing_value_by_default(self):
        with self.assertRaises(KeyError):
            encode_tag_bundle(
                {_SOME_KEY: "unknown_value"}, self.emb, self.dim,
            )

    def test_falls_back_to_zeros_when_allowed(self):
        encoded = encode_tag_bundle(
            {_SOME_KEY: "unknown_value"}, self.emb, self.dim,
            allow_missing_text_embeddings=True,
        )
        self.assertEqual(len(encoded["text_embeddings"]), 1)
        self.assertTrue(torch.allclose(
            encoded["text_embeddings"][0], torch.zeros(self.dim),
        ))


class TestComputeCrossFeatures(unittest.TestCase):
    def test_feature_count_and_none_branch(self):
        feats = compute_cross_features({}, {}, text_embeddings=None)
        self.assertEqual(len(feats), NUM_CROSS_FEATURES)
        self.assertEqual(feats, [0.0, 0.0, 0.0, 0.0])

    def test_text_similarity_features_when_both_present(self):
        emb = {
            "cafe": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "coffee": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        feats = compute_cross_features(
            {_SOME_KEY: "cafe"}, {_SOME_KEY: "coffee"}, text_embeddings=emb,
        )
        # max and mean should be ~1.0 (identical vectors = perfect cosine sim)
        self.assertAlmostEqual(feats[0], 1.0, places=5)
        self.assertAlmostEqual(feats[1], 1.0, places=5)

    def test_housenumber_overlap_exact(self):
        # Range 100-110 overlaps single value 105.
        feats = compute_cross_features(
            {HOUSENUMBER_KEY: "100-110"},
            {HOUSENUMBER_KEY: "105"},
            text_embeddings=None,
        )
        self.assertEqual(feats[-1], 1.0)

    def test_housenumber_no_overlap(self):
        feats = compute_cross_features(
            {HOUSENUMBER_KEY: "100-110"},
            {HOUSENUMBER_KEY: "200"},
            text_embeddings=None,
        )
        self.assertEqual(feats[-1], 0.0)

    def test_housenumber_alpha_is_zero(self):
        # Alpha suffix returns nan; cross feature falls back to 0.
        feats = compute_cross_features(
            {HOUSENUMBER_KEY: "512D"},
            {HOUSENUMBER_KEY: "500-520"},
            text_embeddings=None,
        )
        self.assertEqual(feats[-1], 0.0)


class TestPadSide(unittest.TestCase):
    def test_min_length_one(self):
        keys, txt, mask = _pad_side(
            [{"key_indices": [], "text_embeddings": []}], text_input_dim=3,
        )
        self.assertEqual(keys.shape, (1, 1))
        self.assertEqual(txt.shape, (1, 1, 3))
        self.assertEqual(mask.shape, (1, 1))
        self.assertFalse(bool(mask[0, 0]))

    def test_padding_matches_max(self):
        encoded_list = [
            {"key_indices": [0, 1], "text_embeddings": [torch.ones(3), torch.zeros(3)]},
            {"key_indices": [0], "text_embeddings": [torch.ones(3)]},
        ]
        keys, txt, mask = _pad_side(encoded_list, text_input_dim=3)
        self.assertEqual(keys.shape, (2, 2))
        self.assertTrue(bool(mask[0, 0]) and bool(mask[0, 1]))
        self.assertTrue(bool(mask[1, 0]))
        self.assertFalse(bool(mask[1, 1]))


class TestCollateCorrespondence(unittest.TestCase):
    def test_mixed_batch_with_empty_pano_side(self):
        # First sample has zero pano tags, second has one. Collate must not
        # fall back to a hardcoded dim.
        collate = functools.partial(collate_correspondence, text_input_dim=3)
        sample_empty = {
            "pano": {"key_indices": [], "text_embeddings": []},
            "osm": {"key_indices": [TAG_KEY_TO_IDX[_SOME_KEY]],
                    "text_embeddings": [torch.ones(3)]},
            "cross_features": [0.0] * NUM_CROSS_FEATURES,
            "label": 0.0,
        }
        sample_full = {
            "pano": {"key_indices": [TAG_KEY_TO_IDX[_SOME_KEY]],
                     "text_embeddings": [torch.ones(3)]},
            "osm": {"key_indices": [TAG_KEY_TO_IDX[_SOME_KEY]],
                    "text_embeddings": [torch.ones(3)]},
            "cross_features": [0.0] * NUM_CROSS_FEATURES,
            "label": 1.0,
        }
        batch = collate([sample_empty, sample_full])
        self.assertIsInstance(batch, CorrespondenceBatch)
        self.assertEqual(batch.pano_text_embeddings.shape[-1], 3)
        self.assertEqual(batch.osm_text_embeddings.shape[-1], 3)
        # Labels should come through as float tensor.
        torch.testing.assert_close(batch.labels, torch.tensor([0.0, 1.0]))


class TestCorrespondenceBatchTo(unittest.TestCase):
    def _make_batch(self):
        B, T, D = 2, 1, 3
        return CorrespondenceBatch(
            pano_key_indices=torch.zeros(B, T, dtype=torch.long),
            pano_text_embeddings=torch.zeros(B, T, D),
            pano_tag_mask=torch.zeros(B, T, dtype=torch.bool),
            osm_key_indices=torch.zeros(B, T, dtype=torch.long),
            osm_text_embeddings=torch.zeros(B, T, D),
            osm_tag_mask=torch.zeros(B, T, dtype=torch.bool),
            cross_features=torch.zeros(B, NUM_CROSS_FEATURES),
            labels=torch.zeros(B),
        )

    def test_to_uses_dataclass_fields(self):
        batch = self._make_batch()
        moved = batch.to("cpu")
        self.assertIsInstance(moved, CorrespondenceBatch)
        # All tensor fields preserved.
        for name in batch.__dataclass_fields__:
            self.assertTrue(torch.equal(getattr(batch, name), getattr(moved, name)))


class TestLandmarkCorrespondenceDataset(unittest.TestCase):
    def _make_pairs(self, values_in_emb: bool):
        val = "cafe" if values_in_emb else "missing_value"
        return [
            CorrespondencePair(
                pano_tags={_SOME_KEY: val},
                osm_tags={_SOME_KEY: val},
                label=1.0, difficulty="positive",
                uniqueness_score=None, pano_id="p0",
            ),
            CorrespondencePair(
                pano_tags={_SOME_KEY: val},
                osm_tags={_SOME_KEY: val},
                label=0.0, difficulty="hard",
                uniqueness_score=None, pano_id="p0",
            ),
        ]

    def test_raises_by_default_on_missing_embeddings(self):
        pairs = self._make_pairs(values_in_emb=False)
        with self.assertRaises(KeyError):
            LandmarkCorrespondenceDataset(
                pairs, text_embeddings={}, text_input_dim=4,
                include_difficulties=("positive", "hard"),
            )

    def test_include_difficulties_filters_pairs(self):
        pairs = self._make_pairs(values_in_emb=True)
        emb = {"cafe": torch.ones(4)}
        ds = LandmarkCorrespondenceDataset(
            pairs, text_embeddings=emb, text_input_dim=4,
            include_difficulties=("positive",),
        )
        self.assertEqual(len(ds), 1)

    def test_allow_missing_does_not_raise(self):
        pairs = self._make_pairs(values_in_emb=False)
        ds = LandmarkCorrespondenceDataset(
            pairs, text_embeddings={}, text_input_dim=4,
            include_difficulties=("positive",),
            allow_missing_text_embeddings=True,
        )
        sample = ds[0]
        self.assertIn("pano", sample)
        self.assertIn("osm", sample)
        self.assertEqual(len(sample["cross_features"]), NUM_CROSS_FEATURES)


if __name__ == "__main__":
    unittest.main()
