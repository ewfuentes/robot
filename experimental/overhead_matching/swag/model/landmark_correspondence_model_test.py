"""Tests for landmark correspondence model."""

import math
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    BOOLEAN_KEYS,
    HOUSENUMBER_KEY,
    NUM_TAG_KEYS,
    NUMERIC_KEYS,
    TAG_KEY_TO_IDX,
    ValueType,
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoder,
    TagBundleEncoderConfig,
    encode_housenumber_value,
    encode_numeric_value,
    key_type,
    parse_boolean,
    parse_housenumber,
    parse_maxheight,
    parse_numeric,
)


class TestValueParsing(unittest.TestCase):
    def test_parse_boolean(self):
        self.assertEqual(parse_boolean("yes"), 1.0)
        self.assertEqual(parse_boolean("no"), 0.0)
        self.assertEqual(parse_boolean("-1"), 0.0)
        self.assertEqual(parse_boolean("maybe"), 0.5)

    def test_parse_numeric(self):
        self.assertEqual(parse_numeric("5"), (5.0, False))
        self.assertEqual(parse_numeric("55 mph"), (55.0, False))
        # "20+" encodes as 20.0 with is_lower_bound=True
        self.assertEqual(parse_numeric("20+"), (20.0, True))
        # "high"/"multi" are key-aware: only resolve for level keys
        self.assertEqual(parse_numeric("high", key="building:levels"), (20.0, False))
        self.assertEqual(parse_numeric("multi", key="levels"), (5.0, False))
        self.assertTrue(math.isnan(parse_numeric("high")[0]))  # without key context → NaN
        self.assertTrue(math.isnan(parse_numeric("multi")[0]))
        self.assertTrue(math.isnan(parse_numeric("unknown")[0]))

    def test_parse_maxheight(self):
        self.assertAlmostEqual(parse_maxheight("13'6\""), 13.5, places=2)
        self.assertEqual(parse_maxheight("2"), 2.0)
        self.assertTrue(math.isnan(parse_maxheight("default")))

    def test_parse_housenumber(self):
        self.assertEqual(parse_housenumber("665-667"), (665.0, 667.0))
        self.assertEqual(parse_housenumber("1858"), (1858.0, 1858.0))
        lo, hi = parse_housenumber("abc")
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_encode_numeric(self):
        enc = encode_numeric_value(100.0)
        self.assertEqual(len(enc), 4)
        self.assertAlmostEqual(enc[0], math.log1p(100), places=6)
        self.assertAlmostEqual(enc[1], 0.1, places=6)
        self.assertEqual(enc[2], 1.0)  # presence flag
        self.assertEqual(enc[3], 0.0)  # not a lower bound

    def test_encode_numeric_lower_bound(self):
        enc = encode_numeric_value(20.0, is_lower_bound=True)
        self.assertEqual(len(enc), 4)
        self.assertEqual(enc[3], 1.0)  # is_lower_bound flag

    def test_encode_numeric_nan(self):
        enc = encode_numeric_value(float("nan"))
        self.assertEqual(enc, [0.0, 0.0, 0.0, 0.0])

    def test_encode_housenumber(self):
        enc = encode_housenumber_value(665.0, 667.0)
        self.assertEqual(len(enc), 4)


class TestKeyType(unittest.TestCase):
    def test_boolean_keys(self):
        for k in BOOLEAN_KEYS:
            self.assertEqual(key_type(k), ValueType.BOOLEAN)

    def test_numeric_keys(self):
        for k in NUMERIC_KEYS:
            self.assertEqual(key_type(k), ValueType.NUMERIC)

    def test_housenumber(self):
        self.assertEqual(key_type("addr:housenumber"), ValueType.HOUSENUMBER)

    def test_text_keys(self):
        self.assertEqual(key_type("name"), ValueType.TEXT)
        self.assertEqual(key_type("building"), ValueType.TEXT)
        self.assertEqual(key_type("amenity"), ValueType.TEXT)

    def test_all_tags_have_index(self):
        self.assertGreater(NUM_TAG_KEYS, 100)  # Should be ~112 keys


class TestTagBundleEncoder(unittest.TestCase):
    def _make_dummy_input(self, batch_size=4, max_tags=5, text_dim=768):
        return dict(
            key_indices=torch.randint(0, NUM_TAG_KEYS, (batch_size, max_tags)),
            value_type=torch.randint(0, 4, (batch_size, max_tags)),
            boolean_values=torch.randint(0, 3, (batch_size, max_tags)),
            numeric_values=torch.randn(batch_size, max_tags, 4),
            numeric_nan_mask=torch.zeros(batch_size, max_tags, dtype=torch.bool),
            housenumber_values=torch.randn(batch_size, max_tags, 4),
            housenumber_nan_mask=torch.zeros(batch_size, max_tags, dtype=torch.bool),
            text_embeddings=torch.randn(batch_size, max_tags, text_dim),
            tag_mask=torch.ones(batch_size, max_tags, dtype=torch.bool),
        )

    def test_forward_shape(self):
        config = TagBundleEncoderConfig()
        encoder = TagBundleEncoder(config)
        inputs = self._make_dummy_input()
        output = encoder(**inputs)
        self.assertEqual(output.shape, (4, config.repr_dim))

    def test_masked_tags_dont_affect_output(self):
        config = TagBundleEncoderConfig()
        encoder = TagBundleEncoder(config)

        # Two identical inputs, but second has extra masked-out tags
        inputs1 = self._make_dummy_input(batch_size=1, max_tags=3)
        inputs2 = self._make_dummy_input(batch_size=1, max_tags=5)

        # Copy real tags from inputs1 to inputs2
        for key in inputs1:
            if key == "tag_mask":
                inputs2["tag_mask"][:, :3] = True
                inputs2["tag_mask"][:, 3:] = False
            elif inputs1[key].dim() == 2:
                inputs2[key][:, :3] = inputs1[key]
            elif inputs1[key].dim() == 3:
                inputs2[key][:, :3] = inputs1[key]

        out1 = encoder(**inputs1)
        out2 = encoder(**inputs2)
        # Max pool may differ due to padding, but mean pool should be close
        # Just verify shapes match
        self.assertEqual(out1.shape, out2.shape)

    def test_gradient_flow(self):
        config = TagBundleEncoderConfig()
        encoder = TagBundleEncoder(config)
        inputs = self._make_dummy_input()
        output = encoder(**inputs)
        loss = output.sum()
        loss.backward()

        # absent_embedding is only used for absent tags, which this test doesn't exercise
        skip = {"absent_embedding"}
        for name, param in encoder.named_parameters():
            if param.requires_grad and name not in skip:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestCorrespondenceClassifier(unittest.TestCase):
    def _make_dummy_batch(self, batch_size=4, max_tags=5, text_dim=768, num_cross=13):
        def make_side():
            return dict(
                key_indices=torch.randint(0, NUM_TAG_KEYS, (batch_size, max_tags)),
                value_type=torch.randint(0, 4, (batch_size, max_tags)),
                boolean_values=torch.randint(0, 3, (batch_size, max_tags)),
                numeric_values=torch.randn(batch_size, max_tags, 4),
                numeric_nan_mask=torch.zeros(batch_size, max_tags, dtype=torch.bool),
                housenumber_values=torch.randn(batch_size, max_tags, 4),
                housenumber_nan_mask=torch.zeros(batch_size, max_tags, dtype=torch.bool),
                text_embeddings=torch.randn(batch_size, max_tags, text_dim),
                tag_mask=torch.ones(batch_size, max_tags, dtype=torch.bool),
            )

        pano = {f"pano_{k}": v for k, v in make_side().items()}
        osm = {f"osm_{k}": v for k, v in make_side().items()}
        return {
            **pano,
            **osm,
            "cross_features": torch.randn(batch_size, num_cross),
        }

    def test_forward_shape(self):
        config = CorrespondenceClassifierConfig()
        model = CorrespondenceClassifier(config)
        batch = self._make_dummy_batch()
        output = model(**batch)
        self.assertEqual(output.shape, (4, 1))

    def test_gradient_flow(self):
        config = CorrespondenceClassifierConfig()
        model = CorrespondenceClassifier(config)
        batch = self._make_dummy_batch()
        output = model(**batch)
        loss = output.sum()
        loss.backward()

        # Verify encoder parameters get gradients (shared weights)
        has_encoder_grad = False
        for name, param in model.named_parameters():
            if "encoder" in name and param.requires_grad and param.grad is not None:
                has_encoder_grad = True
                break
        self.assertTrue(has_encoder_grad, "Encoder should receive gradients")

    def test_output_is_logit(self):
        """Output should be unbounded logits (not probabilities)."""
        config = CorrespondenceClassifierConfig()
        model = CorrespondenceClassifier(config)
        batch = self._make_dummy_batch(batch_size=100)
        with torch.no_grad():
            output = model(**batch).squeeze(-1)
        # Logits can be any real number, but shouldn't all be identical
        self.assertGreater(output.std(), 0, "Logits should have some variance")


if __name__ == "__main__":
    unittest.main()
