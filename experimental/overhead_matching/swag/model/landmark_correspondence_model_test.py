"""Tests for the landmark correspondence model (text-only variant)."""

import unittest

import common.torch.load_torch_deps  # noqa: F401

import torch

from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    NUM_CROSS_FEATURES,
    NUM_TAG_KEYS,
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoder,
    TagBundleEncoderConfig,
)


def _make_inputs(batch_size: int, seq_len: int, text_input_dim: int,
                 tag_mask: torch.Tensor | None = None):
    key_indices = torch.randint(0, NUM_TAG_KEYS, (batch_size, seq_len))
    text_embeddings = torch.randn(batch_size, seq_len, text_input_dim)
    if tag_mask is None:
        tag_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return key_indices, text_embeddings, tag_mask


class TestTagBundleEncoder(unittest.TestCase):
    def setUp(self):
        self.cfg = TagBundleEncoderConfig(text_input_dim=16, text_proj_dim=8, per_tag_dim=4)
        self.encoder = TagBundleEncoder(self.cfg).eval()

    def test_output_shape(self):
        keys, txt, mask = _make_inputs(3, 5, self.cfg.text_input_dim)
        out = self.encoder(keys, txt, mask)
        self.assertEqual(out.shape, (3, self.cfg.repr_dim))

    def test_repr_dim_is_2x_per_tag(self):
        self.assertEqual(self.cfg.repr_dim, 2 * self.cfg.per_tag_dim)

    def test_all_padded_sample_emits_zeros(self):
        # One sample has tag_mask=[False, False]; the encoder must not leak
        # -1e9 from the max-pool into its output.
        keys, txt, mask = _make_inputs(2, 2, self.cfg.text_input_dim)
        mask[0] = torch.tensor([False, False])
        mask[1] = torch.tensor([True, True])
        with torch.no_grad():
            out = self.encoder(keys, txt, mask)
        self.assertTrue(torch.isfinite(out).all())
        # All-padded row should be exactly zero on both halves (mean and max).
        self.assertTrue(torch.allclose(out[0], torch.zeros_like(out[0])))
        # Non-all-padded row should be (in general) non-zero.
        self.assertFalse(torch.allclose(out[1], torch.zeros_like(out[1])))

    def test_padded_positions_dont_affect_output(self):
        # Two batches: one with seq_len=2 all real, one with seq_len=3 where
        # the extra slot is padded. Outputs should match modulo padding.
        cfg = self.cfg
        torch.manual_seed(0)
        keys_short = torch.randint(0, NUM_TAG_KEYS, (1, 2))
        txt_short = torch.randn(1, 2, cfg.text_input_dim)
        mask_short = torch.ones(1, 2, dtype=torch.bool)

        keys_long = torch.cat([keys_short, torch.zeros(1, 1, dtype=torch.long)], dim=1)
        txt_long = torch.cat(
            [txt_short, torch.zeros(1, 1, cfg.text_input_dim)], dim=1,
        )
        mask_long = torch.cat([mask_short, torch.zeros(1, 1, dtype=torch.bool)], dim=1)

        with torch.no_grad():
            out_short = self.encoder(keys_short, txt_short, mask_short)
            out_long = self.encoder(keys_long, txt_long, mask_long)
        torch.testing.assert_close(out_short, out_long, atol=1e-6, rtol=1e-6)


class TestCorrespondenceClassifier(unittest.TestCase):
    def setUp(self):
        self.enc_cfg = TagBundleEncoderConfig(
            text_input_dim=16, text_proj_dim=8, per_tag_dim=4,
        )
        self.cfg = CorrespondenceClassifierConfig(
            encoder=self.enc_cfg, mlp_hidden_dim=8, dropout=0.0,
        )
        self.model = CorrespondenceClassifier(self.cfg).eval()

    def test_forward_shape(self):
        B, T = 4, 3
        pano_keys, pano_txt, pano_mask = _make_inputs(B, T, self.enc_cfg.text_input_dim)
        osm_keys, osm_txt, osm_mask = _make_inputs(B, T, self.enc_cfg.text_input_dim)
        cross = torch.randn(B, NUM_CROSS_FEATURES)
        with torch.no_grad():
            out = self.model(
                pano_keys, pano_txt, pano_mask,
                osm_keys, osm_txt, osm_mask,
                cross,
            )
        self.assertEqual(out.shape, (B, 1))

    def test_classify_from_reprs_shape(self):
        B = 5
        pano_repr = torch.randn(B, self.enc_cfg.repr_dim)
        osm_repr = torch.randn(B, self.enc_cfg.repr_dim)
        cross = torch.randn(B, NUM_CROSS_FEATURES)
        with torch.no_grad():
            out = self.model.classify_from_reprs(pano_repr, osm_repr, cross)
        self.assertEqual(out.shape, (B, 1))


if __name__ == "__main__":
    unittest.main()
