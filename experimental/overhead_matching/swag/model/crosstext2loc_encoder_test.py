"""Batching/ordering checks for the CrossText2Loc encoder wrapper, with a stub model and a preprocessor that
mimics the upstream one (clip.tokenize(...).squeeze(0): a single text comes back 1-D)."""
import tempfile
import unittest
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
from PIL import Image

from experimental.overhead_matching.swag.model import crosstext2loc_encoder as enc

CTX = 300


def stub_preprocessor(image, text):
    if isinstance(text, str):
        text = [text]
    tokens = torch.zeros(len(text), CTX, dtype=torch.long)
    for i, t in enumerate(text):
        tokens[i, 0] = len(t)  # a per-text fingerprint so output order is checkable
    gray = torch.tensor(image.resize((2, 2)).convert("L").getdata(), dtype=torch.float32)
    return gray.view(1, 2, 2), tokens.squeeze(0)


class StubModel:
    def encode_text(self, tokens):
        assert tokens.dim() == 2, tokens.shape
        return torch.stack([tokens[:, 0].float(), torch.ones(tokens.shape[0])], dim=1)

    def encode_image(self, images):
        assert images.dim() == 4, images.shape
        return torch.stack([images.mean(dim=(1, 2, 3)), torch.ones(images.shape[0])], dim=1)


class CrossText2LocEncoderTest(unittest.TestCase):
    def test_tokenize_single_text_keeps_batch_dim(self):
        self.assertEqual(enc.tokenize(stub_preprocessor, ["abc"]).shape, (1, CTX))
        self.assertEqual(enc.tokenize(stub_preprocessor, ["a", "bb"]).shape, (2, CTX))

    def test_encode_texts_trailing_batch_of_one(self):
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]  # 5 % 2 == 1 -> last batch has one text
        feats = enc.encode_texts(StubModel(), stub_preprocessor, texts, torch.device("cpu"), batch_size=2)
        self.assertEqual(feats.shape, (5, 2))
        torch.testing.assert_close(feats.norm(dim=1), torch.ones(5))
        # Input order preserved: the fingerprint (text length) is monotone in the first feature's unnormalised value.
        self.assertTrue(torch.all(feats[1:, 0] > feats[:-1, 0]))

    def test_encode_images_order(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, gray in enumerate([10, 200, 90]):
                p = Path(d) / f"{i}.png"
                Image.new("RGB", (4, 4), (gray, gray, gray)).save(p)
                paths.append(str(p))
            feats = enc.encode_images(StubModel(), stub_preprocessor, paths, torch.device("cpu"),
                                      batch_size=2, num_workers=0)
        self.assertEqual(feats.shape, (3, 2))
        torch.testing.assert_close(feats.norm(dim=1), torch.ones(3))
        self.assertEqual(feats[:, 0].argsort().tolist(), [0, 2, 1])


if __name__ == "__main__":
    unittest.main()
