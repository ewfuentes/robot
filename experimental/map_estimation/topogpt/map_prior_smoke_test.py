"""Proves the vendored MapPrior package is importable and its model classes construct.

TopoGPT is developed with `rootutils` putting its source tree on sys.path, which masks packaging
defects that only appear once it is pip-installed -- most of `MapPrior.modules` ships without an
`__init__.py` upstream and is silently dropped by `find_packages`, so a stock install yields a
`MapPrior` package with no model in it. This test is the tripwire for that class of breakage, and
for the dependency closure the imports drag in -- 22 MapPrior modules and 15 third-party packages.
Reaching `gpt` alone pulls `datasets.map_dataset` (cv2, shapely, pyquaternion, scikit-learn,
lightning), `modules.util` (matplotlib), and `MapPrior.utils` (hydra, omegaconf, rich); the fork
declares all of them, so this also checks that declaration is complete.

The three imported classes are exactly the `_target_`s of `configs/model/pretrain.yaml`.
"""

import tempfile
import unittest
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  -- must precede torch or the CUDA libs miss
import torch
from lightning import LightningModule
from omegaconf import OmegaConf

from MapPrior.modules.autoregression.gpt import GPT_S, GPT_models
from MapPrior.modules.autoregression.vector_ar import VectorARLightning
from MapPrior.modules.bev_encoder.simple_bev_encoder import LaneMaskEncoder, LaneMaskProjector
from MapPrior.modules.util import load_pretrained_weights


class MapPriorSmokeTest(unittest.TestCase):
    def test_pretrain_config_targets_resolve(self):
        """The top-level `_target_` of configs/model/pretrain.yaml is importable and is a module.

        Constructing it needs the full optimizer/scheduler/loss/sample config tree, so this
        checks the class rather than an instance; the pieces it composes are exercised below.
        """
        self.assertTrue(issubclass(VectorARLightning, LightningModule))

    def test_lane_mask_projector_flattens_bev_to_tokens(self):
        """The conditioning path: a rasterized BEV lane mask becomes a sequence of GPT tokens."""
        # Values mirror configs/model/pretrain.yaml. The 128x208 input is that config's 16x26
        # bev_shape times its downsample_rate of 8, so this checks those two settings -- which
        # live in different halves of the config and are never validated against each other --
        # actually agree.
        projector = LaneMaskProjector(
            mask_encoder_net=LaneMaskEncoder(
                in_channels=2,
                out_channels=768,
                downsample_rate=8,
                num_blocks=1,
                base_channels=64,
            )
        )
        tokens = projector(torch.zeros(1, 2, 128, 208))
        self.assertEqual(tuple(tokens.shape), (1, 16 * 26, 768))

    def test_gpt_constructs(self):
        """The generator itself builds, with the attention backend the released config selects."""
        # GPT_S rather than the config's GPT_L: identical Transformer code path, an order of
        # magnitude fewer parameters to allocate. bev_cfg has no usable default -- Transformer
        # indexes bev_shape unconditionally, so omitting it is a TypeError rather than a default.
        # attn_op_type="kernel" selects F.scaled_dot_product_attention over the hand-written
        # fallback; neither needs flash-attn or xformers.
        model = GPT_S(
            pos_vocab_size=4096,
            seq_len=50,
            num_control_points=4,
            attn_op_type="kernel",
            bev_cfg={"bev_shape": [16, 26]},
        )
        self.assertEqual(model.bev_token_num, 16 * 26)
        self.assertIn("GPT_L", GPT_models)

    def test_loads_a_checkpoint_carrying_hydra_hyperparameters(self):
        """A Lightning-shaped checkpoint loads without weakening torch's weights_only check.

        The released pretrain.ckpt and finetune.ckpt both store save_hyperparameters() output --
        omegaconf config objects -- beside the tensors, which torch >= 2.6 refuses to construct
        by default. The fork allowlists the specific inert container types instead of passing
        weights_only=False. This reproduces that shape locally rather than downloading 1.26 GB:
        if the allowlist regresses, torch raises UnpicklingError here.
        """
        model = torch.nn.Linear(4, 2)
        checkpoint = {
            "state_dict": {"weight": torch.ones(2, 4), "bias": torch.zeros(2)},
            "hyper_parameters": OmegaConf.create(
                {"sample_cfg": {"temperature": 1.0, "spatial_bin_num": [64, 64]}}
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake_lightning.ckpt"
            torch.save(checkpoint, path)

            # Without the allowlist this file is rejected -- asserted so the test below cannot
            # pass vacuously if some future torch admits omegaconf by default.
            with self.assertRaises(Exception) as rejected:
                torch.load(path, map_location="cpu")
            self.assertIn("omegaconf", str(rejected.exception).lower())

            load_pretrained_weights(model, str(path))

        torch.testing.assert_close(model.weight, torch.ones(2, 4))


if __name__ == "__main__":
    unittest.main()
