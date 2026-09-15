import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    score_localization_inputs,
)


class _DescriptorStub(torch.nn.Module):
    """Small deterministic substitute for the checkpointed query network."""

    def forward(self, images):
        values = torch.arange(images.shape[0] * 512, device=images.device,
                              dtype=torch.float32).reshape(images.shape[0], 512)
        return torch.nn.functional.normalize(values + 1.0, dim=1)


class ScoreLocalizationInputsTest(unittest.TestCase):
    def test_writes_primary_and_top_k_variant_from_one_invocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pano_dir = root / "panorama"
            pano_dir.mkdir()
            Image.fromarray(np.zeros((180, 360, 3), dtype=np.uint8)).save(
                pano_dir / "frame_000.jpg")
            weights = root / "weights.npz"
            weights.write_bytes(b"smoke-test-weights")
            db_dir = root / "db"
            db_dir.mkdir()
            (db_dir / "manifest.json").write_text(json.dumps({
                "lattice": {"crs": "EPSG:4326", "spacing_m": 100.0},
                "render_config": {"n_yaw": 12, "fov_deg": 60.0},
            }))
            primary_dir = root / "all_crops"
            top6_dir = root / "top6"
            descriptors = torch.nn.functional.normalize(
                torch.arange(2 * 12 * 512, dtype=torch.float32).reshape(
                    2, 12, 512) + 1.0, dim=-1)
            db = {
                "descriptors": descriptors,
                "x_m": np.array([-71.0, -70.9]),
                "y_m": np.array([42.3, 42.4]),
                "manifest": json.loads((db_dir / "manifest.json").read_text()),
            }
            paths = types.SimpleNamespace(
                dataset="smoke", panorama_dir=pano_dir, dataset_base=root)
            frames = [types.SimpleNamespace(frame_idx=7, pano_stem="frame_000")]
            argv = [
                "score_localization_inputs.py", "--dataset", "smoke",
                "--db_dir", str(db_dir), "--weights", str(weights),
                "--output_dir", str(primary_dir), "--device", "cpu",
                "--crop_top_k_variant_output_dir", "6", str(top6_dir),
            ]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(score_localization_inputs.paths_lib, "resolve",
                                   return_value=paths), \
                 mock.patch.object(score_localization_inputs.dataset_lib, "load_frames",
                                   return_value=frames), \
                 mock.patch.object(score_localization_inputs.render_db, "load_database",
                                   return_value=db), \
                 mock.patch.object(score_localization_inputs.crosslocate_net,
                                   "CrossLocateVGG16MAC", return_value=_DescriptorStub()), \
                 mock.patch.object(score_localization_inputs.crosslocate_net,
                                   "load_converted_weights"):
                score_localization_inputs.main()

            primary_meta = json.loads((primary_dir / "retrieval_meta.json").read_text())
            top6_meta = json.loads((top6_dir / "retrieval_meta.json").read_text())
            self.assertNotIn("crop_top_k=", primary_meta["scorer"])
            self.assertIn("crop_top_k=6", top6_meta["scorer"])
            for output_dir in (primary_dir, top6_dir):
                fields = np.load(output_dir / "retrieval_fields.npz")
                self.assertEqual(fields["scores"].shape, (1, 2, 12))
                self.assertEqual(fields["keyframe_idx"].tolist(), [7])
                self.assertEqual(fields["pano_ids"].tolist(), ["frame_000"])


if __name__ == "__main__":
    unittest.main()
