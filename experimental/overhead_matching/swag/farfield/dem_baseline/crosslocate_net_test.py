import unittest
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    crosslocate_net,
)

CONVERTED_WEIGHTS = Path(
    "/data/farfield_matching/models/crosslocate/"
    "AlpsPhotosToDepthCompact_31_2/converted_weights.npz")


class ArchitectureTest(unittest.TestCase):
    def test_descriptor_shape_and_norm(self):
        model = crosslocate_net.CrossLocateVGG16MAC().eval()
        with torch.inference_mode():
            out = model(torch.rand(2, 3, 224, 224) * 255.0)
        self.assertEqual(tuple(out.shape),
                         (2, crosslocate_net.DESCRIPTOR_DIM))
        np.testing.assert_allclose(
            out.norm(dim=1).numpy(), 1.0, atol=1e-5)

    def test_parameter_count_matches_release(self):
        # 13 convs x (kernels + biases) == the 26 tensors the TF release
        # loads before stopping at index 26.
        model = crosslocate_net.CrossLocateVGG16MAC()
        self.assertEqual(len(list(model.parameters())), 26)

    def test_native_input_resolution_works(self):
        model = crosslocate_net.CrossLocateVGG16MAC().eval()
        with torch.inference_mode():
            out = model(torch.rand(1, 3, 500, 500) * 255.0)
        self.assertEqual(tuple(out.shape),
                         (1, crosslocate_net.DESCRIPTOR_DIM))

    def test_input_helpers(self):
        rgb = (np.random.default_rng(0).random((375, 500, 3)) * 255).astype(
            np.uint8)
        tensor = crosslocate_net.rgb_query_tensor(rgb)
        self.assertEqual(tuple(tensor.shape),
                         (3, *crosslocate_net.NATIVE_INPUT_HW))
        self.assertGreater(tensor.max().item(), 1.5)  # raw 0-255, not scaled

        depth = np.full((500, 500), 1234.5, dtype=np.float32)
        depth[0, 0] = np.inf
        dt = crosslocate_net.depth_render_tensor(depth, sky_fill_m=0.0)
        self.assertEqual(tuple(dt.shape),
                         (3, *crosslocate_net.NATIVE_INPUT_HW))
        self.assertTrue(torch.isfinite(dt).all())
        # Channel replication: all three channels identical.
        np.testing.assert_array_equal(dt[0].numpy(), dt[1].numpy())

    def test_distances_convention(self):
        # Unit vectors: squared euclidean == 2 - 2 cos.
        q = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
        db = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        dist = crosslocate_net.descriptor_distances(q, db)
        expected = 2.0 - 2.0 * (q @ db.t())
        np.testing.assert_allclose(dist.numpy(), expected.numpy(), atol=1e-5)


class ConvertedWeightsTest(unittest.TestCase):
    @unittest.skipUnless(CONVERTED_WEIGHTS.exists(),
                         "converted checkpoint not on this machine")
    def test_load_and_run(self):
        model = crosslocate_net.CrossLocateVGG16MAC().eval()
        crosslocate_net.load_converted_weights(model, CONVERTED_WEIGHTS)
        with torch.inference_mode():
            out = model(torch.rand(1, 3, 500, 500) * 255.0)
        self.assertTrue(torch.isfinite(out).all())
        np.testing.assert_allclose(out.norm(dim=1).numpy(), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
