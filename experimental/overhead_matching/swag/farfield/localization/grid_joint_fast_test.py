"""Check fused/eager joint factors against the reference likelihood."""
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    grid_filter,
    grid_joint_fast,
)


class JointBackendTest(unittest.TestCase):
    def check_backend(self, device, compiled):
        torch.manual_seed(13)
        for count in (0, 1, 17, 65):
            for variant in (0, 1, 2):
                with self.subTest(count=count, variant=variant):
                    grid = grid_filter.Grid(-1000, 1000, -800, 800, 100)
                    belief = grid_filter.GridBelief(grid, 6, device)
                    east = torch.randn(count, device=device) * 1800
                    north = torch.randn(count, device=device) * 1800
                    weights = torch.full((count,), .5 / max(count, 1), device=device)
                    epochs = [
                        (20.*i, -3.*i, .02*i, .3+.1*i, .05,
                         None if variant == 0 else 1500., .002*i, 2.*i)
                        for i in range(1 if variant == 0 else 8)]
                    options = dict(
                        cap=20 if variant == 1 else None,
                        temper=.5 if variant == 2 else 1.,
                        range_floor=.05 if variant == 2 else 0.,
                        quantization_comp=variant != 2)
                    reference = belief.track_joint_likelihood(
                        epochs, east, north, weights, 10., .2, .5, **options)
                    for chunk in (16, 64, 128):
                        actual = grid_joint_fast.joint_likelihood(
                            belief, epochs, east, north, weights, 10., .2, .5,
                            **options, compiled=compiled, chunk=chunk)
                        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                        torch.testing.assert_close(
                            actual / actual.sum(), reference / reference.sum(),
                            atol=1e-5, rtol=1e-4)

    def test_eager_cpu(self):
        self.check_backend('cpu', compiled=False)

    def test_fused_cpu(self):
        self.check_backend('cpu', compiled=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_fused_cuda(self):
        self.check_backend('cuda', compiled=True)


if __name__ == '__main__':
    unittest.main()
