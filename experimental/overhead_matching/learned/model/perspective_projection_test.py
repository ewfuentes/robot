
import unittest

import common.torch.load_torch_deps
import math
import torch
from experimental.overhead_matching.learned.model import perspective_projection


class PerspectiveProjectionTest(unittest.TestCase):
    def test_perspective_projection(self):
        # Setup
        x = 3.0
        y = 4.0
        z = 5.0
        input_pt = torch.tensor([x, y, z])
        expected = torch.tensor([math.sqrt(x*x + y*y + z*z), math.pi / 4.0, math.atan2(y, x)])

        # Action
        output = perspective_projection.spherical_projection(input_pt)

        # Verification
        self.assertTrue(torch.allclose(output, expected))

    def test_perspective_projection_batching(self):
        # Setup
        x = 3.0
        y = 4.0
        z = 5.0
        input_pt = torch.tensor([x, y, z])
        expected = torch.tensor([math.sqrt(x*x + y*y + z*z), math.pi / 4.0, math.atan2(y, x)])
        for i in range(1, 10):
            input_expanded = input_pt.expand(*list(range(1, i)), 3)
            expected_expanded = expected.expand(*list(range(1, i)), 3)

            # Action
            output = perspective_projection.spherical_projection(input_expanded)

            # Verification
            self.assertTrue(torch.allclose(output, expected_expanded))


if __name__ == "__main__":
    unittest.main()
