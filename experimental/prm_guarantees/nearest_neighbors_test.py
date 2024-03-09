import unittest
import torch

from experimental.prm_guarantees.nearest_neighbors import nearest_neighbors


class NearestNeighborsTest(unittest.TestCase):
    def test_nearest_neighbors(self):
        torch.manual_seed(0)
        N_DIM = 20
        SIZE = 10000
        x = torch.rand((SIZE, N_DIM), device="cuda")

        output_triton = nearest_neighbors(x)

        print(output_triton)


if __name__ == "__main__":
    unittest.main()
