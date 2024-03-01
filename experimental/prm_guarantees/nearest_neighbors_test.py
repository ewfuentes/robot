
import unittest
import torch

from experimental.prm_guarantees import nearest_neighbors

class NearestNeighborsTest(unittest.TestCase):
    def test_nearest_neighbors(self):
        torch.manual_seed(0)
        N_DIM = 20
        SIZE = 1000000
        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')

        output_torch = x + y
        output_triton = nearest_neighbors.add(x, y)

        print(output_torch)
        print(output_triton)

        print('Error: ', torch.sum(torch.abs(output_triton - output_torch)))


if __name__ == "__main__":
    unittest.main()
