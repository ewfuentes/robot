import unittest
import common.torch.load_torch_deps
import torch
import common.torch.example_kernel_python as ekp

class ExampleKernelTest(unittest.TestCase):
    def test_example_kernel_float32(self):
        # Setup
        input = torch.arange(1, 1024, dtype=torch.float32).cuda()
        expected_output = input ** 2

        # Action
        output = ekp.square(input)

        # Verification
        self.assertTrue(torch.allclose(output, expected_output))

    def test_example_kernel_float64(self):
        # Setup
        input = torch.arange(1, 1024, dtype=torch.float64).cuda()
        expected_output = input ** 2

        # Action
        output = ekp.square(input)

        # Verification
        self.assertTrue(torch.allclose(output, expected_output))

    def test_example_kernel_int32(self):
        # Setup
        input = torch.arange(1, 1024, dtype=torch.int32).cuda()
        expected_output = input ** 2

        # Action
        output = ekp.square(input)

        # Verification
        self.assertTrue(torch.allclose(output, expected_output))

if __name__ == "__main__":
    unittest.main()
