import experimental.color_histogram.color_histogram as ch

import unittest
import torch


class ColorHistogramTest(unittest.TestCase):
    def test_no_stride(self):
        # Setup
        t = torch.arange(120, dtype=torch.float32)
        t = t.reshape((2, 3, 4, 5))

        # Action
        hist, edges = ch.color_histogram(t, bins=5, stride=1, size=3)

        # Verification
        self.assertEqual(hist.shape[0], 2)
        self.assertEqual(hist.shape[1], 3)
        self.assertEqual(hist.shape[2], 2)
        self.assertEqual(hist.shape[3], 3)
        self.assertEqual(hist.shape[4], 5)

    def test_with_stride(self):
        # Setup
        t = torch.arange(120, dtype=torch.float32)
        t = t.reshape((2, 3, 4, 5))

        # Action
        hist, edges = ch.color_histogram(t, bins=5, stride=2, size=3)

        # Verification
        self.assertEqual(hist.shape[0], 2)
        self.assertEqual(hist.shape[1], 3)
        self.assertEqual(hist.shape[2], 1)
        self.assertEqual(hist.shape[3], 2)
        self.assertEqual(hist.shape[4], 5)


if __name__ == "__main__":
    unittest.main()
