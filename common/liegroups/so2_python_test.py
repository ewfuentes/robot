
import unittest

import common.liegroups.so2_python as sop
import math
import numpy as np


class SO2PythonTest(unittest.TestCase):
    def test_construct_from_angle(self):
        # Setup
        b_from_a = sop.SO2(math.pi / 3.0)

        # Action
        b_from_a_mat = b_from_a.matrix()

        # Verification
        self.assertAlmostEqual(b_from_a_mat[0, 0], 0.5)
        self.assertAlmostEqual(b_from_a_mat[1, 1], 0.5)
        self.assertAlmostEqual(b_from_a_mat[0, 1], -math.sqrt(3) / 2.0)
        self.assertAlmostEqual(b_from_a_mat[1, 0], math.sqrt(3) / 2.0)

    def test_construct_from_exp(self):
        # Setup
        b_from_a = sop.SO2.exp(math.pi / 3.0)

        # Action
        b_from_a_mat = b_from_a.matrix()

        # Verification
        self.assertAlmostEqual(b_from_a_mat[0, 0], 0.5)
        self.assertAlmostEqual(b_from_a_mat[1, 1], 0.5)
        self.assertAlmostEqual(b_from_a_mat[0, 1], -math.sqrt(3) / 2.0)
        self.assertAlmostEqual(b_from_a_mat[1, 0], math.sqrt(3) / 2.0)

    def test_group_operation(self):
        # Setup
        b_from_a = sop.SO2(math.pi / 3.0)
        c_from_b = sop.SO2(math.pi / 2.0)
        c_from_a_expected = sop.SO2(5 * math.pi / 6.0)

        # Action
        c_from_a = c_from_b * b_from_a

        # Verification
        c_from_a_mat = c_from_a.matrix()
        c_from_a_expected_mat = c_from_a_expected.matrix()
        self.assertAlmostEqual(c_from_a_mat[0, 0], c_from_a_expected_mat[0, 0])
        self.assertAlmostEqual(c_from_a_mat[0, 1], c_from_a_expected_mat[0, 1])
        self.assertAlmostEqual(c_from_a_mat[1, 0], c_from_a_expected_mat[1, 0])
        self.assertAlmostEqual(c_from_a_mat[1, 1], c_from_a_expected_mat[1, 1])

    def test_group_operation_in_place(self):
        # Setup
        b_from_a = sop.SO2(math.pi / 3.0)
        c_from_b = sop.SO2(math.pi / 2.0)
        c_from_a_expected = sop.SO2(5 * math.pi / 6.0)

        # Action
        c_from_b *= b_from_a
        c_from_a = c_from_b

        # Verification
        c_from_a_mat = c_from_a.matrix()
        c_from_a_expected_mat = c_from_a_expected.matrix()
        self.assertAlmostEqual(c_from_a_mat[0, 0], c_from_a_expected_mat[0, 0])
        self.assertAlmostEqual(c_from_a_mat[0, 1], c_from_a_expected_mat[0, 1])
        self.assertAlmostEqual(c_from_a_mat[1, 0], c_from_a_expected_mat[1, 0])
        self.assertAlmostEqual(c_from_a_mat[1, 1], c_from_a_expected_mat[1, 1])

    def test_inverse(self):
        # Setup
        b_from_a = sop.SO2(math.pi / 3.0)

        # Action
        a_from_b = b_from_a.inverse()

        # Verification
        b_from_a_mat = b_from_a.matrix()
        a_from_b_mat = a_from_b.matrix()

        self.assertAlmostEqual(b_from_a_mat[0, 0], a_from_b_mat[0, 0])
        self.assertAlmostEqual(b_from_a_mat[0, 1], a_from_b_mat[1, 0])
        self.assertAlmostEqual(b_from_a_mat[1, 0], a_from_b_mat[0, 1])
        self.assertAlmostEqual(b_from_a_mat[1, 1], a_from_b_mat[1, 1])

    def test_group_action_on_point(self):
        # Setup
        b_from_a = sop.SO2(math.pi / 3.0)
        pt_in_a = np.array([1.0, 0.0])

        # Action
        pt_in_b = b_from_a * pt_in_a

        # Verification
        self.assertAlmostEqual(pt_in_b[0], 0.5)
        self.assertAlmostEqual(pt_in_b[1], math.sqrt(3) / 2.0)

if __name__ == "__main__":
    unittest.main()
