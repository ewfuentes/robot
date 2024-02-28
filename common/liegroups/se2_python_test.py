
import unittest

import common.liegroups.se2_python as sep
import math
import numpy as np


class SE2PythonTest(unittest.TestCase):
    def test_construct_from_angle(self):
        # Setup
        b_from_a = sep.SE2(math.pi / 3.0)

        # Action
        b_from_a_mat = b_from_a.matrix()

        # Verification
        self.assertAlmostEqual(b_from_a_mat[0, 0], 0.5)
        self.assertAlmostEqual(b_from_a_mat[1, 1], 0.5)
        self.assertAlmostEqual(b_from_a_mat[0, 1], -math.sqrt(3) / 2.0)
        self.assertAlmostEqual(b_from_a_mat[1, 0], math.sqrt(3) / 2.0)

        self.assertAlmostEqual(b_from_a_mat[0, 2], 0.0)
        self.assertAlmostEqual(b_from_a_mat[1, 2], 0.0)

    def test_construct_from_translation(self):
        # Setup
        b_from_a = sep.SE2(np.array([1.0, 2.0]))

        # Action
        b_from_a_mat = b_from_a.matrix()

        # Verification
        self.assertAlmostEqual(b_from_a_mat[0, 0], 1.0)
        self.assertAlmostEqual(b_from_a_mat[1, 1], 1.0)
        self.assertAlmostEqual(b_from_a_mat[0, 1], 0.0)
        self.assertAlmostEqual(b_from_a_mat[1, 0], 0.0)

        self.assertAlmostEqual(b_from_a_mat[0, 2], 1.0)
        self.assertAlmostEqual(b_from_a_mat[1, 2], 2.0)

        trans = b_from_a.translation()
        self.assertAlmostEqual(trans[0], 1.0)
        self.assertAlmostEqual(trans[1], 2.0)

        b_from_a_rot = b_from_a.so2()
        b_from_a_rot_mat = b_from_a_rot.matrix()
        self.assertAlmostEqual(b_from_a_rot_mat[0, 0], 1.0)
        self.assertAlmostEqual(b_from_a_rot_mat[1, 1], 1.0)
        self.assertAlmostEqual(b_from_a_rot_mat[0, 1], 0.0)
        self.assertAlmostEqual(b_from_a_rot_mat[1, 0], 0.0)


    def test_group_operation(self):
        # Setup
        b_from_a = sep.SE2(math.pi / 3.0)
        c_from_b = sep.SE2(np.array([3, 4]))
        c_from_a_expected = sep.SE2(math.pi / 3.0, np.array([3, 4]))

        # Action
        c_from_a = c_from_b * b_from_a

        # Verification
        c_from_a_mat = c_from_a.matrix()
        c_from_a_expected_mat = c_from_a_expected.matrix()
        self.assertAlmostEqual(c_from_a_mat[0, 0], c_from_a_expected_mat[0, 0])
        self.assertAlmostEqual(c_from_a_mat[0, 1], c_from_a_expected_mat[0, 1])
        self.assertAlmostEqual(c_from_a_mat[1, 0], c_from_a_expected_mat[1, 0])
        self.assertAlmostEqual(c_from_a_mat[1, 1], c_from_a_expected_mat[1, 1])

        self.assertAlmostEqual(c_from_a_mat[0, 2], c_from_a_expected_mat[0, 2])
        self.assertAlmostEqual(c_from_a_mat[1, 2], c_from_a_expected_mat[1, 2])

    def test_group_operation_in_place(self):
        # Setup
        b_from_a = sep.SE2(math.pi / 3.0)
        c_from_b = sep.SE2(np.array([3, 4]))
        c_from_a_expected = sep.SE2(math.pi / 3.0, np.array([3, 4]))

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

        self.assertAlmostEqual(c_from_a_mat[0, 2], c_from_a_expected_mat[0, 2])
        self.assertAlmostEqual(c_from_a_mat[1, 2], c_from_a_expected_mat[1, 2])

    def test_inverse(self):
        # Setup
        b_from_a = sep.SE2(math.pi / 3.0, np.array([4, 5]))

        # Action
        a_from_b = b_from_a.inverse()

        # Verification
        a_from_a_mat = (a_from_b * b_from_a).matrix()

        total_error = np.sum(np.abs(a_from_a_mat - np.identity(3)))
        self.assertAlmostEqual(total_error, 0.0)


if __name__ == "__main__":
    unittest.main()

