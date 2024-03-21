
import unittest

import numpy as np

from common.liegroups.se3_python import SE3
from common.liegroups.so3_python import SO3


class SE3PythonTest(unittest.TestCase):
    def test_default_is_identity(self):
        # Action
        a_from_a = SE3()

        # Verification
        a_from_a_mat = a_from_a.matrix()
        for i in range(4):
            for j in range(4):
                self.assertEqual(a_from_a_mat[i, j], 1.0 if i == j else 0.0)

    def test_construct_from_so3_and_translation(self):
        # Action
        b_from_a_rot = SO3.rotX(np.pi / 2.0)
        b_from_a_trans = np.array([1.0, 2.0, 3.0])
        b_from_a = SE3(b_from_a_rot, b_from_a_trans)
        y_in_a = np.array([0, 1.0, 0])
        expected_y_in_b = np.array([1.0, 2.0, 4.0])

        # Verification
        y_in_b = b_from_a * y_in_a
        self.assertAlmostEqual(y_in_b[0], expected_y_in_b[0])
        self.assertAlmostEqual(y_in_b[1], expected_y_in_b[1])
        self.assertAlmostEqual(y_in_b[2], expected_y_in_b[2])


if __name__ == "__main__":
    unittest.main()

