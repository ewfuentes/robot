
import unittest
import numpy as np

from common.liegroups.so3_python import SO3

class SO3PythonTest(unittest.TestCase):

    def test_default_construction_is_identity(self):
        # Action
        b_from_a = SO3()

        # Verification
        quat = b_from_a.quaternion()
        self.assertEqual(quat[0], 0.0)
        self.assertEqual(quat[1], 0.0)
        self.assertEqual(quat[2], 0.0)
        self.assertEqual(quat[3], 1.0)

        mat = b_from_a.matrix()
        for i in range(3):
            for j in range(3):
                self.assertEqual(mat[i, j], 1.0 if i == j else 0.0)

    def test_construct_from_quat(self):
        # Setup
        rotation_axis = np.array([1, 2, 3])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = 0.456
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)
        b_from_a_quat = np.concatenate([rotation_axis * s, [c]])
        a_from_b_quat = np.concatenate([-rotation_axis * s, [c]])

        # Action
        b_from_a = SO3.from_quat(b_from_a_quat)
        a_from_b = SO3.from_quat(a_from_b_quat)

        # Verification
        a_from_a = a_from_b * b_from_a
        a_from_a_quat = a_from_a.quaternion()
        self.assertEqual(a_from_a_quat[0], 0.0)
        self.assertEqual(a_from_a_quat[1], 0.0)
        self.assertEqual(a_from_a_quat[2], 0.0)
        self.assertEqual(a_from_a_quat[3], 1.0)

    def test_inverse(self):
        # Setup
        rotation_axis = np.array([1, 2, 3])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = 0.456
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)
        b_from_a_quat = np.concatenate([rotation_axis * s, [c]])

        # Action
        b_from_a = SO3.from_quat(b_from_a_quat)
        a_from_b = b_from_a.inverse()

        # Verification
        a_from_a = a_from_b * b_from_a
        a_from_a_quat = a_from_a.quaternion()
        self.assertEqual(a_from_a_quat[0], 0.0)
        self.assertEqual(a_from_a_quat[1], 0.0)
        self.assertEqual(a_from_a_quat[2], 0.0)
        self.assertEqual(a_from_a_quat[3], 1.0)

    def test_rot_x(self):
        # Setup
        rotation_angle = 0.456
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)

        # Action
        b_from_a = SO3.rotX(rotation_angle)

        # Verification
        b_from_a_quat = b_from_a.quaternion()
        self.assertAlmostEqual(b_from_a_quat[0], s)
        self.assertAlmostEqual(b_from_a_quat[1], 0.0)
        self.assertAlmostEqual(b_from_a_quat[2], 0.0)
        self.assertAlmostEqual(b_from_a_quat[3], c)

    def test_rot_y(self):
        # Setup
        rotation_angle = 0.456
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)

        # Action
        b_from_a = SO3.rotY(rotation_angle)

        # Verification
        b_from_a_quat = b_from_a.quaternion()
        self.assertAlmostEqual(b_from_a_quat[0], 0.0)
        self.assertAlmostEqual(b_from_a_quat[1], s)
        self.assertAlmostEqual(b_from_a_quat[2], 0.0)
        self.assertAlmostEqual(b_from_a_quat[3], c)

    def test_rot_z(self):
        # Setup
        rotation_angle = 0.456
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)

        # Action
        b_from_a = SO3.rotZ(rotation_angle)

        # Verification
        b_from_a_quat = b_from_a.quaternion()
        self.assertAlmostEqual(b_from_a_quat[0], 0.0)
        self.assertAlmostEqual(b_from_a_quat[1], 0.0)
        self.assertAlmostEqual(b_from_a_quat[2], s)
        self.assertAlmostEqual(b_from_a_quat[3], c)

    def test_exp_log(self):
        # Setup
        rotation_angle = 0.456
        rotation_axis = np.array([1, 2, 3])
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        c = np.cos(rotation_angle / 2.0)
        s = np.sin(rotation_angle / 2.0)
        tangent = rotation_axis * rotation_angle
        b_from_a_quat_expected = np.concatenate([rotation_axis * s, [c]])

        # Action
        b_from_a = SO3.exp(tangent)

        # Verification
        b_from_a_quat = b_from_a.quaternion()
        b_from_a_log = b_from_a.log()

        self.assertAlmostEqual(b_from_a_quat[0], b_from_a_quat_expected[0])
        self.assertAlmostEqual(b_from_a_quat[1], b_from_a_quat_expected[1])
        self.assertAlmostEqual(b_from_a_quat[2], b_from_a_quat_expected[2])
        self.assertAlmostEqual(b_from_a_quat[3], b_from_a_quat_expected[3])

        self.assertAlmostEqual(b_from_a_log[0], tangent[0])
        self.assertAlmostEqual(b_from_a_log[1], tangent[1])
        self.assertAlmostEqual(b_from_a_log[2], tangent[2])

    def test_group_action(self):
        # Setup
        b_from_a = SO3.rotX(np.pi / 2.0)
        x_in_a = np.array([1, 0, 0])
        y_in_a = np.array([0, 1, 0])

        # Action
        x_in_b = b_from_a * x_in_a
        y_in_b = b_from_a * y_in_a

        # Verification
        self.assertAlmostEqual(x_in_b[0], x_in_a[0])
        self.assertAlmostEqual(x_in_b[1], x_in_a[1])
        self.assertAlmostEqual(x_in_b[2], x_in_a[2])

        self.assertAlmostEqual(y_in_b[0], 0.0)
        self.assertAlmostEqual(y_in_b[1], 0.0)
        self.assertAlmostEqual(y_in_b[2], 1.0)

if __name__ == "__main__":
    unittest.main()
