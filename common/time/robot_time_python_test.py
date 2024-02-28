
import unittest

import common.time.robot_time_python as rtp


class RobotTimeDurationPythonTest(unittest.TestCase):
    def test_equality(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        d2 = rtp.as_duration(1.0)

        # Action + Verification
        self.assertEqual(d1, d2)

    def test_addition(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        d2 = rtp.as_duration(2.0)

        # Action + Verification
        self.assertEqual(d1 + d1, d2)

    def test_in_place_addition(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        d2 = rtp.as_duration(2.0)

        # Action
        d1 += d1

        # Verification
        self.assertEqual(d1, d2)

    def test_scalar_multiplication(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        d2 = rtp.as_duration(2.0)

        # Action + Verification
        self.assertEqual(2 * d1, d2)


class RobotTimePythonTest(unittest.TestCase):
    def test_add_duration(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()

        # Action
        t2 = t1 + d1

        # Verification
        self.assertEqual(t2.time_since_epoch(), d1)

    def test_radd_duration(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()

        # Action
        t2 = d1 + t1

        # Verification
        self.assertEqual(t2.time_since_epoch(), d1)

    def test_sub(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp() + d1
        t2 = rtp.RobotTimestamp() + 2 * d1

        # Action
        d2 = t2 - t1

        # Verification
        self.assertEqual(d1, d2)

    def test_inplace_addition(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()

        # Action
        t1 += d1

        # Verification
        self.assertEqual(t1, rtp.RobotTimestamp() + d1)

    def test_equality(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp() + d1
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertTrue(t1 == t2)
        self.assertFalse(t1 == t3)

    def test_inequality(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp() + d1
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertFalse(t1 != t2)
        self.assertTrue(t1 != t3)

    def test_less_than(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertFalse(t1 < t1)
        self.assertTrue(t1 < t2)
        self.assertFalse(t3 < t2)

    def test_less_than_or_equal(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertTrue(t1 <= t2)
        self.assertTrue(t2 <= t2)
        self.assertFalse(t3 <= t2)

    def test_greater_than(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertFalse(t1 > t1)
        self.assertFalse(t1 > t2)
        self.assertTrue(t3 > t2)

    def test_greater_than_or_equal(self):
        # Setup
        d1 = rtp.as_duration(1.0)
        t1 = rtp.RobotTimestamp()
        t2 = rtp.RobotTimestamp() + d1
        t3 = rtp.RobotTimestamp() + 2 * d1

        # Verification
        self.assertFalse(t1 >= t2)
        self.assertTrue(t2 >= t2)
        self.assertTrue(t3 >= t2)

if __name__ == "__main__":
    unittest.main()
