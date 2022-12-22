
import unittest

from common.python.pybind_example_python import add

class PybindExampleTest(unittest.TestCase):
    def test_pybind_test(self):
        self.assertEqual(add(1.0, 2.0), 3.0)

if __name__ == "__main__":
    unittest.main()
