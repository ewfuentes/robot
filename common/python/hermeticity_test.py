
import unittest

import sys

class HermeticityTest(unittest.TestCase):
    def test_no_user_site_packages(self):
        for p in sys.path:
            self.assertNotIn('.local/lib', p)

if __name__ == "__main__":
    unittest.main()
