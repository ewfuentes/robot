import unittest
from haversine import find_d_on_unit_circle

class TestHaversine(unittest.TestCase):
    def test_same_point(self):
        # Distance between the same point should be 0
        point = (73.0, -44.0)
        self.assertAlmostEqual(find_d_on_unit_circle(point, point), 0.0, places=6)

    def test_equator_points(self):
        # Distance between two points on the equator
        point_one = (0.0, 0.0)
        point_two = (0.0, 90.0)
        expected_distance = 1.5708  # π/2 radians
        self.assertAlmostEqual(find_d_on_unit_circle(point_one, point_two), expected_distance, places=4)

    def test_poles(self):
        # Distance between the North Pole and South Pole
        north_pole = (90.0, 0.0)
        south_pole = (-90.0, 0.0)
        expected_distance = 3.1416  # π radians
        self.assertAlmostEqual(find_d_on_unit_circle(north_pole, south_pole), expected_distance, places=4)

    def test_arbitrary_points(self):
        # Distance between two arbitrary points
        point_one = (34.0522, -118.2437)  # Los Angeles
        point_two = (40.7128, -74.0060)   # New York
        # Expected value is approximate since it's on a unit circle
        expected_distance = 0.6178
        self.assertAlmostEqual(find_d_on_unit_circle(point_one, point_two), expected_distance, places=4)

if __name__ == "__main__":
    unittest.main()