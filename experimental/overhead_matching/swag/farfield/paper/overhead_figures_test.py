import math
import unittest

from experimental.overhead_matching.swag.farfield.paper import overhead_figures


class OverheadFiguresTest(unittest.TestCase):

    def test_square_bounds_are_square_in_ground_distance(self):
        west, south, east, north = overhead_figures.square_bounds(
            (-71.13, 42.19, -70.84, 42.42))
        width = (east - west) * math.cos(math.radians((south + north) / 2.0))
        self.assertAlmostEqual(width, north - south)

        self.assertEqual(
            overhead_figures.trajectory_bounds([
                ((-2.0, 3.0), (4.0, -1.0)),
                ((0.0, 5.0),),
            ]),
            (-2.0, -1.0, 4.0, 5.0),
        )


if __name__ == "__main__":
    unittest.main()
