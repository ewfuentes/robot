"""Tests for the vehicle wireframe's placement in the egovehicle frame.

Worth testing because the wireframe is the thing you will judge every future pose overlay
against: if it sits behind the origin, or points the wrong way, every bug you chase afterwards
looks like a pose bug. These assertions pin it to the frame AV2 actually uses -- origin at the
center of the rear axle at ground level, x forward, y left, z up.
"""

import unittest

import numpy as np

from experimental.map_estimation.viz import av2_scene


class VehicleWireframeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.strips = av2_scene.vehicle_wireframe()
        self.wheels = av2_scene.wheel_outlines()
        self.body = np.vstack(self.strips)

    def test_every_strip_is_a_3d_polyline(self):
        for strip in self.strips + self.wheels:
            self.assertEqual(strip.ndim, 2)
            self.assertEqual(strip.shape[1], 3)
            self.assertGreaterEqual(strip.shape[0], 2)

    def test_body_spans_the_declared_footprint(self):
        self.assertAlmostEqual(self.body[:, 0].min(), -av2_scene._REAR_OVERHANG_M, places=6)
        self.assertAlmostEqual(self.body[:, 0].max(), av2_scene._NOSE_M, places=6)
        self.assertAlmostEqual(self.body[:, 1].min(), -av2_scene._HALF_WIDTH_M, places=6)
        self.assertAlmostEqual(self.body[:, 1].max(), av2_scene._HALF_WIDTH_M, places=6)

    def test_body_sits_between_sill_and_roof(self):
        self.assertAlmostEqual(self.body[:, 2].min(), av2_scene._SILL_M, places=6)
        self.assertAlmostEqual(self.body[:, 2].max(), av2_scene._ROOF_M, places=6)

    def test_more_of_the_car_is_ahead_of_the_origin_than_behind_it(self):
        """The origin is the rear axle, not the vehicle center -- a easy thing to get backwards."""
        self.assertGreater(abs(self.body[:, 0].max()), 3.0 * abs(self.body[:, 0].min()))

    def test_wheels_rest_on_the_ground_plane(self):
        all_wheels = np.vstack(self.wheels)
        self.assertAlmostEqual(all_wheels[:, 2].min(), 0.0, places=6)
        self.assertAlmostEqual(all_wheels[:, 2].max(), 2.0 * av2_scene._WHEEL_RADIUS_M, places=6)

    def test_wheels_are_centered_on_the_two_axles(self):
        self.assertEqual(len(self.wheels), 4)
        # Midpoint of the extent, not the vertex mean: closing the loop repeats a vertex, which
        # would drag a mean off center.
        centers = sorted(
            {round(float(w[:, 0].min() + w[:, 0].max()) / 2, 3) for w in self.wheels}
        )
        self.assertEqual(centers, [0.0, round(av2_scene._WHEELBASE_M, 3)])

    def test_wheels_straddle_the_centerline(self):
        offsets = sorted({round(float(w[:, 1].mean()), 3) for w in self.wheels})
        self.assertEqual(
            offsets, [-round(av2_scene._HALF_WIDTH_M, 3), round(av2_scene._HALF_WIDTH_M, 3)]
        )

    def test_every_loop_is_closed(self):
        """Loops must return to their first vertex or the outline renders with a gap."""
        loops = [s for s in self.strips + self.wheels if s.shape[0] > 2]
        for loop in loops:
            np.testing.assert_allclose(loop[0], loop[-1])


class NoseChevronTest(unittest.TestCase):
    def test_tip_points_forward_past_the_bumper(self):
        chevron = av2_scene.nose_chevron()
        tip = chevron[1]
        self.assertGreater(tip[0], av2_scene._NOSE_M)
        self.assertAlmostEqual(float(tip[1]), 0.0, places=6)

    def test_arms_are_symmetric_about_the_centerline(self):
        chevron = av2_scene.nose_chevron()
        self.assertAlmostEqual(float(chevron[0][1]), -float(chevron[2][1]), places=6)


class EntityPathTest(unittest.TestCase):
    def test_prediction_paths_hang_off_the_documented_frames(self):
        """Predictions must inherit the same transforms as ground truth, or overlays drift."""
        self.assertTrue(av2_scene.PREDICTION_CITY.startswith(f"{av2_scene.WORLD}/"))
        self.assertTrue(av2_scene.PREDICTION_EGO.startswith(f"{av2_scene.EGO}/"))

    def test_vehicle_is_a_child_of_the_ego_transform(self):
        self.assertTrue(av2_scene.WIREFRAME.startswith(f"{av2_scene.EGO}/"))


if __name__ == "__main__":
    unittest.main()
