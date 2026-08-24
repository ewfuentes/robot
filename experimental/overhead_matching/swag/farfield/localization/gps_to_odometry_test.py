import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    gps_to_odometry,
)


def derive(east, north, **kwargs):
    kwargs.setdefault("sigma_pair_m", 1.0)
    kwargs.setdefault("displacement_gate_m", 2.0)
    kwargs.setdefault("stationary_sigma_m", 3.0)
    kwargs.setdefault("slow_yaw_sigma_deg", 30.0)
    kwargs.setdefault("reverse_keyframe_ranges", ())
    return gps_to_odometry.derive_increments(east, north, **kwargs)


class DeriveIncrementsTest(unittest.TestCase):
    def test_straight_track_east(self):
        east = np.arange(5) * 40.0
        north = np.zeros(5)
        increments = derive(east, north)
        self.assertEqual([i.keyframe_idx for i in increments], [1, 2, 3, 4])
        for increment in increments:
            self.assertAlmostEqual(increment.forward_m, 40.0, places=9)
            self.assertEqual(increment.left_m, 0.0)
            self.assertEqual(increment.delta_yaw_cw_rad, 0.0)
            self.assertEqual(increment.sigma_m, 1.0)
        # First step has no previous course: slow/gapped sigma. Later steps
        # carry the honest geometric budget hypot(atan(1/40), atan(1/40)).
        self.assertAlmostEqual(increments[0].sigma_yaw_rad,
                               math.radians(30.0), places=9)
        expected = math.hypot(math.atan(1.0 / 40.0), math.atan(1.0 / 40.0))
        for increment in increments[1:]:
            self.assertAlmostEqual(increment.sigma_yaw_rad, expected,
                                   places=9)

    def test_turn_dyaw_is_differenced_course(self):
        # Two steps east, then two steps north: one -90 deg course change.
        east = np.array([0.0, 30.0, 60.0, 60.0, 60.0])
        north = np.array([0.0, 0.0, 0.0, 30.0, 60.0])
        increments = derive(east, north)
        dyaws = [math.degrees(i.delta_yaw_cw_rad) for i in increments]
        self.assertAlmostEqual(dyaws[0], 0.0, places=9)
        self.assertAlmostEqual(dyaws[1], 0.0, places=9)
        self.assertAlmostEqual(dyaws[2], -90.0, places=9)
        self.assertAlmostEqual(dyaws[3], 0.0, places=9)

    def test_speed_gate_and_gap_catch_up(self):
        # East, then two crawling steps (below min_step_m), then north:
        # the gated steps emit dyaw 0 at the slow sigma, and the catch-up
        # step spans the whole gap's course change.
        east = np.array([0.0, 30.0, 30.5, 31.0, 31.0])
        north = np.array([0.0, 0.0, 0.0, 0.0, 30.0])
        increments = derive(east, north)
        slow = math.radians(30.0)
        for gated in increments[1:3]:
            self.assertEqual(gated.delta_yaw_cw_rad, 0.0)
            self.assertAlmostEqual(gated.sigma_yaw_rad, slow, places=9)
            self.assertEqual(gated.forward_m, 0.0)
            self.assertEqual(gated.sigma_m, 3.0)
        self.assertAlmostEqual(math.degrees(increments[3].delta_yaw_cw_rad), -90.0,
                               places=9)
        self.assertLess(increments[3].sigma_yaw_rad, slow)

    def test_declared_course_sigma_tracks_step_length(self):
        east = np.array([0.0, 50.0, 55.0])
        north = np.zeros(3)
        increments = derive(east, north)
        expected = math.hypot(math.atan(1.0 / 50.0), math.atan(1.0 / 5.0))
        self.assertAlmostEqual(increments[1].sigma_yaw_rad, expected,
                               places=9)

    def test_stationary_jitter_does_not_accumulate_translation(self):
        east = [0.0, 0.4, -0.3, 0.2, -0.1]
        north = [0.0, -0.2, 0.1, 0.3, -0.4]
        increments = derive(east, north)
        self.assertEqual([item.forward_m for item in increments],
                         [0.0, 0.0, 0.0, 0.0])
        self.assertEqual([item.left_m for item in increments],
                         [0.0, 0.0, 0.0, 0.0])

    def test_human_reverse_ranges_make_distance_negative_without_false_yaw(self):
        east = [0.0, 10.0, 20.0, 10.0, 0.0]
        north = [0.0] * 5
        increments = derive(
            east, north, reverse_keyframe_ranges=((3, 4),))
        self.assertEqual([item.forward_m for item in increments],
                         [10.0, 10.0, -10.0, -10.0])
        for item in increments:
            self.assertAlmostEqual(item.delta_yaw_cw_rad, 0.0, places=9)

    def test_turning_path_reconstructs_under_rotate_then_move(self):
        east = [0.0, 10.0, 20.0, 20.0, 20.0]
        north = [0.0, 0.0, 0.0, 10.0, 20.0]
        increments = derive(east, north)
        reconstructed_east = east[0]
        reconstructed_north = north[0]
        forward_world_cw_rad = math.pi / 2.0  # first usable chord is east
        positions = [(reconstructed_east, reconstructed_north)]
        for item in increments:
            forward_world_cw_rad += item.delta_yaw_cw_rad
            reconstructed_east += (
                item.forward_m * math.sin(forward_world_cw_rad)
                - item.left_m * math.cos(forward_world_cw_rad))
            reconstructed_north += (
                item.forward_m * math.cos(forward_world_cw_rad)
                + item.left_m * math.sin(forward_world_cw_rad))
            positions.append((reconstructed_east, reconstructed_north))
        np.testing.assert_allclose(positions, list(zip(east, north)), atol=1e-9)

    def test_noise_injection_declares_itself(self):
        east = np.arange(20) * 40.0
        north = np.zeros(20)
        clean = derive(east, north)
        noisy = derive(east, north, extra_sigma_m=3.0,
                       extra_yaw_sigma_deg=2.0, noise_seed=7)
        self.assertNotEqual([i.forward_m for i in clean],
                            [i.forward_m for i in noisy])
        for a, b in zip(clean, noisy):
            self.assertAlmostEqual(b.sigma_m, math.hypot(a.sigma_m, 3.0),
                                   places=9)
            self.assertAlmostEqual(
                b.sigma_yaw_rad,
                math.hypot(a.sigma_yaw_rad, math.radians(2.0)), places=9)

    def test_deterministic(self):
        east = np.arange(10) * 40.0
        north = np.linspace(0.0, 90.0, 10)
        a = derive(east, north, extra_sigma_m=1.0)
        b = derive(east, north, extra_sigma_m=1.0)
        self.assertEqual(a, b)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            derive([0.0], [0.0])
        with self.assertRaises(ValueError):
            derive([0.0, 1.0], [0.0, 1.0], sigma_pair_m=0.0)
        with self.assertRaises(ValueError):
            derive([0.0, 10.0], [0.0, 0.0],
                   reverse_keyframe_ranges=((0, 1),))
        with self.assertRaises(ValueError):
            derive([0.0, float("nan")], [0.0, 1.0])

    def test_modeling_knobs_are_required(self):
        with self.assertRaises(TypeError):
            gps_to_odometry.derive_increments([0.0, 40.0], [0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
