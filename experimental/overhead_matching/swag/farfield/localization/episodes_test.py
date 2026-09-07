"""Episode planning and derivation: coverage, alternation, exact reversal."""
import math
import unittest

from experimental.overhead_matching.swag.farfield.localization import (
    episodes,
    structs,
)


def _trajectory(n=60, step=50.0):
    """A curving path with exact odometry (rotate-then-move) and truth."""
    east, north, heading = 0.0, 0.0, math.radians(20.0)
    truth, odometry = [], []
    for k in range(n):
        truth.append(structs.TruthPose(keyframe_idx=k, east_m=east, north_m=north,
                                       course_world_cw_deg=math.degrees(heading) % 360.0))
        if k == n - 1:
            break
        dyaw = math.radians(3.0) if k % 7 else math.radians(-8.0)
        left = 2.0 if k % 5 == 0 else 0.0
        heading += dyaw
        east += step * math.sin(heading) - left * math.cos(heading)
        north += step * math.cos(heading) + left * math.sin(heading)
        odometry.append(structs.OdometryDelta(keyframe_idx=k + 1, forward_m=step, left_m=left,
                                              delta_yaw_cw_rad=dyaw, sigma_m=1.0, sigma_yaw_rad=0.01))
    return truth, odometry


def _integrate(start: structs.TruthPose, odometry):
    east, north, heading = start.east_m, start.north_m, math.radians(start.course_world_cw_deg)
    poses = [(east, north, heading)]
    for d in odometry:
        heading += d.delta_yaw_cw_rad
        east += d.forward_m * math.sin(heading) - d.left_m * math.cos(heading)
        north += d.forward_m * math.cos(heading) + d.left_m * math.sin(heading)
        poses.append((east, north, heading))
    return poses


class PlanTest(unittest.TestCase):
    def test_segments_cover_and_alternate(self):
        truth, _ = _trajectory()
        plan = episodes.plan(truth, 1000.0)  # ~2950 m of path -> about 6 episodes
        self.assertGreaterEqual(len(plan), 4)
        self.assertEqual(plan[0].start_keyframe, 0)
        self.assertEqual(plan[-1].end_keyframe, truth[-1].keyframe_idx)
        self.assertEqual([ep.reverse for ep in plan], [i % 2 == 1 for i in range(len(plan))])
        for ep in plan:
            self.assertLessEqual(ep.length_m, 1000.0 + 1e-6)
            self.assertGreater(ep.n_keyframes, 2)

    def test_short_recording_is_whole_forward_and_reverse(self):
        truth, _ = _trajectory(n=10)
        plan = episodes.plan(truth, 3000.0)
        self.assertEqual(len(plan), 2)
        self.assertEqual((plan[0].start_keyframe, plan[0].end_keyframe), (0, 9))
        self.assertEqual((plan[1].start_keyframe, plan[1].end_keyframe), (0, 9))
        self.assertFalse(plan[0].reverse)
        self.assertTrue(plan[1].reverse)


class DeriveTest(unittest.TestCase):
    def setUp(self):
        self.truth, self.odometry = _trajectory()
        self.measurements = [
            structs.TrackletMeasurement(tracklet_id="T1", anchor_keyframe_idx=k,
                                        bearing_forward_cw_deg=float(10 * k % 360), kappa=3000.0,
                                        range_max_m=2000.0)
            for k in range(0, 60, 5)]

    def test_forward_slice_reindexes(self):
        ep = episodes.Episode(0, 10, 30, False, 0.0, 0.0)
        odo, meas, truth = episodes.derive(self.odometry, self.measurements, self.truth, ep)
        self.assertEqual([d.keyframe_idx for d in odo], list(range(1, 21)))
        self.assertEqual([p.keyframe_idx for p in truth], list(range(21)))
        self.assertEqual(sorted(m.anchor_keyframe_idx for m in meas), [0, 5, 10, 15, 20])
        # forward odometry integrated from the episode's first truth pose lands on its truth
        poses = _integrate(truth[0], odo)
        for (e, n, _), p in zip(poses, truth):
            self.assertAlmostEqual(e, p.east_m, places=6)
            self.assertAlmostEqual(n, p.north_m, places=6)

    def test_reverse_is_exact(self):
        ep = episodes.Episode(1, 10, 30, True, 0.0, 0.0)
        odo, meas, truth = episodes.derive(self.odometry, self.measurements, self.truth, ep)
        self.assertEqual([d.keyframe_idx for d in odo], list(range(1, 21)))
        self.assertEqual([p.keyframe_idx for p in truth], list(range(21)))
        # new keyframe 0 is original 30; course flipped by 180
        self.assertAlmostEqual(truth[0].east_m, self.truth[30].east_m)
        self.assertAlmostEqual(truth[0].course_world_cw_deg,
                               (self.truth[30].course_world_cw_deg + 180.0) % 360.0)
        # reversed odometry integrated from the reversed start reproduces the path backwards
        poses = _integrate(truth[0], odo)
        for (e, n, h), p in zip(poses, truth):
            self.assertAlmostEqual(e, p.east_m, places=6)
            self.assertAlmostEqual(n, p.north_m, places=6)
            self.assertAlmostEqual(math.degrees(h) % 360.0, p.course_world_cw_deg, places=6)
        # bearings gain 180 and anchors re-index: original 30 -> 0, 10 -> 20
        by_anchor = {m.anchor_keyframe_idx: m for m in meas}
        self.assertEqual(sorted(by_anchor), [0, 5, 10, 15, 20])
        self.assertAlmostEqual(by_anchor[0].bearing_forward_cw_deg, (10 * 30 % 360 + 180) % 360)
        self.assertAlmostEqual(by_anchor[20].bearing_forward_cw_deg, (10 * 10 % 360 + 180) % 360)
        self.assertEqual(odo[0].sigma_yaw_rad, 0.01)


if __name__ == "__main__":
    unittest.main()
