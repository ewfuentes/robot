"""Window-joint proposal on a synthetic field of interchangeable landmarks."""
import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog,
    structs,
    window_proposal,
)

TRUE_EAST, TRUE_NORTH, TRUE_HEADING = 1000.0, 800.0, math.radians(35.0)
STEP_M = 20.0
N_KEYFRAMES = 30
SIGMA_DEG = 0.7


def _field(rng):
    """A 6 x 6 lattice of look-alike landmarks with 60 m jitter, 400 m pitch."""
    east, north, ids = [], [], []
    for i in range(6):
        for j in range(6):
            east.append(i * 400.0 + rng.normal(0, 60))
            north.append(j * 400.0 + rng.normal(0, 60))
            ids.append(f"lm{i}{j}")
    return filter_catalog.LandmarkCatalog(ids, east, north,
                                          max_visible_range_m=5000.0)


def _trajectory():
    """Straight, then a gentle turn; poses at every keyframe (truth)."""
    poses = []
    east, north, heading = TRUE_EAST, TRUE_NORTH, TRUE_HEADING
    deltas = []
    for k in range(N_KEYFRAMES):
        poses.append((east, north, heading))
        if k == N_KEYFRAMES - 1:
            break
        dyaw = math.radians(1.5) if k > 15 else 0.0
        heading += dyaw
        east += STEP_M * math.sin(heading)
        north += STEP_M * math.cos(heading)
        deltas.append(structs.OdometryDelta(
            keyframe_idx=k + 1, forward_m=STEP_M, left_m=0.0,
            delta_yaw_cw_rad=dyaw, sigma_m=0.5, sigma_yaw_rad=0.002))
    return poses, deltas


def _epochs(catalog, poses, rng, watched, outlier=None):
    """One epoch every 5 keyframes per tracklet; `outlier` names a tracklet
    whose bearings point at nothing in the catalog."""
    measurements = []
    kappa = 1.0 / math.radians(SIGMA_DEG) ** 2
    for tid, lid in watched.items():
        idx = catalog.index_of(lid)
        for k in range(2, N_KEYFRAMES, 5):
            east, north, heading = poses[k]
            bearing = math.atan2(catalog.east_m[idx] - east,
                                 catalog.north_m[idx] - north) - heading
            if tid == outlier:
                bearing += math.radians(37.0)
            bearing += rng.normal(0, math.radians(SIGMA_DEG))
            measurements.append(structs.TrackletMeasurement(
                tracklet_id=tid, anchor_keyframe_idx=k,
                bearing_forward_cw_deg=math.degrees(bearing) % 360.0,
                kappa=kappa, range_max_m=2000.0))
    return measurements


def _tables(catalog, tids):
    entries = [structs.CompatibilityEntry(landmark_id=lid, log_lr=4.0)
               for lid in catalog.landmark_ids]
    return {tid: structs.CompatibilityTable(
        tracklet_id=tid, matcher_version="test", entries=entries,
        default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="refined") for tid in tids}


class RelativePosesTest(unittest.TestCase):
    def test_back_integration_matches_truth(self):
        poses, deltas = _trajectory()
        kf = N_KEYFRAMES - 1
        rel = window_proposal.relative_poses(deltas, kf, 20)
        east_k, north_k, heading_k = poses[kf]
        for k, (u, v, dtheta, _) in rel.items():
            east, north, heading = poses[k]
            # world = pose_kf + R(heading_kf) (u, v)
            world_e = east_k + u * math.cos(heading_k) + v * math.sin(heading_k)
            world_n = north_k - u * math.sin(heading_k) + v * math.cos(heading_k)
            self.assertAlmostEqual(world_e, east, places=6)
            self.assertAlmostEqual(world_n, north, places=6)
            self.assertAlmostEqual(heading_k + dtheta, heading, places=9)


class ProposeTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.catalog = _field(self.rng)
        self.poses, self.deltas = _trajectory()
        # Six tracklets on six distinct landmarks around the route.
        self.watched = {"T1": "lm22", "T2": "lm33", "T3": "lm41",
                        "T4": "lm14", "T5": "lm52", "T6": "lm25"}
        self.config = structs.ProposalConfig(
            generator="window_joint", window_joint_keyframes=20,
            window_joint_max_tracklets=8, window_joint_min_tracklets=4,
            window_joint_max_outlier_tracklets=2)

    def _run(self, measurements, kf=N_KEYFRAMES - 1):
        return window_proposal.propose(
            measurements, self.deltas, _tables(self.catalog, self.watched),
            self.catalog, self.config, event_id=0, keyframe_idx=kf,
            trigger="init", particle_budget=5000, rng=self.rng)[0]

    def _best(self, result):
        return max(result.hypotheses, key=lambda h: h.compatibility_mass)

    def test_recovers_truth_in_a_lattice(self):
        measurements = _epochs(self.catalog, self.poses, self.rng, self.watched)
        result = self._run(measurements)
        self.assertTrue(result.hypotheses)
        kf = N_KEYFRAMES - 1
        east, north, heading = self.poses[kf]
        best = self._best(result)
        self.assertLess(math.hypot(best.east_m - east, best.north_m - north), 40.0)
        self.assertLess(abs(math.degrees(
            (best.heading_rad - heading + math.pi) % (2 * math.pi) - math.pi)), 2.0)
        # Two of the six lattice landmarks can sit on one ray from the route,
        # in which case the generator folds their tracklets as duplicates.
        self.assertGreaterEqual(len(best.tracklet_ids), 5)
        for tid, lid in zip(best.tracklet_ids, best.landmark_ids):
            self.assertEqual(self.watched[tid], lid)

    def test_tolerates_an_outlier_tracklet(self):
        measurements = _epochs(self.catalog, self.poses, self.rng, self.watched,
                               outlier="T4")
        result = self._run(measurements)
        self.assertTrue(result.hypotheses)
        east, north, _ = self.poses[N_KEYFRAMES - 1]
        best = self._best(result)
        self.assertLess(math.hypot(best.east_m - east, best.north_m - north), 40.0)
        self.assertNotIn("T4", best.tracklet_ids)

    def test_too_few_tracklets_yields_nothing(self):
        two = {k: v for k, v in list(self.watched.items())[:2]}
        measurements = _epochs(self.catalog, self.poses, self.rng, two)
        result, memory = window_proposal.propose(
            measurements, self.deltas, _tables(self.catalog, two), self.catalog,
            self.config, event_id=0, keyframe_idx=N_KEYFRAMES - 1,
            trigger="init", particle_budget=5000, rng=self.rng)
        self.assertEqual(result.hypotheses, [])
        self.assertIsNone(memory)

    def test_accumulate_credits_only_the_persistent_site(self):
        kf = N_KEYFRAMES - 1
        east, north, heading = self.poses[kf]
        east0, north0, heading0 = self.poses[20]
        memory = window_proposal.WindowMemory(
            keyframe_idx=20,
            poses=np.array([[east0, north0, heading0], [east0 + 800.0, north0, heading0]]),
            scores=np.array([-1.0, -3.0]))
        poses = np.array([[east, north, heading], [east + 800.0, north, heading],
                          [east + 3000.0, north, heading]])
        score = window_proposal.accumulate(poses, np.zeros(3), memory, self.deltas, kf)
        # floor = -3 - 1: persistent truth earns 0.9 * 3, the alias 0.9 * 1, the new site 0.
        self.assertAlmostEqual(score[0], 0.9 * 3.0, places=6)
        self.assertAlmostEqual(score[1], 0.9 * 1.0, places=6)
        self.assertAlmostEqual(score[2], 0.0, places=6)

    def test_memory_rewards_the_persistent_site(self):
        measurements = _epochs(self.catalog, self.poses, self.rng, self.watched)
        tables = _tables(self.catalog, self.watched)
        first, memory = window_proposal.propose(
            measurements, self.deltas, tables, self.catalog, self.config,
            event_id=0, keyframe_idx=20, trigger="init", particle_budget=5000,
            rng=self.rng)
        self.assertTrue(first.hypotheses)
        self.assertEqual(memory.keyframe_idx, 20)
        second, memory2 = window_proposal.propose(
            measurements, self.deltas, tables, self.catalog, self.config,
            event_id=1, keyframe_idx=N_KEYFRAMES - 1, trigger="diffuse",
            particle_budget=5000, rng=self.rng, memory=memory)
        east, north, _ = self.poses[N_KEYFRAMES - 1]
        best = self._best(second)
        self.assertLess(math.hypot(best.east_m - east, best.north_m - north), 40.0)
        # Lattice copies persist too, so the truth need not dominate; it must
        # stay the best and the memory must roll forward.
        first_best = self._best(first)
        self.assertLess(math.hypot(first_best.east_m - self.poses[20][0],
                                   first_best.north_m - self.poses[20][1]), 40.0)
        self.assertEqual(memory2.keyframe_idx, N_KEYFRAMES - 1)

    def test_incumbent_score_ranks_truth_first(self):
        measurements = _epochs(self.catalog, self.poses, self.rng, self.watched)
        kf = N_KEYFRAMES - 1
        east, north, heading = self.poses[kf]
        score = window_proposal.incumbent_score(
            np.array([east, east + 400.0, east + 30.0]),
            np.array([north, north, north + 30.0]),
            np.array([heading, heading, heading]),
            measurements, self.deltas, _tables(self.catalog, self.watched),
            self.catalog, self.config, kf)
        self.assertGreaterEqual(int(score.n_consistent[0]), 5)
        self.assertEqual(int(np.argmax(score.score)), 0)


if __name__ == "__main__":
    unittest.main()
