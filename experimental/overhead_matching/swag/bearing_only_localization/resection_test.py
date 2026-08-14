"""Resection geometry (design doc T-U2).

The load-bearing test is the round trip: place a pose, compute exact bearings
from it, and require resection to recover that pose. Everything else here is
about the degenerate configurations that return plausible-looking numbers
instead of failing, which is the way this class of code goes wrong.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
    resection,
)


def _bearings_from(pose, landmarks):
    """Exact body-frame bearings from a (east, north, heading) pose."""
    east, north, heading = pose
    return [float(geodesy.wrap_rad(
        geodesy.compass_bearing_rad(le - east, ln - north) - heading))
        for le, ln in landmarks]


class SubtendedAngleTest(unittest.TestCase):
    def test_is_heading_independent(self):
        """The whole reason resection needs no compass."""
        landmarks = [(2000.0, 1500.0), (-800.0, 2200.0)]
        pose = (100.0, -300.0, 0.0)
        reference = None
        for heading_deg in (0.0, 37.0, 180.0, -95.0):
            bearings = _bearings_from(
                (pose[0], pose[1], math.radians(heading_deg)), landmarks)
            gamma = resection.subtended_angle_rad(*bearings)
            if reference is None:
                reference = gamma
            self.assertAlmostEqual(gamma, reference, places=12)


class InscribedArcTest(unittest.TestCase):
    def test_every_arc_point_reproduces_the_angle(self):
        """Inscribed-angle theorem, asserted directly on samples."""
        rng = np.random.default_rng(0)
        east_a, north_a, east_b, north_b = 1500.0, 900.0, -1200.0, 2100.0
        for gamma_deg in (20.0, 55.0, 90.0, 130.0):
            gamma = math.radians(gamma_deg)
            arcs = resection.inscribed_angle_arcs(east_a, north_a, east_b,
                                                  north_b, gamma)
            self.assertEqual(len(arcs), 2, "both sides must be hypotheses")
            for arc in arcs:
                east, north = resection.sample_arc(
                    arc, east_a, north_a, east_b, north_b, 200, rng)
                self.assertEqual(east.size, 200)
                observed = np.abs(geodesy.wrap_rad(
                    geodesy.compass_bearing_rad(east_b - east, north_b - north)
                    - geodesy.compass_bearing_rad(east_a - east,
                                                  north_a - north)))
                np.testing.assert_allclose(observed, gamma, atol=1e-6)

    def test_both_arcs_are_on_opposite_sides(self):
        """Dropping one arc silently halves the hypothesis space, which
        reads as bad luck rather than as a bug."""
        arcs = resection.inscribed_angle_arcs(-1000.0, 0.0, 1000.0, 0.0,
                                              math.radians(60.0))
        norths = sorted(arc.center_north_m for arc in arcs)
        self.assertLess(norths[0], 0.0)
        self.assertGreater(norths[1], 0.0)
        self.assertAlmostEqual(norths[0], -norths[1], places=9)

    def test_signed_angle_selects_exactly_one_side(self):
        """Both arcs carry the same unsigned angle, but once the bearings
        are assigned to landmarks the sign picks a side. Keeping both would
        put half the proposal's particles on a contradicted hypothesis."""
        landmarks = [(-1200.0, 400.0), (900.0, 1500.0)]
        for pose in ((0.0, -800.0, 0.4), (-200.0, 2400.0, -1.2)):
            bearings = _bearings_from(pose, landmarks)
            signed = resection.subtended_angle_rad(*bearings)
            unsigned = resection.inscribed_angle_arcs(
                landmarks[0][0], landmarks[0][1], landmarks[1][0],
                landmarks[1][1], signed)
            selected = resection.arcs_for_signed_angle(
                landmarks[0][0], landmarks[0][1], landmarks[1][0],
                landmarks[1][1], signed)
            self.assertEqual(len(unsigned), 2)
            self.assertEqual(len(selected), 1)
            # The surviving arc must be the one the observer is actually on.
            distance = math.hypot(pose[0] - selected[0].center_east_m,
                                  pose[1] - selected[0].center_north_m)
            self.assertAlmostEqual(distance, selected[0].radius_m, delta=1e-6)

    def test_selected_arc_reproduces_the_signed_angle(self):
        landmarks = [(-1200.0, 400.0), (900.0, 1500.0)]
        pose = (0.0, -800.0, 0.4)
        bearings = _bearings_from(pose, landmarks)
        signed = resection.subtended_angle_rad(*bearings)
        arc = resection.arcs_for_signed_angle(
            landmarks[0][0], landmarks[0][1], landmarks[1][0],
            landmarks[1][1], signed)[0]
        east, north = resection.sample_arc(
            arc, landmarks[0][0], landmarks[0][1], landmarks[1][0],
            landmarks[1][1], 300, np.random.default_rng(0))
        observed = geodesy.wrap_rad(
            geodesy.compass_bearing_rad(landmarks[1][0] - east,
                                        landmarks[1][1] - north)
            - geodesy.compass_bearing_rad(landmarks[0][0] - east,
                                          landmarks[0][1] - north))
        np.testing.assert_allclose(observed, signed, atol=1e-6)

    def test_radius_matches_closed_form(self):
        gamma = math.radians(30.0)
        arcs = resection.inscribed_angle_arcs(0.0, 0.0, 1000.0, 0.0, gamma)
        # R = baseline / (2 sin gamma) = 1000 / (2 * 0.5) = 1000 (by hand).
        for arc in arcs:
            self.assertAlmostEqual(arc.radius_m, 1000.0, places=6)

    def test_rejects_degenerate_geometry(self):
        # Subtended angle too near 0 or pi: radius blows up, no constraint.
        self.assertEqual(resection.inscribed_angle_arcs(
            0.0, 0.0, 1000.0, 0.0, math.radians(1.0)), [])
        self.assertEqual(resection.inscribed_angle_arcs(
            0.0, 0.0, 1000.0, 0.0, math.radians(179.0)), [])
        # Landmarks too close together to define a baseline.
        self.assertEqual(resection.inscribed_angle_arcs(
            0.0, 0.0, 1.0, 1.0, math.radians(45.0)), [])


class ResectThreeTest(unittest.TestCase):
    _LANDMARKS = [(2200.0, 1800.0), (2600.0, -2100.0), (-1500.0, 1200.0)]

    def test_all_pairings_are_tried(self):
        """An observer nearly in line with one landmark pair sees ~180 deg
        between them, which that pair cannot constrain. Fixing the pairing
        in advance would discard the pose; the other pairings still fix it."""
        pose = (300.0, -400.0, 0.9)
        bearings = _bearings_from(pose, self._LANDMARKS)
        gamma_12 = abs(math.degrees(resection.subtended_angle_rad(
            bearings[1], bearings[2])))
        self.assertGreater(gamma_12, 170.0, "fixture no longer ill-posed")
        solutions = resection.resect_three(
            self._LANDMARKS, bearings, residual_tolerance_rad=1e-4)
        self.assertEqual(len(solutions), 1)
        self.assertAlmostEqual(solutions[0].east_m, pose[0], delta=1e-3)
        self.assertAlmostEqual(solutions[0].north_m, pose[1], delta=1e-3)

    def test_round_trip_recovers_pose(self):
        """T-U2 proper: pose -> exact bearings -> resection -> same pose."""
        rng = np.random.default_rng(7)
        tested = 0
        for _ in range(200):
            pose = (float(rng.uniform(-1500, 1500)),
                    float(rng.uniform(-1500, 1500)),
                    float(rng.uniform(-math.pi, math.pi)))
            bearings = _bearings_from(pose, self._LANDMARKS)
            solutions = resection.resect_three(self._LANDMARKS, bearings)
            if not solutions:
                continue  # legitimately rejected (danger circle etc.)
            tested += 1
            # With exact bearings the true pose must rank first: its residual
            # is zero and every spurious intersection's is not.
            best = solutions[0]
            self.assertAlmostEqual(best.east_m, pose[0], delta=1e-3)
            self.assertAlmostEqual(best.north_m, pose[1], delta=1e-3)
            self.assertAlmostEqual(
                float(geodesy.wrap_rad(best.heading_rad - pose[2])), 0.0,
                delta=1e-6)
            self.assertLess(best.residual_rad, 1e-6)
        self.assertGreater(tested, 150, "too many configurations rejected")

    def test_tolerance_separates_truth_from_near_misses(self):
        """Spurious circle intersections can sit only a couple of degrees
        off, so the tolerance is load-bearing: too loose and fiction
        survives, too tight and every noisy fix is discarded."""
        pose = (300.0, -400.0, 0.9)
        bearings = _bearings_from(pose, self._LANDMARKS)
        tight = resection.resect_three(self._LANDMARKS, bearings,
                                       residual_tolerance_rad=1e-4)
        self.assertEqual(len(tight), 1)
        loose = resection.resect_three(self._LANDMARKS, bearings,
                                       residual_tolerance_rad=math.radians(10))
        self.assertGreater(len(loose), 1)
        # Whatever the tolerance, truth ranks first and the extras are real
        # alternatives rather than duplicates.
        self.assertAlmostEqual(loose[0].east_m, pose[0], delta=1e-3)
        self.assertTrue(all(
            math.hypot(a.east_m - b.east_m, a.north_m - b.north_m) > 1.0
            for i, a in enumerate(loose) for b in loose[i + 1:]))

    def test_rejects_collinear_landmarks(self):
        collinear = [(-1000.0, 0.0), (0.0, 0.0), (1000.0, 0.0)]
        pose = (200.0, 900.0, 0.3)
        self.assertEqual(
            resection.resect_three(collinear, _bearings_from(pose, collinear)),
            [])

    def test_rejects_observer_on_the_danger_circle(self):
        """Snellius-Pothenot: on the circle through all three landmarks the
        fix is indeterminate, so returning any number would be fiction."""
        landmarks = self._LANDMARKS
        (e1, n1), (e2, n2), (e3, n3) = landmarks
        d = 2.0 * (e1 * (n2 - n3) + e2 * (n3 - n1) + e3 * (n1 - n2))
        sq1, sq2, sq3 = e1 ** 2 + n1 ** 2, e2 ** 2 + n2 ** 2, e3 ** 2 + n3 ** 2
        center_e = (sq1 * (n2 - n3) + sq2 * (n3 - n1) + sq3 * (n1 - n2)) / d
        center_n = (sq1 * (e3 - e2) + sq2 * (e1 - e3) + sq3 * (e2 - e1)) / d
        radius = math.hypot(e1 - center_e, n1 - center_n)
        for theta in np.linspace(0.0, 2 * math.pi, 12, endpoint=False):
            pose = (center_e + radius * math.cos(theta),
                    center_n + radius * math.sin(theta), 0.4)
            self.assertEqual(
                resection.resect_three(landmarks,
                                       _bearings_from(pose, landmarks)),
                [], f"accepted a danger-circle fix at theta={theta:.2f}")

    def test_noise_degrades_gracefully(self):
        """With bearing noise the fix should move by a bounded amount, not
        jump to an unrelated part of the region."""
        rng = np.random.default_rng(11)
        pose = (400.0, -200.0, 1.1)
        exact = _bearings_from(pose, self._LANDMARKS)
        errors = []
        for _ in range(120):
            noisy = [b + float(rng.normal(0.0, math.radians(1.0)))
                     for b in exact]
            solutions = resection.resect_three(self._LANDMARKS, noisy)
            if solutions:
                errors.append(min(math.hypot(s.east_m - pose[0],
                                             s.north_m - pose[1])
                                  for s in solutions))
        self.assertGreater(len(errors), 60, "noise rejected too many fixes")
        self.assertLess(float(np.median(errors)), 250.0)


class HeadingRecoveryTest(unittest.TestCase):
    def test_heading_from_bearing(self):
        # Landmark due north of the observer, seen 30 deg to starboard =>
        # the vehicle points 30 deg to port of north (hand-computed).
        heading = resection.heading_from_bearing(0.0, 0.0, 0.0, 1000.0,
                                                 math.radians(30.0))
        self.assertAlmostEqual(math.degrees(heading), -30.0, places=9)

    def test_round_trip_against_arbitrary_poses(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            east, north = rng.uniform(-2000, 2000, 2)
            heading = float(rng.uniform(-math.pi, math.pi))
            lm_east, lm_north = rng.uniform(-3000, 3000, 2)
            bearing = float(geodesy.wrap_rad(
                geodesy.compass_bearing_rad(lm_east - east, lm_north - north)
                - heading))
            recovered = resection.heading_from_bearing(east, north, lm_east,
                                                       lm_north, bearing)
            self.assertAlmostEqual(
                float(geodesy.wrap_rad(recovered - heading)), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
