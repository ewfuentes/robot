import math
import unittest
from types import SimpleNamespace

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    metrics,
    structs,
)


class MapPoseTest(unittest.TestCase):
    def test_picks_the_dense_mode_not_the_midpoint(self):
        """For a bimodal belief the weighted mean sits between the modes and
        describes no hypothesis the filter holds (A-9)."""
        east = np.concatenate([np.full(600, -1000.0), np.full(400, 1000.0)])
        north = np.zeros(1000)
        belief = pf.ParticleBelief(east + np.linspace(-5, 5, 1000), north,
                                   np.zeros(1000), np.zeros(1000))
        mean_east = metrics.mean_pose(belief)[0]
        map_east = metrics.map_pose(belief, cell_size_m=100.0)[0]
        self.assertLess(abs(mean_east - (-200.0)), 60.0)
        self.assertLess(abs(map_east - (-1000.0)), 60.0)

    def test_mass_within_radius(self):
        belief = pf.ParticleBelief(
            np.array([0.0, 10.0, 500.0]), np.zeros(3), np.zeros(3),
            np.zeros(3))
        self.assertAlmostEqual(
            metrics.mass_within_radius(belief, 0.0, 0.0, 100.0),
            2.0 / 3.0, places=6)

    def test_position_mass_config_has_stable_identity_and_strict_radii(self):
        config = metrics.position_mass_metric_config([50, 100.0])
        self.assertEqual(config.metric_id,
                         metrics.POSITION_MASS_METRIC_ID)
        self.assertEqual(config.metric_version,
                         metrics.POSITION_MASS_METRIC_VERSION)
        self.assertEqual(
            metrics.position_mass_metric_key(config, 50.0),
            "posterior_position_probability_mass_within_radius@1:radius_m=50")
        with self.assertRaisesRegex(ValueError, "sorted"):
            metrics.position_mass_metric_config([100.0, 50.0])


class ErrorSeriesTest(unittest.TestCase):
    """The canonical error helpers SKIP keyframes absent from truth.

    On real exports truth exists only where GPS was valid; the old helper
    raised KeyError on such keyframes while every downstream consumer
    re-implemented the skip — so the skip is now the contract, pinned here.
    """

    @staticmethod
    def _health(keyframe_idx, east, north, heading_deg=0.0):
        return structs.HealthRecord(
            keyframe_idx=keyframe_idx, ess=1.0, resampled=False,
            mean_east_m=east, mean_north_m=north,
            mean_heading_deg=heading_deg,
            map_east_m=east, map_north_m=north, map_heading_deg=heading_deg,
            position_std_m=1.0, heading_std_deg=1.0, n_measurements=0)

    def test_skips_keyframes_absent_from_truth(self):
        health = [self._health(0, 0.0, 0.0),
                  self._health(1, 999.0, 999.0),  # no truth for keyframe 1
                  self._health(2, 3.0, 4.0, heading_deg=10.0)]
        truth = [structs.TruthPose(0, 0.0, 0.0, 0.0),
                 structs.TruthPose(2, 0.0, 0.0, 350.0)]

        errors = metrics.position_errors_m(health, truth)
        np.testing.assert_allclose(errors, [0.0, 5.0])
        np.testing.assert_array_equal(errors.keyframe_idx, [0, 2])
        map_errors = metrics.map_position_errors_m(health, truth)
        np.testing.assert_allclose(map_errors, [0.0, 5.0])
        np.testing.assert_array_equal(map_errors.keyframe_idx, [0, 2])
        # Heading errors wrap: 10 vs 350 is 20 degrees, not 340.
        heading = metrics.heading_errors_deg(health, truth)
        np.testing.assert_allclose(heading, [0.0, 20.0])
        np.testing.assert_array_equal(heading.keyframe_idx, [0, 2])

    def test_full_truth_coverage_matches_health_length(self):
        health = [self._health(k, float(k), 0.0) for k in range(4)]
        truth = [structs.TruthPose(k, 0.0, 0.0, 0.0) for k in range(4)]
        errors = metrics.position_errors_m(health, truth)
        self.assertEqual(len(errors), len(health))
        np.testing.assert_allclose(errors, [0.0, 1.0, 2.0, 3.0])


class PositionNeesTest(unittest.TestCase):
    def test_matches_hand_computed_isotropic_case(self):
        rng = np.random.default_rng(0)
        n = 200000
        sigma = 25.0
        belief = pf.ParticleBelief(
            east_m=rng.normal(100.0, sigma, n),
            north_m=rng.normal(-50.0, sigma, n),
            heading_rad=np.zeros(n), log_weight=np.zeros(n))
        # Truth one sigma off along east: NEES ~ (sigma/sigma)^2 = 1.
        nees = metrics.position_nees(belief, 100.0 + sigma, -50.0)
        self.assertAlmostEqual(nees, 1.0, delta=0.05)
        std = metrics.position_std_m(belief)
        self.assertAlmostEqual(std, sigma, delta=0.3)
        cov = metrics.position_covariance(belief)
        self.assertAlmostEqual(math.sqrt(cov[0, 0]), sigma, delta=0.3)


class BearingResidualDiagnosticTest(unittest.TestCase):
    def test_signed_null_stratified_and_mode_pose_consistent(self):
        catalog = SimpleNamespace(
            east_m=np.array([100.0]), north_m=np.array([0.0]),
            index_of=lambda landmark_id: {"lm": 0}[landmark_id])
        measurement = structs.TrackletMeasurement("trk", 0, 100.0, 10.0)
        mode = structs.ModeRecord(
            mode_id=7, weight=1.0, n_particles=10,
            mean_east_m=0.0, mean_north_m=100.0,
            mean_heading_deg=0.0, position_std_m=1.0,
            heading_std_deg=1.0, birth_keyframe_idx=0)
        whole = structs.AssociationPosterior(
            "trk", 0, 0.1, {"lm": 0.8})
        mode_specific = structs.AssociationPosterior(
            "trk", 0, 0.9, {"lm": 0.1}, mode_id=7)
        health = structs.HealthRecord(
            keyframe_idx=0, ess=10.0, resampled=False,
            mean_east_m=0.0, mean_north_m=0.0, mean_heading_deg=0.0,
            map_east_m=0.0, map_north_m=0.0, map_heading_deg=0.0,
            position_std_m=1.0, heading_std_deg=1.0, n_measurements=1,
            associations=[whole, mode_specific], modes=[mode])

        diagnostics = metrics.bearing_residual_diagnostics(
            catalog, [measurement], [health])

        self.assertEqual(len(diagnostics), 2)
        self.assertAlmostEqual(diagnostics[0].signed_residual_deg, 10.0)
        self.assertFalse(diagnostics[0].null_dominated)
        self.assertEqual(diagnostics[1].mode_id, 7)
        self.assertEqual(diagnostics[1].pose_north_m, 100.0)
        self.assertAlmostEqual(diagnostics[1].signed_residual_deg, -35.0)
        self.assertTrue(diagnostics[1].null_dominated)


if __name__ == "__main__":
    unittest.main()
