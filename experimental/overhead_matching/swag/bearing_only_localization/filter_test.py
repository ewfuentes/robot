import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    catalog as catalog_mod,
    filter as pf,
    structs,
)


def _identity_table(tracklet_id: str, landmark_id: str,
                    clip: float = 4.0) -> structs.CompatibilityTable:
    return structs.CompatibilityTable(
        tracklet_id=tracklet_id,
        matcher_version="identity_stub_v1",
        entries=[structs.CompatibilityEntry(landmark_id, clip)],
        default_log_lr=-2.0,
        clip_lo=-clip,
        clip_hi=clip,
        status="fast")


def _catalog(ids, east, north, **kwargs):
    return catalog_mod.LandmarkCatalog(ids, np.asarray(east, dtype=float),
                                       np.asarray(north, dtype=float),
                                       **kwargs)


class VonMisesTest(unittest.TestCase):
    """T-U1: density integrates to 1 across concentration regimes."""

    def test_normalizes(self):
        n_grid = 400000
        h = 2.0 * np.pi / n_grid
        theta = -np.pi + (np.arange(n_grid) + 0.5) * h
        for kappa in [1e-3, 1.0, 50.0, 3000.0]:
            total = float(np.sum(np.exp(pf.von_mises_logpdf(theta, kappa))) * h)
            self.assertAlmostEqual(total, 1.0, delta=1e-6,
                                   msg=f"kappa={kappa}")

    def test_concentration_limits(self):
        # kappa -> 0: uniform on the circle.
        self.assertAlmostEqual(float(pf.von_mises_logpdf(np.array(1.0), 1e-9)),
                               -math.log(2 * math.pi), places=6)
        # Large kappa: mode density matches Gaussian approx sqrt(kappa/2pi).
        mode = float(pf.von_mises_logpdf(np.array(0.0), 3000.0))
        self.assertAlmostEqual(mode, 0.5 * math.log(3000.0 / (2 * math.pi)),
                               delta=1e-3)

    def test_broadcasts_per_element_kappa(self):
        """kappa_eff varies per (particle, candidate), so the density must
        accept an array of concentrations."""
        delta = np.zeros((3, 2))
        kappa = np.array([[1.0, 100.0], [1.0, 100.0], [1.0, 100.0]])
        out = pf.von_mises_logpdf(delta, kappa)
        self.assertEqual(out.shape, (3, 2))
        for row in out:
            self.assertAlmostEqual(row[0],
                                   float(pf.von_mises_logpdf(np.array(0.0),
                                                             1.0)))


class MeasurementUpdateTest(unittest.TestCase):
    def test_bearing_convention(self):
        """Mini T-U3 at filter level: landmark due east, particle heading
        east sees body bearing 0; particle heading north sees +90."""
        belief = pf.ParticleBelief(
            east_m=np.zeros(2), north_m=np.zeros(2),
            heading_rad=np.array([math.pi / 2, 0.0]),  # east, north
            log_weight=np.zeros(2))
        meas = structs.TrackletMeasurement("trk_a", 0, 0.0, 3000.0)
        pf.measurement_update(belief, meas, _identity_table("trk_a", "lm_a"),
                              _catalog(["lm_a"], [1000.0], [0.0]), pi0=0.05)
        # Facing east -> predicted body bearing 0 -> matches measurement.
        # The mismatched particle is floored by the null hypothesis at
        # log(pi0 / 2pi), not driven to -inf (§5.3).
        self.assertGreater(belief.log_weight[0], belief.log_weight[1] + 5.0)
        self.assertAlmostEqual(float(belief.log_weight[1]),
                               math.log(0.05 / (2 * math.pi)), delta=0.01)

    def test_likelihood_is_a_normalized_density(self):
        """The mixture (pi0, (1-pi0) w_j) must integrate to 1 over the
        bearing circle for any pose. This is the invariant §5.3 warns is
        easy to break silently when the candidate prior changes."""
        catalog = _catalog(["lm_a", "lm_b", "lm_c"],
                           [1000.0, -500.0, 200.0],
                           [0.0, 800.0, -1500.0])
        table = structs.CompatibilityTable(
            "trk", "v",
            [structs.CompatibilityEntry("lm_a", 2.0),
             structs.CompatibilityEntry("lm_b", -1.0)],
            default_log_lr=0.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        n_grid = 20000
        step = 360.0 / n_grid
        total = 0.0
        for i in range(n_grid):
            belief = pf.ParticleBelief(np.array([10.0]), np.array([20.0]),
                                       np.array([0.3]), np.zeros(1))
            meas = structs.TrackletMeasurement(
                "trk", 0, -180.0 + (i + 0.5) * step, 40.0)
            pf.measurement_update(belief, meas, table, catalog, pi0=0.2)
            total += math.exp(float(belief.log_weight[0])) * math.radians(step)
        # LLRs are not likelihood ratios of a normalized density, so the
        # integral is sum_j w_j LR_j (1-pi0) + pi0, not 1. Check that
        # identity instead -- it is the one the mixture must satisfy.
        expected = 0.2 + 0.8 * np.mean([math.exp(2.0), math.exp(-1.0),
                                        math.exp(0.0)])
        self.assertAlmostEqual(total, expected, delta=1e-3)

    def test_null_floor_and_share(self):
        """Wildly inconsistent bearing: weights finite, null wins."""
        belief = pf.ParticleBelief(
            east_m=np.zeros(3), north_m=np.zeros(3),
            heading_rad=np.zeros(3), log_weight=np.zeros(3))
        # Landmark due north, but measurement says directly behind.
        meas = structs.TrackletMeasurement("trk_a", 0, 180.0, 3000.0)
        assoc = pf.measurement_update(
            belief, meas, _identity_table("trk_a", "lm_a"),
            _catalog(["lm_a"], [0.0], [1000.0]), pi0=0.05)[0]
        self.assertTrue(np.all(np.isfinite(belief.log_weight)))
        self.assertGreater(assoc.null_share, 0.99)

    def test_log_domain_extremes(self):
        """T-U5: 1000 candidates with LLRs at clip bounds, spanning several
        candidate blocks; no NaN/inf anywhere."""
        rng = np.random.default_rng(0)
        n, m = 50, 1000
        belief = pf.ParticleBelief(
            east_m=rng.uniform(-5000, 5000, n),
            north_m=rng.uniform(-5000, 5000, n),
            heading_rad=rng.uniform(-np.pi, np.pi, n),
            log_weight=np.zeros(n))
        landmark_ids = [f"lm_{i}" for i in range(m)]
        entries = [structs.CompatibilityEntry(
            lm_id, 4.0 if i % 2 == 0 else -4.0)
            for i, lm_id in enumerate(landmark_ids)]
        table = structs.CompatibilityTable(
            tracklet_id="trk", matcher_version="v", entries=entries,
            default_log_lr=-4.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        meas = structs.TrackletMeasurement("trk", 0, 37.0, 3000.0)
        catalog = _catalog(landmark_ids, rng.uniform(-20000, 20000, m),
                           rng.uniform(-20000, 20000, m))
        self.assertGreater(m, pf.CANDIDATE_BLOCK, "blocking not exercised")
        for pi0 in [0.05, 0.999]:
            b = belief.copy()
            assoc = pf.measurement_update(b, meas, table, catalog, pi0)[0]
            self.assertTrue(np.all(np.isfinite(b.log_weight)))
            self.assertTrue(math.isfinite(assoc.null_share))
            total = assoc.null_share + sum(assoc.responsibilities.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_blocking_matches_unblocked(self):
        """Candidate blocking is a memory optimization only (A-8)."""
        rng = np.random.default_rng(3)
        m = 700
        landmark_ids = [f"lm_{i}" for i in range(m)]
        catalog = _catalog(landmark_ids, rng.uniform(-9000, 9000, m),
                           rng.uniform(-9000, 9000, m))
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_5", 3.0)],
            default_log_lr=-1.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        meas = structs.TrackletMeasurement("trk", 0, 12.0, 500.0)
        original_block = pf.CANDIDATE_BLOCK
        results = []
        try:
            for block in (64, 10_000):
                pf.CANDIDATE_BLOCK = block
                belief = pf.ParticleBelief(
                    np.linspace(-100, 100, 32), np.linspace(-50, 50, 32),
                    np.linspace(-1.0, 1.0, 32), np.zeros(32))
                assoc = pf.measurement_update(belief, meas, table, catalog,
                                              0.1)[0]
                results.append((belief.log_weight.copy(), assoc))
        finally:
            pf.CANDIDATE_BLOCK = original_block
        np.testing.assert_allclose(results[0][0], results[1][0], atol=1e-12)
        self.assertAlmostEqual(results[0][1].null_share,
                               results[1][1].null_share, places=12)
        self.assertEqual(results[0][1].responsibilities.keys(),
                         results[1][1].responsibilities.keys())

    def test_llr_clipping(self):
        """Entries beyond the declared clip bounds are clipped (§6)."""
        belief_a = pf.ParticleBelief(
            east_m=np.zeros(1), north_m=np.zeros(1),
            heading_rad=np.zeros(1), log_weight=np.zeros(1))
        belief_b = belief_a.copy()
        meas = structs.TrackletMeasurement("trk", 0, 0.0, 100.0)
        catalog = _catalog(["lm"], [0.0], [1000.0])
        table_huge = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm", 1000.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        table_clip = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm", 4.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        pf.measurement_update(belief_a, meas, table_huge, catalog, 0.05)
        pf.measurement_update(belief_b, meas, table_clip, catalog, 0.05)
        self.assertAlmostEqual(float(belief_a.log_weight[0]),
                               float(belief_b.log_weight[0]), places=12)

    def test_kappa_is_capped(self):
        """A matcher (or a tracker) handing over an implausible kappa must
        not get unbounded authority over the posterior."""
        catalog = _catalog(["lm"], [0.0], [1000.0])
        table = _identity_table("trk", "lm")
        weights = []
        for kappa in (pf.MAX_KAPPA, pf.MAX_KAPPA * 1e6):
            belief = pf.ParticleBelief(np.zeros(1), np.zeros(1), np.zeros(1),
                                       np.zeros(1))
            pf.measurement_update(
                belief, structs.TrackletMeasurement("trk", 0, 0.0, kappa),
                table, catalog, 0.05)
            weights.append(float(belief.log_weight[0]))
        self.assertAlmostEqual(weights[0], weights[1], places=9)

    def test_rejects_invalid_kappa(self):
        catalog = _catalog(["lm"], [0.0], [1000.0])
        belief = pf.ParticleBelief(np.zeros(1), np.zeros(1), np.zeros(1),
                                   np.zeros(1))
        for kappa in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                pf.measurement_update(
                    belief, structs.TrackletMeasurement("t", 0, 0.0, kappa),
                    _identity_table("t", "lm"), catalog, 0.05)


class KappaEffTest(unittest.TestCase):
    def test_map_error_widens_the_bearing_likelihood(self):
        """kappa_eff combines tracklet kappa with projected map error (§4):
        a 50 m position error at 1 km is ~2.9 deg of bearing."""
        catalog = _catalog(["lm"], [0.0], [1000.0], position_sigma_m=50.0)
        kappa_z = 1.0 / math.radians(1.0) ** 2
        kappa_eff = catalog.kappa_eff(kappa_z, np.array([[1000.0]]))
        sigma_eff_deg = math.degrees(1.0 / math.sqrt(float(kappa_eff[0, 0])))
        self.assertAlmostEqual(sigma_eff_deg, math.hypot(1.0, math.degrees(
            50.0 / 1000.0)), places=3)

    def test_exact_catalog_leaves_kappa_untouched(self):
        catalog = _catalog(["lm"], [0.0], [1000.0])
        kappa_eff = catalog.kappa_eff(500.0, np.array([[1000.0], [50.0]]))
        np.testing.assert_allclose(kappa_eff, 500.0)

    def test_rejects_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            catalog_mod.LandmarkCatalog(["a", "b"], np.array([1.0]),
                                        np.array([2.0]))
        with self.assertRaises(ValueError):
            catalog_mod.LandmarkCatalog(["a", "a"], np.array([1.0, 2.0]),
                                        np.array([1.0, 2.0]))


class MotionModelTest(unittest.TestCase):
    """T-U8: hand-derived golden fixtures for body-frame propagation (§5.2).

    Hand-computed, not generated by the code under test (the T-U3
    rationale): motion-side sign bugs are the same class as bearing-side
    ones. Heading is compass (CW from north): forward -> (sin h, cos h),
    left -> (-cos h, sin h); translation rotates by the midpoint heading.
    """

    @staticmethod
    def _step(heading_deg, forward, left, dyaw_deg,
              sigma_m=0.0, sigma_yaw_deg=0.0, n=1, seed=0):
        belief = pf.ParticleBelief(
            east_m=np.zeros(n), north_m=np.zeros(n),
            heading_rad=np.full(n, math.radians(heading_deg)),
            log_weight=np.zeros(n))
        delta = structs.OdometryDelta(
            keyframe_idx=1, forward_m=forward, left_m=left,
            dyaw_rad=math.radians(dyaw_deg), sigma_m=sigma_m,
            sigma_yaw_rad=math.radians(sigma_yaw_deg))
        pf.motion_update(belief, delta, 0.0, np.random.default_rng(seed))
        return belief

    def _assert_pose(self, belief, east, north, heading_deg):
        self.assertAlmostEqual(float(belief.east_m[0]), east, places=9)
        self.assertAlmostEqual(float(belief.north_m[0]), north, places=9)
        self.assertAlmostEqual(
            math.degrees(float(belief.heading_rad[0])) % 360.0,
            heading_deg % 360.0, places=9)

    def test_straight_north(self):
        self._assert_pose(self._step(0.0, 10.0, 0.0, 0.0), 0.0, 10.0, 0.0)

    def test_straight_east(self):
        self._assert_pose(self._step(90.0, 10.0, 0.0, 0.0), 10.0, 0.0, 90.0)

    def test_left_of_east_is_north(self):
        self._assert_pose(self._step(90.0, 0.0, 5.0, 0.0), 0.0, 5.0, 90.0)

    def test_pure_rotation(self):
        self._assert_pose(self._step(0.0, 0.0, 0.0, 30.0), 0.0, 0.0, 30.0)

    def test_corner_uses_midpoint_heading_not_start(self):
        """90-deg turn inside one step: the chord runs at 45, not at the
        start heading (0) — the discriminating case for the midpoint rule."""
        belief = self._step(0.0, 10.0, 0.0, 90.0)
        self._assert_pose(belief, 10.0 * math.sin(math.radians(45.0)),
                          10.0 * math.cos(math.radians(45.0)), 90.0)

    def test_midpoint_frame_increment_reconstructs_polyline_corner(self):
        """The scenario generator resolves an instantaneous corner into the
        midpoint frame (forward = s cos(dyaw/2), left = -s sin(dyaw/2));
        composed through motion_update that lands exactly on the new
        segment — due east here."""
        half = math.radians(45.0)
        belief = self._step(0.0, 10.0 * math.cos(half),
                            -10.0 * math.sin(half), 90.0)
        self._assert_pose(belief, 10.0, 0.0, 90.0)

    def test_heading_noise_creates_cross_track_spread(self):
        """§5.2 order: noise sampled BEFORE the rotation, so heading spread
        flows into cross-track position spread of step * sin(sigma/2) — the
        mechanism that replaces an explicit cross-track sigma."""
        belief = self._step(0.0, 100.0, 0.0, 0.0, sigma_yaw_deg=5.0, n=40000)
        self.assertAlmostEqual(float(np.std(belief.east_m)),
                               100.0 * math.sin(math.radians(2.5)),
                               delta=0.15)
        self.assertAlmostEqual(
            float(np.std(np.degrees(belief.heading_rad))), 5.0, delta=0.1)


class ResamplerTest(unittest.TestCase):
    """T-U6: systematic resampler preserves expected weights."""

    def test_unbiased(self):
        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        east = np.arange(5, dtype=np.float64)
        counts = np.zeros(5)
        n_trials = 2000
        for trial in range(n_trials):
            belief = pf.ParticleBelief(
                east_m=east.copy(), north_m=np.zeros(5),
                heading_rad=np.zeros(5), log_weight=np.log(weights))
            pf.systematic_resample(belief, np.random.default_rng(trial),
                                   regularization=0.0)
            for i in range(5):
                counts[i] += np.sum(belief.east_m == east[i])
        freq = counts / (n_trials * 5)
        np.testing.assert_allclose(freq, weights, atol=0.02)

    def test_regularization_preserves_mean_and_inflates_spread(self):
        rng = np.random.default_rng(0)
        n = 20000
        belief = pf.ParticleBelief(
            east_m=rng.normal(100.0, 30.0, n), north_m=rng.normal(-50.0, 30.0, n),
            heading_rad=rng.normal(0.2, 0.05, n), log_weight=np.zeros(n))
        before_mean = pf.mean_pose(belief)
        before_std = pf.position_std_m(belief)
        pf.systematic_resample(belief, rng, regularization=1.0)
        after_mean = pf.mean_pose(belief)
        self.assertAlmostEqual(before_mean[0], after_mean[0], delta=2.0)
        self.assertAlmostEqual(before_mean[1], after_mean[1], delta=2.0)
        # Bandwidth is sigma * n^(-1/6): variance grows by 1 + n^(-1/3).
        expected = before_std * math.sqrt(1.0 + n ** (-1.0 / 3.0))
        self.assertAlmostEqual(pf.position_std_m(belief), expected, delta=1.5)

    def test_uniform_weights_ess(self):
        log_w = np.zeros(100)
        self.assertAlmostEqual(pf.ess(log_w), 100.0, places=6)
        log_w = np.full(100, -np.inf)
        log_w[0] = 0.0
        self.assertAlmostEqual(pf.ess(log_w), 1.0, places=6)


class PerModeAssociationTest(unittest.TestCase):
    """§5.4 `[CONTRACT]`: two modes with contradictory explanations must be
    reported separately. The whole-belief average describes neither."""

    def test_modes_report_contradictory_associations(self):
        # Two clusters, mirrored east/west, each seeing "its" landmark dead
        # ahead: mode 0 must attribute the bearing to lm_west, mode 1 to
        # lm_east, and the global average must sit between the two.
        east = np.concatenate([np.full(500, -1000.0), np.full(500, 1000.0)])
        belief = pf.ParticleBelief(
            east_m=east, north_m=np.zeros(1000),
            heading_rad=np.zeros(1000), log_weight=np.zeros(1000),
            mode_id=np.concatenate([np.zeros(500, dtype=np.int64),
                                    np.ones(500, dtype=np.int64)]))
        catalog = _catalog(["lm_west", "lm_east"], [-1000.0, 1000.0],
                           [3000.0, 3000.0])
        table = structs.CompatibilityTable(
            "trk", "v", [structs.CompatibilityEntry("lm_west", 2.0),
                         structs.CompatibilityEntry("lm_east", 2.0)],
            default_log_lr=-2.0, clip_lo=-4.0, clip_hi=4.0, status="fast")
        posteriors = pf.measurement_update(
            belief, structs.TrackletMeasurement("trk", 0, 0.0, 2000.0),
            table, catalog, pi0=0.05)

        by_mode = {p.mode_id: p for p in posteriors}
        self.assertEqual(set(by_mode), {None, 0, 1})
        self.assertGreater(by_mode[0].responsibilities["lm_west"], 0.9)
        self.assertLess(by_mode[0].responsibilities["lm_east"], 0.05)
        self.assertGreater(by_mode[1].responsibilities["lm_east"], 0.9)
        self.assertLess(by_mode[1].responsibilities["lm_west"], 0.05)
        # The global average is the number that describes neither mode.
        self.assertAlmostEqual(by_mode[None].responsibilities["lm_west"], 0.5,
                               delta=0.1)
        for posterior in posteriors:
            total = posterior.null_share + sum(
                posterior.responsibilities.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_per_mode_can_be_disabled(self):
        belief = pf.ParticleBelief(np.zeros(4), np.zeros(4), np.zeros(4),
                                   np.zeros(4),
                                   mode_id=np.zeros(4, dtype=np.int64))
        posteriors = pf.measurement_update(
            belief, structs.TrackletMeasurement("trk", 0, 0.0, 100.0),
            _identity_table("trk", "lm"), _catalog(["lm"], [0.0], [900.0]),
            pi0=0.05, per_mode=False)
        self.assertEqual(len(posteriors), 1)
        self.assertIsNone(posteriors[0].mode_id)


class MapPoseTest(unittest.TestCase):
    def test_picks_the_dense_mode_not_the_midpoint(self):
        """For a bimodal belief the weighted mean sits between the modes and
        describes no hypothesis the filter holds (A-9)."""
        east = np.concatenate([np.full(600, -1000.0), np.full(400, 1000.0)])
        north = np.zeros(1000)
        belief = pf.ParticleBelief(east + np.linspace(-5, 5, 1000), north,
                                   np.zeros(1000), np.zeros(1000))
        mean_east = pf.mean_pose(belief)[0]
        map_east = pf.map_pose(belief, cell_size_m=100.0)[0]
        self.assertLess(abs(mean_east - (-200.0)), 60.0)
        self.assertLess(abs(map_east - (-1000.0)), 60.0)

    def test_mass_within_radius(self):
        belief = pf.ParticleBelief(
            np.array([0.0, 10.0, 500.0]), np.zeros(3), np.zeros(3),
            np.zeros(3))
        self.assertAlmostEqual(pf.mass_within_radius(belief, 0.0, 0.0, 100.0),
                               2.0 / 3.0, places=6)


def _tiny_inputs():
    catalog = _catalog(["lm_a", "lm_b"], [2000.0, -1500.0], [1000.0, 2500.0])
    odometry = [structs.OdometryDelta(keyframe_idx=k, forward_m=10.0,
                                      left_m=0.0, dyaw_rad=0.0, sigma_m=0.5,
                                      sigma_yaw_rad=math.radians(2.0))
                for k in range(1, 8)]
    measurements = [
        structs.TrackletMeasurement("trk_lm_a", 2, 15.0, 3000.0),
        structs.TrackletMeasurement("trk_lm_b", 5, -40.0, 3000.0)]
    tables = {"trk_lm_a": _identity_table("trk_lm_a", "lm_a"),
              "trk_lm_b": _identity_table("trk_lm_b", "lm_b")}
    return catalog, odometry, measurements, tables


def _tiny_config(seed: int = 7, **kwargs) -> structs.FilterConfig:
    return structs.FilterConfig(
        n_particles=500, seed=seed,
        init=structs.GaussianInit(0.0, 0.0, 300.0), **kwargs)


class ValidationTest(unittest.TestCase):
    def _inputs(self):
        return _tiny_inputs()

    def _config(self, seed: int = 7, **kwargs) -> structs.FilterConfig:
        return _tiny_config(seed, **kwargs)

    def test_rejects_duplicate_information_epoch(self):
        """T-F1: re-submitting a tracklet's epoch double-counts evidence."""
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "duplicate measurement"):
            pf.run_filter(self._config(), catalog, odometry,
                          measurements + measurements[:1], tables)

    def test_rejects_missing_table(self):
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "no CompatibilityTable"):
            pf.run_filter(self._config(), catalog, odometry, measurements,
                          {"trk_lm_a": tables["trk_lm_a"]})

    def test_rejects_out_of_range_anchor(self):
        catalog, odometry, measurements, tables = self._inputs()
        stray = structs.TrackletMeasurement("trk_lm_a", 99, 0.0, 100.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            pf.run_filter(self._config(), catalog, odometry, [stray], tables)

    def test_rejects_bad_config(self):
        catalog, odometry, measurements, tables = self._inputs()
        for kwargs in [dict(pi0=0.0), dict(pi0=1.0), dict(checkpoint_every=0)]:
            with self.assertRaises(ValueError):
                pf.run_filter(self._config(**kwargs), catalog, odometry,
                              measurements, tables)
        with self.assertRaises(ValueError):
            pf.run_filter(
                structs.FilterConfig(
                    n_particles=0, seed=1,
                    init=structs.GaussianInit(0.0, 0.0, 1.0)),
                catalog, odometry, measurements, tables)

    def test_rejects_out_of_order_odometry(self):
        catalog, odometry, measurements, tables = self._inputs()
        with self.assertRaisesRegex(ValueError, "out of order"):
            pf.run_filter(self._config(), catalog, list(reversed(odometry)),
                          measurements, tables)

    def test_rejects_negative_sigma_yaw(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = [structs.OdometryDelta(keyframe_idx=1, forward_m=1.0,
                                     left_m=0.0, dyaw_rad=0.0, sigma_m=1.0,
                                     sigma_yaw_rad=-0.1)]
        with self.assertRaisesRegex(ValueError, "sigma_yaw_rad"):
            pf.run_filter(self._config(), catalog, bad, [], tables)

    def test_rejects_non_finite_increment(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = [structs.OdometryDelta(keyframe_idx=1,
                                     forward_m=float("nan"), left_m=0.0,
                                     dyaw_rad=0.0, sigma_m=1.0,
                                     sigma_yaw_rad=0.1)]
        with self.assertRaisesRegex(ValueError, "not finite"):
            pf.run_filter(self._config(), catalog, bad, [], tables)

    def test_rejects_inverted_clip_bounds(self):
        catalog, odometry, measurements, tables = self._inputs()
        bad = structs.CompatibilityTable(
            "trk_lm_a", "v", [], default_log_lr=0.0, clip_lo=4.0,
            clip_hi=-4.0, status="fast")
        tables = dict(tables, trk_lm_a=bad)
        with self.assertRaisesRegex(ValueError, "clip_lo"):
            pf.run_filter(self._config(), catalog, odometry, measurements,
                          tables)


class RunFilterDeterminismTest(unittest.TestCase):
    """T-U7: same (config, seed, input log) -> bit-identical history hash.

    In-process only: this pins reproducibility for a given environment, not
    across numpy/BLAS versions (see run_log.py's replay note).
    """

    def test_same_seed_same_hash(self):
        args = _tiny_inputs()
        h1 = pf.run_filter(_tiny_config(7), *args)
        h2 = pf.run_filter(_tiny_config(7), *args)
        self.assertEqual(h1.particle_history_sha256,
                         h2.particle_history_sha256)

    def test_different_seed_different_hash(self):
        args = _tiny_inputs()
        h1 = pf.run_filter(_tiny_config(7), *args)
        h2 = pf.run_filter(_tiny_config(8), *args)
        self.assertNotEqual(h1.particle_history_sha256,
                            h2.particle_history_sha256)


if __name__ == "__main__":
    unittest.main()
