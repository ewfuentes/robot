"""Filter consistency: is the reported uncertainty honest? (design doc T-F2)

Everything else in the suite scores error *magnitude*. This file scores
*calibration* — whether the covariance and the credible sets the filter
publishes match the errors it actually makes. That is the property that
matters for a component whose output is a belief consumed downstream (§5.1),
and it is the test that catches the whole class of defects where the filter
converges to a plausible-looking answer and is far more certain of it than
the evidence supports.

Treat this file as the gate: if it fails, no accuracy number elsewhere in
the suite means anything.
"""

import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as pf,
    scenario,
    structs,
)

_PERIOD_S = 5.0
_N_SEEDS = 10
# chi^2(2 dof): mean 2.0, 95th percentile 5.99.
_NEES_DOF = 2
_NEES_95 = 5.99

# Multi-seed runs are the expensive part of this file and several tests want
# the same ones; memoize on the exact configuration.
_RUN_CACHE = {}


def _run_seeds(scenario_config, n_particles=12000, n_seeds=_N_SEEDS,
               **config_overrides):
    """Run one scenario across seeds; return (final beliefs, truth)."""
    # ScenarioConfig is frozen but holds lists, so it is unhashable; its repr
    # is a faithful stand-in within a process.
    key = (repr(scenario_config), n_particles, n_seeds,
           tuple(sorted(config_overrides.items())))
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]
    data = scenario.generate(scenario_config)
    start = data.truth[0]
    beliefs = []
    for seed in range(n_seeds):
        config = structs.FilterConfig(
            n_particles=n_particles, seed=seed,
            init=structs.GaussianInit(start.east_m + 300.0,
                                      start.north_m - 200.0, 500.0),
            checkpoint_every=1000, **config_overrides)
        history = pf.run_filter(config, data.catalog, data.odometry,
                                data.measurements, data.tables)
        beliefs.append(history.final_belief)
    _RUN_CACHE[key] = (beliefs, data.truth)
    return beliefs, data.truth


class PositionConsistencyTest(unittest.TestCase):
    def test_nees_within_chi_squared_bounds(self):
        """T-F2: average NEES near the 2-dof mean, and the 95% credible
        region must actually contain truth about 95% of the time."""
        beliefs, truth = _run_seeds(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S))
        final = truth[-1]
        nees = np.array([pf.position_nees(b, final.east_m, final.north_m)
                         for b in beliefs])

        # Bound on the mean of N samples of chi^2(2): var = 2*dof, so the
        # standard error of the mean is sqrt(2*dof/N).
        stderr = math.sqrt(2.0 * _NEES_DOF / len(nees))
        self.assertLess(float(np.mean(nees)), _NEES_DOF + 4.0 * stderr,
                        f"overconfident: mean NEES {np.mean(nees):.1f} "
                        f"(ideal {_NEES_DOF}); per-seed {np.round(nees, 1)}")
        # Being wildly *under*-confident is also a defect: it means the
        # belief is uninformative even though it converged.
        self.assertGreater(float(np.mean(nees)), 0.05,
                           f"implausibly conservative: mean NEES "
                           f"{np.mean(nees):.3f}")
        exceed = float(np.mean(nees > _NEES_95))
        self.assertLess(exceed, 0.35,
                        f"{exceed:.0%} of seeds outside the 95% bound")

    def test_error_and_reported_sigma_agree(self):
        """The headline symptom, stated directly: mean error must be within
        a small multiple of the mean reported sigma."""
        beliefs, truth = _run_seeds(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S))
        final = truth[-1]
        errors = np.array([
            math.hypot(pf.mean_pose(b)[0] - final.east_m,
                       pf.mean_pose(b)[1] - final.north_m) for b in beliefs])
        sigmas = np.array([pf.position_std_m(b) for b in beliefs])
        ratio = float(np.mean(errors) / np.mean(sigmas))
        self.assertLess(ratio, 2.0,
                        f"error {np.mean(errors):.1f} m vs reported sigma "
                        f"{np.mean(sigmas):.1f} m (ratio {ratio:.1f})")


class ResamplingRegularizationTest(unittest.TestCase):
    def test_unregularized_resampling_is_overconfident(self):
        """Documents *why* `resample_regularization` defaults to 1.0: with
        it switched off, repeated resampling collapses particle diversity and
        NEES blows up. Guards against someone defaulting it back to 0."""
        final_truth = None
        nees_by_setting = {}
        for regularization in (0.0, 1.0):
            beliefs, truth = _run_seeds(
                scenario.harbor_loop(keyframe_period_s=_PERIOD_S),
                n_particles=4000, n_seeds=8,
                resample_regularization=regularization)
            final_truth = truth[-1]
            nees_by_setting[regularization] = float(np.mean([
                pf.position_nees(b, final_truth.east_m, final_truth.north_m)
                for b in beliefs]))
        self.assertGreater(nees_by_setting[0.0], 4.0 * nees_by_setting[1.0],
                           f"expected unregularized resampling to be much "
                           f"more overconfident, got {nees_by_setting}")
        self.assertLess(nees_by_setting[1.0], 6.0)


class NoiseFloorConsistencyTest(unittest.TestCase):
    """T-F12: filter consistency must not depend on producers keeping sigma
    generous (§5.2 [CONTRACT]). Particle diversity is owned by the
    kernel-regularized resampler, so NEES must hold even when the odometry
    is nearly noiseless — the regime where the archived model relied on
    odometry noise as its only diversity between resamples."""

    def test_nees_with_starved_odometry_noise(self):
        beliefs, truth = _run_seeds(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                 odom_sigma_m=0.05, dyaw_sigma_deg=0.1),
            n_seeds=6)
        final = truth[-1]
        nees = np.array([pf.position_nees(b, final.east_m, final.north_m)
                         for b in beliefs])
        self.assertLess(float(np.mean(nees)), 8.0,
                        f"cloud starvation at small sigma: per-seed NEES "
                        f"{np.round(nees, 1)}")


class CorrelatedOdometryMismatchTest(unittest.TestCase):
    """T-F13: correlated systematic odometry errors (gyro-rate bias, scale
    error) violate per-step independence anti-conservatively — true error
    grows with N steps while the filter budgets sqrt(N). This test owns the
    §5.2 decision between sigma-inflation and per-particle nuisance states:
    it measures the NEES cost at plausible mismatch levels and verifies that
    inflating the heading random walk recovers consistency."""

    _MISMATCH = dict(gyro_bias_deg_per_hr=60.0, odom_scale_error=0.03)

    def test_modest_bias_and_scale_stay_bounded(self):
        beliefs, truth = _run_seeds(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                 **self._MISMATCH),
            n_seeds=6)
        final = truth[-1]
        nees = np.array([pf.position_nees(b, final.east_m, final.north_m)
                         for b in beliefs])
        self.assertLess(float(np.mean(nees)), 12.0,
                        f"correlated mismatch blew up NEES: per-seed "
                        f"{np.round(nees, 1)} — time to revisit the §5.2 "
                        f"sigma-inflation-vs-nuisance-state decision")

    def test_sigma_inflation_recovers_consistency(self):
        beliefs, truth = _run_seeds(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S,
                                 **self._MISMATCH),
            n_seeds=6, heading_random_walk_deg=2.0)
        final = truth[-1]
        nees = np.array([pf.position_nees(b, final.east_m, final.north_m)
                         for b in beliefs])
        self.assertLess(float(np.mean(nees)), 6.0,
                        f"inflation failed to recover consistency: per-seed "
                        f"{np.round(nees, 1)}")


class HeadingConsistencyTest(unittest.TestCase):
    def test_reported_heading_sigma_covers_error(self):
        """A-1's signature was a reported heading sigma *below* the
        single-sample course sigma while the actual error sat above it —
        the fingerprint of reusing one course sample as both prior and
        likelihood."""
        config = scenario.harbor_loop(keyframe_period_s=_PERIOD_S)
        beliefs, truth = _run_seeds(config)
        errors = np.array([
            abs(math.degrees(float(np.asarray(
                pf.mean_pose(b)[2] - math.radians(truth[-1].heading_deg)))))
            for b in beliefs])
        errors = np.abs((errors + 180.0) % 360.0 - 180.0)
        sigmas = np.array([pf.heading_std_deg(b) for b in beliefs])
        self.assertLess(float(np.mean(errors)), 2.0 * float(np.mean(sigmas)),
                        f"heading error {np.mean(errors):.2f} deg vs reported "
                        f"sigma {np.mean(sigmas):.2f} deg")


class BearingsAnchorHeadingTest(unittest.TestCase):
    def test_heading_is_unobservable_without_bearings(self):
        """The odometry carries only yaw increments, so absolute heading
        must come from landmark bearings alone (§5.2). Without bearings the
        heading prior must stay essentially uniform — if it collapses, an
        absolute heading signal is leaking in somewhere."""
        data = scenario.generate(
            scenario.harbor_loop(keyframe_period_s=_PERIOD_S))
        config = structs.FilterConfig(
            n_particles=8000, seed=3,
            init=structs.UniformBoxInit(-2500.0, 2500.0, -2500.0, 2500.0),
            checkpoint_every=1000)
        without = pf.run_filter(config, data.catalog, data.odometry, [], {})
        self.assertGreater(pf.heading_std_deg(without.final_belief), 60.0,
                           "heading collapsed with no bearing measurements: "
                           "absolute course is being used as evidence")

        with_bearings = pf.run_filter(config, data.catalog, data.odometry,
                                      data.measurements, data.tables)
        self.assertLess(pf.heading_std_deg(with_bearings.final_belief), 5.0,
                        "bearings failed to constrain heading")


if __name__ == "__main__":
    unittest.main()
