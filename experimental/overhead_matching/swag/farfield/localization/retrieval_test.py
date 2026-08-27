"""Retrieval observation source: engine math, artifact I/O, and end-to-end
filter behaviour on synthetic score fields (CLD-3)."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter as pf,
    metrics,
    retrieval,
    scenario,
    structs,
)


def _grid_fields(scores: np.ndarray, spacing_m: float = 100.0,
                 frame: geo.RegionFrame | None = None,
                 keyframe_idx=None) -> retrieval.ScoreFields:
    """ScoreFields over a full rows x cols grid centred on the origin.

    scores: (K, rows, cols, N) -> flattened row-major to (K, L, N).
    """
    k, rows, cols, n_bins = scores.shape
    east = (np.arange(cols) - (cols - 1) / 2.0) * spacing_m
    north = (np.arange(rows) - (rows - 1) / 2.0) * spacing_m
    ee, nn = np.meshgrid(east, north)
    meta = retrieval.RetrievalFieldsMeta(
        schema_version=structs.SCHEMA_VERSION, dataset="test",
        n_keyframes=k, n_nodes=rows * cols, n_heading_bins=n_bins,
        node_spacing_m=spacing_m, db_dir="test", db_manifest_sha256="0" * 64,
        scorer="test")
    return retrieval.ScoreFields(
        meta=meta, east_m=ee.ravel(), north_m=nn.ravel(),
        scores=scores.reshape(k, rows * cols, n_bins).astype(np.float32),
        keyframe_idx=np.asarray(
            keyframe_idx if keyframe_idx is not None else np.arange(k)),
        pano_ids=[f"p{i}" for i in range(k)])


_CONFIG = structs.RetrievalConfig(temperature=1.0, outlier_epsilon=0.05)


class EngineMathTest(unittest.TestCase):
    def test_peak_beats_far_node_and_floor_holds_outside(self):
        scores = np.zeros((1, 5, 5, 4), dtype=np.float32)
        scores[0, 2, 2, 1] = 10.0  # centre node, heading bin 1 (90 deg)
        fields = _grid_fields(scores)
        engine = retrieval.RetrievalEngine(fields, _CONFIG)

        heading = math.radians(90.0)
        at_peak = engine.log_likelihood(0, 0.0, 0.0, heading)
        at_far = engine.log_likelihood(0, 200.0, 200.0, heading)
        outside = engine.log_likelihood(0, 10_000.0, 0.0, heading)
        self.assertGreater(float(at_peak), float(at_far))
        expected_floor = math.log(0.05) - math.log(5 * 5 * 4)
        self.assertAlmostEqual(float(outside), expected_floor, places=5)
        # Inside the support, the epsilon floor bounds every pose below.
        self.assertGreaterEqual(float(at_far), expected_floor)

    def test_likelihood_normalizes_over_support(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=(1, 6, 7, 3)).astype(np.float32)
        fields = _grid_fields(scores)
        engine = retrieval.RetrievalEngine(fields, _CONFIG)
        # Summing L(x) over every discrete cell must give 1 (§5.5).
        total = 0.0
        for node in range(fields.scores.shape[1]):
            for b in range(3):
                total += math.exp(engine.log_likelihood(
                    0, float(fields.east_m[node]),
                    float(fields.north_m[node]),
                    b * 2.0 * math.pi / 3))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_heading_interpolation_is_circular_linear(self):
        scores = np.zeros((1, 3, 3, 4), dtype=np.float32)
        scores[0, 1, 1, 0] = 2.0
        scores[0, 1, 1, 3] = 4.0
        fields = _grid_fields(scores)
        engine = retrieval.RetrievalEngine(fields, _CONFIG)
        # Halfway between bin 3 (270 deg) and bin 0 (360 == 0 deg): the wrap.
        halfway = engine.log_likelihood(0, 0.0, 0.0, math.radians(315.0))
        at3 = engine.log_likelihood(0, 0.0, 0.0, math.radians(270.0))
        at0 = engine.log_likelihood(0, 0.0, 0.0, 0.0)
        self.assertGreater(float(halfway), float(at0))
        self.assertLess(float(halfway), float(at3))

    def test_update_mutates_weights_and_returns_no_associations(self):
        scores = np.zeros((1, 3, 3, 4), dtype=np.float32)
        fields = _grid_fields(scores)
        engine = retrieval.RetrievalEngine(fields, _CONFIG)
        belief = pf.init_belief(
            structs.FilterConfig(
                n_particles=64, seed=0,
                init=structs.UniformBoxInit(-100.0, 100.0, -100.0, 100.0)),
            np.random.default_rng(0))
        before = belief.log_weight.copy()
        out = engine.update(belief, structs.RetrievalMeasurement(
            keyframe_idx=0, field_idx=0, pano_id="p0"))
        self.assertEqual(out, [])
        self.assertFalse(np.array_equal(before, belief.log_weight))

    def test_bad_calibration_rejected(self):
        fields = _grid_fields(np.zeros((1, 3, 3, 4), dtype=np.float32))
        for temperature, epsilon in ((0.0, 0.05), (1.0, 0.0), (1.0, 1.0)):
            with self.assertRaises(ValueError):
                retrieval.RetrievalEngine(fields, structs.RetrievalConfig(
                    temperature=temperature, outlier_epsilon=epsilon))

    def test_rotated_lattice_supported(self):
        # A metric-CRS lattice arrives rotated in the run's ENU frame by the
        # meridian convergence; nearest-node lookup must not assume axis
        # alignment. Peak at the centre node survives a 3-degree rotation.
        scores = np.zeros((1, 5, 5, 4), dtype=np.float32)
        scores[0, 2, 2, 0] = 10.0
        fields = _grid_fields(scores)
        theta = math.radians(3.0)
        east = (fields.east_m * math.cos(theta)
                - fields.north_m * math.sin(theta))
        north = (fields.east_m * math.sin(theta)
                 + fields.north_m * math.cos(theta))
        rotated = retrieval.ScoreFields(
            meta=fields.meta, east_m=east, north_m=north,
            scores=fields.scores, keyframe_idx=fields.keyframe_idx,
            pano_ids=fields.pano_ids)
        engine = retrieval.RetrievalEngine(rotated, _CONFIG)
        at_peak = engine.log_likelihood(0, 0.0, 0.0, 0.0)
        far_node = engine.log_likelihood(0, east[0], north[0], 0.0)
        self.assertGreater(float(at_peak), float(far_node))

    def test_missing_nodes_fall_back_to_floor(self):
        # A lattice with a hole (water mask): the hole's cell reports floor.
        scores = np.zeros((1, 3, 3, 2), dtype=np.float32)
        fields = _grid_fields(scores)
        keep = np.arange(9) != 4  # drop the centre node
        holey = retrieval.ScoreFields(
            meta=retrieval.RetrievalFieldsMeta(
                schema_version=structs.SCHEMA_VERSION, dataset="test",
                n_keyframes=1, n_nodes=8, n_heading_bins=2,
                node_spacing_m=100.0, db_dir="test",
                db_manifest_sha256="0" * 64, scorer="test"),
            east_m=fields.east_m[keep], north_m=fields.north_m[keep],
            scores=fields.scores[:, keep], keyframe_idx=fields.keyframe_idx,
            pano_ids=fields.pano_ids)
        engine = retrieval.RetrievalEngine(holey, _CONFIG)
        centre = engine.log_likelihood(0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(
            float(centre), math.log(0.05) - math.log(8 * 2), places=5)


class FieldsIoTest(unittest.TestCase):
    def test_write_load_round_trip(self):
        frame = geo.RegionFrame(44.0, -71.0)
        fields = _grid_fields(
            np.random.default_rng(1).normal(
                size=(3, 4, 5, 12)).astype(np.float32),
            keyframe_idx=[7, 2, 9])
        lat, lon = frame.latlon_from_enu(fields.east_m, fields.north_m)
        with tempfile.TemporaryDirectory() as tmp:
            retrieval.write_fields(
                Path(tmp), fields.meta, lat, lon, fields.scores,
                fields.keyframe_idx, fields.pano_ids)
            loaded = retrieval.load_fields(Path(tmp), frame)
        np.testing.assert_allclose(loaded.east_m, fields.east_m, atol=1e-6)
        np.testing.assert_allclose(loaded.north_m, fields.north_m, atol=1e-6)
        # fp16 storage: scores agree to half precision.
        np.testing.assert_allclose(loaded.scores, fields.scores, atol=2e-3)
        measurements = retrieval.measurements_from_fields(loaded)
        self.assertEqual([m.keyframe_idx for m in measurements], [2, 7, 9])
        self.assertEqual(measurements[0].field_idx, 1)

    def test_duplicate_keyframe_refused(self):
        frame = geo.RegionFrame(44.0, -71.0)
        fields = _grid_fields(np.zeros((2, 3, 3, 4), dtype=np.float32),
                              keyframe_idx=[5, 5])
        lat, lon = frame.latlon_from_enu(fields.east_m, fields.north_m)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                retrieval.write_fields(
                    Path(tmp), fields.meta, lat, lon, fields.scores,
                    fields.keyframe_idx, fields.pano_ids)


def _fields_from_truth(data, *, spacing_m=100.0, half_extent_m=2500.0,
                       n_bins=12, bump_sigma_m=150.0, every=3,
                       mirror_offset=None):
    """Synthetic per-keyframe fields peaked at the truth pose (optionally
    also at a translated mirror, for ambiguity tests)."""
    n_side = int(2 * half_extent_m / spacing_m) + 1
    east = (np.arange(n_side) - (n_side - 1) / 2.0) * spacing_m
    ee, nn = np.meshgrid(east, east)
    ee, nn = ee.ravel(), nn.ravel()
    keyframes = list(range(0, len(data.truth), every))
    scores = np.zeros((len(keyframes), len(ee), n_bins), dtype=np.float32)
    for i, kf in enumerate(keyframes):
        truth = data.truth[kf]
        centres = [(truth.east_m, truth.north_m)]
        if mirror_offset is not None:
            centres.append((truth.east_m + mirror_offset[0],
                            truth.north_m + mirror_offset[1]))
        heading_bin = int(round(truth.course_world_cw_deg / (360.0 / n_bins))) \
            % n_bins
        for ce, cn in centres:
            bump = 8.0 * np.exp(
                -((ee - ce) ** 2 + (nn - cn) ** 2)
                / (2.0 * bump_sigma_m ** 2))
            scores[i, :, heading_bin] += bump.astype(np.float32)
    meta = retrieval.RetrievalFieldsMeta(
        schema_version=structs.SCHEMA_VERSION, dataset="synthetic",
        n_keyframes=len(keyframes), n_nodes=len(ee), n_heading_bins=n_bins,
        node_spacing_m=spacing_m, db_dir="synthetic",
        db_manifest_sha256="0" * 64, scorer="synthetic")
    fields = retrieval.ScoreFields(
        meta=meta, east_m=ee, north_m=nn, scores=scores,
        keyframe_idx=np.asarray(keyframes),
        pano_ids=[f"kf{kf}" for kf in keyframes])
    return fields, retrieval.measurements_from_fields(fields)


def _retrieval_filter_config(**overrides):
    defaults = dict(
        n_particles=40000, seed=3,
        init=structs.UniformBoxInit(-2500.0, 2500.0, -2500.0, 2500.0),
        position_roughening_m=15.0, heading_roughening_deg=1.0,
        checkpoint_every=50,
        proposal=structs.ProposalConfig(enabled=False),
        retrieval=structs.RetrievalConfig(
            temperature=1.0, outlier_epsilon=0.05))
    defaults.update(overrides)
    return structs.FilterConfig(**defaults)


class RetrievalFilterTest(unittest.TestCase):
    def _scenario(self):
        return scenario.generate(scenario.harbor_loop(
            max_visible_range_m=10000.0, keyframe_period_s=5.0))

    def test_uniform_box_converges_on_retrieval_alone(self):
        data = self._scenario()
        fields, measurements = _fields_from_truth(data)
        history = pf.run_filter(
            _retrieval_filter_config(), data.catalog, data.odometry, [], {},
            retrieval_fields=fields, retrieval_measurements=measurements)
        final_truth = data.truth[-1]
        mass = metrics.mass_within_radius(
            history.final_belief, final_truth.east_m, final_truth.north_m,
            300.0)
        self.assertGreater(mass, 0.5,
                           f"only {mass:.0%} of the posterior within 300 m")
        map_errors = metrics.map_position_errors_m(history.health,
                                                   data.truth)
        self.assertLess(float(map_errors[-1]), 300.0)
        # The health stream counts retrieval factors as measurements.
        self.assertEqual(history.health[0].n_measurements, 1)
        self.assertEqual(history.health[0].associations, [])

    def test_translated_ambiguity_is_preserved(self):
        """Two identical bumps a constant offset apart are indistinguishable
        to relative odometry; deleting either would be overconfidence."""
        data = self._scenario()
        # Truth spans east [-800, 800]; the mirror must stay inside the
        # field support (+-2500 m) for the whole leg or the filter would be
        # RIGHT to kill it.
        offset = (-1500.0, 0.0)
        fields, measurements = _fields_from_truth(data,
                                                  mirror_offset=offset)
        history = pf.run_filter(
            _retrieval_filter_config(seed=11), data.catalog, data.odometry,
            [], {}, retrieval_fields=fields,
            retrieval_measurements=measurements)
        final_truth = data.truth[-1]
        near_true = metrics.mass_within_radius(
            history.final_belief, final_truth.east_m, final_truth.north_m,
            400.0)
        near_mirror = metrics.mass_within_radius(
            history.final_belief, final_truth.east_m + offset[0],
            final_truth.north_m + offset[1], 400.0)
        self.assertGreater(near_true, 0.15)
        self.assertGreater(near_mirror, 0.15)

    def test_deterministic(self):
        data = self._scenario()
        fields, measurements = _fields_from_truth(data)
        config = _retrieval_filter_config(n_particles=5000)
        first = pf.run_filter(config, data.catalog, data.odometry, [], {},
                              retrieval_fields=fields,
                              retrieval_measurements=measurements)
        second = pf.run_filter(config, data.catalog, data.odometry, [], {},
                               retrieval_fields=fields,
                               retrieval_measurements=measurements)
        self.assertEqual(first.particle_history_sha256,
                         second.particle_history_sha256)

    def test_mixed_bearings_and_retrieval(self):
        data = self._scenario()
        fields, measurements = _fields_from_truth(data, every=5)
        history = pf.run_filter(
            _retrieval_filter_config(n_particles=5000),
            data.catalog, data.odometry, data.measurements, data.tables,
            retrieval_fields=fields, retrieval_measurements=measurements)
        counted = sum(h.n_measurements for h in history.health)
        self.assertEqual(counted,
                         len(data.measurements) + len(measurements))

    def test_validation(self):
        data = self._scenario()
        fields, measurements = _fields_from_truth(data, every=50)
        config = _retrieval_filter_config(n_particles=100)

        with self.assertRaises(ValueError):  # proposals are bearing-only
            pf.run_filter(
                _retrieval_filter_config(
                    n_particles=100,
                    proposal=structs.ProposalConfig(enabled=True)),
                data.catalog, data.odometry, [], {},
                retrieval_fields=fields,
                retrieval_measurements=measurements)
        with self.assertRaises(ValueError):  # calibration is not optional
            pf.run_filter(
                _retrieval_filter_config(n_particles=100, retrieval=None),
                data.catalog, data.odometry, [], {},
                retrieval_fields=fields,
                retrieval_measurements=measurements)
        with self.assertRaises(ValueError):  # fields and events together
            pf.run_filter(config, data.catalog, data.odometry, [], {},
                          retrieval_fields=fields)
        with self.assertRaises(ValueError):  # one field per keyframe
            pf.run_filter(config, data.catalog, data.odometry, [], {},
                          retrieval_fields=fields,
                          retrieval_measurements=measurements
                          + measurements[:1])


if __name__ == "__main__":
    unittest.main()
