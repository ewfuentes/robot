"""Latency model of the histogram path evaluation: slow streams are captured at
every pano, carried forward by odometry, and integrated once their delay elapsed."""

import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

from common.gps import web_mercator
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    ObservationLogLikelihoodAggregator,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
    HistogramFilterConfig,
    elapsed_seconds,
    run_histogram_filter_on_path,
    step_distances_m,
)

DEVICE = torch.device("cpu")


class _TwoStream(ObservationLogLikelihoodAggregator):
    def __init__(self, image: dict, landmark: dict):
        self.image, self.landmark = image, landmark

    def __call__(self, pano_id):
        return self.image[pano_id] + self.landmark[pano_id]

    def streams(self, pano_id):
        return {"image": self.image[pano_id], "landmark": self.landmark[pano_id]}


class _World:
    """A 20x20 grid of 50 px cells, one satellite patch per cell, and a straight
    path heading east one cell per step. The landmark stream points at the
    true cell (log-likelihood 0 there, -4 elsewhere); the image stream is flat."""

    def __init__(self, path_len=8):
        self.grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.70, max_lat=37.7035, min_lon=-122.42, max_lon=-122.4165,
            zoom_level=20, cell_size_px=50.0)
        centers_px = self.grid_spec.get_all_cell_centers_px(DEVICE)
        self.mapping = build_cell_to_patch_mapping(self.grid_spec, centers_px, 26.0, DEVICE)
        self.num_patches = centers_px.shape[0]
        n_cols = self.grid_spec.num_cols
        row, col0 = self.grid_spec.num_rows // 2, 3
        self.true_cells = [(row, col0 + i) for i in range(path_len)]
        # Cell centers in float64 so each step is an exact whole-cell shift
        # (GridSpec's own float32 lat/lon would quantize the steps by ~7 px).
        gs = self.grid_spec
        rows = torch.tensor([r for r, _ in self.true_cells], dtype=torch.float64)
        cols = torch.tensor([c for _, c in self.true_cells], dtype=torch.float64)
        lat, lon = web_mercator.pixel_coords_to_latlon(
            gs.origin_row_px + (rows + 0.5) * gs.cell_size_px,
            gs.origin_col_px + (cols + 0.5) * gs.cell_size_px, gs.zoom_level)
        self.true_latlons = torch.stack([lat, lon], dim=-1)
        self.motion_deltas = self.true_latlons[1:] - self.true_latlons[:-1]
        self.path = [f"p{i}" for i in range(path_len)]
        image = {pid: torch.zeros(self.num_patches) for pid in self.path}
        landmark = {}
        for pid, (r, c) in zip(self.path, self.true_cells):
            ll = torch.full((self.num_patches,), -4.0)
            ll[r * n_cols + c] = 0.0
            landmark[pid] = ll
        self.agg = _TwoStream(image, landmark)
        self.image_only = _TwoStream(image, {pid: torch.zeros(self.num_patches) for pid in self.path})
        self.step_dist = step_distances_m(self.true_latlons)

    def run(self, agg, **config_kw):
        config = HistogramFilterConfig(motion_noise_frac=0.0, **config_kw)
        belief = HistogramBelief.from_uniform(self.grid_spec, DEVICE)
        elapsed = None
        if config.slow_streams != "none" and config.slow_stream_delay_s > 0:
            elapsed = elapsed_seconds(self.step_dist, config.speed_m_s)
        return run_histogram_filter_on_path(
            belief=belief, motion_deltas=self.motion_deltas, path_pano_ids=self.path,
            log_likelihood_aggregator=agg, mapping=self.mapping, config=config,
            true_latlons=self.true_latlons, convergence_radii=[50],
            elapsed_s=elapsed,
        )


class TestLatencyModel(unittest.TestCase):
    def setUp(self):
        self.w = _World()

    def test_no_slow_stream_reproduces_the_plain_loop(self):
        w = self.w
        result = w.run(w.agg)
        belief = HistogramBelief.from_uniform(w.grid_spec, DEVICE)
        for i, pid in enumerate(w.path):
            belief.apply_observation(w.agg(pid), w.mapping)
            if i < len(w.path) - 1:
                belief.apply_motion(w.motion_deltas[i], 0.0)
        self.assertTrue(torch.equal(result.final_belief.get_log_belief(), belief.get_log_belief()))

    def test_zero_delay_slow_stream_matches_immediate_integration(self):
        w = self.w
        ref = w.run(w.agg)
        out = w.run(w.agg, slow_streams="landmark", slow_stream_delay_s=0.0)
        self.assertTrue(torch.allclose(out.final_belief.get_log_belief(),
                                       ref.final_belief.get_log_belief(), atol=1e-5))

    def test_delay_longer_than_path_is_image_only(self):
        w = self.w
        ref = w.run(w.image_only)
        out = w.run(w.agg, slow_streams="landmark", slow_stream_delay_s=1e9, speed_m_s=2.0)
        self.assertTrue(torch.allclose(out.final_belief.get_log_belief(),
                                       ref.final_belief.get_log_belief(), atol=1e-6))

    def test_full_dropout_is_image_only(self):
        w = self.w
        ref = w.run(w.image_only)
        config = HistogramFilterConfig(motion_noise_frac=0.0, slow_streams="landmark",
                                       slow_stream_dropout=1.0)
        out = run_histogram_filter_on_path(
            belief=HistogramBelief.from_uniform(w.grid_spec, DEVICE),
            motion_deltas=w.motion_deltas, path_pano_ids=w.path,
            log_likelihood_aggregator=w.agg, mapping=w.mapping, config=config,
            true_latlons=w.true_latlons, convergence_radii=[50],
            slow_drop=torch.ones(len(w.path), dtype=torch.bool))
        self.assertTrue(torch.allclose(out.final_belief.get_log_belief(),
                                       ref.final_belief.get_log_belief(), atol=1e-6))

    def test_lag_only_changes_which_observations_are_in_the_belief(self):
        """Integrating pano k's landmark evidence two steps late must leave the
        belief at the end identical to a no-lag run that simply never saw the
        observations still in flight at the end (the last two panos). Exact
        here because the steps are whole cells; with fractional shifts the
        bilinear resampling of a product is not the product of resamplings,
        and carrying the in-flight grids through the same per-step operator
        is the closest the histogram filter can get."""
        w = self.w
        step_m = float(w.step_dist[0])
        for noise in (0.0, 0.141):
            lagged = HistogramFilterConfig(motion_noise_frac=noise, slow_streams="landmark",
                                           slow_stream_delay_s=1.5 * step_m, speed_m_s=1.0)
            elapsed = elapsed_seconds(w.step_dist, 1.0)
            out = run_histogram_filter_on_path(
                belief=HistogramBelief.from_uniform(w.grid_spec, DEVICE),
                motion_deltas=w.motion_deltas, path_pano_ids=w.path,
                log_likelihood_aggregator=w.agg, mapping=w.mapping, config=lagged,
                elapsed_s=elapsed)
            landmark = dict(w.agg.landmark)
            for pid in w.path[-2:]:
                landmark[pid] = torch.zeros(w.num_patches)
            ref = run_histogram_filter_on_path(
                belief=HistogramBelief.from_uniform(w.grid_spec, DEVICE),
                motion_deltas=w.motion_deltas, path_pano_ids=w.path,
                log_likelihood_aggregator=_TwoStream(w.agg.image, landmark),
                mapping=w.mapping, config=HistogramFilterConfig(motion_noise_frac=noise))
            a, b = out.final_belief.get_belief(), ref.final_belief.get_belief()
            self.assertTrue(torch.allclose(a, b, atol=1e-4),  # bilinear resampling at integer positions
                            f"noise={noise} max|diff|={(a - b).abs().max().item():.3e} "
                            f"peak={a.max().item():.3f} argmax {a.argmax().item()} vs {b.argmax().item()}")

    def test_lagged_observations_are_carried_forward(self):
        """With a two-step lag the landmark evidence about pano k lands when the
        robot is at k+2; shifted by the odometry it must still support the true
        cell, so the final mode is the true final position, not two cells back."""
        w = self.w
        step_m = float(w.step_dist[0])
        out = w.run(w.agg, slow_streams="landmark", slow_stream_delay_s=1.5 * step_m,
                    speed_m_s=1.0)
        err_m = 6_371_000.0 * torch.deg2rad(out.mode_history[-1] - w.true_latlons[-1]).norm()
        self.assertLess(err_m.item(), 0.5 * step_m)
        # Mass within 50 m should be high at the end: the lagged evidence was used.
        self.assertGreater(out.prob_mass_by_radius[50][-1].item(), 0.9)
        # And unlike zero delay, the first observation lands later: at step 0
        # only the (flat) image stream has been integrated.
        self.assertLess(out.prob_mass_by_radius[50][1].item(), 0.1)


class TestElapsedSeconds(unittest.TestCase):
    def test_distance_over_speed(self):
        dist = torch.tensor([10.0, 30.0])
        self.assertEqual(elapsed_seconds(dist, 2.0).tolist(), [0.0, 5.0, 20.0])
        for bad in (None, 0.0, -1.0):
            with self.assertRaises(ValueError):
                elapsed_seconds(dist, bad)


if __name__ == "__main__":
    unittest.main()
