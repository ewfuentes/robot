import math
import unittest
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    grid_filter,
    structs,
)


class MotionPlanningTest(unittest.TestCase):
    def test_left_motion_uses_post_turn_heading(self):
        belief = grid_filter.GridBelief(
            grid_filter.Grid(0.0, 50.0, 0.0, 50.0, 10.0), 4, "cpu")
        belief.belief.zero_()
        belief.belief[0, 2, 2] = 1.0

        belief.motion(
            structs.OdometryDelta(
                1, 0.0, 10.0, math.radians(90.0), 0.0, 0.0),
            1.0, 0.0, 0.0)

        expected = torch.zeros_like(belief.belief)
        expected[1, 3, 2] = 1.0
        torch.testing.assert_close(belief.belief, expected)

    def test_turn_carries_subcell_translation_into_new_heading(self):
        belief = grid_filter.GridBelief(
            grid_filter.Grid(0.0, 50.0, 0.0, 50.0, 10.0), 4, "cpu")
        belief.belief.zero_()
        belief.belief[0, 2, 2] = 1.0

        belief.motion(
            structs.OdometryDelta(1, 4.0, 0.0, 0.0, 0.0, 0.0),
            1.0, 0.0, 0.0)
        belief.motion(
            structs.OdometryDelta(
                2, 4.0, 0.0, math.radians(60.0), 0.0, 0.0),
            1.0, 0.0, 0.0)

        expected = torch.zeros_like(belief.belief)
        expected[1, 3, 2] = 1.0
        torch.testing.assert_close(belief.belief, expected)

    def test_convolution_materializes_residual(self):
        belief = grid_filter.GridBelief(
            grid_filter.Grid(0.0, 50.0, 0.0, 50.0, 10.0), 4, "cpu")
        belief.belief.zero_()
        belief.belief[0, 2, 2] = 1.0
        belief.motion(
            structs.OdometryDelta(1, 4.0, 0.0, 0.0, 0.0, 0.0),
            1.0, 0.0, 0.0)

        plan = belief.plan_motion(
            structs.OdometryDelta(2, 0.0, 0.0, 0.0, 0.0, 0.6),
            1.0, 0.0, 0.0)

        self.assertGreater(len(plan.heading_offsets), 1)
        self.assertEqual(
            plan.weighted_cell_shifts[0],
            ((0, 0, 0.6), (1, 0, 0.4)))
        spatial = torch.zeros_like(belief.belief)
        spatial[0, 2, 2] = 0.6
        spatial[0, 3, 2] = 0.4
        expected = torch.zeros_like(spatial)
        for offset, weight in zip(
                plan.heading_offsets, plan.heading_weights, strict=True):
            expected += weight * torch.roll(spatial, offset, dims=0)
        torch.testing.assert_close(
            grid_filter.apply_motion(belief.belief, plan), expected)
        self.assertFalse(belief.pending_de.any())
        self.assertFalse(belief.pending_dn.any())


class SmootherAdjointTest(unittest.TestCase):
    def test_transposed_motion_is_the_adjoint(self):
        torch.manual_seed(0)
        belief = grid_filter.GridBelief(
            grid_filter.Grid(0.0, 80.0, 0.0, 80.0, 10.0), 6, "cpu")
        # a noisy turn (weighted shifts + heading kernel + diffusion), then a
        # clean straight step (integer shifts only): every operator covered
        turn = belief.plan_motion(
            structs.OdometryDelta(
                1, 14.0, 3.0, math.radians(50.0), 4.0, 0.4),
            1.0, math.radians(2.0), 3.0)
        straight = belief.plan_motion(
            structs.OdometryDelta(2, 10.0, 0.0, 0.0, 0.0, 0.0),
            1.0, 0.0, 0.0)
        self.assertTrue(turn.weighted_cell_shifts and turn.heading_offsets
                        and turn.diffusion_offsets)
        self.assertTrue(straight.cell_shifts and not straight.heading_offsets)
        for plan in (turn, straight):
            x = torch.rand_like(belief.belief)
            y = torch.rand_like(belief.belief)
            lhs = (grid_filter.apply_motion(x, plan) * y).sum()
            rhs = (x * grid_filter.apply_motion_transposed(y, plan)).sum()
            torch.testing.assert_close(lhs, rhs)


class CausalReplayTest(unittest.TestCase):
    def test_delayed_factors_change_only_current_and_future_scores(self):
        first = structs.TrackletMeasurement("a", 0, 0.0, 1.0)
        second = structs.TrackletMeasurement("b", 1, 0.0, 1.0)
        releases = [
            SimpleNamespace(
                release_keyframe_idx=2, tracklet_id="a",
                measurements=(first,)),
            SimpleNamespace(
                release_keyframe_idx=3, tracklet_id="b",
                measurements=(second,)),
        ]
        factors = {
            "a": torch.tensor([2.0, 1.0]),
            "b": torch.tensor([1.0, 3.0]),
        }

        def apply(message, _keyframe, measurements):
            for measurement in measurements:
                message = message * factors[measurement.tracklet_id]
            return message

        final, scores, stats = grid_filter.causal_replay(
            torch.tensor([0.5, 0.5]),
            [None, grid_filter.MotionPlan(), grid_filter.MotionPlan(),
             grid_filter.MotionPlan()],
            releases, apply, lambda message, _kf: message.clone(), 2)

        torch.testing.assert_close(scores[0], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(scores[1], torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(
            scores[2], torch.tensor([2.0 / 3.0, 1.0 / 3.0]))
        torch.testing.assert_close(final, torch.tensor([0.4, 0.6]))
        torch.testing.assert_close(scores[3], final)
        self.assertEqual(stats["n_track_releases"], 2)
        self.assertEqual(stats["n_replay_keyframe_steps"], 6)
        self.assertEqual(stats["max_rollback_keyframes"], 3)

    def test_eof_corelease_matches_eager_final_without_revising_past(self):
        measurements = [
            structs.TrackletMeasurement("a", 0, 0.0, 1.0),
            structs.TrackletMeasurement("b", 2, 0.0, 1.0),
            structs.TrackletMeasurement("c", 3, 0.0, 1.0),
        ]
        releases = [
            SimpleNamespace(
                release_keyframe_idx=2, tracklet_id="a",
                measurements=(measurements[0],)),
            SimpleNamespace(
                release_keyframe_idx=4, tracklet_id="b",
                measurements=(measurements[1],)),
            SimpleNamespace(
                release_keyframe_idx=4, tracklet_id="c",
                measurements=(measurements[2],)),
        ]
        factors = {
            "a": torch.tensor([2.0, 1.0]),
            "b": torch.tensor([1.0, 3.0]),
            "c": torch.tensor([4.0, 1.0]),
        }

        def apply(message, _keyframe, selected):
            for measurement in selected:
                message = message * factors[measurement.tracklet_id]
            return message

        initial = torch.tensor([0.5, 0.5])
        plans = [None] + [grid_filter.MotionPlan()] * 4
        final, scores, _ = grid_filter.causal_replay(
            initial, plans, releases, apply,
            lambda message, _kf: message.clone(), 2)

        eager = initial
        by_keyframe = {item.anchor_keyframe_idx: [item]
                       for item in measurements}
        for keyframe in range(len(plans)):
            eager = grid_filter._normalized(  # noqa: SLF001
                apply(eager, keyframe, by_keyframe.get(keyframe, ())))

        torch.testing.assert_close(final, eager)
        torch.testing.assert_close(scores[0], initial)
        torch.testing.assert_close(scores[1], initial)
        torch.testing.assert_close(
            scores[2], torch.tensor([2.0 / 3.0, 1.0 / 3.0]))
        torch.testing.assert_close(scores[3], scores[2])


class ReleaseModeHandoffTest(unittest.TestCase):
    def test_online_map_uses_position_marginal_and_conditional_heading(self):
        grid = grid_filter.Grid(0.0, 20.0, 0.0, 10.0, 10.0)
        message = torch.tensor([
            [[0.15, 0.30]],
            [[0.16, 0.04]],
            [[0.20, 0.03]],
            [[0.09, 0.03]],
        ])

        state = grid_filter._online_map_state(  # noqa: SLF001
            message, message.sum(dim=0).reshape(-1), grid)

        self.assertEqual(state, {
            "east_m": 5.0,
            "north_m": 5.0,
            "heading_world_cw_deg": 180.0,
        })
        self.assertEqual(
            grid_filter._map_state(message, grid)["east_m"],  # noqa: SLF001
            15.0)

    def test_snapshot_is_release_only_and_sorts_corelease_ids(self):
        grid = grid_filter.Grid(0.0, 20.0, 0.0, 10.0, 10.0)
        message = torch.tensor([[[0.25, 0.75]]])

        self.assertIsNone(grid_filter._release_top_mode_snapshot(  # noqa: SLF001
            message, grid, 1, 0.0, 0.0, 3, ()))
        snapshot = grid_filter._release_top_mode_snapshot(  # noqa: SLF001
            message, grid, 1, 0.0, 0.0, 3, ("z", "a"))

        self.assertEqual(snapshot["keyframe_idx"], 3)
        self.assertEqual(snapshot["released_tracklet_ids"], ["a", "z"])
        self.assertEqual(snapshot["returned"], 1)
        self.assertEqual(snapshot["modes"][0]["east_m"], 15.0)
        self.assertEqual(
            grid_filter._map_state(message, grid),  # noqa: SLF001
            {"east_m": 15.0, "north_m": 5.0,
             "heading_world_cw_deg": 0.0})


if __name__ == "__main__":
    unittest.main()
