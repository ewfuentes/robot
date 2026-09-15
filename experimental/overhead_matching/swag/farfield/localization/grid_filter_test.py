import math
import unittest
from types import SimpleNamespace

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    grid_filter,
    structs,
)


class LikelihoodCacheTest(unittest.TestCase):
    def test_loci_log_likelihood_is_applied_directly(self):
        prior = torch.tensor([0.25, 0.75])
        log_likelihood = torch.log(torch.tensor([4.0, 1.0]))
        actual = grid_filter._apply_log_likelihood_factor(  # noqa: SLF001
            prior, log_likelihood)
        torch.testing.assert_close(actual, torch.tensor([4.0 / 7.0, 3.0 / 7.0]))

    def test_consumer_misses_preserve_producer_tail(self):
        cache = grid_filter.LikelihoodCache(8, "cpu")
        for key in range(5):
            cache.get(key, lambda: torch.tensor([float(key)]))
        for key in range(5):
            actual = cache.get(key, lambda: torch.tensor([float(key)]), admit=False)
            self.assertEqual(actual.item(), key)
        self.assertEqual((cache.hits, cache.bytes, cache.evictions), (2, 8, 3))

    def test_single_frame_key_ignores_only_episode_frame(self):
        def release(frame, bearing=45):
            return SimpleNamespace(measurements=[structs.TrackletMeasurement("raw", frame, bearing, 2)])
        key = grid_filter.single_frame_cache_key
        self.assertEqual(key(release(20), 20, "table", "grid"),
                         key(release(0), 0, "table", "grid"))
        self.assertIsNone(key(release(0), 1, "table", "grid"))
        self.assertIsNone(key(SimpleNamespace(measurements=release(0).measurements * 2),
                              0, "table", "grid"))
        baseline = key(release(0), 0, "table", "grid")
        for actual in (key(release(0, 46), 0, "table", "grid"),
                       key(release(0), 0, "changed table", "grid"),
                       key(release(0), 0, "table", "changed grid")):
            self.assertNotEqual(baseline, actual)

    def test_lru_eviction_and_byte_accounting(self):
        cache = grid_filter.LikelihoodCache(24, "cpu")
        values = {key: torch.full((size,), float(index))
                  for index, (key, size) in enumerate(
                      [("a", 2), ("b", 4), ("c", 3), ("large", 7)])}
        for key in ("a", "b", "a", "c"):
            self.assertTrue(torch.equal(cache.get(key, lambda: values[key]), values[key]))
        self.assertEqual(list(cache.store), ["a", "c"])
        self.assertEqual((cache.bytes, cache.hits, cache.misses, cache.evictions),
                         (20, 1, 3, 1))
        cache.get("large", lambda: values["large"])
        self.assertEqual(list(cache.store), ["a", "c"])
        self.assertEqual((cache.bytes, cache.skipped), (20, 1))
        cache.get("b", lambda: values["b"])
        self.assertEqual(list(cache.store), ["b"])
        self.assertEqual((cache.bytes, cache.evictions), (16, 3))

    def test_disabled_and_invalid_budgets(self):
        cache = grid_filter.LikelihoodCache(0, "cpu")
        for _ in range(2):
            cache.get("a", lambda: torch.ones(4))
        self.assertEqual((len(cache.store), cache.bytes, cache.misses, cache.skipped),
                         (0, 0, 2, 2))
        for bad in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                grid_filter.LikelihoodCache(bad, "cpu")

    def test_eviction_preserves_forward_and_backward_messages_exactly(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            belief = grid_filter.GridBelief(
                grid_filter.Grid(0, 50, 0, 40, 10), 6, device)
            tensor_bytes = belief.belief.numel() * belief.belief.element_size()
            def rollout(budget):
                cache = grid_filter.LikelihoodCache(budget, device)
                message = belief.belief.clone()
                messages = []
                def compute(key):
                    return belief.track_likelihood(
                        key * 0.1, 0.05,
                        torch.tensor([100., 200.], device=device),
                        torch.tensor([200., 100.], device=device),
                        torch.tensor([0.4, 0.5], device=device), 25., 0.2, 0.1)
                for keys in (range(5), range(4, -1, -1), [1, 2, 1, 3, 4]):
                    for key in keys:
                        factor = cache.get(key, lambda: compute(key))
                        message = grid_filter._apply_likelihood_factors(message, [factor])
                        messages.append(message.clone())
                        self.assertLessEqual(cache.bytes, budget)
                return messages, cache
            reference, _ = rollout(0)
            for budget in (tensor_bytes * 2, tensor_bytes * 5):
                actual, cache = rollout(budget)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference, actual)))
                if budget == tensor_bytes * 2:
                    self.assertGreater(cache.evictions, 0)


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
    def test_forward_backward_product_preserves_tiny_shared_mass(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                alpha = torch.tensor([1.0, 0.0, 1e-30, 2e-30], device=device)
                beta = torch.tensor([0.0, 1.0, 1e-30, 1e-30], device=device)
                self.assertEqual(float((alpha * beta).sum()), 0.0)
                expected = alpha.double() * beta.double()
                expected /= expected.sum()
                actual = grid_filter._apply_likelihood_factors(alpha, [beta])
                torch.testing.assert_close(actual, expected.float(),
                                           rtol=2e-5, atol=0.0)
                # Exact disjoint support is still invalid, not filled in.
                beta[2:] = 0.0
                with self.assertRaises(ValueError):
                    grid_filter._apply_likelihood_factors(alpha, [beta])

    def test_coreleased_factors_do_not_overflow_or_underflow(self):
        for scale in (1e28, 1e-28):
            prior = torch.tensor([0.25, 0.75, 0.0])
            factors = [torch.tensor([2.0, 1.0, 3.0]) * scale,
                       torch.tensor([5.0, 4.0, 2.0]) * scale]
            naive = prior * factors[0] * factors[1]
            self.assertTrue(not torch.isfinite(naive).all() or naive.sum() == 0)
            expected = prior.double() * factors[0].double() * factors[1].double()
            expected /= expected.sum()
            actual = grid_filter._apply_likelihood_factors(prior, factors)
            torch.testing.assert_close(actual, expected.float(), rtol=2e-5, atol=1e-7)
            self.assertEqual(float(actual[2]), 0.0)

    def test_stable_product_includes_incoming_message_and_rejects_invalid_mass(self):
        prior = torch.tensor([1e-30, 1.0])
        factors = [torch.tensor([1e30, 1.0])] * 2
        actual = grid_filter._apply_likelihood_factors(prior, factors)
        torch.testing.assert_close(actual, torch.tensor([1.0, 1e-30]),
                                   rtol=2e-5, atol=0.0)
        torch.testing.assert_close(
            grid_filter._apply_likelihood_factors(torch.tensor([2.0, 3.0]), []),
            torch.tensor([0.4, 0.6]))
        for bad in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                grid_filter._apply_likelihood_factors(torch.ones(2),
                                                       [torch.full((2,), bad)])

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


class DetectionAuditReplayTest(unittest.TestCase):
    def test_replacement_replays_diffusion_and_preserves_every_prefix(self):
        release_type = grid_filter.release_schedule_lib.TrackRelease

        def release(name, arrival, anchors):
            return release_type(arrival, name, tuple(
                structs.TrackletMeasurement(name, k, 0., 1.) for k in anchors), None)

        raw = [release("d0", 0, [0]), release("d1", 1, [1]), release("d2", 2, [2])]
        audits = [release("a", 3, [0, 1]), release("b", 4, [1, 2])]
        removals = {"a": ["d0", "d1"], "b": ["d1", "d2"]}
        factors = {k: torch.tensor(v).reshape(1, 1, 3) for k, v in {
            "d0": [9., 1., 2.], "d1": [2., 1., 7.], "d2": [1., 8., 2.],
            "a": [1., 2., 6.], "b": [4., 1., 2.]}.items()}
        initial = torch.tensor([[[.2, .5, .3]]])
        plans = [None] + [grid_filter.MotionPlan(
            weighted_cell_shifts=(((0, 0, .8), (0, 1, .2)),))] * 5

        def brute(stop, audit_list):
            available = [r for r in audit_list if r.release_keyframe_idx <= stop]
            removed = {k for r in available for k in removals[r.tracklet_id]}
            active = available + [r for r in raw if r.tracklet_id not in removed]
            message = initial.clone()
            for step in range(stop + 1):
                if step:
                    message = grid_filter.apply_motion(message, plans[step])
                for r in sorted(active, key=lambda r: r.tracklet_id):
                    if r.release_keyframe_idx == step:
                        message = grid_filter._normalized(message * factors[r.tracklet_id])
                message = grid_filter._normalized(message)
            return message

        for stride in (1, 2, 8):
            for stop in range(1, len(plans)):
                _, scores, stats = grid_filter.detection_audit_replay(
                    initial, plans[:stop + 1], raw, audits, removals,
                    lambda r, _: factors[r.tracklet_id], lambda m, _: m.clone(), stride)
                for k, score in enumerate(scores):
                    torch.testing.assert_close(score, brute(k, audits))
                if stop >= 4:
                    self.assertEqual(stats["removed_detection_factors"], 3)
                    self.assertEqual(stats["already_removed_shared_claims"], 1)
                    self.assertEqual(stats["remaining_detection_factors"], 0)
        _, scores, _ = grid_filter.detection_audit_replay(
            initial, plans, raw, [], {}, lambda r, _: factors[r.tracklet_id],
            lambda m, _: m.clone(), 2)
        for k, score in enumerate(scores):
            torch.testing.assert_close(score, brute(k, []))
        with self.assertRaises(ValueError):
            grid_filter.detection_audit_replay(
                initial, plans, raw, [release("a", 0, [0])], {"a": ["d2"]},
                lambda r, _: factors[r.tracklet_id], lambda m, _: m, 2)

    def test_support_mapping_excludes_bystanders_and_records_missing(self):
        tracks = [{"track_id": 1, "birth_obs_id": "o0", "records": [
            {"supports": [{"class": "continue_clean", "obs_id": "o1"},
                          {"class": "none", "obs_id": "bystander"},
                          {"class": "weak", "obs_id": "missing"}]}]}]
        detections = [{"track_id": i, "birth_obs_id": o, "status": "closed",
                       "birth_keyframe": i, "last_keyframe": i}
                      for i, o in enumerate(("o0", "o1", "bystander"))]
        mapping, missing = grid_filter.detection_audit.member_mapping(
            tracks, detections, ["tracked#T1"], [f"raw#T{i}" for i in range(3)])
        self.assertEqual(mapping, {"tracked#T1": ["raw#T0", "raw#T1"]})
        self.assertEqual(missing, {"tracked#T1": ["missing"]})
        with self.assertRaises(ValueError):
            grid_filter.detection_audit.member_mapping(tracks, detections * 2, [], [])


class OnlineMapStateTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
