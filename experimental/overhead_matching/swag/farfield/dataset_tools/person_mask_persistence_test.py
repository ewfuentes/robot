import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    person_mask_persistence as persistence,
)


def _constant_flow(shape, dx=0.0, dy=0.0):
    flow = np.zeros((*shape, 2), dtype=np.float32)
    flow[..., 0] = dx
    flow[..., 1] = dy
    return flow


def _translation_flows(shape, dx=0.0, dy=0.0):
    return persistence.GapFlows(
        previous_to_middle=_constant_flow(shape, dx, dy),
        middle_to_previous=_constant_flow(shape, -dx, -dy),
        next_to_middle=_constant_flow(shape, -dx, -dy),
        middle_to_next=_constant_flow(shape, dx, dy),
    )


def _rect(shape, x0, y0, x1, y1):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _circular_rect(shape, x0, y0, width, height):
    mask = np.zeros(shape, dtype=bool)
    rows = np.arange(y0, min(shape[0], y0 + height))
    columns = np.mod(np.arange(x0, x0 + width), shape[1])
    mask[np.ix_(rows, columns)] = True
    return mask


class PersonMaskPersistenceTest(unittest.TestCase):

    def setUp(self):
        self.shape = (48, 96)
        self.frames = tuple(
            np.zeros(self.shape, dtype=np.uint8) for _ in range(3))

    def bridge(self, previous, middle, following, **kwargs):
        return persistence.bridge_one_frame_gap(
            *self.frames,
            previous, middle, following,
            flows=kwargs.pop("flows", _translation_flows(self.shape)),
            **kwargs,
        )

    def test_translating_object_with_missing_middle_is_filled(self):
        previous = _rect(self.shape, 20, 12, 32, 30)
        expected = _rect(self.shape, 24, 12, 36, 30)
        following = _rect(self.shape, 28, 12, 40, 30)
        result = self.bridge(
            previous, np.zeros(self.shape, bool), following,
            flows=_translation_flows(self.shape, dx=4))
        np.testing.assert_array_equal(result.temporal_fill_mask, expected)
        np.testing.assert_array_equal(result.accepted_mask, expected)
        self.assertEqual(len(result.fills), 1)
        self.assertEqual(
            result.fills[0].mode, "synthesized_endpoint_consensus")
        self.assertEqual(
            result.fills[0].metrics["direct_proposal_coverage"], 0.0)
        self.assertFalse(result.review_required)

    def test_disagreeing_endpoints_are_reviewed_not_filled(self):
        previous = _rect(self.shape, 8, 10, 20, 28)
        following = _rect(self.shape, 68, 10, 80, 28)
        result = self.bridge(previous, np.zeros(self.shape, bool), following)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertTrue(result.review_required)
        self.assertTrue(all(
            "endpoint_disagreement" in flag.reasons
            for flag in result.review_flags))

    def test_subthreshold_endpoint_flow_remnant_is_not_review_noise(self):
        one_pixel = np.zeros(self.shape, dtype=bool)
        one_pixel[18, 30] = True
        result = self.bridge(
            one_pixel, np.zeros(self.shape, bool),
            np.zeros(self.shape, bool))
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertFalse(result.review_required)

    def test_subthreshold_middle_candidate_is_review_only(self):
        candidate = _rect(self.shape, 30, 18, 32, 20)
        result = self.bridge(
            np.zeros(self.shape, bool), np.zeros(self.shape, bool),
            np.zeros(self.shape, bool),
            middle_candidate_mask=candidate)

        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertFalse(np.any(result.accepted_mask))
        self.assertEqual(result.fills, ())
        self.assertEqual(len(result.review_flags), 1)
        flag = result.review_flags[0]
        self.assertEqual(flag.reasons, ("candidate_too_small",))
        np.testing.assert_array_equal(flag.mask, candidate)
        self.assertEqual(flag.metrics["pixel_count"], 4.0)
        self.assertEqual(flag.metrics["uncovered_pixel_count"], 4.0)
        self.assertEqual(
            flag.metrics["min_mask_pixels"],
            float(persistence.PersistenceConfig().min_mask_pixels))

    def test_subthreshold_candidate_flags_only_uncovered_pixels(self):
        direct = np.zeros(self.shape, dtype=bool)
        direct[18, 30] = True
        candidate = _rect(self.shape, 30, 18, 32, 20)
        expected_review = candidate & ~direct

        result = self.bridge(
            np.zeros(self.shape, bool), direct,
            np.zeros(self.shape, bool),
            middle_candidate_mask=candidate)

        np.testing.assert_array_equal(result.accepted_mask, direct)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(result.fills, ())
        self.assertEqual(len(result.review_flags), 1)
        flag = result.review_flags[0]
        self.assertEqual(flag.reasons, ("candidate_too_small",))
        np.testing.assert_array_equal(flag.mask, expected_review)
        self.assertEqual(flag.metrics["pixel_count"], 4.0)
        self.assertEqual(flag.metrics["uncovered_pixel_count"], 3.0)

    def test_matching_low_confidence_candidate_is_preferred(self):
        endpoint = _rect(self.shape, 24, 12, 36, 30)
        candidate = _rect(self.shape, 23, 11, 37, 31)
        result = self.bridge(
            endpoint, np.zeros(self.shape, bool), endpoint,
            middle_candidate_mask=candidate)
        np.testing.assert_array_equal(result.temporal_fill_mask, candidate)
        self.assertEqual(len(result.fills), 1)
        self.assertEqual(result.fills[0].mode, "promoted_middle_candidate")
        self.assertFalse(result.review_required)

    def test_existing_direct_detection_is_preserved_without_fill(self):
        endpoint = _rect(self.shape, 24, 12, 36, 30)
        direct = endpoint.copy()
        result = self.bridge(endpoint, direct, endpoint)
        np.testing.assert_array_equal(result.accepted_mask, direct)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(result.fills, ())
        self.assertFalse(result.review_required)

    def test_fragmented_direct_union_suppresses_boundary_only_fill(self):
        endpoint = _rect(self.shape, 20, 12, 40, 30)
        direct = np.zeros(self.shape, dtype=bool)
        direct[12:30, 20:25] = True
        direct[12:30, 26:31] = True
        direct[12:30, 32:37] = True
        result = self.bridge(endpoint, direct, endpoint)
        np.testing.assert_array_equal(result.accepted_mask, direct)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(result.fills, ())
        self.assertFalse(result.review_required)
        self.assertAlmostEqual(
            result.metrics["max_direct_proposal_coverage"], 0.75)
        self.assertEqual(
            result.metrics["direct_coverage_short_circuit_count"], 1.0)

    def test_partial_head_outside_well_covered_direct_mask_is_reviewed(self):
        body = _rect(self.shape, 24, 16, 36, 30)
        head = _rect(self.shape, 27, 12, 33, 16)
        endpoint = body | head

        result = self.bridge(endpoint, body, endpoint)

        np.testing.assert_array_equal(result.accepted_mask, body)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(result.fills, ())
        self.assertTrue(result.review_required)
        self.assertEqual(len(result.review_flags), 1)
        flag = result.review_flags[0]
        self.assertEqual(flag.reasons, ("uncovered_endpoint_consensus",))
        np.testing.assert_array_equal(flag.mask, head)
        self.assertAlmostEqual(
            flag.metrics["direct_proposal_coverage"], 0.875)
        self.assertEqual(
            flag.metrics["uncovered_consensus_pixel_count"], 24.0)
        self.assertGreaterEqual(
            flag.metrics["max_uncovered_consensus_depth_px"], 4.0)

    def test_direct_detection_short_circuits_bad_flow_and_appearance(self):
        endpoint = _rect(self.shape, 24, 12, 36, 30)
        direct = endpoint.copy()
        candidate = _rect(self.shape, 25, 13, 35, 29)
        previous_frame = np.zeros(self.shape, dtype=np.uint8)
        middle_frame = previous_frame.copy()
        middle_frame[direct] = 255
        next_frame = previous_frame.copy()
        bad_flows = persistence.GapFlows(
            previous_to_middle=_constant_flow(self.shape, 8),
            middle_to_previous=_constant_flow(self.shape, 0),
            next_to_middle=_constant_flow(self.shape, -8),
            middle_to_next=_constant_flow(self.shape, 0),
        )
        result = persistence.bridge_one_frame_gap(
            previous_frame, middle_frame, next_frame,
            endpoint, direct, endpoint,
            middle_candidate_mask=candidate,
            flows=bad_flows,
        )
        np.testing.assert_array_equal(result.accepted_mask, direct)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(result.fills, ())
        self.assertFalse(result.review_required)

    def test_horizontal_seam_crossing_stays_a_narrow_mask(self):
        previous = _circular_rect(self.shape, 90, 12, 8, 18)
        expected = _circular_rect(self.shape, 94, 12, 8, 18)
        following = _circular_rect(self.shape, 98, 12, 8, 18)
        result = self.bridge(
            previous, np.zeros(self.shape, bool), following,
            flows=_translation_flows(self.shape, dx=4))
        np.testing.assert_array_equal(result.temporal_fill_mask, expected)
        self.assertEqual(np.count_nonzero(result.temporal_fill_mask), 8 * 18)
        occupied_columns = np.flatnonzero(np.any(
            result.temporal_fill_mask, axis=0))
        self.assertEqual(len(occupied_columns), 8)
        self.assertFalse(result.review_required)

    def test_dense_flow_estimator_tracks_a_real_seam_crossing(self):
        shape = (64, 128)
        random = np.random.default_rng(123)
        previous_frame = random.integers(0, 256, shape, dtype=np.uint8)
        middle_frame = np.roll(previous_frame, 3, axis=1)
        next_frame = np.roll(previous_frame, 6, axis=1)
        previous = _circular_rect(shape, 118, 20, 8, 24)
        expected = np.roll(previous, 3, axis=1)
        following = np.roll(previous, 6, axis=1)
        result = persistence.bridge_one_frame_gap(
            previous_frame, middle_frame, next_frame,
            previous, np.zeros(shape, bool), following)
        intersection = np.count_nonzero(
            result.temporal_fill_mask & expected)
        union = np.count_nonzero(result.temporal_fill_mask | expected)
        self.assertGreater(intersection / union, 0.95)
        self.assertLessEqual(np.count_nonzero(
            np.any(result.temporal_fill_mask, axis=0)), 9)
        self.assertFalse(result.review_required)

    def test_top_and_bottom_are_not_connected_through_a_polar_wrap(self):
        top = _rect(self.shape, 30, 0, 42, 5)
        bottom = _rect(self.shape, 30, 43, 42, 48)
        result = self.bridge(top, np.zeros(self.shape, bool), bottom)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertEqual(len(result.review_flags), 2)
        self.assertTrue(all(
            "endpoint_disagreement" in flag.reasons
            for flag in result.review_flags))

    def test_forward_backward_cycle_failure_is_reviewed(self):
        endpoint = _rect(self.shape, 24, 12, 36, 30)
        bad_flows = persistence.GapFlows(
            previous_to_middle=_constant_flow(self.shape, 8),
            middle_to_previous=_constant_flow(self.shape, 0),
            next_to_middle=_constant_flow(self.shape, -8),
            middle_to_next=_constant_flow(self.shape, 0),
        )
        result = self.bridge(
            endpoint, np.zeros(self.shape, bool), endpoint,
            flows=bad_flows)
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertTrue(any(
            "previous_flow_cycle_invalid" in flag.reasons
            or "next_flow_cycle_invalid" in flag.reasons
            for flag in result.review_flags))

    def test_scene_cut_rejects_even_geometrically_matching_masks(self):
        previous = np.zeros(self.shape, dtype=np.uint8)
        middle = np.full(self.shape, 255, dtype=np.uint8)
        following = np.zeros(self.shape, dtype=np.uint8)
        endpoint = _rect(self.shape, 24, 12, 36, 30)
        result = persistence.bridge_one_frame_gap(
            previous, middle, following,
            endpoint, np.zeros(self.shape, bool), endpoint,
            flows=_translation_flows(self.shape),
        )
        self.assertFalse(np.any(result.temporal_fill_mask))
        self.assertTrue(result.review_required)
        self.assertTrue(any(
            reason.startswith("scene_cut_")
            for reason in result.review_flags[0].reasons))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            persistence.bridge_one_frame_gap(
                self.frames[0], self.frames[1][:-1], self.frames[2],
                np.zeros(self.shape, bool), np.zeros(self.shape, bool),
                np.zeros(self.shape, bool))


if __name__ == "__main__":
    unittest.main()
