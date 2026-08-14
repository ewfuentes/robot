import unittest

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    bearing_matcher as bm,
    harbor_catalog as hc,
)


def entry(landmark_id, east, north, tags=None, source="osm"):
    return hc.CatalogEntry(
        landmark_id=landmark_id, source=source, east_m=east, north_m=north,
        position_sigma_m=10.0, tags=tags or {})


def obs(keyframe, east, north, bearing, half=0.0):
    return bm.Observation(anchor_keyframe_idx=keyframe, east_m=east,
                          north_m=north, bearing_world_deg=bearing,
                          half_width_deg=half)




class CompatibilityTableTest(unittest.TestCase):
    """The format contract with bearing_only_localization.structs."""

    def test_field_names_and_types_match_the_struct(self):
        table = bm.to_compatibility_table("LT1", {"osm:node:1": 2.0},
                                          "tag_rule_v1")
        self.assertEqual(set(table), {
            "tracklet_id", "matcher_version", "entries", "default_log_lr",
            "clip_lo", "clip_hi", "status"})
        self.assertEqual(set(table["entries"][0]), {"landmark_id", "log_lr"})
        self.assertIsInstance(table["entries"][0]["log_lr"], float)
        self.assertIsInstance(table["default_log_lr"], float)
        self.assertEqual(table["status"], "fast")

    def test_scores_are_clipped_both_ways(self):
        table = bm.to_compatibility_table(
            "LT1", {"hi": 100.0, "lo": -100.0}, "v", clip=4.0)
        by_id = {e["landmark_id"]: e["log_lr"] for e in table["entries"]}
        self.assertEqual(by_id["hi"], 4.0)
        self.assertEqual(by_id["lo"], -4.0)
        self.assertEqual(table["clip_lo"], -4.0)
        self.assertEqual(table["clip_hi"], 4.0)

    def test_affine_transform_applied_before_clipping(self):
        table = bm.to_compatibility_table(
            "LT1", {"x": 2.0}, "v", scale=1.5, offset=-1.0, clip=10.0)
        self.assertAlmostEqual(table["entries"][0]["log_lr"], 2.0)

    def test_entries_matching_the_default_are_omitted(self):
        # The struct's contract: absent landmarks score default_log_lr, so
        # emitting them would be redundant.
        table = bm.to_compatibility_table(
            "LT1", {"same": -2.0, "other": 1.0}, "v", default_log_lr=-2.0)
        self.assertEqual([e["landmark_id"] for e in table["entries"]],
                         ["other"])

    def test_entries_sorted_by_descending_log_lr(self):
        table = bm.to_compatibility_table(
            "LT1", {"a": 0.5, "b": 3.0, "c": 1.5}, "v")
        lrs = [e["log_lr"] for e in table["entries"]]
        self.assertEqual(lrs, sorted(lrs, reverse=True))

    def test_empty_scores_give_a_valid_empty_table(self):
        table = bm.to_compatibility_table("LT1", {}, "v")
        self.assertEqual(table["entries"], [])
        self.assertEqual(table["tracklet_id"], "LT1")




class EffectiveCandidatesTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(bm.effective_candidates({}), 0.0)

    def test_single_dominant_candidate_pins_the_match(self):
        # "One International Place": one candidate far above the rest.
        n_eff = bm.effective_candidates({"a": 10.0, "b": 0.0, "c": 0.0})
        self.assertLess(n_eff, 1.2)

    def test_indistinguishable_candidates_spread_the_match(self):
        # "a tower": twelve identical charted towers, none preferred.
        n_eff = bm.effective_candidates({f"t{i}": 1.0 for i in range(12)})
        self.assertAlmostEqual(n_eff, 12.0, places=4)

    def test_named_match_is_more_specific_than_category_match(self):
        named = bm.effective_candidates({"oip": 4.0, **{f"t{i}": 1.0
                                                        for i in range(11)}})
        category = bm.effective_candidates({f"t{i}": 1.0 for i in range(12)})
        self.assertLess(named, category)


class EstimateMountOffsetTest(unittest.TestCase):
    """A rigid camera implies ONE offset; disagreement is the signal."""

    def _obs(self, keyframe, east, north, course, camera_az):
        o = bm.Observation(keyframe, east, north, 0.0)
        o.course_deg = course
        o.bearing_camera_deg = camera_az
        return o

    def test_recovers_a_known_offset(self):
        target = entry("lm", 0.0, 1000.0)   # due north of every pose
        true_offset = 222.4
        observations = []
        for i, east in enumerate((-300.0, -100.0, 100.0)):
            true_bearing = hc.world_bearing_deg(east, 0.0, 0.0, 1000.0)
            course = 90.0
            observations.append(self._obs(
                i, east, 0.0, course,
                (true_bearing + true_offset - course) % 360.0))
        per, consensus = bm.estimate_mount_offset(
            {"T": observations}, {"T": "lm"}, {"lm": target})
        self.assertAlmostEqual(per["T"][0], true_offset, places=3)
        self.assertLess(per["T"][1], 1e-6)
        self.assertAlmostEqual(consensus, true_offset, places=3)

    def test_too_few_observations_are_skipped(self):
        target = entry("lm", 0.0, 1000.0)
        per, _ = bm.estimate_mount_offset(
            {"T": [self._obs(0, 0.0, 0.0, 0.0, 10.0)]}, {"T": "lm"},
            {"lm": target})
        self.assertEqual(per, {})



class TriangulateTest(unittest.TestCase):
    def _rays_to(self, target_e, target_n, poses, noise=None):
        out = []
        for i, (e, n) in enumerate(poses):
            bearing = hc.world_bearing_deg(e, n, target_e, target_n)
            if noise:
                bearing += noise[i]
            out.append(obs(i, e, n, bearing % 360.0))
        return out

    def test_needs_two_observations(self):
        self.assertIsNone(bm.triangulate([]))
        self.assertIsNone(bm.triangulate([obs(0, 0, 0, 0.0)]))

    def test_recovers_an_exact_intersection(self):
        rays = self._rays_to(1200.0, 3000.0,
                             [(-500.0, 0.0), (0.0, 0.0), (500.0, 0.0)])
        east, north, residual, _ = bm.triangulate(rays)
        self.assertAlmostEqual(east, 1200.0, places=3)
        self.assertAlmostEqual(north, 3000.0, places=3)
        self.assertLess(residual, 1e-6)

    def test_parallel_rays_are_underdetermined(self):
        # Same pose repeated: no baseline, no intersection.
        self.assertIsNone(bm.triangulate(
            [obs(0, 0.0, 0.0, 30.0), obs(1, 0.0, 0.0, 30.0)]))

    def test_residual_reports_inconsistent_bearings(self):
        clean = self._rays_to(0.0, 4000.0,
                              [(-600.0, 0.0), (0.0, 0.0), (600.0, 0.0)])
        noisy = self._rays_to(0.0, 4000.0,
                              [(-600.0, 0.0), (0.0, 0.0), (600.0, 0.0)],
                              noise=[8.0, -8.0, 8.0])
        self.assertLess(bm.triangulate(clean)[2], 1e-6)
        self.assertGreater(bm.triangulate(noisy)[2], 3.0)

    def test_short_baseline_is_ill_conditioned_despite_small_residual(self):
        """A tiny residual must not be read as a confident position."""
        short = self._rays_to(0.0, 20000.0, [(-20.0, 0.0), (20.0, 0.0)])
        long_ = self._rays_to(0.0, 20000.0, [(-4000.0, 0.0), (4000.0, 0.0)])
        self.assertLess(short[2] if (short := bm.triangulate(short)) else 9,
                        1e-6)
        long_ = bm.triangulate(long_)
        self.assertGreater(short[3], long_[3])



class TriangulateBehindObserverTest(unittest.TestCase):
    """Lines intersect where rays do not; a solution behind the sensor is not
    a triangulation. One leg1 tracklet reported a 179.7 deg residual this way."""

    def test_solution_behind_the_observations_is_rejected(self):
        # The west pose looks south-WEST and the east pose south-EAST, so the
        # rays diverge going south; extended backwards the lines still cross,
        # ~2.8 km due north of both. That crossing is behind the sensors and
        # is not a triangulation.
        rays = [obs(0, -500.0, 0.0, 190.0), obs(1, 500.0, 0.0, 170.0)]
        self.assertIsNone(bm.triangulate(rays))

    def test_converging_rays_south_are_accepted(self):
        # Mirror image of the above: swapping the bearings makes the rays
        # converge, and the same crossing point is now legitimately ahead.
        rays = [obs(0, -500.0, 0.0, 170.0), obs(1, 500.0, 0.0, 190.0)]
        result = bm.triangulate(rays)
        self.assertIsNotNone(result)
        self.assertLess(result[1], 0.0)      # south of the baseline

    def test_forward_solution_still_accepted(self):
        rays = [obs(0, -500.0, 0.0, 10.0), obs(1, 500.0, 0.0, 350.0)]
        result = bm.triangulate(rays)
        self.assertIsNotNone(result)
        self.assertGreater(result[1], 0.0)      # north of the baseline

if __name__ == "__main__":
    unittest.main()
