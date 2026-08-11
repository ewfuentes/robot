import unittest

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    pano_geometry as pg,
    track_merge as tm,
)

PANO_W = 7680


def make_track(track_id, boxes_by_kf):
    """boxes_by_kf: {keyframe: (x0, y0, x1, y1)} in pano coords."""
    kfs = sorted(boxes_by_kf)
    records = []
    for kf in kfs:
        x0, y0, x1, y1 = boxes_by_kf[kf]
        records.append({"keyframe": kf, "action": "continue_mask",
                        "window_origin": [0.0, 0.0], "window_px": 1024,
                        "mask_bbox_window": [x0, y0, x1, y1]})
    return {"track_id": track_id, "birth_keyframe": kfs[0],
            "end_keyframe": kfs[-1], "last_keyframe": kfs[-1],
            "status": "closed", "close_reason": "starved",
            "n_supported_keyframes": len(kfs), "records": records}


def evidence(n_supports=10, names=None, tags=None, n_kf=10):
    return {"n_supports": n_supports, "n_supported_keyframes": n_kf,
            "name_votes": names or {}, "tag_votes": tags or {}}


def constant(track_id, x0, kfs, width=100, height=200, y0=1000):
    return make_track(track_id,
                      {kf: (x0, y0, x0 + width, y0 + height) for kf in kfs})


class BoxOverlapTest(unittest.TestCase):
    def test_identical_boxes(self):
        box = (100.0, 0.0, 200.0, 100.0)
        iou, contain, frac = tm.box_overlap(box, box, PANO_W)
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(contain, 1.0)
        self.assertAlmostEqual(frac, 1.0)

    def test_disjoint_boxes(self):
        iou, contain, _ = tm.box_overlap((0.0, 0.0, 100.0, 100.0),
                                         (500.0, 0.0, 600.0, 100.0), PANO_W)
        self.assertEqual(iou, 0.0)
        self.assertEqual(contain, 0.0)

    def test_contained_box(self):
        big = (0.0, 0.0, 400.0, 400.0)
        small = (100.0, 100.0, 200.0, 200.0)
        iou, contain, frac = tm.box_overlap(big, small, PANO_W)
        self.assertAlmostEqual(contain, 1.0)
        self.assertLess(iou, 0.1)
        self.assertAlmostEqual(frac, 1.0 / 16.0)

    def test_overlap_is_wrap_safe(self):
        # Boxes straddling x=0 must register as overlapping, not opposite.
        a = (PANO_W - 50.0, 0.0, PANO_W - 50.0 + 100.0, 100.0)
        b = (0.0, 0.0, 100.0, 100.0)
        iou, _, _ = tm.box_overlap(a, b, PANO_W)
        self.assertGreater(iou, 0.3)

    def test_angular_separation_wraps(self):
        sep = tm.angular_separation_deg((PANO_W - 10.0, 0, PANO_W, 10),
                                        (0.0, 0, 10.0, 10), PANO_W)
        self.assertLess(sep, 2.0)


class ComparePairTest(unittest.TestCase):
    def _cmp(self, a, b, cfg=None):
        cfg = cfg or tm.MergeConfig()
        return tm.compare_pair(a, tm.mask_boxes_by_keyframe(a), b,
                               tm.mask_boxes_by_keyframe(b), PANO_W, cfg)

    def test_coincident_tracks_are_duplicates(self):
        a = constant(1, 1000, range(10))
        b = constant(2, 1005, range(10))
        stats = self._cmp(a, b)
        self.assertEqual(stats.verdict, tm.DUPLICATE)
        self.assertGreater(stats.median_iou, 0.5)

    def test_separated_tracks_are_distinct(self):
        # The Custom House case: co-visible, far apart, cannot be one object.
        a = constant(1, 1000, range(10))
        b = constant(2, 3000, range(10))
        stats = self._cmp(a, b)
        self.assertEqual(stats.verdict, tm.DISTINCT)
        self.assertGreater(stats.median_sep_deg, 5.0)

    def test_contained_track_is_parent_child(self):
        # Fort on an island: small mask inside a much larger one.
        big = make_track(1, {kf: (1000.0, 1000.0, 1800.0, 1400.0)
                             for kf in range(10)})
        small = make_track(2, {kf: (1200.0, 1100.0, 1350.0, 1200.0)
                               for kf in range(10)})
        stats = self._cmp(big, small)
        self.assertEqual(stats.verdict, tm.PARENT_CHILD)
        self.assertEqual(stats.parent, 1)
        self.assertEqual(stats.child, 2)

    def test_partial_overlap_is_ambiguous_not_distinct(self):
        # Tobin-Bridge shape: same object, different mask extents. Must not
        # merge, but must not assert a cannot-link either.
        a = make_track(1, {kf: (1000.0, 1000.0, 1300.0, 1200.0)
                           for kf in range(10)})
        b = make_track(2, {kf: (1150.0, 1000.0, 1450.0, 1200.0)
                           for kf in range(10)})
        stats = self._cmp(a, b)
        self.assertEqual(stats.verdict, tm.AMBIGUOUS)
        self.assertGreater(stats.median_iou, tm.MergeConfig().ambiguous_min_iou)
        self.assertLess(stats.median_iou, tm.MergeConfig().duplicate_min_iou)

    def test_ambiguous_pairs_neither_merge_nor_block(self):
        a = make_track(1, {kf: (1000.0, 1000.0, 1300.0, 1200.0)
                           for kf in range(10)})
        b = make_track(2, {kf: (1150.0, 1000.0, 1450.0, 1200.0)
                           for kf in range(10)})
        ev = {1: evidence(20), 2: evidence(20)}
        landmarks, stats = tm.merge_tracks(
            {t["track_id"]: t for t in (a, b)}, {}, ev, {}, PANO_W,
            tm.MergeConfig())
        self.assertEqual(len(landmarks), 2)
        review = [r for lm in landmarks for r in lm.review_pairs]
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["status"], "geometry_inconclusive")

    def test_non_overlapping_lifetimes_are_disjoint_with_gap(self):
        a = constant(1, 1000, range(0, 10))
        b = constant(2, 1000, range(20, 30))
        stats = self._cmp(a, b)
        self.assertEqual(stats.verdict, tm.DISJOINT)
        self.assertEqual(stats.gap_keyframes, 10)

    def test_thin_covisibility_never_asserts_distinct(self):
        # 2 shared keyframes is too little to hard-block a later merge.
        cfg = tm.MergeConfig(min_covisible_keyframes=3)
        a = constant(1, 1000, range(0, 12))
        b = constant(2, 5000, range(10, 20))
        stats = self._cmp(a, b, cfg)
        self.assertEqual(stats.verdict, tm.DISJOINT)

    def test_valid_segments_restrict_comparison(self):
        # A track whose tail drifted away: with the audit's valid segment
        # applied, only the good span participates.
        drifting = make_track(1, {kf: (1000.0 + 400 * max(0, kf - 5), 1000.0,
                                       1100.0 + 400 * max(0, kf - 5), 1200.0)
                                  for kf in range(10)})
        boxes_all = tm.mask_boxes_by_keyframe(drifting)
        boxes_valid = tm.mask_boxes_by_keyframe(
            drifting, [{"start_t": 0, "end_t": 5}])
        self.assertEqual(len(boxes_all), 10)
        self.assertEqual(len(boxes_valid), 6)
        self.assertEqual(max(boxes_valid), 5)


class ClusterTest(unittest.TestCase):
    def test_duplicates_group_and_distinct_stay_apart(self):
        stats = [
            tm.PairStats(1, 2, tm.DUPLICATE, 10, 0.9, 0.1),
            tm.PairStats(1, 3, tm.DISTINCT, 10, 0.0, 30.0),
            tm.PairStats(2, 3, tm.DISTINCT, 10, 0.0, 30.0),
        ]
        groups, conflicts = tm.cluster(stats, [1, 2, 3], tm.MergeConfig())
        sizes = sorted(len(v) for v in groups.values())
        self.assertEqual(sizes, [1, 2])
        self.assertEqual(conflicts, [])

    def test_contradictory_chain_is_split_and_reported(self):
        # A~B and B~C by overlap, but A and C are provably different: the
        # weld must not silently survive.
        stats = [
            tm.PairStats(1, 2, tm.DUPLICATE, 10, 0.9, 0.1),
            tm.PairStats(2, 3, tm.DUPLICATE, 10, 0.6, 0.1),
            tm.PairStats(1, 3, tm.DISTINCT, 10, 0.0, 40.0),
        ]
        groups, conflicts = tm.cluster(stats, [1, 2, 3], tm.MergeConfig())
        for members in groups.values():
            self.assertNotEqual(sorted(members), [1, 2, 3])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["dropped_edge"], (2, 3))


class MergeTracksTest(unittest.TestCase):
    def _run(self, tracks, evidences, audits=None, cfg=None):
        return tm.merge_tracks(
            {t["track_id"]: t for t in tracks},
            {}, evidences, audits or {}, PANO_W, cfg or tm.MergeConfig())

    def test_merges_duplicates_and_sums_evidence(self):
        a = constant(1, 1000, range(10))
        b = constant(2, 1005, range(10))
        ev = {1: evidence(30, {"Tobin Bridge": 5}, {"man_made=bridge": 30}),
              2: evidence(12, {"Tobin Bridge": 3}, {"man_made=bridge": 12})}
        landmarks, _ = self._run([a, b], ev)
        self.assertEqual(len(landmarks), 1)
        lm = landmarks[0]
        self.assertEqual(lm.track_ids, [1, 2])
        self.assertEqual(lm.n_supports, 42)
        self.assertEqual(lm.name_votes["Tobin Bridge"], 8)
        self.assertFalse(lm.name_contested)

    def test_same_name_different_place_does_not_merge(self):
        # Three tracks all named 'Custom House Tower', pairwise separated.
        tracks = [constant(i, 1000 + 2000 * i, range(10)) for i in (1, 2, 3)]
        ev = {i: evidence(20, {"Custom House Tower": 10}) for i in (1, 2, 3)}
        landmarks, _ = self._run(tracks, ev)
        self.assertEqual(len(landmarks), 3)

    def test_parent_child_links_without_merging(self):
        island = make_track(1, {kf: (1000.0, 1000.0, 1800.0, 1400.0)
                                for kf in range(10)})
        fort = make_track(2, {kf: (1200.0, 1100.0, 1350.0, 1200.0)
                              for kf in range(10)})
        ev = {1: evidence(40, tags={"place=island": 40}),
              2: evidence(10, tags={"historic=fort": 10})}
        landmarks, _ = self._run([island, fort], ev)
        self.assertEqual(len(landmarks), 2)
        by_tracks = {tuple(lm.track_ids): lm for lm in landmarks}
        self.assertEqual(len(by_tracks[(1,)].parent_of), 1)
        self.assertEqual(len(by_tracks[(2,)].child_of), 1)

    def test_handoff_is_proposed_not_merged(self):
        a = constant(1, 1000, range(0, 10))
        b = constant(2, 1000, range(20, 30))
        ev = {1: evidence(20, {"Custom House Tower": 8}),
              2: evidence(20, {"Custom House Tower": 6})}
        landmarks, _ = self._run([a, b], ev)
        self.assertEqual(len(landmarks), 2)
        proposals = [p for lm in landmarks for p in lm.handoff_proposals]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["shared_names"],
                         ["Custom House Tower"])
        self.assertEqual(proposals[0]["status"], "needs_ego_motion_check")

    def test_merged_group_reports_contested_names(self):
        a = constant(1, 1000, range(10))
        b = constant(2, 1005, range(10))
        ev = {1: evidence(20, {"A": 5, "B": 4}), 2: evidence(20, {"C": 4})}
        landmarks, _ = self._run([a, b], ev)
        self.assertTrue(landmarks[0].name_contested)


class BearingSeriesTest(unittest.TestCase):
    def test_azimuth_convention_matches_pano_geometry(self):
        # Pin to the shared convention rather than restating it: whatever
        # pano_geometry says a mask-centre pixel's azimuth is, that is what
        # the bearing series must report.
        for centre_x in (0.0, PANO_W / 4, PANO_W / 2, 3 * PANO_W / 4):
            track = constant(1, centre_x - 50, range(3), width=100)
            expected, _ = pg.direction_from_pano_px(centre_x, 0.0, PANO_W, 1)
            series = tm.bearing_series(track, PANO_W)
            self.assertEqual(len(series), 3)
            for _, az, width in series:
                self.assertAlmostEqual(az, expected, places=5)
                self.assertAlmostEqual(width, 100.0 / PANO_W * 360.0,
                                       places=5)

    def test_fuse_produces_one_measurement_per_epoch(self):
        track = constant(1, 1000, range(20), width=100)
        series = tm.bearing_series(track, PANO_W)
        fused = tm.fuse_bearings(series, epoch_keyframes=5,
                                 bearing_sigma_deg=1.0)
        self.assertEqual(len(fused), 4)
        for anchor, _, kappa in fused:
            self.assertIn(anchor, range(20))
            self.assertGreater(kappa, 0.0)

    def test_wide_object_gets_lower_concentration(self):
        narrow = tm.fuse_bearings(
            tm.bearing_series(constant(1, 1000, range(5), width=20), PANO_W),
            5, 1.0)
        wide = tm.fuse_bearings(
            tm.bearing_series(constant(2, 1000, range(5), width=2000),
                              PANO_W), 5, 1.0)
        self.assertGreater(narrow[0][2], wide[0][2])

    def test_fused_azimuth_is_circular_mean_across_wrap(self):
        # Azimuth 0 sits at pano x = W/2, so straddle THAT to cross the
        # 359->0 wrap (x=0 is azimuth 180, nowhere near it).
        half = PANO_W / 2
        track = make_track(1, {0: (half - 100.0, 0, half - 50.0, 10),
                               1: (half, 0, half + 50.0, 10)})
        fused = tm.fuse_bearings(tm.bearing_series(track, PANO_W), 5, 1.0)
        self.assertEqual(len(fused), 1)
        az = fused[0][1]
        self.assertTrue(az > 355.0 or az < 5.0, f"az={az} not near wrap")


if __name__ == "__main__":
    unittest.main()
