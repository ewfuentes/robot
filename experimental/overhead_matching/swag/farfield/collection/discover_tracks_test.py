#!/usr/bin/env python3
"""Tests for the discovery-time trip clustering.

    bazel test //experimental/overhead_matching/swag/farfield/collection:discover_tracks_test

No network: candidates are built by hand. The point under test is the grouping
contract — same creator + endpoints within max_gap_km + start times within
max_time_gap_h — and especially that a multi-year campaign no longer collapses
into one "trip" the way the pre-time-condition clustering did.
"""

import unittest

from experimental.overhead_matching.swag.farfield.collection.discover_tracks import cluster_tracks

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS


def _cand(seq_id, creator, t_ms, start=(0.0, 0.0), end=(0.01, 0.0), km=5.0):
    """A minimal candidate; coords are (lon, lat) like the real ones."""
    return {
        "sequence_id": seq_id,
        "creator_id": creator,
        "captured_ms": t_ms,
        "length_km": km,
        "is_pano": True,
        "captured_year": 2026,
        "coords": [list(start), list(end)],
    }


def _groups(cands, **kwargs):
    return {frozenset(c["sequence_id"] for c in g)
            for g in cluster_tracks(cands, **kwargs)}


class ClusterTracksTest(unittest.TestCase):

    def test_same_outing_merges(self):
        # Two fragments of one drive: same creator, shared endpoint, the second
        # starts 40 min after the first (the first fragment's duration).
        a = _cand("a", "u1", 0, end=(0.05, 0.0))
        b = _cand("b", "u1", 40 * 60 * 1000, start=(0.05, 0.0), end=(0.10, 0.0))
        self.assertEqual(_groups([a, b]), {frozenset({"a", "b"})})

    def test_campaign_splits_on_time(self):
        # Same creator leaving the same driveway three days apart used to chain
        # into one fictional multi-day "trip"; the time condition splits it.
        a = _cand("a", "u1", 0)
        b = _cand("b", "u1", 3 * DAY_MS)
        self.assertEqual(_groups([a, b]),
                         {frozenset({"a"}), frozenset({"b"})})

    def test_disabling_time_condition_restores_old_behaviour(self):
        a = _cand("a", "u1", 0)
        b = _cand("b", "u1", 3 * DAY_MS)
        self.assertEqual(_groups([a, b], max_time_gap_h=0),
                         {frozenset({"a", "b"})})

    def test_long_outing_chains_transitively(self):
        # A 4.5 h outing in three fragments: each consecutive pair is inside
        # the 2 h window even though the ends are not. Single-linkage is the
        # right behaviour here, not a bug.
        a = _cand("a", "u1", 0, end=(0.05, 0.0))
        b = _cand("b", "u1", int(1.5 * HOUR_MS), start=(0.05, 0.0), end=(0.10, 0.0))
        c = _cand("c", "u1", 3 * HOUR_MS, start=(0.10, 0.0), end=(0.15, 0.0))
        self.assertEqual(_groups([a, b, c]), {frozenset({"a", "b", "c"})})

    def test_different_creator_never_merges(self):
        a = _cand("a", "u1", 0)
        b = _cand("b", "u2", 0)
        self.assertEqual(_groups([a, b]),
                         {frozenset({"a"}), frozenset({"b"})})

    def test_distant_endpoints_never_merge(self):
        # ~11 km apart at the equator, same creator, same instant.
        a = _cand("a", "u1", 0)
        b = _cand("b", "u1", 0, start=(0.10, 0.0), end=(0.11, 0.0))
        self.assertEqual(_groups([a, b]),
                         {frozenset({"a"}), frozenset({"b"})})

    def test_missing_timestamp_never_merges(self):
        # An unknown time is not evidence of adjacency; strict is deliberate.
        a = _cand("a", "u1", None)
        b = _cand("b", "u1", 0)
        self.assertEqual(_groups([a, b]),
                         {frozenset({"a"}), frozenset({"b"})})

    def test_mass_duplicated_stamp_treated_as_missing(self):
        # A clock-less camera stamps every fragment with one default instant
        # (Denver: 180 fragments all saying 1989-05-29 06:01:00). Identical
        # stamps satisfy "adjacent in time" vacuously, so 4+ verbatim repeats
        # from one creator are treated as no timestamp at all.
        chain = [_cand(f"c{i}", "u1", 12345,
                       start=(0.02 * i, 0.0), end=(0.02 * (i + 1), 0.0))
                 for i in range(5)]
        self.assertEqual(_groups(chain),
                         {frozenset({c["sequence_id"]}) for c in chain})

    def test_few_duplicate_stamps_still_merge(self):
        # Two equal stamps can be rounding; only mass duplication is a broken
        # clock. A pair sharing one timestamp with touching endpoints merges.
        a = _cand("a", "u1", 12345, end=(0.05, 0.0))
        b = _cand("b", "u1", 12345, start=(0.05, 0.0), end=(0.10, 0.0))
        self.assertEqual(_groups([a, b]), {frozenset({"a", "b"})})

    def test_duplicate_stamps_across_creators_do_not_poison(self):
        # The multiplicity count is per creator: many uploaders legitimately
        # share a coarse timestamp without impugning each other's clocks.
        pairs = []
        for i in range(4):
            u = f"u{i}"
            pairs += [_cand(f"{u}a", u, 999, end=(0.05, 0.0)),
                      _cand(f"{u}b", u, 999, start=(0.05, 0.0))]
        got = _groups(pairs)
        self.assertIn(frozenset({"u0a", "u0b"}), got)
        self.assertEqual(len(got), 4)

    def test_groups_sorted_by_total_km(self):
        small = _cand("s", "u1", 0, km=2.0)
        big1 = _cand("b1", "u2", 0, km=8.0, end=(0.05, 0.0))
        big2 = _cand("b2", "u2", HOUR_MS, km=8.0,
                     start=(0.05, 0.0), end=(0.10, 0.0))
        clusters = cluster_tracks([small, big1, big2])
        self.assertEqual([c["sequence_id"] for c in clusters[0]], ["b1", "b2"])
        self.assertEqual([c["sequence_id"] for c in clusters[1]], ["s"])


if __name__ == "__main__":
    unittest.main()
