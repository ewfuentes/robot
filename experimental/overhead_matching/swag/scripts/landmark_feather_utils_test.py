"""Tests for landmark_feather_utils (dedupe + merge)."""

import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from experimental.overhead_matching.swag.scripts import landmark_feather_utils as lfu

# ~1e-4 deg latitude is ~11 m, so these offsets straddle the 10 m default.
TOUCHING = 1e-6      # ~0.1 m apart
FAR = 1e-2           # ~1.1 km apart


def make_frame(rows: list[dict], geoms: list, landmark_type="enc") -> gpd.GeoDataFrame:
    data = {"id": [f"('t', {i})" for i in range(len(rows))],
            "landmark_type": [landmark_type] * len(rows)}
    for key in sorted({k for r in rows for k in r}):
        data[key] = [r.get(key) for r in rows]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


class DedupeTest(unittest.TestCase):
    def test_touching_same_tag_features_merge(self):
        """One bridge stored as two abutting segments becomes one landmark."""
        frame = make_frame(
            [{"man_made": "bridge", "name": "Summer Street Bridge"},
             {"man_made": "bridge", "name": "Summer Street Bridge"}],
            [LineString([(-71.0, 42.3), (-71.0 + TOUCHING, 42.3)]),
             LineString([(-71.0 + 2 * TOUCHING, 42.3), (-71.0 + 3 * TOUCHING, 42.3)])])
        out = lfu.dedupe_exact_duplicates(frame, verbose=False)
        self.assertEqual(len(out), 1)
        # The survivor covers both original segments.
        self.assertAlmostEqual(out.geometry.iloc[0].bounds[0], -71.0, places=9)
        self.assertAlmostEqual(out.geometry.iloc[0].bounds[2],
                               -71.0 + 3 * TOUCHING, places=9)

    def test_distant_same_tag_features_are_kept(self):
        """Two unrelated islands really are both named "Deer Island"."""
        frame = make_frame(
            [{"place": "island", "name": "Deer Island"},
             {"place": "island", "name": "Deer Island"}],
            [Point(-71.0, 42.3), Point(-71.0 + FAR, 42.3)])
        self.assertEqual(len(lfu.dedupe_exact_duplicates(frame, verbose=False)), 2)

    def test_distinct_tags_never_merge_even_when_coincident(self):
        frame = make_frame(
            [{"man_made": "lighthouse", "name": "Boston Light"},
             {"man_made": "beacon"}],
            [Point(-71.0, 42.3), Point(-71.0 + TOUCHING, 42.3)])
        self.assertEqual(len(lfu.dedupe_exact_duplicates(frame, verbose=False)), 2)

    def test_many_generic_features_survive(self):
        """1638 distinct Boston piers share man_made=pier; spacing keeps them."""
        rows = [{"man_made": "pier"} for _ in range(20)]
        geoms = [Point(-71.0 + i * FAR, 42.3) for i in range(20)]
        out = lfu.dedupe_exact_duplicates(make_frame(rows, geoms), verbose=False)
        self.assertEqual(len(out), 20)

    def test_transitive_chain_collapses_once(self):
        """A ---touching--- B ---touching--- C is one physical feature."""
        rows = [{"man_made": "quay"}] * 3
        geoms = [Point(-71.0 + i * TOUCHING, 42.3) for i in range(3)]
        out = lfu.dedupe_exact_duplicates(make_frame(rows, geoms), verbose=False)
        self.assertEqual(len(out), 1)

    def test_tolerance_zero_disabled_via_caller(self):
        rows = [{"man_made": "quay"}] * 2
        geoms = [Point(-71.0, 42.3), Point(-71.0 + TOUCHING, 42.3)]
        out = lfu.dedupe_exact_duplicates(make_frame(rows, geoms),
                                          tolerance_m=0.0, verbose=False)
        self.assertEqual(len(out), 2)

    def test_empty_frame(self):
        frame = make_frame([], [])
        self.assertEqual(len(lfu.dedupe_exact_duplicates(frame, verbose=False)), 0)

    def test_preserves_columns_and_crs(self):
        frame = make_frame(
            [{"man_made": "pier", "name": "A"}, {"man_made": "quay", "name": "B"}],
            [Polygon([(-71.0, 42.3), (-71.0, 42.31), (-70.99, 42.31)]),
             Point(-70.9, 42.3)])
        out = lfu.dedupe_exact_duplicates(frame, verbose=False)
        self.assertEqual(list(out.columns), list(frame.columns))
        self.assertEqual(str(out.crs), "EPSG:4326")


class MergeTest(unittest.TestCase):
    def test_column_union_and_provenance(self):
        osm = make_frame([{"man_made": "pier", "name": "Rowes Wharf"}],
                         [Point(-71.05, 42.35)], landmark_type="historical")
        enc = make_frame([{"place": "island", "name": "Georges Island"}],
                         [Point(-70.92, 42.32)], landmark_type="enc")
        enc["id"] = ["('enc', 'X1')"]
        merged = lfu.merge_feathers([osm, enc])
        self.assertEqual(len(merged), 2)
        self.assertEqual(set(merged["landmark_type"]), {"historical", "enc"})
        for column in ("man_made", "place", "name"):
            self.assertIn(column, merged.columns)
        # The OSM row has no `place` tag; pd.concat fills it with NaN, which
        # prune_landmark drops via pd.isna just like a None.
        self.assertTrue(pd.isna(merged["place"][0]))
        self.assertEqual(merged["place"][1], "island")

    def test_duplicate_ids_raise(self):
        a = make_frame([{"man_made": "pier"}], [Point(-71.0, 42.3)])
        b = make_frame([{"man_made": "quay"}], [Point(-70.9, 42.3)])
        with self.assertRaises(ValueError):
            lfu.merge_feathers([a, b])

    def test_pruned_props_rejected(self):
        frame = make_frame([{"man_made": "pier"}], [Point(-71.0, 42.3)])
        frame["pruned_props"] = [frozenset()]
        with self.assertRaises(ValueError):
            lfu.merge_feathers([frame])

    def test_missing_crs_rejected(self):
        frame = make_frame([{"man_made": "pier"}], [Point(-71.0, 42.3)])
        with self.assertRaises(ValueError):
            lfu.merge_feathers([frame.set_crs(None, allow_override=True)])


if __name__ == "__main__":
    unittest.main()
