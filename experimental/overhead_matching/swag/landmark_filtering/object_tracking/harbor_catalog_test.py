import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    harbor_catalog as hc,
)


def point_entry(landmark_id, east, north, tags=None):
    return hc.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=hc.OSM_POSITION_SIGMA_M, tags=tags or {})


def extended_entry(landmark_id, east, north, half_size):
    """Square footprint centred on (east, north)."""
    e = np.array([east - half_size, east + half_size,
                  east + half_size, east - half_size], dtype=np.float64)
    n = np.array([north - half_size, north - half_size,
                  north + half_size, north + half_size], dtype=np.float64)
    return hc.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=hc.OSM_POSITION_SIGMA_M, tags={},
        hull_east_m=e, hull_north_m=n)


class EnuTest(unittest.TestCase):
    def test_anchor_maps_to_origin(self):
        e, n = hc.enu_from_latlon(42.335, -70.99, 42.335, -70.99)
        self.assertAlmostEqual(float(e), 0.0, places=6)
        self.assertAlmostEqual(float(n), 0.0, places=6)

    def test_north_and_east_signs_and_scale(self):
        # 0.01 deg of latitude is ~1113 m north; longitude is shorter by
        # cos(lat) at this latitude.
        e, n = hc.enu_from_latlon(42.345, -70.99, 42.335, -70.99)
        self.assertGreater(float(n), 1100.0)
        self.assertLess(float(n), 1120.0)
        self.assertAlmostEqual(float(e), 0.0, places=6)
        e2, _ = hc.enu_from_latlon(42.335, -70.98, 42.335, -70.99)
        self.assertGreater(float(e2), 0.0)
        self.assertLess(float(e2), float(n))


class BearingTest(unittest.TestCase):
    def test_cardinal_bearings(self):
        self.assertAlmostEqual(hc.world_bearing_deg(0, 0, 0, 100), 0.0)
        self.assertAlmostEqual(hc.world_bearing_deg(0, 0, 100, 0), 90.0)
        self.assertAlmostEqual(hc.world_bearing_deg(0, 0, 0, -100), 180.0)
        self.assertAlmostEqual(hc.world_bearing_deg(0, 0, -100, 0), 270.0)

    def test_angular_delta_wraps(self):
        self.assertAlmostEqual(hc.angular_delta_deg(359.0, 1.0), -2.0)
        self.assertAlmostEqual(hc.angular_delta_deg(1.0, 359.0), 2.0)


class BearingSpanTest(unittest.TestCase):
    def test_point_feature_has_zero_width(self):
        centre, half = hc.bearing_span_from(point_entry("p", 0, 1000), 0, 0)
        self.assertAlmostEqual(centre, 0.0)
        self.assertEqual(half, 0.0)

    def test_extended_feature_subtends_expected_angle(self):
        # 200 m square centred 1000 m north. The widest angle is subtended by
        # the NEAR corners at range 900, not by the centroid range:
        # 2*atan(100/900) = 12.68 deg. Using the centroid range would
        # understate the extent, which is the bias this function exists to
        # avoid.
        entry = extended_entry("big", 0.0, 1000.0, 100.0)
        centre, half = hc.bearing_span_from(entry, 0.0, 0.0)
        self.assertAlmostEqual(centre, 0.0, places=3)
        self.assertAlmostEqual(2 * half, 12.68, places=1)

    def test_span_is_wrap_safe_across_north(self):
        # Straddling due north must not report a ~180 deg half-width.
        entry = extended_entry("north", 0.0, 500.0, 50.0)
        centre, half = hc.bearing_span_from(entry, 0.0, 0.0)
        self.assertLess(half, 20.0)
        self.assertTrue(centre < 5.0 or centre > 355.0)


class WedgeTest(unittest.TestCase):
    def setUp(self):
        self.entries = [
            point_entry("ahead", 0.0, 1000.0),
            point_entry("right", 1000.0, 0.0),
            point_entry("behind", 0.0, -1000.0),
            point_entry("far", 0.0, 50000.0),
            point_entry("near", 0.0, 10.0),
        ]

    def test_wedge_selects_by_bearing(self):
        got = hc.wedge_candidates(self.entries, 0, 0, 0.0, 5.0)
        self.assertEqual({e.landmark_id for e, *_ in got},
                         {"near", "ahead", "far"})

    def test_range_gates(self):
        got = hc.wedge_candidates(self.entries, 0, 0, 0.0, 5.0,
                                  max_range_m=20000.0, min_range_m=100.0)
        self.assertEqual({e.landmark_id for e, *_ in got}, {"ahead"})

    def test_results_sorted_by_range(self):
        got = hc.wedge_candidates(self.entries, 0, 0, 0.0, 5.0)
        ranges = [r for _, r, _, _ in got]
        self.assertEqual(ranges, sorted(ranges))

    def test_extended_candidate_qualifies_on_span_not_centroid(self):
        # Centroid 30 deg off-axis, but the footprint reaches into a narrow
        # wedge: an island must not be rejected for its centroid.
        wide = extended_entry("island", 600.0, 1000.0, 700.0)
        got = hc.wedge_candidates([wide], 0, 0, 0.0, 2.0)
        self.assertEqual(len(got), 1)
        narrow = point_entry("speck", 600.0, 1000.0)
        self.assertEqual(hc.wedge_candidates([narrow], 0, 0, 0.0, 2.0), [])

    def test_wedge_is_wrap_safe(self):
        entries = [point_entry("just_west_of_north", -20.0, 1000.0)]
        got = hc.wedge_candidates(entries, 0, 0, 359.0, 5.0)
        self.assertEqual(len(got), 1)


class TagsToTextTest(unittest.TestCase):
    def test_unique_nonempty_sorted(self):
        self.assertEqual(
            hc.tags_to_text({"a": "x", "b": "x", "c": "y", "d": " "}),
            ["x", "y"])




class HarborTagPruningTest(unittest.TestCase):
    def test_keeps_maritime_vocabulary_the_street_list_drops(self):
        tags = hc.prune_harbor_tags({
            "seamark:type": "light_major",
            "seamark:light:character": "Fl",
            "object_class": "LNDMRK",
            "man_made": "lighthouse",
            "name": "Boston Light",
        })
        self.assertEqual(set(tags), {
            "seamark:type", "seamark:light:character", "object_class",
            "man_made", "name"})

    def test_drops_street_furniture_unobservable_at_range(self):
        tags = hc.prune_harbor_tags({
            "addr:housenumber": "42", "addr:street": "Atlantic Ave",
            "opening_hours": "24/7", "payment:cash": "yes",
            "massgis:way_id": "1234", "name:fr": "Phare",
            "building": "yes",
        })
        self.assertEqual(set(tags), {"building"})

    def test_skips_missing_and_placeholder_values(self):
        tags = hc.prune_harbor_tags(
            {"name": float("nan"), "man_made": "  ", "building": "none",
             "historic": "fort"})
        self.assertEqual(set(tags), {"historic"})

    def test_values_are_strings(self):
        tags = hc.prune_harbor_tags({"height": 42.5, "building:levels": 3})
        self.assertEqual(tags, {"height": "42.5", "building:levels": "3"})



class IdTextTest(unittest.TestCase):
    def test_tuple_ids_keep_every_part(self):
        # The element kind discriminates: node 123 and way 123 differ.
        self.assertEqual(hc._id_text(("node", 123)), "node:123")
        self.assertEqual(hc._id_text(("way", 123)), "way:123")
        self.assertNotEqual(hc._id_text(("node", 123)), hc._id_text(("way", 123)))

    def test_stringified_tuple_ids_are_parsed(self):
        # The feather stores the repr, not the tuple.
        self.assertEqual(hc._id_text("('node', 31419650)"), "node:31419650")
        self.assertEqual(hc._id_text("('enc', '022602110C34FB5F')"),
                         "enc:022602110C34FB5F")

    def test_scalar_ids_pass_through(self):
        self.assertEqual(hc._id_text("way/123"), "way/123")
        self.assertEqual(hc._id_text(42), "42")

if __name__ == "__main__":
    unittest.main()
