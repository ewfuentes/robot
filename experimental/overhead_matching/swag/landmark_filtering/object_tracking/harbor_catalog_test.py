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



class TagsToTextTest(unittest.TestCase):
    def test_unique_nonempty_sorted(self):
        self.assertEqual(
            hc.tags_to_text({"a": "x", "b": "x", "c": "y", "d": " "}),
            ["x", "y"])




class HarborTagPruningTest(unittest.TestCase):
    def test_keeps_maritime_vocabulary_the_street_list_drops(self):
        tags = hc.prune_far_field_tags({
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
        tags = hc.prune_far_field_tags({
            "addr:housenumber": "42", "addr:street": "Atlantic Ave",
            "opening_hours": "24/7", "payment:cash": "yes",
            "massgis:way_id": "1234", "name:fr": "Phare",
            "building": "yes",
        })
        self.assertEqual(set(tags), {"building"})

    def test_keeps_religion_the_only_discriminating_tag_on_a_temple(self):
        # amenity=place_of_worship is ~90% christian in an Anglophone harbour and
        # a genuine 3-way split in CJK; religion is what separates a torii from a
        # steeple, and a distant observer can see the difference.
        tags = hc.prune_far_field_tags({
            "amenity": "place_of_worship", "religion": "buddhist",
            "denomination": "jodo_shinshu", "name": "\uc5ed\uc0ac\uc0ac"})
        self.assertEqual(tags["religion"], "buddhist")
        self.assertEqual(tags["denomination"], "jodo_shinshu")

    def test_keeps_latin_name_variants_and_drops_the_rest(self):
        tags = hc.prune_far_field_tags({
            "name": "\ud3ec\ud56d\uc2dc", "name:en": "Pohang",
            "name:ko-Latn": "Pohang-si", "name:ko": "\ud3ec\ud56d\uc2dc",
            "name:ja": "\u30dd\u30bf\u30f3", "name:el": "noise",
            "place": "city"})
        self.assertEqual(tags["name:en"], "Pohang")
        self.assertEqual(tags["name:ko-Latn"], "Pohang-si")
        for dropped in ("name:ko", "name:ja", "name:el"):
            self.assertNotIn(dropped, tags)

    def test_name_variant_exception_does_not_admit_other_prefixes(self):
        # The exception is checked before FAR_FIELD_DROP_PREFIXES, so it must be
        # narrow enough not to reopen addr:/source:/ref:.
        self.assertTrue(hc.keeps_tag_key("name:en"))
        self.assertTrue(hc.keeps_tag_key("name:zh-Latn"))
        self.assertFalse(hc.keeps_tag_key("name:zh"))
        self.assertFalse(hc.keeps_tag_key("addr:street"))
        self.assertFalse(hc.keeps_tag_key("source:name"))
        self.assertFalse(hc.keeps_tag_key("ref:name"))

    def test_skips_missing_and_placeholder_values(self):
        tags = hc.prune_far_field_tags(
            {"name": float("nan"), "man_made": "  ", "building": "none",
             "historic": "fort"})
        self.assertEqual(set(tags), {"historic"})

    def test_values_are_strings(self):
        tags = hc.prune_far_field_tags({"height": 42.5, "building:levels": 3})
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
