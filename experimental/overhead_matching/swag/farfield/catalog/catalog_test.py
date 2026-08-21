import tempfile
import unittest
from pathlib import Path

import numpy as np
import shapely

from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as cat,
    schema,
)

ANCHOR = (42.35, -71.05)


def point_entry(landmark_id, east, north, tags=None):
    return cat.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=cat.OSM_POSITION_SIGMA_M, tags=tags or {})


def extended_entry(landmark_id, east, north, half_size):
    """Square footprint centred on (east, north)."""
    e = np.array([east - half_size, east + half_size,
                  east + half_size, east - half_size], dtype=np.float64)
    n = np.array([north - half_size, north - half_size,
                  north + half_size, north + half_size], dtype=np.float64)
    return cat.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=cat.OSM_POSITION_SIGMA_M, tags={},
        hull_east_m=e, hull_north_m=n)


class BearingSpanTest(unittest.TestCase):
    def test_point_feature_has_zero_width(self):
        centre, half = cat.bearing_span_from(point_entry("p", 0, 1000), 0, 0)
        self.assertAlmostEqual(centre, 0.0)
        self.assertEqual(half, 0.0)

    def test_extended_feature_subtends_expected_angle(self):
        # 200 m square centred 1000 m north. The widest angle is subtended by
        # the NEAR corners at range 900, not by the centroid range:
        # 2*atan(100/900) = 12.68 deg. Using the centroid range would
        # understate the extent, which is the bias this function exists to
        # avoid.
        entry = extended_entry("big", 0.0, 1000.0, 100.0)
        centre, half = cat.bearing_span_from(entry, 0.0, 0.0)
        self.assertAlmostEqual(centre, 0.0, places=3)
        self.assertAlmostEqual(2 * half, 12.68, places=1)

    def test_span_is_wrap_safe_across_north(self):
        # Straddling due north must not report a ~180 deg half-width.
        entry = extended_entry("north", 0.0, 500.0, 50.0)
        centre, half = cat.bearing_span_from(entry, 0.0, 0.0)
        self.assertLess(half, 20.0)
        self.assertTrue(centre < 5.0 or centre > 355.0)


class TagPruningTest(unittest.TestCase):
    def test_keeps_maritime_vocabulary_the_street_list_drops(self):
        tags = cat.prune_far_field_tags({
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
        tags = cat.prune_far_field_tags({
            "addr:housenumber": "42", "addr:street": "Atlantic Ave",
            "opening_hours": "24/7", "payment:cash": "yes",
            "massgis:way_id": "1234", "name:fr": "Phare",
            "building": "yes",
        })
        self.assertEqual(set(tags), {"building"})

    def test_keeps_religion_the_only_discriminating_tag_on_a_temple(self):
        tags = cat.prune_far_field_tags({
            "amenity": "place_of_worship", "religion": "buddhist",
            "denomination": "jodo_shinshu", "name": "역사사"})
        self.assertEqual(tags["religion"], "buddhist")
        self.assertEqual(tags["denomination"], "jodo_shinshu")

    def test_keeps_latin_name_variants_and_drops_the_rest(self):
        tags = cat.prune_far_field_tags({
            "name": "포항시", "name:en": "Pohang",
            "name:ko-Latn": "Pohang-si", "name:ko": "포항시",
            "name:ja": "ポタン", "name:el": "noise",
            "place": "city"})
        self.assertEqual(tags["name:en"], "Pohang")
        self.assertEqual(tags["name:ko-Latn"], "Pohang-si")
        for dropped in ("name:ko", "name:ja", "name:el"):
            self.assertNotIn(dropped, tags)

    def test_name_variant_exception_does_not_admit_other_prefixes(self):
        self.assertTrue(cat.keeps_tag_key("name:en"))
        self.assertTrue(cat.keeps_tag_key("name:zh-Latn"))
        self.assertFalse(cat.keeps_tag_key("name:zh"))
        self.assertFalse(cat.keeps_tag_key("addr:street"))
        self.assertFalse(cat.keeps_tag_key("source:name"))
        self.assertFalse(cat.keeps_tag_key("ref:name"))

    def test_skips_missing_and_placeholder_values(self):
        tags = cat.prune_far_field_tags(
            {"name": float("nan"), "man_made": "  ", "building": "none",
             "historic": "fort"})
        self.assertEqual(set(tags), {"historic"})

    def test_values_are_strings(self):
        tags = cat.prune_far_field_tags({"height": 42.5,
                                         "building:levels": 3})
        self.assertEqual(tags, {"height": "42.5", "building:levels": "3"})


class IdTextTest(unittest.TestCase):
    def test_tuple_ids_keep_every_part(self):
        # The element kind discriminates: node 123 and way 123 differ.
        self.assertEqual(cat._id_text(("node", 123)), "node:123")
        self.assertEqual(cat._id_text(("way", 123)), "way:123")

    def test_stringified_tuple_ids_are_parsed(self):
        # The feather stores the repr, not the tuple.
        self.assertEqual(cat._id_text("('node', 31419650)"), "node:31419650")
        self.assertEqual(cat._id_text("('enc', '022602110C34FB5F')"),
                         "enc:022602110C34FB5F")

    def test_scalar_ids_pass_through(self):
        self.assertEqual(cat._id_text("way/123"), "way/123")
        self.assertEqual(cat._id_text(42), "42")


class LoadCatalogTest(unittest.TestCase):
    def _write_feather(self, tmp):
        # A point lighthouse ~1.1 km north of the anchor and a ~200 m-wide
        # polygon island east of it.
        lat0, lon0 = ANCHOR
        island = shapely.Polygon([
            (lon0 + 0.012, lat0 - 0.001), (lon0 + 0.014, lat0 - 0.001),
            (lon0 + 0.014, lat0 + 0.001), (lon0 + 0.012, lat0 + 0.001)])
        frame = schema.build_frame(
            ids=["('node', 1)", "('way', 2)"],
            geometries=[shapely.Point(lon0, lat0 + 0.01), island],
            landmark_types=["osm", "enc"],
            tags=[{"man_made": "lighthouse", "name": "Boston Light",
                   "addr:street": "dropped"},
                  {"natural": "island", "name": "Georges Island"}],
        )
        path = Path(tmp) / "v1.feather"
        frame.to_feather(path)
        return path

    def test_entries_land_in_enu_with_pruned_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = cat.load_catalog(self._write_feather(tmp), *ANCHOR)
        self.assertEqual(len(entries), 2)
        light, island = entries
        self.assertEqual(light.landmark_id, "osm:node:1")
        self.assertEqual(island.landmark_id, "enc:way:2")
        self.assertEqual(island.position_sigma_m, cat.ENC_POSITION_SIGMA_M)
        # ~0.01 deg lat north of the anchor.
        self.assertAlmostEqual(light.north_m, 1113.2, delta=1.0)
        self.assertAlmostEqual(light.east_m, 0.0, places=3)
        self.assertNotIn("addr:street", light.tags)
        self.assertEqual(light.tags["name"], "Boston Light")
        # The polygon carries a hull; the point does not.
        self.assertTrue(island.is_extended)
        self.assertFalse(light.is_extended)

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            first = cat.load_catalog_cached(path, *ANCHOR,
                                            cache_dir=cache_dir)
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 1)
            second = cat.load_catalog_cached(path, *ANCHOR,
                                             cache_dir=cache_dir)
        self.assertEqual([e.landmark_id for e in first],
                         [e.landmark_id for e in second])
        # A different anchor is a different cache entry, not a stale hit.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            cat.load_catalog_cached(path, *ANCHOR, cache_dir=cache_dir)
            moved = cat.load_catalog_cached(path, ANCHOR[0] + 0.1, ANCHOR[1],
                                            cache_dir=cache_dir)
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 2)
            self.assertNotAlmostEqual(moved[0].north_m, 1113.2, delta=1.0)


if __name__ == "__main__":
    unittest.main()
