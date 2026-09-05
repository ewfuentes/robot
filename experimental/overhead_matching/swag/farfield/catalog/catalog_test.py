import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import shapely

from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as cat,
    schema,
)

ANCHOR = (42.35, -71.05)
TEST_SIGMA_M = 12.5


def point_entry(landmark_id, east, north, tags=None):
    return cat.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=TEST_SIGMA_M, tags=tags or {})


def extended_entry(landmark_id, east, north, half_size):
    """Square footprint centred on (east, north)."""
    e = np.array([east - half_size, east + half_size,
                  east + half_size, east - half_size], dtype=np.float64)
    n = np.array([north - half_size, north - half_size,
                  north + half_size, north + half_size], dtype=np.float64)
    return cat.CatalogEntry(
        landmark_id=landmark_id, source="osm", east_m=east, north_m=north,
        position_sigma_m=TEST_SIGMA_M, tags={},
        hull_east_m=e, hull_north_m=n)


class BearingSpanTest(unittest.TestCase):
    def test_point_feature_has_zero_width(self):
        centre, half = cat.bearing_span_from(point_entry("p", 0, 1000), 0, 0)
        self.assertAlmostEqual(centre, 0.0)
        self.assertEqual(half, 0.0)

    def test_two_coordinate_hull_is_extended(self):
        entry = cat.CatalogEntry(
            landmark_id="line", source="osm", east_m=0.0, north_m=1000.0,
            position_sigma_m=TEST_SIGMA_M, tags={},
            hull_east_m=np.array([-100.0, 100.0]),
            hull_north_m=np.array([1000.0, 1000.0]),
        )
        self.assertTrue(entry.is_extended)
        _, half = cat.bearing_span_from(entry, 0.0, 0.0)
        self.assertGreater(half, 0.0)

    def test_extended_feature_subtends_expected_angle(self):
        # The near corners of this square determine its widest angle.
        entry = extended_entry("big", 0.0, 1000.0, 100.0)
        centre, half = cat.bearing_span_from(entry, 0.0, 0.0)
        self.assertAlmostEqual(centre, 0.0, places=3)
        self.assertAlmostEqual(2 * half, 12.68, places=1)

    def test_span_is_wrap_safe_across_north(self):
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
        self.assertEqual(cat._id_text(("node", 123)), "node:123")
        self.assertEqual(cat._id_text(("way", 123)), "way:123")

    def test_stringified_tuple_ids_are_parsed(self):
        self.assertEqual(cat._id_text("('node', 31419650)"), "node:31419650")
        self.assertEqual(cat._id_text("('enc', '022602110C34FB5F')"),
                         "enc:022602110C34FB5F")

    def test_scalar_ids_pass_through(self):
        self.assertEqual(cat._id_text("way/123"), "way/123")
        self.assertEqual(cat._id_text(42), "42")


class LoadCatalogTest(unittest.TestCase):
    def _write_feather(self, tmp, filename="v1.feather"):
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
        path = Path(tmp) / filename
        frame.to_feather(path)
        return path

    def test_entries_land_in_enu_with_uniform_sigma_and_pruned_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = cat.load_catalog(
                self._write_feather(tmp), *ANCHOR,
                position_sigma_m=TEST_SIGMA_M)
        self.assertEqual(len(entries), 2)
        light, island = entries
        self.assertEqual(light.landmark_id, "osm:node:1")
        self.assertEqual(island.landmark_id, "enc:way:2")
        self.assertEqual(
            [entry.position_sigma_m for entry in entries],
            [TEST_SIGMA_M, TEST_SIGMA_M])
        self.assertAlmostEqual(light.north_m, 1113.2, delta=1.0)
        self.assertAlmostEqual(light.east_m, 0.0, places=3)
        self.assertNotIn("addr:street", light.tags)
        self.assertEqual(light.tags["name"], "Boston Light")
        self.assertTrue(island.is_extended)
        self.assertFalse(light.is_extended)

    def test_position_sigma_is_required_positive_and_finite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            with self.assertRaises(TypeError):
                cat.load_catalog(path, *ANCHOR)
            for sigma in (True, 0.0, -1.0, float("inf"), float("nan")):
                with self.subTest(sigma=sigma):
                    with self.assertRaises(ValueError):
                        cat.load_catalog(
                            path, *ANCHOR, position_sigma_m=sigma)

    def test_rejects_unknown_source_instead_of_coercing_to_osm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            frame = schema.read_frame(path)
            frame.loc[0, "landmark_type"] = "survey"
            frame.to_feather(path)
            with self.assertRaisesRegex(
                    schema.CatalogSchemaError,
                    "landmark_type must be one of"):
                cat.load_catalog(
                    path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)

    def test_source_namespace_and_element_kind_prevent_id_collisions(self):
        lat0, lon0 = ANCHOR
        frame = schema.build_frame(
            ids=["('node', 7)", "('way', 7)", "('enc', '7')"],
            geometries=[
                shapely.Point(lon0, lat0),
                shapely.Point(lon0 + 0.001, lat0),
                shapely.Point(lon0 + 0.002, lat0),
            ],
            landmark_types=["osm", "osm", "enc"],
            tags=[{}, {}, {}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.feather"
            frame.to_feather(path)
            entries = cat.load_catalog(
                path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)
        self.assertEqual(
            [entry.landmark_id for entry in entries],
            ["osm:node:7", "osm:way:7", "enc:7"])

    def test_two_point_and_collinear_lines_retain_two_endpoint_hulls(self):
        lat0, lon0 = ANCHOR
        frame = schema.build_frame(
            ids=["two", "collinear"],
            geometries=[
                shapely.LineString([
                    (lon0 - 0.001, lat0 + 0.01),
                    (lon0 + 0.001, lat0 + 0.01),
                ]),
                shapely.LineString([
                    (lon0 - 0.002, lat0 + 0.02),
                    (lon0, lat0 + 0.02),
                    (lon0 + 0.002, lat0 + 0.02),
                ]),
            ],
            landmark_types=["osm", "osm"],
            tags=[{}, {}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lines.feather"
            frame.to_feather(path)
            entries = cat.load_catalog(
                path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)
        for entry in entries:
            self.assertEqual(entry.hull_east_m.size, 2)
            self.assertTrue(entry.is_extended)
            _, half_width = cat.bearing_span_from(entry, 0.0, 0.0)
            self.assertGreater(half_width, 0.0)

    def test_polygon_retains_complete_convex_hull_without_truncation(self):
        lat0, lon0 = ANCHOR
        angles = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
        polygon = shapely.Polygon([
            (lon0 + 0.01 + 0.002 * np.cos(angle),
             lat0 + 0.01 + 0.002 * np.sin(angle))
            for angle in angles
        ])
        expected_count = len(shapely.get_coordinates(polygon.convex_hull))
        self.assertGreater(expected_count, 24)
        frame = schema.build_frame(
            ids=["many"],
            geometries=[polygon],
            landmark_types=["osm"],
            tags=[{}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "polygon.feather"
            frame.to_feather(path)
            entry = cat.load_catalog(
                path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)[0]
        self.assertEqual(entry.hull_east_m.size, expected_count)

    def test_multipolygon_uses_one_complete_overall_convex_hull(self):
        lat0, lon0 = ANCHOR
        left = shapely.box(
            lon0 + 0.005, lat0 + 0.005, lon0 + 0.006, lat0 + 0.006)
        right = shapely.box(
            lon0 + 0.01, lat0 + 0.005, lon0 + 0.011, lat0 + 0.006)
        geometry = shapely.MultiPolygon([left, right])
        expected_count = len(
            shapely.get_coordinates(shapely.convex_hull(geometry)))
        frame = schema.build_frame(
            ids=["multi"], geometries=[geometry],
            landmark_types=["osm"], tags=[{}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.feather"
            frame.to_feather(path)
            entry = cat.load_catalog(
                path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)[0]
        self.assertEqual(entry.hull_east_m.size, expected_count)


class CatalogCacheTest(unittest.TestCase):
    def _write_feather(self, directory, name="Boston Light"):
        lat0, lon0 = ANCHOR
        path = Path(directory) / "catalog.feather"
        schema.build_frame(
            ids=["('node', 1)"],
            geometries=[shapely.Point(lon0, lat0 + 0.01)],
            landmark_types=["osm"],
            tags=[{"name": name}],
        ).to_feather(path)
        return path

    def _load(self, path, cache_dir, **kwargs):
        return cat.load_catalog_cached(
            path, *ANCHOR, cache_dir=cache_dir,
            position_sigma_m=kwargs.pop("position_sigma_m", TEST_SIGMA_M),
            **kwargs)

    def test_cache_directory_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            with self.assertRaises(TypeError):
                cat.load_catalog_cached(
                    path, *ANCHOR, position_sigma_m=TEST_SIGMA_M)
            self.assertFalse((path.parent / "catalog_cache").exists())

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            first = self._load(path, cache_dir)
            second = self._load(path, cache_dir)
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 1)
        self.assertEqual(
            [entry.landmark_id for entry in first],
            [entry.landmark_id for entry in second])

    def test_exact_anchor_and_all_loader_options_affect_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            self._load(path, cache_dir)
            cat.load_catalog_cached(
                path, ANCHOR[0] + 1e-10, ANCHOR[1],
                cache_dir=cache_dir, position_sigma_m=TEST_SIGMA_M)
            self._load(path, cache_dir, keep_hulls=False)
            different_sigma = self._load(
                path, cache_dir, position_sigma_m=TEST_SIGMA_M + 1.0)
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 4)
            self.assertEqual(
                different_sigma[0].position_sigma_m, TEST_SIGMA_M + 1.0)

    def test_same_size_restored_mtime_mutation_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp, name="Old")
            cache_dir = Path(tmp) / "cache"
            old = self._load(path, cache_dir)
            old_stat = path.stat()
            self._write_feather(tmp, name="New")
            self.assertEqual(path.stat().st_size, old_stat.st_size)
            os.utime(
                path,
                ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns),
            )
            self.assertEqual(path.stat().st_mtime_ns, old_stat.st_mtime_ns)
            fresh = self._load(path, cache_dir)
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 2)
        self.assertEqual(old[0].tags["name"], "Old")
        self.assertEqual(fresh[0].tags["name"], "New")

    def test_corrupt_cache_is_discarded_and_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            expected = self._load(path, cache_dir)
            cache_path = next(cache_dir.glob("catalog_*.pkl"))
            cache_path.write_bytes(b"not a pickle")
            rebuilt = self._load(path, cache_dir)
            self.assertGreater(cache_path.stat().st_size, len(b"not a pickle"))
        self.assertEqual(
            [entry.landmark_id for entry in rebuilt],
            [entry.landmark_id for entry in expected])

    def test_concurrent_builders_publish_one_complete_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_feather(tmp)
            cache_dir = Path(tmp) / "cache"
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._load, path, cache_dir)
                    for _ in range(4)
                ]
                results = [future.result() for future in futures]
            self.assertEqual(len(list(cache_dir.glob("catalog_*.pkl"))), 1)
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])
            again = self._load(path, cache_dir)
        self.assertTrue(all(
            result[0].landmark_id == again[0].landmark_id
            for result in results))


if __name__ == "__main__":
    unittest.main()
