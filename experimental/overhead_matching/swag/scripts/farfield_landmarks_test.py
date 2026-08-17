import unittest

from experimental.overhead_matching.swag.scripts import farfield_landmarks as fl


class ParseLengthTest(unittest.TestCase):
    def test_bare_number_is_metres(self):
        self.assertAlmostEqual(fl._parse_length_m("42"), 42.0)

    def test_metre_suffixes(self):
        for text in ("42 m", "42m", "42 metres", "42 meters"):
            self.assertAlmostEqual(fl._parse_length_m(text), 42.0, msg=text)

    def test_comma_decimal(self):
        self.assertAlmostEqual(fl._parse_length_m("42,5"), 42.5)

    def test_feet_suffix(self):
        self.assertAlmostEqual(fl._parse_length_m("100 ft"), 30.48, places=4)

    def test_feet_inches_notation(self):
        self.assertAlmostEqual(fl._parse_length_m("10'6\""), 3.2004, places=4)
        self.assertAlmostEqual(fl._parse_length_m("10'"), 3.048, places=4)

    def test_unparseable_returns_none_not_zero(self):
        # Returning 0.0 here would silently sink a landmark below the horizon
        # instead of falling back to the class default.
        for text in ("about 40", "tall", "", "~30m", None):
            self.assertIsNone(fl._parse_length_m(text), msg=repr(text))


class ClassifyTest(unittest.TestCase):
    def test_peak_is_in_dem_with_zero_structure(self):
        kind, height, in_dem = fl._classify({"natural": "peak", "ele": "4808"})
        self.assertEqual(kind, "natural:peak")
        self.assertEqual(height, 0.0)
        self.assertTrue(in_dem)

    def test_mast_is_not_in_dem(self):
        kind, height, in_dem = fl._classify({"man_made": "mast"})
        self.assertEqual(kind, "man_made:mast")
        self.assertGreater(height, 0.0)
        self.assertFalse(in_dem)

    def test_specific_class_wins_over_generic_tower(self):
        # A node carrying both must not be filed as a generic 40 m tower.
        kind, _height, _in_dem = fl._classify({"man_made": "communications_tower",
                                               "tower:type": "communication"})
        self.assertEqual(kind, "man_made:communications_tower")

    def test_street_furniture_is_out_of_scope(self):
        for tags in ({"amenity": "bench"}, {"highway": "crossing"},
                     {"man_made": "surveillance"}, {}):
            self.assertIsNone(fl._classify(tags), msg=str(tags))

    def test_linear_cliff_is_out_of_scope(self):
        # Cliffs are ways, so they would enter as a meaningless centroid -- and
        # at 60% of the Lake Geneva catalog they would dominate every count.
        self.assertIsNone(fl._classify({"natural": "cliff"}))

    def test_tall_building_admitted_by_height(self):
        kind, height, in_dem = fl._classify({"building": "yes", "height": "310 m"})
        self.assertEqual(kind, "building:tall")
        self.assertAlmostEqual(height, 310.0)
        self.assertFalse(in_dem)

    def test_short_building_rejected(self):
        self.assertIsNone(fl._classify({"building": "yes", "height": "12 m"}))

    def test_untagged_building_rejected(self):
        # Nearly half a harbour feather is untagged `building=yes`; admitting
        # those would swamp the catalog with things no distant observer sees.
        self.assertIsNone(fl._classify({"building": "yes"}))


class ElementsToLandmarksTest(unittest.TestCase):
    def test_peak_ignores_ele_for_structure_height(self):
        elements = [{"type": "node", "id": 1, "lat": 45.83, "lon": 6.86,
                     "tags": {"natural": "peak", "name": "Mont Blanc", "ele": "4808"}}]
        landmark, = fl.elements_to_landmarks(elements)

        self.assertEqual(landmark["structure_height_m"], 0.0)
        self.assertTrue(landmark["in_dem"])
        self.assertEqual(landmark["height_source"], "dem")
        # `ele` is kept for cross-checking the DEM, just not used as height.
        self.assertAlmostEqual(landmark["ele_m"], 4808.0)

    def test_structure_uses_tagged_height(self):
        elements = [{"type": "node", "id": 2, "lat": 52.0, "lon": 5.0,
                     "tags": {"man_made": "mast", "height": "230"}}]
        landmark, = fl.elements_to_landmarks(elements)
        self.assertAlmostEqual(landmark["structure_height_m"], 230.0)
        self.assertEqual(landmark["height_source"], "tag")

    def test_structure_falls_back_to_class_default(self):
        elements = [{"type": "node", "id": 3, "lat": 52.0, "lon": 5.0,
                     "tags": {"man_made": "mast"}}]
        landmark, = fl.elements_to_landmarks(elements)
        self.assertAlmostEqual(landmark["structure_height_m"], 100.0)
        self.assertEqual(landmark["height_source"], "default")

    def test_building_levels_fallback(self):
        elements = [{"type": "way", "id": 4, "center": {"lat": 40.0, "lon": -74.0},
                     "tags": {"building": "church", "building:levels": "10"}}]
        landmark, = fl.elements_to_landmarks(elements)
        self.assertAlmostEqual(landmark["structure_height_m"], 32.0)
        self.assertEqual(landmark["height_source"], "levels")

    def test_way_uses_center(self):
        elements = [{"type": "way", "id": 5, "center": {"lat": 51.5, "lon": -0.1},
                     "tags": {"man_made": "tower"}}]
        landmark, = fl.elements_to_landmarks(elements)
        self.assertAlmostEqual(landmark["lat"], 51.5)
        self.assertEqual(landmark["osm_id"], "way/5")

    def test_way_without_center_is_dropped(self):
        elements = [{"type": "way", "id": 6, "tags": {"man_made": "tower"}}]
        self.assertEqual(fl.elements_to_landmarks(elements), [])

    def test_duplicate_ids_collapse(self):
        element = {"type": "node", "id": 7, "lat": 1.0, "lon": 1.0,
                   "tags": {"man_made": "mast"}}
        self.assertEqual(len(fl.elements_to_landmarks([element, dict(element)])), 1)

    def test_name_falls_back_to_ref(self):
        elements = [{"type": "node", "id": 8, "lat": 1.0, "lon": 1.0,
                     "tags": {"man_made": "mast", "ref": "KXYZ"}}]
        self.assertEqual(fl.elements_to_landmarks(elements)[0]["name"], "KXYZ")


class QueryTest(unittest.TestCase):
    def test_bbox_is_emitted_south_west_north_east(self):
        # Overpass orders its bbox filter (south, west, north, east), the
        # transpose of the (west, south, east, north) used everywhere else in
        # this pipeline. Reversing it silently queries the wrong hemisphere.
        query = fl.build_query((6.0, 46.2, 7.4, 46.9))
        self.assertIn("(46.2,6.0,46.9,7.4)", query)

    def test_query_covers_both_nodes_and_ways(self):
        query = fl.build_query((0.0, 0.0, 1.0, 1.0))
        self.assertIn('node["natural"="peak"]', query)
        self.assertIn('way["man_made"="mast"]', query)

    def test_out_center_requested_for_ways(self):
        self.assertIn("out center tags;", fl.build_query((0.0, 0.0, 1.0, 1.0)))


class SplitBboxTest(unittest.TestCase):
    def test_small_bbox_is_one_cell(self):
        cells = list(fl._split_bbox((0.0, 0.0, 0.5, 0.5), max_span_deg=0.75))
        self.assertEqual(len(cells), 1)

    def test_large_bbox_is_gridded(self):
        cells = list(fl._split_bbox((0.0, 0.0, 3.0, 2.0), max_span_deg=1.0))
        self.assertEqual(len(cells), 6)

    def test_cells_tile_the_bbox_exactly(self):
        bbox = (0.0, 0.0, 3.0, 2.0)
        cells = list(fl._split_bbox(bbox, max_span_deg=1.0))
        self.assertAlmostEqual(min(c[0] for c in cells), bbox[0])
        self.assertAlmostEqual(min(c[1] for c in cells), bbox[1])
        self.assertAlmostEqual(max(c[2] for c in cells), bbox[2])
        self.assertAlmostEqual(max(c[3] for c in cells), bbox[3])


if __name__ == "__main__":
    unittest.main()
