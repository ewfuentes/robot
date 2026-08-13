"""Tests for trim_landmark_feather.

The rules encode judgements about what is visible from a vessel, so the tests
are written as those judgements: a container crane survives without a height
tag, a bench does not, and a bare shed does not while a grain terminal does.
"""

import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from experimental.overhead_matching.swag.scripts import trim_landmark_feather as tlf


def square(lon, lat, side_m):
    """Axis-aligned square footprint of roughly side_m x side_m.

    A degree of longitude is only ~82 km at Boston, so the lon and lat steps
    differ; using one step for both makes a rectangle, not a square.
    """
    import math
    d_lat = side_m / 110574.0
    d_lon = side_m / (111320.0 * math.cos(math.radians(lat)))
    return Polygon([(lon, lat), (lon + d_lon, lat),
                    (lon + d_lon, lat + d_lat), (lon, lat + d_lat)])


def drops(tags: list[dict], areas=None, min_area=2000.0, min_levels=6.0):
    areas = np.zeros(len(tags)) if areas is None else np.asarray(areas)
    masks = tlf.evaluate_rules(tags, areas, min_area, min_levels)
    return {name: mask.tolist() for name, mask in masks.items()}


class RuleTest(unittest.TestCase):
    def test_untagged_row_dropped(self):
        self.assertEqual(drops([{}])["no_harbor_tags"], [True])

    def test_street_furniture_dropped(self):
        for tag in [{"amenity": "bench"}, {"natural": "tree"},
                    {"man_made": "surveillance"}, {"highway": "crossing"},
                    {"highway": "footway"}, {"amenity": "waste_basket"},
                    {"barrier": "fence"}, {"power": "pole"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} should be unobservable from a vessel")

    def test_navigation_aids_and_structures_kept(self):
        for tag in [{"seamark:type": "buoy_lateral", "name": "Buoy 12"},
                    {"man_made": "lighthouse"},
                    {"man_made": "crane"},          # no height tag: still 50-80 m
                    {"object_class": "BCNLAT"},
                    {"place": "island", "name": "Georges Island"},
                    {"historic": "fort"},
                    {"natural": "cliff"},
                    {"amenity": "ferry_terminal"},
                    {"man_made": "water_tower"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_name_rescues_a_soft_unobservable_tag(self):
        """Regression: Bunker Hill Monument is a 67 m obelisk whose only OSM
        tags are a name and tourism=information. A name outranks a weak tag."""
        result = drops([{"name": "Bunker Hill Monument",
                         "tourism": "information"}])
        self.assertFalse(any(v[0] for v in result.values()), result)

    def test_unnamed_information_board_still_dropped(self):
        self.assertEqual(drops([{"tourism": "information"}])["unobservable_only"],
                         [True])

    def test_name_does_not_rescue_a_hard_unobservable_tag(self):
        """Regression: named bus stops are real names on invisible objects, and
        admitting them put 11,357 roads and stops into the catalog."""
        for tag in [{"highway": "bus_stop", "name": "Dorchester Ave @ Dix St",
                     "operator": "MBTA"},
                    {"highway": "traffic_signals",
                     "name": "Thomas F. Kennedy Square"},
                    {"amenity": "parking", "name": "Garage A"},
                    {"amenity": "post_box", "name": "Post Box 12"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} must not be rescued by its name")

    def test_tenant_businesses_dropped_even_when_named(self):
        """A restaurant occupies a building rather than being one; the host
        building is its own row, so the tenant adds nothing at range."""
        for tag in [{"amenity": "restaurant", "name": "Legal Sea Foods"},
                    {"amenity": "cafe", "name": "Thinking Cup"},
                    {"amenity": "bank", "name": "Santander"},
                    {"amenity": "fountain", "name": "Rings Fountain"},
                    {"leisure": "pitch", "name": "Field 3"},
                    {"landuse": "grass"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} is not far-field visible")

    def test_civic_and_waterfront_amenities_kept(self):
        """Values that name a whole building or a waterfront structure stay."""
        for tag in [{"amenity": "theatre", "name": "Leader Bank Pavilion"},
                    {"amenity": "school", "name": "Hull High School"},
                    {"amenity": "ferry_terminal"},
                    {"amenity": "place_of_worship", "name": "Old North Church"},
                    {"tourism": "hotel", "name": "Boston Harbor Hotel"},
                    {"tourism": "museum", "name": "ICA"},
                    {"leisure": "marina"},
                    {"leisure": "slipway"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_zoning_class_needs_a_name(self):
        """Regression: `landuse=residential; name=Harbor Towers` is a labelled
        positive, while an unnamed residential polygon is not a landmark."""
        self.assertFalse(
            any(v[0] for v in drops([{"landuse": "residential",
                                      "name": "Harbor Towers"}]).values()))
        self.assertEqual(
            drops([{"landuse": "residential"}])["unobservable_only"], [True])

    def test_rail_infrastructure_dropped_even_when_named(self):
        """Track and MBTA subway stops have no silhouette from the water, and
        every one of them is named."""
        for tag in [{"railway": "rail", "name": "Dorchester Branch",
                     "operator": "MBTA"},
                    {"railway": "station", "name": "Mattapan",
                     "operator": "MBTA"},
                    {"railway": "subway", "name": "Red Line"},
                    {"railway": "subway_entrance", "name": "Andrew"},
                    {"railway": "platform"}, {"railway": "abandoned"}]:
            with self.subTest(tag=tag):
                self.assertEqual(drops([tag])["unobservable_only"], [True],
                                 f"{tag} is not far-field visible")

    def test_railyards_and_rail_bridges_kept(self):
        """The two rail things that do read from the water."""
        for tag in [{"landuse": "railway", "name": "Southampton Street Yard",
                     "operator": "Amtrak"},
                    {"landuse": "railway"},               # unnamed yard
                    {"railway": "yard"}, {"railway": "depot"},
                    {"railway": "service_station"},
                    {"railway": "rail", "bridge": "yes",
                     "name": "Dorchester Branch"},
                    {"railway": "rail", "bridge": "yes"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} must survive: {result}")

    def test_bridges_are_kept_named_or_not(self):
        """Bridges survive despite carrying a hard-blocked highway tag."""
        for tag in [{"bridge": "yes", "highway": "footway"},
                    {"bridge": "yes", "highway": "motorway",
                     "name": "Maurice J. Tobin Memorial Bridge"},
                    {"man_made": "bridge", "name": "Tobin Bridge"},
                    {"bridge": "viaduct", "highway": "primary"},
                    {"object_class": "BRIDGE", "man_made": "bridge"}]:
            with self.subTest(tag=tag):
                result = drops([tag])
                self.assertFalse(any(v[0] for v in result.values()),
                                 f"{tag} is a bridge and must survive: {result}")

    def test_highway_with_structure_survives(self):
        """A bridge carrying a road is still a bridge."""
        result = drops([{"highway": "primary", "man_made": "bridge",
                         "name": "Tobin Bridge"}])
        self.assertFalse(any(v[0] for v in result.values()))

    def test_generic_small_building_dropped(self):
        self.assertEqual(
            drops([{"building": "yes"}], areas=[200.0])["generic_small_building"],
            [True])

    def test_building_survives_on_its_own_merits(self):
        cases = [
            ({"building": "yes", "name": "Custom House Tower"}, 200.0),
            ({"building": "yes", "height": "150"}, 200.0),
            ({"building": "yes", "building:levels": "40"}, 200.0),
            ({"building": "commercial"}, 200.0),          # landmark use
            ({"building": "yes"}, 9000.0),                # big footprint
            ({"building": "yes", "man_made": "chimney"}, 200.0),
        ]
        for tag, area in cases:
            with self.subTest(tag=tag, area=area):
                self.assertEqual(
                    drops([tag], areas=[area])["generic_small_building"], [False],
                    f"{tag} @ {area} m2 should survive")

    def test_numeric_parsing_tolerates_osm_noise(self):
        self.assertEqual(tlf._numeric("12 m"), 12.0)
        self.assertEqual(tlf._numeric("3;4"), 3.0)
        self.assertEqual(tlf._numeric("about ten"), 0.0)
        self.assertEqual(tlf._numeric(None), 0.0)

    def test_rules_are_independent(self):
        """Every row gets a verdict from every rule, so counts stay auditable."""
        tags = [{}, {"amenity": "bench"}, {"building": "yes"},
                {"man_made": "lighthouse"}]
        masks = tlf.evaluate_rules(tags, np.zeros(4), 2000.0, 6.0)
        self.assertEqual(set(masks), {"no_harbor_tags", "unobservable_only",
                                      "generic_small_building"})
        for mask in masks.values():
            self.assertEqual(len(mask), 4)


class ColumnSelectionTest(unittest.TestCase):
    def test_harbor_columns_match_prune_harbor_tags(self):
        """Pre-selecting columns must not change what prune_harbor_tags keeps."""
        columns = ["id", "geometry", "landmark_type", "name", "man_made",
                   "addr:housenumber", "seamark:type", "payment:cash",
                   "building:levels", "name:fr", "opening_hours", "height"]
        selected = set(tlf.harbor_tag_columns(columns))
        row = {c: "x" for c in columns if c not in ("id", "geometry",
                                                    "landmark_type")}
        direct = set(tlf.hc.prune_harbor_tags(row))
        self.assertEqual(selected, direct)

    def test_footprint_area_of_polygon(self):
        gdf = gpd.GeoDataFrame(
            {"id": ["a", "b", "c"]},
            geometry=[square(-71.0, 42.3, 100.0), Point(-71.0, 42.3),
                      LineString([(-71.0, 42.3), (-70.99, 42.3)])],
            crs="EPSG:4326")
        areas = tlf.footprint_area_m2(gdf)
        self.assertAlmostEqual(areas[0], 100.0 * 100.0, delta=2000.0)
        self.assertEqual(areas[1], 0.0)
        self.assertEqual(areas[2], 0.0)


if __name__ == "__main__":
    unittest.main()
