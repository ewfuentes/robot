"""Tests for trim_landmark_feather.

The rules encode judgements about what is visible from a vessel, so the tests
are written as those judgements: a container crane survives without a height
tag, a bench does not, and a bare shed does not while a grain terminal does.
"""

import json
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(drops([{}])["no_far_field_tags"], [True])

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
        self.assertEqual(set(masks), {"no_far_field_tags", "unobservable_only",
                                      "generic_small_building"})
        for mask in masks.values():
            self.assertEqual(len(mask), 4)


class ColumnSelectionTest(unittest.TestCase):
    def test_far_field_columns_match_prune_far_field_tags(self):
        """Pre-selecting columns must not change what prune_far_field_tags keeps."""
        columns = ["id", "geometry", "landmark_type", "name", "man_made",
                   "addr:housenumber", "seamark:type", "payment:cash",
                   "building:levels", "name:fr", "opening_hours", "height"]
        selected = set(tlf.far_field_tag_columns(columns))
        row = {c: "x" for c in columns if c not in ("id", "geometry",
                                                    "landmark_type")}
        direct = set(tlf.hc.prune_far_field_tags(row))
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


def catalog(rows) -> gpd.GeoDataFrame:
    """Tiny dict-schema catalog: [(id, tags, geometry), ...]."""
    return gpd.GeoDataFrame(
        {"id": [r[0] for r in rows],
         "landmark_type": ["osm"] * len(rows),
         "tags": [json.dumps(r[1]) for r in rows]},
        geometry=[r[2] for r in rows], crs="EPSG:4326")


def write_matches(path: Path, entries) -> Path:
    """m9's matches.json shape: [(tracklet, signature, confidence)]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        tracklet: {"matches": [{"landmark_id": f"osm:way:{i}",
                                "signature": signature,
                                "confidence": confidence,
                                "match_type": "instance"}]}
        for i, (tracklet, signature, confidence) in enumerate(entries)}))
    return path


class MatchedRecallGuardTest(unittest.TestCase):
    """The guard that asks a stronger question than the positive set: would a
    rule drop something a matcher already chose on a real run?"""

    def test_reads_matches_and_honours_the_confidence_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_matches(Path(tmp) / "run" / "matching" / "matches.json",
                                 [("LT0", "man_made=lighthouse", 0.9),
                                  ("LT1", "amenity=bench", 0.2)])
            found = tlf.matched_signatures([path.parent.parent], 0.5)
        self.assertEqual(sorted(found), ["man_made=lighthouse"])
        self.assertEqual(found["man_made=lighthouse"][0][1], "LT0")

    def test_accepts_the_matches_file_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_matches(Path(tmp) / "matches.json",
                                 [("LT0", "man_made=pier", 0.6)])
            self.assertEqual(sorted(tlf.matched_signatures([path], 0.5)),
                             ["man_made=pier"])

    def test_missing_matches_file_is_an_error_not_a_silent_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                tlf.matched_signatures([Path(tmp)], 0.5)

    def test_refuses_to_write_when_a_matched_signature_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("osm:node:1", {"amenity": "bench"}, Point(-71.0, 42.3))]
                    ).to_feather(source)
            matches = write_matches(tmp / "matches.json",
                                    [("LT0", "amenity=bench", 0.9)])
            with self.assertRaises(SystemExit) as caught:
                tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                         matched_from=[matches])
            self.assertIn("refusing to write", str(caught.exception))
            self.assertFalse((tmp / "v2_trimmed.feather").exists())

    def test_allow_recall_loss_is_the_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("osm:node:1", {"amenity": "bench"}, Point(-71.0, 42.3))]
                    ).to_feather(source)
            matches = write_matches(tmp / "matches.json",
                                    [("LT0", "amenity=bench", 0.9)])
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                     matched_from=[matches], allow_recall_loss=True)
            record = json.loads((tmp / "v2_trimmed.provenance.json").read_text())
        self.assertEqual(record["recall_guard"]["lost_signatures"],
                         ["amenity=bench"])

    def test_signatures_from_another_region_do_not_block_a_write(self):
        """A run from a different dataset matches signatures this table never
        held; that is not a rule dropping something."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("osm:node:1", {"man_made": "lighthouse"},
                      Point(-70.89, 42.32))]).to_feather(source)
            matches = write_matches(tmp / "matches.json",
                                    [("LT0", "natural=peak; name=Mt Adams", 0.9)])
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                     matched_from=[matches])
            record = json.loads((tmp / "v2_trimmed.provenance.json").read_text())
        self.assertEqual(record["recall_guard"]["lost_signatures"], [])

    def test_passes_when_the_matched_signature_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("osm:node:1", {"man_made": "lighthouse",
                                     "name": "Boston Light"},
                      Point(-70.89, 42.32))]).to_feather(source)
            matches = write_matches(tmp / "matches.json",
                                    [("LT0", "man_made=lighthouse; "
                                             "name=Boston Light", 0.9)])
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                     matched_from=[matches])
            record = json.loads((tmp / "v2_trimmed.provenance.json").read_text())
        self.assertEqual(record["recall_guard"]["lost_signatures"], [])
        self.assertEqual(record["rows_out"], 1)


class WriteProtectionTest(unittest.TestCase):
    """A catalog is part of the problem definition, so it is versioned rather
    than overwritten -- every past number was computed against the old one."""

    def build(self, tmp: Path) -> Path:
        source = tmp / "v1.feather"
        catalog([("osm:node:1", {"man_made": "lighthouse"},
                  Point(-70.89, 42.32))]).to_feather(source)
        return source

    def test_existing_catalog_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = self.build(tmp)
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
            with self.assertRaises(SystemExit) as caught:
                tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
        self.assertIn("versioned, not overwritten", str(caught.exception))

    def test_force_replaces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = self.build(tmp)
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False)
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                     force=True)

    def test_dry_run_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = self.build(tmp)
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, True)
            self.assertFalse((tmp / "v2_trimmed.feather").exists())


class RuleFingerprintTest(unittest.TestCase):

    def test_stable_for_the_same_rules(self):
        self.assertEqual(tlf.rule_fingerprint(2000.0, 6.0),
                         tlf.rule_fingerprint(2000.0, 6.0))

    def test_changes_when_a_threshold_changes(self):
        self.assertNotEqual(tlf.rule_fingerprint(2000.0, 6.0),
                            tlf.rule_fingerprint(400.0, 6.0))

    def test_changes_when_a_rule_set_changes(self):
        original = tlf.HARD_UNOBSERVABLE_TAGS
        try:
            tlf.HARD_UNOBSERVABLE_TAGS = frozenset(original | {("amenity", "zzz")})
            changed = tlf.rule_fingerprint(2000.0, 6.0)
        finally:
            tlf.HARD_UNOBSERVABLE_TAGS = original
        self.assertNotEqual(changed, tlf.rule_fingerprint(2000.0, 6.0))


class ClipBoxTest(unittest.TestCase):

    def test_keeps_inside_and_drops_outside(self):
        inside = Point(-71.08, 42.36)
        outside = Point(-71.08, 42.56)          # ~22 km north
        gdf = catalog([("a", {"man_made": "tower"}, inside),
                       ("b", {"man_made": "tower"}, outside)])
        mask = tlf.clip_mask(gdf, 42.36, -71.08, 25.0)
        self.assertEqual(mask.tolist(), [True, False])

    def test_box_is_square_in_metres_not_degrees(self):
        """A degree of longitude is only ~82 km at Boston against ~111 km for a
        degree of latitude, so equal degree offsets are unequal distances: 0.1
        deg is 8.2 km east but 11.1 km north. A box measured in degrees would
        admit or reject the pair together; measured in metres it splits them."""
        east = Point(-71.08 + 0.10, 42.36)     # 8.2 km east
        north = Point(-71.08, 42.36 + 0.10)    # 11.1 km north
        gdf = catalog([("e", {"man_made": "tower"}, east),
                       ("n", {"man_made": "tower"}, north)])
        self.assertEqual(tlf.clip_mask(gdf, 42.36, -71.08, 25.0).tolist(),
                         [True, True])       # half-box 12.5 km: both inside
        self.assertEqual(tlf.clip_mask(gdf, 42.36, -71.08, 20.0).tolist(),
                         [True, False])      # half-box 10 km: only the east one

    def test_clip_without_a_centre_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("a", {"man_made": "tower"}, Point(-71.0, 42.3))]
                    ).to_feather(source)
            with self.assertRaises(SystemExit) as caught:
                tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                         clip_km=25.0)
        self.assertIn("impossible to reproduce", str(caught.exception))

    def test_clip_is_reported_as_its_own_rule_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "v1.feather"
            catalog([("a", {"man_made": "tower"}, Point(-71.08, 42.36)),
                     ("b", {"man_made": "tower"}, Point(-71.08, 42.56))]
                    ).to_feather(source)
            tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                     clip_km=25.0, clip_center_lat=42.36,
                     clip_center_lon=-71.08)
            record = json.loads((tmp / "v2_trimmed.provenance.json").read_text())
        self.assertEqual(record["drops_per_rule"]["outside_clip_box"], 1)
        self.assertEqual(record["rows_out"], 1)
        self.assertEqual(record["arguments"]["clip_km"], 25.0)


class ProvenanceTest(unittest.TestCase):
    """The provenance file has one job: make the catalog reproducible."""

    def run_once(self, tmp: Path):
        source = tmp / "v1.feather"
        catalog([("a", {"man_made": "lighthouse", "name": "Boston Light"},
                  Point(-70.89, 42.32))]).to_feather(source)
        tlf.main(source, tmp / "v2_trimmed", None, 2000.0, 6.0, False,
                 clip_km=25.0, clip_center_lat=42.32, clip_center_lon=-70.89)
        return json.loads((tmp / "v2_trimmed.provenance.json").read_text()), source

    def test_records_every_argument_and_the_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, source = self.run_once(Path(tmp))
            self.assertEqual(record["input_sha256"], tlf.sha256_of(source))
        for key in ("input", "output", "min_building_area_m2",
                    "min_building_levels", "clip_km", "clip_center_lat",
                    "clip_center_lon", "confidence_floor"):
            self.assertIn(key, record["arguments"])

    def test_reproduce_command_carries_the_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, _ = self.run_once(Path(tmp))
        command = record["reproduce"]
        self.assertIn("trim_landmark_feather", command)
        self.assertIn("--clip_km 25.0", command)
        self.assertIn("--min_building_area_m2 2000.0", command)
        self.assertNotIn("None", command)

if __name__ == "__main__":
    unittest.main()
