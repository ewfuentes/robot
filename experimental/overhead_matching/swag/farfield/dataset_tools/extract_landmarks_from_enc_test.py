"""Tests for extract_landmarks_from_enc (hermetic — synthetic layers + the
real S-57 enum tables shipped inside the pyogrio wheel)."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import geopandas as gpd
from shapely.geometry import LineString, Point

from experimental.overhead_matching.swag.farfield.catalog import (
    catalog as catalog_lib,
)
from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    download_enc_cells,
)
from experimental.overhead_matching.swag.farfield.catalog import (
    schema as ls,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    extract_landmarks_from_enc as ele,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    feather_utils,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)


def make_layer(rows: list, geoms: list) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    if "LNAM" not in frame.columns:
        frame["LNAM"] = [f"FAKE{i:04d}" for i in range(len(frame))]
    return frame


class EnumTableTest(unittest.TestCase):
    """Decode against the REAL GDAL tables so mapping keys stay in sync."""

    @classmethod
    def setUpClass(cls):
        cls.enums = ele.load_s57_enum_tables()

    def test_known_meanings(self):
        self.assertEqual(ele.decode_enum(self.enums, "COLOUR", "3"), ["red"])
        self.assertEqual(ele.decode_enum(self.enums, "COLOUR", "3,1"),
                         ["red", "white"])
        self.assertEqual(ele.decode_enum(self.enums, "CATLMK", 17),
                         ["tower"])
        self.assertEqual(ele.decode_enum(self.enums, "CATSIL", 4.0),
                         ["water tower"])
        self.assertEqual(ele.decode_enum(self.enums, "CONVIS", 1),
                         ["visual conspicuous"])

    def test_missing_and_unknown_values(self):
        self.assertEqual(ele.decode_enum(self.enums, "COLOUR", None), [])
        self.assertEqual(ele.decode_enum(self.enums, "COLOUR", float("nan")),
                         [])
        self.assertEqual(ele.decode_enum(self.enums, "COLOUR", "9999"), [])

    def test_mapping_tables_match_real_meanings(self):
        """Every key in the CAT*->tags dicts must be an actual catalog
        meaning, so a GDAL table revision that renames meanings fails loudly
        here."""
        for acronym, table in [("CATLMK", ele.CATLMK_TO_TAGS),
                               ("CATSLC", ele.CATSLC_TO_TAGS),
                               ("CATSIL", ele.CATSIL_TO_TAGS),
                               ("CATHAF", ele.CATHAF_TO_TAGS)]:
            real_meanings = set(self.enums[acronym].values())
            for key in table:
                self.assertIn(key, real_meanings, f"{acronym}: {key}")


class MapClassRowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enums = ele.load_s57_enum_tables()

    def test_lateral_buoy(self):
        tags = ele.map_class_row("BOYLAT", {
            "OBJNAM": "Boston Main Channel Lighted Buoy 12",
            "BOYSHP": 4, "CATLAM": 2, "COLOUR": "3",
        }, self.enums)
        self.assertEqual(tags["man_made"], "buoy")
        self.assertEqual(tags["seamark:type"], "buoy_lateral")
        self.assertEqual(tags["colour"], "red")
        self.assertEqual(tags["name"],
                         "Boston Main Channel Lighted Buoy 12")
        self.assertIn("pillar buoy", tags["description"])
        self.assertIn("starboard-hand lateral mark", tags["description"])

    def test_landmark_tower_conspicuous(self):
        tags = ele.map_class_row("LNDMRK", {
            "CATLMK": "17", "CONVIS": 1, "HEIGHT": 34.0,
        }, self.enums)
        self.assertEqual(tags["man_made"], "tower")
        self.assertEqual(tags["seamark:type"], "landmark")
        self.assertEqual(tags["height"], "34")
        self.assertEqual(tags["description"], "visually conspicuous")

    def test_landmark_light_support_becomes_lighthouse(self):
        functn_codes = [code for code, meaning in self.enums["FUNCTN"].items()
                        if "light support" in meaning]
        self.assertEqual(len(functn_codes), 1)
        tags = ele.map_class_row(
            "LNDMRK", {"FUNCTN": str(functn_codes[0])}, self.enums)
        self.assertEqual(tags["man_made"], "lighthouse")

    def test_unnamed_light_skipped_named_kept(self):
        self.assertIsNone(
            ele.map_class_row("LIGHTS", {"HEIGHT": 4.0}, self.enums))
        tags = ele.map_class_row(
            "LIGHTS", {"OBJNAM": "Long Island Head Light"}, self.enums)
        self.assertEqual(tags["man_made"], "lighthouse")

    def test_slcons_categories(self):
        pier = ele.map_class_row("SLCONS", {"CATSLC": 4.0}, self.enums)
        self.assertEqual(pier["man_made"], "pier")
        wharf = ele.map_class_row("SLCONS", {"CATSLC": "6"}, self.enums)
        self.assertEqual(wharf["man_made"], "quay")
        # rip rap (8) and category-less shoreline construction are skipped
        self.assertIsNone(
            ele.map_class_row("SLCONS", {"CATSLC": 8.0}, self.enums))
        self.assertIsNone(ele.map_class_row("SLCONS", {}, self.enums))

    def test_buisgl_kept_only_when_named_or_conspicuous(self):
        self.assertIsNone(ele.map_class_row("BUISGL", {}, self.enums))
        named = ele.map_class_row("BUISGL", {"OBJNAM": "Custom House"},
                                  self.enums)
        self.assertEqual(named["building"], "yes")
        conspicuous = ele.map_class_row("BUISGL", {"CONVIS": 1}, self.enums)
        self.assertEqual(conspicuous["building"], "yes")

    def test_daymark_and_water_tower(self):
        daymark = ele.map_class_row("DAYMAR", {}, self.enums)
        self.assertEqual(daymark["man_made"], "beacon")
        self.assertEqual(daymark["seamark:type"], "daymark")
        tank = ele.map_class_row("SILTNK", {"CATSIL": "4"}, self.enums)
        self.assertEqual(tank["man_made"], "water_tower")

    def test_crane_category_lands_in_description_not_a_pruned_tag(self):
        """`crane:` is not a kept prefix in the far-field vocabulary, so a
        crane:type tag would be silently pruned at catalog load; the category
        must travel in `description` instead."""
        crane = ele.map_class_row("CRANES", {"CATCRN": "2"}, self.enums)
        self.assertEqual(crane["man_made"], "crane")
        self.assertNotIn("crane:type", crane)
        self.assertIn("description", crane)
        self.assertFalse(catalog_lib.keeps_tag_key("crane:type"))

    def test_toponym_inference(self):
        self.assertEqual(ele.tags_from_toponym("Georges Island"),
                         [("place", "island")])
        self.assertEqual(ele.tags_from_toponym("Nantasket Beach"),
                         [("natural", "beach")])
        self.assertEqual(ele.tags_from_toponym("Moon Head"),
                         [("natural", "cape")])
        self.assertEqual(ele.tags_from_toponym("Atlantic Hill"),
                         [("natural", "peak")])
        self.assertIsNone(ele.tags_from_toponym("Boston Common"))
        self.assertIsNone(ele.tags_from_toponym("Rowes Wharf"))
        # ENC does carry bare generic names (one Boston HRBFAC is just
        # "Yacht Club"), and matching on them is still correct.
        self.assertEqual(ele.tags_from_toponym("Island "),
                         [("place", "island")])

    def test_land_areas_and_regions(self):
        # Unnamed land is the mainland and bare rocks -- always skipped.
        self.assertIsNone(ele.map_class_row("LNDARE", {}, self.enums))
        self.assertIsNone(
            ele.map_class_row("LNDRGN", {"CATLND": "11"}, self.enums))
        # Toponym wins over the per-class default...
        deer = ele.map_class_row("LNDRGN", {"OBJNAM": "Deer Island"},
                                 self.enums)
        self.assertEqual(deer, {"name": "Deer Island", "place": "island"})
        beach = ele.map_class_row("LNDRGN", {"OBJNAM": "Nantasket Beach"},
                                  self.enums)
        self.assertEqual(beach["natural"], "beach")
        self.assertNotIn("place", beach)
        # ...and the defaults differ: a named marine land area is an island,
        # a named land region without a recognized generic term is not.
        self.assertEqual(
            ele.map_class_row("LNDARE", {"OBJNAM": "Great Brewster"},
                              self.enums)["place"],
            "island")
        self.assertEqual(
            ele.map_class_row("LNDRGN", {"OBJNAM": "Boston Common"},
                              self.enums)["place"],
            "locality")

    def test_harbour_facilities(self):
        marina = ele.map_class_row(
            "HRBFAC", {"OBJNAM": "Hull Yacht Club", "CATHAF": "5"},
            self.enums)
        self.assertEqual(marina,
                         {"name": "Hull Yacht Club", "leisure": "marina"})
        ferry = ele.map_class_row("HRBFAC", {"CATHAF": "3"}, self.enums)
        self.assertEqual(ferry["amenity"], "ferry_terminal")
        yard = ele.map_class_row("HRBFAC", {"CATHAF": "9"}, self.enums)
        self.assertEqual(yard["landuse"], "industrial")
        self.assertEqual(yard["industrial"], "shipyard")
        # No category means nothing usable to match on.
        self.assertIsNone(ele.map_class_row("HRBFAC", {"OBJNAM": "X"},
                                            self.enums))

    def test_all_emitted_keys_survive_the_far_field_vocabulary(self):
        """Every tag key the mapper can emit must survive
        catalog.keeps_tag_key, or the catalog silently prunes it at load
        time. This is the single-vocabulary contract: the ENC extractor and
        the catalog must agree on what a tag key is worth."""
        rows = [
            ("BOYLAT", {"OBJNAM": "B", "BOYSHP": 4, "CATLAM": 1,
                        "COLOUR": "3"}),
            ("BCNLAT", {"COLOUR": "4"}),
            ("DAYMAR", {}),
            ("LIGHTS", {"OBJNAM": "L", "HEIGHT": 10.0}),
            ("LNDMRK", {"CATLMK": "20", "CONVIS": 1}),  # spire -> tower:type
            ("LNDMRK", {"CATLMK": "17"}),               # tower
            ("SLCONS", {"CATSLC": 1.0}),
            ("SILTNK", {"CATSIL": "2"}),
            ("BUISGL", {"OBJNAM": "X", "FUNCTN": "17"}),
            ("BRIDGE", {"CATBRG": "1"}),
            ("CRANES", {"CATCRN": "2"}),
            ("FORSTC", {"CATFOR": "2"}),
            ("LNDARE", {"OBJNAM": "Georges Island"}),
            ("LNDRGN", {"OBJNAM": "Nantasket Beach"}),
            ("LNDRGN", {"OBJNAM": "Boston Common"}),
            ("HRBFAC", {"OBJNAM": "Hull Yacht Club", "CATHAF": "5"}),
            ("HRBFAC", {"CATHAF": "9"}),
        ]
        for object_class, row in rows:
            tags = ele.map_class_row(object_class, row, self.enums)
            self.assertIsNotNone(tags, object_class)
            for key in tags:
                self.assertTrue(
                    catalog_lib.keeps_tag_key(key),
                    f"{object_class} emits key {key!r} that "
                    f"catalog.keeps_tag_key would prune")
            # And the round trip through the catalog's pruner keeps
            # everything.
            self.assertEqual(dict(catalog_lib.prune_far_field_tags(tags)),
                             tags, object_class)


class AssembleFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.enums = ele.load_s57_enum_tables()

    def test_buoy_light_suppressed_fixed_light_kept(self):
        per_class = {
            "BOYLAT": make_layer(
                [{"OBJNAM": "Buoy 2", "CATLAM": 1}], [Point(-70.95, 42.30)]),
            "LIGHTS": make_layer(
                [
                    {"OBJNAM": "Buoy 2 Light"},        # ~5 m from the buoy
                    {"OBJNAM": "Deer Island Light"},   # far from any buoy
                ],
                [Point(-70.95005, 42.30), Point(-70.9545, 42.3396)]),
        }
        features, skipped = ele.assemble_features(per_class, self.enums)
        light_names = [f["tags"].get("name") for f in features
                       if f["object_class"] == "LIGHTS"]
        self.assertEqual(light_names, ["Deer Island Light"])
        self.assertEqual(skipped["LIGHTS: on buoy"], 1)

    def test_lndmrk_near_light_promoted_to_lighthouse(self):
        per_class = {
            "LNDMRK": make_layer(
                [{"OBJNAM": "Deer Island Light", "CONVIS": 1},
                 {"CATLMK": "17"}],
                [Point(-70.9545, 42.3396), Point(-70.90, 42.40)]),
            "LIGHTS": make_layer([{}], [Point(-70.95451, 42.33961)]),
        }
        features, _ = ele.assemble_features(per_class, self.enums)
        by_name = {f["tags"].get("name"): f["tags"] for f in features
                   if f["object_class"] == "LNDMRK"}
        self.assertEqual(by_name["Deer Island Light"]["man_made"],
                         "lighthouse")
        self.assertEqual(by_name[None]["man_made"], "tower")

    def test_exclude_buoys(self):
        per_class = {
            "BOYLAT": make_layer([{"OBJNAM": "Buoy 2"}],
                                 [Point(-70.95, 42.30)]),
            "DAYMAR": make_layer([{}], [Point(-70.96, 42.31)]),
        }
        features, skipped = ele.assemble_features(
            per_class, self.enums, include_buoys=False)
        self.assertEqual([f["object_class"] for f in features], ["DAYMAR"])
        self.assertEqual(skipped["BOYLAT: buoys excluded"], 1)


class FeatherFrameTest(unittest.TestCase):
    def test_output_schema(self):
        features = [
            {"lnam": "0226DAA31A00", "object_class": "BOYLAT",
             "geometry": Point(-70.95, 42.30),
             "tags": {"man_made": "buoy", "name": "Buoy 2"}},
            {"lnam": "0226DAA31B00", "object_class": "SLCONS",
             "geometry": LineString([(-70.96, 42.31), (-70.961, 42.311)]),
             "tags": {"man_made": "pier"}},
        ]
        gdf = ele.features_to_geodataframe(features, "enc")
        self.assertEqual(gdf["id"].tolist(),
                         ["('enc', '0226DAA31A00')",
                          "('enc', '0226DAA31B00')"])
        self.assertEqual(gdf["landmark_type"].tolist(), ["enc", "enc"])
        self.assertNotIn("pruned_props", gdf.columns)
        self.assertEqual(tuple(gdf.columns),
                         (*ls.META_COLUMNS, "object_class"))
        self.assertEqual(str(gdf.crs), "EPSG:4326")
        # Tags live in the `tags` dict column (catalog/schema.py), and a
        # landmark carries only the keys it actually has.
        self.assertIn(ls.TAGS_COLUMN, gdf.columns)
        tag_dicts = ls.tag_dicts(gdf)
        self.assertEqual(tag_dicts[0],
                         {"man_made": "buoy", "name": "Buoy 2",
                          "object_class": "BOYLAT"})
        self.assertEqual(tag_dicts[1],
                         {"man_made": "pier", "object_class": "SLCONS"})
        self.assertNotIn("name", tag_dicts[1])
        self.assertEqual(gdf["object_class"].tolist(), ["BOYLAT", "SLCONS"])

    def test_dedupe_excludes_object_class_from_decoded_tags(self):
        features = [
            {"lnam": "A", "object_class": "DAYMAR",
             "geometry": Point(-70.95, 42.30),
             "tags": {"man_made": "beacon"}},
            {"lnam": "B", "object_class": "BCNLAT",
             "geometry": Point(-70.95, 42.30),
             "tags": {"man_made": "beacon"}},
        ]
        gdf = ele.features_to_geodataframe(features, "enc")
        self.assertEqual(feather_utils.tag_signatures(gdf),
                         [(('man_made', 'beacon'),)] * 2)
        deduped = feather_utils.dedupe_exact_duplicates(
            gdf, tolerance_m=1.0, verbose=False)
        self.assertEqual(len(deduped), 1)

    def test_bbox_filter(self):
        features = [
            {"lnam": "A", "object_class": "BOYLAT",
             "geometry": Point(-70.95, 42.30), "tags": {}},
            {"lnam": "B", "object_class": "BOYLAT",
             "geometry": Point(-70.50, 42.30), "tags": {}},
        ]
        kept = ele.filter_features_to_bbox(features,
                                           (-71.0, 42.2, -70.9, 42.4))
        self.assertEqual([f["lnam"] for f in kept], ["A"])


class PublicationTest(unittest.TestCase):

    def test_main_binds_validated_cell_bytes_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(
                    "ENC_ROOT/US5BOSCD/US5BOSCD.000", b"chart bytes")
            download_enc_cells.download_cell(
                "US5BOSCD", root, fetch_fn=lambda _: buffer.getvalue())
            selection = root / "selection.json"
            download_enc_cells.main(
                cells=["US5BOSCD"], catalog_state=None, bbox=None, band=5,
                output_dir=root, selection_output=selection, force=False,
                fetch_fn=lambda _: self.fail("cached cell should not fetch"))
            layer = make_layer([{"LNAM": "A"}], [Point(-71.0, 42.0)])
            output = root / "enc_v1"
            with mock.patch.object(ele, "load_s57_enum_tables",
                                   return_value={}), mock.patch.object(
                                       ele, "read_cells",
                                       return_value={"DAYMAR": layer}) as read:
                result = ele.main(
                    root, selection, output, bbox=None,
                    include_buoys=True, landmark_type="enc",
                    dedupe_tolerance_m=0.0)
                reused = ele.main(
                    root, selection, output, bbox=None,
                    include_buoys=True, landmark_type="enc",
                    dedupe_tolerance_m=0.0)
                with self.assertRaisesRegex(ValueError,
                                            "provenance differs"):
                    ele.main(
                        root, selection, output, bbox=None,
                        include_buoys=True, landmark_type="enc",
                        dedupe_tolerance_m=1.0)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(reused["id"].tolist(), result["id"].tolist())
            feather, sidecar, staging = source_publication.output_paths(output)
            published = ls.read_frame(feather)
            self.assertEqual(schema_ids := published["id"].tolist(),
                             result["id"].tolist())
            self.assertEqual(schema_ids, ["('enc', 'A')"])
            self.assertEqual(result["object_class"].tolist(), ["DAYMAR"])
            self.assertEqual(published["object_class"].tolist(), ["DAYMAR"])
            self.assertEqual(ls.tag_dicts(published)[0]["object_class"],
                             "DAYMAR")
            record = json.loads(sidecar.read_text())
            self.assertEqual(record["schema"],
                             source_publication.SOURCE_PROVENANCE_SCHEMA)
            self.assertTrue(record["complete"])
            self.assertEqual(record["output_sha256"],
                             artifact.sha256_file(feather))
            self.assertIn("US5BOSCD", record["input_digests"])
            self.assertEqual(record["arguments"]["selection_sha256"],
                             artifact.sha256_file(selection))
            self.assertFalse(staging.exists())
            self.assertEqual(reused["object_class"].tolist(), ["DAYMAR"])
            self.assertEqual(ls.tag_dicts(reused)[0]["object_class"],
                             "DAYMAR")


class BboxFromDatasetTest(unittest.TestCase):
    """bbox_from_dataset reads only the current farfield dataset contract."""

    def test_reads_pipeline_metadata_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            (ds / "pipeline_metadata.json").write_text(json.dumps(
                {"bbox": {"west": -71.1, "south": 42.2, "east": -70.9,
                          "north": 42.4}}))
            self.assertEqual(feather_utils.bbox_from_dataset(ds),
                             (-71.1, 42.2, -70.9, 42.4))

    def test_satellite_bbox_is_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp)
            (ds / "satellite_bbox.json").write_text(json.dumps(
                {"west": -71.1, "south": 42.2, "east": -70.9,
                 "north": 42.4}))
            with self.assertRaises(FileNotFoundError):
                feather_utils.bbox_from_dataset(ds)

    def test_missing_bbox_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                feather_utils.bbox_from_dataset(Path(tmp))

    def test_buffered_bbox_grows_every_side(self):
        self.assertEqual(
            feather_utils.buffered_bbox((0.0, 10.0, 1.0, 12.0), 0.1),
            (-0.1, 9.8, 1.1, 12.2))


class DesignatorTest(unittest.TestCase):
    """A mark's board number must be published as its own `ref` tag.

    91% of the named buoys in the Boston Harbor catalog share their
    designator with another buoy, and same-designator marks sit a median
    14.7 km apart, so the number is a disjunctive constraint and never an
    identity. Leaving it buried inside `name` let a detection named `16` be
    matched to one of three "Buoy 16" rows 19.4 km apart.
    """

    def test_pulls_trailing_designator(self):
        for name, want in [
            ("Hingham Harbor Channel Buoy 16", "16"),
            ("Boston Main Channel Lighted Buoy 5A", "5A"),
            ("Squantum Channel Buoy 1SC", "1SC"),
            ("President Roads Anchorage Buoy C", "C"),
            ("Nantasket Roads Channel Lighted Bell Buoy 3", "3"),
            ("Outer Seal Rock Isolated Danger Buoy DSR", "DSR"),
        ]:
            self.assertEqual(ele.designator_from_name(name), want, name)

    def test_leaves_ordinary_names_alone(self):
        # No trailing designator, so no ref -- never invent one.
        for name in ["Fan Pier South Hazard Lighted Buoy",
                     "Spectacle Island",
                     "New England Aquarium Intake Buoy", "Boston Fish Pier",
                     "", "Buoy"]:
            self.assertIsNone(ele.designator_from_name(name), name)

    def test_common_tags_publishes_ref_alongside_name(self):
        tags, _ = ele._common_tags({"OBJNAM": "Weir River Buoy 11"}, None)
        self.assertEqual(tags["name"], "Weir River Buoy 11")
        self.assertEqual(tags["ref"], "11")

    def test_common_tags_omits_ref_when_there_is_none(self):
        tags, _ = ele._common_tags({"OBJNAM": "Boston Fish Pier"}, None)
        self.assertNotIn("ref", tags)


if __name__ == "__main__":
    unittest.main()
