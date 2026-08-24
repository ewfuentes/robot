import json
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import shapely

from experimental.overhead_matching.swag.farfield.catalog import schema


def valid_frame():
    return schema.build_frame(
        ids=["('node', 1)", "('way', 2)"],
        geometries=[
            shapely.Point(-71.05, 42.35),
            shapely.LineString([(-71.04, 42.35), (-71.03, 42.36)]),
        ],
        landmark_types=["osm", "enc"],
        tags=[
            {"name": "Boston Light", "man_made": "lighthouse"},
            {},
        ],
    )


class CompactSchemaTest(unittest.TestCase):
    def test_build_frame_uses_canonical_json_tags(self):
        frame = valid_frame()
        self.assertEqual(
            frame["tags"].iloc[0],
            '{"man_made":"lighthouse","name":"Boston Light"}')
        self.assertEqual(
            schema.tag_dicts(frame)[0],
            {"man_made": "lighthouse", "name": "Boston Light"})
        self.assertEqual(
            list(frame.columns), ["id", "geometry", "landmark_type", "tags"])

    def test_build_and_read_feather_round_trip(self):
        frame = valid_frame()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.feather"
            frame.to_feather(path)
            back = schema.read_frame(path)
        self.assertEqual(schema.tag_dicts(back), schema.tag_dicts(frame))
        self.assertEqual(back.crs.to_epsg(), 4326)
        self.assertEqual(back.geometry.iloc[0].x, -71.05)

    def test_projected_catalog_with_known_crs_is_valid(self):
        projected = valid_frame().to_crs("EPSG:3857")
        self.assertEqual(len(schema.tag_dicts(projected)), 2)

    def test_summarize_names_owned_schema_version(self):
        summary = schema.summarize(valid_frame())
        self.assertIn("compact JSON-tags schema v1", summary)
        self.assertIn("2 tag values", summary)


class InvalidFrameTest(unittest.TestCase):
    def test_normal_reader_rejects_legacy_wide_schema_actionably(self):
        wide = valid_frame().drop(columns=["tags"])
        wide["name"] = ["Boston Light", None]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.feather"
            wide.to_feather(path)
            with self.assertRaisesRegex(
                    schema.CatalogSchemaError,
                    "wide landmark schema.*JSON 'tags'.*convert"):
                schema.read_frame(path)

    def test_permits_optional_object_class_with_exact_tag_mirror(self):
        frame = valid_frame()
        frame["object_class"] = [None, "LNDMRK"]
        frame.at[1, "tags"] = '{"object_class":"LNDMRK"}'
        self.assertEqual(schema.tag_dicts(frame)[1]["object_class"], "LNDMRK")

    def test_rejects_unexpected_columns(self):
        frame = valid_frame()
        frame["review_note"] = ["one", "two"]
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "unexpected columns.*review_note"):
            schema.tag_dicts(frame)

    def test_rejects_conflicting_optional_structural_tag(self):
        frame = valid_frame()
        frame["object_class"] = [None, "LNDMRK"]
        frame.at[1, "tags"] = '{"object_class":"BOYLAT"}'
        with self.assertRaisesRegex(
                schema.CatalogSchemaError,
                "tag 'object_class' does not match.*LNDMRK"):
            schema.tag_dicts(frame)

    def test_rejects_missing_structural_column(self):
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "missing required columns.*id"):
            schema.tag_dicts(valid_frame().drop(columns=["id"]))

    def test_rejects_unknown_crs(self):
        frame = gpd.GeoDataFrame({
            "id": ["one"],
            "geometry": [shapely.Point(0, 0)],
            "landmark_type": ["osm"],
            "tags": ["{}"],
        }, geometry="geometry")
        with self.assertRaisesRegex(schema.CatalogSchemaError, "CRS is missing"):
            schema.tag_dicts(frame)

    def test_rejects_null_and_duplicate_ids(self):
        frame = valid_frame()
        frame.loc[0, "id"] = None
        with self.assertRaisesRegex(schema.CatalogSchemaError, "id is null"):
            schema.tag_dicts(frame)

        frame = valid_frame()
        frame.loc[1, "id"] = frame.loc[0, "id"]
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "ids must be unique"):
            schema.tag_dicts(frame)

    def test_rejects_unknown_or_missing_source(self):
        for source in ("other", None, "OSM"):
            with self.subTest(source=source):
                frame = valid_frame()
                frame.loc[0, "landmark_type"] = source
                with self.assertRaisesRegex(
                        schema.CatalogSchemaError,
                        "exactly 'osm' or 'enc'"):
                    schema.tag_dicts(frame)

    def test_rejects_null_empty_and_invalid_geometry(self):
        frame = valid_frame()
        frame.loc[0, "geometry"] = None
        with self.assertRaisesRegex(schema.CatalogSchemaError, "geometry is null"):
            schema.tag_dicts(frame)

        frame = valid_frame()
        frame.loc[0, "geometry"] = shapely.Point()
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "geometry is empty"):
            schema.tag_dicts(frame)

        frame = valid_frame()
        frame.loc[0, "geometry"] = shapely.Polygon(
            [(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "geometry is invalid"):
            schema.tag_dicts(frame)


class StrictTagsTest(unittest.TestCase):
    def assert_bad_tags(self, value, pattern):
        frame = valid_frame()
        frame["tags"] = frame["tags"].astype(object)
        frame.at[0, "tags"] = value
        with self.assertRaisesRegex(schema.CatalogSchemaError, pattern):
            schema.tag_dicts(frame)

    def test_rejects_non_json_text_cell(self):
        self.assert_bad_tags({"name": "x"}, "JSON object text")

    def test_rejects_malformed_json(self):
        self.assert_bad_tags('{"name":', "invalid JSON")

    def test_read_frame_tag_error_identifies_file_and_row(self):
        frame = valid_frame()
        frame.at[1, "tags"] = '{"name":'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "malformed.feather"
            frame.to_feather(path)
            with self.assertRaises(schema.CatalogSchemaError) as caught:
                schema.read_frame(path)
        self.assertIn("row 1", str(caught.exception))
        self.assertIn(str(path), str(caught.exception))

    def test_rejects_json_non_object(self):
        self.assert_bad_tags('["name"]', "must decode to a JSON object")

    def test_rejects_non_string_value(self):
        self.assert_bad_tags('{"height":42}', "string value")

    def test_rejects_empty_key(self):
        self.assert_bad_tags('{"":"value"}', "non-empty strings")

    def test_rejects_duplicate_json_key(self):
        self.assert_bad_tags(
            '{"name":"first","name":"second"}', "duplicate key")

    def test_rejects_structural_tag_collision(self):
        for structural in schema.META_COLUMNS:
            with self.subTest(structural=structural):
                self.assert_bad_tags(
                    json.dumps({structural: "shadow"}),
                    "collides with a structural catalog field")

    def test_builder_enforces_strict_tags(self):
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "string value"):
            schema.build_frame(
                ids=["one"],
                geometries=[shapely.Point(0, 0)],
                landmark_types=["osm"],
                tags=[{"height": 42}],
            )
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "collides with a structural"):
            schema.build_frame(
                ids=["one"],
                geometries=[shapely.Point(0, 0)],
                landmark_types=["osm"],
                tags=[{"id": "shadow"}],
            )

    def test_builder_rejects_non_string_key_before_json_coercion(self):
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "tag keys.*strings"):
            schema.build_frame(
                ids=["one"], geometries=[shapely.Point(0, 0)],
                landmark_types=["osm"], tags=[{1: "one"}],
            )

    def test_builder_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(
                schema.CatalogSchemaError, "equal lengths"):
            schema.build_frame(
                ids=["one", "two"],
                geometries=[shapely.Point(0, 0)],
                landmark_types=["osm"],
                tags=[{}],
            )


if __name__ == "__main__":
    unittest.main()
