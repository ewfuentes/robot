import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.swag.farfield.catalog import schema


def dict_frame():
    return pd.DataFrame({
        "id": ["('node', 1)", "('way', 2)"],
        "geometry": [b"", b""],
        "landmark_type": ["osm", "enc"],
        "tags": [json.dumps({"name": "Boston Light", "man_made": "lighthouse"}),
                 json.dumps({})],
    })


def wide_frame():
    return pd.DataFrame({
        "id": ["('node', 1)", "('way', 2)"],
        "geometry": [b"", b""],
        "landmark_type": ["osm", "osm"],
        "name": ["Boston Light", None],
        "man_made": ["lighthouse", None],
        "natural": [None, "peak"],
    })


class LayoutTest(unittest.TestCase):
    def test_layout_detection(self):
        self.assertTrue(schema.is_dict_schema(dict_frame()))
        self.assertFalse(schema.is_dict_schema(wide_frame()))

    def test_tag_dicts_agree_across_layouts(self):
        self.assertEqual(
            schema.tag_dicts(dict_frame())[0],
            {"name": "Boston Light", "man_made": "lighthouse"})
        wide = schema.tag_dicts(wide_frame())
        self.assertEqual(wide[0],
                         {"name": "Boston Light", "man_made": "lighthouse"})
        self.assertEqual(wide[1], {"natural": "peak"})  # nulls dropped

    def test_empty_tags_cell(self):
        self.assertEqual(schema.tag_dicts(dict_frame())[1], {})

    def test_row_dicts_flatten_meta_and_tags(self):
        rows = schema.row_dicts(dict_frame())
        self.assertEqual(rows[0]["landmark_type"], "osm")
        self.assertEqual(rows[0]["name"], "Boston Light")

    def test_widen_round_trips_tag_values(self):
        widened = schema.widen(dict_frame())
        self.assertNotIn(schema.TAGS_COLUMN, widened.columns)
        self.assertEqual(widened["name"][0], "Boston Light")
        self.assertIsNone(widened["name"][1])

    def test_unsupported_tags_cell_raises(self):
        frame = dict_frame()
        frame.loc[0, "tags"] = 42
        with self.assertRaises(TypeError):
            schema.tag_dicts(frame)


class BuildAndReadTest(unittest.TestCase):
    def test_build_frame_feather_round_trip(self):
        import shapely

        frame = schema.build_frame(
            ids=["('node', 1)"],
            geometries=[shapely.Point(-71.05, 42.35)],
            landmark_types=["osm"],
            tags=[{"name": "Custom House Tower", "height": "151"}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v1.feather"
            frame.to_feather(path)
            back = schema.read_frame(path)
        self.assertTrue(schema.is_dict_schema(back))
        self.assertEqual(schema.tag_dicts(back)[0]["name"],
                         "Custom House Tower")
        # Geometry comes back as shapely objects, not WKB bytes.
        self.assertEqual(back.geometry.iloc[0].x, -71.05)

    def test_summarize_names_the_layout(self):
        self.assertIn("dict schema", schema.summarize(dict_frame()))
        self.assertIn("wide schema", schema.summarize(wide_frame()))


if __name__ == "__main__":
    unittest.main()
