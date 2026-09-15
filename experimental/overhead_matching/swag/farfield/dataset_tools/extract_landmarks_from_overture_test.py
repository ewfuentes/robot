import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from experimental.overhead_matching.swag.farfield.catalog import catalog, schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    extract_landmarks_from_overture as overture,
)
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    source_publication,
)

BBOX = (129.0, 35.5, 130.0, 36.5)


def _place(uid, name, hierarchy, confidence, lon=129.5, lat=36.0,
           brand=None, common=None):
    return {
        "id": uid,
        "geometry": shapely.Point(lon, lat).wkb,
        "names": {"primary": name, "common": common},
        "categories": {"primary": hierarchy[-1] if hierarchy else None,
                       "alternate": None},
        "taxonomy": {"primary": hierarchy[-1] if hierarchy else None,
                     "hierarchy": hierarchy or None},
        "brand": {"wikidata": None,
                  "names": {"primary": brand, "common": None}},
        "confidence": confidence,
        "sources": [{"dataset": "meta", "provider": "meta",
                     "license": "CDLA-Permissive-2.0", "record_id": "1",
                     "update_time": "2026-08-10T00:00:00Z", "property": ""}],
        "version": 3,
    }


SCHEMA = pa.schema([
    ("id", pa.string()),
    ("geometry", pa.binary()),
    ("names", pa.struct([("primary", pa.string()),
                         ("common", pa.map_(pa.string(), pa.string()))])),
    ("categories", pa.struct([("primary", pa.string()),
                              ("alternate", pa.list_(pa.string()))])),
    ("taxonomy", pa.struct([("primary", pa.string()),
                            ("hierarchy", pa.list_(pa.string()))])),
    ("brand", pa.struct([
        ("wikidata", pa.string()),
        ("names", pa.struct([("primary", pa.string()),
                             ("common", pa.map_(pa.string(), pa.string()))])),
    ])),
    ("confidence", pa.float64()),
    ("sources", pa.list_(pa.struct([
        ("dataset", pa.string()), ("provider", pa.string()),
        ("license", pa.string()), ("record_id", pa.string()),
        ("update_time", pa.string()), ("property", pa.string())]))),
    ("version", pa.int32()),
])


def write_parquet(path: Path, places: list[dict]) -> Path:
    pq.write_table(pa.Table.from_pylist(places, schema=SCHEMA), path)
    return path


PLACES = [
    _place("a", "코모도 호텔", ["lodging", "hotel"], 0.9, brand="Commodore",
           common=[("en", "Commodore Hotel")]),
    _place("b", "포항물회", ["food_and_drink", "restaurant",
                          "korean_restaurant"], 0.7),
    _place("c", "  ", ["lodging", "hotel"], 0.9),
    _place("d", "무명", [], 0.9),
    _place("e", "저신뢰", ["shopping", "convenience_store"], 0.2),
    _place("f", "먼곳", ["shopping", "convenience_store"], 0.9, lon=131.0),
    _place("g", "신분류", ["brand_new_root", "thing"], 0.9),
]


class ExtractLandmarksFromOvertureTest(unittest.TestCase):

    def test_mapping_prefers_leaf_then_root(self):
        self.assertEqual(overture.tags_for_hierarchy(["lodging", "hotel"]),
                         ({"tourism": "hotel"}, False))
        self.assertEqual(
            overture.tags_for_hierarchy(
                ["food_and_drink", "restaurant", "korean_restaurant"]),
            ({"amenity": "restaurant"}, True))
        self.assertEqual(overture.tags_for_hierarchy(["nope", "x"]),
                         (None, False))
        self.assertEqual(overture.tags_for_hierarchy([]), (None, False))

    def test_every_mapped_tag_survives_far_field_pruning(self):
        # A mapped key outside the keep vocabulary would leave the row with
        # nothing but its name once loaded, which is allowed (OSM shops behave
        # the same) but must be a deliberate choice: list them here.
        name_only = {"shop", "office"}
        for leaf, tags in {**overture.ROOT_TAGS, **overture.LEAF_TAGS}.items():
            kept = {key for key in tags if catalog.keeps_tag_key(key)}
            with self.subTest(leaf=leaf):
                self.assertTrue(kept or set(tags) <= name_only, tags)

    def test_extract_maps_filters_and_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = write_parquet(Path(tmp) / "p.parquet", PLACES)
            frame = overture.main(parquet, "2026-08-19.0", BBOX, 0.5,
                                  Path(tmp) / "out" / "overture_v1")
            tags = schema.tag_dicts(frame)
            self.assertEqual(list(frame["id"]),
                             ["overture:a", "overture:b"])
            self.assertEqual(list(frame["landmark_type"]),
                             ["overture", "overture"])
            self.assertEqual(tags[0]["name"], "코모도 호텔")
            self.assertEqual(tags[0]["name:en"], "Commodore Hotel")
            self.assertEqual(tags[0]["brand"], "Commodore")
            self.assertEqual(tags[0]["tourism"], "hotel")
            self.assertEqual(tags[0]["overture:hierarchy"], "lodging/hotel")
            self.assertEqual(tags[0]["overture:confidence"], "0.9000")
            self.assertEqual(
                json.loads(tags[0]["overture:sources"])[0]["license"],
                "CDLA-Permissive-2.0")
            self.assertEqual(tags[1]["amenity"], "restaurant")
            self.assertNotIn("brand", tags[1])

            sidecar = source_publication.output_paths(
                Path(tmp) / "out" / "overture_v1")[1]
            document = json.loads(sidecar.read_text())
            self.assertEqual(document["report"]["rows_in"], 7)
            self.assertEqual(document["report"]["rows_out"], 2)
            self.assertEqual(document["report"]["dropped"], {
                "low_confidence": 1, "no_name": 1, "no_taxonomy": 1,
                "outside_bbox": 1, "unmapped_root": 1})
            self.assertEqual(document["report"]["leaf_fallbacks"],
                             {"korean_restaurant": 1})
            self.assertEqual(document["arguments"]["release"], "2026-08-19.0")

            again = overture.main(parquet, "2026-08-19.0", BBOX, 0.5,
                                  Path(tmp) / "out" / "overture_v1")
            self.assertEqual(list(again["id"]), list(frame["id"]))
            with self.assertRaises(ValueError):
                overture.main(parquet, "2026-08-19.0", BBOX, 0.6,
                              Path(tmp) / "out" / "overture_v1")

    def test_rejects_output_stem_with_a_dot(self):
        # "overture_2026-08-19.0_v1" would otherwise be written as
        # "overture_2026-08-19.feather" by Path.with_suffix.
        with tempfile.TemporaryDirectory() as tmp:
            parquet = write_parquet(Path(tmp) / "p.parquet", PLACES)
            with self.assertRaisesRegex(ValueError, "must not contain a dot"):
                overture.main(parquet, "2026-08-19.0", BBOX, 0.5,
                              Path(tmp) / "overture_2026-08-19.0_v1")

    def test_requires_taxonomy_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = pa.Table.from_pylist(PLACES[:1], schema=SCHEMA)
            path = Path(tmp) / "old.parquet"
            pq.write_table(table.drop_columns(["taxonomy"]), path)
            with self.assertRaisesRegex(ValueError, "taxonomy"):
                overture.read_places(path)


if __name__ == "__main__":
    unittest.main()
