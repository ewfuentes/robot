"""Focused, hermetic tests for the dedicated far-field OSM writer."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from common.openstreetmap import extract_landmarks_python as elm
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.dataset_tools import (
    extract_landmarks_from_osm as subject,
)


def point(lon: float, lat: float):
    geometry = elm.PointGeometry()
    geometry.coord = elm.Coordinate(lon, lat)
    return geometry


def line(coordinates):
    geometry = elm.LineStringGeometry()
    geometry.coords = [elm.Coordinate(lon, lat) for lon, lat in coordinates]
    return geometry


def polygon(exterior, holes=()):
    geometry = elm.PolygonGeometry()
    geometry.exterior = [
        elm.Coordinate(lon, lat) for lon, lat in exterior
    ]
    geometry.holes = [
        [elm.Coordinate(lon, lat) for lon, lat in ring]
        for ring in holes
    ]
    return geometry


def multipolygon(polygons):
    geometry = elm.MultiPolygonGeometry()
    geometry.polygons = list(polygons)
    return geometry


def feature(osm_type, osm_id: int, geometry, tags: dict):
    return SimpleNamespace(
        osm_type=osm_type,
        osm_id=osm_id,
        geometry=geometry,
        tags=tags,
    )


class ValidationTest(unittest.TestCase):
    def test_bbox_is_explicit_and_ordered(self):
        self.assertEqual(
            subject.validate_bbox((-71.2, 42.2, -70.8, 42.5)),
            (-71.2, 42.2, -70.8, 42.5),
        )
        for bbox in (
            (-70.8, 42.2, -71.2, 42.5),
            (-71.2, 42.5, -70.8, 42.2),
            (-181, 42.2, -70.8, 42.5),
            (-71.2, -91, -70.8, 42.5),
            (-71.2, 42.2, float("nan"), 42.5),
            (-71.2, 42.2, -70.8),
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                subject.validate_bbox(bbox)

    def test_node_margin_requires_explicit_full_or_bounded_mode(self):
        self.assertEqual(subject.validate_node_margin_deg(-1), -1.0)
        self.assertEqual(subject.validate_node_margin_deg(0.25), 0.25)
        for value in (-0.1, float("inf"), True, "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.validate_node_margin_deg(value)

    def test_filters_include_farfield_osm_families(self):
        self.assertTrue({
            "place",
            "seamark:type",
            "man_made",
            "historic",
            "natural",
            "building",
            "bridge",
            "waterway",
        }.issubset(subject.TAG_FILTER_KEYS))


class GeometryConversionTest(unittest.TestCase):
    def test_all_common_geometry_variants(self):
        point_out = subject.create_shapely_geometry(point(-71.0, 42.0))
        self.assertIsInstance(point_out, Point)

        line_out = subject.create_shapely_geometry(
            line([(-71.0, 42.0), (-70.9, 42.1)]))
        self.assertIsInstance(line_out, LineString)

        exterior = [
            (-71.0, 42.0),
            (-70.8, 42.0),
            (-70.8, 42.2),
            (-71.0, 42.2),
            (-71.0, 42.0),
        ]
        hole = [
            (-70.95, 42.05),
            (-70.85, 42.05),
            (-70.85, 42.15),
            (-70.95, 42.15),
            (-70.95, 42.05),
        ]
        polygon_geometry = polygon(exterior, [hole])
        polygon_out = subject.create_shapely_geometry(polygon_geometry)
        self.assertIsInstance(polygon_out, Polygon)
        self.assertEqual(len(polygon_out.interiors), 1)

        multi_out = subject.create_shapely_geometry(
            multipolygon([polygon_geometry]))
        self.assertIsInstance(multi_out, MultiPolygon)
        self.assertEqual(len(multi_out.geoms), 1)


class CompactFrameTest(unittest.TestCase):
    def test_source_qualified_ids_and_canonical_complete_tags(self):
        features = [
            feature(
                elm.OsmType.WAY,
                9,
                line([(-71.0, 42.0), (-70.9, 42.1)]),
                {"name": "Pier", "man_made": "pier"},
            ),
            feature(
                elm.OsmType.NODE,
                20,
                point(-70.95, 42.05),
                {
                    "source": "survey",
                    "seamark:type": "beacon_lateral",
                    "name": "Light 4",
                },
            ),
            feature(
                elm.OsmType.NODE,
                3,
                point(-70.96, 42.06),
                {"name": "Harbor Island", "place": "island"},
            ),
        ]

        frame = subject.features_to_geodataframe(features)

        self.assertEqual(list(frame.columns), list(schema.META_COLUMNS))
        self.assertEqual(
            frame["id"].tolist(),
            ["osm:node:3", "osm:node:20", "osm:way:9"],
        )
        self.assertEqual(frame["landmark_type"].tolist(), ["osm"] * 3)
        self.assertEqual(frame.crs.to_epsg(), 4326)
        # schema.build_frame owns canonical JSON ordering and compact encoding.
        self.assertEqual(
            frame.iloc[1]["tags"],
            '{"name":"Light 4","seamark:type":"beacon_lateral",'
            '"source":"survey"}',
        )
        # Source extraction is lossless.  A future trim may drop source=survey,
        # but changing a trim rule never requires re-reading the PBF.
        self.assertEqual(schema.tag_dicts(frame)[1]["source"], "survey")

    def test_invalid_geometry_names_the_source_feature(self):
        empty = elm.MultiPolygonGeometry()
        empty.polygons = []
        with self.assertRaisesRegex(ValueError, "OSM relation 77.*empty"):
            subject.features_to_geodataframe([
                feature(elm.OsmType.RELATION, 77, empty, {"place": "island"})
            ])


class PublicationTest(unittest.TestCase):
    def test_main_writes_compact_feather_and_extraction_provenance(self):
        extracted = feature(
            elm.OsmType.NODE,
            42,
            point(-71.0, 42.0),
            {
                "name": "Outer Light",
                "man_made": "lighthouse",
                "seamark:type": "light_major",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pbf = root / "source.osm.pbf"
            pbf.write_bytes(b"small hermetic stand-in")
            output = root / "catalog"

            with mock.patch.object(
                    subject.elm, "extract_landmarks",
                    return_value=[(subject.REGION_ID, extracted)]) as extract:
                with mock.patch.object(
                        subject.provenance, "git_commit",
                        return_value="abc123"):
                    returned = subject.main(
                        pbf,
                        (-71.2, 41.8, -70.8, 42.2),
                        output,
                        0.5,
                    )

            feather = output.with_suffix(".feather")
            sidecar = output.with_suffix(".provenance.json")
            persisted = schema.read_frame(feather)
            self.assertEqual(returned["id"].tolist(), ["osm:node:42"])
            self.assertEqual(persisted["id"].tolist(), ["osm:node:42"])
            self.assertEqual(list(persisted.columns), list(schema.META_COLUMNS))

            args, _ = extract.call_args
            self.assertEqual(args[0], str(pbf))
            self.assertEqual(set(args[1]), {subject.REGION_ID})
            self.assertEqual(
                (args[1][subject.REGION_ID].left_deg,
                 args[1][subject.REGION_ID].bottom_deg,
                 args[1][subject.REGION_ID].right_deg,
                 args[1][subject.REGION_ID].top_deg),
                (-71.2, 41.8, -70.8, 42.2),
            )
            self.assertTrue(args[2]["place"])
            self.assertTrue(args[2]["seamark:type"])
            self.assertEqual(args[3], 0.5)

            record = json.loads(sidecar.read_text())
            self.assertEqual(record["schema"], subject.PROVENANCE_SCHEMA)
            self.assertEqual(record["git_commit"], "abc123")
            self.assertEqual(
                record["arguments"]["bbox_wgs84"],
                [-71.2, 41.8, -70.8, 42.2],
            )
            self.assertEqual(
                record["arguments"]["geometry_index_mode"],
                "bounded_degraded",
            )
            self.assertEqual(
                record["inputs"]["pbf"]["sha256"],
                hashlib.sha256(pbf.read_bytes()).hexdigest(),
            )
            self.assertEqual(record["diagnostics"]["rows_out"], 1)
            self.assertEqual(record["diagnostics"]["by_osm_type"], {"node": 1})
            self.assertEqual(record["diagnostics"]["seamark_type_rows"], 1)
            self.assertEqual(
                record["output"]["columns"], list(schema.META_COLUMNS))
            self.assertEqual(
                record["output"]["sha256"], subject._sha256(feather))
            self.assertEqual(list(root.glob("*.partial.*")), [])
            self.assertEqual(list(root.glob(".*.partial.*")), [])

    def test_empty_extraction_still_publishes_a_valid_empty_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pbf = root / "source.osm.pbf"
            pbf.write_bytes(b"empty result stand-in")
            output = root / "empty.feather"
            with mock.patch.object(
                    subject.elm, "extract_landmarks", return_value=[]):
                frame = subject.main(
                    pbf, (-71.2, 41.8, -70.8, 42.2), output, -1)

            self.assertEqual(len(frame), 0)
            persisted = schema.read_frame(output)
            self.assertEqual(len(persisted), 0)
            record = json.loads(
                output.with_suffix(".provenance.json").read_text())
            self.assertEqual(record["arguments"]["geometry_index_mode"], "full")
            self.assertEqual(record["diagnostics"]["rows_out"], 0)
            self.assertIsNone(
                record["diagnostics"]["geometry_bounds_wgs84"])


if __name__ == "__main__":
    unittest.main()
