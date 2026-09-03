import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import shapely
from shapely.geometry import Point, Polygon

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.loci import osm, region


DATASET = "example"


class LociOsmTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.region_bbox = (-71.10, 42.34, -71.08, 42.36)
        self.plan = {
            "schema": region.SCHEMA,
            "bbox_wsen": list(self.region_bbox),
            "grid": region.build_grid(self.region_bbox),
        }
        footprint = self.plan["grid"]["footprint_bbox_wsen"]
        west, south, east, north = footprint
        mid_lat = (south + north) / 2.0
        mid_lon = (west + east) / 2.0
        self.crossing = Polygon([
            (west - 0.05, mid_lat - 0.001),
            (west + 0.000001, mid_lat - 0.001),
            (west + 0.000001, mid_lat + 0.001),
            (west - 0.05, mid_lat + 0.001),
        ])
        self.assertFalse(
            shapely.box(*footprint).contains(
                self.crossing.representative_point()))
        self.rows = [
            ("inside", Point(mid_lon, mid_lat), "osm",
             {"amenity": "school", "source": "survey"}),
            ("crossing", self.crossing, "osm",
             {"bridge": "yes", "source": "survey"}),
            ("outside", Point(east + 0.01, mid_lat), "osm",
             {"amenity": "cafe"}),
            ("enc", Point(mid_lon, mid_lat), "enc",
             {"amenity": "ferry_terminal"}),
            ("empty", Point(mid_lon, mid_lat), "osm",
             {"source": "survey"}),
        ]
        self.catalog_dir, self.catalog_ref = self._publish_catalog(
            "full_v1", self.rows)
        self.region_dir, self.region_ref = self._publish_region(
            "area150km2_v1", self.catalog_ref)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _publish_catalog(self, version, rows):
        directory = self.root / "artifacts" / "catalogs" / DATASET / version
        frame = schema.build_frame(
            ids=[row[0] for row in rows],
            geometries=[row[1] for row in rows],
            landmark_types=[row[2] for row in rows],
            tags=[row[3] for row in rows],
            crs="EPSG:4326",
        )
        with artifact.ArtifactDirectoryBuilder(
                directory, kind="catalogs", dataset=DATASET,
                version=version, generator="osm_test",
                config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "bbox_wsen": [-71.2, 42.2, -70.9, 42.5],
                }, declared_outputs=("catalog.feather",)) as builder:
            frame.to_feather(builder.output_path("catalog.feather"))
        return directory, artifact.open_artifact(directory)

    def _publish_region(self, version, catalog_ref, *, dataset=DATASET):
        directory = (
            self.root / "artifacts" / region.ARTIFACT_KIND
            / dataset / version)
        with artifact.ArtifactDirectoryBuilder(
                directory, kind=region.ARTIFACT_KIND, dataset=dataset,
                version=version, generator="osm_test",
                upstreams=(catalog_ref,), config={"schema": region.SCHEMA},
                declared_outputs=(region.REGION_OUTPUT,)) as builder:
            artifact.atomic_write_json(
                builder.output_path(region.REGION_OUTPUT), self.plan)
        return directory, artifact.open_artifact(directory)

    def test_selects_intersecting_osm_and_writes_only_loci_tags(self):
        output, stats = osm.select_landmarks(
            schema.read_frame(self.catalog_dir / "catalog.feather"),
            self.plan["grid"]["footprint_bbox_wsen"])

        self.assertEqual(output["id"].tolist(), ["inside", "crossing"])
        tags = schema.tag_dicts(output)
        self.assertEqual(tags, [
            {"amenity": "school"},
            {"bridge": "yes"},
        ])
        crossing = output.loc[output["id"] == "crossing"].geometry.iloc[0]
        self.assertTrue(crossing.equals_exact(self.crossing, 0.0))
        self.assertEqual(stats["source_rows"], 5)
        self.assertEqual(stats["source_osm_rows"], 4)
        self.assertEqual(stats["spatially_intersecting_osm_rows"], 3)
        self.assertEqual(stats["empty_loci_tag_rows_dropped"], 1)
        self.assertEqual(stats["output_rows"], 2)

    @mock.patch(
        "experimental.overhead_matching.swag.farfield.viewers.indexes.refresh")
    def test_materialize_publishes_strict_typed_artifact(self, refresh):
        reference = osm.materialize(
            farfield_root=self.root, dataset=DATASET,
            region_dir=self.region_dir, catalog_dir=self.catalog_dir,
            version="area150km2_osm260101_v1")
        output_dir = Path(reference.path)
        loaded_ref, frame, stats = osm.load_loci_osm_artifact(output_dir)

        self.assertEqual(reference, loaded_ref)
        self.assertEqual(frame["id"].tolist(), ["inside", "crossing"])
        self.assertEqual(stats["output_rows"], 2)
        manifest = artifact.load_manifest(output_dir)
        self.assertEqual(
            set(manifest.upstreams), {self.region_ref, self.catalog_ref})
        self.assertEqual(manifest.config["schema"], osm.SCHEMA)
        self.assertEqual(
            manifest.config["footprint_bbox_wsen"],
            self.plan["grid"]["footprint_bbox_wsen"])
        self.assertFalse(manifest.config["geometry_clipped"])
        self.assertEqual(refresh.call_count, 1)

        reused = osm.materialize(
            farfield_root=self.root, dataset=DATASET,
            region_dir=self.region_dir, catalog_dir=self.catalog_dir,
            version="area150km2_osm260101_v1")
        self.assertEqual(reused, reference)
        self.assertEqual(refresh.call_count, 1)

    def test_rejects_region_bound_to_another_full_catalog(self):
        other_dir, _ = self._publish_catalog("other_full_v1", self.rows)
        with self.assertRaisesRegex(
                osm.LociOsmError, "not derived from the selected"):
            osm.materialize(
                farfield_root=self.root, dataset=DATASET,
                region_dir=self.region_dir, catalog_dir=other_dir,
                version="invalid")

    @mock.patch(
        "experimental.overhead_matching.swag.farfield.viewers.indexes.refresh")
    def test_shared_scope_uses_explicit_catalog_dataset(self, _refresh):
        shared_region_dir, _ = self._publish_region(
            "area150km2_shared_v1", self.catalog_ref,
            dataset="example_shared")

        reference = osm.materialize(
            farfield_root=self.root,
            dataset="example_shared",
            region_dir=shared_region_dir,
            catalog_dir=self.catalog_dir,
            catalog_dataset=DATASET,
            version="area150km2_osm_shared_v1",
        )

        manifest = artifact.load_manifest(Path(reference.path))
        self.assertEqual(manifest.dataset, "example_shared")
        self.assertEqual(manifest.config["catalog_dataset"], DATASET)


if __name__ == "__main__":
    unittest.main()
