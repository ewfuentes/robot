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


DATASET = "charles_river_20260727"
PAPER_GROUP = "charles"


class LociOsmTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.region_bbox = (-71.10, 42.34, -71.08, 42.36)
        self.plan = {
            "schema": region.SCHEMA,
            "source_bbox_wsen": [-71.2, 42.2, -70.9, 42.5],
            "bbox_wsen": list(self.region_bbox),
            "metric_reference_lat_deg": 42.35,
            "minimum_trajectory_margin_m": 500.0,
            "trajectory": {
                "datasets": [DATASET],
                "bbox_wsen": [-71.09, 42.35, -71.089, 42.351],
            },
            "grid": region.build_grid(self.region_bbox),
        }
        self.trajectory = region.TrajectoryExtent(
            datasets=(DATASET,), n_points=2,
            bbox_wsen=tuple(self.plan["trajectory"]["bbox_wsen"]),
            dataset_tables={})
        self._trajectory_patch = mock.patch.object(
            region, "load_trajectory_extent", return_value=self.trajectory)
        self._trajectory_patch.start()
        self.addCleanup(self._trajectory_patch.stop)
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
            ("faa", Point(mid_lon, mid_lat), "faa",
             {"man_made": "tower", "faa:oas": "1"}),
            ("empty", Point(mid_lon, mid_lat), "osm",
             {"source": "survey"}),
        ]
        self.full_dir, self.full_ref = self._publish_catalog(
            "full_v1", self.rows[:-2], full=True)
        self.catalog_dir, self.catalog_ref = self._publish_catalog(
            "full_plus_faa_v1", self.rows, upstreams=(self.full_ref,))
        self.selected_dir, self.selected_ref = self._publish_catalog(
            "trim625_20260910_v1", self.rows,
            upstreams=(self.catalog_ref,),
            region_bbox=[-71.2, 42.2, -70.9, 42.5])
        self.region_dir, self.region_ref = self._publish_region(
            "area150km2_v1", self.selected_ref)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _publish_catalog(self, version, rows, *, upstreams=(), full=False,
                         region_bbox=None):
        directory = self.root / "artifacts" / "catalogs" / DATASET / version
        frame = schema.build_frame(
            ids=[row[0] for row in rows],
            geometries=[row[1] for row in rows],
            landmark_types=[row[2] for row in rows],
            tags=[row[3] for row in rows],
            crs="EPSG:4326",
        )
        config = {}
        if full:
            config = {
                "schema": schema.FULL_ARTIFACT_SCHEMA,
                "bbox_wsen": [-71.2, 42.2, -70.9, 42.5],
                "source_coverage": {
                    "schema": "farfield_catalog_source_coverage/v2",
                    "status": "passed",
                    "message": "test coverage",
                    "details": [],
                },
            }
        if region_bbox is not None:
            config.update({
                "region_bbox_wsen": region_bbox,
                "region_source": "clip_bbox_wsen",
                "clip_plan": {
                    "scope": "charles_river_20260727",
                    "bbox_datasets": [DATASET],
                    "bbox_wsen": region_bbox,
                },
            })
        with artifact.ArtifactDirectoryBuilder(
                directory, kind="catalogs", dataset=DATASET,
                version=version, generator="osm_test",
                upstreams=upstreams, config=config,
                declared_outputs=("catalog.feather",)) as builder:
            frame.to_feather(builder.output_path("catalog.feather"))
        return directory, artifact.open_artifact(directory)

    def _publish_region(self, version, catalog_ref, *, dataset=DATASET,
                        legacy=False):
        directory = (
            self.root / "artifacts" / region.ARTIFACT_KIND
            / dataset / version)
        config = {
            "schema": region.SCHEMA,
            "catalog_manifest_digest": catalog_ref.manifest_digest,
            "trajectory_datasets": [DATASET],
        }
        if not legacy:
            config.update({
                "paper_group": PAPER_GROUP,
                "paper_region_bbox_wsen": [-71.2, 42.2, -70.9, 42.5],
                "untrimmed_catalog_manifest_digest": (
                    self.catalog_ref.manifest_digest),
            })
        with artifact.ArtifactDirectoryBuilder(
                directory, kind=region.ARTIFACT_KIND, dataset=dataset,
                version=version, generator="osm_test",
                upstreams=(catalog_ref,), config=config,
                declared_outputs=(region.REGION_OUTPUT,)) as builder:
            artifact.atomic_write_json(
                builder.output_path(region.REGION_OUTPUT), self.plan)
        return directory, artifact.open_artifact(directory)

    def test_selects_all_sources_and_writes_only_loci_tags(self):
        output, stats = osm.select_landmarks(
            schema.read_frame(self.catalog_dir / "catalog.feather"),
            self.plan["grid"]["footprint_bbox_wsen"])

        self.assertEqual(
            output["id"].tolist(), ["inside", "crossing", "enc", "faa"])
        self.assertEqual(
            output["landmark_type"].tolist(), ["osm", "osm", "enc", "faa"])
        tags = schema.tag_dicts(output)
        self.assertEqual(tags, [
            {"amenity": "school"},
            {"bridge": "yes"},
            {"amenity": "ferry_terminal"},
            {"man_made": "tower"},
        ])
        crossing = output.loc[output["id"] == "crossing"].geometry.iloc[0]
        self.assertTrue(crossing.equals_exact(self.crossing, 0.0))
        self.assertEqual(stats["source_rows"], 6)
        self.assertEqual(stats["source_rows_by_landmark_type"], {
            "enc": 1, "faa": 1, "osm": 4})
        self.assertEqual(stats["spatially_intersecting_rows"], 5)
        self.assertEqual(stats["empty_loci_tag_rows_dropped"], 1)
        self.assertEqual(stats["output_rows"], 4)

    @mock.patch(
        "experimental.overhead_matching.swag.farfield.viewers.indexes.refresh")
    def test_materialize_publishes_strict_typed_artifact(self, refresh):
        reference = osm.materialize(
            farfield_root=self.root, dataset=DATASET,
            paper_group=PAPER_GROUP, region_dir=self.region_dir,
            version="area150km2_osm260101_v1")
        output_dir = Path(reference.path)
        loaded_ref, frame, stats = osm.load_loci_osm_artifact(output_dir)

        self.assertEqual(reference, loaded_ref)
        self.assertEqual(
            frame["id"].tolist(), ["inside", "crossing", "enc", "faa"])
        self.assertEqual(stats["output_rows"], 4)
        manifest = artifact.load_manifest(output_dir)
        self.assertEqual(
            set(manifest.upstreams), {
                self.region_ref, self.selected_ref, self.catalog_ref})
        self.assertEqual(manifest.config["schema"], osm.SCHEMA)
        self.assertEqual(
            manifest.config["footprint_bbox_wsen"],
            self.plan["grid"]["footprint_bbox_wsen"])
        self.assertFalse(manifest.config["geometry_clipped"])
        self.assertEqual(refresh.call_count, 1)

        reused = osm.materialize(
            farfield_root=self.root, dataset=DATASET,
            paper_group=PAPER_GROUP, region_dir=self.region_dir,
            version="area150km2_osm260101_v1")
        self.assertEqual(reused, reference)
        self.assertEqual(refresh.call_count, 1)

    def test_rejects_region_catalog_config_mismatch(self):
        bad_region_dir, _ = self._publish_region(
            "bad_region_v1", self.selected_ref)
        manifest_path = bad_region_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["catalog_manifest_digest"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        with self.assertRaisesRegex(
                osm.LociOsmError, "config disagrees"):
            osm.materialize(
                farfield_root=self.root, dataset=DATASET,
                paper_group=PAPER_GROUP, region_dir=bad_region_dir,
                version="invalid")

    def test_rejects_region_that_no_longer_contains_current_trajectory(self):
        moved = region.TrajectoryExtent(
            datasets=(DATASET,), n_points=1,
            bbox_wsen=(-71.07, 42.35, -71.069, 42.351),
            dataset_tables={})
        with mock.patch.object(
                region, "load_trajectory_extent", return_value=moved):
            with self.assertRaisesRegex(
                    osm.LociOsmError, "current canonical trajectories"):
                osm.materialize(
                    farfield_root=self.root, dataset=DATASET,
                    paper_group=PAPER_GROUP, region_dir=self.region_dir,
                    version="moved_trajectory_v1")

    def test_rejects_untrimmed_catalog_missing_required_source(self):
        source = schema.read_frame(self.catalog_dir / "catalog.feather")
        source = source.loc[source["landmark_type"] != "enc"]
        with mock.patch.object(
                osm, "_load_source_catalog", return_value=source):
            with self.assertRaisesRegex(
                    osm.LociOsmError, "missing required landmark sources"):
                osm.materialize(
                    farfield_root=self.root, dataset=DATASET,
                    paper_group=PAPER_GROUP, region_dir=self.region_dir,
                    version="missing_enc_v1")

    @mock.patch(
        "experimental.overhead_matching.swag.farfield.viewers.indexes.refresh")
    def test_legacy_region_grid_binds_current_paper_catalog(self, _refresh):
        legacy_dir, legacy_ref = self._publish_region(
            "legacy_region_v1", self.full_ref, legacy=True)

        reference = osm.materialize(
            farfield_root=self.root, dataset=DATASET,
            paper_group=PAPER_GROUP, region_dir=legacy_dir,
            version="legacy_grid_current_catalog_v1")

        manifest = artifact.load_manifest(Path(reference.path))
        self.assertEqual(
            set(manifest.upstreams), {
                legacy_ref, self.selected_ref, self.catalog_ref})
        self.assertEqual(manifest.config["paper_group"], PAPER_GROUP)

    @mock.patch(
        "experimental.overhead_matching.swag.farfield.viewers.indexes.refresh")
    def test_shared_scope_uses_explicit_catalog_dataset(self, _refresh):
        shared_region_dir, _ = self._publish_region(
            "area150km2_shared_v1", self.selected_ref,
            dataset="example_shared")

        reference = osm.materialize(
            farfield_root=self.root,
            dataset="example_shared",
            paper_group=PAPER_GROUP,
            region_dir=shared_region_dir,
            version="area150km2_osm_shared_v1",
        )

        manifest = artifact.load_manifest(Path(reference.path))
        self.assertEqual(manifest.dataset, "example_shared")
        self.assertEqual(manifest.config["catalog_dataset"], DATASET)


if __name__ == "__main__":
    unittest.main()
