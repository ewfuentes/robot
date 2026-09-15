import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.catalog import schema
from experimental.overhead_matching.swag.farfield.loci import region
from experimental.overhead_matching.swag.farfield.paper import table_common


class RegionTest(unittest.TestCase):
    def setUp(self):
        self.trajectory = region.TrajectoryExtent(
            datasets=("charles_river_20260727",),
            n_points=513,
            bbox_wsen=(-71.0899627, 42.3532798,
                       -71.0771052, 42.3604385),
            dataset_tables={},
        )
        self.source_bbox = (
            -71.3938738436199, 42.12870117897012,
            -70.7731941563801, 42.58501782102988,
        )

    def _publish_catalog(self, root, version, *, config, upstreams=()):
        directory = root / "artifacts" / "catalogs" \
            / "charles_river_20260727" / version
        with artifact.ArtifactDirectoryBuilder(
                directory, kind="catalogs",
                dataset="charles_river_20260727", version=version,
                generator="region_test", upstreams=upstreams, config=config,
                declared_outputs=("catalog.feather",)) as builder:
            builder.output_path("catalog.feather").write_bytes(b"test")
        return directory, artifact.open_artifact(directory)

    def test_charles_150_square_kilometre_contract(self):
        plan = region.derive_region(
            self.source_bbox, self.trajectory, target_area_km2=150.0)
        self.assertAlmostEqual(plan["actual_area_km2"], 150.0, places=9)
        self.assertAlmostEqual(
            plan["uniform_inset_m"], 19339.59056105644, places=6)
        self.assertEqual(
            plan["bbox_wsen"],
            [-71.15877339231294, 42.30243167686623,
             -71.00829460768706, 42.41128732313377])
        self.assertEqual(plan["grid"]["shape_xy"], [351, 344])
        self.assertEqual(plan["grid"]["n_patches"], 120744)
        self.assertEqual(
            plan["grid"]["source_tile_range_xyxy"],
            [317021, 387630, 317461, 388062])
        self.assertEqual(plan["grid"]["n_source_tiles"], 190953)
        self.assertGreater(
            min(plan["trajectory"]["clearance_m"].values()), 5600.0)
        self.assertFalse(plan["containment_limited"])

    def test_containment_caps_the_inset_and_reports_larger_area(self):
        track = region.TrajectoryExtent(
            datasets=("long",), n_points=2,
            bbox_wsen=(-0.44, -0.01, 0.44, 0.01),
            dataset_tables={})
        plan = region.derive_region(
            (-0.5, -0.5, 0.5, 0.5), track,
            target_area_km2=1.0, minimum_trajectory_margin_m=100.0)
        self.assertTrue(plan["containment_limited"])
        self.assertGreater(plan["actual_area_km2"], 1.0)
        self.assertGreaterEqual(
            min(plan["trajectory"]["clearance_m"].values()),
            100.0 - 1e-5)

    def test_source_smaller_than_target_is_rejected(self):
        with self.assertRaisesRegex(region.RegionError, "exceeds source area"):
            region.derive_region(
                (-71.1, 42.35, -71.07, 42.37), self.trajectory,
                target_area_km2=150.0)

    def test_patch_footprint_must_fit_inside_source_catalog(self):
        source = (-0.01, -0.01, 0.01, 0.01)
        width_m, height_m = region.metric_dimensions(source)
        track = region.TrajectoryExtent(
            datasets=("tiny",), n_points=2,
            bbox_wsen=(-0.001, -0.001, 0.001, 0.001),
            dataset_tables={})
        with self.assertRaisesRegex(
                region.RegionError, "footprint extends outside"):
            region.derive_region(
                source, track,
                target_area_km2=width_m * height_m / 1e6 * 0.99999,
                minimum_trajectory_margin_m=0.0)

    def test_persisted_region_rechecks_patch_footprint_coverage(self):
        plan = region.derive_region(
            self.source_bbox, self.trajectory, target_area_km2=150.0)
        plan["source_bbox_wsen"] = plan["bbox_wsen"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / region.REGION_OUTPUT).write_text(json.dumps(plan))
            with mock.patch.object(
                    region.artifact, "open_artifact", return_value=object()):
                with self.assertRaisesRegex(
                        region.RegionError, "footprint extends outside"):
                    region.load_region(path)

    def test_grid_iteration_matches_recorded_count_and_last_center(self):
        grid = region.build_grid(
            (-71.16, 42.30, -71.01, 42.41))
        centres = list(region.iter_grid_centres(grid))
        self.assertEqual(len(centres), grid["n_patches"])
        self.assertEqual(list(centres[-1]), grid["last_center_pixel_xy"])
        self.assertTrue(all(math.isfinite(item) for item in centres[-1]))

    def test_full_paper_region_keeps_patch_footprints_inside_authority(self):
        paper_bbox = (-71.24, 42.24, -70.93, 42.47)
        plan = region.derive_paper_region(paper_bbox, self.trajectory)

        self.assertEqual(plan["bbox_wsen"], list(paper_bbox))
        self.assertTrue(plan["grid"]["contain_footprints"])
        west, south, east, north = plan["grid"]["footprint_bbox_wsen"]
        self.assertGreaterEqual(west, paper_bbox[0])
        self.assertGreaterEqual(south, paper_bbox[1])
        self.assertLessEqual(east, paper_bbox[2])
        self.assertLessEqual(north, paper_bbox[3])

    def test_source_tile_range_uses_quantized_fractional_crop_origin(self):
        zoom = 2
        north, west = region.pixel_to_lat_lon(448.6, 448.6, zoom)
        south, east = region.pixel_to_lat_lon(448.7, 448.7, zoom)

        grid = region.build_grid(
            (west, south, east, north), zoom=zoom, source_px=640)

        self.assertEqual(grid["shape_xy"], [1, 1])
        self.assertEqual(grid["source_tile_range_xyxy"], [0, 0, 3, 3])
        self.assertEqual(grid["n_source_tiles"], 16)

    def test_selected_catalog_resolves_direct_untrimmed_composite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full_dir, full_ref = self._publish_catalog(
                root, "full_v1", config={
                    "schema": schema.FULL_ARTIFACT_SCHEMA,
                    "bbox_wsen": list(self.source_bbox),
                    "source_coverage": {
                        "schema": "farfield_catalog_source_coverage/v2",
                        "status": "passed",
                        "message": "test coverage",
                        "details": [],
                    },
                })
            composite_dir, composite_ref = self._publish_catalog(
                root, "full_plus_faa_v1", config={"rows_out": 2},
                upstreams=(full_ref,))
            selected_dir, selected_ref = self._publish_catalog(
                root, "trim625_v1", config={
                    "region_bbox_wsen": [-71.24, 42.24, -70.93, 42.47],
                    "region_source": "clip_bbox_wsen",
                    "clip_plan": {
                        "scope": "charles_river_20260727",
                        "bbox_datasets": ["charles_river_20260727"],
                        "bbox_wsen": [-71.24, 42.24, -70.93, 42.47],
                    },
                }, upstreams=(composite_ref,))
            group = replace(
                table_common.DATASET_GROUP_BY_KEY["charles"],
                catalog_version="trim625_v1")

            inputs = region.load_paper_catalog(
                selected_dir, group=group)

            self.assertEqual(inputs.selected_ref, selected_ref)
            self.assertEqual(inputs.untrimmed_ref, composite_ref)
            self.assertEqual(inputs.source_bbox_wsen, self.source_bbox)
            self.assertTrue(full_dir.is_dir())
            self.assertTrue(composite_dir.is_dir())

    def test_materialize_rejects_footprint_outside_paper_region(self):
        catalog = region.PaperCatalogInputs(
            selected_ref=object(),
            selected_region_bbox_wsen=(-71.10, 42.34, -71.07, 42.37),
            untrimmed_ref=object(),
            source_bbox_wsen=self.source_bbox,
        )
        group = SimpleNamespace(sequences=("charles_river_20260727",))
        with mock.patch.object(
                region, "resolve_paper_group",
                return_value=(group, Path("/fake/catalog"))):
            with mock.patch.object(
                    region, "load_paper_catalog", return_value=catalog):
                with mock.patch.object(
                        region, "load_trajectory_extent",
                        return_value=self.trajectory):
                    with self.assertRaisesRegex(
                            region.RegionError,
                            "outside the table_common-selected"):
                        region.materialize(
                            farfield_root=Path("/unused"),
                            dataset="charles_river_20260727",
                            paper_group="charles",
                            version="area150km2_test",
                            target_area_km2=150.0,
                        )

    def test_paper_group_owns_catalog_pin_and_trajectories(self):
        group, path = region.resolve_paper_group(
            Path("/farfield"), "boston_harbor")

        self.assertEqual(group.catalog_version, "trim625_20260911_v1")
        self.assertEqual(group.sequences, (
            "boston_harbor_leg1",
            "boston_harbor_leg2",
            "boston_harbor_leg3",
        ))
        self.assertEqual(
            path,
            Path("/farfield/artifacts/catalogs/boston_harbor_leg1/"
                 "trim625_20260911_v1"),
        )

    def test_grid_zoom_controls_patch_density(self):
        z20 = region.build_grid(self.source_bbox, zoom=20)
        z19 = region.build_grid(self.source_bbox, zoom=19)

        self.assertLess(z19["n_patches"], z20["n_patches"])
        self.assertAlmostEqual(
            z19["patch_ground_m_at_mid_lat"],
            2.0 * z20["patch_ground_m_at_mid_lat"],
        )


if __name__ == "__main__":
    unittest.main()
