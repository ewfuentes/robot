import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield.loci import region


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

    def test_source_tile_range_uses_quantized_fractional_crop_origin(self):
        zoom = 2
        north, west = region.pixel_to_lat_lon(448.6, 448.6, zoom)
        south, east = region.pixel_to_lat_lon(448.7, 448.7, zoom)

        grid = region.build_grid(
            (west, south, east, north), zoom=zoom, source_px=640)

        self.assertEqual(grid["shape_xy"], [1, 1])
        self.assertEqual(grid["source_tile_range_xyxy"], [0, 0, 3, 3])
        self.assertEqual(grid["n_source_tiles"], 16)

    def test_materialize_requires_matching_full_catalog(self):
        catalog_ref = object()
        manifest = SimpleNamespace(config={
            "schema": "farfield_catalog_trim/v1",
            "bbox_wsen": list(self.source_bbox),
        })
        with mock.patch.object(
                region.artifact, "open_artifact",
                return_value=catalog_ref) as open_artifact:
            with mock.patch.object(
                    region.artifact, "load_manifest",
                    return_value=manifest):
                with mock.patch.object(
                        region, "load_trajectory_extent") as load_trajectory:
                    with self.assertRaisesRegex(
                            region.RegionError, "must be a full catalog"):
                        region.materialize(
                            farfield_root=Path("/unused"),
                            dataset="charles_river_20260727",
                            trajectory_datasets=("charles_river_20260727",),
                            catalog_dir=Path("/fake/catalog"),
                            version="area150km2_test",
                            target_area_km2=150.0,
                        )

        open_artifact.assert_called_once_with(
            Path("/fake/catalog"),
            expected_kind=region.paths_lib.CATALOGS,
            expected_dataset="charles_river_20260727",
        )
        load_trajectory.assert_not_called()

    def test_shared_scope_names_its_catalog_dataset_explicitly(self):
        manifest = SimpleNamespace(config={
            "schema": "not-a-full-catalog",
            "bbox_wsen": list(self.source_bbox),
        })
        with mock.patch.object(
                region.artifact, "open_artifact",
                return_value=object()) as open_artifact:
            with mock.patch.object(
                    region.artifact, "load_manifest",
                    return_value=manifest):
                with self.assertRaisesRegex(
                        region.RegionError, "must be a full catalog"):
                    region.materialize(
                        farfield_root=Path("/unused"),
                        dataset="boston_harbor_shared",
                        trajectory_datasets=(
                            "boston_harbor_leg1",
                            "boston_harbor_leg2",
                            "boston_harbor_leg3",
                        ),
                        catalog_dir=Path("/fake/catalog"),
                        catalog_dataset="boston_harbor_leg1",
                        version="area150km2_test",
                        target_area_km2=150.0,
                    )

        open_artifact.assert_called_once_with(
            Path("/fake/catalog"),
            expected_kind=region.paths_lib.CATALOGS,
            expected_dataset="boston_harbor_leg1",
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
