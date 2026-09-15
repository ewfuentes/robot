import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    lidar_dsm,
    terrain,
)


def _write_las(path: Path, x, y, z, classification, crs="EPSG:6348"):
    import laspy
    import pyproj

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_crs(pyproj.CRS(crs))
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = np.asarray(x, dtype=np.float64)
    las.y = np.asarray(y, dtype=np.float64)
    las.z = np.asarray(z, dtype=np.float64)
    las.classification = np.asarray(classification, dtype=np.uint8)
    las.write(path)


class RasterAccumulatorTest(unittest.TestCase):
    def test_rejects_non_integer_bounds(self):
        with self.assertRaises(ValueError):
            lidar_dsm.RasterAccumulator.create(
                bounds_xy=(0.5, 0.0, 10.0, 10.0), resolution_m=1.0,
                crs="EPSG:6348")

    def test_max_and_counts(self):
        acc = lidar_dsm.RasterAccumulator.create(
            bounds_xy=(0.0, 0.0, 4.0, 4.0), resolution_m=1.0,
            crs="EPSG:6348")
        used = acc.add_points(np.array([0.5, 0.5, 3.5, 9.0]),
                              np.array([3.5, 3.5, 0.5, 9.0]),
                              np.array([1.0, 5.0, 2.0, 99.0]))
        self.assertEqual(used, 3)  # the (9, 9) point is out of bounds
        # (0.5, 3.5) is row 0, col 0; max of the two returns wins.
        self.assertEqual(acc.elevation[0, 0], 5.0)
        self.assertEqual(acc.counts[0, 0], 2)
        # (3.5, 0.5) is row 3, col 3.
        self.assertEqual(acc.elevation[3, 3], 2.0)
        self.assertEqual(acc.counts[3, 3], 1)


class StreamTileTest(unittest.TestCase):
    def test_class_filter_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            las_path = Path(tmp) / "tile.las"
            # ground plane at z=10, a building return at z=30, a vegetation
            # return at z=50 (excluded), a noise return at z=500 (excluded).
            _write_las(las_path,
                       x=[100.5, 100.5, 101.5, 101.5],
                       y=[100.5, 101.5, 100.5, 101.5],
                       z=[10.0, 30.0, 50.0, 500.0],
                       classification=[2, 6, 5, 7])
            acc = lidar_dsm.RasterAccumulator.create(
                bounds_xy=(100.0, 100.0, 102.0, 102.0), resolution_m=1.0,
                crs="EPSG:6348")
            stats = lidar_dsm.stream_tile(
                las_path, acc, keep_classes=lidar_dsm.DSM_KEEP_CLASSES)
        self.assertEqual(stats["points_total"], 4)
        self.assertEqual(stats["points_kept_class"], 2)
        self.assertEqual(stats["points_in_bounds"], 2)
        self.assertFalse(stats["reprojected"])  # same CRS -> identity
        self.assertEqual(acc.elevation[1, 0], 10.0)  # ground
        self.assertEqual(acc.elevation[0, 0], 30.0)  # building
        self.assertEqual(acc.counts[1, 1], 0)  # vegetation excluded
        self.assertFalse(stats["skipped_disjoint_bbox"])

    def test_disjoint_tile_is_skipped_without_reading_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            las_path = Path(tmp) / "tile.las"
            _write_las(las_path, x=[5000.5], y=[5000.5], z=[1.0],
                       classification=[2])
            acc = lidar_dsm.RasterAccumulator.create(
                bounds_xy=(100.0, 100.0, 102.0, 102.0), resolution_m=1.0,
                crs="EPSG:6348")
            stats = lidar_dsm.stream_tile(
                las_path, acc, keep_classes=lidar_dsm.DSM_KEEP_CLASSES)
        self.assertTrue(stats["skipped_disjoint_bbox"])
        self.assertEqual(stats["points_total"], 1)
        self.assertEqual(stats["points_in_bounds"], 0)
        self.assertEqual(int(acc.counts.sum()), 0)


class FillHolesTest(unittest.TestCase):
    def test_pinhole_filled_large_hole_left(self):
        elevation = np.full((8, 8), 7.0, dtype=np.float32)
        valid = np.ones((8, 8), dtype=bool)
        valid[2, 2] = False  # pinhole: 8 valid neighbors
        valid[5:8, 5:8] = False  # 3x3 hole: corner cell has 3 valid neighbors
        elevation[~valid] = np.nan
        filled, filled_mask = lidar_dsm.fill_holes(elevation, valid, passes=1)
        self.assertEqual(filled[2, 2], 7.0)
        self.assertTrue(filled_mask[2, 2])
        self.assertTrue(np.isnan(filled[6, 6]))  # interior of the big hole
        self.assertFalse(filled_mask[6, 6])

    def test_block_sweep_matches_single_block(self):
        rng = np.random.default_rng(1)
        elevation = rng.uniform(0, 100, (30, 17)).astype(np.float32)
        valid = rng.uniform(size=(30, 17)) > 0.25
        elevation[~valid] = np.nan
        whole, whole_mask = lidar_dsm.fill_holes(elevation, valid, passes=2,
                                                 block_rows=1024)
        blocked, blocked_mask = lidar_dsm.fill_holes(elevation, valid,
                                                     passes=2, block_rows=4)
        np.testing.assert_array_equal(whole_mask, blocked_mask)
        np.testing.assert_allclose(np.nan_to_num(whole, nan=-1),
                                   np.nan_to_num(blocked, nan=-1))


class ComposeSurfaceTest(unittest.TestCase):
    def test_provenance_and_dem_fallback(self):
        acc = lidar_dsm.RasterAccumulator.create(
            bounds_xy=(0.0, 0.0, 8.0, 8.0), resolution_m=1.0, crs="EPSG:6348")
        # LiDAR covers the west half except one pinhole.
        xs, ys = np.meshgrid(np.arange(4) + 0.5, np.arange(8) + 0.5)
        acc.add_points(xs.ravel(), ys.ravel(),
                       np.full(xs.size, 20.0))
        pin_row, pin_col = 3, 2
        acc.counts[pin_row, pin_col] = 0
        acc.elevation[pin_row, pin_col] = -np.inf
        # DEM (hydro-flattened water at z=5) covers the whole grid, except
        # its own no-data corner.
        dem = terrain.HeightField(
            elevation=np.full((8, 8), 5.0, dtype=np.float32),
            x0=0.0, y0=8.0, res=1.0, crs="EPSG:6348",
            nodata_mask=np.zeros((8, 8), dtype=bool))
        dem.nodata_mask[0, 7] = True
        field, provenance = lidar_dsm.compose_surface(
            acc, fill_passes=1, dem_fallback=dem)
        self.assertEqual(provenance[4, 1], lidar_dsm.PROV_LIDAR)
        self.assertEqual(provenance[pin_row, pin_col],
                         lidar_dsm.PROV_HOLE_FILLED)
        self.assertEqual(field.elevation[pin_row, pin_col], 20.0)
        self.assertEqual(provenance[4, 6], lidar_dsm.PROV_DEM_FALLBACK)
        self.assertEqual(field.elevation[4, 6], 5.0)  # water surface
        self.assertEqual(provenance[0, 7], lidar_dsm.PROV_EMPTY)
        self.assertTrue(field.nodata_mask[0, 7])
        self.assertFalse(field.nodata_mask[4, 6])

    def test_no_dem_leaves_empty_as_nodata(self):
        acc = lidar_dsm.RasterAccumulator.create(
            bounds_xy=(0.0, 0.0, 4.0, 4.0), resolution_m=1.0, crs="EPSG:6348")
        acc.add_points(np.array([0.5]), np.array([0.5]), np.array([3.0]))
        field, provenance = lidar_dsm.compose_surface(
            acc, fill_passes=0, dem_fallback=None)
        self.assertEqual(int((provenance == lidar_dsm.PROV_LIDAR).sum()), 1)
        self.assertEqual(int(field.nodata_mask.sum()), 15)


class CompareStatisticsTest(unittest.TestCase):
    def test_quantile_below_max_with_spike(self):
        with tempfile.TemporaryDirectory() as tmp:
            las_path = Path(tmp) / "tile.las"
            rng = np.random.default_rng(0)
            n = 400
            x = np.full(n, 10.0) + rng.uniform(0, 1, n)
            y = np.full(n, 10.0) + rng.uniform(0, 1, n)
            z = np.full(n, 10.0)
            z[0] = 42.0  # single-return spike in the one occupied cell
            _write_las(las_path, x, y, z, np.full(n, 2))
            report = lidar_dsm.compare_statistics(
                las_path, resolution_m=1.0, keep_classes=(2,), quantile=0.98)
        self.assertEqual(report["n_cells"], 1)
        delta = report["delta_max_minus_quantile_m"]["max"]
        self.assertGreater(delta, 30.0)
        self.assertEqual(report["cells_delta_over_1m"], 1)


if __name__ == "__main__":
    unittest.main()
