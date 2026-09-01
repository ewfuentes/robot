import tempfile
import unittest
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    lattice as lattice_lib,
    terrain,
)


def _ramp_field(size: int = 50, res: float = 10.0) -> terrain.HeightField:
    """Elevation increases 1 m per meter eastward: z == x."""
    cols = np.arange(size, dtype=np.float32)
    elevation = np.tile((cols + 0.5) * res, (size, 1)).astype(np.float32)
    return terrain.HeightField(
        elevation=elevation, x0=0.0, y0=size * res, res=res,
        crs="EPSG:32619", nodata_mask=np.zeros((size, size), dtype=bool))


class HeightFieldTest(unittest.TestCase):
    def test_bilinear_sample_on_ramp(self):
        hf = _ramp_field()
        xs = np.array([5.0, 123.4, 250.0, 499.0])
        ys = np.array([250.0, 250.0, 10.0, 490.0])
        values = hf.sample(xs, ys)
        # Pixel centers span x in [5, 495]; beyond the last center the
        # clamped interpolation holds the edge value.
        np.testing.assert_allclose(values, np.clip(xs, 5.0, 495.0), atol=1e-4)

    def test_outside_is_nan(self):
        hf = _ramp_field()
        self.assertTrue(np.isnan(hf.sample(-100.0, 250.0)))
        self.assertTrue(np.isnan(hf.sample(250.0, 1e6)))

    def test_save_load_round_trip(self):
        hf = _ramp_field()
        hf.nodata_mask[3, 4] = True
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "surface"
            hf.save(base, extra_manifest={"source": "test"})
            loaded = terrain.HeightField.load(base)
        np.testing.assert_array_equal(loaded.elevation, hf.elevation)
        np.testing.assert_array_equal(loaded.nodata_mask, hf.nodata_mask)
        self.assertEqual(loaded.crs, hf.crs)
        self.assertEqual(loaded.res, hf.res)

    def test_bounds(self):
        hf = _ramp_field(size=50, res=10.0)
        self.assertEqual(hf.bounds, (0.0, 0.0, 500.0, 500.0))


class BuildHeightFieldTest(unittest.TestCase):
    def test_mosaic_from_geotiffs(self):
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            # Two 20x20 tiles side by side at 10 m; z == x as in the ramp.
            for i, x_off in enumerate((0.0, 200.0)):
                path = Path(tmp) / f"tile{i}.tif"
                cols = np.arange(20, dtype=np.float32)
                data = np.tile(x_off + (cols + 0.5) * 10.0, (20, 1))
                with rasterio.open(
                        path, "w", driver="GTiff", height=20, width=20,
                        count=1, dtype="float32", crs="EPSG:32619",
                        transform=from_origin(x_off, 200.0, 10.0, 10.0),
                        nodata=-9999.0) as dst:
                    dst.write(data.astype(np.float32), 1)
                paths.append(path)

            hf = terrain.build_height_field(
                paths, dst_crs="EPSG:32619", resolution_m=10.0,
                bounds_xy=(0.0, 0.0, 400.0, 200.0))

        self.assertEqual(hf.elevation.shape, (20, 40))
        xs = np.array([55.0, 355.0])
        ys = np.array([100.0, 100.0])
        np.testing.assert_allclose(hf.sample(xs, ys), xs, atol=1.0)
        self.assertFalse(hf.nodata_mask.any())


class LatticeTest(unittest.TestCase):
    def test_spacing_and_count(self):
        hf = _ramp_field(size=50, res=10.0)  # 500 x 500 m
        lat = lattice_lib.build_lattice(hf, spacing_m=100.0)
        self.assertEqual(len(lat), 25)
        self.assertEqual(lat.n_dropped_nodata, 0)
        self.assertAlmostEqual(float(lat.x_m.min()), 50.0)
        self.assertAlmostEqual(float(lat.y_m.max()), 450.0)

    def test_nodata_cells_dropped(self):
        hf = _ramp_field(size=50, res=10.0)
        hf.nodata_mask[:, :25] = True  # west half invalid
        lat = lattice_lib.build_lattice(hf, spacing_m=100.0)
        self.assertEqual(len(lat), 15)
        self.assertEqual(lat.n_dropped_nodata, 10)
        self.assertTrue((lat.x_m > 240.0).all())

    def test_disjoint_bounds_raise(self):
        hf = _ramp_field()
        with self.assertRaises(ValueError):
            lattice_lib.build_lattice(
                hf, spacing_m=10.0, bounds_xy=(1e5, 1e5, 2e5, 2e5))


class LatticeBackgroundTest(unittest.TestCase):
    """A candidate is dropped only when no surface has data there."""

    def _field(self, nodata: np.ndarray, res: float) -> terrain.HeightField:
        return terrain.HeightField(
            elevation=np.zeros(nodata.shape, dtype=np.float32),
            x0=0.0, y0=float(nodata.shape[0]) * res, res=res,
            crs="EPSG:32619", nodata_mask=nodata)

    def test_background_keeps_candidates_over_a_hole(self):
        nodata = np.zeros((100, 100), dtype=bool)
        nodata[:, 50:] = True  # right half of the fine surface is empty
        fine = self._field(nodata, res=10.0)
        coarse = self._field(np.zeros((10, 10), dtype=bool), res=100.0)

        alone = lattice_lib.build_lattice(fine, spacing_m=100.0)
        composed = lattice_lib.build_lattice(fine, spacing_m=100.0,
                                             backgrounds=[coarse])
        self.assertEqual(alone.n_dropped_nodata, 50)
        self.assertEqual(composed.n_dropped_nodata, 0)
        self.assertEqual(len(composed), len(alone) + 50)

    def test_background_that_also_lacks_data_drops_the_candidate(self):
        nodata = np.ones((100, 100), dtype=bool)
        fine = self._field(nodata, res=10.0)
        coarse = self._field(np.ones((10, 10), dtype=bool), res=100.0)
        result = lattice_lib.build_lattice(
            fine, spacing_m=100.0, backgrounds=[coarse])
        self.assertEqual(len(result), 0)
        self.assertEqual(result.n_dropped_nodata, 100)


if __name__ == "__main__":
    unittest.main()
