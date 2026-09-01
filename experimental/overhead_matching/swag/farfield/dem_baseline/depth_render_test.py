import dataclasses
import math
import unittest

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry
from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    terrain,
)


def _flat_field(elevation_m: float = 0.0, size: int = 900,
                res: float = 100.0) -> terrain.HeightField:
    """Flat terrain centered on (0, 0), plenty larger than the render range."""
    shape = (size, size)
    return terrain.HeightField(
        elevation=np.full(shape, elevation_m, dtype=np.float32),
        x0=-size * res / 2.0, y0=size * res / 2.0, res=res, crs="EPSG:32619",
        nodata_mask=np.zeros(shape, dtype=bool))


def _small_config(**overrides) -> depth_render.RenderConfig:
    defaults = dict(n_yaw=4, fov_deg=60.0, width=64, height=64,
                    observer_height_m=0.0, max_range_m=40000.0,
                    min_range_m=1.0, curvature=False)
    defaults.update(overrides)
    return depth_render.RenderConfig(**defaults)


class FlatPlaneTest(unittest.TestCase):
    """Analytic case: observer at height h over an infinite plane."""

    def test_depth_matches_h_over_sin(self):
        hf = _flat_field(0.0)
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        config = _small_config(observer_height_m=50.0, max_range_m=20000.0)
        ring = depth_render.render_ring(tt, config, 0.0, 0.0)

        pixel_elev = depth_render.row_elevation_angles_rad(
            config, "cpu").numpy()
        for view in range(config.n_yaw):
            depth = ring.depth_m[view].numpy()
            below = pixel_elev < math.radians(-0.5)
            expected = 50.0 / np.sin(-pixel_elev[below])
            np.testing.assert_allclose(depth[below], expected, rtol=0.02)

    def test_upward_pixels_are_sky(self):
        hf = _flat_field(0.0)
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        config = _small_config(observer_height_m=50.0)
        ring = depth_render.render_ring(tt, config, 0.0, 0.0)
        pixel_elev = depth_render.row_elevation_angles_rad(
            config, "cpu").numpy()
        up = pixel_elev > 0.0
        self.assertTrue(np.isinf(ring.depth_m[0].numpy()[up]).all())

    def test_full_coverage_on_valid_terrain(self):
        hf = _flat_field(0.0)
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        ring = depth_render.render_ring(
            tt, _small_config(max_range_m=10000.0), 0.0, 0.0)
        np.testing.assert_allclose(ring.coverage, 1.0, atol=1e-6)


class CylinderTest(unittest.TestCase):
    def test_flat_plane_depth_and_sky(self):
        hf = _flat_field(0.0)
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        config = _small_config(observer_height_m=50.0, max_range_m=20000.0)
        cyl = depth_render.render_cylinder(
            tt, config, 0.0, 0.0, n_az=90, elev_min_deg=-20.0,
            elev_max_deg=10.0, n_rows=60)
        self.assertEqual(cyl.depth_m.shape, (60, 90))
        elev = np.radians(cyl.elev_deg)
        below = elev < math.radians(-0.5)
        depth = cyl.depth_m.numpy()
        expected = 50.0 / np.sin(-elev[below])
        for col in (0, 33, 89):
            np.testing.assert_allclose(depth[below, col], expected,
                                       rtol=0.02)
        self.assertTrue(np.isinf(depth[elev > math.radians(0.5)]).all())
        self.assertEqual(cyl.coverage, 1.0)

    def test_wall_appears_at_its_azimuth(self):
        hf = _flat_field(0.0)
        # 200 m wall due east, 1 km away (columns near azimuth 90).
        col0 = int((1000.0 - hf.x0) / hf.res)
        hf.elevation[:, col0:col0 + 2] = 200.0
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        config = _small_config(observer_height_m=10.0, max_range_m=20000.0)
        cyl = depth_render.render_cylinder(
            tt, config, 0.0, 0.0, n_az=360, elev_min_deg=-5.0,
            elev_max_deg=8.0, n_rows=26)
        depth = cyl.depth_m.numpy()
        up_rows = cyl.elev_deg > 2.0
        east = depth[np.ix_(up_rows, np.arange(85, 96))]
        north = depth[np.ix_(up_rows, np.arange(0, 10))]
        self.assertTrue((east < 1500.0).all())
        self.assertTrue(np.isinf(north).all())


class BackgroundTest(unittest.TestCase):
    """A coarse background supplies geometry the fine grid does not have."""

    def _fine_with_hole(self) -> terrain.HeightField:
        # 4 km of fine grid around the origin, with an interior hole.
        size, res = 400, 10.0
        hf = terrain.HeightField(
            elevation=np.zeros((size, size), dtype=np.float32),
            x0=-size * res / 2.0, y0=size * res / 2.0, res=res,
            crs="EPSG:32619",
            nodata_mask=np.zeros((size, size), dtype=bool))
        hf.nodata_mask[100:140, 100:140] = True
        return hf

    def test_far_wall_only_in_background_is_rendered(self):
        # A 400 m wall 8 km north: outside the fine grid entirely.
        coarse = _flat_field(0.0, size=900, res=100.0)
        wall_row = int((coarse.y0 - 8000.0) / coarse.res)
        coarse.elevation[wall_row - 1:wall_row + 1, :] = 400.0
        background = depth_render.TerrainTensor.from_height_field(
            coarse, device="cpu")
        config = _small_config(observer_height_m=2.0, max_range_m=20000.0)

        alone = depth_render.render_cylinder(
            depth_render.TerrainTensor.from_height_field(
                self._fine_with_hole(), device="cpu"),
            config, 0.0, 0.0, n_az=36, elev_min_deg=-2.0, elev_max_deg=6.0,
            n_rows=16)
        composed = depth_render.render_cylinder(
            depth_render.TerrainTensor.from_height_field(
                self._fine_with_hole(), device="cpu", background=background),
            config, 0.0, 0.0, n_az=36, elev_min_deg=-2.0, elev_max_deg=6.0,
            n_rows=16)

        # The wall top is atan(398 / 8000) = 2.85 deg above the observer, so
        # rows below that see it and rows above it stay sky.
        on_wall = (composed.elev_deg > 1.0) & (composed.elev_deg < 2.5)
        above = composed.elev_deg > 3.2
        north = composed.depth_m.numpy()[np.ix_(on_wall, [0])]
        np.testing.assert_allclose(north, 8000.0, rtol=0.05)
        self.assertTrue(np.isinf(
            composed.depth_m.numpy()[np.ix_(above, [0])]).all())
        self.assertTrue(np.isinf(
            alone.depth_m.numpy()[np.ix_(on_wall, [0])]).all())

    def test_background_fills_an_interior_hole(self):
        hf = self._fine_with_hole()
        # Fine grid is flat 0; the hole would read as edge-clamped 0 anyway,
        # so give the background a distinctive height to see which one wins.
        coarse = _flat_field(50.0, size=900, res=100.0)
        tt = depth_render.TerrainTensor.from_height_field(
            hf, device="cpu",
            background=depth_render.TerrainTensor.from_height_field(
                coarse, device="cpu"))
        hole_x = torch.tensor([[hf.x0 + 120 * hf.res]])
        hole_y = torch.tensor([[hf.y0 - 120 * hf.res]])
        edge_x = torch.tensor([[hf.x0 + 10 * hf.res]])
        edge_y = torch.tensor([[hf.y0 - 10 * hf.res]])
        self.assertAlmostEqual(float(tt.sample(hole_x, hole_y)), 50.0)
        self.assertAlmostEqual(float(tt.sample(edge_x, edge_y)), 0.0)
        self.assertAlmostEqual(float(tt.sample_valid(hole_x, hole_y)), 1.0)

    def test_coverage_counts_background_data(self):
        hf = self._fine_with_hole()
        config = _small_config(observer_height_m=2.0, max_range_m=10000.0)
        alone = depth_render.render_cylinder(
            depth_render.TerrainTensor.from_height_field(hf, device="cpu"),
            config, 0.0, 0.0, n_az=36, n_rows=8)
        composed = depth_render.render_cylinder(
            depth_render.TerrainTensor.from_height_field(
                hf, device="cpu",
                background=depth_render.TerrainTensor.from_height_field(
                    _flat_field(0.0, size=900, res=100.0), device="cpu")),
            config, 0.0, 0.0, n_az=36, n_rows=8)
        self.assertLess(alone.coverage, 0.5)  # fine grid is 4 km of 10 km
        self.assertAlmostEqual(composed.coverage, 1.0, places=6)


class WallTest(unittest.TestCase):
    """A tall north wall must appear in the north view at the right depth
    and the right heading, and be absent looking south."""

    def test_wall_depth_and_heading(self):
        hf = _flat_field(0.0)
        wall_north_m = 10000.0
        wall_height_m = 800.0
        wall_row = int((hf.y0 - wall_north_m) / hf.res)
        hf.elevation[wall_row - 2:wall_row + 1, :] = wall_height_m

        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        config = _small_config(observer_height_m=10.0, max_range_m=30000.0)
        ring = depth_render.render_ring(tt, config, 0.0, 0.0)

        pixel_elev = depth_render.row_elevation_angles_rad(
            config, "cpu").numpy()
        center_col = config.width // 2
        wall_top_angle = math.atan2(wall_height_m - 10.0, wall_north_m)

        north = ring.depth_m[0].numpy()[:, center_col]
        elev_col = pixel_elev[:, center_col]
        on_wall = (elev_col > math.radians(0.5)) \
            & (elev_col < wall_top_angle - math.radians(0.5))
        self.assertGreater(on_wall.sum(), 3)
        expected = wall_north_m / np.cos(elev_col[on_wall])
        # One grid cell (100 m) of quantization plus the wall's 300 m footprint.
        np.testing.assert_allclose(north[on_wall], expected, rtol=0.05)

        south = ring.depth_m[2].numpy()[:, center_col]
        self.assertTrue(np.isinf(south[elev_col > math.radians(0.5)]).all())


class CurvatureTest(unittest.TestCase):
    """Earth curvature hides the ground beyond the horizon."""

    def test_slightly_below_horizontal_ray(self):
        hf = _flat_field(0.0)
        tt = depth_render.TerrainTensor.from_height_field(hf, device="cpu")
        obs_h = 100.0
        # A ray ~0.2 deg below horizontal hits flat ground at ~29 km, but
        # with curvature (R_eff = R / (1 - 0.13)) the quadratic
        # s^2/(2 R_eff) - s tan(theta) + h = 0 has no real root -> sky.
        # The window where both claims hold for obs_h=100 is roughly
        # (-0.29 deg, -0.11 deg); a 512-row image has pixels inside it.
        config = _small_config(observer_height_m=obs_h, height=512,
                               max_range_m=44000.0)
        flat = depth_render.render_ring(
            tt, dataclasses.replace(config, curvature=False), 0.0, 0.0)
        curved = depth_render.render_ring(
            tt, dataclasses.replace(config, curvature=True), 0.0, 0.0)

        pixel_elev = depth_render.row_elevation_angles_rad(
            config, "cpu").numpy()
        center_col = config.width // 2
        row = int(np.argmin(
            np.abs(pixel_elev[:, center_col] - math.radians(-0.2))))
        actual_elev = float(pixel_elev[row, center_col])
        self.assertLess(actual_elev, math.radians(-0.14))
        self.assertGreater(actual_elev, math.radians(-0.28))

        flat_depth = flat.depth_m[0, row, center_col].item()
        expected_flat = obs_h / math.sin(-actual_elev)
        self.assertAlmostEqual(flat_depth / expected_flat, 1.0, delta=0.02)

        r_eff = geometry.MEAN_EARTH_RADIUS_M / (1.0 - 0.13)
        discriminant = math.tan(-actual_elev) ** 2 - 2.0 * obs_h / r_eff
        self.assertLess(discriminant, 0.0)  # the analytic claim itself
        self.assertTrue(math.isinf(curved.depth_m[0, row, center_col].item()))


class ScheduleTest(unittest.TestCase):
    def test_march_schedule_monotone_and_bounded(self):
        config = _small_config(max_range_m=30000.0)
        s = depth_render.march_schedule(config, 10.0, "cpu").numpy()
        self.assertTrue((np.diff(s) > 0).all())
        self.assertAlmostEqual(s[-1], 30000.0)
        self.assertLess(len(s), 4000)


if __name__ == "__main__":
    unittest.main()
