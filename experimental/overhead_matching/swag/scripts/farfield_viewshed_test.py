import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.scripts import farfield_viewshed as fv


def flat_dem(elevation_m=0.0, north=1.0, west=0.0, span_deg=1.0, posts=1201):
    """A constant-elevation mosaic, for tests that only care about geometry."""
    grid = np.full((posts, posts), elevation_m, dtype=np.float32)
    return fv.DemMosaic(grid, north=north, west=west,
                        step_deg=span_deg / (posts - 1), stride=1)


class DemMosaicTest(unittest.TestCase):
    def test_row_zero_is_north_edge(self):
        # The .hgt convention puts north at row 0. Getting this inverted
        # mirrors every viewshed about the parallel and is invisible on flat
        # terrain, so it is pinned explicitly.
        grid = np.zeros((3, 3), dtype=np.float32)
        grid[0, :] = 100.0   # north edge
        grid[2, :] = 900.0   # south edge
        dem = fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=0.5, stride=1)

        self.assertAlmostEqual(float(dem.elevation(1.0, 0.0)), 100.0, delta=0.01)
        self.assertAlmostEqual(float(dem.elevation(0.0, 0.0)), 900.0, delta=0.01)

    def test_column_zero_is_west_edge(self):
        grid = np.zeros((3, 3), dtype=np.float32)
        grid[:, 0] = 10.0
        grid[:, 2] = 50.0
        dem = fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=0.5, stride=1)

        self.assertAlmostEqual(float(dem.elevation(0.5, 0.0)), 10.0, places=3)
        self.assertAlmostEqual(float(dem.elevation(0.5, 1.0)), 50.0, places=3)

    def test_bilinear_midpoint(self):
        grid = np.array([[0.0, 100.0], [0.0, 100.0]], dtype=np.float32)
        dem = fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=1.0, stride=1)
        self.assertAlmostEqual(float(dem.elevation(0.5, 0.5)), 50.0, places=3)

    def test_out_of_bounds_clamps(self):
        dem = flat_dem(42.0)
        self.assertAlmostEqual(float(dem.elevation(-40.0, -40.0)), 42.0, places=3)

    def test_vectorised_over_arrays(self):
        dem = flat_dem(7.0)
        out = dem.elevation(np.zeros((4, 5)), np.zeros((4, 5)))
        self.assertEqual(out.shape, (4, 5))


class MaxPoolTest(unittest.TestCase):
    def test_takes_the_maximum_not_a_sample(self):
        # A one-post ridge must survive decimation. Subsampling would drop it
        # and make everything behind it spuriously visible.
        grid = np.zeros((6, 6), dtype=np.float32)
        grid[1, :] = 2000.0
        pooled = fv._max_pool(grid, 3)
        self.assertEqual(pooled.shape, (2, 2))
        self.assertTrue((pooled[0, :] == 2000.0).all())

    def test_trims_partial_cells(self):
        grid = np.zeros((7, 7), dtype=np.float32)
        self.assertEqual(fv._max_pool(grid, 3).shape, (2, 2))

    def test_stride_one_is_identity_shape(self):
        grid = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(fv._max_pool(grid, 1), grid)

    def test_pooled_terrain_is_never_lower(self):
        # Pooled terrain is >= what it replaces everywhere. That makes it
        # conservative in the *occluder* role only -- the same grid also
        # supplies target elevations, where being higher makes landmarks more
        # visible, and measurement shows that effect dominates. See
        # DemMosaic.for_bbox; this asserts the property, not a safety claim.
        rng = np.random.default_rng(0)
        grid = rng.normal(500, 200, size=(30, 30)).astype(np.float32)
        pooled = fv._max_pool(grid, 3)
        for i in range(10):
            for j in range(10):
                block = grid[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]
                self.assertGreaterEqual(pooled[i, j], block.max() - 1e-4)


class HorizonTest(unittest.TestCase):
    def test_eye_level_horizon_is_about_five_km(self):
        # 3.86*sqrt(h) with standard optical refraction, not the 3.57*sqrt(h)
        # geometric figure -- refraction is worth ~8% of range and the two are
        # easy to conflate.
        self.assertAlmostEqual(fv.horizon_range_km(2.0, 0.0), 5.41, delta=0.05)

    def test_tall_peak_reaches_hundreds_of_km(self):
        self.assertAlmostEqual(fv.horizon_range_km(2.0, 3000.0), 215.0, delta=2.0)

    def test_taller_target_sees_further(self):
        self.assertLess(fv.horizon_range_km(2.0, 30.0), fv.horizon_range_km(2.0, 150.0))

    def test_refraction_extends_the_horizon(self):
        without = fv.horizon_range_km(2.0, 100.0, refraction_k=0.0)
        with_standard = fv.horizon_range_km(2.0, 100.0, refraction_k=fv.REFRACTION_K)
        self.assertGreater(with_standard, without)


class LineOfSightTest(unittest.TestCase):
    def test_tall_target_visible_over_flat_ground(self):
        dem = flat_dem(0.0, north=1.0, west=0.0)
        mast = fv.Landmark(lat=0.5 + 0.09, lon=0.5, kind="mast",
                           structure_height_m=200.0, in_dem=False)
        sightings = fv.visible_landmarks(dem, 0.5, 0.5, [mast], max_range_km=30.0)

        self.assertEqual(len(sightings), 1)
        self.assertTrue(sightings[0].visible)
        self.assertAlmostEqual(sightings[0].bearing_deg, 0.0, delta=1.0)

    def test_short_target_beyond_horizon_is_dropped(self):
        # 5 m tall at 40 km on flat ground: below the horizon on curvature
        # alone. Note the two different ways this function says "not visible":
        # terrain-occluded targets come back with visible=False, but targets
        # curvature already hides are culled before the ray march and do not
        # appear at all. Callers counting rows must not assume one row per
        # landmark.
        dem = flat_dem(0.0, north=1.0, west=0.0)
        post = fv.Landmark(lat=0.5 + 0.36, lon=0.5, kind="post",
                           structure_height_m=5.0, in_dem=False)
        sightings = fv.visible_landmarks(dem, 0.5, 0.5, [post], max_range_km=60.0)

        self.assertEqual(sightings, [])

    def test_ridge_occludes(self):
        posts = 1201
        grid = np.zeros((posts, posts), dtype=np.float32)
        dem = fv.DemMosaic(grid, north=1.0, west=0.0,
                           step_deg=1.0 / (posts - 1), stride=1)
        target = fv.Landmark(lat=0.5 + 0.09, lon=0.5, kind="mast",
                             structure_height_m=200.0, in_dem=False)

        clear = fv.visible_landmarks(dem, 0.5, 0.5, [target], max_range_km=30.0)
        self.assertTrue(clear[0].visible)

        # Raise a wall halfway along the ray, taller than the target.
        mid_row = int(round((1.0 - (0.5 + 0.045)) / dem.step_deg))
        grid[mid_row - 1:mid_row + 2, :] = 2000.0
        blocked = fv.visible_landmarks(dem, 0.5, 0.5, [target], max_range_km=30.0)
        self.assertFalse(blocked[0].visible)
        self.assertLess(blocked[0].grazing_deg, 0.0)

    def test_peak_does_not_occlude_itself(self):
        # The regression this guards: a peak's own summit post is the last
        # sample on the ray, so without the near-target exclusion every peak
        # in the catalog reports grazing ~0 and half of them fail the cut.
        posts = 1201
        grid = np.zeros((posts, posts), dtype=np.float32)
        step = 1.0 / (posts - 1)
        dem = fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=step, stride=1)

        peak_lat = 0.5 + 0.09
        peak_row = int(round((1.0 - peak_lat) / step))
        peak_col = int(round((0.5 - 0.0) / step))
        grid[peak_row - 2:peak_row + 3, peak_col - 2:peak_col + 3] = 1500.0

        peak = fv.Landmark(lat=peak_lat, lon=0.5, kind="natural:peak",
                           structure_height_m=0.0, in_dem=True)
        sightings = fv.visible_landmarks(dem, 0.5, 0.5, [peak], max_range_km=30.0)

        self.assertTrue(sightings[0].visible)
        self.assertGreater(sightings[0].grazing_deg, 1.0)

    def test_in_dem_selects_rather_than_adds(self):
        # A peak must take its height from the DEM and ignore any structure
        # height; a mast must add its own to bare ground.
        dem = flat_dem(1000.0, north=1.0, west=0.0)
        as_peak = fv.Landmark(lat=0.51, lon=0.5, kind="peak",
                              structure_height_m=500.0, in_dem=True)
        as_mast = fv.Landmark(lat=0.51, lon=0.5, kind="mast",
                              structure_height_m=500.0, in_dem=False)

        peak_sight = fv.visible_landmarks(dem, 0.5, 0.5, [as_peak], max_range_km=5.0)[0]
        mast_sight = fv.visible_landmarks(dem, 0.5, 0.5, [as_mast], max_range_km=5.0)[0]

        self.assertLess(peak_sight.elevation_angle_deg, 0.1)
        self.assertGreater(mast_sight.elevation_angle_deg, 10.0)

    def test_bearings_are_clockwise_from_north(self):
        dem = flat_dem(0.0, north=1.0, west=0.0)
        targets = [
            fv.Landmark(lat=0.52, lon=0.50, kind="n", structure_height_m=300.0),
            fv.Landmark(lat=0.50, lon=0.52, kind="e", structure_height_m=300.0),
            fv.Landmark(lat=0.48, lon=0.50, kind="s", structure_height_m=300.0),
            fv.Landmark(lat=0.50, lon=0.48, kind="w", structure_height_m=300.0),
        ]
        by_kind = {s.kind: s.bearing_deg
                   for s in fv.visible_landmarks(dem, 0.5, 0.5, targets, max_range_km=10.0)}

        self.assertAlmostEqual(by_kind["n"], 0.0, delta=1.0)
        self.assertAlmostEqual(by_kind["e"], 90.0, delta=1.0)
        self.assertAlmostEqual(by_kind["s"], 180.0, delta=1.0)
        self.assertAlmostEqual(by_kind["w"], 270.0, delta=1.0)

    def test_range_band_filters(self):
        dem = flat_dem(0.0, north=1.0, west=0.0)
        near = fv.Landmark(lat=0.5001, lon=0.5, kind="near", structure_height_m=300.0)
        far = fv.Landmark(lat=0.9, lon=0.5, kind="far", structure_height_m=300.0)
        sightings = fv.visible_landmarks(dem, 0.5, 0.5, [near, far],
                                         min_range_km=1.0, max_range_km=20.0)
        self.assertEqual([s.kind for s in sightings], [])


class HorizonCullTest(unittest.TestCase):
    """The prefilter must never remove something the ray march would have seen."""

    def test_elevated_observer_still_sees_targets_below(self):
        # The regression: culling on height-above-*observer* caps everything
        # lower than the observer at the observer's own ~5 km horizon, deleting
        # the entire view from a ridge road or an overlook. Measuring both
        # heights above the lower of the two is what keeps this case.
        #
        # The observer stands on a cliff edge, not a plateau: a flat top would
        # legitimately occlude a shallow depression angle, so a plateau here
        # would test the cull and the terrain march at once and pass for the
        # wrong reason.
        posts = 1201
        step = 1.0 / (posts - 1)
        grid = np.full((posts, posts), 100.0, dtype=np.float32)
        obs_row = int(round((1.0 - 0.5) / step))
        grid[:obs_row + 1, :] = 1200.0        # high ground north of the observer
        dem = fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=step, stride=1)

        below = fv.Landmark(lat=0.5 - 0.18, lon=0.5, kind="mast",
                            structure_height_m=60.0, in_dem=False)
        sightings = fv.visible_landmarks(dem, 0.5, 0.5, [below], max_range_km=40.0)

        self.assertEqual(len(sightings), 1, "target below the observer was culled")
        self.assertTrue(sightings[0].visible)
        self.assertLess(sightings[0].elevation_angle_deg, 0.0)  # looking down

    def test_short_distant_target_is_culled(self):
        dem = flat_dem(0.0, north=1.0, west=0.0)
        low = fv.Landmark(lat=0.5 + 0.45, lon=0.5, kind="bridge",
                          structure_height_m=15.0, in_dem=False)
        self.assertEqual(
            fv.visible_landmarks(dem, 0.5, 0.5, [low], max_range_km=80.0), [])

    def test_cull_agrees_with_full_march(self):
        # Nothing the prefilter removes should have come back visible.
        rng = np.random.default_rng(7)
        posts = 601
        grid = rng.normal(300, 150, size=(posts, posts)).astype(np.float32)
        dem = fv.DemMosaic(grid, north=1.0, west=0.0,
                           step_deg=1.0 / (posts - 1), stride=1)
        landmarks = [
            fv.Landmark(lat=0.5 + 0.004 * i, lon=0.5 + 0.003 * (i % 7),
                        kind="mast", structure_height_m=20.0 * (i % 9),
                        in_dem=False)
            for i in range(1, 60)
        ]
        seen = {s.index for s in fv.visible_landmarks(dem, 0.5, 0.5, landmarks,
                                                      max_range_km=60.0)
                if s.visible}
        # Re-run with the cull effectively disabled by making every landmark
        # tall enough to clear its own horizon, then check the culled ones were
        # not visible on their real heights.
        for index in range(len(landmarks)):
            if index in seen:
                continue
            single = fv.visible_landmarks(dem, 0.5, 0.5, [landmarks[index]],
                                          max_range_km=60.0)
            self.assertTrue(not single or not single[0].visible,
                            f"landmark {index} culled but visible")


class SpreadTest(unittest.TestCase):
    def test_antipodal_pair_is_not_spread(self):
        # The whole reason for doubling the angle: bearings 180 apart constrain
        # the same axis, so they are redundant, not complementary.
        self.assertAlmostEqual(fv.axial_spread([0.0, 180.0]), 0.0, places=6)

    def test_identical_bearings_are_not_spread(self):
        self.assertAlmostEqual(fv.axial_spread([37.0, 37.0, 37.0]), 0.0, places=6)

    def test_orthogonal_pair_is_fully_spread(self):
        self.assertAlmostEqual(fv.axial_spread([0.0, 90.0]), 1.0, places=6)

    def test_even_ring_is_fully_spread(self):
        self.assertAlmostEqual(fv.axial_spread(np.arange(0, 360, 30.0)), 1.0, places=6)

    def test_empty_is_zero(self):
        self.assertEqual(fv.axial_spread([]), 0.0)

    def test_azimuth_coverage_counts_distinct_bins(self):
        self.assertAlmostEqual(fv.azimuth_coverage([0.0, 1.0, 2.0], n_bins=36), 1 / 36)
        self.assertAlmostEqual(fv.azimuth_coverage(np.arange(0, 360, 10.0), n_bins=36), 1.0)


class PositionCovarianceTest(unittest.TestCase):
    def test_collinear_bearings_leave_one_axis_free(self):
        major, minor, _cond = fv.position_covariance([0.0, 180.0], [10.0, 10.0])
        self.assertEqual(major, float("inf"))
        self.assertTrue(math.isfinite(minor))

    def test_orthogonal_bearings_are_well_conditioned(self):
        major, minor, cond = fv.position_covariance([0.0, 90.0], [1.0, 1.0])
        self.assertTrue(math.isfinite(major))
        self.assertAlmostEqual(major, minor, places=6)
        self.assertAlmostEqual(cond, 1.0, places=6)

    def test_near_landmarks_dominate(self):
        # The claim that far-field is a heading cue, not a position cue: a
        # landmark ten times closer carries a hundred times the information,
        # so adding distant ones barely moves the position covariance.
        near_only, _, _ = fv.position_covariance([0.0, 90.0], [1.0, 1.0])
        with_far, _, _ = fv.position_covariance([0.0, 90.0, 45.0, 135.0],
                                                [1.0, 1.0, 50.0, 50.0])
        self.assertLess(abs(with_far - near_only) / near_only, 0.01)

    def test_sigma_scales_linearly_with_range(self):
        close, _, _ = fv.position_covariance([0.0, 90.0], [1.0, 1.0])
        distant, _, _ = fv.position_covariance([0.0, 90.0], [10.0, 10.0])
        self.assertAlmostEqual(distant / close, 10.0, places=4)

    def test_empty_is_infinite(self):
        major, minor, cond = fv.position_covariance([], [])
        self.assertEqual(major, float("inf"))
        self.assertEqual(minor, float("inf"))


class SiteMetricsTest(unittest.TestCase):
    @staticmethod
    def _sighting(range_km, bearing, grazing=1.0, visible=True):
        return fv.Sighting(index=0, name="", kind="k", range_km=range_km,
                           bearing_deg=bearing, elevation_angle_deg=0.5,
                           grazing_deg=grazing, visible=visible)

    def test_marginal_grazing_is_excluded(self):
        marginal = [self._sighting(20.0, b, grazing=0.001) for b in (0, 90, 180)]
        self.assertEqual(fv.site_metrics(marginal)["n_visible"], 0)

    def test_invisible_excluded(self):
        hidden = [self._sighting(20.0, 0.0, grazing=-1.0, visible=False)]
        self.assertEqual(fv.site_metrics(hidden)["n_visible"], 0)

    def test_far_count_uses_the_band(self):
        mixed = [self._sighting(1.0, 0.0), self._sighting(30.0, 90.0)]
        metrics = fv.site_metrics(mixed, far_km=5.0)
        self.assertEqual(metrics["n_visible"], 2)
        self.assertEqual(metrics["n_far"], 1)

    def test_score_vetoes_on_single_axis(self):
        one_axis = [self._sighting(30.0, 90.0), self._sighting(35.0, 270.0)]
        spread = [self._sighting(30.0, 0.0), self._sighting(35.0, 90.0)]
        self.assertAlmostEqual(fv.farfield_score(fv.site_metrics(one_axis)), 0.0, places=6)
        self.assertGreater(fv.farfield_score(fv.site_metrics(spread)), 0.0)

    def test_score_is_zero_without_far_landmarks(self):
        self.assertEqual(fv.farfield_score(fv.site_metrics([])), 0.0)


class ScoreTrackTest(unittest.TestCase):
    @staticmethod
    def _dem_with_a_peak():
        posts = 1201
        step = 1.0 / (posts - 1)
        grid = np.zeros((posts, posts), dtype=np.float32)
        row = int(round((1.0 - 0.62) / step))
        col = int(round(0.5 / step))
        grid[row - 2:row + 3, col - 2:col + 3] = 2500.0
        return fv.DemMosaic(grid, north=1.0, west=0.0, step_deg=step, stride=1)

    def test_far_km_reaches_site_metrics(self):
        # Regression: far_km was accepted by the CLI and never threaded into
        # score_track, so --far_km silently did nothing for track scoring.
        dem = self._dem_with_a_peak()
        peak = fv.Landmark(lat=0.62, lon=0.5, kind="natural:peak", in_dem=True)
        coords = [(0.5, 0.50), (0.5, 0.52)]

        near_band = fv.score_track(dem, coords, [peak], n_samples=3, far_km=5.0,
                                   max_range_km=40.0)
        far_band = fv.score_track(dem, coords, [peak], n_samples=3, far_km=30.0,
                                  max_range_km=40.0)
        self.assertGreater(near_band["n_far_union"], 0)
        self.assertEqual(far_band["n_far_union"], 0)

    def test_union_respects_the_grazing_cut(self):
        # n_far_union must not exceed the per-sample counts by using a looser
        # admission rule than site_metrics does.
        dem = self._dem_with_a_peak()
        peak = fv.Landmark(lat=0.62, lon=0.5, kind="natural:peak", in_dem=True)
        coords = [(0.5, 0.50), (0.5, 0.52)]
        result = fv.score_track(dem, coords, [peak], n_samples=3,
                                min_grazing_deg=90.0, max_range_km=40.0)
        self.assertEqual(result["n_far_union"], 0)

    def test_empty_track_scores_zero(self):
        dem = flat_dem(0.0)
        self.assertEqual(fv.score_track(dem, [], [], n_samples=4)["score"], 0.0)


class SampleTrackTest(unittest.TestCase):
    def test_samples_by_arc_length_not_vertex_index(self):
        # Vertices bunched at the start, one long leg after. Index sampling
        # would put nearly every observer in the first 100 m.
        coords = [(0.0, 0.0), (0.0, 0.0005), (0.0, 0.001), (0.0, 0.0015), (0.0, 1.0)]
        samples = fv.sample_track(coords, n_samples=4)
        self.assertEqual(len(samples), 4)
        self.assertGreater(samples[-1][0], 0.5)
        self.assertGreater(max(lat for lat, _ in samples) - min(lat for lat, _ in samples), 0.4)

    def test_single_point_track(self):
        self.assertEqual(fv.sample_track([(3.0, 4.0)], n_samples=5), [(4.0, 3.0)])

    def test_degenerate_zero_length_track(self):
        samples = fv.sample_track([(1.0, 2.0), (1.0, 2.0)], n_samples=3)
        self.assertEqual(samples, [(2.0, 1.0)])

    def test_returns_lat_lon_order(self):
        samples = fv.sample_track([(6.0, 46.0), (6.0, 46.5)], n_samples=1)
        self.assertAlmostEqual(samples[0][0], 46.25, places=3)
        self.assertAlmostEqual(samples[0][1], 6.0, places=3)


if __name__ == "__main__":
    unittest.main()
