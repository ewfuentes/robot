import math
import unittest

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog,
)


def make_catalog(**kwargs):
    kwargs.setdefault("max_visible_range_m", 10000.0)
    return filter_catalog.LandmarkCatalog(
        ["a", "b", "c"],
        np.array([0.0, 1000.0, -500.0]),
        np.array([1000.0, 0.0, -500.0]),
        **kwargs)


class ConstructionTest(unittest.TestCase):
    def test_max_visible_range_is_required(self):
        with self.assertRaises(TypeError):
            filter_catalog.LandmarkCatalog(["a"], [0.0], [1.0])
        with self.assertRaises(ValueError):
            filter_catalog.LandmarkCatalog(["a"], [0.0], [1.0],
                                           max_visible_range_m=None)

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ValueError):
            filter_catalog.LandmarkCatalog(
                ["a", "a"], [0.0, 1.0], [0.0, 1.0],
                max_visible_range_m=1000.0)

    def test_shape_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            filter_catalog.LandmarkCatalog(
                ["a", "b"], [0.0], [0.0, 1.0], max_visible_range_m=1000.0)

    def test_coordinates_and_visible_range_must_be_finite(self):
        for east, north in ((float("nan"), 0.0),
                            (0.0, float("inf"))):
            with self.subTest(east=east, north=north):
                with self.assertRaisesRegex(ValueError, "coordinates"):
                    filter_catalog.LandmarkCatalog(
                        ["a"], [east], [north], max_visible_range_m=1000.0)
        for visible_range in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(visible_range=visible_range):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    filter_catalog.LandmarkCatalog(
                        ["a"], [0.0], [1.0],
                        max_visible_range_m=visible_range)

    def test_position_sigma_is_one_uniform_finite_value(self):
        with self.assertRaisesRegex(ValueError, "uniform recorded value"):
            make_catalog(position_sigma_m=[10.0, 20.0, 10.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            make_catalog(position_sigma_m=float("nan"))

    def test_uniform_prior(self):
        cat = make_catalog()
        np.testing.assert_allclose(cat.log_prior, -math.log(3))

    def test_index_and_contains(self):
        cat = make_catalog()
        self.assertEqual(cat.index_of("b"), 1)
        self.assertIn("c", cat)
        with self.assertRaises(ValueError):
            cat.index_of("nope")


class KappaEffTest(unittest.TestCase):
    def test_exact_map_returns_kappa_unchanged(self):
        cat = make_catalog()
        out = cat.kappa_eff(100.0, np.array([500.0, 1000.0, 2000.0]))
        np.testing.assert_allclose(out, 100.0)

    def test_map_error_softens_kappa_with_range(self):
        cat = make_catalog(position_sigma_m=15.0)
        ranges = np.array([100.0, 1000.0, 10000.0])
        out = cat.kappa_eff(100.0, ranges)
        # Nearer landmarks project more angular error: kappa_eff smaller.
        self.assertLess(out[0], out[1])
        self.assertLess(out[1], out[2])
        expected_far = 1.0 / (1.0 / 100.0 + (15.0 / 10000.0) ** 2)
        self.assertAlmostEqual(out[2], expected_far, places=6)


class BearingsFromTest(unittest.TestCase):
    def test_shapes_and_values(self):
        cat = make_catalog()
        bearings, ranges = cat.bearings_from(np.array([0.0]),
                                             np.array([0.0]))
        self.assertEqual(bearings.shape, (1, 3))
        # Landmark a is due north; b due east.
        self.assertAlmostEqual(math.degrees(bearings[0, 0]) % 360.0, 0.0)
        self.assertAlmostEqual(math.degrees(bearings[0, 1]) % 360.0, 90.0)
        self.assertAlmostEqual(ranges[0, 0], 1000.0)


class PerturbedTest(unittest.TestCase):
    def test_jitter_declares_its_uniform_uncertainty(self):
        cat = make_catalog()
        rng = np.random.default_rng(3)
        jittered = cat.perturbed(25.0, rng)
        self.assertFalse(np.allclose(jittered.east_m, cat.east_m))
        np.testing.assert_allclose(jittered.position_sigma_m, 25.0)
        np.testing.assert_allclose(jittered.max_visible_range_m,
                                   cat.max_visible_range_m)


if __name__ == "__main__":
    unittest.main()
