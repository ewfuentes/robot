"""Tests for odometry noise model."""

import unittest
import math
import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.evaluation.odometry_noise import (
    OdometryNoiseConfig,
    add_noise_to_motion_deltas,
    compute_positions_from_deltas,
    METERS_PER_DEG_LAT,
)


# Seattle-ish latitude for realistic test scenarios
START_LATLON = torch.tensor([47.6, -122.3], dtype=torch.float64)


def _make_straight_north_deltas(num_steps: int, step_size_m: float) -> torch.Tensor:
    """Create motion deltas for a straight-north path.

    Returns (num_steps, 2) tensor of [delta_lat, delta_lon] in degrees.
    """
    step_deg = step_size_m / METERS_PER_DEG_LAT
    deltas = torch.zeros(num_steps, 2, dtype=torch.float64)
    deltas[:, 0] = step_deg  # north only
    return deltas


class TestZeroNoise(unittest.TestCase):
    """Zero noise should return input unchanged."""

    def test_zero_noise(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.0)
        deltas = _make_straight_north_deltas(10, 50.0)
        result = add_noise_to_motion_deltas(deltas, START_LATLON, config)
        torch.testing.assert_close(result, deltas, atol=1e-12, rtol=1e-12)


class TestOutputShape(unittest.TestCase):
    """Output shape should match input shape."""

    def test_shape(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        for n in [1, 5, 50]:
            deltas = _make_straight_north_deltas(n, 30.0)
            result = add_noise_to_motion_deltas(deltas, START_LATLON, config)
            self.assertEqual(result.shape, (n, 2))


class TestSeedReproducibility(unittest.TestCase):
    """Same seed should produce identical results."""

    def test_same_seed_same_result(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        deltas = _make_straight_north_deltas(20, 50.0)

        gen1 = torch.Generator().manual_seed(42)
        result1 = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        result2 = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen2)

        torch.testing.assert_close(result1, result2)

    def test_different_seeds_differ(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        deltas = _make_straight_north_deltas(20, 50.0)

        gen1 = torch.Generator().manual_seed(42)
        result1 = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen1)

        gen2 = torch.Generator().manual_seed(999)
        result2 = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen2)

        self.assertFalse(torch.allclose(result1, result2))


class TestNoiseScalesWithStepSize(unittest.TestCase):
    """Noise magnitude should be proportional to step size."""

    def test_noise_scales_with_step_size(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.1)
        num_trials = 500

        ref_lat_rad = math.radians(START_LATLON[0].item())
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)

        errors_small = []
        errors_large = []
        for trial in range(num_trials):
            gen_s = torch.Generator().manual_seed(trial)
            small = _make_straight_north_deltas(1, 25.0)
            noised_s = add_noise_to_motion_deltas(small, START_LATLON, config, generator=gen_s)
            diff_s = noised_s - small
            err_s = math.sqrt(
                (diff_s[0, 0].item() * METERS_PER_DEG_LAT) ** 2 +
                (diff_s[0, 1].item() * meters_per_deg_lon) ** 2
            )
            errors_small.append(err_s)

            gen_l = torch.Generator().manual_seed(trial)
            large = _make_straight_north_deltas(1, 50.0)
            noised_l = add_noise_to_motion_deltas(large, START_LATLON, config, generator=gen_l)
            diff_l = noised_l - large
            err_l = math.sqrt(
                (diff_l[0, 0].item() * METERS_PER_DEG_LAT) ** 2 +
                (diff_l[0, 1].item() * meters_per_deg_lon) ** 2
            )
            errors_large.append(err_l)

        mean_small = sum(errors_small) / len(errors_small)
        mean_large = sum(errors_large) / len(errors_large)
        ratio = mean_large / mean_small
        # Should be ~2.0 (50/25)
        self.assertGreater(ratio, 1.5, f"Ratio {ratio:.2f} should be ~2.0")
        self.assertLess(ratio, 2.5, f"Ratio {ratio:.2f} should be ~2.0")


class TestNoiseIsIsotropic(unittest.TestCase):
    """North and east noise should have similar magnitude."""

    def test_isotropic(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.1)
        num_trials = 500

        ref_lat_rad = math.radians(START_LATLON[0].item())
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)

        north_errors = []
        east_errors = []
        for trial in range(num_trials):
            gen = torch.Generator().manual_seed(trial)
            deltas = _make_straight_north_deltas(1, 50.0)
            noised = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen)
            diff = noised - deltas
            north_errors.append(abs(diff[0, 0].item() * METERS_PER_DEG_LAT))
            east_errors.append(abs(diff[0, 1].item() * meters_per_deg_lon))

        mean_north = sum(north_errors) / len(north_errors)
        mean_east = sum(east_errors) / len(east_errors)
        ratio = mean_north / mean_east
        # Should be ~1.0 (isotropic)
        self.assertGreater(ratio, 0.7, f"Ratio {ratio:.2f} should be ~1.0")
        self.assertLess(ratio, 1.4, f"Ratio {ratio:.2f} should be ~1.0")


class TestPositionErrorGrows(unittest.TestCase):
    """Per-step noise on deltas should cause position error to grow with path length."""

    def test_error_grows_over_path(self):
        """Position error at late steps should exceed error at early steps."""
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        num_steps = 200
        step_size_m = 30.0
        deltas = _make_straight_north_deltas(num_steps, step_size_m)
        num_trials = 200

        ref_lat_rad = math.radians(START_LATLON[0].item())
        meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)

        early_errors = []
        late_errors = []
        for trial in range(num_trials):
            gen = torch.Generator().manual_seed(trial * 7 + 13)
            noised = add_noise_to_motion_deltas(deltas, START_LATLON, config, generator=gen)

            true_pos = compute_positions_from_deltas(START_LATLON, deltas)
            noised_pos = compute_positions_from_deltas(START_LATLON, noised)

            diff = noised_pos - true_pos
            diff_m = torch.stack([
                diff[:, 0] * METERS_PER_DEG_LAT,
                diff[:, 1] * meters_per_deg_lon,
            ], dim=1)
            dist_error = torch.sqrt((diff_m ** 2).sum(dim=1))

            early_errors.append(dist_error[20].item())
            late_errors.append(dist_error[-1].item())

        mean_early = sum(early_errors) / len(early_errors)
        mean_late = sum(late_errors) / len(late_errors)

        # Per-step i.i.d. noise on deltas → position error grows as sqrt(N) (random walk)
        self.assertGreater(mean_late, mean_early * 1.5,
                           f"Late error {mean_late:.1f} should be significantly larger than "
                           f"early error {mean_early:.1f}")


class TestComputePositionsFromDeltas(unittest.TestCase):
    """Test the helper function for integrating deltas to positions."""

    def test_basic_integration(self):
        start = torch.tensor([10.0, 20.0], dtype=torch.float64)
        deltas = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
        positions = compute_positions_from_deltas(start, deltas)

        expected = torch.tensor([
            [10.0, 20.0],
            [11.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
        ], dtype=torch.float64)
        torch.testing.assert_close(positions, expected)

    def test_roundtrip_deltas_to_positions(self):
        """Positions → deltas → positions should be identity."""
        start = torch.tensor([47.6, -122.3], dtype=torch.float64)
        original_deltas = torch.randn(15, 2, dtype=torch.float64) * 0.001

        positions = compute_positions_from_deltas(start, original_deltas)
        recovered_deltas = torch.diff(positions, dim=0)

        torch.testing.assert_close(recovered_deltas, original_deltas, atol=1e-12, rtol=1e-12)


class TestDeviceAndDtype(unittest.TestCase):
    """Output should match input device and dtype."""

    def test_float32_output(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        deltas = _make_straight_north_deltas(5, 50.0).float()
        start = START_LATLON.float()
        result = add_noise_to_motion_deltas(deltas, start, config)
        self.assertEqual(result.dtype, torch.float32)

    def test_float64_output(self):
        config = OdometryNoiseConfig(sigma_noise_frac=0.05)
        deltas = _make_straight_north_deltas(5, 50.0).double()
        start = START_LATLON.double()
        result = add_noise_to_motion_deltas(deltas, start, config)
        self.assertEqual(result.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
