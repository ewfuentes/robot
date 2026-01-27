"""Tests for convergence metrics.

These tests verify the convergence metrics using controlled scenarios where
ground truth values can be computed analytically.
"""

import unittest
import common.torch.load_torch_deps
import torch
import math

from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    get_meters_per_pixel,
    compute_probability_mass_within_radius,
    compute_convergence_cost,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
)
from common.gps import web_mercator
from common.math.haversine import find_d_on_unit_circle

EARTH_RADIUS_M = 6_371_000.0


# =============================================================================
# Test 1: Web Mercator Scale Factor Tests
# =============================================================================


class TestGetMetersPerPixel(unittest.TestCase):
    """Verify get_meters_per_pixel against known values."""

    def test_equator_zoom_20(self):
        """At equator, meters_per_pixel has an exact formula."""
        lat = 0.0
        zoom = 20
        earth_circumference_m = 2 * math.pi * 6_371_000.0
        map_size_px = 2 ** (8 + zoom)
        expected = earth_circumference_m / map_size_px

        result = get_meters_per_pixel(lat, zoom)
        self.assertAlmostEqual(result, expected, places=6)

    def test_lat_60_is_half_equator(self):
        """At lat=60°, cos(60°) = 0.5, so meters_per_pixel is half of equator."""
        zoom = 20
        equator_mpp = get_meters_per_pixel(0.0, zoom)
        lat60_mpp = get_meters_per_pixel(60.0, zoom)

        # cos(60°) = 0.5
        expected_ratio = 0.5
        actual_ratio = lat60_mpp / equator_mpp

        self.assertAlmostEqual(actual_ratio, expected_ratio, places=6)

    def test_cosine_scaling(self):
        """Verify cos(lat) scaling at various latitudes."""
        zoom = 18
        equator_mpp = get_meters_per_pixel(0.0, zoom)

        test_lats = [30.0, 45.0, 60.0, 75.0]
        for lat in test_lats:
            expected_ratio = math.cos(math.radians(lat))
            actual_mpp = get_meters_per_pixel(lat, zoom)
            actual_ratio = actual_mpp / equator_mpp

            self.assertAlmostEqual(
                actual_ratio, expected_ratio, places=6,
                msg=f"At lat={lat}°"
            )

    def test_zoom_level_scaling(self):
        """Each zoom level doubles the map size, halving meters_per_pixel."""
        lat = 40.0
        mpp_z18 = get_meters_per_pixel(lat, 18)
        mpp_z19 = get_meters_per_pixel(lat, 19)
        mpp_z20 = get_meters_per_pixel(lat, 20)

        self.assertAlmostEqual(mpp_z18 / mpp_z19, 2.0, places=6)
        self.assertAlmostEqual(mpp_z19 / mpp_z20, 2.0, places=6)


# =============================================================================
# Test 2: Delta Belief Tests (all mass in single cell)
# =============================================================================


class TestDeltaBelief(unittest.TestCase):
    """Test probability mass computation with all mass concentrated in one cell."""

    def setUp(self):
        """Create a small 5x5 grid centered around New York."""
        # Create grid from valid lat/lon bounds (small area around lat=40.7, lon=-74)
        self.grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.70,
            max_lat=40.71,
            min_lon=-74.01,
            max_lon=-74.00,
            zoom_level=20,
            cell_size_px=100.0,
        )
        self.device = torch.device("cpu")

    def test_delta_at_center_radius_covers(self):
        """All mass in center cell, true position at center, large radius → prob=1.0."""
        belief = HistogramBelief(self.grid_spec, self.device)
        num_rows = self.grid_spec.num_rows
        num_cols = self.grid_spec.num_cols

        # Put all mass in center cell
        center_row = num_rows // 2
        center_col = num_cols // 2
        belief._log_belief = torch.full((num_rows, num_cols), -float("inf"))
        belief._log_belief[center_row, center_col] = 0.0  # log(1) = 0

        # Get center cell's lat/lon
        cell_centers = self.grid_spec.get_all_cell_centers_latlon(self.device)
        center_idx = center_row * num_cols + center_col
        true_latlon = cell_centers[center_idx]

        # Large radius should capture all mass
        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, 1000.0)
        self.assertAlmostEqual(prob_mass, 1.0, places=6)

    def test_delta_at_center_radius_zero(self):
        """All mass in center cell, true position at center, zero radius → prob=0."""
        belief = HistogramBelief(self.grid_spec, self.device)
        num_rows = self.grid_spec.num_rows
        num_cols = self.grid_spec.num_cols

        # Put all mass in center cell
        center_row = num_rows // 2
        center_col = num_cols // 2
        belief._log_belief = torch.full((num_rows, num_cols), -float("inf"))
        belief._log_belief[center_row, center_col] = 0.0

        # Get center cell's lat/lon
        cell_centers = self.grid_spec.get_all_cell_centers_latlon(self.device)
        center_idx = center_row * num_cols + center_col
        true_latlon = cell_centers[center_idx]

        # Zero radius shouldn't capture any mass
        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, 0.0)
        self.assertEqual(prob_mass, 0.0)

    def test_delta_far_from_true_position(self):
        """All mass in corner cell, true position at opposite corner → prob=0 for small radius."""
        belief = HistogramBelief(self.grid_spec, self.device)
        num_rows = self.grid_spec.num_rows
        num_cols = self.grid_spec.num_cols

        # Put all mass in top-left cell (0, 0)
        belief._log_belief = torch.full((num_rows, num_cols), -float("inf"))
        belief._log_belief[0, 0] = 0.0

        # True position at bottom-right cell
        cell_centers = self.grid_spec.get_all_cell_centers_latlon(self.device)
        true_idx = (num_rows - 1) * num_cols + (num_cols - 1)
        true_latlon = cell_centers[true_idx]

        # Small radius shouldn't capture any mass
        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, 1.0)
        self.assertEqual(prob_mass, 0.0)


# =============================================================================
# Test 3: Known Geometry Grid Tests (hand-calculable scenarios)
# =============================================================================


class TestKnownGeometryGrid(unittest.TestCase):
    """Test with small grids where we can hand-calculate expected values."""

    def test_uniform_grid_center_captures_center_cell(self):
        """Uniform grid, true at center, small radius captures only center cell."""
        device = torch.device("cpu")

        # Create grid from valid lat/lon bounds
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.701,
            min_lon=-74.001,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        belief = HistogramBelief.from_uniform(grid_spec, device)
        num_cells = grid_spec.num_rows * grid_spec.num_cols

        # True position at center cell
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_row = grid_spec.num_rows // 2
        center_col = grid_spec.num_cols // 2
        center_idx = center_row * grid_spec.num_cols + center_col
        true_latlon = cell_centers[center_idx]

        # Get cell size in meters at this latitude
        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Radius that captures only the center cell (less than half cell size)
        small_radius = cell_size_m * 0.4

        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, small_radius)

        # Should capture 1/num_cells of the mass (only center cell)
        expected = 1.0 / num_cells
        self.assertAlmostEqual(prob_mass, expected, delta=0.01)

    def test_uniform_grid_large_radius_captures_all(self):
        """Uniform grid, very large radius should capture all mass."""
        device = torch.device("cpu")

        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.701,
            min_lon=-74.001,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        belief = HistogramBelief.from_uniform(grid_spec, device)

        # True position at center
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_row = grid_spec.num_rows // 2
        center_col = grid_spec.num_cols // 2
        center_idx = center_row * grid_spec.num_cols + center_col
        true_latlon = cell_centers[center_idx]

        # Very large radius should capture everything
        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, 10000.0)

        self.assertAlmostEqual(prob_mass, 1.0, places=6)

    def test_two_cells_equal_weight(self):
        """Two cells with equal weight, radius captures one → prob=0.5."""
        device = torch.device("cpu")

        # Create a grid and use only two cells
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.7000,
            max_lat=40.7005,
            min_lon=-74.001,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )
        num_rows = grid_spec.num_rows
        num_cols = grid_spec.num_cols

        belief = HistogramBelief(grid_spec, device)
        # Put equal mass in first and last cells
        belief._log_belief = torch.full((num_rows, num_cols), -float("inf"))
        belief._log_belief[0, 0] = math.log(0.5)
        belief._log_belief[num_rows - 1, num_cols - 1] = math.log(0.5)

        # True position at first cell (0, 0)
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        true_latlon = cell_centers[0]

        # Get cell size in meters
        mpp = get_meters_per_pixel(true_latlon[0].item(), grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Radius that captures only the first cell
        small_radius = cell_size_m * 0.4

        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, small_radius)

        # Should capture only first cell's mass = 0.5
        self.assertAlmostEqual(prob_mass, 0.5, delta=0.01)

    def test_weighted_cells_large_radius(self):
        """Non-uniform weights, large radius captures all."""
        device = torch.device("cpu")

        # Create grid from valid lat/lon bounds
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.7000,
            max_lat=40.7002,
            min_lon=-74.0002,
            max_lon=-74.0000,
            zoom_level=20,
            cell_size_px=100.0,
        )
        num_rows = grid_spec.num_rows
        num_cols = grid_spec.num_cols

        belief = HistogramBelief(grid_spec, device)
        # Set probabilities that sum to 1
        total_cells = num_rows * num_cols
        probs = torch.arange(1, total_cells + 1, dtype=torch.float32).reshape(num_rows, num_cols)
        probs = probs / probs.sum()
        belief._log_belief = torch.log(probs)

        # True position at center of grid
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        true_latlon = cell_centers.mean(dim=0)

        # Large radius captures all
        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, 10000.0)
        self.assertAlmostEqual(prob_mass, 1.0, places=6)


# =============================================================================
# Test 4: Convergence Cost Analytical Tests
# =============================================================================


class TestConvergenceCostAnalytical(unittest.TestCase):
    """Test convergence cost computation against analytically derived values."""

    def test_constant_probability_mass(self):
        """Constant prob_mass → cost = (1 - p) * total_distance."""
        n_steps = 10
        p = 0.6
        total_distance = 100.0

        prob_mass = torch.full((n_steps + 1,), p)  # +1 for initial
        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Expected: (1 - 0.6) * 100 = 40
        expected = (1 - p) * total_distance
        self.assertAlmostEqual(cost, expected, delta=1.0)

    def test_zero_probability_mass(self):
        """Zero prob_mass throughout → cost = total_distance."""
        n_steps = 10
        total_distance = 100.0

        prob_mass = torch.zeros(n_steps + 1)
        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Expected: 1.0 * 100 = 100
        expected = total_distance
        self.assertAlmostEqual(cost, expected, delta=1.0)

    def test_full_probability_mass(self):
        """Full prob_mass = 1.0 throughout → cost = 0."""
        n_steps = 10
        total_distance = 100.0

        prob_mass = torch.ones(n_steps + 1)
        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Expected: 0
        self.assertAlmostEqual(cost, 0.0, places=6)

    def test_step_function_at_midpoint(self):
        """prob_mass jumps from 0 to 1 at midpoint → cost ≈ total_distance / 2."""
        n_steps = 100
        total_distance = 100.0

        # Create step function: 0 for first half, 1 for second half
        prob_mass = torch.zeros(n_steps + 1)
        prob_mass[n_steps // 2:] = 1.0

        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Expected: ~50 (integral of (1-0) from 0 to 50)
        expected = total_distance / 2
        self.assertAlmostEqual(cost, expected, delta=2.0)

    def test_linear_ramp(self):
        """prob_mass ramps linearly from 0 to 1 → cost ≈ total_distance / 2."""
        n_steps = 100
        total_distance = 100.0

        # Linear ramp from 0 to 1
        prob_mass = torch.linspace(0, 1, n_steps + 1)
        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Integral of (1 - x/L) from 0 to L = L - L/2 = L/2
        expected = total_distance / 2
        self.assertAlmostEqual(cost, expected, delta=2.0)

    def test_quadratic_convergence(self):
        """prob_mass = (d/D)² → cost can be computed analytically."""
        n_steps = 100
        total_distance = 100.0

        # Quadratic: p(d) = (d/D)²
        d_values = torch.linspace(0, total_distance, n_steps + 1)
        prob_mass = (d_values / total_distance) ** 2

        distance_traveled = torch.linspace(0, total_distance, n_steps)

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # Integral of (1 - (x/L)²) from 0 to L = L - L/3 = 2L/3
        expected = 2 * total_distance / 3
        self.assertAlmostEqual(cost, expected, delta=2.0)

    def test_empty_inputs(self):
        """Edge case: very short inputs should return 0."""
        prob_mass = torch.tensor([0.5])
        distance_traveled = torch.tensor([0.0])

        cost = compute_convergence_cost(prob_mass, distance_traveled)
        self.assertEqual(cost, 0.0)

    def test_two_steps(self):
        """Minimum valid case with actual distance segment."""
        prob_mass = torch.tensor([0.0, 0.0, 1.0])  # Initial, step 0, step 1
        distance_traveled = torch.tensor([0.0, 50.0])  # At step 0 and step 1

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        # delta_dist = [50]
        # We use prob_mass[2:] = [1.0], so missing_prob = [0.0]
        # cost = 0 * 50 = 0
        expected = 0.0
        self.assertAlmostEqual(cost, expected, places=6)


# =============================================================================
# Test 5: Integration / Round-Trip Tests
# =============================================================================


class TestIntegration(unittest.TestCase):
    """Integration tests verifying consistency across the full pipeline."""

    def test_symmetry_around_true_position(self):
        """Symmetric belief around true position should give positive prob mass."""
        device = torch.device("cpu")

        # Create grid from valid lat/lon bounds
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.705,
            min_lon=-74.005,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        # Create symmetric belief (Gaussian-like, centered at grid center)
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_row = grid_spec.num_rows // 2
        center_col = grid_spec.num_cols // 2
        center_idx = center_row * grid_spec.num_cols + center_col
        center_latlon = cell_centers[center_idx]

        belief = HistogramBelief.from_gaussian(
            grid_spec, center_latlon, std_deg=0.0001, device=device
        )

        # Probability mass should be positive for any reasonable radius
        prob_mass = compute_probability_mass_within_radius(belief, center_latlon, 5.0)
        self.assertGreater(prob_mass, 0)

    def test_increasing_radius_increases_prob_mass(self):
        """Larger radius should always capture more or equal probability mass."""
        device = torch.device("cpu")

        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.710,
            min_lon=-74.010,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_row = grid_spec.num_rows // 2
        center_col = grid_spec.num_cols // 2
        center_idx = center_row * grid_spec.num_cols + center_col
        true_latlon = cell_centers[center_idx]

        radii = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
        prev_mass = 0.0

        for radius in radii:
            mass = compute_probability_mass_within_radius(belief, true_latlon, radius)
            self.assertGreaterEqual(
                mass, prev_mass - 1e-6,
                msg=f"Probability mass decreased: {prev_mass} -> {mass} at radius {radius}"
            )
            prev_mass = mass

    def test_prob_mass_bounded_zero_one(self):
        """Probability mass should always be in [0, 1]."""
        device = torch.device("cpu")

        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.705,
            min_lon=-74.005,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        num_cells = grid_spec.num_rows * grid_spec.num_cols

        # Test various positions and radii
        test_indices = [0, num_cells // 2, num_cells - 1]
        for idx in test_indices:
            true_latlon = cell_centers[idx]
            for radius in [0.0, 1.0, 10.0, 100.0, 10000.0]:
                mass = compute_probability_mass_within_radius(belief, true_latlon, radius)
                self.assertGreaterEqual(mass, 0.0)
                self.assertLessEqual(mass, 1.0 + 1e-6)


# =============================================================================
# Test 6: Haversine Cross-Validation
# =============================================================================


class TestHaversineCrossValidation(unittest.TestCase):
    """Cross-validate our distance calculations with haversine formula."""

    def test_meters_per_pixel_vs_haversine_equator(self):
        """At equator, verify meters_per_pixel against haversine for a known pixel distance."""
        zoom = 20
        lat = 0.0

        # Get meters per pixel from our function
        mpp = get_meters_per_pixel(lat, zoom)

        # Verify using haversine: move 1000 pixels east at equator
        px_distance = 1000.0
        expected_meters = mpp * px_distance

        # Get lon change for this pixel distance
        map_size_px = 2 ** (8 + zoom)
        degrees_per_pixel = 360.0 / map_size_px
        lon_change = px_distance * degrees_per_pixel

        # Use haversine to compute actual distance
        p1 = (lat, 0.0)
        p2 = (lat, lon_change)
        d_unit_circle = find_d_on_unit_circle(p1, p2)
        haversine_meters = d_unit_circle * EARTH_RADIUS_M

        # Should be very close (within 0.1%)
        rel_error = abs(expected_meters - haversine_meters) / haversine_meters
        self.assertLess(rel_error, 0.001, f"Relative error {rel_error:.4f} too large")

    def test_meters_per_pixel_vs_haversine_midlat(self):
        """At mid-latitude (45°), verify meters_per_pixel against haversine."""
        zoom = 20
        lat = 45.0

        mpp = get_meters_per_pixel(lat, zoom)
        px_distance = 1000.0
        expected_meters = mpp * px_distance

        # At non-equator, we need to account for web mercator projection
        # The y pixel coordinate changes with latitude
        # For east-west movement, lon change is same as at equator
        map_size_px = 2 ** (8 + zoom)
        degrees_per_pixel = 360.0 / map_size_px
        lon_change = px_distance * degrees_per_pixel

        p1 = (lat, 0.0)
        p2 = (lat, lon_change)
        d_unit_circle = find_d_on_unit_circle(p1, p2)
        haversine_meters = d_unit_circle * EARTH_RADIUS_M

        # Should be very close (within 0.1%)
        rel_error = abs(expected_meters - haversine_meters) / haversine_meters
        self.assertLess(rel_error, 0.001, f"Relative error {rel_error:.4f} too large")

    def test_cell_distance_vs_haversine(self):
        """Verify cell center distances computed via pixels match haversine."""
        device = torch.device("cpu")

        # Create a small grid
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=40.700,
            max_lat=40.710,
            min_lon=-74.010,
            max_lon=-74.000,
            zoom_level=20,
            cell_size_px=100.0,
        )

        cell_centers_latlon = grid_spec.get_all_cell_centers_latlon(device)
        cell_centers_px = grid_spec.get_all_cell_centers_px(device)

        # Pick two cells far apart
        idx1 = 0
        idx2 = len(cell_centers_latlon) - 1

        lat1, lon1 = cell_centers_latlon[idx1].tolist()
        lat2, lon2 = cell_centers_latlon[idx2].tolist()

        # Haversine distance
        d_unit = find_d_on_unit_circle((lat1, lon1), (lat2, lon2))
        haversine_dist_m = d_unit * EARTH_RADIUS_M

        # Pixel-based distance (using our mpp approximation)
        px1 = cell_centers_px[idx1]
        px2 = cell_centers_px[idx2]
        px_dist = torch.norm(px2 - px1).item()

        # Use average latitude for mpp
        avg_lat = (lat1 + lat2) / 2
        mpp = get_meters_per_pixel(avg_lat, grid_spec.zoom_level)
        pixel_based_dist_m = px_dist * mpp

        # For small distances (< 1km), should be within 1%
        rel_error = abs(pixel_based_dist_m - haversine_dist_m) / haversine_dist_m
        self.assertLess(
            rel_error, 0.01,
            f"Distance mismatch: pixel={pixel_based_dist_m:.2f}m, haversine={haversine_dist_m:.2f}m"
        )


# =============================================================================
# Test 7: Exact Geometric Cell Layout Tests
# =============================================================================


class TestExactGeometricCellLayout(unittest.TestCase):
    """Test with precisely constructed grids where cell counts are known exactly."""

    def _make_grid_with_exact_size(self, num_rows: int, num_cols: int, cell_size_px: float = 100.0):
        """Create a grid with exact dimensions using valid lat/lon bounds."""
        # Use small bounds around NYC that will create the desired grid size
        # The from_bounds_and_cell_size method computes num_rows and num_cols based on bounds
        # For precise control, we'll use a different approach: compute valid pixel coords

        # At zoom 20, the equator is at row 2^27 = 134217728
        # Latitude 40.7° (NYC) is approximately at row 134217728 - offset
        # Let's compute valid origin for lat ~40.7
        base_row = 100_000_000.0  # This maps to roughly lat ~64° which is valid
        base_col = 100_000_000.0

        return GridSpec(
            origin_row_px=base_row,
            origin_col_px=base_col,
            num_rows=num_rows,
            num_cols=num_cols,
            zoom_level=20,
            cell_size_px=cell_size_px,
        )

    def test_known_3x3_grid_center_cell(self):
        """3x3 grid with uniform belief, center position captures 1/9."""
        device = torch.device("cpu")

        grid_spec = self._make_grid_with_exact_size(3, 3)

        belief = HistogramBelief.from_uniform(grid_spec, device)

        # Get center cell lat/lon
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_idx = 4  # Center of 3x3 = index 4
        true_latlon = cell_centers[center_idx]

        # Get cell size in meters
        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Radius that captures only center (< half diagonal)
        small_radius = cell_size_m * 0.4

        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, small_radius)

        # Should capture 1/9 of mass
        self.assertAlmostEqual(prob_mass, 1.0 / 9.0, delta=0.001)

    def test_known_3x3_grid_all_cells(self):
        """3x3 grid, radius that captures all cells should give prob=1.0."""
        device = torch.device("cpu")

        grid_spec = self._make_grid_with_exact_size(3, 3)

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        center_idx = 4
        true_latlon = cell_centers[center_idx]

        # Get cell size in meters
        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Radius that captures all 9 cells (diagonal of 3x3 grid)
        diagonal = math.sqrt(2) * 3 * cell_size_m
        large_radius = diagonal

        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, large_radius)
        self.assertAlmostEqual(prob_mass, 1.0, places=4)

    def test_2x2_grid_corner_position(self):
        """2x2 grid, position at one corner, small radius captures 1/4."""
        device = torch.device("cpu")

        grid_spec = self._make_grid_with_exact_size(2, 2)

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        corner_idx = 0  # Top-left corner
        true_latlon = cell_centers[corner_idx]

        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        small_radius = cell_size_m * 0.4

        prob_mass = compute_probability_mass_within_radius(belief, true_latlon, small_radius)
        self.assertAlmostEqual(prob_mass, 0.25, delta=0.001)

    def test_4x1_row_grid_capture_count(self):
        """4x1 row grid, verify we can selectively capture 1, 2, 3, or 4 cells."""
        device = torch.device("cpu")

        grid_spec = self._make_grid_with_exact_size(1, 4)

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        # Position at leftmost cell (index 0)
        true_latlon = cell_centers[0]

        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Test radii that capture 1, 2, 3, 4 cells
        test_cases = [
            (cell_size_m * 0.4, 1),      # Just the first cell
            (cell_size_m * 1.1, 2),      # First two cells
            (cell_size_m * 2.1, 3),      # First three cells
            (cell_size_m * 3.5, 4),      # All four cells
        ]

        for radius, expected_cells in test_cases:
            prob_mass = compute_probability_mass_within_radius(belief, true_latlon, radius)
            expected_prob = expected_cells / 4.0
            self.assertAlmostEqual(
                prob_mass, expected_prob, delta=0.01,
                msg=f"Radius {radius:.1f}m: expected {expected_cells}/4={expected_prob}, got {prob_mass}"
            )


# =============================================================================
# Test 8: Exhaustive Invariant Tests
# =============================================================================


class TestExhaustiveInvariants(unittest.TestCase):
    """Property-based testing without hypothesis: test invariants across many cases."""

    def _make_valid_grid(self, num_rows: int, num_cols: int, zoom: int):
        """Create a grid with valid lat/lon coordinates."""
        # Map size at zoom level z is 2^(8+z) pixels
        # Use a pixel coordinate at 40% of the map size for ~mid-latitude
        map_size = 2 ** (8 + zoom)
        origin = map_size * 0.4
        return GridSpec(
            origin_row_px=origin,
            origin_col_px=origin,
            num_rows=num_rows,
            num_cols=num_cols,
            zoom_level=zoom,
            cell_size_px=100.0,
        )

    def test_monotonicity_across_radii_many_grids(self):
        """Test that prob_mass is monotonically increasing with radius."""
        device = torch.device("cpu")

        # Test various grid sizes
        grid_sizes = [(3, 3), (5, 5), (10, 10), (7, 4)]
        zoom_levels = [18, 19, 20]

        for num_rows, num_cols in grid_sizes:
            for zoom in zoom_levels:
                grid_spec = self._make_valid_grid(num_rows, num_cols, zoom)

                belief = HistogramBelief.from_uniform(grid_spec, device)

                cell_centers = grid_spec.get_all_cell_centers_latlon(device)
                center_idx = (num_rows * num_cols) // 2
                true_latlon = cell_centers[center_idx]

                # Test increasing radii
                radii = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
                prev_mass = 0.0

                for radius in radii:
                    mass = compute_probability_mass_within_radius(belief, true_latlon, radius)
                    self.assertGreaterEqual(
                        mass, prev_mass - 1e-6,
                        msg=f"Grid {num_rows}x{num_cols} zoom={zoom}: "
                            f"monotonicity violated at radius {radius}"
                    )
                    prev_mass = mass

    def test_bounds_invariant_many_positions(self):
        """Test that prob_mass is always in [0, 1] for many positions."""
        device = torch.device("cpu")

        grid_spec = self._make_valid_grid(10, 10, 20)

        # Test various belief distributions
        for belief_type in ["uniform", "concentrated", "spread"]:
            if belief_type == "uniform":
                belief = HistogramBelief.from_uniform(grid_spec, device)
            elif belief_type == "concentrated":
                belief = HistogramBelief(grid_spec, device)
                belief._log_belief = torch.full((10, 10), -float("inf"))
                belief._log_belief[5, 5] = 0.0  # All mass at center
            else:  # spread
                belief = HistogramBelief(grid_spec, device)
                # Arbitrary spread
                log_probs = torch.randn(10, 10)
                belief._log_belief = log_probs - torch.logsumexp(log_probs.flatten(), dim=0)

            cell_centers = grid_spec.get_all_cell_centers_latlon(device)

            # Test all cells as true positions
            for idx in range(len(cell_centers)):
                true_latlon = cell_centers[idx]

                for radius in [0.0, 1.0, 10.0, 100.0, 10000.0]:
                    mass = compute_probability_mass_within_radius(belief, true_latlon, radius)
                    self.assertGreaterEqual(
                        mass, 0.0,
                        msg=f"Negative mass at idx={idx}, radius={radius}, type={belief_type}"
                    )
                    self.assertLessEqual(
                        mass, 1.0 + 1e-6,
                        msg=f"Mass > 1 at idx={idx}, radius={radius}, type={belief_type}"
                    )

    def test_belief_sum_invariant(self):
        """Large radius should always capture probability summing to ~1.0."""
        device = torch.device("cpu")

        grid_sizes = [(3, 3), (5, 7), (10, 10)]

        for num_rows, num_cols in grid_sizes:
            grid_spec = self._make_valid_grid(num_rows, num_cols, 20)

            belief = HistogramBelief.from_uniform(grid_spec, device)

            cell_centers = grid_spec.get_all_cell_centers_latlon(device)
            center_idx = (num_rows * num_cols) // 2
            true_latlon = cell_centers[center_idx]

            # Very large radius should capture everything
            mass = compute_probability_mass_within_radius(belief, true_latlon, 100000.0)
            self.assertAlmostEqual(
                mass, 1.0, places=4,
                msg=f"Grid {num_rows}x{num_cols}: large radius didn't capture all mass"
            )


# =============================================================================
# Test 9: Numerical Edge Cases with Log Probs
# =============================================================================


class TestNumericalEdgeCasesLogProbs(unittest.TestCase):
    """Test numerical stability with extreme log probabilities."""

    def _make_valid_grid(self, num_rows: int, num_cols: int):
        """Create a grid with valid lat/lon coordinates."""
        return GridSpec(
            origin_row_px=100_000_000.0,
            origin_col_px=100_000_000.0,
            num_rows=num_rows,
            num_cols=num_cols,
            zoom_level=20,
            cell_size_px=100.0,
        )

    def test_very_small_probabilities(self):
        """Test with very small (but valid) probabilities."""
        device = torch.device("cpu")

        grid_spec = self._make_valid_grid(4, 4)

        belief = HistogramBelief(grid_spec, device)

        # Set very small but valid log probabilities
        # Most cells have log prob = -100 (prob ≈ 3.7e-44)
        # One cell has most of the mass
        belief._log_belief = torch.full((4, 4), -100.0)
        belief._log_belief[2, 2] = 0.0  # This cell has almost all mass

        # Normalize (most mass goes to [2, 2])
        log_sum = torch.logsumexp(belief._log_belief.flatten(), dim=0)
        belief._log_belief = belief._log_belief - log_sum

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)

        # Position at the high-prob cell should give ~1.0
        center_idx = 2 * 4 + 2
        true_latlon = cell_centers[center_idx]
        mass = compute_probability_mass_within_radius(belief, true_latlon, 1000.0)
        self.assertAlmostEqual(mass, 1.0, places=4)

    def test_single_cell_grid(self):
        """Test edge case with a 1x1 grid."""
        device = torch.device("cpu")

        grid_spec = self._make_valid_grid(1, 1)

        belief = HistogramBelief.from_uniform(grid_spec, device)

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        true_latlon = cell_centers[0]

        # Any positive radius should capture all mass (only one cell)
        mass_small = compute_probability_mass_within_radius(belief, true_latlon, 0.1)
        mass_large = compute_probability_mass_within_radius(belief, true_latlon, 1000.0)

        # Zero radius at exact cell center: distance=0, and 0 <= 0 is True, so captures the cell
        mass_zero = compute_probability_mass_within_radius(belief, true_latlon, 0.0)

        # When true position is exactly at cell center, even radius=0 captures it (dist=0 <= 0)
        self.assertAlmostEqual(mass_zero, 1.0, places=6)
        self.assertAlmostEqual(mass_large, 1.0, places=6)
        # Small radius should also capture the only cell
        self.assertGreater(mass_small, 0.0)

    def test_concentrated_belief_numerical_stability(self):
        """Test with belief concentrated at a single cell (log prob = 0, others = -inf)."""
        device = torch.device("cpu")

        grid_spec = self._make_valid_grid(5, 5)

        belief = HistogramBelief(grid_spec, device)
        belief._log_belief = torch.full((5, 5), -float("inf"))
        belief._log_belief[3, 3] = 0.0  # All mass here

        cell_centers = grid_spec.get_all_cell_centers_latlon(device)

        # True position at the concentrated cell
        idx = 3 * 5 + 3
        true_latlon = cell_centers[idx]

        center_lat = true_latlon[0].item()
        mpp = get_meters_per_pixel(center_lat, grid_spec.zoom_level)
        cell_size_m = grid_spec.cell_size_px * mpp

        # Small radius captures the cell
        mass = compute_probability_mass_within_radius(belief, true_latlon, cell_size_m * 0.5)
        self.assertAlmostEqual(mass, 1.0, places=6)

        # True position far from concentrated cell
        far_idx = 0
        far_latlon = cell_centers[far_idx]
        mass_far = compute_probability_mass_within_radius(belief, far_latlon, cell_size_m * 0.5)
        self.assertAlmostEqual(mass_far, 0.0, places=6)

    def test_log_sum_exp_stability(self):
        """Verify belief is properly normalized when using log probs."""
        device = torch.device("cpu")

        grid_spec = self._make_valid_grid(3, 3)

        # Create belief with manually set log probs
        belief = HistogramBelief(grid_spec, device)

        # Set raw log probs
        raw_log_probs = torch.tensor([
            [-10.0, -5.0, -10.0],
            [-5.0, 0.0, -5.0],
            [-10.0, -5.0, -10.0],
        ])

        # Normalize properly
        belief._log_belief = raw_log_probs - torch.logsumexp(raw_log_probs.flatten(), dim=0)

        # Verify normalization
        total_prob = belief.get_belief().sum().item()
        self.assertAlmostEqual(total_prob, 1.0, places=6)

        # Now test prob mass - large radius should get all
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)
        true_latlon = cell_centers[4]  # Center

        mass = compute_probability_mass_within_radius(belief, true_latlon, 10000.0)
        self.assertAlmostEqual(mass, 1.0, places=4)


# =============================================================================
# Test 10: Explicit Alignment Tests for Convergence Cost
# =============================================================================


class TestConvergenceCostAlignment(unittest.TestCase):
    """Verify correct alignment between prob_mass and distance_traveled arrays."""

    def test_indexing_manual_verification(self):
        """Manually verify the index alignment in convergence cost computation."""
        # prob_mass has (path_len + 1) entries:
        #   [0] = initial (before any observation)
        #   [1] = after step 0
        #   [2] = after step 1
        #   ...
        #   [path_len] = after step (path_len - 1)

        # distance_traveled has (path_len) entries:
        #   [0] = cumulative distance at step 0
        #   [1] = cumulative distance at step 1
        #   ...
        #   [path_len - 1] = cumulative distance at step (path_len - 1)

        # The cost integrates (1 - prob_mass) * delta_distance
        # delta_distance[i] = distance_traveled[i+1] - distance_traveled[i]
        # This represents distance from step i to step i+1

        # For alignment, we use prob_mass at the END of each segment:
        # Segment from step i to i+1 uses prob_mass[i+2]
        # (because prob_mass[i+1] is "after step i", so prob_mass[i+2] is "after step i+1")

        path_len = 5

        # Create test data where we can verify by hand
        # prob_mass: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  (6 entries for path_len=5)
        # distance:  [0, 10, 20, 30, 40]              (5 entries)
        # delta_dist = [10, 10, 10, 10]               (4 segments)

        # According to our formula:
        # missing_prob = 1 - prob_mass[2:6] = [0.8, 0.7, 0.6, 0.5]
        # cost = 0.8*10 + 0.7*10 + 0.6*10 + 0.5*10 = 26

        prob_mass = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        distance_traveled = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0])

        cost = compute_convergence_cost(prob_mass, distance_traveled)

        expected = 0.8 * 10 + 0.7 * 10 + 0.6 * 10 + 0.5 * 10
        self.assertAlmostEqual(cost, expected, places=4)

    def test_alignment_with_step_change(self):
        """Test alignment by introducing a step change at a known point."""
        # If prob_mass jumps from 0 to 1 at step 3, the cost should be
        # the integral of 1.0 from step 0 to step 3

        path_len = 6
        prob_mass = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # 7 entries
        distance_traveled = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])

        # delta_dist = [10, 10, 10, 10, 10]  (5 segments)
        # missing_prob = 1 - prob_mass[2:7] = [1, 1, 0, 0, 0]
        # cost = 1*10 + 1*10 + 0*10 + 0*10 + 0*10 = 20

        cost = compute_convergence_cost(prob_mass, distance_traveled)
        self.assertAlmostEqual(cost, 20.0, places=4)

    def test_varying_step_sizes(self):
        """Test with non-uniform step sizes to verify delta_dist computation."""
        # prob_mass constant at 0.5
        # distance: [0, 5, 15, 30, 50]  (steps of 5, 10, 15, 20)
        # delta_dist = [5, 10, 15, 20]
        # cost = 0.5 * (5 + 10 + 15 + 20) = 0.5 * 50 = 25

        prob_mass = torch.full((6,), 0.5)  # path_len=5, so 6 entries
        distance_traveled = torch.tensor([0.0, 5.0, 15.0, 30.0, 50.0])

        cost = compute_convergence_cost(prob_mass, distance_traveled)
        self.assertAlmostEqual(cost, 25.0, places=4)

    def test_edge_case_path_len_2(self):
        """Minimum valid case: path_len=2 (one segment to compute)."""
        # prob_mass: [0.0, 0.0, 0.8]  (3 entries)
        # distance: [0, 100]          (2 entries)
        # delta_dist = [100]          (1 segment)
        # missing_prob = 1 - prob_mass[2:3] = [0.2]
        # cost = 0.2 * 100 = 20

        prob_mass = torch.tensor([0.0, 0.0, 0.8])
        distance_traveled = torch.tensor([0.0, 100.0])

        cost = compute_convergence_cost(prob_mass, distance_traveled)
        self.assertAlmostEqual(cost, 20.0, places=4)

    def test_late_convergence_vs_early_convergence(self):
        """Verify that late convergence has higher cost than early convergence."""
        # Early convergence: prob_mass jumps to 1.0 early
        # Late convergence: prob_mass stays at 0.0 then jumps to 1.0 late

        path_len = 10
        distance = torch.linspace(0, 100, path_len)

        # Early: jumps at step 2
        early_prob = torch.zeros(path_len + 1)
        early_prob[3:] = 1.0

        # Late: jumps at step 8
        late_prob = torch.zeros(path_len + 1)
        late_prob[9:] = 1.0

        early_cost = compute_convergence_cost(early_prob, distance)
        late_cost = compute_convergence_cost(late_prob, distance)

        self.assertLess(early_cost, late_cost)
        # More specifically:
        # Early cost ≈ (1.0) * distance from 0 to step 2
        # Late cost ≈ (1.0) * distance from 0 to step 8


if __name__ == "__main__":
    unittest.main()
