import unittest
import common.torch.load_torch_deps
import torch
import math

from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    CellToPatchMapping,
    _make_gaussian_kernel_1d,
    _shift_grid,
    segment_logsumexp,
    build_cell_to_patch_mapping,
)


class TestGridSpec(unittest.TestCase):
    def test_from_bounds_creates_valid_grid(self):
        """Test that GridSpec.from_bounds_and_cell_size creates a valid grid."""
        # San Francisco area bounds (roughly)
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.8,
            min_lon=-122.5,
            max_lon=-122.4,
            zoom_level=20,
            cell_size_px=160.0,
        )

        self.assertGreater(grid_spec.num_rows, 0)
        self.assertGreater(grid_spec.num_cols, 0)
        self.assertEqual(grid_spec.zoom_level, 20)
        self.assertEqual(grid_spec.cell_size_px, 160.0)

    def test_latlon_to_cell_indices_roundtrip(self):
        """Test that converting lat/lon to cell indices and back is consistent."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.8,
            min_lon=-122.5,
            max_lon=-122.4,
            zoom_level=20,
            cell_size_px=160.0,
        )

        # Test point in the middle of the grid
        lat = torch.tensor(37.75)
        lon = torch.tensor(-122.45)

        row_idx, col_idx = grid_spec.latlon_to_cell_indices(lat, lon)
        lat_back, lon_back = grid_spec.cell_indices_to_latlon(
            row_idx.float(), col_idx.float()
        )

        # Should be within one cell of original
        lat_diff = abs(lat_back.item() - lat.item())
        lon_diff = abs(lon_back.item() - lon.item())

        # At zoom 20, 160px is roughly 24m, which is ~0.0002 degrees
        self.assertLess(lat_diff, 0.001, f"Latitude diff too large: {lat_diff}")
        self.assertLess(lon_diff, 0.001, f"Longitude diff too large: {lon_diff}")

    def test_get_all_cell_centers_shape(self):
        """Test that get_all_cell_centers_latlon returns correct shape."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.8,
            min_lon=-122.5,
            max_lon=-122.4,
            zoom_level=20,
            cell_size_px=160.0,
        )

        centers = grid_spec.get_all_cell_centers_latlon(torch.device("cpu"))

        expected_count = grid_spec.num_rows * grid_spec.num_cols
        self.assertEqual(centers.shape, (expected_count, 2))

    def test_cell_indices_in_bounds(self):
        """Test that points inside bounds map to valid cell indices."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.8,
            min_lon=-122.5,
            max_lon=-122.4,
            zoom_level=20,
            cell_size_px=160.0,
        )

        # Point in center
        lat = torch.tensor(37.75)
        lon = torch.tensor(-122.45)
        row_idx, col_idx = grid_spec.latlon_to_cell_indices(lat, lon)

        self.assertGreaterEqual(row_idx.item(), 0)
        self.assertLess(row_idx.item(), grid_spec.num_rows)
        self.assertGreaterEqual(col_idx.item(), 0)
        self.assertLess(col_idx.item(), grid_spec.num_cols)


class TestGaussianKernel(unittest.TestCase):
    def test_kernel_sums_to_one(self):
        """Test that Gaussian kernel sums to 1."""
        kernel = _make_gaussian_kernel_1d(2.0, torch.device("cpu"))
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=5)

    def test_kernel_is_symmetric(self):
        """Test that Gaussian kernel is symmetric."""
        kernel = _make_gaussian_kernel_1d(2.0, torch.device("cpu"))
        self.assertTrue(torch.allclose(kernel, kernel.flip(0)))

    def test_small_sigma_returns_identity(self):
        """Test that very small sigma returns identity kernel."""
        kernel = _make_gaussian_kernel_1d(0.05, torch.device("cpu"))
        self.assertEqual(len(kernel), 1)
        self.assertEqual(kernel[0].item(), 1.0)


class TestShiftGrid(unittest.TestCase):
    def test_zero_shift_preserves_grid(self):
        """Test that zero shift doesn't change the grid."""
        log_belief = torch.randn(10, 10)
        shifted = _shift_grid(log_belief, 0.0, 0.0)

        # Should be very close (bilinear interpolation at integer points)
        self.assertTrue(torch.allclose(shifted, log_belief, atol=1e-4))

    def test_integer_shift(self):
        """Test that integer shift moves the grid correctly."""
        # Create a grid with a single peak
        log_belief = torch.full((10, 10), -10.0)
        log_belief[5, 5] = 0.0  # Peak at center

        # Shift by 2 cells right and 1 cell down
        shifted = _shift_grid(log_belief, 1.0, 2.0)

        # Peak should have moved (some spreading due to bilinear interpolation)
        # The new peak should be near (6, 7)
        max_idx = shifted.argmax()
        max_row = max_idx // 10
        max_col = max_idx % 10

        self.assertEqual(max_row.item(), 6, f"Expected row 6, got {max_row}")
        self.assertEqual(max_col.item(), 7, f"Expected col 7, got {max_col}")

    def test_shift_out_of_bounds_zeros(self):
        """Test that shifting out of bounds produces zeros (absorbing boundary)."""
        log_belief = torch.zeros((10, 10))  # All probability = 1

        # Shift far right - most content should fall off
        shifted = _shift_grid(log_belief, 0.0, 8.0)

        # Left side should now be very low probability (-inf after log)
        self.assertLess(shifted[:, :3].max().item(), -30)


class TestHistogramBelief(unittest.TestCase):
    def test_uniform_sums_to_one(self):
        """Test that uniform belief sums to 1."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        belief = HistogramBelief.from_uniform(grid_spec, torch.device("cpu"))

        prob_sum = belief.get_belief().sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_gaussian_sums_to_one(self):
        """Test that Gaussian belief sums to 1."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        mean = torch.tensor([37.705, -122.495])
        belief = HistogramBelief.from_gaussian(
            grid_spec, mean, std_deg=0.002, device=torch.device("cpu")
        )

        prob_sum = belief.get_belief().sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_gaussian_mean_near_specified(self):
        """Test that Gaussian belief mean is near the specified center."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        mean = torch.tensor([37.705, -122.495])
        belief = HistogramBelief.from_gaussian(
            grid_spec, mean, std_deg=0.002, device=torch.device("cpu")
        )

        computed_mean = belief.get_mean_latlon()

        # Should be very close to specified mean
        self.assertAlmostEqual(computed_mean[0].item(), mean[0].item(), places=3)
        self.assertAlmostEqual(computed_mean[1].item(), mean[1].item(), places=3)

    def test_normalize_preserves_relative_probabilities(self):
        """Test that normalize preserves relative probabilities."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        belief = HistogramBelief(grid_spec, torch.device("cpu"))

        # Set some arbitrary log probabilities
        belief._log_belief = torch.randn(grid_spec.num_rows, grid_spec.num_cols)
        original_diff = belief._log_belief[0, 0] - belief._log_belief[0, 1]

        belief.normalize()

        # Relative differences should be preserved
        new_diff = belief._log_belief[0, 0] - belief._log_belief[0, 1]
        self.assertAlmostEqual(original_diff.item(), new_diff.item(), places=4)

    def test_motion_blur_preserves_probability_mass(self):
        """Test that motion blur approximately preserves total probability."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        mean = torch.tensor([37.705, -122.495])
        belief = HistogramBelief.from_gaussian(
            grid_spec, mean, std_deg=0.001, device=torch.device("cpu")
        )

        prob_before = belief.get_belief().sum().item()
        belief.apply_motion_blur(sigma_cells=2.0)
        prob_after = belief.get_belief().sum().item()

        # Some probability may be lost at edges, but should be close
        # (For a centered Gaussian with small blur, loss should be minimal)
        self.assertGreater(prob_after, 0.9 * prob_before)

    def test_motion_blur_increases_variance(self):
        """Test that motion blur increases the variance of the belief."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        mean = torch.tensor([37.705, -122.495])
        belief = HistogramBelief.from_gaussian(
            grid_spec, mean, std_deg=0.001, device=torch.device("cpu")
        )

        var_before = belief.get_variance_deg_sq().sum().item()
        belief.apply_motion_blur(sigma_cells=2.0)
        belief.normalize()
        var_after = belief.get_variance_deg_sq().sum().item()

        self.assertGreater(var_after, var_before)

    def test_apply_motion_shifts_mean(self):
        """Test that apply_motion shifts the belief mean."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        mean = torch.tensor([37.705, -122.495])
        belief = HistogramBelief.from_gaussian(
            grid_spec, mean, std_deg=0.001, device=torch.device("cpu")
        )

        mean_before = belief.get_mean_latlon()

        # Apply motion: move north (+lat) and east (+lon)
        motion_delta = torch.tensor([0.001, 0.001])
        belief.apply_motion(motion_delta, noise_percent=0.02)

        mean_after = belief.get_mean_latlon()

        # Mean should have shifted in the direction of motion
        self.assertGreater(mean_after[0].item(), mean_before[0].item(),
                          "Latitude should have increased")
        self.assertGreater(mean_after[1].item(), mean_before[1].item(),
                          "Longitude should have increased")

    def test_clone_is_independent(self):
        """Test that cloned belief is independent of original."""
        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=37.7,
            max_lat=37.71,
            min_lon=-122.5,
            max_lon=-122.49,
            zoom_level=20,
            cell_size_px=160.0,
        )
        original = HistogramBelief.from_uniform(grid_spec, torch.device("cpu"))
        cloned = original.clone()

        # Modify original
        original._log_belief[0, 0] = 0.0

        # Clone should be unchanged
        self.assertNotEqual(cloned._log_belief[0, 0].item(), 0.0)


class TestSegmentLogsumexp(unittest.TestCase):
    def test_single_segment(self):
        """Test logsumexp over a single segment."""
        values = torch.tensor([1.0, 2.0, 3.0])
        offsets = torch.tensor([0, 3])
        segment_ids = torch.tensor([0, 0, 0])

        result = segment_logsumexp(values, offsets, segment_ids)

        expected = torch.logsumexp(values, dim=0)
        self.assertTrue(torch.allclose(result, expected.unsqueeze(0), atol=1e-5))

    def test_multiple_segments(self):
        """Test logsumexp over multiple segments."""
        # Segment 0: [1, 2], Segment 1: [3, 4, 5]
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        offsets = torch.tensor([0, 2, 5])
        segment_ids = torch.tensor([0, 0, 1, 1, 1])

        result = segment_logsumexp(values, offsets, segment_ids)

        expected_0 = torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0)
        expected_1 = torch.logsumexp(torch.tensor([3.0, 4.0, 5.0]), dim=0)

        self.assertAlmostEqual(result[0].item(), expected_0.item(), places=5)
        self.assertAlmostEqual(result[1].item(), expected_1.item(), places=5)

    def test_empty_segment(self):
        """Test handling of empty segments."""
        # Segment 0: [1], Segment 1: [], Segment 2: [2]
        values = torch.tensor([1.0, 2.0])
        offsets = torch.tensor([0, 1, 1, 2])  # Segment 1 is empty
        segment_ids = torch.tensor([0, 2])

        result = segment_logsumexp(values, offsets, segment_ids)

        self.assertAlmostEqual(result[0].item(), 1.0, places=5)
        self.assertEqual(result[1].item(), -float("inf"))  # Empty segment
        self.assertAlmostEqual(result[2].item(), 2.0, places=5)


class TestBuildCellToPatchMapping(unittest.TestCase):
    def test_simple_grid(self):
        """Test mapping construction with a simple grid."""
        grid_spec = GridSpec(
            zoom_level=20,
            cell_size_px=100.0,
            origin_row_px=0.0,
            origin_col_px=0.0,
            num_rows=3,
            num_cols=3,
        )

        # Single patch centered at (150, 150) with half-size 200
        # This should cover all 9 cells
        patch_positions = torch.tensor([[150.0, 150.0]])
        patch_half_size = 200.0

        mapping = build_cell_to_patch_mapping(
            grid_spec, patch_positions, patch_half_size, torch.device("cpu")
        )

        # All 9 cells should overlap with the single patch
        self.assertEqual(len(mapping.patch_indices), 9)
        self.assertTrue(torch.all(mapping.patch_indices == 0))

    def test_non_overlapping_patches(self):
        """Test with patches that don't overlap some cells."""
        grid_spec = GridSpec(
            zoom_level=20,
            cell_size_px=100.0,
            origin_row_px=0.0,
            origin_col_px=0.0,
            num_rows=2,
            num_cols=2,
        )

        # Patch only covers top-left cell
        patch_positions = torch.tensor([[50.0, 50.0]])
        patch_half_size = 60.0  # Only covers cells whose center is within 60px

        mapping = build_cell_to_patch_mapping(
            grid_spec, patch_positions, patch_half_size, torch.device("cpu")
        )

        # Only cell (0,0) with center at (50, 50) should overlap
        self.assertEqual(len(mapping.patch_indices), 1)

    def test_multiple_overlapping_patches(self):
        """Test cells overlapping with multiple patches."""
        grid_spec = GridSpec(
            zoom_level=20,
            cell_size_px=100.0,
            origin_row_px=0.0,
            origin_col_px=0.0,
            num_rows=1,
            num_cols=1,
        )

        # Two overlapping patches
        patch_positions = torch.tensor([[50.0, 50.0], [60.0, 60.0]])
        patch_half_size = 100.0

        mapping = build_cell_to_patch_mapping(
            grid_spec, patch_positions, patch_half_size, torch.device("cpu")
        )

        # The single cell should overlap with both patches
        self.assertEqual(len(mapping.patch_indices), 2)


class TestApplyObservation(unittest.TestCase):
    def test_observation_concentrates_belief(self):
        """Test that observation update concentrates belief toward high-likelihood areas."""
        grid_spec = GridSpec(
            zoom_level=20,
            cell_size_px=100.0,
            origin_row_px=0.0,
            origin_col_px=0.0,
            num_rows=3,
            num_cols=3,
        )

        # Create a patch for each cell
        # Patch positions at cell centers
        patch_positions = torch.tensor([
            [50.0, 50.0], [50.0, 150.0], [50.0, 250.0],
            [150.0, 50.0], [150.0, 150.0], [150.0, 250.0],
            [250.0, 50.0], [250.0, 150.0], [250.0, 250.0],
        ])
        patch_half_size = 60.0  # Each cell overlaps only its corresponding patch

        mapping = build_cell_to_patch_mapping(
            grid_spec, patch_positions, patch_half_size, torch.device("cpu")
        )

        # Start with uniform belief
        belief = HistogramBelief.from_uniform(grid_spec, torch.device("cpu"))

        # Create similarity matrix with high similarity at center patch (index 4)
        similarity = torch.zeros(9)
        similarity[4] = 0.9  # High similarity at center
        similarity[0] = 0.1  # Low elsewhere

        belief.apply_observation(similarity, mapping, sigma=0.1)

        # Center cell should have highest probability
        probs = belief.get_belief()
        center_prob = probs[1, 1].item()
        corner_prob = probs[0, 0].item()

        self.assertGreater(center_prob, corner_prob)

    def test_observation_preserves_normalization(self):
        """Test that observation update keeps belief normalized."""
        grid_spec = GridSpec(
            zoom_level=20,
            cell_size_px=100.0,
            origin_row_px=0.0,
            origin_col_px=0.0,
            num_rows=2,
            num_cols=2,
        )

        patch_positions = torch.tensor([
            [50.0, 50.0], [50.0, 150.0],
            [150.0, 50.0], [150.0, 150.0],
        ])
        patch_half_size = 60.0

        mapping = build_cell_to_patch_mapping(
            grid_spec, patch_positions, patch_half_size, torch.device("cpu")
        )

        belief = HistogramBelief.from_uniform(grid_spec, torch.device("cpu"))
        similarity = torch.rand(4)

        belief.apply_observation(similarity, mapping, sigma=0.1)

        # Should still sum to 1
        prob_sum = belief.get_belief().sum().item()
        self.assertAlmostEqual(prob_sum, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
