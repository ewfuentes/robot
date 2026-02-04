"""Profile GPU memory consumption of histogram filter evaluation.

Instruments memory measurement at key operations to identify which tensors
and operations dominate GPU memory when increasing the subdivision factor.
"""

import json
import math
from pathlib import Path

import common.torch.load_torch_deps
import torch
import torch.nn.functional as F

import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    compute_probability_mass_within_radius,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    CellToPatchMapping,
    build_cell_to_patch_mapping,
    segment_max,
    _shift_grid,
    _make_gaussian_kernel_1d,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    aggregator_from_config,
    load_aggregator_config,
)
from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
    HistogramFilterConfig,
    get_dataset_bounds,
    get_patch_positions_px,
    construct_path_eval_inputs_from_args,
)
from common.gps import web_mercator


def gpu_mem_mb():
    """Return (allocated_MB, max_allocated_MB)."""
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return alloc, peak


def tensor_mb(t: torch.Tensor) -> float:
    """Return size of a tensor in MB."""
    return t.element_size() * t.nelement() / (1024 * 1024)


def describe(name: str, t: torch.Tensor):
    """Print shape, dtype, and MB for a tensor."""
    print(f"      {name:45s} {str(t.shape):25s} {str(t.dtype):15s} {tensor_mb(t):8.2f} MB")


def checkpoint(label, prev_alloc=0.0):
    """Record a memory checkpoint and print it."""
    alloc, peak = gpu_mem_mb()
    delta = alloc - prev_alloc
    print(f"    >> {label:50s}  alloc={alloc:8.2f} MB  peak={peak:8.2f} MB  delta={delta:+8.2f} MB")
    return alloc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile GPU memory for histogram filter")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--paths-path", type=str, required=True)
    parser.add_argument("--aggregator-config", type=str, required=True)
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--subdivision-factor", type=int, required=True)
    parser.add_argument("--noise-percent", type=float, default=0.02)
    parser.add_argument("--convergence-radii", type=str, default="25,50,100")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005)
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640)
    args = parser.parse_args()

    convergence_radii = [int(r.strip()) for r in args.convergence_radii.split(",")]

    DEVICE = "cuda:0"

    print(f"\n{'='*80}")
    print(f"  GPU Memory Profile  --  subdivision_factor = {args.subdivision_factor}")
    print(f"{'='*80}\n")

    torch.cuda.reset_peak_memory_stats()
    prev = checkpoint("Baseline (empty)")

    # ---- 1. Load dataset (no models) ----
    vigor_dataset, _, _, paths_data = construct_path_eval_inputs_from_args(
        dataset_path=args.dataset_path,
        paths_path=args.paths_path,
        panorama_neighbor_radius_deg=args.panorama_neighbor_radius_deg,
        panorama_landmark_radius_px=args.panorama_landmark_radius_px,
        device=DEVICE,
        landmark_version=args.landmark_version,
    )
    prev = checkpoint("After dataset load", prev)

    # ---- 2. Create aggregator (loads similarity matrix) ----
    agg_config = load_aggregator_config(Path(args.aggregator_config))
    log_likelihood_aggregator = aggregator_from_config(agg_config, vigor_dataset, DEVICE)
    prev = checkpoint("After aggregator creation", prev)
    print(f"      similarity_matrix lives on: {log_likelihood_aggregator.similarity_matrix.device}")
    describe("similarity_matrix", log_likelihood_aggregator.similarity_matrix)

    # ---- 3. Build grid spec ----
    config = HistogramFilterConfig(
        noise_percent=args.noise_percent,
        subdivision_factor=args.subdivision_factor,
    )
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    cell_size_px = config.patch_size_px / config.subdivision_factor

    patch_half_size_px = config.patch_size_px / 2.0
    center_lat = (min_lat + max_lat) / 2
    ref_y, ref_x = web_mercator.latlon_to_pixel_coords(center_lat, min_lon, config.zoom_level)
    buf_lat, _ = web_mercator.pixel_coords_to_latlon(ref_y - patch_half_size_px, ref_x, config.zoom_level)
    _, buf_lon = web_mercator.pixel_coords_to_latlon(ref_y, ref_x + patch_half_size_px, config.zoom_level)
    lat_buffer = buf_lat - center_lat
    lon_buffer = buf_lon - min_lon

    grid_spec = GridSpec.from_bounds_and_cell_size(
        min_lat=min_lat - lat_buffer,
        max_lat=max_lat + lat_buffer,
        min_lon=min_lon - lon_buffer,
        max_lon=max_lon + lon_buffer,
        zoom_level=config.zoom_level,
        cell_size_px=cell_size_px,
    )
    H, W = grid_spec.num_rows, grid_spec.num_cols
    num_cells = H * W
    num_patches = len(vigor_dataset._satellite_metadata)
    print(f"\n  Grid: {H} x {W} = {num_cells} cells  (cell_size={cell_size_px:.1f} px)")
    print(f"  num_patches: {num_patches}")

    # ---- 4. build_cell_to_patch_mapping ----
    patch_positions_px = get_patch_positions_px(vigor_dataset, DEVICE)
    torch.cuda.reset_peak_memory_stats()
    mapping = build_cell_to_patch_mapping(
        grid_spec=grid_spec,
        patch_positions_px=patch_positions_px,
        patch_half_size_px=patch_half_size_px,
        device=DEVICE,
    )
    prev = checkpoint("After build_cell_to_patch_mapping()", prev)
    total_overlaps = len(mapping.patch_indices)

    print(f"\n  === Persistent mapping tensors (CSR format) ===")
    describe("mapping.patch_indices", mapping.patch_indices)
    describe("mapping.cell_offsets", mapping.cell_offsets)
    describe("mapping.segment_ids", mapping.segment_ids)
    mapping_mb = tensor_mb(mapping.patch_indices) + tensor_mb(mapping.cell_offsets) + tensor_mb(mapping.segment_ids)
    print(f"      Total mapping:  {mapping_mb:.2f} MB   (total_overlaps={total_overlaps})")
    avg_overlaps = total_overlaps / num_cells if num_cells > 0 else 0
    print(f"      Avg overlaps/cell: {avg_overlaps:.1f}")

    # ---- 5. Filter loop (1 path), with detailed tensor tracing ----
    paths = paths_data['paths']
    if not paths:
        print("No paths found!")
        return

    path = paths[0]
    print(f"\n{'='*80}")
    print(f"  DETAILED FILTER STEP TRACE  --  1 step on path[0] ({len(path)} panos)")
    print(f"{'='*80}")

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        belief = HistogramBelief.from_uniform(grid_spec=grid_spec, device=DEVICE)
        prev = checkpoint("After HistogramBelief.from_uniform()", prev)
        describe("belief._log_belief", belief._log_belief)

        motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(DEVICE)
        true_latlons = vigor_dataset.get_panorama_positions(path).to(DEVICE)

        # ---- apply_observation, step by step ----
        print(f"\n  --- apply_observation (step 0) ---")
        torch.cuda.reset_peak_memory_stats()

        # 1) aggregator call: look up one row of similarity matrix, compute log-likelihood
        obs_log_ll = log_likelihood_aggregator(path[0])
        prev = checkpoint("aggregator -> obs_log_ll", prev)
        describe("obs_log_ll", obs_log_ll)

        # 2) gather: index into obs_log_ll with patch_indices
        all_log_ll = obs_log_ll[mapping.patch_indices]  # (total_overlaps,)
        prev = checkpoint("gather: obs_log_ll[patch_indices]", prev)
        describe("all_log_ll (gathered)", all_log_ll)

        # 3) segment_max: reduce to per-cell max
        cell_log_ll = segment_max(all_log_ll, mapping.cell_offsets, mapping.segment_ids)
        prev = checkpoint("segment_max -> cell_log_ll", prev)
        describe("cell_log_ll", cell_log_ll)

        # 4) reshape + add to belief + normalize
        cell_log_ll_grid = cell_log_ll.reshape(H, W)
        belief._log_belief = belief._log_belief + cell_log_ll_grid
        belief.normalize()
        prev = checkpoint("add to belief + normalize", prev)
        # clean up the temporaries as the real code would
        del obs_log_ll, all_log_ll, cell_log_ll, cell_log_ll_grid

        # ---- apply_motion, step by step ----
        print(f"\n  --- apply_motion (step 0) ---")
        torch.cuda.reset_peak_memory_stats()

        # 1) _shift_grid internals
        log_belief = belief._log_belief
        belief_prob = torch.exp(log_belief)
        prev = checkpoint("exp(log_belief) -> belief_prob", prev)
        describe("belief_prob", belief_prob)

        y = torch.linspace(-1, 1, H, device=DEVICE, dtype=torch.float32)
        x = torch.linspace(-1, 1, W, device=DEVICE, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        prev = checkpoint("sampling grid for grid_sample", prev)
        describe("grid (sampling coords)", grid)

        belief_prob_4d = belief_prob.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        shifted = F.grid_sample(belief_prob_4d, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        shifted_log = torch.log(shifted.squeeze() + 1e-40)
        prev = checkpoint("grid_sample + log -> shifted_log", prev)
        describe("shifted (output)", shifted)

        del belief_prob, grid, grid_y, grid_x, y, x, belief_prob_4d, shifted

        # 2) apply_motion_blur internals
        # Compute sigma for this step
        reference_latlon = belief.get_mean_latlon()
        y0, x0 = web_mercator.latlon_to_pixel_coords(
            reference_latlon[0], reference_latlon[1], grid_spec.zoom_level)
        y1, x1 = web_mercator.latlon_to_pixel_coords(
            reference_latlon[0] + motion_deltas[0][0],
            reference_latlon[1] + motion_deltas[0][1],
            grid_spec.zoom_level)
        delta_magnitude_px = torch.sqrt((y1-y0)**2 + (x1-x0)**2)
        sigma_cells = (delta_magnitude_px * config.noise_percent / cell_size_px).item()
        kernel = _make_gaussian_kernel_1d(sigma_cells, DEVICE)
        print(f"      sigma_cells={sigma_cells:.4f}, kernel_size={len(kernel)}")
        describe("blur kernel", kernel)

        blur_prob = torch.exp(shifted_log)
        pad_size = len(kernel) // 2
        blur_prob = F.pad(blur_prob.unsqueeze(0).unsqueeze(0),
                          (pad_size, pad_size, pad_size, pad_size),
                          mode="constant", value=0)
        prev = checkpoint("exp + pad for blur", prev)
        describe("padded belief_prob", blur_prob)

        kernel_v = kernel.view(1, 1, -1, 1)
        blur_prob = F.conv2d(blur_prob, kernel_v)
        kernel_h = kernel.view(1, 1, 1, -1)
        blur_prob = F.conv2d(blur_prob, kernel_h)
        prev = checkpoint("separable conv2d blur", prev)
        describe("blurred belief_prob", blur_prob)

        belief._log_belief = torch.log(blur_prob.squeeze() + 1e-40)
        belief.normalize()
        del blur_prob, shifted_log, kernel
        prev = checkpoint("log + normalize", prev)

        # Now actually do the step via the real API to advance state properly
        belief = HistogramBelief.from_uniform(grid_spec=grid_spec, device=DEVICE)
        belief.apply_observation(
            log_likelihood_aggregator(path[0]), mapping)
        belief.apply_motion(motion_deltas[0], config.noise_percent)
        torch.cuda.empty_cache()

        # ---- get_mean_latlon, step by step ----
        print(f"\n  --- get_mean_latlon ---")
        torch.cuda.reset_peak_memory_stats()
        prev = checkpoint("before get_mean_latlon", prev)

        belief_probs = belief.get_belief()   # (H, W) = exp(_log_belief)
        describe("belief_probs = exp(log_belief)", belief_probs)
        row_marginal = belief_probs.sum(dim=1)  # (H,)
        col_marginal = belief_probs.sum(dim=0)  # (W,)
        describe("row_marginal", row_marginal)
        describe("col_marginal", col_marginal)
        prev = checkpoint("marginals computed", prev)
        del belief_probs, row_marginal, col_marginal

        # ---- get_variance_deg_sq, step by step ----
        print(f"\n  --- get_variance_deg_sq ---")
        torch.cuda.reset_peak_memory_stats()
        prev = checkpoint("before get_variance_deg_sq", prev)

        belief_probs = belief.get_belief()
        describe("belief_probs", belief_probs)

        cell_centers = grid_spec.get_all_cell_centers_latlon(DEVICE)
        describe("cell_centers (all latlon)", cell_centers)
        cell_centers_3d = cell_centers.reshape(H, W, 2)
        describe("cell_centers reshaped", cell_centers_3d)

        mean_latlon = belief.get_mean_latlon()
        diff_sq = (cell_centers_3d - mean_latlon) ** 2  # (H, W, 2)
        describe("diff_sq", diff_sq)

        weighted = belief_probs.unsqueeze(-1) * diff_sq  # (H, W, 2)
        describe("weighted = probs * diff_sq", weighted)

        var = weighted.sum(dim=(0, 1))
        prev = checkpoint("variance computed", prev)
        del belief_probs, cell_centers, cell_centers_3d, diff_sq, weighted, var

        # ---- compute_probability_mass_within_radius, step by step ----
        print(f"\n  --- compute_probability_mass_within_radius ---")
        torch.cuda.reset_peak_memory_stats()
        prev = checkpoint("before convergence metric", prev)

        cell_centers_px = grid_spec.get_all_cell_centers_px(DEVICE)
        describe("cell_centers_px", cell_centers_px)

        true_lat = true_latlons[1][0].item()
        true_lon = true_latlons[1][1].item()
        true_y_px, true_x_px = web_mercator.latlon_to_pixel_coords(
            true_lat, true_lon, grid_spec.zoom_level)
        true_pos_px = torch.tensor([[true_y_px, true_x_px]], device=DEVICE, dtype=torch.float32)
        delta = cell_centers_px - true_pos_px
        describe("delta (cell - true)", delta)
        dist_px = torch.norm(delta, dim=1)
        describe("dist_px", dist_px)

        meters_per_pixel = web_mercator.get_meters_per_pixel(true_lat, grid_spec.zoom_level)
        dist_meters = dist_px * meters_per_pixel
        within_mask = dist_meters <= 100.0
        prob_mass = belief.get_belief().flatten()[within_mask].sum()
        prev = checkpoint("convergence metric computed", prev)
        del cell_centers_px, delta, dist_px, dist_meters, within_mask

    print(f"\n{'='*80}")
    print(f"  DONE  --  subdivision_factor = {args.subdivision_factor}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
