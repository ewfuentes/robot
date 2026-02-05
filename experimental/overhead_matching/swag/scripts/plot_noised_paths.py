"""Visualize the effect of odometry noise on evaluation paths.

Loads paths and VIGOR dataset metadata (no images), applies noise at
multiple levels, and produces comparison plots.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:plot_noised_paths -- \
        --dataset-path /data/overhead_matching/datasets/VIGOR/Seattle \
        --paths-path /data/overhead_matching/evaluation/paths/Seattle_1000_goal_directed.json \
        --output /tmp/noised_paths.png
"""

import argparse
import json
import math
from pathlib import Path

import common.torch.load_torch_deps
import torch
import matplotlib.pyplot as plt
import numpy as np

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.odometry_noise import (
    OdometryNoiseConfig,
    add_noise_to_motion_deltas,
    compute_positions_from_deltas,
    METERS_PER_DEG_LAT,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es


def positions_to_meters(positions: torch.Tensor, ref_lat_rad: float) -> torch.Tensor:
    """Convert (N, 2) [lat, lon] degrees to [north_m, east_m] relative to first position."""
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    origin = positions[0:1]
    diff = positions - origin
    north_m = diff[:, 0] * METERS_PER_DEG_LAT
    east_m = diff[:, 1] * meters_per_deg_lon
    return torch.stack([north_m, east_m], dim=1)


def compute_position_errors(
    true_positions: torch.Tensor,
    noised_positions: torch.Tensor,
    ref_lat_rad: float,
) -> torch.Tensor:
    """Compute per-position error in meters."""
    diff = noised_positions - true_positions
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    north_err = diff[:, 0] * METERS_PER_DEG_LAT
    east_err = diff[:, 1] * meters_per_deg_lon
    return torch.sqrt(north_err ** 2 + east_err ** 2)


def plot_paths(
    ax,
    true_pos_m: torch.Tensor,
    motion_deltas: torch.Tensor,
    start_latlon: torch.Tensor,
    ref_lat_rad: float,
    noise_levels: list[float],
    seed: int,
):
    """Plot ground truth vs noised paths at various noise levels."""
    ax.plot(true_pos_m[:, 1].numpy(), true_pos_m[:, 0].numpy(),
            'k-', linewidth=2, label='Ground truth', zorder=10)

    colors = plt.get_cmap('Reds')(np.linspace(0.3, 0.9, len(noise_levels)))
    for val, color in zip(noise_levels, colors):
        if val == 0:
            continue
        config = OdometryNoiseConfig(sigma_noise_frac=val)
        label = f'σ={val}'

        gen = torch.Generator().manual_seed(seed)
        noised = add_noise_to_motion_deltas(motion_deltas, start_latlon, config, generator=gen)
        noised_pos = compute_positions_from_deltas(start_latlon.to(torch.float64), noised)
        noised_m = positions_to_meters(noised_pos, ref_lat_rad)
        ax.plot(noised_m[:, 1].numpy(), noised_m[:, 0].numpy(),
                '-', color=color, alpha=0.8, label=label)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_title('Noised paths')
    ax.legend(fontsize=7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_error_vs_distance(
    ax,
    motion_deltas: torch.Tensor,
    start_latlon: torch.Tensor,
    ref_lat_rad: float,
    noise_levels: list[float],
    num_realizations: int = 50,
    seed: int = 42,
):
    """Plot position error vs distance traveled for swept noise levels."""
    true_pos = compute_positions_from_deltas(start_latlon.to(torch.float64), motion_deltas)

    # Compute cumulative distance traveled in meters
    deltas_m_north = torch.diff(true_pos[:, 0]) * METERS_PER_DEG_LAT
    meters_per_deg_lon = METERS_PER_DEG_LAT * math.cos(ref_lat_rad)
    deltas_m_east = torch.diff(true_pos[:, 1]) * meters_per_deg_lon
    step_dist = torch.sqrt(deltas_m_north ** 2 + deltas_m_east ** 2)
    cum_dist = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(step_dist, dim=0)])

    cmap = plt.get_cmap('tab10')
    configs = [(val, OdometryNoiseConfig(sigma_noise_frac=val)) for val in noise_levels if val > 0]
    for idx, (val, config) in enumerate(configs):
        all_errors = []
        for r in range(num_realizations):
            gen = torch.Generator().manual_seed(seed + r)
            noised = add_noise_to_motion_deltas(motion_deltas, start_latlon, config, generator=gen)
            noised_pos = compute_positions_from_deltas(start_latlon.to(torch.float64), noised)
            errors = compute_position_errors(true_pos, noised_pos, ref_lat_rad)
            all_errors.append(errors.numpy())

        all_errors = np.stack(all_errors)
        mean_err = all_errors.mean(axis=0)
        std_err = all_errors.std(axis=0)

        dist_np = cum_dist.numpy()
        color = cmap(idx / max(len(configs) - 1, 1))
        ax.plot(dist_np, mean_err, '-', color=color, label=f'σ={val}')
        ax.fill_between(dist_np, mean_err - std_err, mean_err + std_err,
                        alpha=0.15, color=color)

    ax.set_xlabel('Distance traveled (m)')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Error vs distance (mean ± std)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Visualize noised evaluation paths")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to VIGOR dataset")
    parser.add_argument("--paths-path", type=str, required=True, help="Path to JSON evaluation paths")
    parser.add_argument("--output", type=str, default=None,
                        help="Save to file (PNG/PDF). If not set, displays interactively.")
    parser.add_argument("--path-index", type=int, default=0, help="Which path to visualize")
    parser.add_argument("--noise-levels", type=str, default="0,0.02,0.05,0.1",
                        help="Comma-separated noise levels (fraction of step distance)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    # Load dataset (metadata only)
    dataset_path = Path(args.dataset_path).expanduser()
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        panorama_neighbor_radius=0.0005,
        should_load_images=False,
        should_load_landmarks=False,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    # Load paths
    with open(args.paths_path, 'r') as f:
        paths_data = json.load(f)
    paths = paths_data['paths']

    if args.path_index >= len(paths):
        raise ValueError(f"Path index {args.path_index} out of range (have {len(paths)} paths)")

    path = paths[args.path_index]
    print(f"Using path {args.path_index} with {len(path)} steps")

    # Get motion deltas and positions
    motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(dtype=torch.float64)
    start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(dtype=torch.float64)
    true_pos = compute_positions_from_deltas(start_latlon, motion_deltas)
    ref_lat_rad = math.radians(start_latlon[0].item())
    true_pos_m = positions_to_meters(true_pos, ref_lat_rad)

    # Create figure: noised paths, error vs distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    plot_paths(
        axes[0], true_pos_m, motion_deltas, start_latlon, ref_lat_rad,
        noise_levels, seed=args.seed,
    )

    plot_error_vs_distance(
        axes[1], motion_deltas, start_latlon, ref_lat_rad,
        noise_levels, num_realizations=50, seed=args.seed,
    )

    fig.suptitle(f'Odometry Noise — Path {args.path_index} ({len(path)} steps)',
                 fontsize=14)
    fig.tight_layout()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
