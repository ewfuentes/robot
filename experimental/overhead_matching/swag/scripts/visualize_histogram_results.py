"""Visualize histogram filter results on paths."""

import common.torch.load_torch_deps
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import argparse
import numpy as np

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es


def load_evaluation_results(eval_path: Path, path_idx: int):
    """Load evaluation results for a specific path."""
    path_dir = eval_path / f"{path_idx:07d}"

    def load_cpu(path):
        t = torch.load(path, map_location='cpu')
        return t.cpu() if hasattr(t, 'cpu') else t

    results = {
        "error": load_cpu(path_dir / "error.pt"),
        "var": load_cpu(path_dir / "var.pt"),
        "path": torch.load(path_dir / "path.pt", map_location='cpu'),  # May be list
        "distance_traveled_m": load_cpu(path_dir / "distance_traveled_m.pt"),
    }

    if (path_dir / "mean_history.pt").exists():
        results["mean_history"] = load_cpu(path_dir / "mean_history.pt")
    if (path_dir / "variance_history.pt").exists():
        results["variance_history"] = load_cpu(path_dir / "variance_history.pt")
    if (path_dir / "other_info.json").exists():
        with open(path_dir / "other_info.json") as f:
            results["other_info"] = json.load(f)

    return results


def plot_path_comparison(
    vigor_dataset: vd.VigorDataset,
    results: dict,
    satellite_positions: torch.Tensor,
    title: str = "Histogram Filter Results",
    ax: plt.Axes = None,
):
    """Plot ground truth path vs estimated path with satellite patches."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    path = results["path"]
    gt_positions = vigor_dataset.get_panorama_positions(path).numpy()

    # Plot satellite patch centers
    sat_latlons = vigor_dataset._satellite_metadata[['lat', 'lon']].values
    ax.scatter(sat_latlons[:, 1], sat_latlons[:, 0],
               c='lightgray', s=10, alpha=0.3, label='Satellite patches')

    # Plot ground truth path
    ax.plot(gt_positions[:, 1], gt_positions[:, 0],
            'g-', linewidth=2, label='Ground truth', zorder=5)
    ax.scatter(gt_positions[:, 1], gt_positions[:, 0],
               c='green', s=30, zorder=6)
    ax.scatter(gt_positions[0, 1], gt_positions[0, 0],
               c='green', s=100, marker='*', zorder=7, label='Start (GT)')

    # Plot estimated path if available
    if "mean_history" in results:
        mean_history = results["mean_history"].numpy()
        # mean_history has one extra entry (initial before first obs)
        # Align with path length
        estimates = mean_history[-len(path):]

        ax.plot(estimates[:, 1], estimates[:, 0],
                'r--', linewidth=2, label='Estimated', zorder=5)
        ax.scatter(estimates[:, 1], estimates[:, 0],
                   c='red', s=30, zorder=6)
        ax.scatter(estimates[0, 1], estimates[0, 0],
                   c='red', s=100, marker='*', zorder=7, label='Start (Est)')

        # Draw error lines between GT and estimates
        for i in range(len(path)):
            ax.plot([gt_positions[i, 1], estimates[i, 1]],
                    [gt_positions[i, 0], estimates[i, 0]],
                    'k-', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')

    return ax


def plot_error_over_time(results: dict, ax: plt.Axes = None):
    """Plot error over distance traveled."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    error = results["error"].numpy()
    distance = results["distance_traveled_m"].numpy()

    # Align lengths
    min_len = min(len(error), len(distance))
    error = error[:min_len]
    distance = distance[:min_len]

    ax.plot(distance, error, 'b-', linewidth=2)
    ax.scatter(distance, error, c='blue', s=30)

    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'Localization Error Over Path (Final: {error[-1]:.1f}m)')
    ax.grid(True, alpha=0.3)

    return ax


def plot_multiple_paths_summary(eval_path: Path, vigor_dataset: vd.VigorDataset, num_paths: int = 10):
    """Plot summary of multiple paths."""
    # Load summary
    with open(eval_path / "summary_statistics.json") as f:
        summary = json.load(f)

    # Collect errors for all paths
    all_final_errors = []
    all_error_curves = []
    all_distance_curves = []

    for i in range(num_paths):
        try:
            results = load_evaluation_results(eval_path, i)
            all_final_errors.append(results["error"][-1].item())
            all_error_curves.append(results["error"].numpy())
            all_distance_curves.append(results["distance_traveled_m"].numpy())
        except FileNotFoundError:
            break

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Final error histogram
    ax = axes[0, 0]
    ax.hist(all_final_errors, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(summary["average_final_error"], color='r', linestyle='--',
               label=f'Mean: {summary["average_final_error"]:.1f}m')
    ax.set_xlabel('Final Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Final Errors')
    ax.legend()

    # Plot 2: Error curves over distance
    ax = axes[0, 1]
    for i, (error, distance) in enumerate(zip(all_error_curves, all_distance_curves)):
        min_len = min(len(error), len(distance))
        ax.plot(distance[:min_len], error[:min_len], alpha=0.5, linewidth=1)
    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Error (m)')
    ax.set_title('Error Curves for All Paths')
    ax.grid(True, alpha=0.3)

    # Plot 3: Best path
    best_idx = np.argmin(all_final_errors)
    best_results = load_evaluation_results(eval_path, best_idx)
    plot_path_comparison(vigor_dataset, best_results, None,
                        title=f'Best Path (#{best_idx}, Error: {all_final_errors[best_idx]:.1f}m)',
                        ax=axes[1, 0])

    # Plot 4: Worst path
    worst_idx = np.argmax(all_final_errors)
    worst_results = load_evaluation_results(eval_path, worst_idx)
    plot_path_comparison(vigor_dataset, worst_results, None,
                        title=f'Worst Path (#{worst_idx}, Error: {all_final_errors[worst_idx]:.1f}m)',
                        ax=axes[1, 1])

    plt.tight_layout()
    return fig


def debug_path(vigor_dataset, results, path_idx):
    """Print debug info about a path."""
    path = results["path"]
    mean_history = results.get("mean_history")

    gt_positions = vigor_dataset.get_panorama_positions(path)

    print(f"\n=== Debug info for path {path_idx} ===")
    print(f"Path length: {len(path)}")
    print(f"GT start: {gt_positions[0].numpy()}")
    print(f"GT end: {gt_positions[-1].numpy()}")
    print(f"GT total displacement: {(gt_positions[-1] - gt_positions[0]).numpy()}")

    if mean_history is not None:
        print(f"\nMean history shape: {mean_history.shape}")
        print(f"Est start: {mean_history[0].numpy()}")
        print(f"Est end: {mean_history[-1].numpy()}")
        print(f"Est total displacement: {(mean_history[-1] - mean_history[0]).numpy()}")

        # Check step-by-step displacements
        displacements = torch.diff(mean_history, dim=0)
        print(f"\nMean step displacement: {torch.norm(displacements, dim=1).mean().item():.8f} deg")
        print(f"Max step displacement: {torch.norm(displacements, dim=1).max().item():.8f} deg")

        print(f"\nFirst 5 estimate displacements (deg):")
        for i in range(min(5, len(displacements))):
            print(f"  Step {i}: {displacements[i].numpy()}")

    # Compare to GT motion
    gt_motions = torch.diff(gt_positions, dim=0)
    print(f"\nFirst 5 GT motions (deg):")
    for i in range(min(5, len(gt_motions))):
        print(f"  Step {i}: {gt_motions[i].numpy()}")

    print(f"\nMean GT motion magnitude: {torch.norm(gt_motions, dim=1).mean().item():.8f} deg")


def main():
    parser = argparse.ArgumentParser(description="Visualize histogram filter results")
    parser.add_argument("--eval-path", type=str, required=True,
                        help="Path to evaluation results directory")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VIGOR dataset")
    parser.add_argument("--path-idx", type=int, default=None,
                        help="Specific path index to visualize (default: show summary)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: display)")
    parser.add_argument("--num-paths", type=int, default=100,
                        help="Number of paths to include in summary")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info")

    args = parser.parse_args()

    eval_path = Path(args.eval_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()

    # Load dataset config from evaluation
    with open(eval_path / "args.json") as f:
        eval_args = json.load(f)

    # Minimal dataset config for loading positions
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=None,
        satellite_tensor_cache_info=None,
        panorama_neighbor_radius=0.0005,
        satellite_patch_size=(640, 640),
        panorama_size=(640, 640),
        factor=0.3,
        landmark_version=eval_args.get("landmark_version", "v4_202001"),
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    if args.path_idx is not None:
        # Visualize specific path
        results = load_evaluation_results(eval_path, args.path_idx)

        if args.debug:
            debug_path(vigor_dataset, results, args.path_idx)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_path_comparison(vigor_dataset, results, None,
                            title=f'Path #{args.path_idx}', ax=axes[0])
        plot_error_over_time(results, ax=axes[1])
        plt.tight_layout()
    else:
        # Show summary
        fig = plot_multiple_paths_summary(eval_path, vigor_dataset, args.num_paths)

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
