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

    path = torch.load(path_dir / "path.pt", map_location='cpu')
    # Check for old format
    if path and isinstance(path[0], int):
        raise ValueError(
            f"path.pt in '{path_dir}' uses old index format (integers). "
            "Re-run evaluation with new path files to get pano_id format (strings)."
        )
    results = {
        "error": load_cpu(path_dir / "error.pt"),
        "var": load_cpu(path_dir / "var.pt"),
        "path": path,
        "distance_traveled_m": load_cpu(path_dir / "distance_traveled_m.pt"),
    }

    if (path_dir / "mean_history.pt").exists():
        results["mean_history"] = load_cpu(path_dir / "mean_history.pt")
    if (path_dir / "variance_history.pt").exists():
        results["variance_history"] = load_cpu(path_dir / "variance_history.pt")
    if (path_dir / "prob_mass_by_radius.pt").exists():
        results["prob_mass_by_radius"] = torch.load(
            path_dir / "prob_mass_by_radius.pt", map_location='cpu'
        )
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


def plot_convergence_curves_single_path(results: dict, ax: plt.Axes = None):
    """Plot probability mass within radius over distance for a single path."""
    if "prob_mass_by_radius" not in results:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    prob_mass_by_radius = results["prob_mass_by_radius"]
    distance = results["distance_traveled_m"].numpy()

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(prob_mass_by_radius)))

    for (radius, prob_mass), color in zip(sorted(prob_mass_by_radius.items()), colors):
        prob_mass_np = prob_mass.numpy()
        # Align lengths (prob_mass has path_len + 1 entries, distance has path_len)
        min_len = min(len(prob_mass_np), len(distance))
        ax.plot(distance[:min_len], prob_mass_np[:min_len],
                linewidth=2, color=color, label=f'{radius}m radius')

    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Probability Mass Within Radius')
    ax.set_title('Convergence: Probability Mass Near True Position')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_convergence_curves_aggregate(
    eval_path: Path,
    num_paths: int,
    ax: plt.Axes = None,
):
    """Plot aggregate convergence curves with median and IQR shading.

    Collects convergence data from all paths and plots median with
    interquartile range (25th-75th percentile) shading.
    """
    # Collect convergence data
    all_prob_mass_by_radius: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    for i in range(num_paths):
        # Some paths may have failed during evaluation or not have results saved
        try:
            results = load_evaluation_results(eval_path, i)
            if "prob_mass_by_radius" not in results:
                continue

            distance = results["distance_traveled_m"].numpy()
            for radius, prob_mass in results["prob_mass_by_radius"].items():
                if radius not in all_prob_mass_by_radius:
                    all_prob_mass_by_radius[radius] = []
                prob_mass_np = prob_mass.numpy()
                min_len = min(len(prob_mass_np), len(distance))
                all_prob_mass_by_radius[radius].append(
                    (distance[:min_len], prob_mass_np[:min_len])
                )
        except FileNotFoundError:
            break

    if not all_prob_mass_by_radius:
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create common distance bins for interpolation
    max_distance = max(
        d.max() for radius_data in all_prob_mass_by_radius.values()
        for d, _ in radius_data
    )
    distance_bins = np.linspace(0, max_distance, 100)

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_prob_mass_by_radius)))

    for (radius, curves), color in zip(sorted(all_prob_mass_by_radius.items()), colors):
        # Interpolate all curves to common distance bins
        interpolated = []
        for dist, prob_mass in curves:
            interp_values = np.interp(
                distance_bins, dist, prob_mass,
                left=prob_mass[0], right=prob_mass[-1]
            )
            interpolated.append(interp_values)

        interpolated = np.array(interpolated)  # (num_paths, num_bins)

        # Compute statistics
        median = np.median(interpolated, axis=0)
        q25 = np.percentile(interpolated, 25, axis=0)
        q75 = np.percentile(interpolated, 75, axis=0)

        ax.fill_between(distance_bins, q25, q75, alpha=0.3, color=color)
        ax.plot(distance_bins, median, linewidth=2, color=color, label=f'{radius}m radius')

    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Probability Mass Within Radius')
    ax.set_title('Convergence: Probability Mass Near True Position\n(Median + IQR)')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


def plot_convergence_curves_aggregate_xlim(
    eval_path: Path,
    num_paths: int,
    ax: plt.Axes,
    xlim: float = None,
    path_indices: list[int] = None,
):
    """Plot aggregate convergence curves with optional x-axis limit.

    Args:
        eval_path: Path to evaluation results
        num_paths: Maximum number of paths to consider
        ax: Matplotlib axes
        xlim: Optional x-axis limit
        path_indices: Optional list of specific path indices to include.
                      If None, includes all paths up to num_paths.
    """
    all_prob_mass_by_radius: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    indices_to_use = path_indices if path_indices is not None else range(num_paths)

    for i in indices_to_use:
        try:
            results = load_evaluation_results(eval_path, i)
            if "prob_mass_by_radius" not in results:
                continue
            distance = results["distance_traveled_m"].numpy()
            for radius, prob_mass in results["prob_mass_by_radius"].items():
                if radius not in all_prob_mass_by_radius:
                    all_prob_mass_by_radius[radius] = []
                prob_mass_np = prob_mass.numpy()
                min_len = min(len(prob_mass_np), len(distance))
                all_prob_mass_by_radius[radius].append(
                    (distance[:min_len], prob_mass_np[:min_len])
                )
        except FileNotFoundError:
            if path_indices is None:
                break  # Stop if sequential and file not found
            continue  # Skip if using specific indices

    if not all_prob_mass_by_radius:
        return

    max_distance = xlim if xlim else max(
        d.max() for radius_data in all_prob_mass_by_radius.values()
        for d, _ in radius_data
    )
    distance_bins = np.linspace(0, max_distance, 100)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(all_prob_mass_by_radius)))

    for (radius, curves), color in zip(sorted(all_prob_mass_by_radius.items()), colors):
        interpolated = []
        for dist, prob_mass in curves:
            interp_values = np.interp(
                distance_bins, dist, prob_mass,
                left=prob_mass[0], right=prob_mass[-1]
            )
            interpolated.append(interp_values)
        interpolated = np.array(interpolated)
        median = np.median(interpolated, axis=0)
        q25 = np.percentile(interpolated, 25, axis=0)
        q75 = np.percentile(interpolated, 75, axis=0)
        ax.fill_between(distance_bins, q25, q75, alpha=0.3, color=color)
        ax.plot(distance_bins, median, linewidth=2, color=color, label=f'{radius}m radius')

    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Probability Mass Within Radius')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)


def plot_error_curves_xlim(all_error_curves, all_distance_curves, ax, xlim=None):
    """Plot error curves with optional x-axis limit and median overlay."""
    for error, distance in zip(all_error_curves, all_distance_curves):
        min_len = min(len(error), len(distance))
        ax.plot(distance[:min_len], error[:min_len], alpha=0.3, linewidth=1)

    if all_distance_curves:
        max_dist = xlim if xlim else max(d.max() for d in all_distance_curves)
        dist_bins = np.linspace(0, max_dist, 100)
        interpolated_errors = []
        for error, distance in zip(all_error_curves, all_distance_curves):
            min_len = min(len(error), len(distance))
            interp_values = np.interp(dist_bins, distance[:min_len], error[:min_len])
            interpolated_errors.append(interp_values)
        interpolated_errors = np.array(interpolated_errors)
        median_error = np.median(interpolated_errors, axis=0)
        ax.plot(dist_bins, median_error, 'r-', linewidth=2, label='Median')
        ax.legend()

    ax.set_xlabel('Distance Traveled (m)')
    ax.set_ylabel('Error (m)')
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(0, xlim)


def plot_convergence_cost_hist(summary: dict, ax: plt.Axes):
    """Plot histogram of integrated convergence metric across all paths."""
    # Get convergence costs for each radius
    colors = plt.cm.viridis(np.linspace(0, 0.8, 3))

    for i, radius in enumerate([25, 50, 100]):
        key = f'convergence_cost_{radius}m'
        if key in summary:
            costs = summary[key]
            ax.hist(costs, bins=30, alpha=0.5, color=colors[i],
                    label=f'{radius}m (mean: {np.mean(costs):.0f})', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Integrated Convergence Cost')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_multiple_paths_summary(eval_path: Path, vigor_dataset: vd.VigorDataset, num_paths: int = 10):
    """Plot summary of multiple paths with full range and zoomed views."""
    # Load summary
    with open(eval_path / "summary_statistics.json") as f:
        summary = json.load(f)

    # Collect errors for all paths
    all_final_errors = []
    all_error_curves = []
    all_distance_curves = []
    has_convergence_data = False

    for i in range(num_paths):
        try:
            results = load_evaluation_results(eval_path, i)
            all_final_errors.append(results["error"][-1].item())
            all_error_curves.append(results["error"].numpy())
            all_distance_curves.append(results["distance_traveled_m"].numpy())
            if "prob_mass_by_radius" in results:
                has_convergence_data = True
        except FileNotFoundError:
            break

    print(f"Loaded {len(all_error_curves)} paths")

    # Separate converged vs non-converged paths
    CONVERGENCE_THRESHOLD = 100  # meters
    converged_indices = [i for i, e in enumerate(all_final_errors) if e < CONVERGENCE_THRESHOLD]
    non_converged_indices = [i for i, e in enumerate(all_final_errors) if e >= CONVERGENCE_THRESHOLD]
    converged_errors = [all_final_errors[i] for i in converged_indices]
    non_converged_errors = [all_final_errors[i] for i in non_converged_indices]

    print(f"Converged (<{CONVERGENCE_THRESHOLD}m): {len(converged_indices)} paths")
    print(f"Non-converged (>={CONVERGENCE_THRESHOLD}m): {len(non_converged_indices)} paths")

    # 3 rows x 3 cols layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Row 1: All paths
    # Plot 1: Convergence curves (full range)
    if has_convergence_data:
        plot_convergence_curves_aggregate_xlim(eval_path, num_paths, axes[0, 0])
        axes[0, 0].set_title('All Paths: Convergence\n(Median + IQR)')
    else:
        axes[0, 0].text(0.5, 0.5, 'No convergence data', ha='center', va='center')
        axes[0, 0].set_title('Convergence (No Data)')

    # Plot 2: Error curves (full range)
    plot_error_curves_xlim(all_error_curves, all_distance_curves, axes[0, 1])
    axes[0, 1].set_title('All Paths: Localization Error Over Distance')

    # Plot 3: Final error histogram (all paths)
    axes[0, 2].hist(all_final_errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(summary["average_final_error"], color='r', linestyle='--',
                       label=f'Mean: {summary["average_final_error"]:.1f}m')
    median_final = np.median(all_final_errors)
    axes[0, 2].axvline(median_final, color='g', linestyle='--',
                       label=f'Median: {median_final:.1f}m')
    axes[0, 2].axvline(CONVERGENCE_THRESHOLD, color='orange', linestyle=':',
                       label=f'Threshold: {CONVERGENCE_THRESHOLD}m')
    axes[0, 2].set_xlabel('Final Error (m)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title(f'All Paths: Final Error Distribution (n={len(all_final_errors)})')
    axes[0, 2].legend()

    # Row 2: Zoomed to first 1km
    ZOOM_DIST = 1000  # meters

    # Plot 4: Convergence curves (zoomed)
    if has_convergence_data:
        plot_convergence_curves_aggregate_xlim(eval_path, num_paths, axes[1, 0], xlim=ZOOM_DIST)
        axes[1, 0].set_title(f'All Paths: Convergence (First {ZOOM_DIST}m)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No convergence data', ha='center', va='center')
        axes[1, 0].set_title(f'Convergence (First {ZOOM_DIST}m, No Data)')

    # Plot 5: Error curves (zoomed)
    plot_error_curves_xlim(all_error_curves, all_distance_curves, axes[1, 1], xlim=ZOOM_DIST)
    axes[1, 1].set_title(f'All Paths: Error Over Distance (First {ZOOM_DIST}m)')

    # Plot 6: Integrated convergence cost histogram
    plot_convergence_cost_hist(summary, axes[1, 2])
    axes[1, 2].set_title('All Paths: Integrated Convergence Cost')

    # Row 3: Converged vs Non-converged analysis
    # Plot 7: Convergence curves for ONLY converged paths
    if has_convergence_data and converged_indices:
        plot_convergence_curves_aggregate_xlim(
            eval_path, num_paths, axes[2, 0], xlim=ZOOM_DIST, path_indices=converged_indices)
        axes[2, 0].set_title(f'Converged Paths Only: Convergence (First {ZOOM_DIST}m)\n(n={len(converged_indices)})')
    else:
        axes[2, 0].text(0.5, 0.5, 'No converged paths', ha='center', va='center')
        axes[2, 0].set_title('Converged Paths (No Data)')

    # Plot 8: Final error histogram for ONLY non-converged paths
    if non_converged_errors:
        axes[2, 1].hist(non_converged_errors, bins=20, edgecolor='black', alpha=0.7, color='coral')
        axes[2, 1].axvline(np.mean(non_converged_errors), color='r', linestyle='--',
                           label=f'Mean: {np.mean(non_converged_errors):.1f}m')
        axes[2, 1].axvline(np.median(non_converged_errors), color='g', linestyle='--',
                           label=f'Median: {np.median(non_converged_errors):.1f}m')
        axes[2, 1].set_xlabel('Final Error (m)')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].set_title(f'Non-Converged Paths: Final Error (n={len(non_converged_indices)})')
        axes[2, 1].legend()
    else:
        axes[2, 1].text(0.5, 0.5, 'All paths converged!', ha='center', va='center')
        axes[2, 1].set_title('Non-Converged Paths (None)')

    # Plot 9: Final error histogram for ONLY converged paths
    if converged_errors:
        axes[2, 2].hist(converged_errors, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[2, 2].axvline(np.mean(converged_errors), color='r', linestyle='--',
                           label=f'Mean: {np.mean(converged_errors):.1f}m')
        axes[2, 2].axvline(np.median(converged_errors), color='g', linestyle='--',
                           label=f'Median: {np.median(converged_errors):.1f}m')
        axes[2, 2].set_xlabel('Final Error (m)')
        axes[2, 2].set_ylabel('Count')
        axes[2, 2].set_title(f'Converged Paths: Final Error (n={len(converged_indices)})')
        axes[2, 2].legend()
    else:
        axes[2, 2].text(0.5, 0.5, 'No paths converged', ha='center', va='center')
        axes[2, 2].set_title('Converged Paths (None)')

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
        factor=1.0,
        landmark_version=eval_args.get("landmark_version", "v4_202001"),
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    if args.path_idx is not None:
        # Visualize specific path
        results = load_evaluation_results(eval_path, args.path_idx)

        if args.debug:
            debug_path(vigor_dataset, results, args.path_idx)

        # Use 3 columns if convergence data is available
        has_convergence = "prob_mass_by_radius" in results
        if has_convergence:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            plot_path_comparison(vigor_dataset, results, None,
                                title=f'Path #{args.path_idx}', ax=axes[0])
            plot_error_over_time(results, ax=axes[1])
            plot_convergence_curves_single_path(results, ax=axes[2])
        else:
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
