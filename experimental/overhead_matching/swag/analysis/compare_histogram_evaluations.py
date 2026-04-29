"""Compare histogram filter convergence across multiple evaluation runs.

Used to look at how quickly different aggregation strategies (e.g., image-only
vs. landmark-fused) converge during histogram filter localization. Produces a
2x3 plot grid: convergence probability at 25m/50m/100m radii vs. distance
traveled, median localization error, path survival, and a summary table. Each
curve shows the median with IQR shading across only the paths that are still
active at that distance. Paths that have ended are excluded from statistics
rather than held at their final value.

Usage:
    bazel run //experimental/overhead_matching/swag/analysis:compare_histogram_evaluations -- \
        --eval-dirs /path/to/run1 /path/to/run2 ... \
        --labels "Baseline" "EA proper_noun" ... \
        --output /tmp/comparison.png \
        --xlim 1000 \
        --title "Norway" \
        --normalize-convergence
"""

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps
import matplotlib.pyplot as plt
import numpy as np
import torch


def collect_path_data(eval_path: Path, num_paths: int):
    """Collect per-path convergence and error data from an evaluation directory."""
    all_prob_mass = {}  # radius -> list of (distance, prob_mass)
    all_errors = []
    all_distances = []

    for i in range(num_paths):
        path_dir = eval_path / f"{i:07d}"
        if not path_dir.exists():
            break
        try:
            distance = torch.load(
                path_dir / "distance_traveled_m.pt", map_location="cpu"
            ).numpy()
            error = torch.load(path_dir / "error.pt", map_location="cpu").numpy()
            all_errors.append(error)
            all_distances.append(distance)

            pm_path = path_dir / "prob_mass_by_radius.pt"
            if pm_path.exists():
                pm = torch.load(pm_path, map_location="cpu")
                for radius, prob_mass in pm.items():
                    if radius not in all_prob_mass:
                        all_prob_mass[radius] = []
                    prob_mass_np = prob_mass.numpy()
                    min_len = min(len(prob_mass_np), len(distance))
                    all_prob_mass[radius].append(
                        (distance[:min_len], prob_mass_np[:min_len])
                    )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Path {i} directory exists but is missing {e.filename}. "
                f"This may indicate a partially completed evaluation run."
            ) from e

    return all_prob_mass, all_errors, all_distances


def interpolate_curves_masked(curves, distance_bins):
    """Interpolate curves onto common bins, masking values beyond each path's end.

    Returns:
        masked: (n_curves, n_bins) masked array — masked where the bin exceeds
                the path's maximum distance.
    """
    n = len(curves)
    m = len(distance_bins)
    values = np.full((n, m), np.nan)

    for i, (dist, vals) in enumerate(curves):
        if len(dist) == 0:
            continue  # leave row as NaN (will be masked)
        max_dist = dist[-1]
        # Only interpolate within the path's range
        valid = distance_bins <= max_dist
        values[i, valid] = np.interp(
            distance_bins[valid], dist, vals, left=vals[0],
        )

    return np.ma.masked_invalid(values)


def compute_path_survival(distances_list, distance_bins):
    """Compute fraction of paths still active at each distance bin.

    Args:
        distances_list: list of distance arrays (one per path)
        distance_bins: common distance grid

    Returns:
        (n_bins,) array of fractions in [0, 1]
    """
    max_dists = np.array([d[-1] for d in distances_list])
    n_paths = len(max_dists)
    survival = np.array([
        np.sum(max_dists >= d) / n_paths for d in distance_bins
    ])
    return survival


def masked_percentile(masked_arr, q):
    """Compute percentile along axis 0, ignoring masked values.

    For bins where all values are masked, returns NaN.
    """
    result = np.full(masked_arr.shape[1], np.nan)
    for j in range(masked_arr.shape[1]):
        col = masked_arr[:, j].compressed()
        if len(col) > 0:
            result[j] = np.percentile(col, q)
    return result


def compute_cc_values(summary, distances, normalize, radii=(25, 50, 100)):
    """Compute convergence cost values for given radii.

    Returns list of floats (one per radius). Uses path-length normalization
    if normalize is True.
    """
    cc_vals = []
    for radius in radii:
        costs = summary.get(f"convergence_cost_{radius}m", [float("nan")])
        if normalize:
            path_lengths = [d[-1] for d in distances]
            normed = [c / l for c, l in zip(costs, path_lengths) if l > 0]
            cc_vals.append(np.mean(normed) if normed else float("nan"))
        else:
            cc_vals.append(np.mean(costs))
    return cc_vals


def main():
    parser = argparse.ArgumentParser(
        description="Compare histogram filter evaluations across multiple approaches"
    )
    parser.add_argument(
        "--eval-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation result directories",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for each evaluation run",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for the plot",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=1000,
        help="X-axis limit in meters (default: 1000)",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5000,
        help="Maximum number of paths to load per run",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title prefix for the plot (e.g., city name)",
    )
    parser.add_argument(
        "--normalize-convergence",
        action="store_true",
        help="Normalize convergence costs by path length (cost / path_distance)",
    )
    args = parser.parse_args()

    assert len(args.eval_dirs) == len(args.labels), (
        f"Number of eval dirs ({len(args.eval_dirs)}) must match "
        f"number of labels ({len(args.labels)})"
    )

    # Load data for all runs
    run_data = {}
    for label, eval_dir in zip(args.labels, args.eval_dirs):
        eval_path = Path(eval_dir)
        print(f"Loading {label} from {eval_path}...")
        pm, errors, dists = collect_path_data(eval_path, args.num_paths)

        summary_path = eval_path / "summary_statistics.json"
        summary = None
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        run_data[label] = {
            "prob_mass": pm,
            "errors": errors,
            "distances": dists,
            "summary": summary,
        }
        print(f"  Loaded {len(errors)} paths")
        if summary:
            print(f"  Final error: {summary['average_final_error']:.2f}m")

    # Auto-detect xlim from data if not specified (xlim <= 0)
    if args.xlim <= 0:
        max_dist = max(
            d.max()
            for data in run_data.values()
            for d in data["distances"]
        )
        xlim = float(max_dist)
    else:
        xlim = args.xlim

    distance_bins = np.linspace(0, xlim, 200)
    colors = [f"C{i}" for i in range(len(run_data))]

    # Find the shortest path end across all runs (for vertical line)
    min_path_end = min(
        d[-1]
        for data in run_data.values()
        for d in data["distances"]
    )

    # Figure layout: top row 3 convergence plots, bottom row error + survival + summary table
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Convergence plots for each radius (top row)
    for ax, radius in zip([axes[0, 0], axes[0, 1], axes[0, 2]], [25, 50, 100]):
        for (label, data), color in zip(run_data.items(), colors):
            if radius not in data["prob_mass"]:
                continue
            curves = data["prob_mass"][radius]
            interp = interpolate_curves_masked(curves, distance_bins)
            median = masked_percentile(interp, 50)
            q25 = masked_percentile(interp, 25)
            q75 = masked_percentile(interp, 75)
            ax.fill_between(distance_bins, q25, q75, alpha=0.2, color=color)

            # Get mean convergence cost for this radius
            s = data["summary"]
            cc_key = f"convergence_cost_{radius}m"
            if s and cc_key in s:
                cc_val = compute_cc_values(s, data["distances"], args.normalize_convergence, [radius])[0]
                if args.normalize_convergence:
                    cc_str = f", nCC={cc_val:.2f}" if not np.isnan(cc_val) else ""
                else:
                    cc_str = f", CC={cc_val:.0f}"
            else:
                cc_str = ""

            ax.plot(
                distance_bins, median, linewidth=2, color=color,
                label=f"{label}{cc_str}",
            )

        ax.axvline(min_path_end, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Distance Traveled (m)")
        ax.set_ylabel("P(mass within radius)")
        ax.set_title(f"Convergence @ {radius}m radius")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, xlim)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom left: median error curves
    ax = axes[1, 0]
    for (label, data), color in zip(run_data.items(), colors):
        error_curves = []
        for error, dist in zip(data["errors"], data["distances"]):
            min_len = min(len(error), len(dist))
            error_curves.append((dist[:min_len], error[:min_len]))
        interp = interpolate_curves_masked(error_curves, distance_bins)
        median = masked_percentile(interp, 50)
        q25 = masked_percentile(interp, 25)
        q75 = masked_percentile(interp, 75)
        ax.fill_between(distance_bins, q25, q75, alpha=0.2, color=color)
        s = data["summary"]
        err_str = f" (mean final: {s['average_final_error']:.1f}m)" if s else ""
        ax.plot(
            distance_bins, median, linewidth=2, color=color,
            label=f"{label}{err_str}",
        )

    ax.axvline(min_path_end, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Distance Traveled (m)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Median Localization Error")
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom middle: path survival
    ax = axes[1, 1]
    # All runs share the same evaluation paths, so just use the first one
    first_data = next(iter(run_data.values()))
    survival = compute_path_survival(first_data["distances"], distance_bins)
    ax.plot(distance_bins, survival, linewidth=2, color="black")
    ax.fill_between(distance_bins, 0, survival, alpha=0.1, color="black")
    ax.axvline(min_path_end, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Distance Traveled (m)")
    ax.set_ylabel("Fraction of paths remaining")
    ax.set_title("Path Survival")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, xlim)
    ax.grid(True, alpha=0.3)

    # Bottom right: summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    cc_label = "nCC" if args.normalize_convergence else "CC"
    col_labels = ["Method", "Error", f"{cc_label}@25", f"{cc_label}@50", f"{cc_label}@100"]
    for label, data in run_data.items():
        s = data["summary"]
        if s is None:
            table_data.append([label, "N/A", "", "", ""])
            continue
        err = s["average_final_error"]
        cc_vals = compute_cc_values(s, data["distances"], args.normalize_convergence)
        if args.normalize_convergence:
            table_data.append([label, f"{err:.1f}", f"{cc_vals[0]:.2f}", f"{cc_vals[1]:.2f}", f"{cc_vals[2]:.2f}"])
        else:
            table_data.append([label, f"{err:.1f}", f"{cc_vals[0]:.0f}", f"{cc_vals[1]:.0f}", f"{cc_vals[2]:.0f}"])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    ax.set_title("Summary")

    # Print summary table to stdout
    print("\n" + "=" * 80)
    header = f"{'Method':<40} {'Error':>8} {cc_label + '@25':>8} {cc_label + '@50':>8} {cc_label + '@100':>8}"
    print(header)
    print("-" * 80)
    for label, data in run_data.items():
        s = data["summary"]
        if s is None:
            print(f"{label:<40} {'N/A':>8}")
            continue
        err = s["average_final_error"]
        cc_vals = compute_cc_values(s, data["distances"], args.normalize_convergence)
        if args.normalize_convergence:
            print(f"{label:<40} {err:>8.2f} {cc_vals[0]:>8.3f} {cc_vals[1]:>8.3f} {cc_vals[2]:>8.3f}")
        else:
            print(f"{label:<40} {err:>8.2f} {cc_vals[0]:>8.1f} {cc_vals[1]:>8.1f} {cc_vals[2]:>8.1f}")
    print("=" * 80)

    # Build subtitle with path count info
    path_counts = [len(d["errors"]) for d in run_data.values()]
    n_paths = path_counts[0] if len(set(path_counts)) == 1 else "/".join(map(str, path_counts))
    title_prefix = f"{args.title} — " if args.title else ""
    fig.suptitle(
        f"{title_prefix}Histogram Filter Comparison (first {xlim:.0f}m)\n"
        f"{n_paths} paths, solid line = median, shading = IQR (25th-75th percentile)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
