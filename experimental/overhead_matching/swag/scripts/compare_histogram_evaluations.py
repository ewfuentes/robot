"""Compare histogram filter evaluations across multiple approaches.

Produces separate convergence plots for each radius (25m, 50m, 100m) and
a final error comparison. Each plot shows median with IQR shading for all
runs, making it easy to compare convergence speed across methods.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:compare_histogram_evaluations -- \
        --eval-dirs /path/to/run1 /path/to/run2 ... \
        --labels "Baseline" "EA proper_noun" ... \
        --output /tmp/comparison.png \
        --xlim 1000
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
        except FileNotFoundError:
            break

    return all_prob_mass, all_errors, all_distances


def interpolate_curves(curves, distance_bins):
    """Interpolate list of (distance, values) curves onto common distance bins."""
    interpolated = []
    for dist, vals in curves:
        interp = np.interp(
            distance_bins, dist, vals, left=vals[0], right=vals[-1]
        )
        interpolated.append(interp)
    return np.array(interpolated)


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

    distance_bins = np.linspace(0, args.xlim, 200)
    colors = [f"C{i}" for i in range(len(run_data))]

    # Figure layout: 2 rows x 2 cols
    # Top row: convergence @ 25m, convergence @ 50m
    # Bottom row: convergence @ 100m, median error + summary table
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Convergence plots for each radius
    for ax, radius in zip([axes[0, 0], axes[0, 1], axes[1, 0]], [25, 50, 100]):
        cc_parts = []
        for (label, data), color in zip(run_data.items(), colors):
            if radius not in data["prob_mass"]:
                continue
            curves = data["prob_mass"][radius]
            interp = interpolate_curves(curves, distance_bins)
            median = np.median(interp, axis=0)
            q25 = np.percentile(interp, 25, axis=0)
            q75 = np.percentile(interp, 75, axis=0)
            ax.fill_between(distance_bins, q25, q75, alpha=0.2, color=color)

            # Get mean convergence cost for this radius
            s = data["summary"]
            cc_key = f"convergence_cost_{radius}m"
            cc_val = np.mean(s[cc_key]) if s and cc_key in s else None
            cc_str = f", CC={cc_val:.0f}" if cc_val is not None else ""
            cc_parts.append(f"{label}: {cc_val:.0f}" if cc_val is not None else label)

            ax.plot(
                distance_bins, median, linewidth=2, color=color,
                label=f"{label}{cc_str}",
            )

        title_cc = "  |  ".join(cc_parts)
        ax.set_xlabel("Distance Traveled (m)")
        ax.set_ylabel("P(mass within radius)")
        ax.set_title(f"Convergence @ {radius}m radius\nMean CC: {title_cc}")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, args.xlim)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom right: median error curves
    ax = axes[1, 1]
    for (label, data), color in zip(run_data.items(), colors):
        interp_errors = []
        for error, dist in zip(data["errors"], data["distances"]):
            min_len = min(len(error), len(dist))
            interp = np.interp(
                distance_bins,
                dist[:min_len],
                error[:min_len],
                left=error[0],
                right=error[-1],
            )
            interp_errors.append(interp)
        interp_errors = np.array(interp_errors)
        median = np.median(interp_errors, axis=0)
        q25 = np.percentile(interp_errors, 25, axis=0)
        q75 = np.percentile(interp_errors, 75, axis=0)
        ax.fill_between(distance_bins, q25, q75, alpha=0.2, color=color)
        s = data["summary"]
        err_str = f" (mean final: {s['average_final_error']:.1f}m)" if s else ""
        ax.plot(
            distance_bins, median, linewidth=2, color=color,
            label=f"{label}{err_str}",
        )

    ax.set_xlabel("Distance Traveled (m)")
    ax.set_ylabel("Error (m)")
    ax.set_title("Median Localization Error")
    ax.set_xlim(0, args.xlim)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Print summary table to stdout
    print("\n" + "=" * 80)
    header = f"{'Method':<40} {'Error':>8} {'CC@25':>8} {'CC@50':>8} {'CC@100':>8}"
    print(header)
    print("-" * 80)
    for label, data in run_data.items():
        s = data["summary"]
        if s is None:
            print(f"{label:<40} {'N/A':>8}")
            continue
        err = s["average_final_error"]
        cc25 = np.mean(s.get("convergence_cost_25m", [float("nan")]))
        cc50 = np.mean(s.get("convergence_cost_50m", [float("nan")]))
        cc100 = np.mean(s.get("convergence_cost_100m", [float("nan")]))
        print(f"{label:<40} {err:>8.2f} {cc25:>8.1f} {cc50:>8.1f} {cc100:>8.1f}")
    print("=" * 80)

    # Build subtitle with path count info
    path_counts = [len(d["errors"]) for d in run_data.values()]
    n_paths = path_counts[0] if len(set(path_counts)) == 1 else "/".join(map(str, path_counts))
    fig.suptitle(
        f"Histogram Filter Comparison (first {args.xlim:.0f}m)\n"
        f"{n_paths} paths, solid line = median, shading = IQR (25th-75th percentile)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
