"""Paper plots: convergence cost and localization error comparison."""

import common.torch.load_torch_deps
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import argparse
from pathlib import Path


ENVIRONMENTS = [
    "Seattle",
    "Boston",
    "Framingham",
    "Middletown",
    "Norway",
    "post_hurricane_ian",
    "SanFrancisco_mapillary",
    "nightdrive",
]

DISPLAY_NAMES = {
    "Boston": "Boston Snowy",
    "Framingham": "Framingham",
    "Middletown": "Middletown",
    "Norway": "Norway",
    "post_hurricane_ian": "Fort Myers",
    "SanFrancisco_mapillary": "San Francisco",
    "Seattle": "Seattle",
    "nightdrive": "Boston Night",
}

RADII = [25, 50, 100]


def load_per_path_convergence_costs(summary_path: Path) -> dict:
    """Load per-path convergence costs from summary_statistics.json."""
    with open(summary_path) as f:
        stats = json.load(f)
    return {
        r: np.array(stats[f"convergence_cost_{r}m"])
        for r in RADII
        if f"convergence_cost_{r}m" in stats
    }


def load_per_path_final_errors(eval_dir: Path) -> np.ndarray:
    """Load final localization error for each path from error.pt files."""
    errors = []
    path_idx = 0
    while True:
        err_file = eval_dir / f"{path_idx:07d}" / "error.pt"
        if not err_file.exists():
            break
        err = torch.load(err_file, map_location="cpu")
        errors.append(err[-1].item())
        path_idx += 1
    return np.array(errors)


def load_per_path_convergence_curves(
    eval_dir: Path, radius: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load per-path convergence curves (prob mass vs distance traveled).

    Returns:
        distances: list of 1-D arrays (distance traveled at each step)
        prob_masses: list of 1-D arrays (prob mass within radius at each step)
    """
    distances = []
    prob_masses = []
    path_idx = 0
    while True:
        path_dir = eval_dir / f"{path_idx:07d}"
        if not path_dir.exists():
            break
        dist = torch.load(path_dir / "distance_traveled_m.pt", map_location="cpu")
        pmr = torch.load(path_dir / "prob_mass_by_radius.pt", map_location="cpu")
        if radius in pmr:
            # prob_mass has shape (path_len + 1): initial + after each step
            # distance has shape (path_len): cumulative distance at each step
            # Align: use prob_mass[1:] (after first observation) with distance
            pm = pmr[radius].numpy()
            d = dist.numpy()
            # Prepend 0 distance for the initial state
            d_with_init = np.concatenate([[0.0], d])
            distances.append(d_with_init)
            prob_masses.append(pm)
        path_idx += 1
    return distances, prob_masses


def interpolate_curves_to_common_grid(
    distances: list[np.ndarray],
    values: list[np.ndarray],
    grid_spacing_m: float = 50.0,
    min_paths_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate variable-length curves onto a common distance grid.

    Returns:
        grid: common distance grid
        mean: mean value at each grid point
        sem: SEM at each grid point
        n_active: number of active paths at each grid point
    """
    max_dist = max(d[-1] for d in distances)
    grid = np.arange(0, max_dist + grid_spacing_m, grid_spacing_m)

    # Interpolate each path onto the grid
    interpolated = np.full((len(distances), len(grid)), np.nan)
    for i, (d, v) in enumerate(zip(distances, values)):
        # Only interpolate within the path's distance range
        mask = grid <= d[-1]
        interpolated[i, mask] = np.interp(grid[mask], d, v)

    # Compute statistics, masking where too few paths are active
    n_active = np.sum(~np.isnan(interpolated), axis=0)
    min_paths = max(1, int(len(distances) * min_paths_fraction))
    valid = n_active >= min_paths

    mean = np.nanmean(interpolated, axis=0)
    std = np.nanstd(interpolated, axis=0, ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_active)

    # Mask invalid points
    mean[~valid] = np.nan
    ci95[~valid] = np.nan

    return grid, mean, ci95, n_active


def compute_stats(values: np.ndarray) -> tuple[float, float]:
    """Compute mean and 95% CI half-width (1.96 * SEM)."""
    mean = np.mean(values)
    ci95 = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, ci95


# -- Data Loading --


def load_method_data(method_dirs: dict[str, dict[str, Path]]):
    """Load all data for all methods across all environments.

    Args:
        method_dirs: {method_name: {env_name: Path}}

    Returns:
        convergence_costs: {method: {env: {radius: array}}}
        final_errors: {method: {env: array}}
    """
    convergence_costs = {}
    final_errors = {}

    for method, env_paths in method_dirs.items():
        convergence_costs[method] = {}
        final_errors[method] = {}
        for env, path in env_paths.items():
            summary = path / "summary_statistics.json"
            if summary.exists():
                convergence_costs[method][env] = load_per_path_convergence_costs(
                    summary
                )
            final_errors[method][env] = load_per_path_final_errors(path)

    return convergence_costs, final_errors


# -- Figure A: Grouped Bar Chart --


def plot_grouped_bar_chart(
    convergence_costs: dict,
    final_errors: dict,
    method_names: list[str],
    output_path: Path,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
):
    """Plot grouped bar chart of convergence cost across environments."""
    if method_colors is None:
        method_colors = {
            method_names[0]: "#888888",
            method_names[1]: "#2196F3",
        }
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.3)

    n_envs = len(ENVIRONMENTS)
    n_methods = len(method_names)
    bar_width = 0.8 / n_methods
    x = np.arange(n_envs)

    for ax, radius in zip(axes, RADII):
        for j, method in enumerate(method_names):
            means = []
            sems = []
            for env in ENVIRONMENTS:
                costs = convergence_costs[method].get(env, {}).get(radius)
                if costs is not None and len(costs) > 0:
                    m, s = compute_stats(costs)
                    means.append(m)
                    sems.append(s)
                else:
                    means.append(0)
                    sems.append(0)

            offset = (j - (n_methods - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                means,
                bar_width,
                yerr=sems,
                label=method_labels[method],
                color=method_colors[method],
                edgecolor="white",
                linewidth=0.5,
                capsize=2,
                error_kw={"linewidth": 0.8},
            )

        ax.set_title(f"r = {radius}m", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [DISPLAY_NAMES[e] for e in ENVIRONMENTS], rotation=45, ha="right", fontsize=11
        )
        ax.set_ylabel("Convergence cost (m)" if radius == RADII[0] else "", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[0].legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path / "fig_a_grouped_bar.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(output_path / "fig_a_grouped_bar.png", bbox_inches="tight", dpi=150)
    print(f"Saved grouped bar chart to {output_path / 'fig_a_grouped_bar.pdf'}")
    plt.close(fig)


# -- Figure B: Convergence Curves Over Distance --


def plot_convergence_curves(
    method_dirs: dict[str, dict[str, Path]],
    method_names: list[str],
    output_path: Path,
    radii: list[int] = [50, 100],
    xlim_m: float | None = 3000,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
):
    """Plot convergence curves (prob mass vs distance) per environment."""
    if method_colors is None:
        method_colors = {
            method_names[0]: "#888888",
            method_names[1]: "#2196F3",
        }
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    # Line styles cycle for radii
    all_linestyles = ["-", "--", ":", "-."]
    radius_linestyles = {r: all_linestyles[i % len(all_linestyles)] for i, r in enumerate(radii)}

    n_envs = len(ENVIRONMENTS)
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), squeeze=False)

    for idx, env in enumerate(ENVIRONMENTS):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for radius in radii:
            ls = radius_linestyles[radius]
            for method in method_names:
                eval_dir = method_dirs[method].get(env)
                if eval_dir is None or not eval_dir.exists():
                    continue

                distances, prob_masses = load_per_path_convergence_curves(
                    eval_dir, radius
                )
                if not distances:
                    continue

                grid, mean, sem, n_active = interpolate_curves_to_common_grid(
                    distances, prob_masses
                )
                valid = ~np.isnan(mean)
                label = f"{method_labels[method]} (r={radius}m)"
                ax.plot(
                    grid[valid],
                    mean[valid],
                    label=label,
                    color=method_colors[method],
                    linestyle=ls,
                    linewidth=1.5,
                )
                ax.fill_between(
                    grid[valid],
                    (mean - sem)[valid],
                    (mean + sem)[valid],
                    alpha=0.15,
                    color=method_colors[method],
                )

        ax.set_title(DISPLAY_NAMES[env], fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        if xlim_m is not None:
            ax.set_xlim(0, xlim_m)
        ax.set_xlabel("Distance traveled (m)", fontsize=12)
        ax.set_ylabel("P(mass within r)" if col == 0 else "", fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
        ax.tick_params(labelsize=10)

    # Hide unused axes
    for idx in range(n_envs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Single legend from first subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(radii) * len(method_names),
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    suffix = "_".join(str(r) for r in radii)
    fig.savefig(
        output_path / f"fig_b_convergence_curves_r{suffix}.pdf",
        bbox_inches="tight",
        dpi=150,
    )
    fig.savefig(
        output_path / f"fig_b_convergence_curves_r{suffix}.png",
        bbox_inches="tight",
        dpi=150,
    )
    print(
        f"Saved convergence curves to {output_path / f'fig_b_convergence_curves_r{suffix}.pdf'}"
    )
    plt.close(fig)


# -- Figure C: Dumbbell / Paired Dot Plot --


def plot_dumbbell(
    convergence_costs: dict,
    method_names: list[str],
    output_path: Path,
    radius: int = 100,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
):
    """Plot dumbbell chart comparing two methods across environments."""
    if method_colors is None:
        method_colors = {
            method_names[0]: "#888888",
            method_names[1]: "#2196F3",
        }
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    fig, ax = plt.subplots(figsize=(8, 5))

    y_positions = np.arange(len(ENVIRONMENTS))

    for j, method in enumerate(method_names):
        means = []
        sems = []
        for env in ENVIRONMENTS:
            costs = convergence_costs[method].get(env, {}).get(radius)
            if costs is not None and len(costs) > 0:
                m, s = compute_stats(costs)
                means.append(m)
                sems.append(s)
            else:
                means.append(np.nan)
                sems.append(0)

        ax.errorbar(
            means,
            y_positions,
            xerr=sems,
            fmt="o",
            color=method_colors[method],
            label=method_labels[method],
            markersize=7,
            capsize=3,
            linewidth=0,
            elinewidth=1.2,
            zorder=3,
        )

    # Draw connecting segments
    for i, env in enumerate(ENVIRONMENTS):
        vals = []
        for method in method_names:
            costs = convergence_costs[method].get(env, {}).get(radius)
            if costs is not None and len(costs) > 0:
                vals.append(np.mean(costs))
            else:
                vals.append(np.nan)
        if len(vals) >= 2 and not any(np.isnan(vals)):
            ax.plot(vals, [i, i], color="#cccccc", linewidth=1.5, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_NAMES[e] for e in ENVIRONMENTS], fontsize=12)
    ax.set_xlabel(f"Convergence cost at r = {radius}m (m)", fontsize=13)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    fig.savefig(
        output_path / f"fig_c_dumbbell_r{radius}.pdf", bbox_inches="tight", dpi=150
    )
    fig.savefig(
        output_path / f"fig_c_dumbbell_r{radius}.png", bbox_inches="tight", dpi=150
    )
    print(f"Saved dumbbell plot to {output_path / f'fig_c_dumbbell_r{radius}.pdf'}")
    plt.close(fig)


# -- Table --


def _get_mean_ci(values: np.ndarray | None) -> tuple[float, float] | None:
    if values is None or len(values) == 0:
        return None
    return compute_stats(values)


def _fmt_val(mean: float, ci: float, bold: bool) -> str:
    """Format a value as mean±ci for LaTeX, optionally bold."""
    s = f"{mean:.0f}$\\pm${ci:.0f}"
    return f"\\textbf{{{s}}}" if bold else s


def print_summary_table(
    convergence_costs: dict,
    final_errors: dict,
    method_names: list[str],
    method_labels: dict[str, str] | None = None,
):
    """Print a paired-comparison LaTeX table. Lower is better; best is bolded."""
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    assert len(method_names) == 2, "Paired table expects exactly 2 methods"
    m_a, m_b = method_names

    # Metrics: (label, radius or None for final error)
    metrics = [(f"CC$_{{100}}$", 100), ("Error", None)]

    n_metrics = len(metrics)
    col_spec = "l" + "rr" * n_metrics
    # Header row 1: metric names spanning 2 columns each
    header1_parts = []
    for label, _ in metrics:
        header1_parts.append(f"\\multicolumn{{2}}{{c}}{{{label}}}")
    header1 = " & ".join([""] + header1_parts) + " \\\\"

    # Header row 2: method names under each metric
    header2_parts = []
    for _ in metrics:
        header2_parts.append(method_labels[m_a])
        header2_parts.append(method_labels[m_b])
    header2 = " & ".join([""] + header2_parts) + " \\\\"

    # cmidrules for grouping
    cmidrules = ""
    for i in range(n_metrics):
        col_start = 2 + i * 2
        col_end = col_start + 1
        cmidrules += f"\\cmidrule(lr){{{col_start}-{col_end}}} "

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Convergence cost (m) and final localization error (m) across environments. "
                  "Values shown as mean $\\pm$ 95\\% CI. \\textbf{Bold} indicates the better method.}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(cmidrules)
    lines.append(header2)
    lines.append("\\midrule")

    for env in ENVIRONMENTS:
        row_parts = [DISPLAY_NAMES[env]]
        for _, radius in metrics:
            if radius is not None:
                val_a = _get_mean_ci(convergence_costs[m_a].get(env, {}).get(radius))
                val_b = _get_mean_ci(convergence_costs[m_b].get(env, {}).get(radius))
            else:
                val_a = _get_mean_ci(final_errors[m_a].get(env))
                val_b = _get_mean_ci(final_errors[m_b].get(env))

            if val_a is None:
                row_parts.append("--")
            elif val_b is None:
                row_parts.append(_fmt_val(*val_a, bold=False))
            else:
                row_parts.append(_fmt_val(*val_a, bold=val_a[0] <= val_b[0]))

            if val_b is None:
                row_parts.append("--")
            elif val_a is None:
                row_parts.append(_fmt_val(*val_b, bold=False))
            else:
                row_parts.append(_fmt_val(*val_b, bold=val_b[0] <= val_a[0]))

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Paper convergence plots")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/paper_plots",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="/data/overhead_matching/evaluation/results/260310_all_non_vigor_datasets",
        help="Base directory for SAFA baseline results",
    )
    parser.add_argument(
        "--ours_dir",
        type=str,
        default="/data/overhead_matching/evaluation/results/260330_panov2_tuned_prompt",
        help="Base directory for our method results",
    )
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="image_only",
        help="Subdirectory name for baseline method within each city",
    )
    parser.add_argument(
        "--ours_method",
        type=str,
        default="ea_safa_corr",
        help="Subdirectory name for our method within each city",
    )
    parser.add_argument(
        "--curve_radii",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Radii for convergence curve plots (Fig B)",
    )
    parser.add_argument(
        "--xlim_m",
        type=float,
        default=3000,
        help="X-axis limit in meters for convergence curves (Fig B). 0 for no limit.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build method directories
    baseline_base = Path(args.baseline_dir)
    ours_base = Path(args.ours_dir)

    # Seattle uses 260325 results (older run) since it's not in 260330
    seattle_base = Path("/data/overhead_matching/evaluation/results/260325_correspondence_fusion")

    method_dirs = {
        "safa": {
            env: (seattle_base if env == "Seattle" else baseline_base) / env / args.baseline_method
            for env in ENVIRONMENTS
        },
        "ours": {
            env: (seattle_base if env == "Seattle" else ours_base) / env / args.ours_method
            for env in ENVIRONMENTS
        },
    }

    method_colors = {"safa": "#888888", "ours": "#2196F3"}
    method_labels = {"safa": "SAFA", "ours": "Ours"}

    # Load data
    print("Loading convergence costs and final errors...")
    convergence_costs, final_errors = load_method_data(method_dirs)

    # Figure A: Grouped bar chart
    print("\nGenerating Figure A: Grouped bar chart...")
    plot_grouped_bar_chart(
        convergence_costs,
        final_errors,
        ["safa", "ours"],
        output_path,
        method_colors=method_colors,
        method_labels=method_labels,
    )

    # Figure B: Convergence curves
    print("\nGenerating Figure B: Convergence curves over distance...")
    xlim = args.xlim_m if args.xlim_m > 0 else None
    plot_convergence_curves(
        method_dirs,
        ["safa", "ours"],
        output_path,
        radii=args.curve_radii,
        xlim_m=xlim,
        method_colors=method_colors,
        method_labels=method_labels,
    )

    # Figure C: Dumbbell plot
    print("\nGenerating Figure C: Dumbbell plot...")
    plot_dumbbell(
        convergence_costs,
        ["safa", "ours"],
        output_path,
        radius=args.curve_radii[-1],
        method_colors=method_colors,
        method_labels=method_labels,
    )

    # Table
    print("\n\n=== SUMMARY TABLE ===\n")
    print_summary_table(
        convergence_costs,
        final_errors,
        ["safa", "ours"],
        method_labels=method_labels,
    )


if __name__ == "__main__":
    main()
