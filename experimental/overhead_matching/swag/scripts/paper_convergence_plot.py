"""Paper plots: convergence cost and localization error comparison."""

import common.torch.load_torch_deps
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import re
import argparse
from pathlib import Path


ENVIRONMENTS = [
    "NewYork",
    "SanFrancisco_mapillary",
    "Boston",
    "nightdrive",
    "Middletown",
    "Norway",
    "post_hurricane_ian_sw",
    "Framingham",
]

DISPLAY_NAMES = {
    "Boston": "Boston Snowy",
    "Framingham": "Framingham",
    "Middletown": "Middletown",
    "Norway": "Norway",
    "post_hurricane_ian": "Fort Myers",
    "post_hurricane_ian_sw": "Fort Myers",
    "SanFrancisco_mapillary": "San Francisco",
    "Seattle": "Seattle",
    "NewYork": "New York",
    "nightdrive": "Boston Night",
}

RADII = [25, 50, 100]

_PATH_DIR_RE = re.compile(r"^\d{7}$")


def _enumerate_path_dirs(eval_dir: Path) -> list[Path]:
    """Return the list of path directories, asserting they are densely 0-indexed.

    Raises if any ``{idx:07d}`` directory is missing between 0 and the highest
    index present — i.e. refuses to silently truncate when a gap would drop
    later paths from the sample.
    """
    present = sorted(
        p for p in eval_dir.iterdir() if p.is_dir() and _PATH_DIR_RE.match(p.name)
    )
    if not present:
        raise FileNotFoundError(f"No path directories (NNNNNNN) under {eval_dir}")
    expected = [eval_dir / f"{i:07d}" for i in range(len(present))]
    if [p.name for p in present] != [p.name for p in expected]:
        missing = [e.name for e in expected if not e.exists()]
        raise FileNotFoundError(
            f"Path directories under {eval_dir} are not densely numbered 0..N; "
            f"missing {missing}"
        )
    return present


def load_per_path_convergence_costs(summary_path: Path) -> dict:
    """Load per-path convergence costs from summary_statistics.json.

    Raises if any radius in ``RADII`` is missing — refuses to silently drop a
    column from the table.
    """
    with open(summary_path) as f:
        stats = json.load(f)
    result = {}
    for r in RADII:
        key = f"convergence_cost_{r}m"
        if key not in stats:
            raise KeyError(f"{summary_path} is missing {key!r}")
        result[r] = np.array(stats[key])
    return result


def load_per_path_final_errors(eval_dir: Path) -> np.ndarray:
    """Load final localization error for each path from error.pt files."""
    errors = []
    for path_dir in _enumerate_path_dirs(eval_dir):
        err = torch.load(path_dir / "error.pt", map_location="cpu")
        errors.append(err[-1].item())
    return np.array(errors)


def load_per_path_convergence_curves(
    eval_dir: Path, radius: int, distance_grid: np.ndarray
) -> np.ndarray:
    """Load per-path prob-mass curves and interpolate onto a common distance grid.

    Each path has its own number of steps and total traveled distance; we
    interpolate ``prob_mass`` over ``distance`` onto ``distance_grid`` so paths
    of different lengths can be averaged at matching distances. Grid points
    beyond a path's final distance are filled with NaN, so callers should use
    nan-aware reductions and track how many paths contributed at each grid
    point.

    Returns: (n_paths, len(distance_grid)) prob-mass array.
    """
    path_dirs = _enumerate_path_dirs(eval_dir)

    prob_masses = []
    for path_dir in path_dirs:
        dist = torch.load(path_dir / "distance_traveled_m.pt", map_location="cpu")
        pmr = torch.load(path_dir / "prob_mass_by_radius.pt", map_location="cpu")
        if radius not in pmr:
            raise KeyError(
                f"{path_dir}/prob_mass_by_radius.pt is missing radius {radius}m; "
                f"available: {sorted(pmr.keys())}"
            )
        pm = pmr[radius].numpy()
        d = dist.numpy()
        d_with_init = np.concatenate([[0.0], d])
        if len(pm) != len(d_with_init):
            raise ValueError(
                f"{path_dir}: prob_mass length {len(pm)} does not match "
                f"distance length {len(d_with_init)} (= path_len + 1)"
            )
        grid_pm = np.interp(distance_grid, d_with_init, pm)
        grid_pm = np.where(distance_grid > d_with_init[-1], np.nan, grid_pm)
        prob_masses.append(grid_pm)

    return np.asarray(prob_masses)


def compute_stats(values: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% normal-approximation CI for the mean
    (i.e. 1.96 · SEM). Bounds the uncertainty in the mean estimate, not the
    spread of the data."""
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
            convergence_costs[method][env] = load_per_path_convergence_costs(summary)
            final_errors[method][env] = load_per_path_final_errors(path)

    return convergence_costs, final_errors


# -- Figure B: Convergence Curves Over Distance --


def plot_convergence_curves(
    method_dirs: dict[str, dict[str, Path]],
    method_names: list[str],
    output_path: Path,
    radii: list[int] = [50, 100],
    xlim_m: float | None = 3000,
    method_colors: dict[str, str] | None = None,
    method_labels: dict[str, str] | None = None,
    grid_resolution_m: float = 25.0,
    min_paths_for_ci: int = 10,
):
    """Plot convergence curves (prob mass vs distance) per environment.

    Path lengths can vary across paths and methods; each path's prob-mass
    curve is interpolated onto a common distance grid (resolution
    ``grid_resolution_m``). Means / CIs use nan-aware reductions, and grid
    points where fewer than ``min_paths_for_ci`` paths contributed are
    truncated from the plot."""
    if method_colors is None:
        default_palette = ["#888888", "#FF9800", "#2196F3", "#4CAF50", "#9C27B0"]
        method_colors = {m: default_palette[i % len(default_palette)] for i, m in enumerate(method_names)}
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    # Line styles cycle for radii
    all_linestyles = ["-", "--", ":", "-."]
    radius_linestyles = {r: all_linestyles[i % len(all_linestyles)] for i, r in enumerate(radii)}

    grid_max = xlim_m if xlim_m is not None else 5000.0
    distance_grid = np.arange(0.0, grid_max + grid_resolution_m, grid_resolution_m)

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
                if env not in method_dirs[method]:
                    continue
                eval_dir = method_dirs[method][env]

                prob_masses = load_per_path_convergence_curves(
                    eval_dir, radius, distance_grid
                )
                n_valid = np.sum(~np.isnan(prob_masses), axis=0)
                keep = n_valid >= min_paths_for_ci
                if not np.any(keep):
                    continue
                with np.errstate(invalid="ignore"):
                    mean_pm = np.nanmean(prob_masses, axis=0)
                    pm_std = np.nanstd(prob_masses, axis=0, ddof=1)
                ci95 = 1.96 * pm_std / np.sqrt(np.maximum(n_valid, 1))

                x = distance_grid[keep]
                y = mean_pm[keep]
                lo = (mean_pm - ci95)[keep]
                hi = (mean_pm + ci95)[keep]

                label = f"{method_labels[method]} (r={radius}m)"
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=method_colors[method],
                    linestyle=ls,
                    linewidth=1.5,
                )
                ax.fill_between(
                    x,
                    lo,
                    hi,
                    alpha=0.15,
                    color=method_colors[method],
                )

        ax.set_title(DISPLAY_NAMES[env], fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        if xlim_m is not None:
            ax.set_xlim(0, xlim_m)
        ax.set_xlabel("Distance traveled (m)", fontsize=12)
        ax.set_ylabel("Probability mass within radius" if col == 0 else "", fontsize=12)
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


# -- Table --


def _fmt_val(mean: float, ci: float, bold: bool) -> str:
    """Format a value as mean±ci for LaTeX, optionally bold.

    Both mean and CI to 1 decimal place.
    """
    s = f"{mean:.1f}$\\pm${ci:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def print_summary_table(
    convergence_costs: dict,
    final_errors: dict,
    method_names: list[str],
    method_labels: dict[str, str] | None = None,
):
    """Print a multi-method LaTeX table. Lower is better; best is bolded per row."""
    if method_labels is None:
        method_labels = {m: m for m in method_names}

    n_methods = len(method_names)
    assert n_methods >= 2, "Table expects at least 2 methods"

    # Metrics: (label, radius or None for final error). All are "lower is
    # better" so each label gets a $\downarrow$ marker. Values are in meters.
    metrics = [
        ("CC$_{100}$ (m) $\\downarrow$", 100),
        ("Final Error (m) $\\downarrow$", None),
    ]

    n_metrics = len(metrics)
    col_spec = "l" + ("r" * n_methods) * n_metrics
    # Header row 1: metric names spanning n_methods columns each
    header1_parts = []
    for label, _ in metrics:
        header1_parts.append(f"\\multicolumn{{{n_methods}}}{{c}}{{{label}}}")
    header1 = " & ".join([""] + header1_parts) + " \\\\"

    # Header row 2: method names under each metric
    header2_parts = []
    for _ in metrics:
        for m in method_names:
            header2_parts.append(method_labels[m])
    header2 = " & ".join([""] + header2_parts) + " \\\\"

    # cmidrules for grouping
    cmidrules = ""
    for i in range(n_metrics):
        col_start = 2 + i * n_methods
        col_end = col_start + n_methods - 1
        cmidrules += f"\\cmidrule(lr){{{col_start}-{col_end}}} "

    # Build all data rows first so we can pad each column to the widest cell,
    # matching the paper's hand-formatted alignment style. Padding is purely
    # cosmetic (LaTeX ignores it in rendering) but makes the .tex source diff
    # cleanly against reviewer edits.
    row_cells: list[list[str]] = []
    for env in ENVIRONMENTS:
        cells = [DISPLAY_NAMES[env]]
        for _, radius in metrics:
            vals = []
            for m in method_names:
                if radius is not None:
                    if env in convergence_costs[m]:
                        v = compute_stats(convergence_costs[m][env][radius])
                    else:
                        v = None
                else:
                    if env in final_errors[m]:
                        v = compute_stats(final_errors[m][env])
                    else:
                        v = None
                vals.append(v)

            non_none = [(i, v) for i, v in enumerate(vals) if v is not None]
            best_idx = min(non_none, key=lambda x: x[1][0])[0] if non_none else -1

            for i, v in enumerate(vals):
                if v is None:
                    cells.append("--")
                else:
                    cells.append(_fmt_val(*v, bold=(i == best_idx)))
        row_cells.append(cells)

    n_cells = len(row_cells[0])
    col_widths = [max(len(row[i]) for row in row_cells) for i in range(n_cells)]

    def _pad_data_row(cells: list[str]) -> str:
        padded = [cells[i].ljust(col_widths[i]) for i in range(n_cells)]
        return " & ".join(padded) + " \\\\"

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{Convergence cost (m) and final localization error (m) across environments. "
                  "Values shown as mean $\\pm$ 95\\% CI using standard error of the mean. "
                  "\\textbf{Bold} indicates the best method.}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append("  \\toprule")
    lines.append("  " + header1)
    lines.append("  " + cmidrules.rstrip())
    lines.append("  " + header2)
    lines.append("  \\midrule")
    for row in row_cells:
        lines.append("  " + _pad_data_row(row))
    lines.append("  \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table*}")

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
        default="/data/overhead_matching/evaluation/results/260428_v6_safa_only_noise014",
        help="Base directory for SAFA baseline results",
    )
    parser.add_argument(
        "--ours_dir",
        type=str,
        default="/data/overhead_matching/evaluation/results/260501_v6_safa_plus_norm_lm",
        help="Base directory for our method results",
    )
    parser.add_argument(
        "--osm_dir",
        type=str,
        default="/data/overhead_matching/evaluation/results/260504_160045_osm_baseline",
        help="Base directory for OSM baseline results (may cover only a subset of envs)",
    )
    parser.add_argument(
        "--seattle_results_dir",
        type=str,
        default=None,
        help="If set, override baseline_dir/ours_dir for the Seattle environment "
             "(useful when Seattle lives in a different evaluation run)",
    )
    parser.add_argument(
        "--baseline_method",
        type=str,
        default="safa_only",
        help="Subdirectory name for baseline method within each city",
    )
    parser.add_argument(
        "--ours_method",
        type=str,
        default="safa_plus_norm_lm",
        help="Subdirectory name for our method within each city",
    )
    parser.add_argument(
        "--osm_method",
        type=str,
        default="dinov3_osm",
        help="Subdirectory name for OSM baseline method within each city",
    )
    parser.add_argument(
        "--osm_label",
        type=str,
        default="DINOv3+OSM",
        help="Display label for the OSM-baseline method in legends and tables",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        default="WAG",
        help="Display label for the SAFA/WAG baseline method in legends and tables",
    )
    parser.add_argument(
        "--early_dir",
        type=str,
        default=None,
        help="Base directory for an additional 'early fusion' method (per-city "
             "subdirs; may cover a subset of envs). Skipped if not set.",
    )
    parser.add_argument(
        "--early_method",
        type=str,
        default="early_fusion_attempt_v1",
        help="Subdirectory name for the early-fusion method within each city",
    )
    parser.add_argument(
        "--early_label",
        type=str,
        default="Early Fusion",
        help="Display label for the early-fusion method in legends and tables",
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
    parser.add_argument(
        "--allow_partial_method_coverage",
        action="store_true",
        help="Allow optional methods (OSM, Early Fusion) to cover only a subset "
             "of ENVIRONMENTS. By default every requested method must have data "
             "for every env or loading throws.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline_base = Path(args.baseline_dir)
    ours_base = Path(args.ours_dir)
    osm_base = Path(args.osm_dir) if args.osm_dir else None
    early_base = Path(args.early_dir) if args.early_dir else None
    seattle_base = Path(args.seattle_results_dir) if args.seattle_results_dir else None

    def base_for(env: str, default_base: Path) -> Path:
        if env == "Seattle" and seattle_base is not None:
            return seattle_base
        return default_base

    method_dirs = {
        "safa": {
            env: base_for(env, baseline_base) / env / args.baseline_method
            for env in ENVIRONMENTS
        },
        "osm": {},
        "early": {},
        "ours": {
            env: base_for(env, ours_base) / env / args.ours_method
            for env in ENVIRONMENTS
        },
    }
    def register_optional(base: Path | None, method_subdir: str) -> dict[str, Path]:
        if base is None:
            return {}
        result = {}
        for env in ENVIRONMENTS:
            candidate = base / env / method_subdir
            if args.allow_partial_method_coverage and not candidate.exists():
                continue
            result[env] = candidate
        return result

    method_dirs["osm"] = register_optional(osm_base, args.osm_method)
    method_dirs["early"] = register_optional(early_base, args.early_method)

    method_names = ["safa", "osm", "early", "ours"]
    if early_base is None:
        method_names = [m for m in method_names if m != "early"]
    method_colors = {
        "safa": "#888888",
        "osm": "#FF9800",
        "early": "#4CAF50",
        "ours": "#2196F3",
    }
    method_labels = {
        "safa": args.baseline_label,
        "osm": args.osm_label,
        "early": args.early_label,
        "ours": "Ours",
    }

    # Load data
    print("Loading convergence costs and final errors...")
    convergence_costs, final_errors = load_method_data(method_dirs)

    # Figure B: Convergence curves
    print("\nGenerating Figure B: Convergence curves over distance...")
    xlim = args.xlim_m if args.xlim_m > 0 else None
    plot_convergence_curves(
        method_dirs,
        method_names,
        output_path,
        radii=args.curve_radii,
        xlim_m=xlim,
        method_colors=method_colors,
        method_labels=method_labels,
    )

    # Table
    print("\n\n=== SUMMARY TABLE ===\n")
    print_summary_table(
        convergence_costs,
        final_errors,
        method_names,
        method_labels=method_labels,
    )


if __name__ == "__main__":
    main()
