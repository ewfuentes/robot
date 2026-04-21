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
    for path_dir in _enumerate_path_dirs(eval_dir):
        err = torch.load(path_dir / "error.pt", map_location="cpu")
        errors.append(err[-1].item())
    return np.array(errors)


def load_per_path_convergence_curves(
    eval_dir: Path, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load per-path convergence curves (prob mass vs distance traveled).

    Returns:
        distances:   (n_paths, path_len + 1) — cumulative distance at each step,
                     with a prepended 0 for the initial state.
        prob_masses: (n_paths, path_len + 1) — prob mass within ``radius`` at
                     each step (initial + post each update).

    Raises if path directories are not densely 0-indexed, if any path lacks
    the requested radius in ``prob_mass_by_radius.pt``, or if paths have
    different lengths.
    """
    path_dirs = _enumerate_path_dirs(eval_dir)

    distances = []
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
        # prob_mass has length path_len + 1 (initial + after each step),
        # distance has length path_len; prepend 0 for the initial state.
        d_with_init = np.concatenate([[0.0], d])
        if len(pm) != len(d_with_init):
            raise ValueError(
                f"{path_dir}: prob_mass length {len(pm)} does not match "
                f"distance length {len(d_with_init)} (= path_len + 1)"
            )
        distances.append(d_with_init)
        prob_masses.append(pm)

    expected_len = len(distances[0])
    for i, (d, pm) in enumerate(zip(distances, prob_masses)):
        if len(d) != expected_len:
            raise ValueError(
                f"{path_dirs[i]}: path length {len(d)} differs from path 0 "
                f"length {expected_len}; this script requires fixed-length paths"
            )
    return np.asarray(distances), np.asarray(prob_masses)


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
            if summary.exists():
                convergence_costs[method][env] = load_per_path_convergence_costs(
                    summary
                )
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
):
    """Plot convergence curves (prob mass vs distance) per environment.

    All paths under each (env, method) must have the same length; the
    shaded band is a 95% CI for the mean (1.96 · SEM across paths)."""
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
                n_paths = len(prob_masses)
                mean_distance = distances.mean(axis=0)
                mean_pm = prob_masses.mean(axis=0)
                pm_std = prob_masses.std(axis=0, ddof=1)
                ci95 = 1.96 * pm_std / np.sqrt(n_paths)

                label = f"{method_labels[method]} (r={radius}m)"
                ax.plot(
                    mean_distance,
                    mean_pm,
                    label=label,
                    color=method_colors[method],
                    linestyle=ls,
                    linewidth=1.5,
                )
                ax.fill_between(
                    mean_distance,
                    mean_pm - ci95,
                    mean_pm + ci95,
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
        "--seattle_results_dir",
        type=str,
        default=None,
        help="If set, override baseline_dir/ours_dir for the Seattle environment "
             "(useful when Seattle lives in a different evaluation run)",
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

    baseline_base = Path(args.baseline_dir)
    ours_base = Path(args.ours_dir)
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
        "ours": {
            env: base_for(env, ours_base) / env / args.ours_method
            for env in ENVIRONMENTS
        },
    }

    method_colors = {"safa": "#888888", "ours": "#2196F3"}
    method_labels = {"safa": "SAFA", "ours": "Ours"}

    # Load data
    print("Loading convergence costs and final errors...")
    convergence_costs, final_errors = load_method_data(method_dirs)

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
