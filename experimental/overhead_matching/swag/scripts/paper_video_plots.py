"""Paper-video convergence plots: one standalone figure per environment.

Curated subset of ``paper_convergence_plot`` for the paper video:
  * one single-panel convergence-curve figure per environment,
  * r=100m only (the r=50m / "CC50" curves are dropped),
  * methods WAG / WAG+OSM / LOCI (LOCI-EF dropped),
  * explicit X and Y axis labels on every panel.

Environments (with their eval-result sources):
  * San Francisco / Boston Snowy -> 260709_new_satellite_imagery (latest)
  * Framingham Mixed-Sat         -> 260522_full_rerun_no_hinge (the leaf-on
    "Framingham w/ Leaves" run, renamed to "Framingham Mixed-Sat")

The two source runs use different directory layouts, so each environment's
per-method eval dirs are listed explicitly rather than derived from a base.
"""

import common.torch.load_torch_deps  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import re
from pathlib import Path

from experimental.overhead_matching.swag.scripts.paper_convergence_plot import (
    load_per_path_convergence_curves,
)

RESULTS = Path("/data/overhead_matching/evaluation/results")
NEW_SAT = RESULTS / "260709_new_satellite_imagery"
LEAFY = RESULTS / "260522_full_rerun_no_hinge"

# Method draw order + colors, matched to paper_convergence_plot (LOCI-EF dropped).
METHOD_ORDER = ["WAG", "WAG+OSM", "LOCI"]
METHOD_COLORS = {"WAG": "#888888", "WAG+OSM": "#FF9800", "LOCI": "#2196F3"}


def _new_sat_env(env: str) -> dict[str, Path]:
    """Per-method eval dirs for an env in the 260709 run (<root>/<m>/<env>/<m>)."""
    return {m: NEW_SAT / sub / env / sub
            for m, sub in [("WAG", "wag"),
                           ("WAG+OSM", "wag_plus_osm"),
                           ("LOCI", "loci")]}


# (display_name, {method: eval_dir}). Order = panel/file order.
ENVIRONMENTS = [
    ("San Francisco", _new_sat_env("SanFrancisco_mapillary")),
    ("Boston Snowy", _new_sat_env("Boston")),
    ("Framingham Mixed-Sat", {
        "WAG": LEAFY / "wag_only_no_hinge" / "Framingham" / "wag_only_no_hinge",
        "WAG+OSM": LEAFY / "wag_plus_osm_no_hinge" / "Framingham" / "wag_plus_osm_no_hinge",
        "LOCI": LEAFY / "cosmos_no_hinge_sigma0.46" / "Framingham" / "cosmos_no_hinge_sigma0.46",
    }),
]


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_one_env(
    display_name: str,
    method_dirs: dict[str, Path],
    output_path: Path,
    radius: int = 100,
    xlim_m: float = 3000,
    grid_resolution_m: float = 25.0,
    min_paths_for_ci: int = 10,
):
    """Plot a single environment's convergence curves (prob mass vs distance).

    Per-path curves are interpolated onto a common distance grid and averaged
    with nan-aware reductions; grid points with fewer than ``min_paths_for_ci``
    contributing paths are truncated. Shaded band is the 95% normal-approx CI
    for the mean (1.96 * SEM). Mirrors paper_convergence_plot's math."""
    distance_grid = np.arange(0.0, xlim_m + grid_resolution_m, grid_resolution_m)

    fig, ax = plt.subplots(figsize=(5, 4))
    for method in METHOD_ORDER:
        eval_dir = method_dirs[method]
        prob_masses = load_per_path_convergence_curves(eval_dir, radius, distance_grid)
        n_valid = np.sum(~np.isnan(prob_masses), axis=0)
        keep = n_valid >= min_paths_for_ci
        if not np.any(keep):
            continue
        with np.errstate(invalid="ignore"):
            mean_pm = np.nanmean(prob_masses, axis=0)
            pm_std = np.nanstd(prob_masses, axis=0, ddof=1)
        ci95 = 1.96 * pm_std / np.sqrt(np.maximum(n_valid, 1))

        x = distance_grid[keep]
        ax.plot(x, mean_pm[keep], label=method, color=METHOD_COLORS[method], linewidth=2.0)
        ax.fill_between(
            x, (mean_pm - ci95)[keep], (mean_pm + ci95)[keep],
            alpha=0.15, color=METHOD_COLORS[method],
        )

    ax.set_title(display_name, fontsize=15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, xlim_m)
    ax.set_xlabel("Distance traveled (m)", fontsize=13)
    ax.set_ylabel(f"Probability mass within {radius} m", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.tick_params(labelsize=11)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()

    slug = _slug(display_name)
    for ext in ("pdf", "png"):
        fig.savefig(output_path / f"convergence_{slug}_r{radius}.{ext}",
                    bbox_inches="tight", dpi=200)
    print(f"Saved {output_path / f'convergence_{slug}_r{radius}.{{pdf,png}}'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Paper-video per-env convergence plots")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(NEW_SAT / "paper_outputs" / "video_plots"),
        help="Directory to save the per-environment figures",
    )
    parser.add_argument("--radius", type=int, default=100, help="Convergence radius (m)")
    parser.add_argument("--xlim_m", type=float, default=3000, help="X-axis limit (m)")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for display_name, method_dirs in ENVIRONMENTS:
        plot_one_env(display_name, method_dirs, output_path,
                     radius=args.radius, xlim_m=args.xlim_m)


if __name__ == "__main__":
    main()
