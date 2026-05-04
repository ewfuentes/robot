"""Grid-search constant (σ_img, σ_lm) directly against trajectory convergence cost.

This is the "oracle constant-σ" baseline that the per-step learned policy
must beat.  It complements :mod:`calibrate_sigma` (per-pair NLL) and
:mod:`calibrate_fusion_sigma` (joint NLL on EA fused softmax) by optimizing
the *trajectory-level* metric the production filter actually cares about
(see ``compute_convergence_cost`` in
``experimental/overhead_matching/swag/evaluation/convergence_metrics.py``).

For each grid point we instantiate an :class:`EntropyAdaptiveAggregator`,
run the existing histogram filter rollout on every Seattle path, and
accumulate per-trajectory metrics:

  * Mean convergence cost at a given radius (lower is better)
  * Mean steps-to-converge (first step where prob_mass ≥ threshold)
  * Mean final position error (mean estimator and MAP estimator)

Outputs:

  * ``grid_search_results.json`` — flat list of one entry per (σ_img, σ_lm)
  * ``grid_heatmap_<radius>.png`` — convergence-cost heatmap per radius

This script intentionally does **not** save per-path tensors to disk: with a
5×5 grid that's 25× the disk usage of one ``evaluate_histogram_on_paths`` run
and we only need the aggregate metrics.
"""

from dataclasses import asdict
from pathlib import Path
import argparse
import itertools
import json

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
    HistogramFilterConfig,
    construct_path_eval_inputs_from_args,
    get_dataset_bounds,
    get_patch_positions_px,
    run_histogram_filter_on_path,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    EntropyAdaptiveAggregator,
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    compute_convergence_cost,
)
from experimental.overhead_matching.swag.evaluation.odometry_noise import (
    OdometryNoiseConfig,
    add_noise_to_motion_deltas,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from common.gps import web_mercator
from common.math.haversine import find_d_on_unit_circle


def parse_sigma_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def steps_to_converge(prob_mass: torch.Tensor, threshold: float) -> int | None:
    """Return the first step index where prob_mass ≥ threshold, or None.

    `prob_mass` is the (path_len + 1,) tensor recorded after each observation
    update by ``run_histogram_filter_on_path``.
    """
    above = prob_mass >= threshold
    if not above.any():
        return None
    return int(above.float().argmax().item())


def final_error_meters(
    vigor_dataset: vd.VigorDataset,
    path: list[str],
    estimate_history: torch.Tensor,
) -> float:
    """Final-step distance error in meters between estimate and ground truth."""
    true_latlon = vigor_dataset.get_panorama_positions(path).to(estimate_history.device)
    estimates = estimate_history[-len(path):]
    d = vd.EARTH_RADIUS_M * find_d_on_unit_circle(true_latlon[-1], estimates[-1])
    return float(d.item())


def evaluate_one_sigma_pair(
    vigor_dataset,
    paths,
    grid_spec,
    mapping,
    img_sim_matrix,
    lm_sim_matrix,
    sigma_img: float,
    sigma_lm: float,
    config: HistogramFilterConfig,
    convergence_radii: list[int],
    converge_threshold: float,
    seed: int,
    device: torch.device,
):
    """Run the histogram filter on every path with a constant (σ_img, σ_lm)."""
    aggregator = EntropyAdaptiveAggregator(
        image_similarity_matrix=img_sim_matrix,
        landmark_similarity_matrix=lm_sim_matrix,
        panorama_metadata=vigor_dataset._panorama_metadata,
        image_sigma=sigma_img,
        landmark_sigma=sigma_lm,
        device=device,
    )

    convergence_costs_by_radius: dict[int, list[float]] = {r: [] for r in convergence_radii}
    steps_to_converge_by_radius: dict[int, list[int | None]] = {
        r: [] for r in convergence_radii
    }
    final_errors = []
    final_mode_errors = []

    for i, path in enumerate(paths):
        generator_seed = seed * (i + 1)

        belief = HistogramBelief.from_uniform(grid_spec=grid_spec, device=device)
        motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(device)

        if config.odometry_noise is not None:
            start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(device)
            noise_gen = torch.Generator(device="cpu").manual_seed(
                config.odometry_noise.seed * (i + 1)
            )
            motion_deltas = add_noise_to_motion_deltas(
                motion_deltas.cpu(), start_latlon.cpu(), config.odometry_noise,
                generator=noise_gen,
            ).to(device)

        true_latlons = vigor_dataset.get_panorama_positions(path).to(device)

        result = run_histogram_filter_on_path(
            belief=belief,
            motion_deltas=motion_deltas,
            path_pano_ids=path,
            log_likelihood_aggregator=aggregator,
            mapping=mapping,
            config=config,
            true_latlons=true_latlons,
            convergence_radii=convergence_radii,
        )

        distance_traveled_m = es.compute_distance_traveled(vigor_dataset, path)
        for radius in convergence_radii:
            prob_mass = result.prob_mass_by_radius[radius]
            convergence_costs_by_radius[radius].append(
                compute_convergence_cost(prob_mass, distance_traveled_m)
            )
            steps_to_converge_by_radius[radius].append(
                steps_to_converge(prob_mass, converge_threshold)
            )

        final_errors.append(final_error_meters(vigor_dataset, path, result.mean_history))
        final_mode_errors.append(
            final_error_meters(vigor_dataset, path, result.mode_history)
        )

    summary = {
        "sigma_img": sigma_img,
        "sigma_lm": sigma_lm,
        "num_paths": len(paths),
        "mean_final_error_m": float(np.mean(final_errors)),
        "mean_final_mode_error_m": float(np.mean(final_mode_errors)),
    }
    for radius in convergence_radii:
        costs = convergence_costs_by_radius[radius]
        summary[f"mean_convergence_cost_{radius}m"] = float(np.mean(costs))
        summary[f"median_convergence_cost_{radius}m"] = float(np.median(costs))
        # steps-to-converge: report mean and fraction-converged (as a path-level
        # binary that we can read alongside).
        steps = steps_to_converge_by_radius[radius]
        converged = [s for s in steps if s is not None]
        summary[f"frac_paths_converged_{radius}m"] = (
            len(converged) / len(steps) if steps else 0.0
        )
        summary[f"mean_steps_to_converge_{radius}m"] = (
            float(np.mean(converged)) if converged else float("nan")
        )
    return summary


def plot_heatmap(
    sigmas_img: list[float],
    sigmas_lm: list[float],
    values: np.ndarray,
    title: str,
    output_path: Path,
    annotate_min: bool = True,
) -> None:
    """Pcolormesh of `values` (shape: K_img × K_lm) on log-scaled σ axes."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    si = np.array(sigmas_img)
    sl = np.array(sigmas_lm)
    pcm = ax.pcolormesh(si, sl, values.T, shading="auto", cmap="viridis")
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label(title.split(" — ")[-1] if " — " in title else "value")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("σ_img")
    ax.set_ylabel("σ_lm")
    if annotate_min:
        flat = int(np.argmin(values))
        bi, bl = flat // values.shape[1], flat % values.shape[1]
        ax.scatter(
            [si[bi]], [sl[bl]], marker="*", color="white", edgecolor="red",
            s=240, lw=1.5, zorder=5,
            label=f"min ({si[bi]:.4f}, {sl[bl]:.4f}) = {values[bi, bl]:.3f}",
        )
        ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-path", type=str, required=True,
                        help="JSON file produced by create_evaluation_paths.")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--img-sim-path", type=str, required=True)
    parser.add_argument("--lm-sim-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--sigmas-img", type=str, default="0.05,0.10,0.13,0.19,0.25",
        help="Comma-separated σ_img values.")
    parser.add_argument(
        "--sigmas-lm", type=str, default="0.13,0.20,0.26,0.40,0.53",
        help="Comma-separated σ_lm values.")
    parser.add_argument(
        "--convergence-radii", type=str, default="25,50,100",
        help="Comma-separated radii (meters) for convergence cost.")
    parser.add_argument(
        "--converge-threshold", type=float, default=0.5,
        help="prob_mass threshold for steps-to-converge.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-noise-frac", type=float, default=0.05,
                        help="Wiener filter noise (m/√m).")
    parser.add_argument("--subdivision-factor", type=int, default=4)
    parser.add_argument("--max-chunk-gib", type=float, default=2.0)
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005)
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640)
    parser.add_argument("--odometry-noise-frac", type=float, default=None)
    parser.add_argument("--odometry-noise-seed", type=int, default=7919)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--max-paths", type=int, default=None,
        help="Subsample to the first N paths (for quick smoke tests).")
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    sigmas_img = parse_sigma_list(args.sigmas_img)
    sigmas_lm = parse_sigma_list(args.sigmas_lm)
    convergence_radii = [int(r.strip()) for r in args.convergence_radii.split(",")]

    print(f"Loading dataset metadata + path set from {args.paths_path}")
    vigor_dataset, _, _, paths_data = construct_path_eval_inputs_from_args(
        dataset_path=args.dataset_path,
        paths_path=args.paths_path,
        panorama_neighbor_radius_deg=args.panorama_neighbor_radius_deg,
        panorama_landmark_radius_px=args.panorama_landmark_radius_px,
        device=device,
        landmark_version=args.landmark_version,
    )
    paths = paths_data["paths"]
    if args.max_paths is not None:
        paths = paths[: args.max_paths]
    print(f"  {len(paths)} paths, dataset factor={paths_data.get('args', {}).get('factor', 1.0)}")

    print(f"Loading image similarity matrix from {args.img_sim_path}")
    img_sim = _load_similarity_matrix(Path(args.img_sim_path))
    print(f"  shape: {tuple(img_sim.shape)}")
    print(f"Loading landmark similarity matrix from {args.lm_sim_path}")
    lm_sim = _load_similarity_matrix(Path(args.lm_sim_path))
    print(f"  shape: {tuple(lm_sim.shape)}")
    assert img_sim.shape == lm_sim.shape, (
        f"shape mismatch: img {tuple(img_sim.shape)} vs lm {tuple(lm_sim.shape)}")

    # Read source_px from satellite_bbox.json if available (matches
    # evaluate_histogram_on_paths' main()).
    source_px = None
    sat_bbox_path = Path(args.dataset_path) / "satellite_bbox.json"
    if sat_bbox_path.exists():
        with open(sat_bbox_path) as f:
            sat_bbox = json.load(f)
        source_px = sat_bbox.get("source_px")

    odometry_noise_config = None
    if args.odometry_noise_frac is not None:
        odometry_noise_config = OdometryNoiseConfig(
            sigma_noise_frac=args.odometry_noise_frac, seed=args.odometry_noise_seed,
        )

    config = HistogramFilterConfig(
        motion_noise_frac=args.motion_noise_frac,
        subdivision_factor=args.subdivision_factor,
        source_px=source_px,
        odometry_noise=odometry_noise_config,
        max_chunk_gib=args.max_chunk_gib,
    )

    # Build grid + cell-to-patch mapping ONCE (expensive).
    print("Building grid spec and cell-to-patch mapping (one-time setup)...")
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    footprint_px = config.footprint_px
    cell_size_px = footprint_px / config.subdivision_factor
    patch_half_size_px = footprint_px / 2.0
    center_lat = (min_lat + max_lat) / 2
    ref_y, ref_x = web_mercator.latlon_to_pixel_coords(
        center_lat, min_lon, config.zoom_level
    )
    buf_lat, _ = web_mercator.pixel_coords_to_latlon(
        ref_y - patch_half_size_px, ref_x, config.zoom_level
    )
    _, buf_lon = web_mercator.pixel_coords_to_latlon(
        ref_y, ref_x + patch_half_size_px, config.zoom_level
    )
    grid_spec = GridSpec.from_bounds_and_cell_size(
        min_lat=min_lat - (buf_lat - center_lat),
        max_lat=max_lat + (buf_lat - center_lat),
        min_lon=min_lon - (buf_lon - min_lon),
        max_lon=max_lon + (buf_lon - min_lon),
        zoom_level=config.zoom_level,
        cell_size_px=cell_size_px,
    )
    print(f"  grid {grid_spec.num_rows}x{grid_spec.num_cols}, "
          f"cell_size_px={cell_size_px:.0f}")
    patch_positions_px = get_patch_positions_px(vigor_dataset, device)
    mapping = build_cell_to_patch_mapping(
        grid_spec=grid_spec,
        patch_positions_px=patch_positions_px,
        patch_half_size_px=patch_half_size_px,
        device=device,
        max_chunk_bytes=int(config.max_chunk_gib * 1024**3),
    )
    print(f"  built mapping with {len(mapping.patch_indices)} overlaps")

    K_img, K_lm = len(sigmas_img), len(sigmas_lm)
    print(f"Sweeping {K_img}x{K_lm} = {K_img * K_lm} (σ_img, σ_lm) pairs over "
          f"{len(paths)} paths each...")

    results = []
    with torch.no_grad():
        for sigma_img, sigma_lm in tqdm.tqdm(
            list(itertools.product(sigmas_img, sigmas_lm))
        ):
            summary = evaluate_one_sigma_pair(
                vigor_dataset=vigor_dataset,
                paths=paths,
                grid_spec=grid_spec,
                mapping=mapping,
                img_sim_matrix=img_sim,
                lm_sim_matrix=lm_sim,
                sigma_img=sigma_img,
                sigma_lm=sigma_lm,
                config=config,
                convergence_radii=convergence_radii,
                converge_threshold=args.converge_threshold,
                seed=args.seed,
                device=device,
            )
            results.append(summary)

    with open(output_path / "grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-radius heatmaps and best-pair summary.
    best_per_radius = {}
    for radius in convergence_radii:
        costs = np.array([
            r[f"mean_convergence_cost_{radius}m"] for r in results
        ]).reshape(K_img, K_lm)
        plot_heatmap(
            sigmas_img, sigmas_lm, costs,
            title=f"Mean convergence cost @ {radius}m — Seattle (n={len(paths)} paths)",
            output_path=output_path / f"grid_heatmap_{radius}m.png",
        )
        flat = int(np.argmin(costs))
        bi, bl = flat // K_lm, flat % K_lm
        best_per_radius[radius] = {
            "sigma_img": sigmas_img[bi],
            "sigma_lm": sigmas_lm[bl],
            "mean_convergence_cost": float(costs[bi, bl]),
        }
        print(f"  radius={radius}m: best (σ_img, σ_lm) = "
              f"({sigmas_img[bi]:.4f}, {sigmas_lm[bl]:.4f}), "
              f"mean_cost = {costs[bi, bl]:.3f}")

    with open(output_path / "best_per_radius.json", "w") as f:
        json.dump(best_per_radius, f, indent=2)

    print(f"\nDone. Outputs in {output_path}")


if __name__ == "__main__":
    main()
