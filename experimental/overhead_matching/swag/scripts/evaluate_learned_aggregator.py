"""Evaluate a trained :class:`LearnedAggregator` on full-length paths.

Counterpart to ``grid_search_convergence``: instead of sweeping constant σ
pairs, plug a trained policy into the same trajectory rollout and report the
same trajectory-level metrics. Also reports the policy's per-step σ / α
distribution so we can tell whether it's doing meaningful per-step adaptation
versus collapsing to near-constant values.
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np
import tqdm

from common.gps import web_mercator
from common.math.haversine import find_d_on_unit_circle
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    compute_convergence_cost,
    compute_probability_mass_within_radius,
)
from experimental.overhead_matching.swag.evaluation.odometry_noise import (
    OdometryNoiseConfig,
    add_noise_to_motion_deltas,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.filter.learned_aggregator import (
    LearnedAggregator,
    SigmaPolicy,
    StepContext,
    HistoryEntry,
    extract_belief_features,
)
from experimental.overhead_matching.swag.scripts.evaluate_histogram_on_paths import (
    HistogramFilterConfig,
    construct_path_eval_inputs_from_args,
    get_dataset_bounds,
    get_patch_positions_px,
)


def evaluate_path(
    aggregator: LearnedAggregator,
    grid_spec,
    mapping,
    vigor_dataset,
    path: list[str],
    motion_noise_frac: float,
    convergence_radii: list[float],
    converge_threshold: float,
    odometry_noise: OdometryNoiseConfig | None,
    odometry_seed: int | None,
    device: torch.device,
):
    """Run the (production) hard-max filter on one path with the learned policy."""
    belief = HistogramBelief.from_uniform(grid_spec=grid_spec, device=device)
    motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(device)
    if odometry_noise is not None and odometry_seed is not None:
        start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(device)
        noise_gen = torch.Generator(device="cpu").manual_seed(odometry_seed)
        motion_deltas = add_noise_to_motion_deltas(
            motion_deltas.cpu(), start_latlon.cpu(), odometry_noise,
            generator=noise_gen,
        ).to(device)
    true_latlons = vigor_dataset.get_panorama_positions(path).to(device)

    distance_traveled_m = es.compute_distance_traveled(vigor_dataset, path)
    path_len = len(path)
    cum_dist = torch.zeros(path_len, device=device)
    pano_positions = true_latlons
    for i in range(1, path_len):
        cum_dist[i] = cum_dist[i - 1] + (
            vd.EARTH_RADIUS_M * find_d_on_unit_circle(
                pano_positions[i - 1], pano_positions[i]
            )
        )

    # One prob_mass_history per radius (parallel lists).
    prob_mass_histories: dict[float, list[float]] = {
        r: [compute_probability_mass_within_radius(belief, true_latlons[0], r)]
        for r in convergence_radii
    }
    sigma_imgs, sigma_lms, alphas = [], [], []
    history_buffer: list[HistoryEntry] = []

    def _make_step_ctx(step_idx: int) -> StepContext:
        norm_step = float(step_idx) / max(path_len - 1, 1)
        step_d = (
            float((cum_dist[step_idx] - cum_dist[step_idx - 1]).item())
            if step_idx > 0 else 0.0
        )
        b_entropy, b_log_var, b_top = extract_belief_features(belief)
        return StepContext(
            belief_entropy=b_entropy, belief_log_trace_var=b_log_var,
            belief_top_cell_mass=b_top, belief_step_index_norm=norm_step,
            norm_cum_distance=float(cum_dist[step_idx].item()) / 5000.0,
            log_step_distance=float(np.log(max(step_d, 1.0))),
            step_idx_over_path_len=norm_step,
            history=list(history_buffer),
        )

    def _push_history(step_idx: int, log_ll: torch.Tensor) -> None:
        b_entropy, b_log_var, b_top = extract_belief_features(belief)
        step_d = (
            float((cum_dist[step_idx] - cum_dist[step_idx - 1]).item())
            if step_idx > 0 else 0.0
        )
        history_buffer.append(HistoryEntry(
            belief_entropy=b_entropy,
            belief_log_trace_var=b_log_var,
            belief_top_cell_mass=b_top,
            log_step_distance=float(np.log(max(step_d, 1.0))),
            last_obs_max_log_p=float(log_ll.detach().max().item()),
        ))

    for step_idx in range(path_len - 1):
        aggregator.set_step_context(_make_step_ctx(step_idx))
        pano_id = path[step_idx]
        pano_index = aggregator._pano_id_index.get_loc(pano_id)
        img_row = aggregator.image_similarity_matrix[pano_index].to(device)
        lm_row = aggregator.landmark_similarity_matrix[pano_index].to(device)
        with torch.no_grad():
            log_ll, sigma_img, sigma_lm, alpha = aggregator.fused_log_likelihood(
                img_row, lm_row
            )
        sigma_imgs.append(float(sigma_img.item()))
        sigma_lms.append(float(sigma_lm.item()))
        alphas.append(float(alpha.item()))

        # Production hard-max path (no surrogate_tau).
        belief.apply_observation(log_ll, mapping)
        for r in convergence_radii:
            prob_mass_histories[r].append(
                compute_probability_mass_within_radius(
                    belief, true_latlons[step_idx + 1], r
                )
            )
        _push_history(step_idx, log_ll)
        belief.apply_motion(motion_deltas[step_idx], motion_noise_frac)

    # Final observation
    aggregator.set_step_context(_make_step_ctx(path_len - 1))
    pano_id = path[-1]
    pano_index = aggregator._pano_id_index.get_loc(pano_id)
    img_row = aggregator.image_similarity_matrix[pano_index].to(device)
    lm_row = aggregator.landmark_similarity_matrix[pano_index].to(device)
    with torch.no_grad():
        log_ll, sigma_img, sigma_lm, alpha = aggregator.fused_log_likelihood(
            img_row, lm_row
        )
    sigma_imgs.append(float(sigma_img.item()))
    sigma_lms.append(float(sigma_lm.item()))
    alphas.append(float(alpha.item()))
    belief.apply_observation(log_ll, mapping)
    for r in convergence_radii:
        prob_mass_histories[r].append(
            compute_probability_mass_within_radius(belief, true_latlons[-1], r)
        )

    # Compute per-radius convergence cost and steps-to-converge.
    costs: dict[float, float] = {}
    steps_to_converge: dict[float, int | None] = {}
    for r, hist in prob_mass_histories.items():
        pm_tensor = torch.tensor(hist)
        costs[r] = compute_convergence_cost(pm_tensor, distance_traveled_m)
        above = pm_tensor >= converge_threshold
        steps_to_converge[r] = (
            int(above.float().argmax().item()) if above.any() else None
        )

    final_mean = belief.get_mean_latlon()
    final_err = float(
        vd.EARTH_RADIUS_M * find_d_on_unit_circle(true_latlons[-1], final_mean).item()
    )

    out = {
        "final_error_m": final_err,
        "sigma_img_mean": float(np.mean(sigma_imgs)),
        "sigma_img_std": float(np.std(sigma_imgs)),
        "sigma_lm_mean": float(np.mean(sigma_lms)),
        "sigma_lm_std": float(np.std(sigma_lms)),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "path_len": path_len,
    }
    for r in convergence_radii:
        out[f"convergence_cost_{int(r)}m"] = costs[r]
        out[f"steps_to_converge_{int(r)}m"] = steps_to_converge[r]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--img-sim-path", type=str, required=True)
    parser.add_argument("--lm-sim-path", type=str, required=True)
    parser.add_argument("--policy-weights", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--motion-noise-frac", type=float, default=0.05)
    parser.add_argument("--subdivision-factor", type=int, default=4)
    parser.add_argument("--convergence-radii", type=str, default="25,50,100",
                        help="Comma-separated radii (m) for convergence cost.")
    parser.add_argument("--converge-threshold", type=float, default=0.5)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--max-chunk-gib", type=float, default=2.0)
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005)
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640)
    parser.add_argument("--odometry-noise-frac", type=float, default=None)
    parser.add_argument("--odometry-noise-seed", type=int, default=7919)
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device(args.device)

    print("Loading dataset / paths...")
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
    print(f"  {len(paths)} paths")

    print("Loading similarity matrices...")
    img_sim = _load_similarity_matrix(Path(args.img_sim_path))
    lm_sim = _load_similarity_matrix(Path(args.lm_sim_path))

    source_px = None
    sat_bbox_path = Path(args.dataset_path) / "satellite_bbox.json"
    if sat_bbox_path.exists():
        with open(sat_bbox_path) as f:
            source_px = json.load(f).get("source_px")
    odometry_noise = None
    if args.odometry_noise_frac is not None:
        odometry_noise = OdometryNoiseConfig(
            sigma_noise_frac=args.odometry_noise_frac, seed=args.odometry_noise_seed,
        )
    hist_cfg = HistogramFilterConfig(
        motion_noise_frac=args.motion_noise_frac,
        subdivision_factor=args.subdivision_factor,
        source_px=source_px,
    )

    print("Building grid + mapping...")
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    footprint_px = hist_cfg.footprint_px
    cell_size_px = footprint_px / hist_cfg.subdivision_factor
    patch_half_size_px = footprint_px / 2.0
    center_lat = (min_lat + max_lat) / 2
    ref_y, ref_x = web_mercator.latlon_to_pixel_coords(
        center_lat, min_lon, hist_cfg.zoom_level
    )
    buf_lat, _ = web_mercator.pixel_coords_to_latlon(
        ref_y - patch_half_size_px, ref_x, hist_cfg.zoom_level
    )
    _, buf_lon = web_mercator.pixel_coords_to_latlon(
        ref_y, ref_x + patch_half_size_px, hist_cfg.zoom_level
    )
    grid_spec = GridSpec.from_bounds_and_cell_size(
        min_lat=min_lat - (buf_lat - center_lat),
        max_lat=max_lat + (buf_lat - center_lat),
        min_lon=min_lon - (buf_lon - min_lon),
        max_lon=max_lon + (buf_lon - min_lon),
        zoom_level=hist_cfg.zoom_level,
        cell_size_px=cell_size_px,
    )
    patch_positions_px = get_patch_positions_px(vigor_dataset, device)
    mapping = build_cell_to_patch_mapping(
        grid_spec=grid_spec,
        patch_positions_px=patch_positions_px,
        patch_half_size_px=patch_half_size_px,
        device=device,
        max_chunk_bytes=int(args.max_chunk_gib * 1024 ** 3),
    )

    print(f"Loading policy from {args.policy_weights}")
    policy = SigmaPolicy()
    state = torch.load(args.policy_weights, map_location="cpu", weights_only=False)
    policy.load_state_dict(state)
    policy.standardizer.frozen = True
    policy.eval()
    aggregator = LearnedAggregator(
        image_similarity_matrix=img_sim,
        landmark_similarity_matrix=lm_sim,
        panorama_metadata=vigor_dataset._panorama_metadata,
        policy=policy,
        device=device,
    )

    convergence_radii = [float(r.strip()) for r in args.convergence_radii.split(",")]

    print(f"Evaluating on {len(paths)} full-length paths "
          f"(radii={convergence_radii})...")
    per_path = []
    for i, path in enumerate(tqdm.tqdm(paths)):
        result = evaluate_path(
            aggregator=aggregator,
            grid_spec=grid_spec,
            mapping=mapping,
            vigor_dataset=vigor_dataset,
            path=path,
            motion_noise_frac=args.motion_noise_frac,
            convergence_radii=convergence_radii,
            converge_threshold=args.converge_threshold,
            odometry_noise=odometry_noise,
            odometry_seed=args.seed * (i + 1) if odometry_noise else None,
            device=device,
        )
        per_path.append(result)

    errs = [r["final_error_m"] for r in per_path]
    si_means = [r["sigma_img_mean"] for r in per_path]
    si_stds = [r["sigma_img_std"] for r in per_path]
    sl_means = [r["sigma_lm_mean"] for r in per_path]
    sl_stds = [r["sigma_lm_std"] for r in per_path]
    a_means = [r["alpha_mean"] for r in per_path]
    a_stds = [r["alpha_std"] for r in per_path]

    summary = {
        "num_paths": len(paths),
        "mean_final_error_m": float(np.mean(errs)),
        "mean_sigma_img_per_path_mean": float(np.mean(si_means)),
        "mean_sigma_img_per_path_std": float(np.mean(si_stds)),
        "std_sigma_img_across_paths": float(np.std(si_means)),
        "mean_sigma_lm_per_path_mean": float(np.mean(sl_means)),
        "mean_sigma_lm_per_path_std": float(np.mean(sl_stds)),
        "std_sigma_lm_across_paths": float(np.std(sl_means)),
        "mean_alpha_per_path_mean": float(np.mean(a_means)),
        "mean_alpha_per_path_std": float(np.mean(a_stds)),
        "std_alpha_across_paths": float(np.std(a_means)),
    }
    for r in convergence_radii:
        rkey = int(r)
        costs_r = [p[f"convergence_cost_{rkey}m"] for p in per_path]
        steps_r = [
            p[f"steps_to_converge_{rkey}m"] for p in per_path
            if p[f"steps_to_converge_{rkey}m"] is not None
        ]
        summary[f"mean_convergence_cost_{rkey}m"] = float(np.mean(costs_r))
        summary[f"median_convergence_cost_{rkey}m"] = float(np.median(costs_r))
        summary[f"frac_paths_converged_{rkey}m"] = len(steps_r) / max(len(paths), 1)
        summary[f"mean_steps_to_converge_{rkey}m"] = (
            float(np.mean(steps_r)) if steps_r else float("nan")
        )
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_path / "per_path.json", "w") as f:
        json.dump(per_path, f, indent=2)

    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
