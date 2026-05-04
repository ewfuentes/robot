"""End-to-end supervised training of the learned per-step fusion policy.

Roll the differentiable histogram filter forward on Seattle paths, score the
trajectory by the same convergence-cost formula production uses, and backprop
through the filter all the way to the policy.

The histogram filter is differentiable when ``apply_observation`` is called
with ``surrogate_tau`` set (soft-max via segment_logsumexp instead of hard
max). All other ops in the filter (motion shift, Gaussian blur, normalize)
are already differentiable.

Outputs (under ``--output-path``):

  * ``policy_weights.pt`` — best-on-validation policy state dict.
  * ``train_log.json`` — per-iteration losses and validation metrics.
  * ``train_args.json`` — full CLI args for reproducibility.

Use the trained policy by adding a ``LearnedAggregatorConfig`` to your
aggregator YAML pointing at ``policy_weights.pt`` — :func:`aggregator_from_config`
will instantiate the :class:`LearnedAggregator` automatically and the
existing ``evaluate_histogram_on_paths`` rollout works unchanged.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import random

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np
import tqdm

from common.gps import web_mercator
from common.gps.web_mercator import get_meters_per_pixel
from common.math.haversine import find_d_on_unit_circle

import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    compute_convergence_cost,
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


# ============ Differentiable convergence reward ============


def prob_mass_within_radius_tensor(
    belief: HistogramBelief,
    true_latlon: torch.Tensor,
    radius_meters: float,
) -> torch.Tensor:
    """Differentiable analogue of ``compute_probability_mass_within_radius``.

    Returns a 0-D tensor that retains the graph back to belief. The
    production version calls ``.item()`` and so cannot be used in training.
    """
    cell_centers_px = belief.grid_spec.get_all_cell_centers_px(belief.device)
    true_lat = (
        true_latlon[0].item() if isinstance(true_latlon[0], torch.Tensor)
        else float(true_latlon[0])
    )
    true_lon = (
        true_latlon[1].item() if isinstance(true_latlon[1], torch.Tensor)
        else float(true_latlon[1])
    )
    true_y_px, true_x_px = web_mercator.latlon_to_pixel_coords(
        true_lat, true_lon, belief.grid_spec.zoom_level
    )
    true_pos_px = torch.tensor(
        [[true_y_px, true_x_px]], device=belief.device, dtype=torch.float32
    )
    dist_px = torch.norm(cell_centers_px - true_pos_px, dim=1)
    meters_per_pixel = get_meters_per_pixel(true_lat, belief.grid_spec.zoom_level)
    dist_meters = dist_px * meters_per_pixel
    within_mask = dist_meters <= radius_meters
    return belief.get_belief().flatten()[within_mask].sum()


# ============ Path-distance helpers (constant w.r.t. policy) ============


def cumulative_distance_meters(
    vigor_dataset: vd.VigorDataset, path: list[str], device: torch.device
) -> torch.Tensor:
    """(path_len,) cumulative distance traveled along the path (in meters)."""
    pano_positions = vigor_dataset.get_panorama_positions(path).to(device)
    cum = [torch.tensor(0.0, device=device)]
    for i in range(len(path) - 1):
        d = vd.EARTH_RADIUS_M * find_d_on_unit_circle(
            pano_positions[i], pano_positions[i + 1]
        )
        cum.append(cum[-1] + d)
    return torch.stack(cum)


# ============ Rollout that keeps gradients flowing ============


@dataclass
class TrainConfig:
    motion_noise_frac: float = 0.05
    subdivision_factor: int = 4
    surrogate_tau: float = 0.5
    radius_m: float = 50.0
    truncate_path_length: int | None = None
    odometry_noise: OdometryNoiseConfig | None = None
    max_chunk_gib: float = 2.0


def diff_rollout_and_loss(
    aggregator: LearnedAggregator,
    grid_spec: GridSpec,
    mapping,
    vigor_dataset: vd.VigorDataset,
    path: list[str],
    cfg: TrainConfig,
    device: torch.device,
    odometry_seed: int | None = None,
    nll_anchor: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Differentiable rollout — returns (loss, diagnostics).

    Loss = mean over path-segments of (1 − prob_mass[t]) · Δd[t]   in meters.
    Optionally adds an NLL-anchor term ``λ · ((logσ_img − logσ_NLL_img)² +
    (logσ_lm − logσ_NLL_lm)²)`` to keep the policy near the calibration
    solution and prevent collapse early in training.

    Diagnostics returned (all detached floats):
      * ``trajectory_convergence_cost``: matches ``compute_convergence_cost``
      * ``mean_sigma_img``, ``mean_sigma_lm``, ``mean_alpha``
      * ``final_position_error_m`` (mean estimator)
    """
    if cfg.truncate_path_length is not None:
        path = path[: cfg.truncate_path_length]
    path_len = len(path)
    if path_len < 3:
        zero = torch.tensor(0.0, device=device)
        return zero, {
            "trajectory_convergence_cost": 0.0,
            "mean_sigma_img": 0.0, "mean_sigma_lm": 0.0, "mean_alpha": 0.0,
            "final_position_error_m": 0.0,
            "path_len": path_len,
        }

    # Set up belief and motion deltas (no_grad; these are policy-independent).
    with torch.no_grad():
        belief = HistogramBelief.from_uniform(grid_spec=grid_spec, device=device)
        motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(device)
        if cfg.odometry_noise is not None and odometry_seed is not None:
            start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(device)
            noise_gen = torch.Generator(device="cpu").manual_seed(odometry_seed)
            motion_deltas = add_noise_to_motion_deltas(
                motion_deltas.cpu(), start_latlon.cpu(), cfg.odometry_noise,
                generator=noise_gen,
            ).to(device)
        true_latlons = vigor_dataset.get_panorama_positions(path).to(device)
        cum_dist = cumulative_distance_meters(vigor_dataset, path, device)
    delta_d = cum_dist[1:] - cum_dist[:-1]  # (path_len - 1,)

    prob_masses: list[torch.Tensor] = [
        prob_mass_within_radius_tensor(belief, true_latlons[0], cfg.radius_m)
    ]

    sigma_imgs, sigma_lms, alphas = [], [], []
    history_buffer: list[HistoryEntry] = []
    pano_positions = true_latlons  # alias for clarity
    pano_count = pano_positions.shape[0]

    def _make_step_ctx(step_idx: int) -> StepContext:
        norm_step = float(step_idx) / max(path_len - 1, 1)
        cur_d = float(cum_dist[step_idx].item())
        step_d = float(
            cum_dist[step_idx].item() - cum_dist[step_idx - 1].item()
        ) if step_idx > 0 else 0.0
        b_entropy, b_log_var, b_top = extract_belief_features(belief)
        return StepContext(
            belief_entropy=b_entropy,
            belief_log_trace_var=b_log_var,
            belief_top_cell_mass=b_top,
            belief_step_index_norm=norm_step,
            norm_cum_distance=cur_d / 5000.0,  # 5km path normalization
            log_step_distance=math.log(max(step_d, 1.0)),
            step_idx_over_path_len=norm_step,
            history=list(history_buffer),
        )

    def _push_history(step_idx: int, log_ll: torch.Tensor) -> None:
        """Append the just-applied step to the history buffer."""
        b_entropy, b_log_var, b_top = extract_belief_features(belief)
        step_d = float(
            cum_dist[step_idx].item() - cum_dist[step_idx - 1].item()
        ) if step_idx > 0 else 0.0
        history_buffer.append(HistoryEntry(
            belief_entropy=b_entropy,
            belief_log_trace_var=b_log_var,
            belief_top_cell_mass=b_top,
            log_step_distance=math.log(max(step_d, 1.0)),
            last_obs_max_log_p=float(log_ll.detach().max().item()),
        ))

    for step_idx in range(path_len - 1):
        aggregator.set_step_context(_make_step_ctx(step_idx))

        # Compute fused log-likelihood (in graph).
        pano_id = path[step_idx]
        pano_index = aggregator._pano_id_index.get_loc(pano_id)
        img_row = aggregator.image_similarity_matrix[pano_index].to(device)
        lm_row = aggregator.landmark_similarity_matrix[pano_index].to(device)
        log_ll, sigma_img, sigma_lm, alpha = aggregator.fused_log_likelihood(
            img_row, lm_row
        )
        sigma_imgs.append(sigma_img)
        sigma_lms.append(sigma_lm)
        alphas.append(alpha)

        belief.apply_observation(log_ll, mapping, surrogate_tau=cfg.surrogate_tau)
        prob_masses.append(
            prob_mass_within_radius_tensor(
                belief, true_latlons[step_idx + 1], cfg.radius_m
            )
        )
        _push_history(step_idx, log_ll)
        # Motion update (no_grad — motion model is policy-independent).
        with torch.no_grad():
            belief.apply_motion(motion_deltas[step_idx], cfg.motion_noise_frac)

    # Final observation step
    aggregator.set_step_context(_make_step_ctx(path_len - 1))
    pano_id = path[-1]
    pano_index = aggregator._pano_id_index.get_loc(pano_id)
    img_row = aggregator.image_similarity_matrix[pano_index].to(device)
    lm_row = aggregator.landmark_similarity_matrix[pano_index].to(device)
    log_ll, sigma_img, sigma_lm, alpha = aggregator.fused_log_likelihood(img_row, lm_row)
    sigma_imgs.append(sigma_img)
    sigma_lms.append(sigma_lm)
    alphas.append(alpha)
    belief.apply_observation(log_ll, mapping, surrogate_tau=cfg.surrogate_tau)
    prob_masses.append(
        prob_mass_within_radius_tensor(belief, true_latlons[-1], cfg.radius_m)
    )

    prob_mass_t = torch.stack(prob_masses)  # (path_len + 1,)
    # Match compute_convergence_cost formula exactly:
    # cost = Σ (1 - prob_mass[t+2]) · Δd[t]   for t in 0..path_len-2.
    # Note: Δd has path_len-1 entries, prob_mass has path_len+1 entries.
    missing = 1.0 - prob_mass_t[2:]  # (path_len - 1,)
    seg_cost = (missing * delta_d).sum()
    # Normalize by total distance so the loss is on the order of 1, not the
    # several-thousand-meter trajectory cost.
    total_distance = delta_d.sum().clamp(min=1.0)
    loss = seg_cost / total_distance

    if nll_anchor is not None and nll_anchor.get("lambda", 0.0) > 0:
        lam = nll_anchor["lambda"]
        log_si_target = math.log(nll_anchor["sigma_img_target"])
        log_sl_target = math.log(nll_anchor["sigma_lm_target"])
        si_mean_log = torch.stack([torch.log(s) for s in sigma_imgs]).mean()
        sl_mean_log = torch.stack([torch.log(s) for s in sigma_lms]).mean()
        loss = loss + lam * (
            (si_mean_log - log_si_target) ** 2 + (sl_mean_log - log_sl_target) ** 2
        )

    diagnostics = {
        "trajectory_convergence_cost": float(seg_cost.detach().item()),
        "mean_sigma_img": float(torch.stack(sigma_imgs).mean().detach().item()),
        "mean_sigma_lm": float(torch.stack(sigma_lms).mean().detach().item()),
        "mean_alpha": float(torch.stack(alphas).mean().detach().item()),
        "final_position_error_m": 0.0,  # filled by caller if asked
        "path_len": path_len,
        "total_distance_m": float(total_distance.detach().item()),
    }
    return loss, diagnostics


# ============ Top-level training loop ============


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--img-sim-path", type=str, required=True)
    parser.add_argument("--lm-sim-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--batch-paths", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--max-paths", type=int, default=None,
                        help="Subsample to first N paths (for smoke runs).")
    parser.add_argument("--truncate-path-length", type=int, default=100,
                        help="Truncate each path to N steps for speed.")
    parser.add_argument("--surrogate-tau", type=float, default=0.5)
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--motion-noise-frac", type=float, default=0.05)
    parser.add_argument("--subdivision-factor", type=int, default=4)
    parser.add_argument("--max-chunk-gib", type=float, default=2.0)
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005)
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640)
    parser.add_argument("--nll-anchor-lambda", type=float, default=0.01)
    parser.add_argument("--nll-anchor-sigma-img", type=float, default=0.05)
    parser.add_argument("--nll-anchor-sigma-lm", type=float, default=0.13)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--odometry-noise-frac", type=float, default=None)
    parser.add_argument("--odometry-noise-seed", type=int, default=7919)
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    # ------ Data ------
    print(f"Loading dataset / paths from {args.paths_path}")
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
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * args.val_frac))
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    print(f"  train: {len(train_paths)} paths, val: {len(val_paths)} paths")

    print("Loading similarity matrices...")
    img_sim = _load_similarity_matrix(Path(args.img_sim_path))
    lm_sim = _load_similarity_matrix(Path(args.lm_sim_path))
    assert img_sim.shape == lm_sim.shape, "img/lm similarity shape mismatch"

    # ------ Filter setup (expensive, do once) ------
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

    train_cfg = TrainConfig(
        motion_noise_frac=args.motion_noise_frac,
        subdivision_factor=args.subdivision_factor,
        surrogate_tau=args.surrogate_tau,
        radius_m=args.radius_m,
        truncate_path_length=args.truncate_path_length,
        odometry_noise=odometry_noise,
        max_chunk_gib=args.max_chunk_gib,
    )
    hist_cfg = HistogramFilterConfig(
        motion_noise_frac=args.motion_noise_frac,
        subdivision_factor=args.subdivision_factor,
        source_px=source_px,
        max_chunk_gib=args.max_chunk_gib,
    )

    print("Building grid spec + cell-to-patch mapping...")
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
        max_chunk_bytes=int(hist_cfg.max_chunk_gib * 1024 ** 3),
    )

    # ------ Policy + aggregator ------
    policy = SigmaPolicy().to(device)
    aggregator = LearnedAggregator(
        image_similarity_matrix=img_sim,
        landmark_similarity_matrix=lm_sim,
        panorama_metadata=vigor_dataset._panorama_metadata,
        policy=policy,
        device=device,
    )
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    nll_anchor = {
        "lambda": args.nll_anchor_lambda,
        "sigma_img_target": args.nll_anchor_sigma_img,
        "sigma_lm_target": args.nll_anchor_sigma_lm,
    }

    # ------ Training loop ------
    log = {"iters": [], "train_loss": [], "val_loss": [],
           "val_mean_convergence_cost": [], "val_mean_sigma_img": [],
           "val_mean_sigma_lm": [], "val_mean_alpha": []}
    best_val = float("inf")

    for it in tqdm.tqdm(range(args.num_iters)):
        policy.train()
        # Update standardizer on a randomly-sampled training path's features
        # by running a quick forward without backward.
        batch = rng.sample(train_paths, k=min(args.batch_paths, len(train_paths)))
        optim.zero_grad()
        batch_loss = 0.0
        diag_accum = []
        for p_idx, path in enumerate(batch):
            loss, diag = diff_rollout_and_loss(
                aggregator=aggregator,
                grid_spec=grid_spec,
                mapping=mapping,
                vigor_dataset=vigor_dataset,
                path=path,
                cfg=train_cfg,
                device=device,
                odometry_seed=args.seed * (it + 1) + p_idx,
                nll_anchor=nll_anchor,
            )
            (loss / len(batch)).backward()
            batch_loss += float(loss.detach().item())
            diag_accum.append(diag)
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=args.grad_clip_norm
        )
        optim.step()

        # Update standardizer with features from the most recent rollout.
        # Cheap & approximate; lets eval-time normalization track training.
        with torch.no_grad():
            sample_idx = aggregator._pano_id_index.get_loc(batch[0][0])
            sample_features = aggregator._compute_features(
                aggregator.image_similarity_matrix[sample_idx].to(device),
                aggregator.landmark_similarity_matrix[sample_idx].to(device),
            )
            policy.standardizer.update(sample_features.unsqueeze(0))

        log["iters"].append(it)
        log["train_loss"].append(batch_loss / max(len(batch), 1))

        if (it + 1) % args.val_every == 0 or it == args.num_iters - 1:
            policy.eval()
            with torch.no_grad():
                val_losses = []
                val_costs = []
                v_si, v_sl, v_a = [], [], []
                for vp in val_paths:
                    vloss, vdiag = diff_rollout_and_loss(
                        aggregator=aggregator,
                        grid_spec=grid_spec,
                        mapping=mapping,
                        vigor_dataset=vigor_dataset,
                        path=vp,
                        cfg=train_cfg,
                        device=device,
                        odometry_seed=None,
                        nll_anchor=None,
                    )
                    val_losses.append(float(vloss.detach().item()))
                    val_costs.append(vdiag["trajectory_convergence_cost"])
                    v_si.append(vdiag["mean_sigma_img"])
                    v_sl.append(vdiag["mean_sigma_lm"])
                    v_a.append(vdiag["mean_alpha"])
            v_mean = float(np.mean(val_losses))
            log["val_loss"].append(v_mean)
            log["val_mean_convergence_cost"].append(float(np.mean(val_costs)))
            log["val_mean_sigma_img"].append(float(np.mean(v_si)))
            log["val_mean_sigma_lm"].append(float(np.mean(v_sl)))
            log["val_mean_alpha"].append(float(np.mean(v_a)))
            print(
                f"  it={it+1}: train={log['train_loss'][-1]:.4f}  "
                f"val={v_mean:.4f}  cc={np.mean(val_costs):.1f}  "
                f"σ_img={np.mean(v_si):.3f}  σ_lm={np.mean(v_sl):.3f}  "
                f"α={np.mean(v_a):.3f}"
            )
            if v_mean < best_val:
                best_val = v_mean
                torch.save(policy.state_dict(), output_path / "policy_weights.pt")

        with open(output_path / "train_log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Outputs in {output_path}")


if __name__ == "__main__":
    main()
