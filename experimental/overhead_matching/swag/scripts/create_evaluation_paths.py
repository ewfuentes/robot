import argparse
import matplotlib.pyplot as plt
import json
import tqdm
import common.torch.load_torch_deps
import torch
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to Vigor dataset")
    parser.add_argument("--factor", type=float, required=False, help="Dataset factor", default=1.0)
    parser.add_argument("--out", type=Path, required=True, help="Save json file path")
    parser.add_argument("--path-length-m", type=float, required=True, help="Path length meters")
    parser.add_argument("--num-paths", type=int, required=True, help="Number of paths to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--skip-plots", action="store_true", help="skip path visualization")
    parser.add_argument("--turn-temperature", type=float, default=0.1, help="Turn temperature. Higher value more turns")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005, help="Panorama neighbor radius deg")
    # Goal-directed path generation arguments
    parser.add_argument("--goal-directed", action="store_true", help="Use goal-directed A* path generation instead of random walk")
    parser.add_argument("--goal-reached-threshold-m", type=float, default=200.0, help="Distance (m) to consider goal reached")
    parser.add_argument("--min-goal-distance-percentile", type=float, default=0.75, help="Select goals at this percentile of max distance")
    parser.add_argument("--min-bbox-coverage-ratio", type=float, default=0.05, help="Reject paths covering less than this fraction of city area")
    parser.add_argument("--goal-resample-distance-m", type=float, default=5000.0, help="Resample goal after traveling this distance")
    parser.add_argument("--edge-cost-noise-factor", type=float, default=0.0, help="Multiplicative noise factor for edge costs")
    parser.add_argument("--quadrant-goals", action="store_true", help="Select goals in different quadrants")
    parser.add_argument("--visited-penalty", type=float, default=0.0,
                        help="Penalty multiplier for edges leaving visited nodes (encourages path diversity)")

    args = parser.parse_args()
    args.out = Path(args.out)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg,
        factor=args.factor)
    dataset = vd.VigorDataset(Path(args.dataset_path), dataset_config)
    paths = []

    # Track visit counts for diversity penalty
    visit_counts = [0] * len(dataset._panorama_metadata) if args.visited_penalty > 0 else None
    base_edge_costs = None
    if args.visited_penalty > 0 and args.goal_directed:
        if not hasattr(dataset, '_cached_edge_costs'):
            dataset._cached_edge_costs = dataset._precompute_edge_costs()
        base_edge_costs = dataset._cached_edge_costs

    for i in tqdm.tqdm(range(args.num_paths)):
        generator = torch.manual_seed(args.seed + i)

        # Apply visited-node penalty if enabled
        if args.visited_penalty > 0 and args.goal_directed and base_edge_costs is not None:
            penalized_edge_costs = []
            for node_idx, node_costs in enumerate(base_edge_costs):
                penalty = 1.0 + args.visited_penalty * visit_counts[node_idx]
                penalized_costs = [cost * penalty for cost in node_costs]
                penalized_edge_costs.append(penalized_costs)
            dataset._cached_edge_costs = penalized_edge_costs

        if args.goal_directed:
            paths.append(dataset.generate_goal_directed_path(
                generator,
                args.path_length_m,
                goal_reached_threshold_m=args.goal_reached_threshold_m,
                min_goal_distance_percentile=args.min_goal_distance_percentile,
                min_bbox_coverage_ratio=args.min_bbox_coverage_ratio,
                goal_resample_distance_m=args.goal_resample_distance_m,
                edge_cost_noise_factor=args.edge_cost_noise_factor,
                quadrant_goals=args.quadrant_goals))
        else:
            paths.append(dataset.generate_random_path(
                generator, args.path_length_m, args.turn_temperature))

        # Update visit counts
        if args.visited_penalty > 0 and args.goal_directed:
            for pano_id in paths[-1]:
                idx = dataset._panorama_metadata[
                    dataset._panorama_metadata['pano_id'] == pano_id].index[0]
                visit_counts[idx] += 1

        if not args.skip_plots:
            plot_dir = args.out.with_suffix('')
            plot_dir.mkdir(exist_ok=True, parents=True)
            fig, ax = dataset.visualize(path=paths[-1])
            plt.savefig(plot_dir / f"{i:06d}.png")
            plt.close(fig)

    # Restore original edge costs
    if base_edge_costs is not None:
        dataset._cached_edge_costs = base_edge_costs

    out = {
        "paths": paths,
        "dataset_hash": hash(dataset),
        "dataset_path": str(args.dataset_path),
        "args": {
            "path_length_m": args.path_length_m,
            "seed": args.seed,
            "turn_temperature": args.turn_temperature,
            "panorama_neighbor_radius_deg": args.panorama_neighbor_radius_deg,
            "factor": args.factor,
            "goal_directed": args.goal_directed,
            "goal_reached_threshold_m": args.goal_reached_threshold_m,
            "min_goal_distance_percentile": args.min_goal_distance_percentile,
            "min_bbox_coverage_ratio": args.min_bbox_coverage_ratio,
            "goal_resample_distance_m": args.goal_resample_distance_m,
            "edge_cost_noise_factor": args.edge_cost_noise_factor,
            "quadrant_goals": args.quadrant_goals,
            "visited_penalty": args.visited_penalty,
        },
    }
    if visit_counts is not None:
        unique_nodes = sum(1 for c in visit_counts if c > 0)
        print(f"Total unique nodes visited: {unique_nodes}")

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
