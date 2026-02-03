import argparse
import logging
import matplotlib.pyplot as plt
import json
import tqdm
import common.torch.load_torch_deps
import torch
import numpy as np
import pandas as pd
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from pathlib import Path

from common.math.haversine import find_d_on_unit_circle
from planning import a_star_python

# Configure logger for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6378137.0


def compute_city_bounding_box(panorama_metadata: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute bounding box of all panoramas in the dataset.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon data.

    Returns:
        (min_lat, max_lat, min_lon, max_lon)
    """
    min_lat = panorama_metadata['lat'].min()
    max_lat = panorama_metadata['lat'].max()
    min_lon = panorama_metadata['lon'].min()
    max_lon = panorama_metadata['lon'].max()
    return (min_lat, max_lat, min_lon, max_lon)


def compute_path_bounding_box(
    panorama_metadata: pd.DataFrame,
    path_indices: list[int]
) -> tuple[float, float, float, float]:
    """Compute bounding box of a path.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon data.
        path_indices: List of panorama indices.

    Returns:
        (min_lat, max_lat, min_lon, max_lon)
    """
    lats = panorama_metadata.loc[path_indices, 'lat']
    lons = panorama_metadata.loc[path_indices, 'lon']
    return (lats.min(), lats.max(), lons.min(), lons.max())


def compute_bbox_coverage_ratio(
    path_bbox: tuple[float, float, float, float],
    city_bbox: tuple[float, float, float, float]
) -> float:
    """Compute the ratio of path bounding box area to city bounding box area.

    Args:
        path_bbox: (min_lat, max_lat, min_lon, max_lon) of path
        city_bbox: (min_lat, max_lat, min_lon, max_lon) of city

    Returns:
        Ratio of path bbox area to city bbox area.
    """
    path_lat_span = path_bbox[1] - path_bbox[0]
    path_lon_span = path_bbox[3] - path_bbox[2]
    city_lat_span = city_bbox[1] - city_bbox[0]
    city_lon_span = city_bbox[3] - city_bbox[2]

    # Avoid division by zero
    if city_lat_span <= 0 or city_lon_span <= 0:
        return 0.0

    path_area = path_lat_span * path_lon_span
    city_area = city_lat_span * city_lon_span
    return path_area / city_area


def get_quadrant(
    lat: float,
    lon: float,
    city_bbox: tuple[float, float, float, float]
) -> int:
    """Determine which quadrant a position is in.

    Args:
        lat: Latitude of position.
        lon: Longitude of position.
        city_bbox: (min_lat, max_lat, min_lon, max_lon) of city.

    Returns:
        Quadrant index: 0=SW, 1=SE, 2=NW, 3=NE
    """
    min_lat, max_lat, min_lon, max_lon = city_bbox
    mid_lat = (min_lat + max_lat) / 2
    mid_lon = (min_lon + max_lon) / 2

    is_north = lat >= mid_lat
    is_east = lon >= mid_lon

    if is_north:
        return 3 if is_east else 2  # NE or NW
    else:
        return 1 if is_east else 0  # SE or SW


def select_distant_goal(
    panorama_metadata: pd.DataFrame,
    current_idx: int,
    generator: torch.Generator,
    percentile: float
) -> int:
    """Select a goal that is at least at the given percentile of maximum distance.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon data.
        current_idx: Current panorama index.
        generator: Random generator for reproducibility.
        percentile: Select goals >= this percentile of max distance from current position.

    Returns:
        Index of selected goal panorama.
    """
    current_pos = panorama_metadata.loc[current_idx, ['lat', 'lon']].values
    all_positions = panorama_metadata[['lat', 'lon']].values

    # Vectorized distance computation using haversine
    distances = EARTH_RADIUS_M * find_d_on_unit_circle(current_pos, all_positions)

    # Find threshold distance at the given percentile
    max_distance = distances.max()
    threshold_distance = percentile * max_distance

    # Get indices of panoramas beyond the threshold
    distant_indices = np.where(distances >= threshold_distance)[0]

    if len(distant_indices) == 0:
        # Fallback: select the farthest panorama
        return int(np.argmax(distances))

    # Randomly select one of the distant panoramas
    rand_idx = torch.randint(0, len(distant_indices), (1,), generator=generator).item()
    return int(distant_indices[rand_idx])


def select_quadrant_goal(
    panorama_metadata: pd.DataFrame,
    current_idx: int,
    generator: torch.Generator,
    city_bbox: tuple[float, float, float, float],
    percentile: float = 0.5,
) -> int:
    """Select a goal in a different quadrant than the current position.

    This encourages paths to go around the perimeter rather than through the center.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon data.
        current_idx: Current panorama index.
        generator: Random generator for reproducibility.
        city_bbox: (min_lat, max_lat, min_lon, max_lon) of city.
        percentile: Also require goal to be at this percentile of max distance.

    Returns:
        Index of selected goal panorama.
    """
    current_pos = panorama_metadata.loc[current_idx, ['lat', 'lon']].values
    current_quadrant = get_quadrant(current_pos[0], current_pos[1], city_bbox)

    all_positions = panorama_metadata[['lat', 'lon']].values

    # Find all panoramas in different quadrants
    different_quadrant_mask = np.array([
        get_quadrant(pos[0], pos[1], city_bbox) != current_quadrant
        for pos in all_positions
    ])

    # Also filter by distance if percentile > 0
    if percentile > 0:
        distances = EARTH_RADIUS_M * find_d_on_unit_circle(current_pos, all_positions)
        max_distance = distances.max()
        threshold_distance = percentile * max_distance
        distance_mask = distances >= threshold_distance
        combined_mask = different_quadrant_mask & distance_mask
    else:
        combined_mask = different_quadrant_mask

    candidate_indices = np.where(combined_mask)[0]

    if len(candidate_indices) == 0:
        # Fallback to regular distant goal selection
        return select_distant_goal(panorama_metadata, current_idx, generator, percentile)

    # Randomly select one of the candidates
    rand_idx = torch.randint(0, len(candidate_indices), (1,), generator=generator).item()
    return int(candidate_indices[rand_idx])


def precompute_edge_costs(panorama_metadata: pd.DataFrame) -> list[list[float]]:
    """Precompute edge costs (haversine distances) for all edges in the panorama graph.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon and neighbor data.

    Returns:
        List of lists where edge_costs[i][j] is the cost from node i to its j-th neighbor.
    """
    all_positions = panorama_metadata[['lat', 'lon']].values
    edge_costs = []
    for idx in range(len(panorama_metadata)):
        neighbors = panorama_metadata.loc[idx, 'neighbor_panorama_idxs']
        if len(neighbors) == 0:
            edge_costs.append([])
            continue
        pos_i = all_positions[idx]
        neighbor_positions = all_positions[neighbors]
        # Vectorized distance computation for all neighbors at once
        costs = EARTH_RADIUS_M * find_d_on_unit_circle(pos_i, neighbor_positions)
        edge_costs.append(costs.tolist())
    return edge_costs


def compute_heuristics_to_goal(
    panorama_metadata: pd.DataFrame,
    goal_idx: int
) -> np.ndarray:
    """Compute heuristic values (haversine distance to goal) for all nodes.

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon data.
        goal_idx: Index of the goal panorama.

    Returns:
        Numpy array of shape (N,) with heuristic values for each node.
    """
    goal_pos = panorama_metadata.loc[goal_idx, ['lat', 'lon']].values
    all_positions = panorama_metadata[['lat', 'lon']].values
    # Vectorized haversine computation
    heuristics = EARTH_RADIUS_M * find_d_on_unit_circle(all_positions, goal_pos)
    return heuristics


def generate_goal_directed_path(
    panorama_metadata: pd.DataFrame,
    generator: torch.Generator,
    max_length_m: float,
    goal_reached_threshold_m: float = 200.0,
    min_goal_distance_percentile: float = 0.75,
    min_bbox_coverage_ratio: float = 0.05,
    max_regeneration_attempts: int = 100,
    max_astar_expansions: int = 50000,
    goal_resample_distance_m: float = 2000.0,
    # Exploration options
    epsilon_greedy: float = 0.0,
    edge_cost_noise_factor: float = 0.0,
    hybrid_random_walk_distance_m: float = 0.0,
    quadrant_goals: bool = False,
    # Optional pre-computed edge costs (for diversity penalty)
    edge_costs_override: list[list[float]] | None = None,
) -> list[str]:
    """Generate a path using goal-directed A* navigation.

    This function generates paths that spread across the city by:
    1. Selecting distant goals (at the given percentile of max distance)
    2. Using A* to navigate to each goal
    3. Sampling new goals after traveling some distance or when goal is reached
    4. Rejecting paths with small bounding boxes

    Args:
        panorama_metadata: DataFrame containing panorama lat/lon and neighbor data.
        generator: Random generator for reproducibility.
        max_length_m: Maximum path length in meters.
        goal_reached_threshold_m: Distance (m) to consider goal reached.
        min_goal_distance_percentile: Select goals at this percentile of max distance.
        min_bbox_coverage_ratio: Reject paths covering less than this fraction of city area.
        max_regeneration_attempts: Maximum retries when starting from disconnected components.
        max_astar_expansions: Maximum A* node expansions before giving up on a goal.
        goal_resample_distance_m: Resample goal after traveling this distance toward current goal.
        epsilon_greedy: Probability (0-1) of taking a random neighbor instead of A* path.
        edge_cost_noise_factor: Add multiplicative noise to edge costs (0=none, 0.5=costs vary 0.5x-1.5x).
        hybrid_random_walk_distance_m: After reaching goal neighborhood, random walk for this distance.
        quadrant_goals: If True, select goals in different quadrants to encourage perimeter exploration.
        edge_costs_override: Optional pre-computed edge costs (for diversity penalty across paths).

    Returns:
        List of pano_ids defining a path through the graph.
    """
    # Pre-compute city bounding box
    city_bbox = compute_city_bounding_box(panorama_metadata)

    # Pre-compute edge costs if not provided
    if edge_costs_override is not None:
        base_edge_costs = edge_costs_override
    else:
        base_edge_costs = precompute_edge_costs(panorama_metadata)

    # Get neighbor adjacency list
    neighbors = [list(row['neighbor_panorama_idxs'])
                 for _, row in panorama_metadata.iterrows()]

    # Helper function for random walk exploration
    def do_random_walk(start_idx: int, target_distance_m: float) -> tuple[list[int], float]:
        """Do a random walk from start_idx for approximately target_distance_m."""
        walk_indices = []
        walk_distance = 0.0
        walk_idx = start_idx
        while walk_distance < target_distance_m:
            walk_neighbors = neighbors[walk_idx]
            if len(walk_neighbors) == 0:
                break
            # Pick random neighbor
            rand_neighbor_i = torch.randint(0, len(walk_neighbors), (1,), generator=generator).item()
            next_walk_idx = walk_neighbors[rand_neighbor_i]
            # Compute distance
            walk_pos = panorama_metadata.loc[walk_idx, ['lat', 'lon']].values
            next_walk_pos = panorama_metadata.loc[next_walk_idx, ['lat', 'lon']].values
            step_dist = EARTH_RADIUS_M * find_d_on_unit_circle(walk_pos, next_walk_pos)
            walk_distance += step_dist
            walk_indices.append(next_walk_idx)
            walk_idx = next_walk_idx
        return walk_indices, walk_distance

    for attempt in range(max_regeneration_attempts):
        path_indices = []
        distance_traveled_m = 0.0
        distance_since_goal_change_m = 0.0
        consecutive_astar_failures = 0
        max_consecutive_failures = 10  # Give up on this start point after N failures

        # Apply edge cost noise if requested (regenerate each attempt for variety)
        if edge_cost_noise_factor > 0:
            edge_costs = []
            for node_costs in base_edge_costs:
                noisy_costs = []
                for cost in node_costs:
                    # Multiplicative noise: cost * uniform(1-factor, 1+factor)
                    noise = 1.0 + edge_cost_noise_factor * (2.0 * torch.rand(1, generator=generator).item() - 1.0)
                    noisy_costs.append(cost * max(0.1, noise))  # Ensure positive
                edge_costs.append(noisy_costs)
        else:
            edge_costs = base_edge_costs

        # Helper to select goal based on quadrant_goals setting
        def select_goal(from_idx: int) -> int:
            if quadrant_goals:
                return select_quadrant_goal(panorama_metadata, from_idx, generator, city_bbox, min_goal_distance_percentile)
            else:
                return select_distant_goal(panorama_metadata, from_idx, generator, min_goal_distance_percentile)

        # Pick random start
        current_idx = torch.randint(0, len(panorama_metadata), (1,), generator=generator).item()
        path_indices.append(current_idx)

        # Select initial distant goal
        goal_idx = select_goal(current_idx)

        while distance_traveled_m < max_length_m:
            # Check if we should resample goal (reached it, or traveled enough toward it)
            current_pos = panorama_metadata.loc[current_idx, ['lat', 'lon']].values
            goal_pos = panorama_metadata.loc[goal_idx, ['lat', 'lon']].values
            distance_to_goal = EARTH_RADIUS_M * find_d_on_unit_circle(current_pos, goal_pos)

            goal_reached = distance_to_goal < goal_reached_threshold_m
            should_resample = goal_reached or distance_since_goal_change_m >= goal_resample_distance_m

            # Hybrid mode: do random walk when reaching goal neighborhood
            if goal_reached and hybrid_random_walk_distance_m > 0:
                walk_indices, walk_distance = do_random_walk(current_idx, hybrid_random_walk_distance_m)
                if walk_indices:
                    path_indices.extend(walk_indices)
                    distance_traveled_m += walk_distance
                    current_idx = walk_indices[-1]
                    if distance_traveled_m >= max_length_m:
                        break

            if should_resample:
                # Pick a new distant goal
                goal_idx = select_goal(current_idx)
                distance_since_goal_change_m = 0.0

            # Compute heuristics for A*
            heuristics = compute_heuristics_to_goal(panorama_metadata, goal_idx)

            # Run A* with expansion limit to avoid getting stuck
            result = a_star_python.find_path(
                neighbors, edge_costs, heuristics, current_idx, goal_idx,
                max_astar_expansions)

            if result is None:
                # No path found or expansion limit hit - pick a new goal and try again
                consecutive_astar_failures += 1
                if consecutive_astar_failures >= max_consecutive_failures:
                    # Graph might be disconnected at this location, start over
                    logger.warning(f"A* failed {consecutive_astar_failures} times from node {current_idx}, trying new start")
                    break
                goal_idx = select_goal(current_idx)
                distance_since_goal_change_m = 0.0
                continue

            # Reset failure counter on success
            consecutive_astar_failures = 0

            # Traverse A* path, accumulating distance
            a_star_path = result.states
            prev_traversed_idx = current_idx  # Track actual previous node for distance
            for i, next_idx in enumerate(a_star_path):
                if next_idx == current_idx and i == 0:
                    # Skip the starting node if it's the same as current
                    continue

                # Epsilon-greedy: with probability epsilon, take random neighbor instead
                if epsilon_greedy > 0 and torch.rand(1, generator=generator).item() < epsilon_greedy:
                    curr_neighbors = neighbors[prev_traversed_idx]
                    if len(curr_neighbors) > 0:
                        rand_i = torch.randint(0, len(curr_neighbors), (1,), generator=generator).item()
                        next_idx = curr_neighbors[rand_i]

                # Always compute distance from previous traversed node
                prev_pos = panorama_metadata.loc[prev_traversed_idx, ['lat', 'lon']].values
                next_pos = panorama_metadata.loc[next_idx, ['lat', 'lon']].values
                step_distance = EARTH_RADIUS_M * find_d_on_unit_circle(prev_pos, next_pos)
                distance_traveled_m += step_distance
                distance_since_goal_change_m += step_distance
                prev_traversed_idx = next_idx

                # Add to path (include all traversed nodes, even revisited ones)
                path_indices.append(next_idx)

                if distance_traveled_m >= max_length_m:
                    break

                # Check if we should resample goal mid-path
                if distance_since_goal_change_m >= goal_resample_distance_m:
                    # Will resample on next iteration
                    break

            # Update current position to end of A* path (or where we stopped)
            current_idx = prev_traversed_idx

        # Only consider this path if we actually reached the target distance
        if distance_traveled_m < max_length_m:
            # Didn't reach target distance (disconnected component), try again
            continue

        # Check path bounding box coverage
        path_bbox = compute_path_bounding_box(panorama_metadata, path_indices)
        coverage = compute_bbox_coverage_ratio(path_bbox, city_bbox)

        if coverage >= min_bbox_coverage_ratio:
            # Convert indices to pano_ids
            return [panorama_metadata.loc[idx, 'pano_id'] for idx in path_indices]

    # If we exhausted all attempts, return the last path anyway
    logger.warning(f"Failed to generate path with bbox coverage >= {min_bbox_coverage_ratio} "
                  f"after {max_regeneration_attempts} attempts. Returning last attempt.")
    return [panorama_metadata.loc[idx, 'pano_id'] for idx in path_indices]


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
        base_edge_costs = precompute_edge_costs(dataset._panorama_metadata)

    for i in tqdm.tqdm(range(args.num_paths)):
        generator = torch.manual_seed(args.seed + i)

        # Apply visited-node penalty if enabled
        edge_costs_for_path = None
        if args.visited_penalty > 0 and args.goal_directed and base_edge_costs is not None:
            penalized_edge_costs = []
            for node_idx, node_costs in enumerate(base_edge_costs):
                penalty = 1.0 + args.visited_penalty * visit_counts[node_idx]
                penalized_costs = [cost * penalty for cost in node_costs]
                penalized_edge_costs.append(penalized_costs)
            edge_costs_for_path = penalized_edge_costs

        if args.goal_directed:
            paths.append(generate_goal_directed_path(
                dataset._panorama_metadata,
                generator,
                args.path_length_m,
                goal_reached_threshold_m=args.goal_reached_threshold_m,
                min_goal_distance_percentile=args.min_goal_distance_percentile,
                min_bbox_coverage_ratio=args.min_bbox_coverage_ratio,
                goal_resample_distance_m=args.goal_resample_distance_m,
                edge_cost_noise_factor=args.edge_cost_noise_factor,
                quadrant_goals=args.quadrant_goals,
                edge_costs_override=edge_costs_for_path))
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
