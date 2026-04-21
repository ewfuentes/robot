"""Generate evaluation paths from sequential trajectory datasets.

Creates paths of a fixed distance by sliding a window along the trajectory
at uniform intervals. Half the paths go forward, half go backward.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:create_eval_paths_from_panorama_trajectory -- \
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/Framingham \
        --target_distance_m 3000 \
        --num_paths 1000 \
        --out /data/overhead_matching/evaluation/paths/mappilary_v2/Framingham.json
"""

import argparse
import csv
import json
from pathlib import Path

from common.math.haversine import find_d_on_unit_circle

EARTH_RADIUS_M = 6378137.0


def load_trajectory(dataset_path: Path) -> tuple[list[str], list[float]]:
    """Load pano IDs and cumulative distances (meters) along the trajectory."""
    mapping = dataset_path / "pano_id_mapping.csv"
    with open(mapping) as f:
        rows = list(csv.DictReader(f))

    pano_ids = [r["pano_id"] for r in rows]
    latlons = [(float(r["lat"]), float(r["lon"])) for r in rows]

    cum_dist = [0.0]
    for i in range(1, len(pano_ids)):
        step_m = EARTH_RADIUS_M * find_d_on_unit_circle(latlons[i - 1], latlons[i])
        cum_dist.append(cum_dist[-1] + step_m)

    return pano_ids, cum_dist


def find_index_at_distance(cum_dist: list[float], start_idx: int, target_m: float, forward: bool) -> int:
    """Find the trajectory index that is ~target_m away from start_idx."""
    if forward:
        target = cum_dist[start_idx] + target_m
        idx = start_idx
        while idx < len(cum_dist) - 1 and cum_dist[idx] < target:
            idx += 1
        return idx
    else:
        target = cum_dist[start_idx] - target_m
        idx = start_idx
        while idx > 0 and cum_dist[idx] > target:
            idx -= 1
        return idx


def generate_paths(
    pano_ids: list[str],
    cum_dist: list[float],
    target_distance_m: float,
    num_paths: int,
) -> list[list[str]]:
    """Generate paths by uniformly spacing start points along the trajectory.

    Half go forward, half go backward. Start points are uniformly spaced
    within the valid range for each direction.
    """
    n = len(pano_ids)
    total_dist = cum_dist[-1]
    num_forward = num_paths // 2
    num_backward = num_paths - num_forward

    if target_distance_m > total_dist:
        raise ValueError(
            f"Target distance {target_distance_m:.0f}m exceeds trajectory length {total_dist:.0f}m"
        )

    paths = []

    # Forward paths: start points spaced uniformly from 0 to (total - target)
    max_forward_start = total_dist - target_distance_m
    for i in range(num_forward):
        start_dist = max_forward_start * i / max(num_forward - 1, 1)
        # Find the trajectory index closest to this distance
        start_idx = min(range(n), key=lambda j: abs(cum_dist[j] - start_dist))
        end_idx = find_index_at_distance(cum_dist, start_idx, target_distance_m, forward=True)
        paths.append(pano_ids[start_idx : end_idx + 1])

    # Backward paths: start points spaced uniformly from target to total
    for i in range(num_backward):
        start_dist = target_distance_m + (total_dist - target_distance_m) * i / max(num_backward - 1, 1)
        start_idx = min(range(n), key=lambda j: abs(cum_dist[j] - start_dist))
        end_idx = find_index_at_distance(cum_dist, start_idx, target_distance_m, forward=False)
        paths.append(list(reversed(pano_ids[end_idx : start_idx + 1])))

    return paths


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-length evaluation paths from trajectory datasets")
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--target_distance_m", type=float, required=True, help="Path length in meters")
    parser.add_argument("--num_paths", type=int, required=True, help="Total paths (half forward, half backward)")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    pano_ids, cum_dist = load_trajectory(args.dataset_path)
    total_dist = cum_dist[-1]
    print(f"Trajectory: {len(pano_ids)} panos, {total_dist:.0f}m")

    paths = generate_paths(pano_ids, cum_dist, args.target_distance_m, args.num_paths)

    # Stats
    lengths = [len(p) for p in paths]
    num_fwd = args.num_paths // 2
    fwd_lengths = lengths[:num_fwd]
    bwd_lengths = lengths[num_fwd:]
    print(f"Forward:  {num_fwd} paths, {min(fwd_lengths)}-{max(fwd_lengths)} panos")
    print(f"Backward: {len(bwd_lengths)} paths, {min(bwd_lengths)}-{max(bwd_lengths)} panos")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "paths": paths,
        "dataset_path": str(args.dataset_path),
        "dataset_hash": "new",
        "args": {
            "target_distance_m": args.target_distance_m,
            "num_paths": args.num_paths,
            "num_forward": num_fwd,
            "num_backward": args.num_paths - num_fwd,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
