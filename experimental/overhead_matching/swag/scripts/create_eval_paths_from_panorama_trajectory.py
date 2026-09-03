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
import hashlib
import json
from pathlib import Path

from common.gps.web_mercator import EARTH_RADIUS_M
from common.math.haversine import find_d_on_unit_circle


def mapping_sha256(dataset_path: Path) -> str:
    """Return the stable identity of the trajectory-to-panorama mapping."""
    return hashlib.sha256(
        (dataset_path / "pano_id_mapping.csv").read_bytes()).hexdigest()


def load_trajectory(dataset_path: Path) -> tuple[list[str], list[float]]:
    """Load pano IDs and cumulative distances (meters) along the trajectory.

    Raises ValueError if the CSV has fewer than two rows — no trajectory
    can be formed from a single point. The row-count check runs before
    column access so a degenerate CSV with missing columns still produces
    a clear error instead of a KeyError.
    """
    mapping = dataset_path / "pano_id_mapping.csv"
    with open(mapping) as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 2:
        raise ValueError(
            f"{mapping} has {len(rows)} rows; at least 2 are required to form a trajectory"
        )

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


def generate_full_trajectory_paths(pano_ids: list[str]) -> list[list[str]]:
    """Return the complete recorded trajectory in both directions."""
    return [list(pano_ids), list(reversed(pano_ids))]


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-length evaluation paths from trajectory datasets")
    parser.add_argument("--dataset_path", type=Path, required=True)
    distance = parser.add_mutually_exclusive_group(required=True)
    distance.add_argument("--target_distance_m", type=float, help="Path length in meters")
    distance.add_argument(
        "--full_trajectory", action="store_true",
        help="Emit the complete recorded leg once forward and once backward",
    )
    parser.add_argument(
        "--num_paths", type=int,
        help="Total fixed-length paths (half forward, half backward)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    pano_ids, cum_dist = load_trajectory(args.dataset_path)
    total_dist = cum_dist[-1]
    print(f"Trajectory: {len(pano_ids)} panos, {total_dist:.0f}m")

    if args.full_trajectory:
        if args.num_paths is not None:
            parser.error("--num_paths cannot be used with --full_trajectory")
        paths = generate_full_trajectory_paths(pano_ids)
        num_fwd = 1
    else:
        if args.num_paths is None:
            parser.error("--num_paths is required with --target_distance_m")
        paths = generate_paths(
            pano_ids, cum_dist, args.target_distance_m, args.num_paths)
        num_fwd = args.num_paths // 2

    # Stats
    lengths = [len(p) for p in paths]
    fwd_lengths = lengths[:num_fwd]
    bwd_lengths = lengths[num_fwd:]
    print(f"Forward:  {num_fwd} paths, {min(fwd_lengths)}-{max(fwd_lengths)} panos")
    print(f"Backward: {len(bwd_lengths)} paths, {min(bwd_lengths)}-{max(bwd_lengths)} panos")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "paths": paths,
        "dataset_path": str(args.dataset_path),
        "dataset_hash": mapping_sha256(args.dataset_path),
        "args": {
            "target_distance_m": args.target_distance_m,
            "full_trajectory": args.full_trajectory,
            "num_paths": len(paths),
            "num_forward": num_fwd,
            "num_backward": len(paths) - num_fwd,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
