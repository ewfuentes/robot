import argparse
import hashlib
import json
import logging

import numpy as np
import pandas as pd
from pathlib import Path

from common.math.haversine import find_d_on_unit_circle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6378137.0


def load_trajectory(dataset_path: Path) -> tuple[pd.DataFrame, str]:
    """Load extraction_log.csv and return sorted trajectory with pano ID column name.

    Returns:
        (df sorted by frame_idx, name of the pano ID column)
    """
    csv_path = dataset_path / "extraction_log.csv"
    df = pd.read_csv(csv_path)

    # Detect pano ID column
    if "mapillary_id" in df.columns:
        pano_col = "mapillary_id"
    elif "uuid" in df.columns:
        pano_col = "uuid"
    elif "new_pano_id" in df.columns:
        pano_col = "new_pano_id"
    else:
        raise ValueError(
            f"Cannot find pano ID column. Available columns: {list(df.columns)}"
        )

    # Extract frame_idx from old_filename if not present (Boston format)
    if "frame_idx" not in df.columns and "old_filename" in df.columns:
        df["frame_idx"] = df["old_filename"].str.extract(r"frame_(\d+)")[0].astype(int)

    df = df.sort_values("frame_idx").reset_index(drop=True)

    # Normalize longitude column to 'lon'
    if "lng" in df.columns and "lon" not in df.columns:
        df = df.rename(columns={"lng": "lon"})

    return df, pano_col


def validate_trajectory(df: pd.DataFrame) -> np.ndarray:
    """Validate frame indices are sequential and GPS spacing is reasonable.

    Returns:
        Cumulative distances along trajectory in meters (length N).
    """
    frame_idxs = df["frame_idx"].values
    assert np.all(np.diff(frame_idxs) > 0), (
        f"frame_idx values are not strictly increasing. "
        f"First: {frame_idxs[0]}, Last: {frame_idxs[-1]}, Count: {len(frame_idxs)}"
    )

    # Compute consecutive haversine distances
    positions = df[["lat", "lon"]].values
    p1 = positions[:-1]
    p2 = positions[1:]
    step_distances = EARTH_RADIUS_M * find_d_on_unit_circle(p1, p2)

    max_step = step_distances.max()
    assert max_step < 150.0, (
        f"Max consecutive distance {max_step:.1f}m exceeds 150m limit. "
        f"Trajectory may have gaps."
    )

    cumulative = np.zeros(len(df))
    cumulative[1:] = np.cumsum(step_distances)

    total_dist = cumulative[-1]
    avg_spacing = total_dist / (len(df) - 1)
    logger.info(
        f"Trajectory: {len(df)} panoramas, "
        f"{total_dist:.0f}m total, "
        f"{avg_spacing:.1f}m avg spacing"
    )

    return cumulative


def compute_dataset_hash(dataset_path: Path) -> str:
    csv_path = dataset_path / "extraction_log.csv"
    h = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16]
    return h


def create_paths(
    df: pd.DataFrame,
    pano_col: str,
    cumulative_distances: np.ndarray,
    num_paths: int,
    max_trim_frac: float,
) -> list[list[str]]:
    """Create paths by trimming different amounts from the start."""
    n = len(df)
    total_dist = cumulative_distances[-1]
    pano_ids = df[pano_col].astype(str).values

    paths = []
    for i in range(num_paths):
        if num_paths == 1:
            trim_frac = 0.0
        else:
            trim_frac = i * max_trim_frac / (num_paths - 1)

        trim_dist = trim_frac * total_dist
        # Find first index where cumulative distance >= trim_dist
        start_idx = int(np.searchsorted(cumulative_distances, trim_dist))
        # Clamp to ensure at least 2 panoramas in the path
        start_idx = min(start_idx, n - 2)

        paths.append(pano_ids[start_idx:].tolist())

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation paths for sequential datasets by trimming from start"
    )
    parser.add_argument(
        "--dataset-path", type=Path, required=True,
        help="Path to VIGOR dataset directory (must contain extraction_log.csv)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--num-paths", type=int, default=10, help="Number of sub-paths to generate"
    )
    parser.add_argument(
        "--max-trim-frac", type=float, default=None,
        help="Maximum fraction of trajectory to trim from start (e.g., 0.9)",
    )
    parser.add_argument(
        "--max-trim-m", type=float, default=None,
        help="Maximum distance in meters to trim from start",
    )
    args = parser.parse_args()

    # Exactly one trim arg required
    if (args.max_trim_frac is None) == (args.max_trim_m is None):
        parser.error("Exactly one of --max-trim-frac or --max-trim-m must be provided")

    df, pano_col = load_trajectory(args.dataset_path)
    cumulative_distances = validate_trajectory(df)
    total_dist = cumulative_distances[-1]

    # Convert max_trim_m to fraction
    if args.max_trim_m is not None:
        max_trim_frac = args.max_trim_m / total_dist
        logger.info(
            f"--max-trim-m {args.max_trim_m:.0f}m = {max_trim_frac:.4f} fraction "
            f"of {total_dist:.0f}m total"
        )
    else:
        max_trim_frac = args.max_trim_frac

    paths = create_paths(df, pano_col, cumulative_distances, args.num_paths, max_trim_frac)

    logger.info(
        f"Generated {len(paths)} paths: "
        f"longest={len(paths[0])} panos, shortest={len(paths[-1])} panos"
    )

    args.out.parent.mkdir(exist_ok=True, parents=True)
    out = {
        "paths": paths,
        "dataset_path": str(args.dataset_path),
        "sequence_dataset_hash": compute_dataset_hash(args.dataset_path),
        "args": {
            "num_paths": args.num_paths,
            "max_trim_frac": max_trim_frac,
            "max_trim_m": args.max_trim_m,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
