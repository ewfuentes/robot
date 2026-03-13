"""Export tag-weight similarity matrices for any VIGOR city.

Loads learned per-tag-key weights (from tag_weight_optimization notebook) and
applies them to a city's parquet match tables to produce a (num_panos, num_sats)
similarity matrix. Reports recall@k and MRR metrics.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:export_tag_weight_similarity -- \
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/Framingham \
        --weights_path ~/scratch/similarities/osm_tag_lbfgs/weights.pt \
        --train_city_name chicago
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 — must precede torch import
import torch

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation import tag_weight_similarity as tws


def auto_detect_landmark_version(dataset_path: Path) -> str:
    """Auto-detect landmark version from single .feather file in landmarks/ dir."""
    landmarks_dir = dataset_path / "landmarks"
    if not landmarks_dir.exists():
        raise FileNotFoundError(f"No landmarks/ directory found at {landmarks_dir}")
    feather_files = list(landmarks_dir.glob("*.feather"))
    if len(feather_files) == 0:
        raise FileNotFoundError(f"No .feather files in {landmarks_dir}")
    if len(feather_files) > 1:
        names = [f.name for f in feather_files]
        raise ValueError(
            f"Multiple .feather files in {landmarks_dir}, specify --landmark_version. "
            f"Found: {names}"
        )
    return feather_files[0].stem


def main():
    parser = argparse.ArgumentParser(
        description="Export tag-weight similarity matrix for a VIGOR city"
    )
    parser.add_argument("--dataset_path", type=Path, required=True,
                        help="Path to VIGOR city dir, e.g. /data/.../VIGOR/Chicago")
    parser.add_argument("--weights_path", type=Path, required=True,
                        help="Path to weights.pt file with learned theta and tag_keys")
    parser.add_argument("--train_city_name", type=str, required=True,
                        help="Name of the city weights were trained on (used in output filename)")
    parser.add_argument("--landmark_tables_dir", type=Path, default=None,
                        help="Dir containing {city}_pano_osm_matches.parquet and "
                             "{city}_sat_osm_table.parquet. Default: {dataset_path}/landmark_tables/")
    parser.add_argument("--output_path", type=Path, default=None,
                        help="Output .pt file path. Default: "
                             "{dataset_path}/similarity_matrices/learned_tag_weight_{train_city_name}.pt")
    parser.add_argument("--city_name", type=str, default=None,
                        help="City name for table filenames (default: inferred from dataset_path, lowercased)")
    parser.add_argument("--landmark_version", type=str, default=None,
                        help="Landmark version string. Default: auto-detect from landmarks/ dir")
    parser.add_argument("--ks", type=str, default="1,5,10",
                        help="Comma-separated top-k values (default: 1,5,10)")
    parser.add_argument("--inflation_factor", type=float, default=1.0,
                        help="Satellite patch inflation factor for landmark correspondence (default: 1.0)")
    parser.add_argument("--dedup_sat_keys", action="store_true",
                        help="Deduplicate tag keys per (pano, sat) pair: each tag_key contributes "
                             "at most once regardless of how many OSM landmarks on the satellite match it")
    parser.add_argument("--no_count", action="store_true",
                        help="Ignore pano landmark counts: each (pano, osm, key) match contributes "
                             "theta[key] regardless of how many pano landmarks matched")
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    weights_path = args.weights_path.expanduser().resolve()
    city_name = args.city_name or dataset_path.name.lower()
    ks = [int(k) for k in args.ks.split(",")]

    # Resolve landmark tables directory
    if args.landmark_tables_dir is not None:
        tables_dir = args.landmark_tables_dir.expanduser().resolve()
    else:
        tables_dir = dataset_path / "landmark_tables"

    matches_path = str(tables_dir / f"{city_name}_pano_osm_matches.parquet")
    sat_osm_path = str(tables_dir / f"{city_name}_sat_osm_table.parquet")

    # Resolve output path
    if args.output_path is not None:
        output_path = args.output_path.expanduser().resolve()
    else:
        output_path = dataset_path / "similarity_matrices" / f"learned_tag_weight_{args.train_city_name}.pt"

    # Auto-detect landmark version if not specified
    landmark_version = args.landmark_version or auto_detect_landmark_version(dataset_path)

    # 1. Load weights
    print(f"Loading weights from {weights_path}")
    weights_data = torch.load(weights_path, weights_only=False)
    theta = weights_data["theta"]
    tag_keys = weights_data["tag_keys"]
    print(f"  {len(tag_keys)} tag keys, theta shape: {theta.shape}")

    # 2. Load VigorDataset (no images, yes landmarks)
    print(f"Loading dataset from {dataset_path} (landmark_version={landmark_version})")
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
        landmark_correspondence_inflation_factor=args.inflation_factor,
    )
    dataset = vd.VigorDataset(dataset_path, config)
    print(f"  {len(dataset._panorama_metadata)} panos, "
          f"{len(dataset._satellite_metadata)} sats, "
          f"{len(dataset._landmark_metadata)} landmarks")

    # 3. Precompute match data
    print(f"Loading match tables from {tables_dir}")
    match_data = tws.precompute_match_data(
        matches_path, sat_osm_path, dataset, tag_keys,
    )

    # 4. Build similarity matrix
    if args.dedup_sat_keys:
        print("Deduplicating: each tag_key counts at most once per (pano, sat) pair")
    if args.no_count:
        print("Ignoring pano landmark counts")
    similarity = tws.build_similarity_matrix(
        match_data, theta, dedup_sat_keys=args.dedup_sat_keys, no_count=args.no_count)
    print(f"Similarity matrix shape: {similarity.shape}")

    # 5. Compute and print metrics
    metrics = tws.compute_top_k_metrics(similarity, dataset, ks=ks)
    print(f"\nMetrics for {city_name} (weights trained on {args.train_city_name}):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 6. Save similarity matrix
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(similarity, output_path)
    print(f"\nSaved similarity matrix {similarity.shape} to {output_path}")


if __name__ == "__main__":
    main()
