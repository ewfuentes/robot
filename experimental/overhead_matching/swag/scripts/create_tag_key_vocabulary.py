"""Extract tag key vocabulary from Chicago training data.

Collects tag keys from:
1. Chicago v4_202001 OSM landmark feather (applying prune_landmark to each row)
2. Chicago pano_v2 embeddings pickle (primary_tag + additional_tags)

Filters to keys appearing >= 5 times across both sources combined.
Writes one key per line to the output file.
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import common.torch.load_torch_deps
import geopandas as gpd

from experimental.overhead_matching.swag.model.semantic_landmark_utils import prune_landmark


def collect_osm_keys(feather_path: Path) -> Counter:
    """Collect tag keys from OSM landmark feather file using prune_landmark."""
    df = gpd.read_feather(feather_path)
    counts = Counter()

    for _, row in df.iterrows():
        pruned = prune_landmark(row.dropna().to_dict())
        for key, _ in pruned:
            counts[key] += 1

    return counts


def collect_pano_keys(pickle_path: Path) -> Counter:
    """Collect tag keys from pano_v2 embeddings pickle."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    counts = Counter()
    for pano_id, pano_data in data.get("panoramas", {}).items():
        for landmark in pano_data.get("landmarks", []):
            primary_tag = landmark.get("primary_tag", {})
            if primary_tag and "key" in primary_tag:
                counts[primary_tag["key"]] += 1

            for tag in landmark.get("additional_tags", []):
                if "key" in tag:
                    counts[tag["key"]] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Create tag key vocabulary")
    parser.add_argument(
        "--dataset_base",
        type=Path,
        default=Path("/data/overhead_matching/datasets/VIGOR"),
    )
    parser.add_argument(
        "--embedding_base",
        type=Path,
        default=Path("/data/overhead_matching/datasets/semantic_landmark_embeddings"),
    )
    parser.add_argument(
        "--city",
        default="Chicago",
    )
    parser.add_argument(
        "--landmark_version",
        default="v4_202001",
    )
    parser.add_argument(
        "--pano_version",
        default="pano_v2",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.embedding_base / "tag_key_vocabulary.txt"

    # Collect from OSM feather
    feather_path = (
        args.dataset_base / args.city / "landmarks" / f"{args.landmark_version}.feather"
    )
    print(f"Loading OSM landmarks from {feather_path}")
    osm_counts = collect_osm_keys(feather_path)
    print(f"  Found {len(osm_counts)} unique keys, {sum(osm_counts.values())} total occurrences")

    # Collect from pano pickle
    pickle_path = (
        args.embedding_base / args.pano_version / args.city / "embeddings" / "embeddings.pkl"
    )
    print(f"Loading pano landmarks from {pickle_path}")
    pano_counts = collect_pano_keys(pickle_path)
    print(f"  Found {len(pano_counts)} unique keys, {sum(pano_counts.values())} total occurrences")

    # Combine counts
    combined = osm_counts + pano_counts
    print(f"\nCombined: {len(combined)} unique keys")

    # Filter by minimum count
    filtered = {k: v for k, v in combined.items() if v >= args.min_count}
    print(f"After filtering (>= {args.min_count}): {len(filtered)} keys")

    # Sort alphabetically for determinism
    sorted_keys = sorted(filtered.keys())

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for key in sorted_keys:
            f.write(key + "\n")

    print(f"\nWrote {len(sorted_keys)} keys to {args.output}")

    # Print top keys by count
    print("\nTop 20 keys by count:")
    for key, count in sorted(combined.items(), key=lambda x: -x[1])[:20]:
        marker = " *" if key in filtered else ""
        print(f"  {key}: {count}{marker}")


if __name__ == "__main__":
    main()
