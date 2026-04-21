#!/usr/bin/env python3
"""Pre-compute text embeddings for all unique tag values.

Two modes:
  1. --data_dir: Scan JSONL batch response files (original mode)
  2. --feather_dirs + --pano_v2_base: Scan .feather landmark files and pano_v2 pickles

Use --base_embeddings to incrementally expand an existing embeddings file: all
base entries are carried forward into the output, and only values not already in
the base are sent to the embedding API. The output is a superset of the base.

Usage:
    # From JSONL responses (original):
    bazel run ...precompute_value_embeddings -- \
        --data_dir /data/.../landmark_correspondence/chicago_seattle_neg_v3_full \
        --output /data/.../text_embeddings.pkl

    # From feather + pano_v2 (expand to new cities):
    bazel run ...precompute_value_embeddings -- \
        --feather_dirs /data/.../VIGOR/mapillary/MiamiBeach /data/.../VIGOR/NewYork \
        --pano_v2_base /data/.../semantic_landmark_embeddings/mapillary \
        --base_embeddings /data/.../eval_text_embeddings.pkl \
        --output /data/.../eval_text_embeddings_expanded.pkl

Prerequisites:
    export GOOGLE_CLOUD_PROJECT=your-project-id
    gcloud auth application-default login
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    parse_prompt_landmarks,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    TAG_KEY_TO_IDX,
)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    extract_panorama_data_across_cities,
)
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import (
    extract_tags_from_pano_data,
)


def collect_unique_text_values(data_dir: Path) -> dict[str, int]:
    """Scan JSONL files and collect unique tag values (any key in TAG_KEY_TO_IDX)
    with counts.

    Args:
        data_dir: Root directory containing {Chicago,Seattle}/responses/ subdirs

    Returns:
        Counter mapping value strings to their occurrence counts
    """
    value_counts: Counter = Counter()
    jsonl_files = list(data_dir.rglob("predictions.jsonl"))
    if not jsonl_files:
        jsonl_files = list(data_dir.rglob("*.jsonl"))

    print(f"Scanning {len(jsonl_files)} JSONL file(s)...")

    total_lines = 0
    skipped = 0
    for jsonl_path in tqdm(jsonl_files, desc="Scanning JSONL"):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    data = json.loads(line)
                    prompt_text = data["request"]["contents"][0]["parts"][0]["text"]
                    set1, set2 = parse_prompt_landmarks(prompt_text)
                    for landmarks in [set1, set2]:
                        for tags in landmarks:
                            for k, v in tags.items():
                                if k in TAG_KEY_TO_IDX:
                                    value_counts[v] += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    skipped += 1
                    continue

    print(f"Scanned {total_lines} lines from {len(jsonl_files)} files")
    if skipped:
        print(f"WARNING: Skipped {skipped} unparseable lines")
    return value_counts


def collect_text_values_from_feather(feather_dirs: list[Path]) -> Counter:
    """Collect text-type tag values from OSM landmark .feather files.

    Scans the pruned_props column (frozenset of (key, value) tuples).
    """
    value_counts: Counter = Counter()
    for city_dir in feather_dirs:
        landmarks_dir = city_dir / "landmarks"
        if not landmarks_dir.exists():
            print(f"  Warning: No landmarks/ dir in {city_dir}")
            continue
        for feather_path in landmarks_dir.glob("*.feather"):
            df = pd.read_feather(feather_path)
            if "pruned_props" in df.columns:
                raise ValueError(
                    f"{feather_path} already has pruned_props column — "
                    f"this should never be precomputed in the feather file"
                )
            from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
                prune_landmark,
            )
            df["pruned_props"] = df.apply(
                lambda row: prune_landmark(row.dropna().to_dict()), axis=1)
            n_lm = 0
            for props in df["pruned_props"]:
                if not props:
                    continue
                n_lm += 1
                for k, v in props:
                    if k in TAG_KEY_TO_IDX:
                        value_counts[v] += 1
            print(f"  {feather_path.name}: {n_lm} landmarks, "
                  f"{len(value_counts)} unique values so far")
    return value_counts


def collect_text_values_from_pano_v2(pano_v2_base: Path) -> Counter:
    """Collect tag values from pano_v2 landmark pickles (any key in TAG_KEY_TO_IDX)."""
    value_counts: Counter = Counter()
    pano_tags = extract_panorama_data_across_cities(
        pano_v2_base, extract_tags_from_pano_data,
    )
    for pano_id, landmarks in pano_tags.items():
        for lm in landmarks:
            for k, v in lm["tags"]:
                if k in TAG_KEY_TO_IDX:
                    value_counts[v] += 1
    print(f"  {len(pano_tags)} panoramas, {len(value_counts)} unique values")
    return value_counts


def embed_texts_vertex(
    texts: list[str],
    model: str = "text-embedding-005",
    task_type: str = "SEMANTIC_SIMILARITY",
    batch_size: int = 250,
    output_dimensionality: int = 768,
) -> np.ndarray:
    """Embed texts using Vertex AI.

    Returns: numpy array of shape (len(texts), output_dimensionality)
    """
    from google import genai
    from google.genai.types import EmbedContentConfig

    client = genai.Client(vertexai=True)
    config = EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )

    all_embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", total=num_batches):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config,
                )
                all_embeddings.extend([emb.values for emb in response.embeddings])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"\nRetry {attempt + 1}/3 after {wait}s: {e}")
                time.sleep(wait)

    return np.array(all_embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings for tag values")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="Root directory with JSONL responses (original mode)",
    )
    parser.add_argument(
        "--feather_dirs", type=Path, nargs="+", default=None,
        help="VIGOR city dirs containing landmarks/*.feather files",
    )
    parser.add_argument(
        "--pano_v2_base", type=Path, default=None,
        help="Base path for pano_v2 embeddings (contains city subdirs)",
    )
    parser.add_argument(
        "--base_embeddings", type=Path, default=None,
        help="Existing embeddings pickle to merge into the output. All base entries "
             "are kept, and only values not in the base are sent to the embedding API.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--model", type=str, default="text-embedding-005",
        help="Embedding model (default: text-embedding-005)",
    )
    parser.add_argument(
        "--output_dimensionality", type=int, default=768,
        help="Embedding dimension (default: 768)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=250,
        help="Batch size for API calls (default: 250, max: 250)",
    )
    parser.add_argument(
        "--stats_only", action="store_true",
        help="Only print statistics, don't embed",
    )
    args = parser.parse_args()

    if not args.data_dir and not args.feather_dirs and not args.pano_v2_base:
        parser.error("Provide --data_dir (JSONL mode) or --feather_dirs/--pano_v2_base (feather mode)")

    # Collect unique values from all sources
    value_counts: Counter = Counter()

    if args.data_dir:
        print("Collecting from JSONL responses...")
        value_counts += collect_unique_text_values(args.data_dir)

    if args.feather_dirs:
        print(f"Collecting from {len(args.feather_dirs)} feather directories...")
        value_counts += collect_text_values_from_feather(args.feather_dirs)

    if args.pano_v2_base:
        print(f"Collecting from pano_v2 at {args.pano_v2_base}...")
        value_counts += collect_text_values_from_pano_v2(args.pano_v2_base)

    print(f"\nTotal unique text values: {len(value_counts):,}")
    print(f"Total occurrences: {sum(value_counts.values()):,}")

    # Load base embeddings to determine what's new
    base_embeddings = {}
    if args.base_embeddings:
        print(f"\nLoading base embeddings from {args.base_embeddings}")
        with open(args.base_embeddings, "rb") as f:
            base_embeddings = pickle.load(f)
        print(f"  {len(base_embeddings):,} existing embeddings")

    already_have = set(base_embeddings.keys())
    new_values = sorted(v for v in value_counts if v not in already_have)
    print(f"Already embedded: {len(already_have & set(value_counts)):,}")
    print(f"New values to embed: {len(new_values):,}")

    # Show top new values
    if new_values:
        print("\nTop 20 most common NEW values:")
        new_counts = {v: value_counts[v] for v in new_values}
        for val, count in Counter(new_counts).most_common(20):
            print(f"  {count:6d}  {val[:80]}")

        lengths = [len(v) for v in new_values]
        print(f"\nNew value length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.1f}")

    if args.stats_only:
        return 0

    if not new_values:
        print("\nNo new values to embed.")
        if base_embeddings and args.output != args.base_embeddings:
            # Copy base to output
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "wb") as f:
                pickle.dump(base_embeddings, f)
            print(f"Copied base embeddings to {args.output}")
        return 0

    # Embed new values
    print(f"\nEmbedding {len(new_values):,} new values with {args.model}...")
    embeddings = embed_texts_vertex(
        new_values,
        model=args.model,
        output_dimensionality=args.output_dimensionality,
        batch_size=args.batch_size,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Merge with base
    result = dict(base_embeddings)
    for i, v in enumerate(new_values):
        result[v] = embeddings[i]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    file_size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved to {args.output} ({file_size_mb:.1f} MB)")
    print(f"  {len(result):,} total embeddings ({len(base_embeddings):,} base + "
          f"{len(new_values):,} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
