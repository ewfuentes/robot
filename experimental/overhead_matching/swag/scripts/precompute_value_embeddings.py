#!/usr/bin/env python3
"""Pre-compute text embeddings for all unique tag values in the JSONL data.

Scans all JSONL response files, collects unique value strings for text-type keys,
embeds them via Vertex AI text-embedding-005, and saves as pickle:
  {value_string: numpy_array (768-dim)}

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:precompute_value_embeddings -- \
        --data_dir /data/overhead_matching/datasets/landmark_correspondence/neg_v3_full \
        --output /data/overhead_matching/datasets/landmark_correspondence/text_embeddings.pkl

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
from tqdm import tqdm

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    parse_prompt_landmarks,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    key_type,
)


def collect_unique_text_values(data_dir: Path) -> dict[str, int]:
    """Scan JSONL files and collect all unique text-type tag values with counts.

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
                                if key_type(k) == "text":
                                    value_counts[v] += 1
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    print(f"Scanned {total_lines} lines from {len(jsonl_files)} files")
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
        "--data_dir", type=Path, required=True,
        help="Root directory with {Chicago,Seattle}/responses/ subdirs",
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

    # Collect unique values
    value_counts = collect_unique_text_values(args.data_dir)
    print(f"\nUnique text values: {len(value_counts):,}")
    print(f"Total occurrences: {sum(value_counts.values()):,}")

    # Show top values
    print("\nTop 20 most common values:")
    for val, count in value_counts.most_common(20):
        print(f"  {count:6d}  {val[:80]}")

    # Show value length stats
    lengths = [len(v) for v in value_counts]
    print(f"\nValue length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}")

    if args.stats_only:
        return 0

    # Embed
    values = sorted(value_counts.keys())  # Deterministic ordering
    print(f"\nEmbedding {len(values):,} unique values with {args.model}...")

    embeddings = embed_texts_vertex(
        values,
        model=args.model,
        output_dimensionality=args.output_dimensionality,
        batch_size=args.batch_size,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Save as {value_string: numpy_array}
    result = {v: embeddings[i] for i, v in enumerate(values)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(result, f)

    file_size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nSaved to {args.output} ({file_size_mb:.1f} MB)")
    print(f"  {len(result):,} embeddings, {args.output_dimensionality}-dim each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
