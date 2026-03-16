"""Batch runner for ollama_osm_extraction over labeled panoramas.

Iterates over panorama IDs from a labels JSON file, running the Ollama
extraction for each one and saving individual JSON results. Skips panos
that already have output files.

Requires the OSM tag server to be running separately::

    bazel run //experimental/overhead_matching/swag/scripts:osm_tag_server -- \
      --feather /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather

Usage::

    bazel run //experimental/overhead_matching/swag/scripts:batch_ollama_extraction -- \
      --city Chicago \
      --output-dir /tmp/ollama_extractions
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import ollama as ollama_sdk

from common.ollama.pyollama import Ollama
from experimental.overhead_matching.swag.scripts.ollama_osm_extraction import (
    DEFAULT_PINHOLE_BASE,
    DEFAULT_TAG_SERVER_URL,
    resolve_pano_images,
    run_extraction,
)

DEFAULT_LABELS = "/home/erick/scratch/overhead_matching/landmark_tagger.labels.json"


def main():
    parser = argparse.ArgumentParser(
        description="Batch Ollama OSM extraction over labeled panoramas"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(DEFAULT_LABELS),
        help="Path to landmark_tagger labels JSON",
    )
    parser.add_argument("--city", type=str, required=True, help="VIGOR city name")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for output JSONs"
    )
    parser.add_argument(
        "--pinhole-base",
        type=Path,
        default=Path(DEFAULT_PINHOLE_BASE),
        help=f"Base directory for pinhole images (default: {DEFAULT_PINHOLE_BASE})",
    )
    parser.add_argument(
        "--tag-server-url",
        type=str,
        default=DEFAULT_TAG_SERVER_URL,
        help=f"URL of the OSM tag server (default: {DEFAULT_TAG_SERVER_URL})",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3.5:35b", help="Ollama model tag"
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=None,
        help="URL to an existing Ollama server (skip managed startup)",
    )
    parser.add_argument(
        "--max-tool-rounds", type=int, default=15, help="Max tool-calling iterations"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of panoramas to process (for testing)",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=None,
        help="Directory with example conversation JSONs for in-context learning",
    )
    args = parser.parse_args()

    # Load labels
    with open(args.labels) as f:
        labels_data = json.load(f)
    pano_ids = sorted(
        set(labels_data.get("panorama_labels", {}).keys())
        | set(labels_data.get("relevant_landmarks", {}).keys())
    )
    print(f"Loaded {len(pano_ids)} labeled panorama IDs", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to those not yet processed
    remaining = [
        pid for pid in pano_ids if not (args.output_dir / f"{pid}.json").exists()
    ]
    if args.limit:
        remaining = remaining[: args.limit]
    print(
        f"{len(remaining)} remaining (of {len(pano_ids)} total)", file=sys.stderr
    )

    def process_all(client: ollama_sdk.Client):
        success = 0
        skipped = 0
        failed = 0
        for i, pano_id in enumerate(remaining):
            output_path = args.output_dir / f"{pano_id}.json"
            print(
                f"\n[{i + 1}/{len(remaining)}] {pano_id}",
                file=sys.stderr,
            )
            try:
                image_paths = resolve_pano_images(
                    args.pinhole_base, args.city, pano_id
                )
            except FileNotFoundError as e:
                print(f"  SKIP (no images): {e}", file=sys.stderr)
                skipped += 1
                continue

            try:
                result = run_extraction(
                    client,
                    args.model,
                    image_paths,
                    args.tag_server_url,
                    args.max_tool_rounds,
                    examples_dir=args.examples_dir,
                )
                output_path.write_text(result.model_dump_json(indent=2))
                num_lm = len(result.landmarks)
                print(f"  OK: {num_lm} landmarks -> {output_path}", file=sys.stderr)
                success += 1
            except Exception:
                print(f"  FAIL:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                failed += 1

        print(
            f"\nDone: {success} success, {skipped} skipped, {failed} failed",
            file=sys.stderr,
        )

    if args.ollama_base_url:
        print(
            f"Using existing Ollama server at {args.ollama_base_url}",
            file=sys.stderr,
        )
        client = ollama_sdk.Client(host=args.ollama_base_url)
        process_all(client)
    else:
        print(
            f"Starting managed Ollama server for model {args.model}...",
            file=sys.stderr,
        )
        with Ollama(args.model) as server:
            client = ollama_sdk.Client(host=server.base_url())
            process_all(client)


if __name__ == "__main__":
    main()
