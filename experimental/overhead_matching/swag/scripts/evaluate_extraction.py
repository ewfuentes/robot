"""Evaluate LLM OSM tag extraction against human labels.

Compares Gemini or Ollama extraction results against human labels from
the landmark tagger, checking:
1. Panorama-level: Does the LLM correctly identify useful/not_useful panoramas?
2. Landmark-level: For panoramas with labeled relevant landmarks, how many
   does the LLM find via keyed substring matching?

Usage::

    bazel run //experimental/overhead_matching/swag/scripts:evaluate_extraction -- \
      --extractions-dir /data/.../sentences/ \
      --feather /data/.../v4_202001.feather
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experimental.overhead_matching.swag.evaluation import (
    proper_noun_matcher_python as pnm,
)
from experimental.overhead_matching.swag.scripts.search_osm_tags import (
    EXCLUDE_KEYS,
    META_COLS,
)


def load_osm_tag_extraction_jsonl(input_dir: Path) -> dict[str, dict]:
    """Load OSM tag extraction predictions from JSONL files."""
    panoramas = {}
    for jsonl_file in input_dir.rglob("*.jsonl"):
        for line in jsonl_file.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pano_key = record["key"]
            try:
                response_text = (
                    record["response"]["candidates"][0]["content"]["parts"][0]["text"]
                )
                parsed = json.loads(response_text)
                panoramas[pano_key] = parsed
            except Exception:
                print(f"Failed to load response for {pano_key} from {jsonl_file}")
    return panoramas

DEFAULT_LABELS = "/home/erick/scratch/overhead_matching/landmark_tagger.labels.json"
DEFAULT_DATASET_BASE = "/data/overhead_matching/datasets/VIGOR"
DEFAULT_FEATHER_NAME = "v4_202001.feather"


def load_extractions(extractions_dir: Path) -> dict[str, dict]:
    """Load extractions, auto-detecting format (JSONL vs individual JSON)."""
    jsonl_files = list(extractions_dir.rglob("*.jsonl"))
    if jsonl_files:
        # Gemini JSONL format: key = "pano_id,lat,lon,"
        raw = load_osm_tag_extraction_jsonl(extractions_dir)
        # Extract pano_id from the key
        return {key.split(",")[0]: val for key, val in raw.items()}
    else:
        # Individual JSON files: {pano_id}.json
        result = {}
        for json_file in extractions_dir.rglob("*.json"):
            pano_id = json_file.stem
            with open(json_file) as f:
                result[pano_id] = json.load(f)
        return result


def extract_llm_tags(landmark: dict) -> list[tuple[str, str]]:
    """Extract (key, value) tag pairs from a single LLM landmark."""
    primary = landmark.get("primary_tag", {})
    tags = []
    if primary.get("key") and primary.get("value"):
        tags.append((primary["key"], primary["value"]))
    for tag in landmark.get("additional_tags", []):
        if tag.get("key") and tag.get("value"):
            tags.append((tag["key"], tag["value"]))
    return tags


def extract_osm_tags(row: pd.Series) -> list[tuple[str, str]]:
    """Extract (key, value) tag pairs from a feather DataFrame row."""
    tags = []
    for col in row.index:
        if col in META_COLS or col in EXCLUDE_KEYS:
            continue
        val = row[col]
        if pd.notna(val) and str(val).strip():
            tags.append((col, str(val)))
    return tags


def evaluate_panorama_level(
    panorama_labels: dict[str, str],
    extractions: dict[str, dict],
) -> dict:
    """Evaluate panorama-level classification (useful vs not_useful)."""
    tp = fp = tn = fn = 0
    missing = 0

    for pano_id, label in panorama_labels.items():
        if pano_id not in extractions:
            missing += 1
            continue
        landmarks = extractions[pano_id].get("landmarks", [])
        has_landmarks = len(landmarks) > 0

        if label == "useful":
            if has_landmarks:
                tp += 1
            else:
                fn += 1
        elif label == "not_useful":
            if has_landmarks:
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total, "missing": missing,
        "accuracy": accuracy, "precision": precision, "recall": recall,
    }


def evaluate_landmark_level(
    relevant_landmarks: dict[str, list[str]],
    extractions: dict[str, dict],
    osm_id_to_row: dict[str, int],
    df: pd.DataFrame,
) -> list[dict]:
    """Evaluate landmark-level matching using keyed substring matching."""
    results = []

    for pano_id, osm_id_strs in relevant_landmarks.items():
        if pano_id not in extractions:
            continue
        if not osm_id_strs:
            continue

        landmarks = extractions[pano_id].get("landmarks", [])

        # Build LLM tag lists
        llm_tags_list = []
        llm_landmark_names = []
        for lm in landmarks:
            tags = extract_llm_tags(lm)
            if tags:
                llm_tags_list.append(tags)
                llm_landmark_names.append(
                    lm.get("primary_tag", {}).get("value", "?")
                )

        # Build OSM tag lists from human labels
        osm_tags_list = []
        osm_ids_found = []
        for osm_id_str in osm_id_strs:
            row_idx = osm_id_to_row.get(osm_id_str)
            if row_idx is None:
                print(
                    f"  WARNING: OSM ID {osm_id_str} not found in feather",
                    file=sys.stderr,
                )
                continue
            tags = extract_osm_tags(df.iloc[row_idx])
            if tags:
                osm_tags_list.append(tags)
                osm_ids_found.append(osm_id_str)

        if not llm_tags_list or not osm_tags_list:
            results.append({
                "pano_id": pano_id,
                "num_llm": len(llm_tags_list),
                "num_osm": len(osm_tags_list),
                "precision": 0.0 if llm_tags_list else float("nan"),
                "recall": 0.0,
                "matches": [],
            })
            continue

        # Compute matches
        match_matrix = pnm.compute_keyed_substring_matches(
            llm_tags_list, osm_tags_list
        )

        # Per-panorama precision and recall
        osm_matched = np.any(match_matrix > 0, axis=0)
        llm_matched = np.any(match_matrix > 0, axis=1)
        recall = float(np.mean(osm_matched))
        precision = float(np.mean(llm_matched))

        # Collect match details
        matches = []
        for i in range(len(llm_tags_list)):
            for j in range(len(osm_tags_list)):
                if match_matrix[i, j] > 0:
                    matches.append({
                        "llm_idx": i,
                        "llm_name": llm_landmark_names[i],
                        "osm_idx": j,
                        "osm_id": osm_ids_found[j],
                    })

        results.append({
            "pano_id": pano_id,
            "num_llm": len(llm_tags_list),
            "num_osm": len(osm_tags_list),
            "precision": precision,
            "recall": recall,
            "matches": matches,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM OSM tag extraction against human labels"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(DEFAULT_LABELS),
        help="Path to landmark_tagger labels JSON",
    )
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        required=True,
        help="Directory of extraction results (JSONL or individual JSON)",
    )
    parser.add_argument(
        "--feather",
        type=Path,
        default=None,
        help="Path to feather file with OSM landmarks",
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="City name to auto-resolve feather file",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=Path(DEFAULT_DATASET_BASE),
        help=f"Base dataset directory (default: {DEFAULT_DATASET_BASE})",
    )
    parser.add_argument(
        "--feather-name",
        type=str,
        default=DEFAULT_FEATHER_NAME,
        help=f"Feather filename (default: {DEFAULT_FEATHER_NAME})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON dump of all match details",
    )
    args = parser.parse_args()

    # Resolve feather path
    feather_path = args.feather
    if feather_path is None:
        if args.city is None:
            parser.error("Provide --feather or --city")
        feather_path = args.dataset_base / args.city / "landmarks" / args.feather_name
    if not feather_path.exists():
        parser.error(f"Feather file not found: {feather_path}")

    # Load labels
    with open(args.labels) as f:
        labels_data = json.load(f)
    panorama_labels = labels_data.get("panorama_labels", {})
    relevant_landmarks = labels_data.get("relevant_landmarks", {})
    print(
        f"Loaded labels: {len(panorama_labels)} panorama labels, "
        f"{len(relevant_landmarks)} panoramas with relevant landmarks"
    )

    # Load feather and build OSM ID index
    df = pd.read_feather(feather_path)
    osm_id_to_row = {str(df.iloc[i]["id"]): i for i in range(len(df))}
    print(f"Loaded feather: {len(df)} landmarks, {len(osm_id_to_row)} unique IDs")

    # Load extractions
    extractions = load_extractions(args.extractions_dir)
    print(f"Loaded {len(extractions)} extractions")

    # --- Panorama-level evaluation ---
    pano_metrics = evaluate_panorama_level(panorama_labels, extractions)
    print("\n=== Panorama-Level Classification ===")
    print(f"  TP={pano_metrics['tp']}  FP={pano_metrics['fp']}")
    print(f"  FN={pano_metrics['fn']}  TN={pano_metrics['tn']}")
    print(f"  Accuracy:  {pano_metrics['accuracy']:.3f}")
    print(f"  Precision: {pano_metrics['precision']:.3f}")
    print(f"  Recall:    {pano_metrics['recall']:.3f}")
    if pano_metrics["missing"] > 0:
        print(f"  ({pano_metrics['missing']} labeled panos had no extraction)")

    # --- Landmark-level evaluation ---
    lm_results = evaluate_landmark_level(
        relevant_landmarks, extractions, osm_id_to_row, df
    )

    if lm_results:
        recalls = [r["recall"] for r in lm_results if not np.isnan(r["recall"])]
        precisions = [r["precision"] for r in lm_results if not np.isnan(r["precision"])]

        print("\n=== Landmark-Level Matching ===")
        print(f"  Panoramas evaluated: {len(lm_results)}")
        if recalls:
            print(f"  Mean recall:    {np.mean(recalls):.3f}")
        if precisions:
            print(f"  Mean precision: {np.mean(precisions):.3f}")

        # Per-panorama details to stderr
        print("\n--- Per-Panorama Details ---", file=sys.stderr)
        for r in sorted(lm_results, key=lambda x: x["recall"]):
            print(
                f"  {r['pano_id']}: "
                f"llm={r['num_llm']} osm={r['num_osm']} "
                f"P={r['precision']:.2f} R={r['recall']:.2f}",
                file=sys.stderr,
            )
            for m in r["matches"]:
                print(
                    f"    matched: LLM[{m['llm_idx']}] "
                    f"'{m['llm_name']}' <-> OSM {m['osm_id']}",
                    file=sys.stderr,
                )
    else:
        print("\nNo landmark-level results (no overlapping pano_ids)")

    # Optional JSON output
    if args.output:
        output_data = {
            "panorama_level": pano_metrics,
            "landmark_level": lm_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
