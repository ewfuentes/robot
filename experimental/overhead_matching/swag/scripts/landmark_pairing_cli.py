#!/usr/bin/env python3
"""Landmark pairing CLI for matching pano_v2 tags to OSM landmarks.

Loads VIGOR dataset + pano_v2 tags, prunes tags through a keep-list, and
builds prompts pairing street-level observations against OSM database entries.
Supports prompt preview and Gemini batch JSONL generation.

Usage:
    # Preview prompts without calling any model
    bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
        --pano_ids "EZd0yEWwrNSoGQhljhV1Uw" --prompt_only

    # Generate Gemini batch JSONL (with hard/easy negatives)
    bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
        --all --with_negatives --generate_batch /tmp/batch.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

import common.torch.load_torch_deps
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    extract_yaw_angles_from_bboxes, extract_panorama_data_across_cities)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    _TAGS_TO_KEEP_SET, _TAGS_TO_KEEP_PREFIXES)


SYSTEM_PROMPT = """You are a landmark matching expert. Given two sets of OpenStreetMap-style tag bundles,
identify which landmarks from Set 1 (extracted from street-level imagery) correspond to
landmarks in Set 2 (from an OpenStreetMap database). Both sets use key=value tag notation.

For each Set 1 landmark that has a match, rate the uniqueness of that landmark's tag set (1-5).
The score describes how distinctive the panorama landmark is on its own — NOT the quality
of the match or how similar the two sides are.
  1 = extremely generic (e.g., building=yes)
  2 = common category (e.g., amenity=restaurant)
  3 = moderately specific (e.g., shop=convenience; brand=7-Eleven)
  4 = quite distinctive (e.g., amenity=library; name=Harold Washington Library)
  5 = highly unique/unmistakable (e.g., tourism=attraction; name=Cloud Gate)

Only propose matches you are confident about. Some landmarks may have no match."""

SYSTEM_PROMPT_WITH_NEGATIVES = """You are a landmark matching expert. Given two sets of OpenStreetMap-style tag bundles,
identify which landmarks from Set 1 (extracted from street-level imagery) represent the same physical object as a
landmark in Set 2 (from an OpenStreetMap database). Both sets use key=value tag notation.

For each Set 1 landmark that has a match, rate the uniqueness of that landmark's tag set (1-5).
The score describes how distinctive the panorama landmark is on its own — NOT the quality
of the match or how similar the two sides are.
  1 = extremely generic (e.g., building=yes)
  2 = common category (e.g., amenity=restaurant)
  3 = moderately specific (e.g., shop=convenience; brand=7-Eleven)
  4 = quite distinctive (e.g., amenity=library; name=Harold Washington Library)
  5 = highly unique/unmistakable (e.g., tourism=attraction; name=Cloud Gate)

For each Set 1 landmark, also provide 0-2 negative examples from Set 2 — landmarks that
are NOT a match. Label each as "hard" or "easy":
  - hard: same general category but UNAMBIGUOUSLY a different landmark. Valid reasons:
      * Names that refer to different entities (e.g., "Lou Malnati's" vs "Portillo's",
        "26th Ave" vs "12th Ave")
      * Vastly different scale (e.g., a 20-story tower vs a 2-story house)
    Do NOT treat these as conflicts — they do not make valid hard negatives:
      * Small numeric differences (building:levels=3 vs 4) — the extractor is often off
      * Different tag specificity for the same thing (building=apartments vs building=yes,
        highway=traffic_signals vs crossing=traffic_signals)
      * One name being a substring/variant of another (Wolf Point vs Wolf Point East)
      * Features that could be spatially contained (grass area within a park)
      * Missing tags on one side (a building with vs without an address)
    When in doubt, do NOT include it as a negative.
  - easy: obviously unrelated (completely different type, e.g., a restaurant vs a street).

Only propose matches you are confident about. Some landmarks may have no match."""

JSON_SCHEMA = json.dumps({
    "type": "object",
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["set_1_id", "set_2_matches", "uniqueness_score"],
                "properties": {
                    "set_1_id": {"type": "integer"},
                    "set_2_matches": {"type": "array", "items": {"type": "integer"}},
                    "uniqueness_score": {"type": "integer"}
                }
            }
        }
    }
})

JSON_SCHEMA_WITH_NEGATIVES = json.dumps({
    "type": "object",
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["set_1_id", "set_2_matches", "uniqueness_score", "negatives"],
                "properties": {
                    "set_1_id": {"type": "integer"},
                    "set_2_matches": {"type": "array", "items": {"type": "integer"}},
                    "uniqueness_score": {"type": "integer"},
                    "negatives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["set_2_id", "difficulty"],
                            "properties": {
                                "set_2_id": {"type": "integer"},
                                "difficulty": {"type": "string", "enum": ["hard", "easy"]}
                            }
                        }
                    }
                }
            }
        }
    }
})


def format_tags(tags):
    """Format a list of (key, value) tuples as 'key=value; key=value'."""
    return '; '.join(f"{k}={v}" for k, v in tags)


def itemized_list(items):
    out = []
    for i, v in enumerate(items):
        out.append(f" {i}. {v}")
    return '\n'.join(out)


def _should_keep_tag(key):
    """Check if a tag key is in the keep list."""
    if key in _TAGS_TO_KEEP_SET:
        return True
    return any(key.startswith(p) for p in _TAGS_TO_KEEP_PREFIXES)


# Track pruning stats globally so we can report after loading
_pano_prune_stats = {"total_landmarks": 0, "kept_landmarks": 0, "total_tags": 0, "kept_tags": 0}


def extract_tags_from_pano_data(pano_id_clean, pano_data):
    """Extract tags from pano_v2 data, filtering through the keep list."""
    landmarks = []
    for lm in pano_data["landmarks"]:
        primary = lm.get("primary_tag", {})
        raw_tags = []
        if primary.get("key") and primary.get("value"):
            raw_tags.append((primary["key"], primary["value"]))
        for tag in lm.get("additional_tags", []):
            if tag.get("key") and tag.get("value"):
                raw_tags.append((tag["key"], tag["value"]))
        if not raw_tags:
            continue

        _pano_prune_stats["total_landmarks"] += 1
        _pano_prune_stats["total_tags"] += len(raw_tags)

        # Filter through keep list
        tags = [(k, v) for k, v in raw_tags if _should_keep_tag(k)]
        _pano_prune_stats["kept_tags"] += len(tags)

        if not tags:
            continue

        _pano_prune_stats["kept_landmarks"] += 1
        yaw_angles = extract_yaw_angles_from_bboxes(lm.get("bounding_boxes", []))
        confidence = lm.get("confidence", "unknown")
        landmarks.append({
            "tags": tags,
            "yaw_angles": yaw_angles,
            "confidence": confidence,
        })
    return landmarks if landmarks else None


def osm_landmarks_from_pano_id(pano_id, dataset):
    pano_info = dataset._panorama_metadata[dataset._panorama_metadata.pano_id == pano_id].iloc[0]
    sat_idxs = pano_info.positive_satellite_idxs + pano_info.semipositive_satellite_idxs
    osm_landmarks = set()
    for sat_idx in sat_idxs:
        osm_landmarks |= set(dataset._satellite_metadata.iloc[sat_idx]["landmark_idxs"])
    all_landmarks = dataset._landmark_metadata.iloc[list(osm_landmarks)][["pruned_props", "id", "geometry", "geometry_px"]]
    # Deduplicate by pruned_props, sort for deterministic ordering across runs
    unique_props = sorted(set(all_landmarks["pruned_props"].values), key=lambda fs: sorted(fs))
    return [list(x) for x in unique_props], all_landmarks


def create_prompt(pano_id, pano_tags_from_pano_id, dataset):
    pano_tags = pano_tags_from_pano_id[pano_id]
    osm_tags, _ = osm_landmarks_from_pano_id(pano_id, dataset)
    return f"""Set 1 (street-level observations):
{itemized_list(format_tags(lm["tags"]) for lm in pano_tags)}

Set 2 (map database):
{itemized_list(format_tags(x) for x in osm_tags)}"""



def main():
    parser = argparse.ArgumentParser(description='Landmark pairing CLI')
    parser.add_argument('--city', default='Chicago', help='City name')
    parser.add_argument('--pano_ids', help='Comma-separated panorama IDs')
    parser.add_argument('--random_n', type=int, help='Pick N random panoramas')
    parser.add_argument('--all', action='store_true', help='Use all panoramas that have both pano_v2 and dataset data')
    parser.add_argument('--prompt_only', action='store_true',
                        help='Print prompts without calling any model')
    parser.add_argument('--generate_batch', type=str, default=None,
                        help='Write Gemini batch JSONL to this path and exit')
    parser.add_argument('--thinking_level', default='HIGH',
                        choices=['NONE', 'LOW', 'MEDIUM', 'HIGH'],
                        help='Gemini thinking level for batch requests (default: HIGH)')
    parser.add_argument('--with_negatives', action='store_true',
                        help='Include hard/easy negative examples in output')
    parser.add_argument('--dataset_base', default='/data/overhead_matching/datasets/VIGOR/',
                        help='Base path for VIGOR dataset')
    parser.add_argument('--pano_v2_base', required=True,
                        help='Base path for pano_v2 embeddings (contains city subdirs)')
    parser.add_argument('--landmark_version', required=True,
                        help='Landmark version for VIGOR dataset')
    args = parser.parse_args()

    if not args.prompt_only and not args.generate_batch:
        parser.error("Must specify --prompt_only or --generate_batch")

    # Load dataset
    dataset_path = Path(args.dataset_base) / args.city
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=args.landmark_version,
    )
    print(f"Loading dataset from {dataset_path}...")
    dataset = vd.VigorDataset(dataset_path=dataset_path, config=config)

    # Load pano_v2 tags
    pano_v2_base = Path(args.pano_v2_base)
    print(f"Loading pano_v2 tags from {pano_v2_base}...")
    _pano_prune_stats["total_landmarks"] = 0
    _pano_prune_stats["kept_landmarks"] = 0
    _pano_prune_stats["total_tags"] = 0
    _pano_prune_stats["kept_tags"] = 0
    pano_tags_from_pano_id = extract_panorama_data_across_cities(
        pano_v2_base, extract_tags_from_pano_data)
    s = _pano_prune_stats
    dropped_lm = s["total_landmarks"] - s["kept_landmarks"]
    dropped_tags = s["total_tags"] - s["kept_tags"]
    print(f"Loaded tags for {len(pano_tags_from_pano_id)} panoramas (across all cities in {pano_v2_base})")
    print(f"  Pano landmarks: {s['kept_landmarks']}/{s['total_landmarks']} kept ({dropped_lm} dropped, all tags pruned)")
    print(f"  Pano tags: {s['kept_tags']}/{s['total_tags']} kept ({dropped_tags} dropped by keep-list filter)")

    # Select panorama IDs
    dataset_pano_ids = set(dataset._panorama_metadata.pano_id.values)
    available = [pid for pid in pano_tags_from_pano_id if pid in dataset_pano_ids]
    print(f"  {len(available)} panoramas overlap with {args.city} VIGOR dataset ({len(dataset_pano_ids)} VIGOR panos)")
    if args.pano_ids:
        pano_ids = args.pano_ids.split(',')
    elif args.all:
        pano_ids = sorted(available)
    elif args.random_n:
        rng = np.random.default_rng(42)
        pano_ids = rng.choice(
            available, size=min(args.random_n, len(available)),
            replace=False).tolist()
    else:
        parser.error("Must specify --pano_ids, --random_n, or --all")

    # Filter to valid pano_ids
    valid_pano_ids = [pid for pid in pano_ids
                      if pid in pano_tags_from_pano_id and pid in dataset_pano_ids]
    skipped = len(pano_ids) - len(valid_pano_ids)
    if skipped:
        print(f"WARNING: {skipped} pano_ids not in pano_v2 or VIGOR data, skipping")
    if not valid_pano_ids:
        print("ERROR: No valid pano_ids remain after filtering.")
        return

    if args.prompt_only:
        for pano_id in valid_pano_ids:
            print(f"\n{'='*80}")
            print(f"Panorama: {pano_id}")
            print(f"{'='*80}")
            print("\n--- System Prompt ---")
            print(SYSTEM_PROMPT)
            print("\n--- User Prompt ---")
            print(create_prompt(pano_id, pano_tags_from_pano_id, dataset))
        return

    if args.generate_batch:
        system_prompt = SYSTEM_PROMPT_WITH_NEGATIVES if args.with_negatives else SYSTEM_PROMPT
        schema = json.loads(JSON_SCHEMA_WITH_NEGATIVES if args.with_negatives else JSON_SCHEMA)
        gen_config = {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        }
        if args.thinking_level != 'NONE':
            gen_config["thinkingConfig"] = {"thinkingLevel": args.thinking_level}
        batch_lines = []
        for pano_id in valid_pano_ids:
            user_prompt = create_prompt(pano_id, pano_tags_from_pano_id, dataset)
            request = {
                "key": pano_id,
                "request": {
                    "contents": [{"parts": [{"text": user_prompt}], "role": "user"}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "generationConfig": gen_config,
                }
            }
            batch_lines.append(json.dumps(request))
        out_path = Path(args.generate_batch)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(batch_lines) + '\n')
        print(f"Wrote {len(batch_lines)} Gemini batch requests to {out_path}")


if __name__ == '__main__':
    main()
