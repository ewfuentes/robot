#!/usr/bin/env python3
"""Landmark pairing CLI prototype using Claude CLI for prompt iteration.

Loads VIGOR dataset + pano_v2 tags, constructs the same prompt format used
for Gemini batch requests, and pipes each prompt to `claude --print` for
interactive testing and iteration.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
        --random_n 3

    bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
        --pano_ids "EZd0yEWwrNSoGQhljhV1Uw,--z0RFQbsumsJC2wWUUKIg" \
        --prompt_only

    bazel run //experimental/overhead_matching/swag/scripts:landmark_pairing_cli -- \
        --random_n 20 --parallel 5 --output /tmp/matches.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

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

Only propose matches you are confident about. Some landmarks may have no match."""

# Looser negative rules (v2), kept for reference.
_SYSTEM_PROMPT_WITH_NEGATIVES_V2 = """You are a landmark matching expert. Given two sets of OpenStreetMap-style tag bundles,
identify which landmarks from Set 1 (extracted from street-level imagery) represent the same physical object as a
landmark in Set 2 (from an OpenStreetMap database). Both sets use key=value tag notation.

For each Set 1 landmark that has a match, rate the uniqueness of that landmark's tag set (1-5).
The score describes how distinctive the panorama landmark is on its own — NOT the quality
of the match or how similar the two sides are.
  1 = extremely generic (e.g., building=yes)
  2 = common category (e.g., amenity=restaurant)
  3 = moderately specific (e.g., shop=convenience; brand=7-Eleven)
  4 = quite distinctive (e.g., amenity=library; name=Harold Washington Library)

For each Set 1 landmark, also provide 0-2 negative examples from Set 2 — landmarks that
are NOT a match. Label each as "hard" or "easy":
  - hard: shares some superficial similarity but is clearly a different landmark.
    If it could possibly be the same landmark (e.g., a tag was just not labeled) do NOT include it as a negative.
  - easy: obviously unrelated (completely different type).

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


# Global list of child processes for Ctrl-C cleanup
_child_procs = []
_child_procs_lock = Lock()


def _cleanup_children(signum, frame):
    with _child_procs_lock:
        for proc in _child_procs:
            try:
                proc.terminate()
            except Exception:
                pass
    sys.exit(1)


def run_claude(cmd):
    """Run claude subprocess, interruptible by Ctrl-C."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with _child_procs_lock:
        _child_procs.append(proc)
    try:
        stdout, stderr = proc.communicate()
    finally:
        with _child_procs_lock:
            if proc in _child_procs:
                _child_procs.remove(proc)
    return proc.returncode, stdout, stderr


def process_pano(pano_id, pano_tags_from_pano_id, dataset, model):
    """Process a single panorama: build prompt, call Claude, parse result."""
    user_prompt = create_prompt(pano_id, pano_tags_from_pano_id, dataset)

    cmd = [
        "claude", "--print",
        "--output-format", "json",
        "--system-prompt", SYSTEM_PROMPT,
        "--json-schema", JSON_SCHEMA,
        "--tools", "",
        "--model", model,
        "-p", user_prompt,
    ]

    returncode, stdout, stderr = run_claude(cmd)

    result = {
        "pano_id": pano_id,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "matches": None,
        "error": None,
    }

    if returncode != 0:
        result["error"] = stderr.strip()
        return result

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        result["error"] = f"Could not parse claude output: {stdout[:200]}"
        return result

    usage = envelope.get("usage", {})
    result["input_tokens"] = usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    result["output_tokens"] = usage.get("output_tokens", 0)
    result["cost_usd"] = envelope.get("total_cost_usd", 0)

    # --json-schema puts parsed result in structured_output
    parsed = envelope.get("structured_output")
    if parsed is None:
        response_text = envelope.get("result", "")
        try:
            parsed = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            result["error"] = f"No structured output: {response_text[:200]}"
            return result

    result["matches"] = parsed
    return result


def _write_outputs(output_path, all_results, log_lines, total_input, total_output, total_cost, completed, total):
    """Write JSON results and human-readable log if output path is set."""
    if not output_path or not all_results:
        return
    out = Path(output_path)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote results to {out}")

    readable_path = out.with_suffix('.readable.txt')
    readable_path.write_text('\n'.join(log_lines) + '\n')
    print(f"Wrote readable log to {readable_path}")


def main():
    parser = argparse.ArgumentParser(description='Landmark pairing CLI using Claude')
    parser.add_argument('--city', default='Chicago', help='City name')
    parser.add_argument('--pano_ids', help='Comma-separated panorama IDs')
    parser.add_argument('--random_n', type=int, help='Pick N random panoramas')
    parser.add_argument('--all', action='store_true', help='Use all panoramas that have both pano_v2 and dataset data')
    parser.add_argument('--prompt_only', action='store_true',
                        help='Print prompt without calling Claude')
    parser.add_argument('--generate_batch', type=str, default=None,
                        help='Write Gemini batch JSONL to this path and exit')
    parser.add_argument('--thinking_level', default='HIGH',
                        choices=['NONE', 'LOW', 'MEDIUM', 'HIGH'],
                        help='Gemini thinking level for batch requests (default: HIGH)')
    parser.add_argument('--with_negatives', action='store_true',
                        help='Include hard/easy negative examples in output')
    parser.add_argument('--model', default='sonnet',
                        help='Claude model to use (default: sonnet)')
    parser.add_argument('--output', type=str, default=None,
                        help='Write results JSON to this path')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of panoramas to process in parallel (default: 1)')
    parser.add_argument('--dataset_base', default='/data/overhead_matching/datasets/VIGOR/',
                        help='Base path for VIGOR dataset')
    parser.add_argument('--pano_v2_base',
                        default='/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/',
                        help='Base path for pano_v2 embeddings')
    parser.add_argument('--landmark_version', default='v4_202001',
                        help='Landmark version for VIGOR dataset')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _cleanup_children)

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
    print(f"Loaded tags for {len(pano_tags_from_pano_id)} panoramas")
    print(f"  Pano landmarks: {s['kept_landmarks']}/{s['total_landmarks']} kept ({dropped_lm} dropped, all tags pruned)")
    print(f"  Pano tags: {s['kept_tags']}/{s['total_tags']} kept ({dropped_tags} dropped by keep-list filter)")

    # Select panorama IDs
    dataset_pano_ids = set(dataset._panorama_metadata.pano_id.values)
    if args.pano_ids:
        pano_ids = args.pano_ids.split(',')
    elif args.all:
        pano_ids = sorted([pid for pid in pano_tags_from_pano_id if pid in dataset_pano_ids])
    elif args.random_n:
        rng = np.random.default_rng(42)
        available = [pid for pid in pano_tags_from_pano_id
                     if pid in dataset_pano_ids]
        pano_ids = rng.choice(
            available, size=min(args.random_n, len(available)),
            replace=False).tolist()
    else:
        parser.error("Must specify --pano_ids, --random_n, or --all")

    # Filter to valid pano_ids
    valid_pano_ids = [pid for pid in pano_ids if pid in pano_tags_from_pano_id]
    skipped = len(pano_ids) - len(valid_pano_ids)
    if skipped:
        print(f"WARNING: {skipped} pano_ids not in pano_v2 data, skipping")

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
        return

    # Resume: load existing results and skip already-processed panos
    all_results = {}
    log_lines = []
    if args.output and Path(args.output).exists():
        try:
            all_results = json.loads(Path(args.output).read_text())
            already_done = set(all_results.keys()) & set(valid_pano_ids)
            if already_done:
                print(f"Resuming: {len(already_done)} panos already in {args.output}, skipping them")
                valid_pano_ids = [pid for pid in valid_pano_ids if pid not in already_done]
                log_lines.append(f"Resumed from {args.output} ({len(already_done)} already done)")
        except (json.JSONDecodeError, TypeError):
            pass

    if not valid_pano_ids:
        print("All panoramas already processed.")
        return

    # Process panoramas (possibly in parallel)
    print(f"\nProcessing {len(valid_pano_ids)} panoramas (parallel={args.parallel})...")
    total_input = 0
    total_output = 0
    total_cost = 0.0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 3
    stop_early = False
    print_lock = Lock()

    def handle_result(r):
        nonlocal total_input, total_output, total_cost, consecutive_errors, stop_early
        total_input += r["input_tokens"]
        total_output += r["output_tokens"]
        total_cost += r["cost_usd"]

        with print_lock:
            lines = []
            lines.append(f"{'='*80}")
            lines.append(f"Panorama: {r['pano_id']}")
            lines.append(f"  Tokens: {r['input_tokens']} in / {r['output_tokens']} out  Cost: ${r['cost_usd']:.4f}")

            if r["error"]:
                consecutive_errors += 1
                lines.append(f"  ERROR: {r['error']}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    lines.append(f"  STOPPING: {consecutive_errors} consecutive errors, likely quota/auth issue")
                    stop_early = True
                for l in lines:
                    print(l)
                log_lines.extend(lines)
                return

            consecutive_errors = 0

            parsed = r["matches"]
            pano_tags = pano_tags_from_pano_id[r["pano_id"]]
            osm_tags, _ = osm_landmarks_from_pano_id(r["pano_id"], dataset)

            for match in parsed.get("matches", []):
                set1 = match["set_1_id"]
                score = match.get("uniqueness_score", 0)
                pano_tag_str = format_tags(pano_tags[set1]["tags"])
                lines.append(f"  [U{score}] Pano {set1}: {pano_tag_str}")
                for set2 in match["set_2_matches"]:
                    osm_tag_str = format_tags(osm_tags[set2])
                    lines.append(f"         -> OSM {set2}: {osm_tag_str}")

            if not parsed.get("matches"):
                lines.append("  (no matches)")

            for l in lines:
                print(l)
            log_lines.extend(lines)

            # Build output entry
            pano_lms = [{"tags": format_tags(lm["tags"]),
                         "confidence": lm["confidence"],
                         "yaw_angles": lm["yaw_angles"]} for lm in pano_tags]

            osm_lms = [{"tags": format_tags(tags), "index": idx}
                       for idx, tags in enumerate(osm_tags)]

            all_results[r["pano_id"]] = {
                "pano": pano_lms,
                "osm": osm_lms,
                "matches": parsed,
            }

    try:
        if args.parallel <= 1:
            for pano_id in valid_pano_ids:
                if stop_early:
                    break
                r = process_pano(pano_id, pano_tags_from_pano_id, dataset, args.model)
                handle_result(r)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(process_pano, pid, pano_tags_from_pano_id, dataset, args.model): pid
                    for pid in valid_pano_ids
                }
                for future in as_completed(futures):
                    handle_result(future.result())
                    if stop_early:
                        for f in futures:
                            f.cancel()
                        break
    except KeyboardInterrupt:
        print("\n\nInterrupted. Cleaning up...")
        with _child_procs_lock:
            for proc in _child_procs:
                try:
                    proc.terminate()
                except Exception:
                    pass
        # Still write partial results if we have an output path
        _write_outputs(args.output, all_results, log_lines, total_input, total_output, total_cost, len(all_results), len(valid_pano_ids))
        sys.exit(1)

    summary = f"Total: {total_input} input + {total_output} output tokens, ${total_cost:.4f}"
    completed = f"Completed: {len(all_results)}/{len(valid_pano_ids)} panoramas"
    print(f"\n{'='*80}")
    print(summary)
    print(completed)
    log_lines.append(f"{'='*80}")
    log_lines.append(summary)
    log_lines.append(completed)

    _write_outputs(args.output, all_results, log_lines, total_input, total_output, total_cost, len(all_results), len(valid_pano_ids))


if __name__ == '__main__':
    main()
