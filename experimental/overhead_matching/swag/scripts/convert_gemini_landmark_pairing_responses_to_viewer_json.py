"""Convert Gemini batch responses to viewer JSON format.

Parses pano/OSM landmarks directly from the request prompt text embedded in
each response record, so landmark indices match exactly what Gemini saw.
"""
import json
import re
import sys
from pathlib import Path

import common.torch.load_torch_deps
import experimental.overhead_matching.swag.data.vigor_dataset as vd


def parse_prompt_landmarks(prompt_text):
    """Parse Set 1 and Set 2 landmarks from the prompt text.

    Returns (pano_landmarks, osm_landmarks) where each is a list of tag strings
    in the same order as the prompt.
    """
    parts = prompt_text.split("Set 2 (map database):")
    if len(parts) != 2:
        return None, None

    def extract_items(section):
        items = []
        for line in section.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('Set '):
                continue
            m = re.match(r'(\d+)\.\s+(.*)', line)
            if m:
                items.append(m.group(2).strip())
        return items

    pano_lms = extract_items(parts[0])
    osm_lms = extract_items(parts[1])
    return pano_lms, osm_lms


def osm_ids_from_tags_str(tags_str, dataset):
    """Look up OSM feature IDs for a tag string by matching against pruned_props."""
    # Parse "key=value; key=value" back to a frozenset
    pairs = set()
    for kv in tags_str.split('; '):
        kv = kv.strip()
        if '=' in kv:
            k, v = kv.split('=', 1)
            pairs.add((k.strip(), v.strip()))
    target_fs = frozenset(pairs)

    matching = dataset._landmark_metadata[dataset._landmark_metadata["pruned_props"] == target_fs]
    if matching.empty:
        return []

    def parse_id(x):
        return json.loads(x.replace('(', '[').replace(')', ']').replace("'", '"'))

    return matching.id.apply(parse_id).values.tolist()


if len(sys.argv) < 3:
    print("Usage: convert_gemini_responses <responses_dir> <output_path> [city]")
    print("  responses_dir: Directory containing .jsonl response files")
    print("  output_path:   Output JSON file path")
    print("  city:          City name (default: Chicago)")
    sys.exit(1)

responses_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
city = sys.argv[3] if len(sys.argv) > 3 else "Chicago"

if not responses_dir.exists():
    print(f"Error: responses_dir does not exist: {responses_dir}")
    sys.exit(1)

jsonl_files = list(responses_dir.rglob("*.jsonl"))
if not jsonl_files:
    print(f"Error: no .jsonl files found in {responses_dir}")
    sys.exit(1)

# Load dataset (only needed for OSM feature ID lookup)
config = vd.VigorDatasetConfig(
    satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
    should_load_images=False, should_load_landmarks=True, landmark_version='v4_202001')
dataset = vd.VigorDataset(dataset_path=Path(f'/data/overhead_matching/datasets/VIGOR/{city}/'), config=config)

# Load responses — parse landmarks from the embedded request prompt
output = {}
total_prompt = 0
total_output = 0
total_thinking = 0
skipped = 0
for jsonl_file in jsonl_files:
    for line in jsonl_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"WARNING: Could not parse JSON line, skipping: {exc}")
            skipped += 1
            continue

        try:
            pano_id = record["key"]

            # Skip error records
            if record.get("error") is not None:
                print(f"WARNING: Skipping error record for {pano_id}: {record['error']}")
                skipped += 1
                continue

            response_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
            matches = json.loads(response_text)

            # Get the prompt that was sent to Gemini
            prompt_text = record["request"]["contents"][0]["parts"][0]["text"]
            pano_tag_strs, osm_tag_strs = parse_prompt_landmarks(prompt_text)
            if pano_tag_strs is None:
                print(f"WARNING: Could not parse prompt for {pano_id}, skipping")
                skipped += 1
                continue

            pano_lms = [{"tags": t} for t in pano_tag_strs]
            osm_lms = [{"tags": t, "ids": osm_ids_from_tags_str(t, dataset), "index": i}
                       for i, t in enumerate(osm_tag_strs)]

            # Filter out matches with out-of-bounds indices (including negatives)
            valid_matches = []
            for m in matches.get("matches", []):
                if m["set_1_id"] < 0 or m["set_1_id"] >= len(pano_lms):
                    print(f"WARNING: {pano_id} set_1_id={m['set_1_id']} out of bounds (max {len(pano_lms)-1}), dropping match")
                    continue
                valid_set2 = [osm_id for osm_id in m["set_2_matches"]
                              if 0 <= osm_id < len(osm_lms)]
                dropped = len(m["set_2_matches"]) - len(valid_set2)
                if dropped:
                    print(f"WARNING: {pano_id} dropped {dropped} out-of-bounds set_2 indices from match set_1_id={m['set_1_id']}")
                m_copy = dict(m)
                m_copy["set_2_matches"] = valid_set2
                valid_matches.append(m_copy)
            matches["matches"] = valid_matches

            # Accumulate token usage
            usage = record["response"].get("usageMetadata", {})
            total_prompt += usage.get("promptTokenCount", 0)
            total_output += usage.get("candidatesTokenCount", 0)
            total_thinking += usage.get("thoughtsTokenCount", 0)

            output[pano_id] = {"pano": pano_lms, "osm": osm_lms, "matches": matches}

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            pano_id = record.get("key", "<unknown>")
            print(f"WARNING: Error processing record {pano_id}: {type(exc).__name__}: {exc}, skipping")
            skipped += 1
            continue

output_path.write_text(json.dumps(output))
print(f"Wrote {len(output)} panoramas to {output_path}")
if skipped:
    print(f"Skipped {skipped} records due to errors")
print(f"\nToken usage:")
print(f"  Prompt:   {total_prompt:,}")
print(f"  Output:   {total_output:,}")
print(f"  Thinking: {total_thinking:,}")
print(f"  Total:    {total_prompt + total_output + total_thinking:,}")
