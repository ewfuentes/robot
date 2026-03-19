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
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    _TAGS_TO_KEEP_SET, _TAGS_TO_KEEP_PREFIXES)


def should_keep_tag(key):
    if key in _TAGS_TO_KEEP_SET:
        return True
    return any(key.startswith(p) for p in _TAGS_TO_KEEP_PREFIXES)


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


responses_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
city = sys.argv[3] if len(sys.argv) > 3 else "Chicago"

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
for jsonl_file in responses_dir.rglob("*.jsonl"):
    for line in jsonl_file.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        pano_id = record["key"]
        response_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
        matches = json.loads(response_text)

        # Get the prompt that was sent to Gemini
        prompt_text = record["request"]["contents"][0]["parts"][0]["text"]
        pano_tag_strs, osm_tag_strs = parse_prompt_landmarks(prompt_text)
        if pano_tag_strs is None:
            print(f"WARNING: Could not parse prompt for {pano_id}, skipping")
            continue

        pano_lms = [{"tags": t} for t in pano_tag_strs]
        osm_lms = [{"tags": t, "ids": osm_ids_from_tags_str(t, dataset), "index": i}
                   for i, t in enumerate(osm_tag_strs)]

        # Validate match indices are in bounds
        for m in matches.get("matches", []):
            if m["set_1_id"] >= len(pano_lms):
                print(f"WARNING: {pano_id} set_1_id={m['set_1_id']} out of bounds (max {len(pano_lms)-1})")
            for osm_id in m["set_2_matches"]:
                if osm_id >= len(osm_lms):
                    print(f"WARNING: {pano_id} set_2_matches={osm_id} out of bounds (max {len(osm_lms)-1})")

        # Accumulate token usage
        usage = record["response"].get("usageMetadata", {})
        total_prompt += usage.get("promptTokenCount", 0)
        total_output += usage.get("candidatesTokenCount", 0)
        total_thinking += usage.get("thoughtsTokenCount", 0)

        output[pano_id] = {"pano": pano_lms, "osm": osm_lms, "matches": matches}

output_path.write_text(json.dumps(output))
print(f"Wrote {len(output)} panoramas to {output_path}")
print(f"\nToken usage:")
print(f"  Prompt:   {total_prompt:,}")
print(f"  Output:   {total_output:,}")
print(f"  Thinking: {total_thinking:,}")
print(f"  Total:    {total_prompt + total_output + total_thinking:,}")
