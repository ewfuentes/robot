import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import common.torch.load_torch_deps
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    return Path, mo, slu, vd


@app.cell
def _():
    import numpy as np
    import pickle
    import json
    return json, np, pickle


@app.cell
def _():
    from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
        extract_yaw_angles_from_bboxes, extract_panorama_data_across_cities)
    from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
        _TAGS_TO_KEEP_SET, _TAGS_TO_KEEP_PREFIXES)
    return _TAGS_TO_KEEP_PREFIXES, _TAGS_TO_KEEP_SET, extract_panorama_data_across_cities, extract_yaw_angles_from_bboxes


@app.cell
def _(Path, vd):
    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago/')
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version='v3'
    )

    new_dataset = vd.VigorDataset(dataset_path=_dataset_path,
                             config=_config)

    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago/')
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version='v4_202001'
    )

    historical_dataset = vd.VigorDataset(dataset_path=_dataset_path,
                             config=_config)
    return historical_dataset, new_dataset


@app.cell
def _(Path, _TAGS_TO_KEEP_PREFIXES, _TAGS_TO_KEEP_SET, extract_panorama_data_across_cities, extract_yaw_angles_from_bboxes):
    _pano_v2_base = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/')
    _total_lm = 0
    _kept_lm = 0
    _total_tags = 0
    _kept_tags = 0

    def _should_keep_tag(key):
        if key in _TAGS_TO_KEEP_SET:
            return True
        return any(key.startswith(p) for p in _TAGS_TO_KEEP_PREFIXES)

    def _extract_tags(pano_id_clean, pano_data):
        nonlocal _total_lm, _kept_lm, _total_tags, _kept_tags
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
            _total_lm += 1
            _total_tags += len(raw_tags)
            tags = [(k, v) for k, v in raw_tags if _should_keep_tag(k)]
            _kept_tags += len(tags)
            if not tags:
                continue
            _kept_lm += 1
            yaw_angles = extract_yaw_angles_from_bboxes(lm.get("bounding_boxes", []))
            confidence = lm.get("confidence", "unknown")
            landmarks.append({
                "tags": tags,
                "yaw_angles": yaw_angles,
                "confidence": confidence,
            })
        return landmarks if landmarks else None

    pano_tags_from_pano_id = extract_panorama_data_across_cities(_pano_v2_base, _extract_tags)
    print(f"Loaded pano_v2 tags for {len(pano_tags_from_pano_id)} panoramas")
    print(f"  Pano landmarks: {_kept_lm}/{_total_lm} kept ({_total_lm - _kept_lm} dropped, all tags pruned)")
    print(f"  Pano tags: {_kept_tags}/{_total_tags} kept ({_total_tags - _kept_tags} dropped by keep-list filter)")
    return (pano_tags_from_pano_id,)


@app.cell
def _(historical_dataset):
    _pano_id = '--z0RFQbsumsJC2wWUUKIg'

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

    osm_landmarks_from_pano_id(_pano_id, historical_dataset)
    return (osm_landmarks_from_pano_id,)


@app.cell
def _(historical_dataset, np, osm_landmarks_from_pano_id, pano_tags_from_pano_id):
    _pano_sizes = [len(v) for v in pano_tags_from_pano_id.values()]
    print("=== Pano Landmark Set Sizes ===")
    print(f"  Count: {len(_pano_sizes)}")
    print(f"  Min: {np.min(_pano_sizes)}, Max: {np.max(_pano_sizes)}")
    print(f"  Mean: {np.mean(_pano_sizes):.1f}, Median: {np.median(_pano_sizes):.1f}")
    print(f"  P95: {np.percentile(_pano_sizes, 95):.1f}, P99: {np.percentile(_pano_sizes, 99):.1f}")

    _osm_sizes = {}
    for _pid in pano_tags_from_pano_id:
        try:
            _osm_lms, _ = osm_landmarks_from_pano_id(_pid, historical_dataset)
            _osm_sizes[_pid] = len(_osm_lms)
        except Exception:
            pass

    _osm_size_vals = list(_osm_sizes.values())
    if _osm_size_vals:
        print("\n=== OSM Landmark Set Sizes (post-dedup) ===")
        print(f"  Count: {len(_osm_size_vals)}")
        print(f"  Min: {np.min(_osm_size_vals)}, Max: {np.max(_osm_size_vals)}")
        print(f"  Mean: {np.mean(_osm_size_vals):.1f}, Median: {np.median(_osm_size_vals):.1f}")
        print(f"  P95: {np.percentile(_osm_size_vals, 95):.1f}, P99: {np.percentile(_osm_size_vals, 99):.1f}")

        _sorted_osm = sorted(_osm_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n=== Top 10 by OSM landmark count ===")
        for _pid, _cnt in _sorted_osm:
            print(f"  {_pid}: {_cnt}")

    _sorted_pano = sorted(pano_tags_from_pano_id.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\n=== Top 10 by Pano landmark count ===")
    for _pid, _lms in _sorted_pano:
        print(f"  {_pid}: {len(_lms)}")
    return


@app.cell
def _(
    historical_dataset,
    osm_landmarks_from_pano_id,
    pano_tags_from_pano_id,
):
    _pano_id = 'EZd0yEWwrNSoGQhljhV1Uw'

    def format_tags(tags):
        """Format a list of (key, value) tuples as 'key=value; key=value'."""
        return '; '.join(f"{k}={v}" for k, v in tags)

    def itemized_list(items):
        out = []
        for i, v in enumerate(items):
            out.append(f" {i}. {v}")
        return '\n'.join(out)

    GEMINI_SCHEMA = {
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
    }

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

    def create_prompt(pano_id):
        pano_tags = pano_tags_from_pano_id[pano_id]
        osm_tags, _ = osm_landmarks_from_pano_id(pano_id, historical_dataset)
        return f"""Set 1 (street-level observations):
{itemized_list(format_tags(lm["tags"]) for lm in pano_tags)}

Set 2 (map database):
{itemized_list(format_tags(x) for x in osm_tags)}"""

    print(create_prompt(_pano_id))
    return GEMINI_SCHEMA, SYSTEM_PROMPT, create_prompt, format_tags, itemized_list


@app.cell
def _(GEMINI_SCHEMA, SYSTEM_PROMPT, create_prompt):
    def create_gemini_request(pano_id):
        return {
            "key": pano_id,
            "request": {
                "contents": [{"parts": [{"text": create_prompt(pano_id)}], "role": "user"}],
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": GEMINI_SCHEMA,
                }
            }
        }

    return (create_gemini_request,)


@app.cell
def _(create_gemini_request, pano_tags_from_pano_id):
    _selected_panoramas = list(pano_tags_from_pano_id.keys())
    requests = [create_gemini_request(_p) for _p in _selected_panoramas]
    return (requests,)


@app.cell(disabled=True)
def _(Path, json, requests):
    _batch_output_dir = Path("/tmp/landmark_pairing_requests/")
    _batch_output_dir.mkdir(parents=True, exist_ok=True)

    _batch_size = 10_000
    for _idx in range(0, len(requests), _batch_size):
        _batch = requests[_idx:_idx + _batch_size]
        _batch_path = _batch_output_dir / f"requests_{_idx // _batch_size:03d}.jsonl"
        _batch_path.write_text('\n'.join(json.dumps(r) for r in _batch))
        print(f"Wrote {len(_batch)} requests to {_batch_path}")
    return


@app.cell
def _(
    Path,
    historical_dataset,
    json,
    slu,
):
    import re as _re
    _results_path = Path('/data/overhead_matching/datasets/landmark_correspondence/v8_gemini_pano_v2/')

    def _parse_prompt_landmarks(prompt_text):
        """Parse Set 1 and Set 2 landmarks from the prompt text."""
        parts = prompt_text.split("Set 2 (map database):")
        if len(parts) != 2:
            return None, None
        def extract_items(section):
            items = []
            for line in section.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('Set '):
                    continue
                m = _re.match(r'(\d+)\.\s+(.*)', line)
                if m:
                    items.append(m.group(2).strip())
            return items
        return extract_items(parts[0]), extract_items(parts[1])

    def _osm_ids_from_tags_str(tags_str, dataset):
        """Look up OSM feature IDs for a tag string."""
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

    def load_matches_from_folder(path, dataset):
        response_json = slu.load_all_jsonl_from_folder(path / "responses")
        output = {}
        for record in response_json:
            # Parse response — auto-detect Gemini vs OpenAI format
            if "key" in record and "response" in record and "candidates" in record.get("response", {}):
                pano_id = record["key"]
                response_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
                matches = json.loads(response_text)
                # Parse landmarks from the request prompt to preserve ordering
                prompt_text = record["request"]["contents"][0]["parts"][0]["text"]
                pano_tag_strs, osm_tag_strs = _parse_prompt_landmarks(prompt_text)
            elif "custom_id" in record and "response" in record and "body" in record.get("response", {}):
                pano_id = record["custom_id"]
                matches = json.loads(record["response"]["body"]["choices"][0]["message"]["content"])
                prompt_text = record["body"]["messages"][-1]["content"]
                pano_tag_strs, osm_tag_strs = _parse_prompt_landmarks(prompt_text)
            else:
                print(f"Unknown response format, skipping: {list(record.keys())}")
                continue

            if pano_tag_strs is None:
                print(f"WARNING: Could not parse prompt for {pano_id}, skipping")
                continue

            pano_lms = [{"tags": t} for t in pano_tag_strs]
            osm_lms = [{"tags": t, "ids": _osm_ids_from_tags_str(t, dataset), "index": i}
                       for i, t in enumerate(osm_tag_strs)]

            output[pano_id] = {
                "pano": pano_lms,
                "osm": osm_lms,
                "matches": matches,
            }

        return output

    if _results_path.exists():
        match_results = load_matches_from_folder(_results_path, historical_dataset)
        Path('/tmp/gemini_pano_v2.json').write_text(json.dumps(match_results))
        print(f"Loaded {len(match_results)} panorama matches")
    else:
        match_results = {}
        print(f"Results path {_results_path} does not exist yet. Generate requests and submit batch first.")
    return load_matches_from_folder, match_results


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
