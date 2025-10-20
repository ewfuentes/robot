"""Utilities for semantic landmark extraction.

This module contains shared functions for loading and processing semantic landmark data
that are used across multiple extractor modules.
"""

import json
from pathlib import Path
import pandas as pd


def load_all_jsonl_from_folder(folder: Path) -> list:
    """Load all JSONL files from a folder.

    Args:
        folder: Path to folder containing JSONL files

    Returns:
        List of JSON objects from all files
    """
    all_json_objs = []
    for file in folder.glob("*"):
        if file.is_dir():
            continue
        with open(file, 'r') as f:
            for line in f:
                all_json_objs.append(json.loads(line))
    return all_json_objs


def make_embedding_dict_from_json(embedding_jsons: list) -> dict[str, list[float]]:
    """Create a dictionary mapping custom_id to embedding vector.

    Args:
        embedding_jsons: List of JSON responses containing embeddings

    Returns:
        Dictionary mapping custom_id to embedding vector
    """
    out = {}
    for response in embedding_jsons:
        assert response["error"] == None
        custom_id = response["custom_id"]
        embedding = response["response"]["body"]["data"][0]["embedding"]
        assert custom_id not in out
        out[custom_id] = embedding
    return out


def make_sentence_dict_from_json(sentence_jsons: list) -> tuple[dict[str, str], int]:
    """Create a dictionary mapping custom_id to sentence description.

    Args:
        sentence_jsons: List of JSON responses containing sentence descriptions

    Returns:
        Tuple of (dict mapping custom_id to sentence, total output tokens)
    """
    out = {}
    output_tokens = 0
    for response in sentence_jsons:
        if len(response['response']['body']) == 0:
            print(f"GOT EMPTY RESPONSE {response}. SKIPPING")
            continue
        assert response["error"] == None and \
            response["response"]["body"]["choices"][0]["finish_reason"] == "stop" and \
            response["response"]["body"]["choices"][0]["message"]["refusal"] == None
        custom_id = response["custom_id"]
        sentence = response["response"]["body"]["choices"][0]["message"]["content"]
        assert custom_id not in out
        out[custom_id] = sentence
        output_tokens += response["response"]["body"]["usage"]["completion_tokens"]
    return out, output_tokens


def make_sentence_dict_from_pano_jsons(sentence_jsons: list) -> tuple[dict[str, str], int]:
    """Create a dictionary mapping landmark_custom_id to sentence description. For panorama outputs where a json object is returned by OpenAI"""
    pano_out = {}
    out, output_tokens = make_sentence_dict_from_json(sentence_jsons)
    for custom_id, content_str in out.items():
        try:
            content = json.loads(content_str)
            landmarks = content.get("landmarks", [])

            # Create entries for each landmark in this panorama
            panorama_id = custom_id
            for idx, landmark in enumerate(landmarks):
                description = landmark.get("description", "")
                if description:
                    landmark_custom_id = f"{panorama_id}__landmark_{idx}"
                    pano_out[landmark_custom_id] = description
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {custom_id}: {e}")
            continue

    return pano_out, output_tokens



def prune_landmark(props):
    """Prune landmark properties by removing unwanted fields.

    Removes metadata fields like indices, timestamps, and administrative data
    that aren't relevant for landmark descriptions.

    Args:
        props: Dictionary of landmark properties

    Returns:
        Frozenset of (key, value) tuples for relevant properties
    """
    to_drop = [
        "index",  # for props that come from a dataloader
        "web_mercator",
        "panorama_idxs",
        "satellite_idxs",
        "landmark_type",
        "element",
        "id",
        "geometry",
        "opening_hours",
        "website",
        "addr:city",
        "addr:state",
        'check_date',
        'checked_exists',
        'opening_date',
        'chicago:building_id',
        'survey:date',
        'payment',
        'disused',
        'time',
        'end_date']
    out = set()
    for (k, v) in props.items():
        should_add = True
        for prefix in to_drop:
            if k.startswith(prefix):
                should_add = False
                break
        if not should_add:
            continue
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            continue
        out.add((k, v))
    return frozenset(out)
