"""Utilities for semantic landmark extraction.

This module contains shared functions for loading and processing semantic landmark data
that are used across multiple extractor modules.
"""

import base64
import hashlib
import json
import pickle
import hashlib
import base64
from pathlib import Path
import pandas as pd
import torch


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
        elif file.suffix == ".pkl":
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
        Dictionary mapping custom_id to embedding vector (as lists)
    """
    out = {}
    for response in embedding_jsons:
        assert response["error"] == None
        custom_id = response["custom_id"]
        embedding = response["response"]["body"]["data"][0]["embedding"]
        assert custom_id not in out
        out[custom_id] = embedding
    return out


def convert_embeddings_to_tensors(embedding_dict: dict[str, list[float]]):
    """Convert embedding dict values from lists to tensors.

    Args:
        embedding_dict: Dictionary mapping custom_id to embedding vector (as lists)

    Returns:
        Dictionary mapping custom_id to embedding tensor
    """
    return {
        custom_id: torch.tensor(embedding, dtype=torch.float32)
        for custom_id, embedding in embedding_dict.items()
    }


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


def make_sentence_dict_from_pano_jsons(sentence_jsons: list) -> tuple[dict[str, str], dict, int]:
    """Create a dictionary mapping landmark_custom_id to sentence description. For panorama outputs where a json object is returned by OpenAI

    Returns:
        tuple of (sentences_dict, metadata_by_pano_id, output_tokens) where:
        - sentences_dict: maps landmark_custom_id to description string
        - metadata_by_pano_id: maps panorama_id to list of landmark metadata dicts
        - output_tokens: total number of output tokens used
    """
    pano_out = {}
    metadata = {}  # Maps pano_id -> list of landmarks
    out, output_tokens = make_sentence_dict_from_json(sentence_jsons)
    for custom_id, content_str in out.items():
        try:
            content = json.loads(content_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {custom_id}: {e}")
            continue

        landmarks = content.get("landmarks", [])

        # Create entries for each landmark in this panorama
        panorama_id = custom_id

        if panorama_id not in metadata:
            metadata[panorama_id] = []

        for idx, landmark in enumerate(landmarks):
            description = landmark["description"]
            yaw_angles = landmark["yaw_angles"]
            if not description:
                raise RuntimeError(f"No description! {landmark}")

            landmark_custom_id = f"{panorama_id}__landmark_{idx}"
            pano_out[landmark_custom_id] = description

            # Append to metadata list for this panorama
            metadata[panorama_id].append({
                "landmark_idx": idx,
                "custom_id": landmark_custom_id,
                "yaw_angles": yaw_angles
            })
    return pano_out, metadata, output_tokens



_TAGS_TO_KEEP = [
    "access",
    "addr:housenumber",
    "addr:street",
    "advertising",
    "amenity",
    "architect",
    "architecture",
    "artwork",
    "artwork_type",
    "banner",
    "barrier",
    "beauty",
    "bicycle_parking",
    "brand",
    "brewery",
    "bridge",
    "bridge:structure",
    "bridge:type",
    "building",
    "building:architecture",
    "building:levels",
    "building:part",
    "building:use",
    "construction",
    "covered",
    "craft",
    "crane:type",
    "crossing",
    "crossing:barrier",
    "cuisine",
    "cycleway",
    "denomination",
    "description",
    "design",
    "drive_through",
    "embankment",
    "emergency",
    "entrance",
    "facade",
    "fence",
    "fence_type",
    "fenced",
    "fire_escape",
    "fire_hydrant:type",
    "footway",
    "garden:type",
    "handrail",
    "healthcare",
    "heritage",
    "highway",
    "historic",
    "industrial",
    "information",
    "inscription",
    "junction",
    "lamp_mount",
    "lamp_type",
    "landuse",
    "lanes",
    "leisure",
    "levels",
    "light_source",
    "man_made",
    "mast:type",
    "maxheight",
    "maxspeed",
    "memorial",
    "mural",
    "name",
    "natural",
    "office",
    "oneway",
    "operator",
    "outdoor_seating",
    "parking",
    "phone",
    "power",
    "public_transport",
    "railway",
    "recycling_type",
    "ref",
    "religion",
    "residential",
    "roof:shape",
    "service",
    "shelter",
    "shelter_type",
    "shop",
    "short_name",
    "social_facility",
    "sport",
    "station",
    "street_lamp",
    "style",
    "substation",
    "support",
    "theatre:type",
    "tourism",
    "tower:type",
    "traffic_calming",
    "traffic_sign",
    "tunnel",
    "type",
    "vending",
    "wall",
    "waste",
    "water",
    "waterway",
]

# Entries ending with ':' are prefix matches (e.g. "building:" matches "building:levels").
# All others are exact key matches.
_TAGS_TO_KEEP_PREFIXES = [t for t in _TAGS_TO_KEEP if t.endswith(':')]
_TAGS_TO_KEEP_SET = frozenset(t for t in _TAGS_TO_KEEP if not t.endswith(':'))


def prune_landmark(props):
    """Keep only tags relevant for landmark matching.

    Retains only tags that describe features visible from street level and
    useful for matching between panorama observations and OSM database entries.

    Args:
        props: Dictionary of landmark properties

    Returns:
        Frozenset of (key, value) tuples for relevant properties
    """
    out = set()
    for (k, v) in props.items():
        if k not in _TAGS_TO_KEEP_SET and not any(k.startswith(p) for p in _TAGS_TO_KEEP_PREFIXES):
            continue
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            continue
        out.add((k, v))
    return frozenset(out)


def load_embeddings_from_pickle(embedding_path: Path) -> tuple[torch.Tensor, dict[str, int]]:
    """Load embeddings from a pickle file.

    Args:
        embedding_path: Path to the embeddings.pkl file

    Returns:
        embeddings_tensor: (num_landmarks, embedding_dim) tensor of embeddings
        landmark_id_to_idx: Dict mapping landmark custom_id to tensor row index
    """
    with open(embedding_path, 'rb') as f:
        embeddings_tensor, landmark_id_to_idx = pickle.load(f)
    return embeddings_tensor, landmark_id_to_idx


def load_embeddings_from_jsonl(embedding_directory: Path) -> tuple[torch.Tensor, dict[str, int]]:
    """Load embeddings from JSONL files in a directory.

    Args:
        embedding_directory: Path to directory containing embedding JSONL files

    Returns:
        embeddings_tensor: (num_landmarks, embedding_dim) tensor of embeddings
        landmark_id_to_idx: Dict mapping landmark custom_id to tensor row index
    """
    embeddings = convert_embeddings_to_tensors(
        make_embedding_dict_from_json(
            load_all_jsonl_from_folder(embedding_directory)))

    embedding_list = []
    landmark_id_to_idx = {}
    for idx, (landmark_id, emb) in enumerate(embeddings.items()):
        embedding_list.append(emb)
        landmark_id_to_idx[landmark_id] = idx

    embeddings_tensor = torch.stack(embedding_list)
    return embeddings_tensor, landmark_id_to_idx


def load_embeddings(
    embedding_directory: Path,
    output_dim: int | None = None,
    normalize: bool = False
) -> tuple[torch.Tensor, dict[str, int]]:
    """Load embeddings from a directory, handling both pickle and JSONL formats.

    Prefers pickle format if embeddings.pkl exists, otherwise loads from JSONL.

    Args:
        embedding_directory: Path to directory containing embeddings
        output_dim: If specified, truncate embeddings to this dimension
        normalize: If True, normalize embeddings to unit length

    Returns:
        embeddings_tensor: (num_landmarks, embedding_dim) tensor of embeddings
        landmark_id_to_idx: Dict mapping landmark custom_id to tensor row index
    """
    pickle_path = embedding_directory / "embeddings.pkl"
    if pickle_path.exists():
        embeddings, landmark_id_to_idx = load_embeddings_from_pickle(pickle_path)
    else:
        embeddings, landmark_id_to_idx = load_embeddings_from_jsonl(embedding_directory)

    if output_dim is not None and embeddings.shape[1] > output_dim:
        embeddings = embeddings[:, :output_dim]

    if normalize:
        embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)

    return embeddings, landmark_id_to_idx


def custom_id_from_props(props: dict | frozenset) -> str:
    """Generate a custom ID from landmark properties using SHA256 hash.

    Args:
        props: Dictionary or frozenset of (key, value) tuples representing landmark properties

    Returns:
        Base64-encoded SHA256 hash of the JSON-serialized properties
    """
    if isinstance(props, frozenset):
        props = dict(props)
    json_props = json.dumps(props, sort_keys=True)
    custom_id = base64.b64encode(
        hashlib.sha256(json_props.encode('utf-8')).digest()
    ).decode('utf-8')
    return custom_id

