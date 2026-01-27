"""Generate spoofed pano-OSM datasets for testing place name generalization.

Creates datasets where each panorama has unique (business_name, address) pairs,
and each satellite aggregates all combos from its matching panoramas.

Output format matches existing pano_v1 and v4_202001_no_addresses datasets.
"""
import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 - must import before torch
import numpy as np
import openai
import pandas as pd
import torch
import tqdm

import geopandas as gpd
from shapely.geometry import Point

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    VigorDatasetConfig,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    custom_id_from_props,
)


# Sentence templates
PANO_BUSINESS_TEMPLATE = (
    "A single-story brick building with large street-facing windows "
    "with a sign for {business_name}"
)
PANO_ADDRESS_TEMPLATE = "A street sign with the address {address}"
OSM_TEMPLATE = "{business_name} location on {address}"

# OpenAI embedding config
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 2048  # OpenAI's max inputs per request


def load_source_data(
    source_path: Path, additional_names_path: Path | None = None
) -> tuple[list[str], list[str]]:
    """Load business names and places from source data.

    Args:
        source_path: Path to dataset.pkl with train/test DataFrames
        additional_names_path: Optional JSON file with additional things/places

    Returns:
        Tuple of (list of things, list of places)
    """
    with open(source_path, "rb") as f:
        data = pickle.load(f)

    # Combine all unique things and places from both splits
    all_things = set()
    all_places = set()
    for split_name in ["train", "test"]:
        if split_name in data:
            all_things.update(data[split_name]["thing"].unique())
            all_places.update(data[split_name]["place"].unique())

    # Add additional names if provided
    if additional_names_path is not None:
        with open(additional_names_path, "r") as f:
            additional = json.load(f)
        all_things.update(additional.get("things", []))
        all_places.update(additional.get("places", []))

    return list(all_things), list(all_places)


def split_things_and_places_by_city(
    things: list[str], places: list[str], cities: list[str], seed: int = 42
) -> dict[str, tuple[list[str], list[str]]]:
    """Split things and places 50/50 between cities with no overlap.

    Args:
        things: List of business names
        places: List of street names
        cities: List of city names (first is train, second is test)
        seed: Random seed for reproducibility

    Returns:
        Dict mapping city -> (things_list, places_list)
    """
    rng = np.random.RandomState(seed)

    things = list(things)
    places = list(places)
    rng.shuffle(things)
    rng.shuffle(places)

    mid_things = len(things) // 2
    mid_places = len(places) // 2

    result = {}
    result[cities[0]] = (things[:mid_things], places[:mid_places])
    result[cities[1]] = (things[mid_things:], places[mid_places:])

    # Verify no overlap
    assert set(result[cities[0]][0]).isdisjoint(set(result[cities[1]][0]))
    assert set(result[cities[0]][1]).isdisjoint(set(result[cities[1]][1]))

    return result


def generate_street_numbers(
    places: list[str], seed: int = 42
) -> dict[str, int]:
    """Generate random street numbers for each place.

    Args:
        places: List of street names
        seed: Random seed

    Returns:
        Dict mapping place -> street number (100-9999)
    """
    rng = np.random.RandomState(seed)
    return {place: rng.randint(100, 10000) for place in places}


def assign_combos_to_panoramas(
    pano_metadata: pd.DataFrame,
    things: list[str],
    places: list[str],
    street_numbers: dict[str, int],
    seed: int = 42,
) -> dict[int, tuple[str, str]]:
    """Assign unique (business_name, address) to each panorama.

    Args:
        pano_metadata: DataFrame with panorama metadata
        things: List of business names for this city
        places: List of street names for this city
        street_numbers: Dict mapping place -> street number
        seed: Random seed

    Returns:
        Dict mapping pano_idx -> (business_name, address)
    """
    rng = np.random.RandomState(seed)
    num_panos = len(pano_metadata)

    # Generate all possible (thing, place) combinations
    # We need at least num_panos unique combinations
    combos = []
    for thing in things:
        for place in places:
            number = street_numbers.get(place, rng.randint(100, 10000))
            address = f"{number} {place}"
            combos.append((thing, address))

    if len(combos) < num_panos:
        raise ValueError(
            f"Not enough unique combos ({len(combos)}) for {num_panos} panoramas. "
            f"Need more things ({len(things)}) or places ({len(places)})."
        )

    # Shuffle and assign
    rng.shuffle(combos)
    pano_to_combo = {}
    for idx, pano_idx in enumerate(pano_metadata.index):
        pano_to_combo[pano_idx] = combos[idx]

    return pano_to_combo


def aggregate_combos_for_satellites(
    pano_metadata: pd.DataFrame, pano_to_combo: dict[int, tuple[str, str]]
) -> dict[int, list[tuple[str, str]]]:
    """Aggregate all panorama combos for each satellite.

    Args:
        pano_metadata: DataFrame with panorama metadata including positive_satellite_idxs
        pano_to_combo: Dict mapping pano_idx -> (business_name, address)

    Returns:
        Dict mapping sat_idx -> list of (business_name, address) tuples
    """
    sat_to_combos = defaultdict(list)

    for pano_idx, row in pano_metadata.iterrows():
        if pano_idx not in pano_to_combo:
            continue
        combo = pano_to_combo[pano_idx]
        for sat_idx in row["positive_satellite_idxs"]:
            if combo not in sat_to_combos[sat_idx]:
                sat_to_combos[sat_idx].append(combo)

    return dict(sat_to_combos)


def generate_pano_custom_id(
    pano_id: str, lat: float, lon: float, landmark_idx: int
) -> str:
    """Generate custom ID for panorama landmark.

    Format: '{pano_id},{lat:.6f},{lon:.6f},__landmark_{idx}'
    """
    return f"{pano_id},{lat:.6f},{lon:.6f},__landmark_{landmark_idx}"


def generate_osm_custom_id(business_name: str, address: str) -> str:
    """Generate custom ID for OSM landmark using SHA256 hash."""
    props = {"name": business_name, "addr:street": address}
    return custom_id_from_props(props)


def create_openai_batch_response_format(custom_id: str, content: str) -> dict:
    """Create a fake OpenAI batch API response format for sentence storage.

    This format matches what make_sentence_dict_from_json expects.

    Args:
        custom_id: The custom ID for the response
        content: The sentence content (plain text for OSM, JSON string for pano)

    Returns:
        Dict in OpenAI batch API response format
    """
    return {
        "id": f"spoofed_{custom_id[:16]}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"spoofed_{custom_id[:16]}",
            "body": {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 0},
            },
        },
        "error": None,
    }


def get_embeddings_batch(
    client: openai.OpenAI, sentences: list[str], max_retries: int = 5
) -> list[list[float]]:
    """Get embeddings for a batch of sentences with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=sentences,
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            wait_time = 2**attempt
            print(f"Rate limited, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2**attempt
            print(f"API error: {e}, retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise RuntimeError("Max retries exceeded")


def generate_all_embeddings(
    sentences: list[str], client: openai.OpenAI
) -> np.ndarray:
    """Generate embeddings for all sentences.

    Args:
        sentences: List of sentences to embed
        client: OpenAI client

    Returns:
        numpy array of shape (len(sentences), EMBEDDING_DIM)
    """
    all_embeddings = []

    for i in tqdm.tqdm(
        range(0, len(sentences), BATCH_SIZE), desc="Generating embeddings"
    ):
        batch = sentences[i : i + BATCH_SIZE]
        embeddings = get_embeddings_batch(client, batch)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def save_pano_embeddings(
    output_path: Path,
    pano_metadata: pd.DataFrame,
    pano_to_combo: dict[int, tuple[str, str]],
    embeddings_array: np.ndarray,
    sentence_to_idx: dict[str, int],
):
    """Save panorama embeddings and metadata.

    Args:
        output_path: Base output path for this city
        pano_metadata: DataFrame with panorama metadata
        pano_to_combo: Dict mapping pano_idx -> (business_name, address)
        embeddings_array: All embeddings as numpy array
        sentence_to_idx: Mapping from sentence to index in embeddings_array
    """
    embeddings_dir = output_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    requests_dir = output_path / "embedding_requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    # Build landmark ID to embedding index mapping
    landmark_id_to_idx = {}
    pano_landmark_metadata = []

    for pano_idx, row in pano_metadata.iterrows():
        if pano_idx not in pano_to_combo:
            continue

        business_name, address = pano_to_combo[pano_idx]
        pano_id = row["pano_id"]
        lat = row["lat"]
        lon = row["lon"]

        # Landmark 0: business name
        business_sentence = PANO_BUSINESS_TEMPLATE.format(
            business_name=business_name
        )
        business_custom_id = generate_pano_custom_id(pano_id, lat, lon, 0)
        landmark_id_to_idx[business_custom_id] = sentence_to_idx[
            business_sentence
        ]
        panorama_id = f"{pano_id},{lat:.6f},{lon:.6f},"
        pano_landmark_metadata.append(
            {
                "custom_id": business_custom_id,
                "panorama_id": panorama_id,
                "landmark_idx": 0,
                "yaw_angles": [0, 90, 180, 270],
            }
        )

        # Landmark 1: address
        address_sentence = PANO_ADDRESS_TEMPLATE.format(address=address)
        address_custom_id = generate_pano_custom_id(pano_id, lat, lon, 1)
        landmark_id_to_idx[address_custom_id] = sentence_to_idx[
            address_sentence
        ]
        pano_landmark_metadata.append(
            {
                "custom_id": address_custom_id,
                "panorama_id": panorama_id,
                "landmark_idx": 1,
                "yaw_angles": [0, 90, 180, 270],
            }
        )

    # Create embeddings tensor with proper indexing
    embedding_indices = list(landmark_id_to_idx.values())
    embeddings_tensor = torch.tensor(
        embeddings_array[embedding_indices], dtype=torch.float32
    )

    # Remap indices to be consecutive
    new_landmark_id_to_idx = {
        lid: new_idx for new_idx, lid in enumerate(landmark_id_to_idx.keys())
    }

    # Save embeddings.pkl
    with open(embeddings_dir / "embeddings.pkl", "wb") as f:
        pickle.dump((embeddings_tensor, new_landmark_id_to_idx), f)

    # Save panorama_metadata.jsonl
    with open(requests_dir / "panorama_metadata.jsonl", "w") as f:
        for metadata in pano_landmark_metadata:
            f.write(json.dumps(metadata) + "\n")

    # Save sentences.jsonl
    sentences_dir = output_path / "sentences"
    sentences_dir.mkdir(parents=True, exist_ok=True)

    with open(sentences_dir / "sentences.jsonl", "w") as f:
        for pano_idx, row in pano_metadata.iterrows():
            if pano_idx not in pano_to_combo:
                continue

            business_name, address = pano_to_combo[pano_idx]
            pano_id = row["pano_id"]
            lat = row["lat"]
            lon = row["lon"]
            panorama_id = f"{pano_id},{lat:.6f},{lon:.6f},"

            business_sentence = PANO_BUSINESS_TEMPLATE.format(
                business_name=business_name
            )
            address_sentence = PANO_ADDRESS_TEMPLATE.format(address=address)

            # Create landmarks JSON content (matching pano_v1 format)
            content = json.dumps(
                {
                    "landmarks": [
                        {
                            "description": business_sentence,
                            "yaw_angles": [0, 90, 180, 270],
                        },
                        {
                            "description": address_sentence,
                            "yaw_angles": [0, 90, 180, 270],
                        },
                    ]
                }
            )
            response = create_openai_batch_response_format(panorama_id, content)
            f.write(json.dumps(response) + "\n")

    print(f"Saved {len(new_landmark_id_to_idx)} pano landmarks to {output_path}")


def save_osm_embeddings(
    output_path: Path,
    sat_to_combos: dict[int, list[tuple[str, str]]],
    embeddings_array: np.ndarray,
    sentence_to_idx: dict[str, int],
):
    """Save OSM embeddings and metadata.

    Args:
        output_path: Base output path for this city
        sat_to_combos: Dict mapping sat_idx -> list of (business_name, address)
        embeddings_array: All embeddings as numpy array
        sentence_to_idx: Mapping from sentence to index in embeddings_array
    """
    embeddings_dir = output_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Build landmark ID to embedding index mapping
    landmark_id_to_idx = {}
    sat_landmark_metadata = []

    for sat_idx, combos in sat_to_combos.items():
        sat_landmarks = []
        for business_name, address in combos:
            osm_sentence = OSM_TEMPLATE.format(
                business_name=business_name, address=address
            )
            osm_custom_id = generate_osm_custom_id(business_name, address)

            if osm_custom_id not in landmark_id_to_idx:
                landmark_id_to_idx[osm_custom_id] = sentence_to_idx[
                    osm_sentence
                ]
            sat_landmarks.append(osm_custom_id)

        sat_landmark_metadata.append(
            {"sat_idx": int(sat_idx), "landmark_custom_ids": sat_landmarks}
        )

    # Create embeddings tensor with proper indexing
    embedding_indices = list(landmark_id_to_idx.values())
    embeddings_tensor = torch.tensor(
        embeddings_array[embedding_indices], dtype=torch.float32
    )

    # Remap indices to be consecutive
    new_landmark_id_to_idx = {
        lid: new_idx for new_idx, lid in enumerate(landmark_id_to_idx.keys())
    }

    # Save embeddings.pkl
    with open(embeddings_dir / "embeddings.pkl", "wb") as f:
        pickle.dump((embeddings_tensor, new_landmark_id_to_idx), f)

    # Save satellite_metadata.jsonl
    with open(output_path / "satellite_metadata.jsonl", "w") as f:
        for metadata in sat_landmark_metadata:
            f.write(json.dumps(metadata) + "\n")

    print(f"Saved {len(new_landmark_id_to_idx)} OSM landmarks to {output_path}")


def save_merged_osm_embeddings(
    output_path: Path,
    all_sat_to_combos: dict[str, dict[int, list[tuple[str, str]]]],
    embeddings_array: np.ndarray,
    sentence_to_idx: dict[str, int],
):
    """Save merged OSM embeddings and metadata across all cities.

    Args:
        output_path: Base output path (e.g., output_base / "osm_spoofed")
        all_sat_to_combos: Dict mapping city -> {sat_idx -> [(business_name, address), ...]}
        embeddings_array: All embeddings as numpy array
        sentence_to_idx: Mapping from sentence to index in embeddings_array
    """
    embeddings_dir = output_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    sentences_dir = output_path / "sentences"
    sentences_dir.mkdir(parents=True, exist_ok=True)

    # Build merged landmark ID to embedding index mapping
    landmark_id_to_idx = {}
    sat_landmark_metadata = []
    sentence_data = []
    seen_custom_ids = set()

    for city, sat_to_combos in all_sat_to_combos.items():
        for sat_idx, combos in sat_to_combos.items():
            sat_landmarks = []
            for business_name, address in combos:
                osm_sentence = OSM_TEMPLATE.format(
                    business_name=business_name, address=address
                )
                osm_custom_id = generate_osm_custom_id(business_name, address)

                if osm_custom_id not in landmark_id_to_idx:
                    landmark_id_to_idx[osm_custom_id] = sentence_to_idx[osm_sentence]

                if osm_custom_id not in seen_custom_ids:
                    seen_custom_ids.add(osm_custom_id)
                    sentence_data.append(
                        create_openai_batch_response_format(osm_custom_id, osm_sentence)
                    )

                sat_landmarks.append(osm_custom_id)

            sat_landmark_metadata.append(
                {
                    "sat_idx": int(sat_idx),
                    "landmark_custom_ids": sat_landmarks,
                    "city": city,
                }
            )

    # Create embeddings tensor with proper indexing
    embedding_indices = list(landmark_id_to_idx.values())
    embeddings_tensor = torch.tensor(
        embeddings_array[embedding_indices], dtype=torch.float32
    )

    # Remap indices to be consecutive
    new_landmark_id_to_idx = {
        lid: new_idx for new_idx, lid in enumerate(landmark_id_to_idx.keys())
    }

    # Save embeddings.pkl
    with open(embeddings_dir / "embeddings.pkl", "wb") as f:
        pickle.dump((embeddings_tensor, new_landmark_id_to_idx), f)

    # Save satellite_metadata.jsonl
    with open(output_path / "satellite_metadata.jsonl", "w") as f:
        for metadata in sat_landmark_metadata:
            f.write(json.dumps(metadata) + "\n")

    # Save sentences.jsonl
    with open(sentences_dir / "sentences.jsonl", "w") as f:
        for sentence_entry in sentence_data:
            f.write(json.dumps(sentence_entry) + "\n")

    print(f"Saved {len(new_landmark_id_to_idx)} merged OSM landmarks to {output_path}")


def save_pano_v2_format(
    output_path: Path,
    pano_metadata: pd.DataFrame,
    pano_to_combo: dict[int, tuple[str, str]],
):
    """Save panorama data in v2.0 pickle format for CLIPTextLandmarkExtractor.

    The v2.0 format is:
    {
        "version": "2.0",
        "panoramas": {
            "pano_id": {
                "landmarks": [
                    {"description": "...", "landmark_idx": 0, "bounding_boxes": [...]}
                ]
            }
        }
    }

    Args:
        output_path: Base output path for this city (e.g., output_base/pano_spoofed_v2/Chicago)
        pano_metadata: DataFrame with panorama metadata
        pano_to_combo: Dict mapping pano_idx -> (business_name, address)
    """
    embeddings_dir = output_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    panoramas = {}

    for pano_idx, row in pano_metadata.iterrows():
        if pano_idx not in pano_to_combo:
            continue

        business_name, address = pano_to_combo[pano_idx]
        pano_id = row["pano_id"]

        business_sentence = PANO_BUSINESS_TEMPLATE.format(business_name=business_name)
        address_sentence = PANO_ADDRESS_TEMPLATE.format(address=address)

        landmarks = [
            {
                "description": business_sentence,
                "landmark_idx": 0,
                "bounding_boxes": [
                    {"yaw_angle": "0"},
                    {"yaw_angle": "90"},
                    {"yaw_angle": "180"},
                    {"yaw_angle": "270"},
                ],
            },
            {
                "description": address_sentence,
                "landmark_idx": 1,
                "bounding_boxes": [
                    {"yaw_angle": "0"},
                    {"yaw_angle": "90"},
                    {"yaw_angle": "180"},
                    {"yaw_angle": "270"},
                ],
            },
        ]

        panoramas[pano_id] = {"landmarks": landmarks}

    output = {"version": "2.0", "panoramas": panoramas}

    with open(embeddings_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(output, f)

    print(f"Saved {len(panoramas)} panoramas in v2.0 format to {output_path}")


def save_landmarks_feather(
    output_path: Path,
    pano_metadata: pd.DataFrame,
    pano_to_combo: dict[int, tuple[str, str]],
    landmark_version: str,
):
    """Save spoofed landmarks as a VIGOR-compatible feather file.

    Creates Point landmarks at each panorama's location with the spoofed
    business name and address. These will be spatially matched to satellites
    by compute_satellite_from_landmarks().

    Args:
        output_path: Path to VIGOR dataset directory (e.g., /data/.../VIGOR/Chicago)
        pano_metadata: DataFrame with panorama metadata (lat, lon)
        pano_to_combo: Dict mapping pano_idx -> (business_name, address)
        landmark_version: Version string for the feather file (e.g., "spoofed_v1")
    """
    landmarks_dir = output_path / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # Build landmark records
    # Only include panoramas that have at least one positive satellite match,
    # since those are the only ones that will have embeddings generated.
    records = []
    for pano_idx, row in pano_metadata.iterrows():
        if pano_idx not in pano_to_combo:
            continue
        # Skip panoramas without satellite matches - they won't have embeddings
        if len(row["positive_satellite_idxs"]) == 0:
            continue

        business_name, address = pano_to_combo[pano_idx]
        lat = row["lat"]
        lon = row["lon"]

        records.append({
            "id": f"('spoofed', {pano_idx})",
            "geometry": Point(lon, lat),  # Note: Point takes (lon, lat)
            "landmark_type": "spoofed",
            "name": business_name,
            "addr:street": address,
        })

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    # Save to feather
    feather_path = landmarks_dir / f"{landmark_version}.feather"
    gdf.to_feather(feather_path)

    print(f"Saved {len(records)} landmarks to {feather_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spoofed pano-OSM datasets"
    )
    parser.add_argument(
        "--vigor_base",
        type=str,
        required=True,
        help="Path to VIGOR dataset base directory",
    )
    parser.add_argument(
        "--source_data",
        type=str,
        required=True,
        help="Path to spoofed_place_location_pairs/dataset.pkl",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Output directory for generated datasets",
    )
    parser.add_argument(
        "--additional_names",
        type=str,
        default=None,
        help="Optional JSON file with additional things/places",
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=["Chicago", "Seattle"],
        help="Cities to process (first is train, second is test)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--landmark_version",
        type=str,
        default="spoofed_v1",
        help="Version string for the landmarks feather file",
    )
    args = parser.parse_args()

    vigor_base = Path(args.vigor_base)
    source_path = Path(args.source_data)
    output_base = Path(args.output_base)
    additional_names = (
        Path(args.additional_names) if args.additional_names else None
    )

    # Load source data
    print("Loading source data...")
    all_things, all_places = load_source_data(source_path, additional_names)
    print(f"Loaded {len(all_things)} things and {len(all_places)} places")

    # Split by city
    print("Splitting things/places by city...")
    city_splits = split_things_and_places_by_city(
        all_things, all_places, args.cities, args.seed
    )
    for city, (things, places) in city_splits.items():
        print(f"  {city}: {len(things)} things, {len(places)} places")

    # Generate street numbers
    street_numbers = generate_street_numbers(all_places, args.seed)

    # Collect all sentences for embedding
    all_sentences = set()
    city_data = {}

    for city in args.cities:
        print(f"\nProcessing {city}...")
        things, places = city_splits[city]

        # Load VIGOR dataset
        config = VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=False,
        )
        dataset = VigorDataset(
            dataset_path=vigor_base / city,
            config=config,
        )

        # Assign combos to panoramas
        pano_to_combo = assign_combos_to_panoramas(
            dataset._panorama_metadata,
            things,
            places,
            street_numbers,
            seed=args.seed + hash(city) % 1000,
        )

        # Filter out panoramas without satellite matches - they can't be used for training
        # and won't have corresponding OSM embeddings
        panos_with_satellites = {
            pano_idx
            for pano_idx, row in dataset._panorama_metadata.iterrows()
            if len(row["positive_satellite_idxs"]) > 0
        }
        original_count = len(pano_to_combo)
        pano_to_combo = {
            k: v for k, v in pano_to_combo.items() if k in panos_with_satellites
        }
        filtered_count = original_count - len(pano_to_combo)
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} panoramas without satellite matches")

        # Aggregate for satellites
        sat_to_combos = aggregate_combos_for_satellites(
            dataset._panorama_metadata, pano_to_combo
        )

        print(
            f"  {len(pano_to_combo)} panoramas, {len(sat_to_combos)} satellites with landmarks"
        )

        # Collect sentences
        for business_name, address in pano_to_combo.values():
            all_sentences.add(
                PANO_BUSINESS_TEMPLATE.format(business_name=business_name)
            )
            all_sentences.add(PANO_ADDRESS_TEMPLATE.format(address=address))
            all_sentences.add(
                OSM_TEMPLATE.format(business_name=business_name, address=address)
            )

        city_data[city] = {
            "pano_metadata": dataset._panorama_metadata,
            "pano_to_combo": pano_to_combo,
            "sat_to_combos": sat_to_combos,
        }

    # Generate embeddings for all unique sentences
    all_sentences = list(all_sentences)
    print(f"\nGenerating embeddings for {len(all_sentences)} unique sentences...")

    client = openai.OpenAI()
    embeddings_array = generate_all_embeddings(all_sentences, client)
    sentence_to_idx = {s: i for i, s in enumerate(all_sentences)}

    print(f"Embeddings shape: {embeddings_array.shape}")

    # Save pano outputs for each city
    for city in args.cities:
        print(f"\nSaving {city} pano outputs...")
        data = city_data[city]

        # Pano embeddings (per city) - old format for quatro_robot
        pano_output = output_base / "pano_spoofed" / city
        save_pano_embeddings(
            pano_output,
            data["pano_metadata"],
            data["pano_to_combo"],
            embeddings_array,
            sentence_to_idx,
        )

        # Pano v2.0 format (per city) - for third_robot CLIPTextLandmarkExtractor
        pano_v2_output = output_base / "pano_spoofed_v2" / city
        save_pano_v2_format(
            pano_v2_output,
            data["pano_metadata"],
            data["pano_to_combo"],
        )

        # Landmarks feather file (for SemanticLandmarkExtractor compatibility)
        save_landmarks_feather(
            vigor_base / city,
            data["pano_metadata"],
            data["pano_to_combo"],
            args.landmark_version,
        )

    # Save merged OSM outputs (all cities combined)
    print("\nSaving merged OSM outputs...")
    osm_output = output_base / "osm_spoofed"
    all_sat_to_combos = {city: city_data[city]["sat_to_combos"] for city in args.cities}
    save_merged_osm_embeddings(
        osm_output,
        all_sat_to_combos,
        embeddings_array,
        sentence_to_idx,
    )

    # Save metadata
    metadata_dir = output_base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Split summary
    split_summary = {
        city: {"things": list(things), "places": list(places)}
        for city, (things, places) in city_splits.items()
    }
    with open(metadata_dir / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    # Per-city pair mappings
    for city in args.cities:
        data = city_data[city]
        pairs = [
            {
                "pano_idx": int(pano_idx),
                "business_name": combo[0],
                "address": combo[1],
            }
            for pano_idx, combo in data["pano_to_combo"].items()
        ]
        with open(metadata_dir / f"{city.lower()}_pairs.json", "w") as f:
            json.dump(pairs, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
