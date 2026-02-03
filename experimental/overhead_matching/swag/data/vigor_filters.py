"""Filters for VIGOR dataset that can be applied after dataset creation."""

import logging
import pickle
from pathlib import Path

import pandas as pd

from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset

logger = logging.getLogger(__name__)

# OSM text keys for proper noun matching
OSM_TEXT_KEYS = ['name', 'brand', 'operator', 'addr:street', 'network',
                 'alt_name', 'official_name', 'short_name', 'old_name', 'description']


def get_osm_text_from_landmark(landmark_dict: dict) -> list[str]:
    """Extract text fields from OSM landmark for matching."""
    texts = []
    for key in OSM_TEXT_KEYS:
        value = landmark_dict.get(key)
        if value and isinstance(value, str):
            texts.append(value)
    return texts


def check_proper_noun_match(proper_noun: str, osm_texts: list[str]) -> bool:
    """Check if proper_noun.lower() appears in any OSM text (case-insensitive)."""
    pn_lower = proper_noun.lower()
    return any(pn_lower in text.lower() for text in osm_texts)


def load_pano_gemini_proper_nouns(base_path: Path, city_names: list[str]) -> dict[str, list[str]]:
    """Load pano_gemini data and return dict mapping pano_id -> all proper nouns."""
    pano_proper_nouns = {}

    for city_name in city_names:
        pickle_path = base_path / city_name / "embeddings" / "embeddings.pkl"
        if not pickle_path.exists():
            logger.warning(f"pano_gemini not found: {pickle_path}")
            continue
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        if data.get('version') != '2.0':
            raise ValueError(f"Expected pano_gemini version '2.0', got '{data.get('version')}' in {pickle_path}")
        for pano_key, pano_data in data.get('panoramas', {}).items():
            pano_id = pano_key.split(',')[0]
            all_pns = []
            for lm in pano_data.get('landmarks', []):
                all_pns.extend(lm.get('proper_nouns', []))
            if all_pns:
                pano_proper_nouns[pano_id] = all_pns
    return pano_proper_nouns


def compute_proper_noun_matches(
    pano_metadata: pd.DataFrame,
    satellite_metadata: pd.DataFrame,
    landmark_metadata: pd.DataFrame,
    pano_proper_nouns: dict[str, list[str]]
) -> dict[int, set[int]]:
    """Return dict mapping pano_idx -> set of sat_idxs with proper noun matches."""
    matching_pairs = {}  # pano_idx -> set of sat_idxs

    for pano_idx, pano_row in pano_metadata.iterrows():
        pano_id = pano_row['pano_id']
        proper_nouns = pano_proper_nouns.get(pano_id, [])
        if not proper_nouns:
            continue

        matching_sats = set()
        for sat_idx in pano_row["positive_satellite_idxs"] + pano_row["semipositive_satellite_idxs"]:
            # Get OSM texts for THIS specific satellite
            osm_texts = []
            landmark_idxs = satellite_metadata.iloc[sat_idx].get('landmark_idxs', [])
            for lm_idx in landmark_idxs:
                lm_dict = landmark_metadata.iloc[lm_idx]['as_dict']
                osm_texts.extend(get_osm_text_from_landmark(dict(lm_dict)))

            # Check if any proper noun matches this satellite's OSM texts
            for pn in proper_nouns:
                if check_proper_noun_match(pn, osm_texts):
                    matching_sats.add(sat_idx)
                    break

        if matching_sats:
            matching_pairs[pano_idx] = matching_sats

    return matching_pairs


def apply_proper_noun_filter(
    dataset: VigorDataset,
    pano_gemini_base_path: Path,
) -> None:
    """Apply proper noun filtering to dataset in-place.

    This filter keeps only panorama-satellite pairs where a proper noun detected
    in the panorama image matches an OSM text field in a landmark visible in
    the satellite patch.

    Panoramas with no matching pairs after filtering are dropped entirely.

    Args:
        dataset: The VigorDataset to filter in-place.
        pano_gemini_base_path: Path to the pano_gemini data directory containing
            per-city embeddings with proper noun extractions.

    Raises:
        ValueError: If the dataset doesn't have landmarks loaded (required for
            proper noun matching).
    """
    if dataset._landmark_metadata is None:
        raise ValueError("apply_proper_noun_filter requires landmarks to be loaded in the dataset")

    city_names = list(set(dataset._panorama_metadata['dataset_key'].unique()))
    logger.info(f"Loading pano_gemini data for cities: {city_names}")

    pano_proper_nouns = load_pano_gemini_proper_nouns(pano_gemini_base_path, city_names)
    logger.info(f"Loaded proper nouns for {len(pano_proper_nouns)} panoramas")

    matching_pairs = compute_proper_noun_matches(
        dataset._panorama_metadata,
        dataset._satellite_metadata,
        dataset._landmark_metadata,
        pano_proper_nouns)

    original_pano_count = len(dataset._panorama_metadata)

    # Build filtered pair lists for ALL panoramas (original length)
    # and determine which panoramas to keep (those with at least one match)
    def filter_sat_idxs(sat_idxs, matching_set):
        return [idx for idx in sat_idxs if idx in matching_set]

    new_positive_sat_idxs = []
    new_semipositive_sat_idxs = []
    panos_to_keep = []

    for pano_idx in range(len(dataset._panorama_metadata)):
        pano_row = dataset._panorama_metadata.iloc[pano_idx]
        matching_sats = matching_pairs.get(pano_idx, set())

        filtered_pos = filter_sat_idxs(pano_row["positive_satellite_idxs"], matching_sats)
        filtered_semipos = filter_sat_idxs(pano_row["semipositive_satellite_idxs"], matching_sats)

        # Store filtered pairs for ALL panoramas (original length required by drop_to_subset)
        new_positive_sat_idxs.append(filtered_pos)
        new_semipositive_sat_idxs.append(filtered_semipos)

        # Track which panoramas have at least one match
        if filtered_pos or filtered_semipos:
            panos_to_keep.append(pano_idx)

    logger.info(f"Proper noun filter: keeping {len(panos_to_keep)}/{original_pano_count} panoramas")

    # Apply the filter using drop_to_subset
    dataset.drop_to_subset(
        new_positive_sat_idxs_per_pano=new_positive_sat_idxs,
        new_semipositive_sat_idxs_per_pano=new_semipositive_sat_idxs,
        pano_idxs_to_keep=panos_to_keep,
    )
