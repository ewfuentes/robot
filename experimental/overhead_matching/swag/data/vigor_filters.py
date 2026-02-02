"""Filters for VIGOR dataset that can be applied after dataset creation."""

import logging
from pathlib import Path

from experimental.overhead_matching.swag.data.vigor_dataset import (
    VigorDataset,
    compute_proper_noun_matches,
    load_pano_gemini_proper_nouns,
)

logger = logging.getLogger(__name__)


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

    # Build filtered pair lists and determine which panoramas to keep
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

        if filtered_pos or filtered_semipos:
            panos_to_keep.append(pano_idx)
            new_positive_sat_idxs.append(filtered_pos)
            new_semipositive_sat_idxs.append(filtered_semipos)

    logger.info(f"Proper noun filter: keeping {len(panos_to_keep)}/{original_pano_count} panoramas")

    # Apply the filter using drop_to_subset
    dataset.drop_to_subset(
        new_positive_sat_idxs_per_pano=new_positive_sat_idxs,
        new_semipositive_sat_idxs_per_pano=new_semipositive_sat_idxs,
        pano_idxs_to_keep=panos_to_keep,
    )
