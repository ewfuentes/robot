"""OSM tag extraction similarity computation.

Loads Gemini OSM tag extraction predictions, joins them with VIGOR datasets,
and computes pano-to-satellite similarity matrices using keyed substring matching.
"""

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import common.torch.load_torch_deps
import torch
import tqdm

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation import proper_noun_matcher_python as pnm


class MatchTableSchema:
    """Shared enum types for match tables, usable in other DataFrames."""

    def __init__(self, pano_ids: list[str], tag_keys: list[str]):
        self.pano_id = pl.Enum(sorted(pano_ids))
        self.tag_key = pl.Enum(sorted(tag_keys))


def build_match_tables(q, t, k, pano_lm_tags, osm_tags, pano_lm_ranges,
                       osm_idxs_from_sat_idx) -> tuple[pl.DataFrame, pl.DataFrame, MatchTableSchema]:
    """Build normalized match tables for tag matching analysis.

    Returns:
        matches: One row per (pano_landmark, osm_landmark, tag) match.
            Columns: pano_id, pano_lm_idx, tag_key, osm_idx,
                     pano_lm_value, sat_lm_value
        osm_to_sat: Mapping from deduplicated OSM landmarks to sat patches.
            Columns: osm_idx, sat_idx, sat_lm_idx
        schema: Shared enum types for use in other DataFrames.

    Join on osm_idx to materialize matches for specific sat patches.
    """
    # Build osm_to_sat table
    osm_to_sat_rows = []
    for sat_idx, osm_idxs in osm_idxs_from_sat_idx.items():
        for local_idx, osm_idx in enumerate(osm_idxs):
            osm_to_sat_rows.append((osm_idx, sat_idx, local_idx))

    osm_to_sat = pl.DataFrame(
        osm_to_sat_rows,
        schema={"osm_idx": pl.Int32, "sat_idx": pl.Int32, "sat_lm_idx": pl.Int32},
        orient="row",
    )

    # Collect all tag keys across pano and osm landmarks
    all_tag_keys = set()
    for tags in pano_lm_tags:
        for key, _ in tags:
            all_tag_keys.add(key)
    for tags in osm_tags:
        for key, _ in tags:
            all_tag_keys.add(key)

    table_schema = MatchTableSchema(
        pano_ids=list(pano_lm_ranges.keys()),
        tag_keys=list(all_tag_keys),
    )
    match_schema = {
        "pano_id": table_schema.pano_id, "pano_lm_idx": pl.Int32,
        "tag_key": table_schema.tag_key,
        "osm_idx": pl.Int32, "pano_lm_value": pl.Utf8, "sat_lm_value": pl.Utf8,
    }
    if len(q) == 0:
        return pl.DataFrame(schema=match_schema), osm_to_sat, table_schema

    q_np = q.numpy() if isinstance(q, torch.Tensor) else q
    t_np = t.numpy() if isinstance(t, torch.Tensor) else t
    k_np = k.numpy() if isinstance(k, torch.Tensor) else k

    # Reverse mapping: global pano lm index -> (pano_id, local_idx)
    pano_id_per_lm = []
    local_idx_per_lm = []
    for pano_id, (start, end) in pano_lm_ranges.items():
        for i in range(end - start):
            pano_id_per_lm.append(pano_id)
            local_idx_per_lm.append(i)

    pano_ids = [pano_id_per_lm[qi] for qi in q_np]
    pano_lm_idxs = [local_idx_per_lm[qi] for qi in q_np]
    tag_keys = [pano_lm_tags[qi][ki][0] for qi, ki in zip(q_np, k_np)]
    pano_values = [pano_lm_tags[qi][ki][1] for qi, ki in zip(q_np, k_np)]
    osm_idxs = t_np.tolist()

    sat_values = []
    for qi, ti, ki in zip(q_np, t_np, k_np):
        pano_key = pano_lm_tags[qi][ki][0]
        sat_val = ""
        for ok, ov in osm_tags[ti]:
            if ok == pano_key:
                sat_val = ov
                break
        sat_values.append(sat_val)

    matches = pl.DataFrame(
        {
            "pano_id": pano_ids,
            "pano_lm_idx": pano_lm_idxs,
            "tag_key": tag_keys,
            "osm_idx": osm_idxs,
            "pano_lm_value": pano_values,
            "sat_lm_value": sat_values,
        },
        schema=match_schema,
    )

    return matches, osm_to_sat, table_schema


def load_osm_tag_extraction_jsonl(input_dir: Path) -> dict[str, dict]:
    """Load OSM tag extraction predictions from JSONL files.

    Args:
        input_dir: Directory containing prediction JSONL files (can be nested)

    Returns:
        Dict mapping pano_key ("pano_id,lat,lon,") to parsed prediction dict
    """
    panoramas = {}
    for jsonl_file in input_dir.rglob("*.jsonl"):
        for line in jsonl_file.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            pano_key = record["key"]  # "pano_id,lat,lon,"
            try:
                response_text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
                parsed = json.loads(response_text)
                panoramas[pano_key] = parsed
            except Exception:
                print(f"Failed to load response for {pano_key} from {jsonl_file}")
    return panoramas


def create_osm_tag_extraction_dataset(
    extraction_path: Path,
    vigor_dataset: vd.VigorDataset,
) -> dict:
    """Create dataset from OSM tag extraction, joined with VIGOR.

    Returns a unified dict matching the existing dataset interface used by
    create_vigor_datasets and create_proper_noun_datasets.

    Args:
        extraction_path: Path to the extraction output directory
        vigor_dataset: Loaded VigorDataset instance

    Returns:
        Dict with keys:
            - pano_data_from_pano_id: {pano_id: {"landmarks": [...], "sat_idxs": [...]}}
            - sat_data_from_sat_id: {sat_idx: {"landmark_ids": [...], "pano_ids": [...]}}
    """
    pano_metadata = vigor_dataset._panorama_metadata.set_index('pano_id')

    raw_predictions = load_osm_tag_extraction_jsonl(extraction_path)

    pano_data_from_pano_id = {}
    for pano_key, pred in raw_predictions.items():
        pano_id = pano_key.split(',')[0]
        if pano_id not in pano_metadata.index:
            continue
        row = pano_metadata.loc[pano_id]
        sat_idxs = row.positive_satellite_idxs + row.semipositive_satellite_idxs
        pano_data_from_pano_id[pano_id] = {
            "landmarks": pred["landmarks"],
            "location_type": pred.get("location_type"),
            "sat_idxs": sat_idxs
        }

    # Build sat_data_from_sat_id from vigor_dataset (matching existing pattern)
    sat_data_from_sat_id = {}
    for sat_idx, row in vigor_dataset._satellite_metadata.iterrows():
        lm_idxs = row.landmark_idxs
        landmark_ids = [vigor_dataset._landmark_metadata.iloc[i].pruned_props for i in lm_idxs]
        pano_idxs = row.positive_panorama_idxs + row.semipositive_panorama_idxs
        pano_ids = vigor_dataset._panorama_metadata.iloc[pano_idxs].pano_id.tolist()
        sat_data_from_sat_id[sat_idx] = {"landmark_ids": landmark_ids, "pano_ids": pano_ids}

    return {
        "pano_data_from_pano_id": pano_data_from_pano_id,
        "sat_data_from_sat_id": sat_data_from_sat_id,
    }


def compute_osm_tag_match_similarity(dataset: dict) -> dict:
    """Compute similarity based on OSM tag substring matching with multiple aggregations.

    Uses C++ implementation for fast keyed substring matching:
    For each (pano_landmark, osm_landmark) pair, checks if any extracted tag
    (key, value) has a matching key in OSM where the extracted value is a
    case-insensitive substring of the OSM value.

    Aggregation methods (applied per pano-satellite pair):
    Binary (unit weight):
    - max_unit_max: 1 if any pano landmark matched any OSM landmark
    - max_unit_mean: fraction of pano landmarks that matched
    - max_unit_sum: count of pano landmarks that matched
    IDF-weighted:
    - sum_idf_max: max IDF-weighted match score across pano landmarks
    - sum_idf_mean: mean IDF-weighted match score across pano landmarks
    - sum_idf_sum: sum of IDF-weighted match scores across pano landmarks

    Args:
        dataset: Dict with "pano_data_from_pano_id" and "sat_data_from_sat_id"

    Returns:
        Dict with aggregation method keys, each containing {"similarity": tensor, "positives": tensor}
    """
    num_sat_patches = max(dataset['sat_data_from_sat_id'].keys()) + 1
    num_panos = len(dataset["pano_data_from_pano_id"])

    # Binary (unit weight) outputs
    out_max_unit_max = torch.full((num_panos, num_sat_patches), -torch.inf)
    out_max_unit_mean = torch.full((num_panos, num_sat_patches), -torch.inf)
    out_max_unit_sum = torch.full((num_panos, num_sat_patches), -torch.inf)
    # IDF-weighted outputs
    out_sum_idf_max = torch.full((num_panos, num_sat_patches), -torch.inf)
    out_sum_idf_mean = torch.full((num_panos, num_sat_patches), -torch.inf)
    out_sum_idf_sum = torch.full((num_panos, num_sat_patches), -torch.inf)
    positives = torch.zeros((num_panos, num_sat_patches), dtype=torch.bool)

    # Build index of unique OSM landmarks
    osm_idx_from_props = {}
    osm_tags_by_idx = []
    for sat_idx, sat_data in dataset['sat_data_from_sat_id'].items():
        for osm_props in sat_data["landmark_ids"]:
            if osm_props not in osm_idx_from_props:
                osm_idx_from_props[osm_props] = len(osm_tags_by_idx)
                osm_tags_by_idx.append(list(osm_props))
    num_osm = len(osm_tags_by_idx)

    # Collect pano landmark data, tracking which pano each landmark belongs to
    pano_for_lm = []
    pano_lm_tags_list = []
    num_lm_per_pano = []  # number of landmarks per pano
    pano_lm_ranges = {}  # {pano_id: (start, end)} into pano_lm_tags_list

    for pano_idx, (pano_id, pano_info) in enumerate(tqdm.tqdm(
            dataset["pano_data_from_pano_id"].items(), desc="collecting_tags")):
        positives[pano_idx, pano_info["sat_idxs"]] = True

        start = len(pano_lm_tags_list)
        lm_count = 0
        for pano_lm in pano_info["landmarks"]:
            primary = pano_lm.get("primary_tag", {})
            extracted = []
            if primary.get("key") and primary.get("value"):
                extracted.append((primary["key"], primary["value"]))
            for tag in pano_lm.get("additional_tags", []):
                if tag.get("key") and tag.get("value"):
                    extracted.append((tag["key"], tag["value"]))
            if extracted:
                pano_for_lm.append(pano_idx)
                pano_lm_tags_list.append(extracted)
                lm_count += 1
        pano_lm_ranges[pano_id] = (start, start + lm_count)
        num_lm_per_pano.append(lm_count)

    # Compute substring matching using C++ implementation
    if len(pano_lm_tags_list) > 0 and num_osm > 0:
        print(f"Computing substring matches: {len(pano_lm_tags_list)} pano landmarks x {num_osm} OSM landmarks")
        q, t, k = pnm.compute_keyed_substring_matches_detailed(
            pano_lm_tags_list, osm_tags_by_idx
        )
        match_matrix_t = torch.zeros(len(pano_lm_tags_list), num_osm)
        if len(q) > 0:
            match_matrix_t[torch.from_numpy(q), torch.from_numpy(t)] = 1.0

        # Compute per-pano-landmark IDF weights
        # Group matches by (q, k) to get df for each pano tag
        pano_lm_idf = torch.zeros(len(pano_lm_tags_list))
        if len(q) > 0:
            qk_pairs = np.stack([q, k], axis=1)
            unique_qk, _, match_counts = np.unique(
                qk_pairs, axis=0, return_inverse=True, return_counts=True
            )
            # Sum IDF of matched tags into each pano landmark's weight
            for i, (qi, ki) in enumerate(unique_qk):
                pano_lm_idf[qi] += math.log(num_osm / match_counts[i])

        pano_for_lm_t = torch.tensor(pano_for_lm, dtype=torch.long)
        num_lm_per_pano_t = torch.tensor(num_lm_per_pano, dtype=torch.float)

        # Build sat_idx -> osm landmark indices mapping
        osm_idxs_from_sat_idx = {}
        for sat_idx, sat_data in dataset['sat_data_from_sat_id'].items():
            osm_idxs_from_sat_idx[sat_idx] = [osm_idx_from_props[x] for x in sat_data["landmark_ids"]]

        # For each satellite patch, compute aggregations using vectorized operations
        for sat_idx, sat_data in tqdm.tqdm(dataset['sat_data_from_sat_id'].items(), desc="aggregating"):
            osm_idxs = osm_idxs_from_sat_idx[sat_idx]
            if len(osm_idxs) == 0:
                continue

            # Get match scores for OSM landmarks in this sat patch
            # sat_matches: (num_pano_lm, num_osm_in_sat)
            sat_matches = match_matrix_t[:, osm_idxs]

            # For each pano landmark, did it match ANY osm landmark in this sat patch?
            # best_per_lm: (num_pano_lm,) - 1 if matched, 0 otherwise
            best_per_lm = sat_matches.max(dim=1).values

            # Aggregate by pano using scatter operations
            # Sum of best scores per pano
            sum_scores = torch.zeros(num_panos)
            sum_scores.scatter_add_(0, pano_for_lm_t, best_per_lm)

            # For max: use scatter_reduce with 'amax'
            max_scores = torch.full((num_panos,), -torch.inf)
            max_scores.scatter_reduce_(0, pano_for_lm_t, best_per_lm, reduce='amax', include_self=False)

            # Mean = sum / count (avoid div by zero)
            mean_scores = sum_scores / num_lm_per_pano_t.clamp(min=1)

            # IDF-weighted: binary best-per-landmark * per-landmark IDF weight
            idf_best_per_lm = best_per_lm * pano_lm_idf

            idf_sum_scores = torch.zeros(num_panos)
            idf_sum_scores.scatter_add_(0, pano_for_lm_t, idf_best_per_lm)

            idf_max_scores = torch.full((num_panos,), -torch.inf)
            idf_max_scores.scatter_reduce_(0, pano_for_lm_t, idf_best_per_lm, reduce='amax', include_self=False)

            idf_mean_scores = idf_sum_scores / num_lm_per_pano_t.clamp(min=1)

            # Set outputs (only for panos with landmarks)
            has_landmarks = num_lm_per_pano_t > 0
            out_max_unit_max[has_landmarks, sat_idx] = max_scores[has_landmarks]
            out_max_unit_mean[has_landmarks, sat_idx] = mean_scores[has_landmarks]
            out_max_unit_sum[has_landmarks, sat_idx] = sum_scores[has_landmarks]
            out_sum_idf_max[has_landmarks, sat_idx] = idf_max_scores[has_landmarks]
            out_sum_idf_mean[has_landmarks, sat_idx] = idf_mean_scores[has_landmarks]
            out_sum_idf_sum[has_landmarks, sat_idx] = idf_sum_scores[has_landmarks]

    matches, osm_to_sat, table_schema = build_match_tables(
        q=q, t=t, k=k,
        pano_lm_tags=pano_lm_tags_list,
        osm_tags=osm_tags_by_idx,
        pano_lm_ranges=pano_lm_ranges,
        osm_idxs_from_sat_idx=osm_idxs_from_sat_idx,
    )

    return {
        "max_unit_max": {"similarity": out_max_unit_max, "positives": positives},
        "max_unit_mean": {"similarity": out_max_unit_mean, "positives": positives},
        "max_unit_sum": {"similarity": out_max_unit_sum, "positives": positives},
        "sum_idf_max": {"similarity": out_sum_idf_max, "positives": positives},
        "sum_idf_mean": {"similarity": out_sum_idf_mean, "positives": positives},
        "sum_idf_sum": {"similarity": out_sum_idf_sum, "positives": positives},
    }, (matches, osm_to_sat, table_schema)
