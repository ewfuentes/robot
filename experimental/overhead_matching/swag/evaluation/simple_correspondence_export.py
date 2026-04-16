"""Export utilities for SimpleCorrespondenceClassifier.

Simplified version of the precompute path in correspondence_matching.py,
using the simple all-text encoding and 4 cross features.
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm

from experimental.overhead_matching.swag.data.simple_correspondence_dataset import (
    encode_tag_bundle,
    compute_cross_features,
    NUM_CROSS_FEATURES,
    TAG_KEY_TO_IDX,
)
from experimental.overhead_matching.swag.evaluation.correspondence_matching import (
    RawCorrespondenceData,
)


def _log(msg: str):
    print(msg)


def _pad_simple_encoded(encoded_list: list[dict], text_input_dim: int) -> dict:
    """Pad a list of simple-encoded tag bundles to uniform length."""
    max_tags = max((len(e["key_indices"]) for e in encoded_list), default=1)
    max_tags = max(max_tags, 1)

    key_indices = []
    text_embs = []
    masks = []

    for e in encoded_list:
        n = len(e["key_indices"])
        pad = max_tags - n
        key_indices.append(
            torch.tensor(e["key_indices"] + [0] * pad, dtype=torch.long)
        )
        if n > 0:
            stacked = torch.stack(e["text_embeddings"])
        else:
            stacked = torch.zeros(0, text_input_dim)
        if pad > 0:
            stacked = torch.cat([stacked, torch.zeros(pad, text_input_dim)])
        text_embs.append(stacked)
        masks.append(torch.tensor([True] * n + [False] * pad, dtype=torch.bool))

    return {
        "key_indices": torch.stack(key_indices),
        "text_embeddings": torch.stack(text_embs),
        "tag_mask": torch.stack(masks),
    }


def simple_batch_encode_landmarks(
    tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    model,  # SimpleCorrespondenceClassifier
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Encode tag bundles through SimpleTagBundleEncoder. Returns (N, repr_dim) on CPU."""
    all_encoded = [encode_tag_bundle(t, text_embeddings, text_input_dim)
                   for t in tags_list]
    all_reprs = []
    for start in range(0, len(all_encoded), batch_size):
        chunk = all_encoded[start:start + batch_size]
        tensors = _pad_simple_encoded(chunk, text_input_dim)
        with torch.no_grad():
            reprs = model.encoder(
                tensors["key_indices"].to(device),
                tensors["text_embeddings"].to(device),
                tensors["tag_mask"].to(device),
            )
        all_reprs.append(reprs.cpu())
    return torch.cat(all_reprs, dim=0)


def _compute_cross_features_for_pairs(
    pano_tags_list: list[dict[str, str]],
    osm_tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
) -> np.ndarray:
    """Compute (n_pano, n_osm, 4) cross features."""
    n_pano = len(pano_tags_list)
    n_osm = len(osm_tags_list)
    result = np.zeros((n_pano, n_osm, NUM_CROSS_FEATURES), dtype=np.float32)
    for i, pt in enumerate(pano_tags_list):
        for j, ot in enumerate(osm_tags_list):
            result[i, j] = compute_cross_features(pt, ot, text_embeddings)
    return result


def simple_precompute_raw_cost_data(
    model,  # SimpleCorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    dataset,  # VigorDataset
    pano_tags_from_pano_id: dict[str, list[dict]],
    device: torch.device,
    max_pairs_per_batch: int = 50000,
) -> RawCorrespondenceData:
    """Precompute (total_pano_lm x total_osm_lm) P(match) matrix for simple model."""
    num_sats = len(dataset._satellite_metadata)

    # Collect unique OSM landmarks
    osm_lm_idx_to_tags: dict[int, dict[str, str]] = {}
    for sat_idx in range(num_sats):
        sat_meta = dataset._satellite_metadata.iloc[sat_idx]
        lm_idxs = sat_meta.get("landmark_idxs", [])
        if lm_idxs is None:
            continue
        for lm_idx in lm_idxs:
            if lm_idx not in osm_lm_idx_to_tags:
                lm_row = dataset._landmark_metadata.iloc[lm_idx]
                pruned = lm_row.get("pruned_props", frozenset())
                if pruned:
                    osm_lm_idx_to_tags[lm_idx] = dict(pruned)

    unique_osm_lm_idxs = sorted(osm_lm_idx_to_tags.keys())
    osm_tags_list = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]
    n_osm = len(unique_osm_lm_idxs)
    _log(f"  {n_osm} unique OSM landmarks")

    # Pre-encode all OSM landmarks
    _log("  Encoding OSM landmarks...")
    osm_reprs = simple_batch_encode_landmarks(
        osm_tags_list, text_embeddings, text_input_dim, model, device,
    )
    _log(f"  OSM representations: {osm_reprs.shape}")

    # Collect panorama data
    pano_ids = sorted(dataset._panorama_metadata.pano_id.values)
    pano_id_to_lm_rows: dict[str, list[int]] = {}
    all_cost_rows = []
    all_pano_lm_tags = []
    current_row = 0

    # Process in batches of panoramas
    batch_size = 64
    pano_list = []
    for pano_id in pano_ids:
        if pano_id not in pano_tags_from_pano_id:
            continue
        pano_landmarks = pano_tags_from_pano_id[pano_id]
        pano_tags_dicts = [dict(lm["tags"]) for lm in pano_landmarks]
        pano_list.append((pano_id, pano_tags_dicts, pano_landmarks))

    _log(f"  {len(pano_list)} panoramas with tags")

    for batch_start in tqdm(range(0, len(pano_list), batch_size),
                            desc="Precomputing cost matrices"):
        batch_panos = pano_list[batch_start:batch_start + batch_size]

        # Collect all pano landmarks in this batch
        all_pano_tags = []
        pano_offsets = []
        for pano_id, pano_tags_dicts, pano_landmarks in batch_panos:
            offset = len(all_pano_tags)
            all_pano_tags.extend(pano_tags_dicts)
            pano_offsets.append((offset, len(pano_tags_dicts)))

        n_pano_lm = len(all_pano_tags)
        if n_pano_lm == 0:
            for pano_id, _, _ in batch_panos:
                pano_id_to_lm_rows[pano_id] = []
            continue

        # Encode pano landmarks
        pano_reprs = simple_batch_encode_landmarks(
            all_pano_tags, text_embeddings, text_input_dim, model, device,
        )

        # Process OSM in chunks for cross features
        osm_chunk_size = 32768
        osm_cost_chunks = []
        for osm_start in range(0, n_osm, osm_chunk_size):
            osm_end = min(osm_start + osm_chunk_size, n_osm)
            osm_chunk_tags = osm_tags_list[osm_start:osm_end]
            n_osm_chunk = osm_end - osm_start

            # Cross features
            cross_feats = _compute_cross_features_for_pairs(
                all_pano_tags, osm_chunk_tags, text_embeddings,
            )
            cross_tensor = torch.from_numpy(
                cross_feats.reshape(n_pano_lm * n_osm_chunk, -1))

            # Expand representations for all pairs
            pano_exp = pano_reprs.unsqueeze(1).expand(
                n_pano_lm, n_osm_chunk, -1).reshape(-1, pano_reprs.shape[1])
            osm_chunk_reprs = osm_reprs[osm_start:osm_end]
            osm_exp = osm_chunk_reprs.unsqueeze(0).expand(
                n_pano_lm, n_osm_chunk, -1).reshape(-1, osm_chunk_reprs.shape[1])

            # MLP in sub-chunks
            chunk_pairs = n_pano_lm * n_osm_chunk
            chunk_probs = []
            for s in range(0, chunk_pairs, max_pairs_per_batch):
                e = min(s + max_pairs_per_batch, chunk_pairs)
                with torch.no_grad():
                    logits = model.classify_from_reprs(
                        pano_exp[s:e].to(device),
                        osm_exp[s:e].to(device),
                        cross_tensor[s:e].to(device),
                    ).squeeze(-1)
                    chunk_probs.append(torch.sigmoid(logits).cpu().numpy())

            osm_cost_chunks.append(
                np.concatenate(chunk_probs).reshape(n_pano_lm, n_osm_chunk))

        all_cost = np.hstack(osm_cost_chunks)

        # Split back to individual panoramas
        for i, (pano_id, pano_tags_dicts, pano_landmarks) in enumerate(batch_panos):
            offset, n_lm = pano_offsets[i]
            cost = all_cost[offset:offset + n_lm]
            row_indices = list(range(current_row, current_row + n_lm))
            pano_id_to_lm_rows[pano_id] = row_indices
            all_cost_rows.append(cost)
            for lm in pano_landmarks:
                all_pano_lm_tags.append(lm["tags"])
            current_row += n_lm

    if all_cost_rows:
        flat_cost = np.vstack(all_cost_rows)
    else:
        flat_cost = np.zeros((0, n_osm), dtype=np.float32)

    _log(f"  Flat cost matrix shape: {flat_cost.shape}")

    osm_lm_tags = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]
    return RawCorrespondenceData(
        cost_matrix=flat_cost,
        pano_id_to_lm_rows=pano_id_to_lm_rows,
        pano_lm_tags=all_pano_lm_tags,
        osm_lm_indices=unique_osm_lm_idxs,
        osm_lm_tags=osm_lm_tags,
    )
