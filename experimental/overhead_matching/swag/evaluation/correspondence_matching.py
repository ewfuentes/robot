"""Core library for correspondence-based landmark matching and similarity.

Provides the main building blocks used by the export script, panorama viewer,
and correspondence explorer:

- `precompute_raw_cost_data`: encode all pano/OSM tag bundles through a
  trained CorrespondenceClassifier and compute a flat P(match) cost matrix
  of shape (total_pano_lm, total_osm_lm).
- `similarity_from_raw_data`: given that flat cost matrix, run per-satellite
  bipartite matching (Hungarian / greedy) and aggregate (sum / max / log-odds)
  to produce a (num_panos, num_sats) similarity matrix.
- `match_and_aggregate` / `_aggregate_score` / `compute_uniqueness_weights`:
  the matching/aggregation primitives, reusable by downstream explorer tools.

There is a single code path: `precompute_raw_cost_data` is always the
entry point, and the persisted `RawCorrespondenceData` can be fed into
`similarity_from_raw_data` either live (same process) or after loading from
disk. There is no in-memory-only "build similarity directly" shortcut.
"""

import enum
import math
import os
from dataclasses import dataclass
from datetime import datetime

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    NUM_CROSS_FEATURES,
    compute_cross_features,
    encode_tag_bundle,
)

_LOG_PATH = "/tmp/correspondence_precompute.log"


def _log(msg: str):
    """Print and append to log file for post-mortem debugging."""
    print(msg)
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _log_memory(label: str):
    """Log current RAM and GPU memory usage."""
    import psutil
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1e9
    gpu_msg = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        gpu_msg = f", gpu alloc={alloc:.2f}GB reserved={reserved:.2f}GB"
    _log(f"  [mem @ {label}] rss={rss_gb:.2f}GB{gpu_msg}")


# ---------------------------------------------------------------------------
# Matching / aggregation primitives
# ---------------------------------------------------------------------------

class MatchingMethod(enum.Enum):
    HUNGARIAN = "hungarian"
    GREEDY = "greedy"


class AggregationMode(enum.Enum):
    SUM = "sum"
    MAX = "max"
    LOG_ODDS = "log_odds"


@dataclass
class MatchResult:
    pano_lm_indices: list[int]
    osm_lm_indices: list[int]
    match_probs: list[float]
    cost_matrix: np.ndarray  # (n_pano_lm, n_osm_lm) P(match)
    similarity_score: float


def _aggregate_score(
    match_probs: list[float],
    aggregation: AggregationMode,
    weights: list[float] | None = None,
) -> float:
    """Aggregate matched probabilities into a single similarity score."""
    if not match_probs:
        return 0.0
    if weights is None:
        weights = [1.0] * len(match_probs)
    if aggregation == AggregationMode.SUM:
        return sum(p * w for p, w in zip(match_probs, weights))
    if aggregation == AggregationMode.MAX:
        return max(p * w for p, w in zip(match_probs, weights))
    if aggregation == AggregationMode.LOG_ODDS:
        total = 0.0
        for p, w in zip(match_probs, weights):
            p_clamped = max(min(p, 1.0 - 1e-7), 1e-7)
            total += w * math.log(p_clamped / (1.0 - p_clamped))
        return total
    raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_uniqueness_weights(
    cost_matrix: np.ndarray,
    prob_threshold: float = 0.3,
) -> np.ndarray:
    """Compute per-pano-landmark uniqueness weights from the cost matrix.

    Weight = 1 / log2(1 + count of OSM landmarks with P(match) > threshold).
    A pano landmark matching 1 OSM landmark gets weight 1.0; one matching
    100 gets weight ~0.15.
    """
    match_counts = (cost_matrix >= prob_threshold).sum(axis=1)
    return 1.0 / np.log2(1.0 + np.maximum(match_counts, 1).astype(np.float64))


def match_and_aggregate(
    cost_matrix: np.ndarray,
    method: MatchingMethod,
    aggregation: AggregationMode,
    prob_threshold: float = 0.3,
    uniqueness_weights: np.ndarray | None = None,
) -> MatchResult:
    """Run bipartite matching on a cost matrix and aggregate to a score."""
    n_pano, n_osm = cost_matrix.shape
    if n_pano == 0 or n_osm == 0:
        return MatchResult(
            pano_lm_indices=[], osm_lm_indices=[], match_probs=[],
            cost_matrix=cost_matrix, similarity_score=0.0,
        )

    if method == MatchingMethod.HUNGARIAN:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        pano_inds, osm_inds, probs = [], [], []
        for r, c in zip(row_ind, col_ind):
            p = cost_matrix[r, c]
            if p >= prob_threshold:
                pano_inds.append(int(r))
                osm_inds.append(int(c))
                probs.append(float(p))
    elif method == MatchingMethod.GREEDY:
        flat_indices = np.argsort(-cost_matrix.ravel())
        used_pano: set[int] = set()
        used_osm: set[int] = set()
        pano_inds, osm_inds, probs = [], [], []
        for flat_idx in flat_indices:
            r, c = divmod(int(flat_idx), n_osm)
            p = cost_matrix[r, c]
            if p < prob_threshold:
                break
            if r in used_pano or c in used_osm:
                continue
            used_pano.add(r)
            used_osm.add(c)
            pano_inds.append(r)
            osm_inds.append(c)
            probs.append(float(p))
            if len(used_pano) == min(n_pano, n_osm):
                break
    else:
        raise ValueError(f"Unknown method: {method}")

    weights = None
    if uniqueness_weights is not None:
        weights = [float(uniqueness_weights[r]) for r in pano_inds]
    score = _aggregate_score(probs, aggregation, weights)
    return MatchResult(
        pano_lm_indices=pano_inds,
        osm_lm_indices=osm_inds,
        match_probs=probs,
        cost_matrix=cost_matrix,
        similarity_score=score,
    )


# ---------------------------------------------------------------------------
# Encoder / cross-feature helpers used by the precompute path
# ---------------------------------------------------------------------------

def _pad_encoded(encoded_list: list[dict], text_input_dim: int) -> dict:
    """Pad a list of encoded tag bundles into stacked (key_indices, text_embeddings, tag_mask) tensors."""
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


def batch_encode_landmarks(
    tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    model,  # CorrespondenceClassifier
    device: torch.device,
    batch_size: int = 4096,
    allow_missing_text_embeddings: bool = False,
) -> torch.Tensor:
    """Encode tag bundles through TagBundleEncoder. Returns (N, repr_dim) on CPU."""
    all_encoded = [
        encode_tag_bundle(
            t, text_embeddings, text_input_dim,
            allow_missing_text_embeddings=allow_missing_text_embeddings,
        )
        for t in tags_list
    ]
    all_reprs = []
    for start in range(0, len(all_encoded), batch_size):
        chunk = all_encoded[start:start + batch_size]
        tensors = _pad_encoded(chunk, text_input_dim)
        with torch.no_grad():
            reprs = model.encoder(
                tensors["key_indices"].to(device),
                tensors["text_embeddings"].to(device),
                tensors["tag_mask"].to(device),
            )
        all_reprs.append(reprs.cpu())
    return torch.cat(all_reprs, dim=0)


def compute_pairs_cost_matrix(
    pano_tags_list: list[dict[str, str]],
    osm_tags_list: list[dict[str, str]],
    model,  # CorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    allow_missing_text_embeddings: bool = False,
) -> np.ndarray:
    """Compute (n_pano, n_osm) P(match) for arbitrary tag-dict lists.

    Lightweight helper for interactive / small-batch use (e.g. the panorama
    viewer). Larger workflows should use `precompute_raw_cost_data` instead.
    """
    n_pano = len(pano_tags_list)
    n_osm = len(osm_tags_list)
    if n_pano == 0 or n_osm == 0:
        return np.zeros((n_pano, n_osm), dtype=np.float32)

    pano_reprs = batch_encode_landmarks(
        pano_tags_list, text_embeddings, text_input_dim, model, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    )
    osm_reprs = batch_encode_landmarks(
        osm_tags_list, text_embeddings, text_input_dim, model, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    )

    cross = _compute_cross_features_for_pairs(
        pano_tags_list, osm_tags_list, text_embeddings,
    )
    cross_tensor = torch.from_numpy(cross.reshape(n_pano * n_osm, -1))

    pano_exp = pano_reprs.unsqueeze(1).expand(
        n_pano, n_osm, -1,
    ).reshape(-1, pano_reprs.shape[1])
    osm_exp = osm_reprs.unsqueeze(0).expand(
        n_pano, n_osm, -1,
    ).reshape(-1, osm_reprs.shape[1])

    total_pairs = n_pano * n_osm
    chunks = []
    for s in range(0, total_pairs, max_pairs_per_batch):
        e = min(s + max_pairs_per_batch, total_pairs)
        with torch.no_grad():
            logits = model.classify_from_reprs(
                pano_exp[s:e].to(device),
                osm_exp[s:e].to(device),
                cross_tensor[s:e].to(device),
            ).squeeze(-1)
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks).reshape(n_pano, n_osm)


def _compute_cross_features_for_pairs(
    pano_tags_list: list[dict[str, str]],
    osm_tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
) -> np.ndarray:
    """Return (n_pano, n_osm, NUM_CROSS_FEATURES) cross features via Python loop.

    Slow but simple reference implementation; see the vectorized replacement
    in a follow-up commit.
    """
    n_pano = len(pano_tags_list)
    n_osm = len(osm_tags_list)
    result = np.zeros((n_pano, n_osm, NUM_CROSS_FEATURES), dtype=np.float32)
    for i, pt in enumerate(pano_tags_list):
        for j, ot in enumerate(osm_tags_list):
            result[i, j] = compute_cross_features(pt, ot, text_embeddings)
    return result


# ---------------------------------------------------------------------------
# Precompute full cost data
# ---------------------------------------------------------------------------

@dataclass
class RawCorrespondenceData:
    """Precomputed P(match) data for all pano landmarks × all OSM landmarks."""
    cost_matrix: np.ndarray  # (total_pano_lm, total_osm_lm) float32
    pano_id_to_lm_rows: dict[str, list[int]]  # pano_id → row indices
    pano_lm_tags: list  # tags per pano landmark row (list of key=value tuples)
    osm_lm_indices: list[int]  # col_idx → dataset landmark_idx
    osm_lm_tags: list[dict[str, str]]  # tags per OSM landmark column


def precompute_raw_cost_data(
    model,  # CorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    dataset,  # VigorDataset
    pano_tags_from_pano_id: dict[str, list[dict]],
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    allow_missing_text_embeddings: bool = False,
) -> RawCorrespondenceData:
    """Precompute the flat (total_pano_lm × total_osm_lm) P(match) matrix."""
    num_sats = len(dataset._satellite_metadata)

    # Collect unique OSM landmarks from satellite metadata
    osm_lm_idx_to_tags: dict[int, dict[str, str]] = {}
    dropped_empty = 0
    for sat_idx in range(num_sats):
        sat_meta = dataset._satellite_metadata.iloc[sat_idx]
        lm_idxs = sat_meta.get("landmark_idxs", [])
        if lm_idxs is None:
            continue
        for lm_idx in lm_idxs:
            if lm_idx in osm_lm_idx_to_tags:
                continue
            lm_row = dataset._landmark_metadata.iloc[lm_idx]
            pruned = lm_row.get("pruned_props", frozenset())
            if pruned:
                osm_lm_idx_to_tags[lm_idx] = dict(pruned)
            else:
                dropped_empty += 1

    if dropped_empty:
        _log(f"  Dropped {dropped_empty} OSM landmarks with empty pruned_props.")

    unique_osm_lm_idxs = sorted(osm_lm_idx_to_tags.keys())
    osm_tags_list = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]
    n_osm = len(unique_osm_lm_idxs)
    _log(f"  {n_osm} unique OSM landmarks")

    _log("  Encoding OSM landmarks...")
    osm_reprs = batch_encode_landmarks(
        osm_tags_list, text_embeddings, text_input_dim, model, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    )
    _log(f"  OSM representations: {osm_reprs.shape}")
    _log_memory("after encoding OSM landmarks")

    pano_ids = sorted(dataset._panorama_metadata.pano_id.values)
    pano_id_to_lm_rows: dict[str, list[int]] = {}
    all_cost_rows = []
    all_pano_lm_tags: list = []
    current_row = 0

    pano_list = []
    for pano_id in pano_ids:
        if pano_id not in pano_tags_from_pano_id:
            continue
        pano_landmarks = pano_tags_from_pano_id[pano_id]
        pano_tags_dicts = [dict(lm["tags"]) for lm in pano_landmarks]
        pano_list.append((pano_id, pano_tags_dicts, pano_landmarks))

    _log(f"  {len(pano_list)} panoramas with tags")

    batch_size = 64
    for batch_start in tqdm(
        range(0, len(pano_list), batch_size),
        desc="Precomputing cost matrices",
    ):
        batch_panos = pano_list[batch_start:batch_start + batch_size]

        all_pano_tags: list[dict[str, str]] = []
        pano_offsets = []
        for _pid, pano_tags_dicts, _plms in batch_panos:
            offset = len(all_pano_tags)
            all_pano_tags.extend(pano_tags_dicts)
            pano_offsets.append((offset, len(pano_tags_dicts)))

        n_pano_lm = len(all_pano_tags)
        if n_pano_lm == 0:
            for pano_id, _, _ in batch_panos:
                pano_id_to_lm_rows[pano_id] = []
            continue

        pano_reprs = batch_encode_landmarks(
            all_pano_tags, text_embeddings, text_input_dim, model, device,
            allow_missing_text_embeddings=allow_missing_text_embeddings,
        )

        osm_chunk_size = 32768
        osm_cost_chunks = []
        for osm_start in range(0, n_osm, osm_chunk_size):
            osm_end = min(osm_start + osm_chunk_size, n_osm)
            osm_chunk_tags = osm_tags_list[osm_start:osm_end]
            n_osm_chunk = osm_end - osm_start

            cross_feats = _compute_cross_features_for_pairs(
                all_pano_tags, osm_chunk_tags, text_embeddings,
            )
            cross_tensor = torch.from_numpy(
                cross_feats.reshape(n_pano_lm * n_osm_chunk, -1),
            )

            pano_exp = pano_reprs.unsqueeze(1).expand(
                n_pano_lm, n_osm_chunk, -1,
            ).reshape(-1, pano_reprs.shape[1])
            osm_chunk_reprs = osm_reprs[osm_start:osm_end]
            osm_exp = osm_chunk_reprs.unsqueeze(0).expand(
                n_pano_lm, n_osm_chunk, -1,
            ).reshape(-1, osm_chunk_reprs.shape[1])

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
                np.concatenate(chunk_probs).reshape(n_pano_lm, n_osm_chunk),
            )

        all_cost = np.hstack(osm_cost_chunks)

        for i, (pano_id, _pano_tags_dicts, pano_landmarks) in enumerate(batch_panos):
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

    return RawCorrespondenceData(
        cost_matrix=flat_cost,
        pano_id_to_lm_rows=pano_id_to_lm_rows,
        pano_lm_tags=all_pano_lm_tags,
        osm_lm_indices=unique_osm_lm_idxs,
        osm_lm_tags=osm_tags_list,
    )


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def similarity_from_raw_data(
    raw: RawCorrespondenceData,
    dataset,  # VigorDataset
    method: MatchingMethod = MatchingMethod.HUNGARIAN,
    aggregation: AggregationMode = AggregationMode.SUM,
    prob_threshold: float = 0.3,
    uniqueness_weighted: bool = False,
) -> torch.Tensor:
    """Build similarity matrix from precomputed raw cost data."""
    num_panos = len(dataset._panorama_metadata)
    num_sats = len(dataset._satellite_metadata)
    similarity = torch.zeros(num_panos, num_sats)

    osm_idx_to_col = {idx: col for col, idx in enumerate(raw.osm_lm_indices)}

    sat_col_positions = []
    for sat_idx in range(num_sats):
        sat_meta = dataset._satellite_metadata.iloc[sat_idx]
        lm_idxs = sat_meta.get("landmark_idxs", [])
        if lm_idxs is None:
            sat_col_positions.append([])
        else:
            sat_col_positions.append(
                [osm_idx_to_col[i] for i in lm_idxs if i in osm_idx_to_col]
            )

    for pano_idx in tqdm(range(num_panos), desc="Building similarity matrix"):
        pano_id = dataset._panorama_metadata.iloc[pano_idx]["pano_id"]
        rows = raw.pano_id_to_lm_rows.get(pano_id)
        if rows is None:
            continue
        pano_cost = raw.cost_matrix[rows]

        u_weights = (
            compute_uniqueness_weights(pano_cost, prob_threshold)
            if uniqueness_weighted else None
        )

        for sat_idx in range(num_sats):
            cols = sat_col_positions[sat_idx]
            if not cols:
                continue
            sub_cost = pano_cost[:, cols]
            result = match_and_aggregate(
                sub_cost, method, aggregation, prob_threshold,
                uniqueness_weights=u_weights,
            )
            similarity[pano_idx, sat_idx] = result.similarity_score

    return similarity
