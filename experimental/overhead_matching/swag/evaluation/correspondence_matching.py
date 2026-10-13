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
import multiprocessing
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    HOUSENUMBER_KEY,
    NUM_CROSS_FEATURES,
    _pad_side,
    encode_tag_bundle,
    parse_housenumber,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    NUM_TAG_KEYS,
    TAG_KEY_TO_IDX,
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
    COUNT = "count"
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
    elif aggregation == AggregationMode.COUNT:
        return sum(weights)
    elif aggregation == AggregationMode.MAX:
        return max(p * w for p, w in zip(match_probs, weights))
    elif aggregation == AggregationMode.LOG_ODDS:
        total = 0.0
        for p, w in zip(match_probs, weights):
            p_clamped = max(min(p, 1.0 - 1e-7), 1e-7)
            total += w * math.log(p_clamped / (1.0 - p_clamped))
        return total
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compute_uniqueness_weights(
    cost_matrix: np.ndarray,
    prob_threshold: float = 0.3,
    weighting: str = "inverse_log",
) -> np.ndarray:
    """Compute per-pano-landmark uniqueness weights from the cost matrix.

    Default weight = 1 / log2(1 + count of OSM landmarks with P(match) >= threshold).
    A pano landmark matching 1 OSM landmark gets weight 1.0; one matching
    100 gets weight ~0.15. `inverse_count` instead uses 1 / max(1, count).
    """
    match_counts = (cost_matrix >= prob_threshold).sum(axis=1)
    counts = np.maximum(match_counts, 1).astype(np.float64)
    if weighting == "inverse_count":
        return 1.0 / counts
    if weighting == "inverse_log":
        return 1.0 / np.log2(1.0 + counts)
    raise ValueError(f"Unknown uniqueness weighting: {weighting}")


def match_and_aggregate(
    cost_matrix: np.ndarray,
    method: MatchingMethod,
    aggregation: AggregationMode,
    prob_threshold: float = 0.3,
    uniqueness_weights: np.ndarray | None = None,
    use_dustbin: bool = True,
) -> MatchResult:
    """Run bipartite matching on a cost matrix and aggregate to a score.

    With `use_dustbin=True` (default, Hungarian only): the cost matrix is
    augmented with n_pano extra "dustbin" columns each valued at
    `prob_threshold`. Hungarian then optimizes jointly over real-column and
    dustbin matches, so it picks a real column iff its probability exceeds
    the threshold — avoiding the pathology where the globally-optimal
    forced 1-to-1 assignment saddles a row with a bad partner that we'd
    later discard anyway. Dustbin matches are stripped before aggregation.

    `use_dustbin=False` reproduces the legacy post-hoc-threshold behavior,
    useful for reproducing older similarity artifacts.
    """
    n_pano, n_osm = cost_matrix.shape
    if n_pano == 0 or n_osm == 0:
        return MatchResult(
            pano_lm_indices=[], osm_lm_indices=[], match_probs=[],
            cost_matrix=cost_matrix, similarity_score=0.0,
        )

    if method == MatchingMethod.HUNGARIAN:
        from scipy.optimize import linear_sum_assignment
        if use_dustbin:
            dustbin = np.full(
                (n_pano, n_pano), prob_threshold, dtype=cost_matrix.dtype,
            )
            aug = np.hstack([cost_matrix, dustbin])
            row_ind, col_ind = linear_sum_assignment(-aug)
        else:
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        pano_inds, osm_inds, probs = [], [], []
        for r, c in zip(row_ind, col_ind):
            if c >= n_osm:  # dustbin column
                continue
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
        key_indices, text_embs, tag_mask = _pad_side(chunk, text_input_dim)
        with torch.no_grad():
            reprs = model.encoder(
                key_indices.to(device),
                text_embs.to(device),
                tag_mask.to(device),
            )
        all_reprs.append(reprs.cpu())
    return torch.cat(all_reprs, dim=0)


@dataclass
class PreparedOsmLandmarks:
    """A list of OSM landmarks encoded once, to score many panoramas against with
    `compute_cost_matrix_prepared`: their tag-bundle representations and their
    text index for the cross features."""
    reprs: torch.Tensor        # (N, repr_dim) on the model's device
    text_index: "OsmTextIndex"

    def __len__(self) -> int:
        return self.reprs.shape[0]


def prepare_osm_landmarks(
    osm_tags_list: list[dict[str, str]],
    model,  # CorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
    allow_missing_text_embeddings: bool = False,
) -> PreparedOsmLandmarks:
    reprs = batch_encode_landmarks(
        osm_tags_list, text_embeddings, text_input_dim, model, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    ).to(device)
    return PreparedOsmLandmarks(
        reprs, build_osm_text_index(osm_tags_list, text_embeddings, text_input_dim, device),
    )


def compute_cost_matrix_prepared(
    pano_tags_list: list[dict[str, str]],
    osm: PreparedOsmLandmarks,
    model,  # CorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    allow_missing_text_embeddings: bool = False,
    osm_chunk_size: int = 32768,
) -> np.ndarray:
    """(n_pano, n_osm) P(match) of each pano tag bundle against landmarks
    prepared once with `prepare_osm_landmarks`. The cross features for every pair
    are computed in one call, (n_pano, n_osm, NUM_CROSS_FEATURES) on `device`;
    the classifier then works through the landmarks in chunks."""
    n_pano = len(pano_tags_list)
    n_osm = len(osm)
    if n_pano == 0 or n_osm == 0:
        return np.zeros((n_pano, n_osm), dtype=np.float32)

    pano_reprs = batch_encode_landmarks(
        pano_tags_list, text_embeddings, text_input_dim, model, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    ).to(device)

    cross_all = compute_cross_features_indexed(
        pano_tags_list, osm.text_index, text_embeddings, text_input_dim, device,
        osm_chunk_size=osm_chunk_size,
    )
    out = np.zeros((n_pano, n_osm), dtype=np.float32)
    for start in range(0, n_osm, osm_chunk_size):
        end = min(start + osm_chunk_size, n_osm)
        c = end - start
        cross = cross_all[:, start:end].reshape(n_pano * c, -1)
        pano_exp = pano_reprs.unsqueeze(1).expand(n_pano, c, -1).reshape(n_pano * c, -1)
        osm_exp = osm.reprs[start:end].unsqueeze(0).expand(n_pano, c, -1).reshape(n_pano * c, -1)
        probs = []
        for s in range(0, n_pano * c, max_pairs_per_batch):
            e = min(s + max_pairs_per_batch, n_pano * c)
            with torch.no_grad():
                logits = model.classify_from_reprs(
                    pano_exp[s:e], osm_exp[s:e], cross[s:e],
                ).squeeze(-1)
            probs.append(torch.sigmoid(logits))
        out[:, start:end] = torch.cat(probs).reshape(n_pano, c).cpu().numpy()
    return out


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
    viewer). Larger workflows should use `precompute_raw_cost_data` instead;
    to score many panoramas against the same landmarks, `prepare_osm_landmarks`
    once and call `compute_cost_matrix_prepared`.
    """
    if len(pano_tags_list) == 0 or len(osm_tags_list) == 0:
        return np.zeros((len(pano_tags_list), len(osm_tags_list)), dtype=np.float32)
    osm = prepare_osm_landmarks(
        osm_tags_list, model, text_embeddings, text_input_dim, device,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    )
    return compute_cost_matrix_prepared(
        pano_tags_list, osm, model, text_embeddings, text_input_dim, device,
        max_pairs_per_batch=max_pairs_per_batch,
        allow_missing_text_embeddings=allow_missing_text_embeddings,
    )


def _LandmarkTextTensors(tags_list: list[dict[str, str]],
                         text_embeddings: dict[str, torch.Tensor],
                         text_input_dim: int,
                         device: torch.device):
    """Build per-landmark tensors used by the vectorized cross-feature path.

    Returns:
        text: (N, NUM_TAG_KEYS, text_input_dim) — text embedding per key, zeros where absent.
        text_norm: (N, NUM_TAG_KEYS) — L2 norm of the above along the last dim.
        has_text: (N, NUM_TAG_KEYS) — 1.0 where text embedding present, else 0.0.
        has_key: (N, NUM_TAG_KEYS) — 1.0 where key appears in the landmark, else 0.0.
        hn_lo / hn_hi: (N,) — parsed housenumber bounds, nan when absent/unparseable.
    """
    n = len(tags_list)
    text = torch.zeros(n, NUM_TAG_KEYS, text_input_dim)
    has_text = torch.zeros(n, NUM_TAG_KEYS)
    has_key = torch.zeros(n, NUM_TAG_KEYS)
    hn_lo = torch.full((n,), float("nan"))
    hn_hi = torch.full((n,), float("nan"))

    for i, tags in enumerate(tags_list):
        for k, v in tags.items():
            idx = TAG_KEY_TO_IDX.get(k)
            if idx is not None:
                has_key[i, idx] = 1.0
                if v in text_embeddings:
                    text[i, idx] = text_embeddings[v]
                    has_text[i, idx] = 1.0
            if k == HOUSENUMBER_KEY:
                lo, hi = parse_housenumber(v)
                hn_lo[i] = lo
                hn_hi[i] = hi

    text = text.to(device)
    text_norm = text.norm(dim=-1).to(device)  # (N, NUM_TAG_KEYS)
    has_text = has_text.to(device)
    has_key = has_key.to(device)
    hn_lo = hn_lo.to(device)
    hn_hi = hn_hi.to(device)
    return text, text_norm, has_text, has_key, hn_lo, hn_hi


@dataclass
class OsmTextIndex:
    """The text side of a list of OSM landmarks, for computing cross features
    against many panoramas without rebuilding it each time.

    Tag values repeat heavily across landmarks, so each landmark keeps, per tag
    key, an index into one table of the distinct values' text embeddings, rather
    than a dense (NUM_TAG_KEYS, text_input_dim) tensor that is almost all zeros.
    """
    value_text: torch.Tensor   # (V, text_input_dim): the distinct values' embeddings
    value_norm: torch.Tensor   # (V,): their L2 norms
    value_idx: torch.Tensor    # (N, NUM_TAG_KEYS) int64: -1 where no key or no text
    hn_lo: torch.Tensor        # (N,): parsed housenumber bounds, nan when absent
    hn_hi: torch.Tensor

    def __len__(self) -> int:
        return self.value_idx.shape[0]

    def rows(self, start: int, end: int) -> "OsmTextIndex":
        """The landmarks [start, end), sharing the value table."""
        return OsmTextIndex(self.value_text, self.value_norm,
                            self.value_idx[start:end],
                            self.hn_lo[start:end], self.hn_hi[start:end])


def build_osm_text_index(
    osm_tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
) -> OsmTextIndex:
    """Index `osm_tags_list` for `compute_cross_features_indexed`."""
    n = len(osm_tags_list)
    value_idx = torch.full((n, NUM_TAG_KEYS), -1, dtype=torch.int64)
    hn_lo = torch.full((n,), float("nan"))
    hn_hi = torch.full((n,), float("nan"))
    value_to_row: dict[str, int] = {}
    for i, tags in enumerate(osm_tags_list):
        for k, v in tags.items():
            idx = TAG_KEY_TO_IDX.get(k)
            if idx is not None and v in text_embeddings:
                row = value_to_row.setdefault(v, len(value_to_row))
                value_idx[i, idx] = row
            if k == HOUSENUMBER_KEY:
                hn_lo[i], hn_hi[i] = parse_housenumber(v)
    value_text = torch.zeros(len(value_to_row), text_input_dim)
    for v, row in value_to_row.items():
        value_text[row] = text_embeddings[v]
    value_text = value_text.to(device)
    return OsmTextIndex(
        value_text=value_text,
        value_norm=value_text.norm(dim=-1),
        value_idx=value_idx.to(device),
        hn_lo=hn_lo.to(device),
        hn_hi=hn_hi.to(device),
    )


def compute_cross_features_indexed(
    pano_tags_list: list[dict[str, str]],
    osm: OsmTextIndex,
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
    osm_chunk_size: int = 32768,
) -> torch.Tensor:
    """The 4 cross features of every (pano, OSM) pair: (n_pano, n_osm,
    NUM_CROSS_FEATURES) on `device`, the same as the Python-loop reference
    (correspondence_matching_test.py) up to floating point tolerance.

    Only the (pano landmark, key) pairs with text can contribute, so their
    similarities to every distinct OSM value are computed once, then gathered for
    each OSM landmark that has text at the same key.
    """
    n_pano = len(pano_tags_list)
    n_osm = len(osm)
    result = torch.zeros(n_pano, n_osm, NUM_CROSS_FEATURES, device=device)
    if n_pano == 0 or n_osm == 0:
        return result

    (pano_text, pano_text_norm, pano_has_text, _pano_has_key,
     pano_hn_lo, pano_hn_hi) = _LandmarkTextTensors(
        pano_tags_list, text_embeddings, text_input_dim, device,
    )

    # The pano's (landmark, key) pairs with text, and their similarity to every
    # distinct OSM value: (Q, V).
    q_pano, q_key = torch.nonzero(pano_has_text > 0, as_tuple=True)
    eps = 1e-8
    q_sim = (pano_text[q_pano, q_key] @ osm.value_text.T) / (
        pano_text_norm[q_pano, q_key].unsqueeze(1) * osm.value_norm.unsqueeze(0) + eps
    )
    name_key_idx = TAG_KEY_TO_IDX.get("name")

    for start in range(0, n_osm, osm_chunk_size):
        end = min(start + osm_chunk_size, n_osm)
        c = end - start

        if len(q_pano) > 0 and osm.value_text.shape[0] > 0:
            # Each pair's OSM value at the same key: (Q, c), -1 where none.
            v = osm.value_idx[start:end][:, q_key].T
            valid = v >= 0
            sim = torch.gather(q_sim, 1, v.clamp(min=0))
            idx = q_pano.unsqueeze(1).expand(-1, c)

            # text_max: max over valid keys; 0 when there are none.
            text_max = torch.full((n_pano, c), -float("inf"), device=device)
            text_max.scatter_reduce_(
                0, idx, torch.where(valid, sim, torch.full_like(sim, -float("inf"))),
                reduce="amax")
            count = torch.zeros(n_pano, c, device=device).scatter_add_(
                0, idx, valid.to(sim.dtype))
            text_max = torch.where(count > 0, text_max, torch.zeros_like(text_max))

            # text_mean: sum / count, 0 when count is 0.
            total = torch.zeros(n_pano, c, device=device).scatter_add_(
                0, idx, sim * valid)
            text_mean = total / count.clamp(min=1)

            # text_name: the similarity at the "name" key, if both have text there.
            text_name = torch.zeros(n_pano, c, device=device)
            if name_key_idx is not None:
                is_name = q_key == name_key_idx
                text_name[q_pano[is_name]] = (sim * valid)[is_name]

            result[:, start:end, 0] = text_max
            result[:, start:end, 1] = text_mean
            result[:, start:end, 2] = text_name

        # Housenumber range overlap.
        plo = pano_hn_lo.unsqueeze(1)
        phi = pano_hn_hi.unsqueeze(1)
        olo = osm.hn_lo[start:end].unsqueeze(0)
        ohi = osm.hn_hi[start:end].unsqueeze(0)
        hn_valid = (
            torch.isfinite(plo) & torch.isfinite(phi)
            & torch.isfinite(olo) & torch.isfinite(ohi)
        )
        hn_overlap = (
            ((plo >= olo) & (plo <= ohi)) | ((olo >= plo) & (olo <= phi))
        ) & hn_valid
        result[:, start:end, 3] = hn_overlap.to(result.dtype)

    return result


def _compute_cross_features_gpu(
    pano_tags_list: list[dict[str, str]],
    osm_tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
) -> np.ndarray:
    """(n_pano, n_osm, NUM_CROSS_FEATURES) cross features as numpy: indexes the
    OSM side and calls `compute_cross_features_indexed`. To score many panoramas
    against the same landmarks, build the index once instead."""
    osm = build_osm_text_index(osm_tags_list, text_embeddings, text_input_dim, device)
    return compute_cross_features_indexed(
        pano_tags_list, osm, text_embeddings, text_input_dim, device,
    ).cpu().numpy()


# ---------------------------------------------------------------------------
# Precompute full cost data
# ---------------------------------------------------------------------------

@dataclass
class RawCorrespondenceData:
    """Precomputed P(match) data for all pano landmarks × all OSM landmarks."""
    cost_matrix: np.ndarray  # (total_pano_lm, total_osm_lm) float32; np.memmap when streamed
    pano_id_to_lm_rows: dict[str, list[int]]  # pano_id → row indices
    pano_lm_tags: list  # tags per pano landmark row (list of key=value tuples)
    osm_lm_indices: list[int]  # col_idx → dataset landmark_idx
    osm_lm_tags: list[dict[str, str]]  # tags per OSM landmark column
    cost_matrix_path: Path | None = None  # canonical .npy path when streamed


def precompute_raw_cost_data(
    model,  # CorrespondenceClassifier
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    dataset,  # VigorDataset
    pano_tags_from_pano_id: dict[str, list[dict]],
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    allow_missing_text_embeddings: bool = False,
    cost_matrix_memmap_path: Path | None = None,
) -> RawCorrespondenceData:
    """Precompute the flat (total_pano_lm × total_osm_lm) P(match) matrix.

    When `cost_matrix_memmap_path` is provided, the cost matrix is streamed
    to a .npy file via `np.lib.format.open_memmap` so it never lives fully in
    RAM. To avoid leaving a half-zeroed file on a mid-precompute crash, rows
    are written to `<path>.partial` and atomically renamed to the canonical
    path only after a successful `flush()`. The returned
    `RawCorrespondenceData.cost_matrix` is the memmap re-opened against the
    canonical path; `cost_matrix_path` is also populated.
    """
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
    osm_text_index = build_osm_text_index(
        osm_tags_list, text_embeddings, text_input_dim, device,
    )
    _log_memory("after encoding OSM landmarks")

    pano_ids = sorted(dataset._panorama_metadata.pano_id.values)
    pano_id_to_lm_rows: dict[str, list[int]] = {}
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

    total_pano_lm = sum(len(pano_tags_dicts) for _, pano_tags_dicts, _ in pano_list)
    partial_memmap_path: Path | None = None
    if cost_matrix_memmap_path is not None:
        cost_matrix_memmap_path.parent.mkdir(parents=True, exist_ok=True)
        partial_memmap_path = cost_matrix_memmap_path.with_suffix(
            cost_matrix_memmap_path.suffix + ".partial"
        )
        flat_cost = np.lib.format.open_memmap(
            partial_memmap_path, mode="w+",
            dtype=np.float32, shape=(total_pano_lm, n_osm),
        )
        all_cost_rows = None
        _log(f"  Streaming cost matrix to {partial_memmap_path} "
             f"(final: {cost_matrix_memmap_path}, "
             f"shape={(total_pano_lm, n_osm)}, "
             f"~{total_pano_lm * n_osm * 4 / 1e9:.1f} GB)")
    else:
        flat_cost = None
        all_cost_rows = []

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
            n_osm_chunk = osm_end - osm_start

            cross_feats = compute_cross_features_indexed(
                all_pano_tags, osm_text_index.rows(osm_start, osm_end),
                text_embeddings, text_input_dim, device,
            )
            cross_tensor = cross_feats.reshape(n_pano_lm * n_osm_chunk, -1).cpu()

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
            if flat_cost is not None:
                flat_cost[current_row:current_row + n_lm] = cost
            else:
                all_cost_rows.append(cost)
            for lm in pano_landmarks:
                all_pano_lm_tags.append(lm["tags"])
            current_row += n_lm

    cost_matrix_path: Path | None = None
    if flat_cost is not None:
        flat_cost.flush()
        # Drop the memmap so the underlying file handle closes before rename.
        del flat_cost
        os.replace(partial_memmap_path, cost_matrix_memmap_path)
        cost_matrix_path = cost_matrix_memmap_path
        flat_cost = np.load(cost_matrix_memmap_path, mmap_mode="r")
    elif all_cost_rows:
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
        cost_matrix_path=cost_matrix_path,
    )


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def _score_panorama(task, raw, sat_col_positions, method, aggregation,
                    prob_threshold, uniqueness_weighted, use_dustbin,
                    uniqueness_weighting):
    pano_idx, pano_id = task
    scores = np.zeros(len(sat_col_positions), dtype=np.float32)
    rows = raw.pano_id_to_lm_rows.get(pano_id)
    if not rows:
        return pano_idx, scores
    pano_cost = raw.cost_matrix[rows]
    weights = (
        compute_uniqueness_weights(pano_cost, prob_threshold, uniqueness_weighting)
        if uniqueness_weighted else None
    )
    for sat_idx, cols in enumerate(sat_col_positions):
        if cols:
            scores[sat_idx] = match_and_aggregate(
                pano_cost[:, cols], method, aggregation, prob_threshold,
                uniqueness_weights=weights, use_dustbin=use_dustbin,
            ).similarity_score
    return pano_idx, scores


def _init_similarity_worker(score_panorama):
    # Fork shares the raw matrix; only panorama IDs and result rows cross the queue.
    global _worker_score_panorama
    _worker_score_panorama = score_panorama


def _score_in_worker(task):
    return _worker_score_panorama(task)


def similarity_from_raw_data(
    raw: RawCorrespondenceData,
    dataset,  # VigorDataset
    method: MatchingMethod = MatchingMethod.HUNGARIAN,
    aggregation: AggregationMode = AggregationMode.SUM,
    prob_threshold: float = 0.3,
    uniqueness_weighted: bool = False,
    use_dustbin: bool = True,
    uniqueness_weighting: str = "inverse_log",
    workers: int = 1,
) -> torch.Tensor:
    """Build similarity matrix from precomputed raw cost data.

    See `match_and_aggregate` for the meaning of `use_dustbin`. Pass
    `use_dustbin=False` to reproduce legacy (post-hoc threshold) artifacts.
    `workers > 1` uses forked CPU workers and must run before CUDA initialization.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > 1 and torch.cuda.is_initialized():
        raise ValueError("Parallel scoring requires a fresh CPU process; use --from_raw")
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

    score_panorama = partial(
        _score_panorama, raw=raw, sat_col_positions=sat_col_positions,
        method=method, aggregation=aggregation, prob_threshold=prob_threshold,
        uniqueness_weighted=uniqueness_weighted, use_dustbin=use_dustbin,
        uniqueness_weighting=uniqueness_weighting,
    )
    tasks = enumerate(dataset._panorama_metadata["pano_id"])
    if workers == 1:
        for idx, scores in tqdm(map(score_panorama, tasks), total=num_panos,
                                desc="Building similarity matrix"):
            similarity[idx] = torch.from_numpy(scores)
    else:
        with multiprocessing.get_context("fork").Pool(
            workers, initializer=_init_similarity_worker, initargs=(score_panorama,),
        ) as pool:
            for idx, scores in tqdm(pool.imap_unordered(_score_in_worker, tasks),
                                    total=num_panos, desc="Building similarity matrix"):
                similarity[idx] = torch.from_numpy(scores)

    return similarity
