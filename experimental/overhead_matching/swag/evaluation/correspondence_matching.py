"""Core library for correspondence-based landmark matching and similarity.

Provides the main building blocks used by the export script, panorama viewer,
and correspondence explorer:

- Cost matrix computation: encode tag bundles through a trained
  CorrespondenceClassifier and compute P(match) for all
  (pano_landmark, osm_landmark) pairs
- Bipartite matching: Hungarian or greedy 1:1 assignment
- Aggregation: combine match probabilities into a single similarity score
  (sum, max, or log-odds)
- Similarity matrix construction: build (num_panos, num_sats) matrices
  from per-pair scores with optional checkpointed precomputation
"""

import enum
import math
import os
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import torch
from tqdm import tqdm

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
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        gpu_msg = f", GPU: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved"
    _log(f"  [{label}] RAM: {rss_gb:.1f}GB{gpu_msg}")

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    CorrespondenceBatch,
    _pad_encoded,
    compute_cross_features,
    encode_tag_bundle,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    TAG_KEY_TO_IDX,
    ValueType,
    key_type,
)


def _tags_encodable(tags: dict[str, str], text_embeddings: dict[str, torch.Tensor] | None) -> bool:
    """Check if all text-type tag values have embeddings available."""
    for k, v in tags.items():
        if k not in TAG_KEY_TO_IDX:
            continue
        if key_type(k) == ValueType.TEXT:
            if text_embeddings is None or v not in text_embeddings:
                return False
    return True


@dataclass
class LandmarkCrossData:
    """Pre-computed data for vectorized cross-feature computation.

    Text embeddings are stored sparsely: per text key, a dense embedding
    vector for each landmark (zero if absent). This avoids the (N, T, D)
    dense matrix that OOMs for large N.
    """
    key_set_bits: np.ndarray   # (N, num_keys) bool — which keys are present
    value_hashes: np.ndarray   # (N, num_keys) int64 — hash of value per key (0 if absent)
    # Per text key: (N, emb_dim) normalized embedding (zero if absent) + (N,) mask
    text_key_embeddings: dict[int, np.ndarray]  # text_key_idx → (N, emb_dim) normed
    text_key_masks: dict[int, np.ndarray]       # text_key_idx → (N,) bool
    text_keys_ordered: list[str]
    name_idx: int              # index of "name" in text_keys_ordered
    numeric_values: np.ndarray # (N, 6) float32 — parsed numeric values (NaN if absent)
    numeric_scales: np.ndarray # (6,) float32 — scale per numeric key
    housenumber_lo: np.ndarray # (N,) float32 — parsed housenumber low (NaN if absent)
    housenumber_hi: np.ndarray # (N,) float32 — parsed housenumber high (NaN if absent)
    n_keys: np.ndarray         # (N,) int — number of keys per landmark


def precompute_cross_data(
    tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor] | None,
    text_keys: list[str] | None = None,
) -> LandmarkCrossData:
    """Pre-compute per-landmark data for vectorized cross features.

    Args:
        text_keys: If provided, use this fixed set of text keys (for alignment
            between pano and osm data). If None, auto-detect from data.
    """
    from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
        HOUSENUMBER_KEY, NUMERIC_KEYS,
        parse_housenumber, parse_maxheight, parse_numeric,
    )

    N = len(tags_list)
    all_keys = sorted(TAG_KEY_TO_IDX.keys())
    key_to_bit = {k: i for i, k in enumerate(all_keys)}
    num_keys = len(all_keys)

    # Identify text keys — use provided set or auto-detect from data
    if text_keys is not None:
        text_keys_ordered = text_keys
    else:
        present_text_keys = set()
        for tags in tags_list:
            for k in tags:
                if k in TAG_KEY_TO_IDX and key_type(k) == ValueType.TEXT:
                    present_text_keys.add(k)
        text_keys_ordered = sorted(present_text_keys)
    text_key_to_idx = {k: i for i, k in enumerate(text_keys_ordered)}
    name_idx = text_key_to_idx.get("name", -1)
    num_text_keys = len(text_keys_ordered)
    emb_dim = next(iter(text_embeddings.values())).shape[0] if text_embeddings else 768

    # Numeric keys
    numeric_keys_ordered = sorted(NUMERIC_KEYS)
    numeric_scales = np.array([10.0 if k == "maxspeed" else 2.0 for k in numeric_keys_ordered],
                              dtype=np.float32)

    # Pre-convert text embeddings to numpy once (avoid per-call .numpy())
    _text_emb_np_cache: dict[str, np.ndarray] = {}
    if text_embeddings is not None:
        for tv, te in text_embeddings.items():
            _text_emb_np_cache[tv] = te.numpy() if isinstance(te, torch.Tensor) else te

    # Pre-compute key type and numeric index lookups (avoid repeated calls)
    _key_types = {k: key_type(k) for k in all_keys}
    _numeric_key_to_idx = {k: i for i, k in enumerate(numeric_keys_ordered)}

    # Allocate arrays
    key_set_bits = np.zeros((N, num_keys), dtype=bool)
    value_hashes = np.zeros((N, num_keys), dtype=np.int64)
    # Sparse text storage: only allocate per-key arrays for keys that appear
    # Each is (N, emb_dim) but allocated lazily
    _text_emb_per_key: dict[int, np.ndarray] = {}  # ti → (N, D)
    _text_mask_per_key: dict[int, np.ndarray] = {}  # ti → (N,)
    numeric_values = np.full((N, 6), np.nan, dtype=np.float32)
    housenumber_lo = np.full(N, np.nan, dtype=np.float32)
    housenumber_hi = np.full(N, np.nan, dtype=np.float32)
    n_keys = np.zeros(N, dtype=np.int32)

    _iter = range(N)
    if N > 10000:
        _iter = tqdm(_iter, desc="  Building cross-feature data")
    for i in _iter:
        tags = tags_list[i]
        n_keys[i] = len(tags)
        for k, v in tags.items():
            if k not in key_to_bit:
                continue
            bit = key_to_bit[k]
            key_set_bits[i, bit] = True
            value_hashes[i, bit] = hash(v)

            kt = _key_types[k]
            if kt == ValueType.TEXT and k in text_key_to_idx:
                ti = text_key_to_idx[k]
                emb = _text_emb_np_cache.get(v)
                if emb is not None:
                    if ti not in _text_emb_per_key:
                        _text_emb_per_key[ti] = np.zeros((N, emb_dim), dtype=np.float32)
                        _text_mask_per_key[ti] = np.zeros(N, dtype=bool)
                    _text_emb_per_key[ti][i] = emb
                    _text_mask_per_key[ti][i] = True

            elif kt == ValueType.NUMERIC:
                ni = _numeric_key_to_idx.get(k, -1)
                if ni >= 0:
                    if k == "maxheight":
                        numeric_values[i, ni] = parse_maxheight(v)
                    else:
                        numeric_values[i, ni], _ = parse_numeric(v, key=k)

            elif kt == ValueType.HOUSENUMBER:
                lo, hi = parse_housenumber(v)
                housenumber_lo[i] = lo
                housenumber_hi[i] = hi

    # Normalize text embeddings for cosine similarity
    text_key_embeddings: dict[int, np.ndarray] = {}
    text_key_masks: dict[int, np.ndarray] = {}
    for ti, emb_arr in _text_emb_per_key.items():
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        text_key_embeddings[ti] = (emb_arr / norms).astype(np.float32)
        text_key_masks[ti] = _text_mask_per_key[ti]

    return LandmarkCrossData(
        key_set_bits=key_set_bits,
        value_hashes=value_hashes,
        text_key_embeddings=text_key_embeddings,
        text_key_masks=text_key_masks,
        text_keys_ordered=text_keys_ordered,
        name_idx=name_idx,
        numeric_values=numeric_values,
        numeric_scales=numeric_scales,
        housenumber_lo=housenumber_lo,
        housenumber_hi=housenumber_hi,
        n_keys=n_keys,
    )


def compute_cross_features_batch(
    pano_data: LandmarkCrossData,
    osm_data: LandmarkCrossData,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute cross features for all pano×osm pairs vectorized.

    Uses GPU if device is provided (much faster for large n_osm).
    Returns: (n_pano, n_osm, 13) float32 numpy array.
    """
    if device is not None and device.type == 'cuda':
        return _compute_cross_features_gpu(pano_data, osm_data, device)
    return _compute_cross_features_cpu(pano_data, osm_data)


def _compute_cross_features_cpu(
    pano_data: LandmarkCrossData,
    osm_data: LandmarkCrossData,
) -> np.ndarray:
    """CPU path for small datasets."""
    n_pano = pano_data.key_set_bits.shape[0]
    n_osm = osm_data.key_set_bits.shape[0]

    shared_counts = pano_data.key_set_bits.astype(np.float32) @ osm_data.key_set_bits.astype(np.float32).T
    union_counts = (pano_data.n_keys[:, None] + osm_data.n_keys[None, :]).astype(np.float32) - shared_counts
    jaccard = shared_counts / np.maximum(union_counts, 1.0)

    both_present = pano_data.key_set_bits[:, None, :] & osm_data.key_set_bits[None, :, :]
    same_hash = pano_data.value_hashes[:, None, :] == osm_data.value_hashes[None, :, :]
    exact_matches = (both_present & same_hash).sum(axis=2).astype(np.float32)

    # Text cosine similarity — iterate over text keys that exist in both sides
    text_max = np.zeros((n_pano, n_osm), dtype=np.float32)
    text_sum = np.zeros((n_pano, n_osm), dtype=np.float32)
    n_text_shared = np.zeros((n_pano, n_osm), dtype=np.float32)
    name_sim = np.zeros((n_pano, n_osm), dtype=np.float32)
    ni = pano_data.name_idx

    common_text_keys = set(pano_data.text_key_embeddings.keys()) & set(osm_data.text_key_embeddings.keys())
    for ti in common_text_keys:
        p_emb = pano_data.text_key_embeddings[ti]  # (p, D) already normalized
        o_emb = osm_data.text_key_embeddings[ti]   # (o, D)
        p_mask = pano_data.text_key_masks[ti]       # (p,)
        o_mask = osm_data.text_key_masks[ti]        # (o,)
        sim = p_emb @ o_emb.T  # (p, o)
        both = p_mask[:, None] & o_mask[None, :]
        sim = np.where(both, sim, 0.0)
        text_sum += sim
        text_max = np.maximum(text_max, sim)
        n_text_shared += both.astype(np.float32)
        if ti == ni:
            name_sim = sim

    text_mean = np.where(n_text_shared > 0, text_sum / np.maximum(n_text_shared, 1), 0.0)

    pano_num = pano_data.numeric_values
    osm_num = osm_data.numeric_values
    diff = np.abs(pano_num[:, None, :] - osm_num[None, :, :])
    numeric_prox = np.exp(-diff / pano_data.numeric_scales[None, None, :])
    both_num = (~np.isnan(pano_num))[:, None, :] & (~np.isnan(osm_num))[None, :, :]
    numeric_prox = np.nan_to_num(np.where(both_num, numeric_prox, 0.0), nan=0.0)

    plo, phi_ = pano_data.housenumber_lo, pano_data.housenumber_hi
    olo, ohi = osm_data.housenumber_lo, osm_data.housenumber_hi
    both_hn = (~np.isnan(plo) & ~np.isnan(phi_))[:, None] & (~np.isnan(olo) & ~np.isnan(ohi))[None, :]
    hn_overlap = np.zeros((n_pano, n_osm), dtype=np.float32)
    if both_hn.any():
        overlap = ((plo[:, None] >= olo[None, :]) & (plo[:, None] <= ohi[None, :])) | \
                  ((olo[None, :] >= plo[:, None]) & (olo[None, :] <= phi_[:, None]))
        hn_overlap = np.where(both_hn, overlap.astype(np.float32), 0.0)

    return np.stack([
        jaccard, shared_counts / 10.0, exact_matches / 10.0,
        text_max, text_mean, name_sim,
        numeric_prox[:, :, 0], numeric_prox[:, :, 1], numeric_prox[:, :, 2],
        numeric_prox[:, :, 3], numeric_prox[:, :, 4], numeric_prox[:, :, 5],
        hn_overlap,
    ], axis=2).astype(np.float32)


def _compute_cross_features_gpu(
    pano_data: LandmarkCrossData,
    osm_data: LandmarkCrossData,
    device: torch.device,
) -> np.ndarray:
    """GPU path — batches over OSM landmarks to avoid OOM."""
    n_pano = pano_data.key_set_bits.shape[0]
    n_osm = osm_data.key_set_bits.shape[0]

    # Pano data is small (~7-200), put it all on GPU
    p_keys = torch.from_numpy(pano_data.key_set_bits.astype(np.float32)).to(device)
    p_nkeys = torch.from_numpy(pano_data.n_keys.astype(np.float32)).to(device)
    p_hashes = torch.from_numpy(pano_data.value_hashes).to(device)
    p_bits = torch.from_numpy(pano_data.key_set_bits).to(device)
    # Pano text embeddings on GPU (sparse, per key)
    p_text_gpu = {ti: torch.from_numpy(emb).to(device)
                  for ti, emb in pano_data.text_key_embeddings.items()}
    p_mask_gpu = {ti: torch.from_numpy(mask).to(device)
                  for ti, mask in pano_data.text_key_masks.items()}
    p_num = torch.from_numpy(np.nan_to_num(pano_data.numeric_values, nan=0.0)).to(device)
    p_has_num = torch.from_numpy(~np.isnan(pano_data.numeric_values)).to(device)
    scales = torch.from_numpy(pano_data.numeric_scales).to(device)
    plo = torch.from_numpy(np.nan_to_num(pano_data.housenumber_lo, nan=-999)).to(device)
    phi_ = torch.from_numpy(np.nan_to_num(pano_data.housenumber_hi, nan=-999)).to(device)
    p_has_hn = torch.from_numpy(~np.isnan(pano_data.housenumber_lo) & ~np.isnan(pano_data.housenumber_hi)).to(device)

    ni = pano_data.name_idx
    common_text_keys = set(pano_data.text_key_embeddings.keys()) & set(osm_data.text_key_embeddings.keys())

    # Batch over OSM landmarks to avoid GPU OOM
    # Per-key text embeddings: chunk_size * D * 4 bytes per key
    # 32K * 768 * 4 ≈ 94 MB per key — safe even with ~15 keys
    osm_chunk_size = 32768
    result_chunks = []

    for osm_start in range(0, n_osm, osm_chunk_size):
        osm_end = min(osm_start + osm_chunk_size, n_osm)
        cs = slice(osm_start, osm_end)
        n_chunk = osm_end - osm_start

        o_keys = torch.from_numpy(osm_data.key_set_bits[cs].astype(np.float32)).to(device)
        o_nkeys = torch.from_numpy(osm_data.n_keys[cs].astype(np.float32)).to(device)

        shared_counts = p_keys @ o_keys.T
        union_counts = p_nkeys[:, None] + o_nkeys[None, :] - shared_counts
        jaccard = shared_counts / torch.clamp(union_counts, min=1.0)

        o_hashes = torch.from_numpy(osm_data.value_hashes[cs]).to(device)
        o_bits = torch.from_numpy(osm_data.key_set_bits[cs]).to(device)
        both = p_bits[:, None, :] & o_bits[None, :, :]
        same = p_hashes[:, None, :] == o_hashes[None, :, :]
        exact_matches = (both & same).sum(dim=2).float()

        # Text sim per key — sparse, only iterate keys that exist on both sides
        text_max_val = torch.zeros(n_pano, n_chunk, device=device)
        text_sum_val = torch.zeros(n_pano, n_chunk, device=device)
        name_sim_val = torch.zeros(n_pano, n_chunk, device=device)
        n_text_shared = torch.zeros(n_pano, n_chunk, device=device)

        for ti in common_text_keys:
            p_emb = p_text_gpu[ti]  # (p, D) already normalized
            o_emb = torch.from_numpy(osm_data.text_key_embeddings[ti][cs]).to(device)  # (chunk, D)
            p_m = p_mask_gpu[ti]    # (p,)
            o_m = torch.from_numpy(osm_data.text_key_masks[ti][cs]).to(device)  # (chunk,)
            sim = p_emb @ o_emb.T   # (p, chunk)
            mask_2d = p_m.unsqueeze(1) & o_m.unsqueeze(0)
            sim_masked = torch.where(mask_2d, sim, torch.zeros_like(sim))
            text_sum_val += sim_masked
            text_max_val = torch.max(text_max_val, sim_masked)
            n_text_shared += mask_2d.float()
            if ti == ni:
                name_sim_val = sim_masked

        text_mean = torch.where(n_text_shared > 0,
                                text_sum_val / torch.clamp(n_text_shared, min=1),
                                torch.zeros_like(text_sum_val))

        # Numeric
        o_num = torch.from_numpy(np.nan_to_num(osm_data.numeric_values[cs], nan=0.0)).to(device)
        o_has = torch.from_numpy(~np.isnan(osm_data.numeric_values[cs])).to(device)
        diff = torch.abs(p_num[:, None, :] - o_num[None, :, :])
        numeric_prox = torch.exp(-diff / scales[None, None, :])
        both_num = p_has_num[:, None, :] & o_has[None, :, :]
        numeric_prox = torch.where(both_num, numeric_prox, torch.zeros_like(numeric_prox))

        # Housenumber
        olo = torch.from_numpy(np.nan_to_num(osm_data.housenumber_lo[cs], nan=-999)).to(device)
        ohi = torch.from_numpy(np.nan_to_num(osm_data.housenumber_hi[cs], nan=-999)).to(device)
        o_has_hn = torch.from_numpy(
            ~np.isnan(osm_data.housenumber_lo[cs]) & ~np.isnan(osm_data.housenumber_hi[cs])).to(device)
        both_hn = p_has_hn[:, None] & o_has_hn[None, :]
        overlap = ((plo[:, None] >= olo[None, :]) & (plo[:, None] <= ohi[None, :])) | \
                  ((olo[None, :] >= plo[:, None]) & (olo[None, :] <= phi_[:, None]))
        hn_overlap = torch.where(both_hn, overlap.float(), torch.zeros(n_pano, n_chunk, device=device))

        chunk_result = torch.stack([
            jaccard, shared_counts / 10.0, exact_matches / 10.0,
            text_max_val, text_mean, name_sim_val,
            numeric_prox[:, :, 0], numeric_prox[:, :, 1], numeric_prox[:, :, 2],
            numeric_prox[:, :, 3], numeric_prox[:, :, 4], numeric_prox[:, :, 5],
            hn_overlap,
        ], dim=2)
        result_chunks.append(chunk_result.cpu())

    return torch.cat(result_chunks, dim=1).numpy()


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


def batch_encode_landmarks(
    tags_list: list[dict[str, str]],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    model,  # CorrespondenceClassifier
    device: torch.device,
    batch_size: int = 4096,
    all_text: bool = False,
) -> torch.Tensor:
    """Encode a list of tag bundles through TagBundleEncoder in batches.

    Returns: (N, repr_dim) tensor of encoder representations on CPU.
    """
    all_encoded = [encode_tag_bundle(t, text_embeddings, text_input_dim, all_text=all_text)
                   for t in tags_list]

    all_reprs = []
    for start in range(0, len(all_encoded), batch_size):
        chunk = all_encoded[start:start + batch_size]
        tensors = _pad_encoded(chunk, text_input_dim)
        with torch.no_grad():
            reprs = model.encoder(
                tensors["key_indices"].to(device),
                tensors["value_type"].to(device),
                tensors["boolean_values"].to(device),
                tensors["numeric_values"].to(device),
                tensors["numeric_nan_mask"].to(device),
                tensors["housenumber_values"].to(device),
                tensors["housenumber_nan_mask"].to(device),
                tensors["text_embeddings"].to(device),
                tensors["tag_mask"].to(device),
            )
        all_reprs.append(reprs.cpu())

    return torch.cat(all_reprs, dim=0)


def _forward_batch(pano_encoded_list, osm_encoded_list, cross_features_list,
                    text_input_dim, model, device):
    """Run a single batch through the model, return probabilities."""
    pano_tensors = _pad_encoded(pano_encoded_list, text_input_dim)
    osm_tensors = _pad_encoded(osm_encoded_list, text_input_dim)
    cross_features = torch.tensor(cross_features_list, dtype=torch.float)
    labels = torch.zeros(len(pano_encoded_list), dtype=torch.float)

    batch = CorrespondenceBatch(
        pano_key_indices=pano_tensors["key_indices"],
        pano_value_type=pano_tensors["value_type"],
        pano_boolean_values=pano_tensors["boolean_values"],
        pano_numeric_values=pano_tensors["numeric_values"],
        pano_numeric_nan_mask=pano_tensors["numeric_nan_mask"],
        pano_housenumber_values=pano_tensors["housenumber_values"],
        pano_housenumber_nan_mask=pano_tensors["housenumber_nan_mask"],
        pano_text_embeddings=pano_tensors["text_embeddings"],
        pano_tag_mask=pano_tensors["tag_mask"],
        osm_key_indices=osm_tensors["key_indices"],
        osm_value_type=osm_tensors["value_type"],
        osm_boolean_values=osm_tensors["boolean_values"],
        osm_numeric_values=osm_tensors["numeric_values"],
        osm_numeric_nan_mask=osm_tensors["numeric_nan_mask"],
        osm_housenumber_values=osm_tensors["housenumber_values"],
        osm_housenumber_nan_mask=osm_tensors["housenumber_nan_mask"],
        osm_text_embeddings=osm_tensors["text_embeddings"],
        osm_tag_mask=osm_tensors["tag_mask"],
        cross_features=cross_features,
        labels=labels,
    ).to(device)

    with torch.no_grad():
        logits = model(
            pano_key_indices=batch.pano_key_indices,
            pano_value_type=batch.pano_value_type,
            pano_boolean_values=batch.pano_boolean_values,
            pano_numeric_values=batch.pano_numeric_values,
            pano_numeric_nan_mask=batch.pano_numeric_nan_mask,
            pano_housenumber_values=batch.pano_housenumber_values,
            pano_housenumber_nan_mask=batch.pano_housenumber_nan_mask,
            pano_text_embeddings=batch.pano_text_embeddings,
            pano_tag_mask=batch.pano_tag_mask,
            osm_key_indices=batch.osm_key_indices,
            osm_value_type=batch.osm_value_type,
            osm_boolean_values=batch.osm_boolean_values,
            osm_numeric_values=batch.osm_numeric_values,
            osm_numeric_nan_mask=batch.osm_numeric_nan_mask,
            osm_housenumber_values=batch.osm_housenumber_values,
            osm_housenumber_nan_mask=batch.osm_housenumber_nan_mask,
            osm_text_embeddings=batch.osm_text_embeddings,
            osm_tag_mask=batch.osm_tag_mask,
            cross_features=batch.cross_features,
        ).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()


def compute_cost_matrix(
    pano_tags_list: list[dict[str, str]],
    osm_tags_list: list[dict[str, str]],
    model: torch.nn.Module,
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    all_text: bool = False,
) -> np.ndarray:
    """Compute P(match) for all (pano_lm, osm_lm) pairs.

    Automatically chunks into batches to avoid OOM.
    Returns: (n_pano, n_osm) numpy array of match probabilities.
    """
    n_pano = len(pano_tags_list)
    n_osm = len(osm_tags_list)
    if n_pano == 0 or n_osm == 0:
        return np.zeros((n_pano, n_osm), dtype=np.float32)

    # How many OSM landmarks per chunk?
    osm_chunk_size = max(1, max_pairs_per_batch // n_pano)

    all_probs = []
    for osm_start in range(0, n_osm, osm_chunk_size):
        osm_end = min(osm_start + osm_chunk_size, n_osm)
        osm_chunk = osm_tags_list[osm_start:osm_end]

        pano_encoded_list = []
        osm_encoded_list = []
        cross_features_list = []

        for pt in pano_tags_list:
            for ot in osm_chunk:
                pano_encoded_list.append(
                    encode_tag_bundle(pt, text_embeddings, text_input_dim, all_text=all_text)
                )
                osm_encoded_list.append(
                    encode_tag_bundle(ot, text_embeddings, text_input_dim, all_text=all_text)
                )
                cross_features_list.append(
                    compute_cross_features(pt, ot, text_embeddings)
                )

        chunk_probs = _forward_batch(
            pano_encoded_list, osm_encoded_list, cross_features_list,
            text_input_dim, model, device,
        )
        # Reshape to (n_pano, chunk_osm_size)
        all_probs.append(chunk_probs.reshape(n_pano, len(osm_chunk)))

    return np.hstack(all_probs)


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
) -> np.ndarray:
    """Compute per-pano-landmark uniqueness weights from the cost matrix.

    Weight = 1 / log2(1 + count of OSM landmarks with P(match) > threshold).
    A pano landmark matching 1 OSM landmark gets weight 1.0.
    One matching 100 gets weight ~0.15.

    Args:
        cost_matrix: (n_pano, n_osm) P(match) matrix
        prob_threshold: threshold for counting matches

    Returns: (n_pano,) float array of weights
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
    """Run bipartite matching on cost matrix and aggregate to score.

    Args:
        cost_matrix: (n_pano, n_osm) P(match) matrix
        method: Hungarian or greedy matching
        aggregation: How to aggregate matched probabilities
        prob_threshold: Minimum P(match) to include in matching
        uniqueness_weights: (n_pano,) per-pano-landmark weights. If None, all 1.0.
    """
    n_pano, n_osm = cost_matrix.shape
    if n_pano == 0 or n_osm == 0:
        return MatchResult(
            pano_lm_indices=[], osm_lm_indices=[], match_probs=[],
            cost_matrix=cost_matrix, similarity_score=0.0,
        )

    if method == MatchingMethod.HUNGARIAN:
        from scipy.optimize import linear_sum_assignment
        # Hungarian minimizes cost; we want to maximize P(match)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        pano_inds, osm_inds, probs = [], [], []
        for r, c in zip(row_ind, col_ind):
            p = cost_matrix[r, c]
            if p >= prob_threshold:
                pano_inds.append(int(r))
                osm_inds.append(int(c))
                probs.append(float(p))

    elif method == MatchingMethod.GREEDY:
        # Greedily pick highest-probability pairs
        flat_indices = np.argsort(-cost_matrix.ravel())
        used_pano = set()
        used_osm = set()
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


@dataclass
class RawCorrespondenceData:
    """Precomputed P(match) data for all pano landmarks × all OSM landmarks."""
    cost_matrix: np.ndarray  # (total_pano_lm, total_osm_lm) float32
    pano_id_to_lm_rows: dict[str, list[int]]  # pano_id → row indices
    pano_lm_tags: list[list[tuple]]  # tags per pano landmark row
    osm_lm_indices: list[int]  # col_idx → dataset landmark_idx
    osm_lm_tags: list[dict[str, str]]  # tags per OSM landmark column


def precompute_raw_cost_data(
    model: torch.nn.Module,
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    dataset,  # VigorDataset
    pano_tags_from_pano_id: dict[str, list[dict]],
    device: torch.device,
    max_pairs_per_batch: int = 50000,
    checkpoint_dir: str | None = None,
    all_text: bool = False,
) -> RawCorrespondenceData:
    """Precompute a flat (total_pano_lm × total_osm_lm) P(match) matrix.

    Pre-encodes all OSM landmarks once, then for each panorama encodes only
    its pano landmarks and combines with the cached OSM encodings.
    Landmarks with text values missing from text_embeddings will raise
    during encode_tag_bundle — expand the embeddings file first.
    """
    num_sats = len(dataset._satellite_metadata)

    # Collect ALL unique OSM landmarks across all satellites
    osm_lm_idx_to_tags: dict[int, dict[str, str]] = {}
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

    unique_osm_lm_idxs = sorted(osm_lm_idx_to_tags.keys())
    osm_tags_list = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]
    n_osm = len(unique_osm_lm_idxs)
    # Clear log
    with open(_LOG_PATH, "w") as f:
        f.write(f"=== precompute_raw_cost_data started {datetime.now()} ===\n")
    _log(f"  {n_osm} unique OSM landmarks across all satellites")
    _log_memory("after collecting OSM landmarks")

    # Pre-encode ALL OSM landmarks through the encoder in one batched pass
    _log("  Encoding OSM landmarks through TagBundleEncoder...")
    osm_reprs = batch_encode_landmarks(
        osm_tags_list, text_embeddings, text_input_dim, model, device,
        all_text=all_text,
    )
    _log(f"  OSM representations: {osm_reprs.shape}")
    _log_memory("after encoding OSM landmarks")

    # Collect all valid panoramas upfront (need their tags to determine text keys)
    pano_batch_data = []  # list of (pano_id, pano_tags_list, pano_landmarks)
    for pano_idx in range(len(dataset._panorama_metadata)):
        pano_row = dataset._panorama_metadata.iloc[pano_idx]
        pano_id = pano_row["pano_id"]
        pano_landmarks = pano_tags_from_pano_id.get(pano_id)
        if pano_landmarks is None:
            continue
        pano_tags_list = [dict(lm["tags"]) for lm in pano_landmarks]
        if not pano_tags_list:
            continue
        pano_batch_data.append((pano_id, pano_tags_list, pano_landmarks))
    _log(f"  {len(pano_batch_data)} panoramas with tags")
    _log_memory("after collecting pano data")

    # Process panoramas in batches — concatenate multiple panos' landmarks
    # into one big batch for the encoder and cross features
    all_cost_rows = []
    pano_id_to_lm_rows: dict[str, list[int]] = {}
    all_pano_lm_tags: list[list[tuple]] = []
    current_row = 0
    # Adaptive batch size: target ~200 total pano landmarks per batch
    # Compute the union of text keys across OSM + all pano landmarks
    all_text_keys = set()
    for tags in osm_tags_list:
        for k in tags:
            if k in TAG_KEY_TO_IDX and key_type(k) == ValueType.TEXT:
                all_text_keys.add(k)
    for _, ptl, _ in pano_batch_data:
        for tags in ptl:
            for k in tags:
                if k in TAG_KEY_TO_IDX and key_type(k) == ValueType.TEXT:
                    all_text_keys.add(k)
    shared_text_keys = sorted(all_text_keys)
    _log(f"  {len(shared_text_keys)} text keys present in data (of 93 total)")

    avg_lm = max(1, sum(len(ptl) for _, ptl, _ in pano_batch_data) // len(pano_batch_data))
    pano_batch_size = max(1, min(64, 200 // avg_lm))

    # OSM chunk size for cross-feature computation to avoid RAM OOM
    # Each text key: (chunk, 768) float32 = chunk * 3KB
    # With ~15 keys: chunk * 45KB. At 32K chunks: ~1.4GB — safe
    osm_cross_chunk = 32768

    total_pano_lm = sum(len(ptl) for _, ptl, _ in pano_batch_data)
    _log(f"  Avg {avg_lm} landmarks/pano, batch size: {pano_batch_size}, "
         f"OSM cross chunk: {osm_cross_chunk}")
    _log(f"  Total pano landmarks: {total_pano_lm}, "
         f"cost matrix will be ({total_pano_lm}, {n_osm}) = "
         f"{total_pano_lm * n_osm * 4 / 1e9:.1f} GB")

    # Checkpoint setup: save partial results to disk so we can resume after OOM
    _ckpt_dir = None
    _ckpt_interval = 100  # save every N pano batches
    _resume_batch = 0
    if checkpoint_dir is not None:
        import json as _json
        _ckpt_dir = Path(checkpoint_dir)
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
        _ckpt_meta = _ckpt_dir / "meta.json"
        # Check for existing checkpoint to resume from
        if _ckpt_meta.exists():
            _meta = _json.loads(_ckpt_meta.read_text())
            _resume_batch = _meta["next_batch_start"]
            current_row = _meta["current_row"]
            pano_id_to_lm_rows = _meta["pano_id_to_lm_rows"]
            all_pano_lm_tags = _meta["pano_lm_tags"]
            # Load saved cost chunks
            for _cf in sorted(_ckpt_dir.glob("cost_*.npy")):
                all_cost_rows.append(np.load(_cf))
            _log(f"  Resumed from checkpoint: batch {_resume_batch}, "
                 f"{len(all_cost_rows)} chunks, {current_row} rows")
        _log(f"  Checkpointing to {_ckpt_dir} every {_ckpt_interval} batches")

    _log_memory("before main loop")

    _last_ckpt_batch = _resume_batch
    for batch_start in tqdm(range(0, len(pano_batch_data), pano_batch_size),
                            desc="Precomputing cost matrices"):
        # Skip already-computed batches when resuming
        if batch_start < _resume_batch:
            continue

        batch_end = min(batch_start + pano_batch_size, len(pano_batch_data))
        batch_panos = pano_batch_data[batch_start:batch_end]

        # Concatenate all pano landmarks in this batch
        all_pano_tags = []
        pano_offsets = []  # (start_idx, n_lm) for each pano in batch
        for _, ptl, _ in batch_panos:
            pano_offsets.append((len(all_pano_tags), len(ptl)))
            all_pano_tags.extend(ptl)

        n_total_pano_lm = len(all_pano_tags)

        # Encode all pano landmarks in one batch
        pano_reprs = batch_encode_landmarks(
            all_pano_tags, text_embeddings, text_input_dim, model, device,
            all_text=all_text,
        )

        # Pano cross data is small (~200 landmarks)
        pano_cross_data = precompute_cross_data(all_pano_tags, text_embeddings,
                                                text_keys=shared_text_keys)

        # Process OSM landmarks in chunks to avoid RAM OOM on cross data
        osm_cost_chunks = []
        for osm_start in range(0, n_osm, osm_cross_chunk):
            osm_end = min(osm_start + osm_cross_chunk, n_osm)
            osm_chunk_tags = osm_tags_list[osm_start:osm_end]
            n_osm_chunk = osm_end - osm_start

            osm_chunk_cross = precompute_cross_data(
                osm_chunk_tags, text_embeddings, text_keys=shared_text_keys)
            cross_feats = compute_cross_features_batch(
                pano_cross_data, osm_chunk_cross, device)
            # cross_feats: (n_total_pano_lm, n_osm_chunk, 13)
            cross_tensor = torch.from_numpy(
                cross_feats.reshape(n_total_pano_lm * n_osm_chunk, -1))

            # Build pairs for this OSM chunk
            pano_exp = pano_reprs.unsqueeze(1).expand(
                n_total_pano_lm, n_osm_chunk, -1).reshape(-1, pano_reprs.shape[1])
            osm_chunk_reprs = osm_reprs[osm_start:osm_end]
            osm_exp = osm_chunk_reprs.unsqueeze(0).expand(
                n_total_pano_lm, n_osm_chunk, -1).reshape(-1, osm_chunk_reprs.shape[1])

            # MLP in sub-chunks
            chunk_pairs = n_total_pano_lm * n_osm_chunk
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
                np.concatenate(chunk_probs).reshape(n_total_pano_lm, n_osm_chunk))

        all_cost = np.hstack(osm_cost_chunks)

        # Split results back to individual panoramas
        for i, (pano_id, pano_tags_list, pano_landmarks) in enumerate(batch_panos):
            offset, n_lm = pano_offsets[i]
            cost = all_cost[offset:offset + n_lm]
            row_indices = list(range(current_row, current_row + n_lm))
            pano_id_to_lm_rows[pano_id] = row_indices
            all_cost_rows.append(cost)
            for lm in pano_landmarks:
                all_pano_lm_tags.append(lm["tags"])
            current_row += n_lm

        # Explicitly free intermediates
        del all_cost, osm_cost_chunks, pano_reprs, pano_cross_data

        # Checkpoint: save partial results to disk
        if _ckpt_dir is not None and (batch_start - _last_ckpt_batch) >= _ckpt_interval * pano_batch_size:
            import json as _json
            # Save new cost chunks since last checkpoint
            for _ci, _chunk in enumerate(all_cost_rows):
                _cf = _ckpt_dir / f"cost_{_ci:06d}.npy"
                if not _cf.exists():
                    np.save(_cf, _chunk)
            # Save metadata
            _meta = {
                "next_batch_start": batch_start + pano_batch_size,
                "current_row": current_row,
                "pano_id_to_lm_rows": pano_id_to_lm_rows,
                "pano_lm_tags": all_pano_lm_tags,
            }
            (_ckpt_dir / "meta.json").write_text(_json.dumps(_meta))
            _last_ckpt_batch = batch_start
            _log(f"  Checkpoint saved at batch {batch_start} ({len(all_cost_rows)} chunks)")
            _log_memory(f"checkpoint batch {batch_start}")

        # Log memory periodically
        elif batch_start % (pano_batch_size * 50) == 0:
            _log_memory(f"batch {batch_start}")

    _log_memory("before vstack")
    if all_cost_rows:
        flat_cost = np.vstack(all_cost_rows)
        del all_cost_rows  # free the list of arrays
    else:
        flat_cost = np.zeros((0, n_osm), dtype=np.float32)

    _log(f"  Flat cost matrix shape: {flat_cost.shape}")
    _log_memory("final")
    return RawCorrespondenceData(
        cost_matrix=flat_cost,
        pano_id_to_lm_rows=pano_id_to_lm_rows,
        pano_lm_tags=all_pano_lm_tags,
        osm_lm_indices=unique_osm_lm_idxs,
        osm_lm_tags=osm_tags_list,
    )


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

    # Build col position lookup
    osm_idx_to_col = {idx: col for col, idx in enumerate(raw.osm_lm_indices)}

    # Pre-build sat → col positions
    sat_col_positions = []
    for sat_idx in range(num_sats):
        sat_meta = dataset._satellite_metadata.iloc[sat_idx]
        lm_idxs = sat_meta.get("landmark_idxs", [])
        if lm_idxs is None:
            sat_col_positions.append([])
        else:
            sat_col_positions.append([osm_idx_to_col[i] for i in lm_idxs if i in osm_idx_to_col])

    for pano_idx in tqdm(range(num_panos), desc="Building similarity matrix"):
        pano_id = dataset._panorama_metadata.iloc[pano_idx]["pano_id"]
        rows = raw.pano_id_to_lm_rows.get(pano_id)
        if rows is None:
            continue
        pano_cost = raw.cost_matrix[rows]

        # Compute uniqueness weights from full pano cost matrix
        u_weights = compute_uniqueness_weights(pano_cost, prob_threshold) if uniqueness_weighted else None

        for sat_idx in range(num_sats):
            cols = sat_col_positions[sat_idx]
            if not cols:
                continue
            sub_cost = pano_cost[:, cols]
            result = match_and_aggregate(sub_cost, method, aggregation, prob_threshold,
                                         uniqueness_weights=u_weights)
            similarity[pano_idx, sat_idx] = result.similarity_score

    return similarity


def build_correspondence_similarity_matrix(
    model: torch.nn.Module,
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    dataset,  # VigorDataset
    pano_tags_from_pano_id: dict[str, list[dict]],
    device: torch.device,
    method: MatchingMethod = MatchingMethod.HUNGARIAN,
    aggregation: AggregationMode = AggregationMode.SUM,
    prob_threshold: float = 0.3,
    uniqueness_weighted: bool = False,
) -> torch.Tensor:
    """Build (num_panos, num_sats) similarity matrix using correspondence matching.

    Args:
        model: Trained CorrespondenceClassifier (must be in eval mode)
        text_embeddings: Pre-computed {value_string: tensor}
        text_input_dim: Dimension of text embeddings
        dataset: VigorDataset with landmarks loaded
        pano_tags_from_pano_id: {pano_id: [{"tags": [(k,v),...], ...}, ...]}
            from extract_panorama_data_across_cities(base, extract_tags_from_pano_data)
        device: torch device
        method: Matching method
        aggregation: Aggregation mode
        prob_threshold: Minimum P(match) to include
        uniqueness_weighted: Weight matches by pano landmark uniqueness
    """
    num_panos = len(dataset._panorama_metadata)
    num_sats = len(dataset._satellite_metadata)
    similarity = torch.zeros(num_panos, num_sats)

    _missing_warned = False

    for pano_idx in tqdm(range(num_panos), desc="Building correspondence similarity"):
        pano_row = dataset._panorama_metadata.iloc[pano_idx]
        pano_id = pano_row["pano_id"]

        # Get pano tags
        pano_landmarks = pano_tags_from_pano_id.get(pano_id)
        if pano_landmarks is None:
            continue
        pano_tags_list = [dict(lm["tags"]) for lm in pano_landmarks]
        if not pano_tags_list:
            continue

        # Collect all unique OSM landmarks across all satellites that have landmarks
        # Map: osm_lm_idx -> dict tags
        # Map: sat_idx -> list of osm_lm_indices
        osm_lm_idx_to_tags = {}
        sat_to_osm_lm_idxs = {}

        for sat_idx in range(num_sats):
            sat_meta = dataset._satellite_metadata.iloc[sat_idx]
            lm_idxs = sat_meta.get("landmark_idxs", [])
            if lm_idxs is None or len(lm_idxs) == 0:
                continue

            sat_osm_indices = []
            for lm_idx in lm_idxs:
                if lm_idx not in osm_lm_idx_to_tags:
                    lm_row = dataset._landmark_metadata.iloc[lm_idx]
                    pruned = lm_row.get("pruned_props", frozenset())
                    if not pruned:
                        continue
                    osm_lm_idx_to_tags[lm_idx] = dict(pruned)
                if lm_idx in osm_lm_idx_to_tags:
                    sat_osm_indices.append(lm_idx)

            if sat_osm_indices:
                sat_to_osm_lm_idxs[sat_idx] = sat_osm_indices

        if not osm_lm_idx_to_tags:
            continue

        # Build ordered list of unique OSM landmarks
        unique_osm_lm_idxs = sorted(osm_lm_idx_to_tags.keys())
        osm_lm_idx_to_pos = {idx: pos for pos, idx in enumerate(unique_osm_lm_idxs)}
        osm_tags_list = [osm_lm_idx_to_tags[idx] for idx in unique_osm_lm_idxs]

        # Single batched forward pass for this panorama
        try:
            full_cost = compute_cost_matrix(
                pano_tags_list, osm_tags_list, model, text_embeddings,
                text_input_dim, device,
            )
        except (KeyError, ValueError) as e:
            if not _missing_warned:
                warnings.warn(f"Skipping pano {pano_id}: {e}")
                _missing_warned = True
            continue

        # Compute uniqueness weights from the full cost matrix
        u_weights = compute_uniqueness_weights(full_cost, prob_threshold) if uniqueness_weighted else None

        # Partition by satellite and run matching
        for sat_idx, sat_osm_lm_idxs in sat_to_osm_lm_idxs.items():
            osm_positions = [osm_lm_idx_to_pos[i] for i in sat_osm_lm_idxs]
            sub_cost = full_cost[:, osm_positions]
            result = match_and_aggregate(sub_cost, method, aggregation, prob_threshold,
                                         uniqueness_weights=u_weights)
            similarity[pano_idx, sat_idx] = result.similarity_score

    return similarity
