"""Tag-weight similarity: build similarity matrices from learned per-tag-key weights.

Extracted from the tag_weight_optimization.py marimo notebook. Provides:
- MatchData: compact per-panorama match data with CSR osm→sat expansion
- precompute_match_data(): builds MatchData from parquet tables + VigorDataset + vocabulary
- build_similarity_matrix(): builds full (num_panos, num_sats) similarity matrix
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from experimental.overhead_matching.swag.data import vigor_dataset as vd


class MatchData:
    """Compact per-panorama match data for on-the-fly score computation.

    Stores condensed (osm_idx, key_idx, count) matches per panorama as flat
    arrays, plus an osm_idx → sat_idx expansion table. Scores are computed
    on-the-fly in build_similarity_matrix — no (pano × sat × key) matrix needed.
    """

    def __init__(self, num_panos, num_sats, num_keys,
                 match_osm_idxs, match_key_idxs, match_counts,
                 pano_boundaries, osm_to_sat_idxs, osm_to_sat_offsets):
        self.num_panos = num_panos
        self.num_sats = num_sats
        self.num_keys = num_keys
        # Flat arrays over all condensed matches, sorted by pano_idx
        self.match_osm_idxs = match_osm_idxs        # int32 (num_condensed,)
        self.match_key_idxs = match_key_idxs        # int16 (num_condensed,)
        self.match_counts = match_counts             # int16 (num_condensed,)
        # (start, end) into match arrays for each pano_idx
        self.pano_boundaries = pano_boundaries       # list of (int, int)
        # Flat CSR for osm_idx → list of sat_idxs
        self.osm_to_sat_idxs = osm_to_sat_idxs      # int32 (num_osm_sat_pairs,)
        self.osm_to_sat_offsets = osm_to_sat_offsets  # int32 (max_osm_idx+2,)


def precompute_match_data(
    matches_path: str,
    sat_osm_path: str,
    vigor_dataset: vd.VigorDataset,
    vocab: list[str],
    chunk_rows: int = 10_000_000,
) -> MatchData:
    """Condense match parquet into compact per-panorama arrays.

    Groups by (pano_id, osm_idx, tag_key) in chunks, then builds flat
    arrays sorted by pano_idx.
    """
    import tempfile

    num_keys = len(vocab)
    key_to_idx = {k: i for i, k in enumerate(vocab)}
    pano_meta = vigor_dataset._panorama_metadata
    num_panos = len(pano_meta)
    num_sats = len(vigor_dataset._satellite_metadata)
    pano_id_to_idx = {row["pano_id"]: idx for idx, (_, row) in enumerate(pano_meta.iterrows())}

    # --- Build osm → sat CSR from small table ---
    sat_osm_df = pl.read_parquet(sat_osm_path).select("osm_idx", "sat_idx").unique()
    if len(sat_osm_df) == 0:
        raise ValueError(f"sat_osm table at {sat_osm_path} is empty — no OSM-satellite correspondences found")
    max_osm = sat_osm_df["osm_idx"].max()
    # Some osm_idxs in matches may not appear in sat table — size CSR to cover all
    max_osm_matches = (
        pl.scan_parquet(matches_path)
        .select(pl.col("osm_idx").max())
        .collect()
        .item()
    )
    if max_osm_matches is None:
        raise ValueError(f"Matches parquet at {matches_path} is empty — no pano-OSM matches found")
    max_osm = max(max_osm, max_osm_matches)
    osm_to_sat_lists = defaultdict(list)
    for osm_idx, sat_idx in zip(sat_osm_df["osm_idx"], sat_osm_df["sat_idx"]):
        osm_to_sat_lists[osm_idx].append(sat_idx)
    # Build CSR
    osm_to_sat_offsets = np.zeros(max_osm + 2, dtype=np.int32)
    all_sat_idxs = []
    for osm in range(max_osm + 1):
        sats = osm_to_sat_lists.get(osm, [])
        all_sat_idxs.extend(sats)
        osm_to_sat_offsets[osm + 1] = osm_to_sat_offsets[osm] + len(sats)
    osm_to_sat_flat = torch.tensor(all_sat_idxs, dtype=torch.int32)
    osm_to_sat_offsets = torch.tensor(osm_to_sat_offsets, dtype=torch.int32)
    del sat_osm_df, osm_to_sat_lists
    print(f"  osm→sat CSR: {len(osm_to_sat_flat)} entries, max_osm={max_osm}")

    # --- Condense matches: group by (pano_id, osm_idx, tag_key) in chunks ---
    total_rows = pl.scan_parquet(matches_path).select(pl.len()).collect().item()
    num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
    print(f"  Total matches: {total_rows}, condensing in {num_chunks} chunks")

    with tempfile.TemporaryDirectory(prefix="tag_weight_") as tmp_dir:
        chunk_paths = []
        for i in tqdm(range(num_chunks), desc="Condensing chunks"):
            offset = i * chunk_rows
            condensed = (
                pl.scan_parquet(matches_path)
                .slice(offset, chunk_rows)
                .filter(pl.col("tag_key").is_in(vocab))
                .select("pano_id", "osm_idx", "tag_key")
                .group_by("pano_id", "osm_idx", "tag_key")
                .len()
                .collect()
            )
            if len(condensed) > 0:
                chunk_path = Path(tmp_dir) / f"chunk_{i:04d}.parquet"
                condensed.write_parquet(chunk_path)
                chunk_paths.append(chunk_path)

        # --- Merge chunks and build flat arrays ---
        # Stream chunks into per-pano lists
        pano_matches = defaultdict(list)  # pano_idx -> list of (osm_idx, key_idx, count)
        total_condensed = 0
        skipped_rows = 0
        for chunk_path in tqdm(chunk_paths, desc="Merging chunks"):
            chunk = pl.read_parquet(chunk_path)
            total_condensed += len(chunk)
            for pano_id, osm_idx, tag_key, count in chunk.iter_rows():
                pano_idx = pano_id_to_idx.get(pano_id)
                if pano_idx is None:
                    skipped_rows += 1
                    continue
                kid = key_to_idx[tag_key]
                pano_matches[pano_idx].append((osm_idx, kid, count))
            del chunk
    if total_condensed > 0 and skipped_rows > 0:
        pct = skipped_rows / total_condensed * 100
        print(f"  WARNING: {skipped_rows}/{total_condensed} rows ({pct:.1f}%) had pano_ids not in dataset")
        if pct > 50:
            raise ValueError(
                f"{pct:.1f}% of rows had unmatched pano_ids — likely a dataset/parquet mismatch"
            )
    print(f"  Condensed rows: {total_condensed}, panos with matches: {len(pano_matches)}")

    # Note: same (pano, osm, key) can appear in multiple chunks — merge counts
    all_osm = []
    all_key = []
    all_count = []
    boundaries = []
    for p in range(num_panos):
        start = len(all_osm)
        if p in pano_matches:
            # Merge duplicates from different chunks
            merged = defaultdict(int)
            for osm_idx, kid, count in pano_matches[p]:
                merged[(osm_idx, kid)] += count
            for (osm_idx, kid), count in merged.items():
                all_osm.append(osm_idx)
                all_key.append(kid)
                all_count.append(count)
        boundaries.append((start, len(all_osm)))
    del pano_matches

    match_osm_idxs = torch.tensor(all_osm, dtype=torch.int32)
    match_key_idxs = torch.tensor(all_key, dtype=torch.int16)
    match_counts = torch.tensor(all_count, dtype=torch.int16)

    total_matches = len(match_osm_idxs)
    mem_mb = (match_osm_idxs.nbytes + match_key_idxs.nbytes + match_counts.nbytes) / 1e6
    print(f"  Final: {total_matches} condensed matches, {mem_mb:.0f} MB")

    return MatchData(
        num_panos=num_panos, num_sats=num_sats, num_keys=num_keys,
        match_osm_idxs=match_osm_idxs,
        match_key_idxs=match_key_idxs,
        match_counts=match_counts,
        pano_boundaries=boundaries,
        osm_to_sat_idxs=osm_to_sat_flat,
        osm_to_sat_offsets=osm_to_sat_offsets,
    )


def build_similarity_matrix(
    match_data: MatchData,
    theta: torch.Tensor,
    dedup_sat_keys: bool = False,
    no_count: bool = False,
) -> torch.Tensor:
    """Build full (num_panos, num_sats) similarity matrix from MatchData and weight vector.

    Args:
        match_data: precomputed MatchData
        theta: learned weight vector of shape (num_keys,)
        dedup_sat_keys: if True, each tag_key contributes at most once per
            (pano, sat) pair. When multiple OSM landmarks on the same satellite
            match the same tag_key, only the match with the highest
            weight is kept.
        no_count: if True, ignore pano landmark counts — each (pano, osm, key)
            match contributes theta[key] regardless of how many pano landmarks matched.

    Returns:
        Float32 tensor of shape (num_panos, num_sats)
    """
    md = match_data
    if theta.shape[0] != md.num_keys:
        raise ValueError(
            f"theta has {theta.shape[0]} elements but match_data has {md.num_keys} keys — "
            f"ensure weights were trained on the same vocabulary as the match data"
        )
    sim = torch.zeros(md.num_panos, md.num_sats, dtype=torch.float32)

    for p in tqdm(range(md.num_panos), desc="Building similarity matrix"):
        start, end = md.pano_boundaries[p]
        if start == end:
            continue
        osm_idxs = md.match_osm_idxs[start:end].long()
        key_idxs = md.match_key_idxs[start:end].long()
        if no_count:
            weights = theta[key_idxs]
        else:
            counts = md.match_counts[start:end].float()
            weights = theta[key_idxs] * counts

        o_starts = md.osm_to_sat_offsets[osm_idxs]
        o_ends = md.osm_to_sat_offsets[osm_idxs + 1]
        sizes = (o_ends - o_starts).long()
        if sizes.sum() == 0:
            continue
        exp_weights = torch.repeat_interleave(weights, sizes)
        exp_key_idxs = torch.repeat_interleave(key_idxs, sizes)
        cumsize = sizes.cumsum(0)
        cumsize_prev = torch.cat([torch.zeros(1, dtype=torch.long), cumsize[:-1]])
        rep_o_starts = torch.repeat_interleave(o_starts.long(), sizes)
        rep_cp = torch.repeat_interleave(cumsize_prev, sizes)
        flat_idx = rep_o_starts + torch.arange(sizes.sum(), dtype=torch.long) - rep_cp
        sat_idxs = md.osm_to_sat_idxs[flat_idx].long()

        if dedup_sat_keys:
            # Deduplicate: for each (sat_idx, key_idx) pair, keep max weight
            pair_ids = sat_idxs * md.num_keys + exp_key_idxs
            unique_pairs, inv = pair_ids.unique(return_inverse=True)
            deduped_weights = torch.zeros(len(unique_pairs), dtype=torch.float32)
            deduped_weights.scatter_reduce_(0, inv, exp_weights, reduce="amax")
            deduped_sat_idxs = unique_pairs // md.num_keys
            sim[p].scatter_add_(0, deduped_sat_idxs, deduped_weights)
        else:
            sim[p].scatter_add_(0, sat_idxs, exp_weights)

    return sim
