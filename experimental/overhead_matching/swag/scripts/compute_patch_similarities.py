#!/usr/bin/env python3
"""
Compute panorama-to-satellite-patch similarities using MaxSim heuristic (SIMPLIFIED VERSION).

This script:
1. Loads VIGOR dataset metadata and spatial relationships
2. Loads panorama and OSM embeddings
3. Computes MaxSim similarities for all pano-patch pairs (scores only, no contribution tracking)
4. Exports JSON files for the spoofed correspondence viewer

Output files:
- pano_patch_similarities.json: Per-panorama top patches with basic scores (no contribution details)
- patch_index.json: Satellite patch metadata (minimal)
- similarity_statistics.json: Global statistics for histogram

OPTIMIZATIONS:
- No contribution tracking (no OSM match details, sentences, tags, locations)
- No second pass for detailed contributions
- Minimal memory usage (~50MB for 25k panos vs ~6GB with contributions)
- Fast processing (~0.8s per pano vs ~1.2s with contributions)
"""

import common.torch.load_torch_deps
import sys
import json
import pickle
import hashlib
import base64
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkb

import torch
import torch.nn.functional as F

# Import VIGOR dataset utilities
from experimental.overhead_matching.swag.data.vigor_dataset import (
    load_satellite_metadata,
    load_panorama_metadata,
    load_landmark_geojson,
    compute_satellite_from_panorama,
    compute_satellite_from_landmarks,
    compute_panorama_from_landmarks,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import prune_landmark

# Configuration
CITY = "Chicago"
ZOOM_LEVEL = 20
SAT_PATCH_SIZE = (640, 640)  # Original satellite patch size
PANO_LANDMARK_RADIUS_PX = 640  # Radius for panorama-landmark association
TOP_K_PERCENT = 0.10  # Store top 10% of patches per panorama
THRESHOLD = 0.0  # Similarity threshold (0.0 = no filtering)

# Paths
VIGOR_BASE = Path("/data/overhead_matching/datasets/VIGOR")
EMBEDDINGS_BASE = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings")
OUTPUT_BASE = Path("/home/ekf/scratch/visualizers_crossview/spoofed_correspondence_viewer/static/data/chicago")


def load_embeddings(embeddings_dir: Path) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load embeddings from pickle file."""
    pkl_path = embeddings_dir / "embeddings" / "embeddings.pkl"
    print(f"Loading embeddings from {pkl_path}...")

    with open(pkl_path, 'rb') as f:
        embeddings_tensor, id_to_idx = pickle.load(f)

    print(f"  Loaded {embeddings_tensor.shape[0]} embeddings of dimension {embeddings_tensor.shape[1]}")
    return embeddings_tensor, id_to_idx


# Sentence loading functions removed - not needed without contribution tracking


def custom_id_from_props(props) -> str:
    """Generate custom_id for OSM landmarks."""
    json_props = json.dumps(dict(props), sort_keys=True)
    custom_id = base64.b64encode(
        hashlib.sha256(json_props.encode('utf-8')).digest()
    ).decode('utf-8')
    return custom_id


def build_landmark_to_custom_id_mapping(
    landmark_metadata: gpd.GeoDataFrame
) -> Dict[int, str]:
    """
    Build minimal mapping from landmark index to custom_id (for embeddings lookup only).

    Returns:
        landmark_idx_to_custom_id: Map landmark index -> custom_id
    """
    print("Building landmark index to custom_id mappings...")

    landmark_idx_to_custom_id = {}

    # Create a mapping from OSM id to row index in landmark_metadata
    # The landmark_metadata has the pruned_props already
    for lm_idx, lm_row in landmark_metadata.iterrows():
        pruned_props = lm_row['pruned_props']

        if not pruned_props:
            continue

        # Generate custom_id
        custom_id = custom_id_from_props(pruned_props)
        landmark_idx_to_custom_id[lm_idx] = custom_id

    print(f"  Built mappings for {len(landmark_idx_to_custom_id)} landmarks")

    return landmark_idx_to_custom_id


# compute_maxsim_similarity() function removed - not needed without contribution tracking


def compute_maxsim_similarity_batch_scores_only(
    pano_embeddings: torch.Tensor,
    patch_osm_data: List[Dict],
    threshold: float = 0.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute MaxSim similarity scores only (no contributions tracking) for speed.

    Args:
        pano_embeddings: Tensor [n_pano_landmarks, embed_dim] on device
        patch_osm_data: List of dicts with 'embeddings' key containing OSM embeddings per patch
        threshold: Optional threshold to exclude low similarities
        device: Device to use for computation

    Returns:
        Tensor of scores [n_patches]
    """
    if len(pano_embeddings) == 0:
        return torch.zeros(len(patch_osm_data), device=device)

    # Normalize pano embeddings once
    pano_norm = F.normalize(pano_embeddings, dim=-1)  # [n_pano, embed_dim]

    # Pre-allocate scores
    scores = torch.zeros(len(patch_osm_data), device=device)

    # Process all patches
    for patch_idx, patch_data in enumerate(patch_osm_data):
        osm_embs = patch_data['embeddings']

        if len(osm_embs) == 0:
            continue

        # Normalize OSM embeddings
        osm_norm = F.normalize(osm_embs, dim=-1)

        # Compute similarity matrix: [n_pano, n_osm]
        sim_matrix = torch.matmul(pano_norm, osm_norm.T)

        # For each pano landmark, find max similarity
        max_similarities = sim_matrix.max(dim=1)[0]

        # Apply threshold
        if threshold > 0:
            max_similarities = torch.where(
                max_similarities >= threshold,
                max_similarities,
                torch.zeros_like(max_similarities)
            )

        # Store score
        scores[patch_idx] = max_similarities.sum()

    return scores


# compute_maxsim_similarity_batch() function removed - not needed without contribution tracking


def main(max_panos: int = None):
    """Main processing loop."""
    print("=" * 80)
    print("Computing Panorama-Satellite Patch Similarities")
    print("=" * 80)
    print(f"City: {CITY}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Top K%: {TOP_K_PERCENT * 100}%")
    if max_panos:
        print(f"Max Panoramas: {max_panos} (testing mode)")
    print()

    # Create output directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load VIGOR dataset metadata
    # -------------------------------------------------------------------------
    print("Step 1: Loading VIGOR dataset metadata...")

    sat_path = VIGOR_BASE / CITY / "satellite"
    pano_path = VIGOR_BASE / CITY / "panorama"
    landmark_path = VIGOR_BASE / CITY / "landmarks" / "v4_202001.geojson"

    sat_metadata = load_satellite_metadata(sat_path, ZOOM_LEVEL)
    pano_metadata = load_panorama_metadata(pano_path, ZOOM_LEVEL)
    landmark_metadata = load_landmark_geojson(landmark_path, ZOOM_LEVEL)

    print(f"  Satellite patches: {len(sat_metadata)}")
    print(f"  Panoramas: {len(pano_metadata)}")
    print(f"  OSM landmarks: {len(landmark_metadata)}")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Compute spatial relationships
    # -------------------------------------------------------------------------
    print("Step 2: Computing spatial relationships...")

    from scipy.spatial import cKDTree
    sat_kdtree = cKDTree(sat_metadata[["web_mercator_x", "web_mercator_y"]].values)

    # Panorama to satellite patches
    sat_from_pano = compute_satellite_from_panorama(
        sat_kdtree, sat_metadata, pano_metadata,
        sat_original_size=SAT_PATCH_SIZE
    )

    # OSM landmarks to satellite patches
    sat_from_landmarks = compute_satellite_from_landmarks(
        sat_metadata, landmark_metadata, SAT_PATCH_SIZE
    )

    # Panorama to OSM landmarks
    pano_from_landmarks = compute_panorama_from_landmarks(
        pano_metadata, landmark_metadata, PANO_LANDMARK_RADIUS_PX
    )

    print("  Spatial relationships computed")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Load embeddings
    # -------------------------------------------------------------------------
    print("Step 3: Loading embeddings...")

    # Load OSM embeddings
    osm_emb_dir = EMBEDDINGS_BASE / "sat_spoof"
    osm_embeddings, osm_id_to_idx = load_embeddings(osm_emb_dir)

    # Load panorama embeddings
    pano_emb_dir = EMBEDDINGS_BASE / "pano_spoof" / CITY
    pano_embeddings, pano_id_to_idx = load_embeddings(pano_emb_dir)

    print()

    # -------------------------------------------------------------------------
    # Step 4: Build minimal custom_id mapping (for embeddings lookup only)
    # -------------------------------------------------------------------------
    print("Step 4: Building OSM custom_id mappings...")

    landmark_idx_to_custom_id = build_landmark_to_custom_id_mapping(landmark_metadata)

    print()

    # -------------------------------------------------------------------------
    # Step 5: Pre-organize OSM embeddings by patch (SIMPLIFIED)
    # -------------------------------------------------------------------------
    print("Step 5: Pre-organizing OSM embeddings by patch...")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Move embeddings to device
    osm_embeddings = osm_embeddings.to(device)
    pano_embeddings = pano_embeddings.to(device)

    # Pre-build OSM embeddings for each patch (embeddings only, no metadata)
    patch_osm_data = []
    for sat_idx in tqdm(range(len(sat_metadata)), desc="Organizing OSM embeddings by patch"):
        osm_landmark_idxs_in_patch = sat_from_landmarks.landmark_idxs_from_sat_idx[sat_idx]

        if len(osm_landmark_idxs_in_patch) == 0:
            patch_osm_data.append({
                'embeddings': torch.empty(0, osm_embeddings.shape[1], device=device)
            })
            continue

        # Map landmark indices to custom_ids
        osm_custom_ids_in_patch = [
            landmark_idx_to_custom_id.get(lm_idx)
            for lm_idx in osm_landmark_idxs_in_patch
        ]

        # Filter out None values and get embedding indices
        osm_emb_indices_in_patch = [
            osm_id_to_idx[cid]
            for cid in osm_custom_ids_in_patch
            if cid is not None and cid in osm_id_to_idx
        ]

        if len(osm_emb_indices_in_patch) == 0:
            patch_osm_data.append({
                'embeddings': torch.empty(0, osm_embeddings.shape[1], device=device)
            })
        else:
            # Get embeddings and keep on GPU
            osm_embs = osm_embeddings[osm_emb_indices_in_patch]
            patch_osm_data.append({
                'embeddings': osm_embs
            })

    print(f"  Pre-organized {len(patch_osm_data)} patches")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Compute similarities for all panoramas (SIMPLIFIED)
    # -------------------------------------------------------------------------
    print("Step 6: Computing similarities for all panoramas...")

    # Data structures for results
    pano_patch_similarities = {}
    all_pair_scores = []  # For histogram

    # Determine how many panoramas to process
    num_panos_to_process = min(max_panos, len(pano_metadata)) if max_panos else len(pano_metadata)

    # Process each panorama
    for pano_idx in tqdm(range(num_panos_to_process), desc="Processing panoramas"):
        pano_row = pano_metadata.iloc[pano_idx]
        pano_id = pano_row['path'].stem.split(',')[0]

        # Get positive satellite patches (true matches)
        positive_sat_idxs = sat_from_pano.positive_sat_idxs_from_pano_idx[pano_idx]

        # Get panorama landmark indices
        pano_landmark_idxs = pano_from_landmarks.landmark_idxs_from_pano_idx[pano_idx]

        # Get panorama embeddings
        pano_custom_ids = [
            f"{pano_id},{pano_row['lat']},{pano_row['lon']},__landmark_{lm_idx}"
            for lm_idx in range(len(pano_landmark_idxs))
        ]
        pano_emb_indices = [
            pano_id_to_idx[cid] for cid in pano_custom_ids if cid in pano_id_to_idx
        ]

        if len(pano_emb_indices) == 0:
            continue

        pano_embs = pano_embeddings[pano_emb_indices]

        # Compute scores only (fast, no contributions)
        scores_tensor = compute_maxsim_similarity_batch_scores_only(pano_embs, patch_osm_data, THRESHOLD, device)
        scores_cpu = scores_tensor.cpu().numpy()

        # Build patch scores list with scores and metadata
        patch_scores = []
        for sat_idx in range(len(sat_metadata)):
            sat_row = sat_metadata.iloc[sat_idx]
            is_true_match = sat_idx in positive_sat_idxs

            patch_scores.append({
                'sat_idx': int(sat_idx),
                'sat_filename': sat_row['path'].name,
                'lat': float(sat_row['lat']),
                'lon': float(sat_row['lon']),
                'score': float(scores_cpu[sat_idx]),
                'is_true_match': bool(is_true_match)
            })

            # Track for histogram
            all_pair_scores.append({
                'pano_id': pano_id,
                'sat_idx': int(sat_idx),
                'score': float(scores_cpu[sat_idx]),
                'is_true_match': bool(is_true_match)
            })

        # Sort by score to find top-K
        patch_scores.sort(key=lambda x: x['score'], reverse=True)

        # Compute ranks
        for rank, patch in enumerate(patch_scores, 1):
            patch['rank'] = rank

        # Keep only top K%
        top_k_count = max(1, int(len(patch_scores) * TOP_K_PERCENT))
        top_patches = patch_scores[:top_k_count]

        # Extract true matches
        true_matches = [
            {
                'sat_idx': int(p['sat_idx']),
                'sat_filename': p['sat_filename'],
                'score': float(p['score']),
                'rank': int(p['rank'])
            }
            for p in patch_scores if p['is_true_match']
        ]

        # Store results
        pano_patch_similarities[pano_id] = {
            'top_patches': top_patches,
            'true_matches': true_matches
        }

    print(f"  Processed {len(pano_patch_similarities)} panoramas")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Compute global statistics
    # -------------------------------------------------------------------------
    print("Step 7: Computing global statistics...")

    # Histogram bins
    bins = [(i/10, (i+1)/10) for i in range(10)]
    histogram_bins = []

    for bin_range in bins:
        true_pairs = [p for p in all_pair_scores if bin_range[0] <= p['score'] < bin_range[1] and p['is_true_match']]
        false_pairs = [p for p in all_pair_scores if bin_range[0] <= p['score'] < bin_range[1] and not p['is_true_match']]

        histogram_bins.append({
            'range': list(bin_range),
            'count_true': len(true_pairs),
            'count_false': len(false_pairs),
            'true_pairs': true_pairs[:100],  # Sample
            'false_pairs': false_pairs[:100]  # Sample
        })

    # Ranking statistics
    true_match_ranks = []
    for pano_id, data in pano_patch_similarities.items():
        for tm in data['true_matches']:
            true_match_ranks.append(tm['rank'])

    if len(true_match_ranks) > 0:
        ranking_statistics = {
            'mean_rank_of_true_matches': float(np.mean(true_match_ranks)),
            'median_rank_of_true_matches': float(np.median(true_match_ranks)),
            'recall_at_1': sum(1 for r in true_match_ranks if r == 1) / len(true_match_ranks),
            'recall_at_10': sum(1 for r in true_match_ranks if r <= 10) / len(true_match_ranks),
            'recall_at_100': sum(1 for r in true_match_ranks if r <= 100) / len(true_match_ranks),
            'mrr': float(np.mean([1.0 / r for r in true_match_ranks]))
        }
    else:
        ranking_statistics = {}

    similarity_statistics = {
        'histogram_bins': histogram_bins,
        'ranking_statistics': ranking_statistics,
        'total_panos': len(pano_metadata),
        'total_patches': len(sat_metadata),
        'threshold_used': THRESHOLD,
        'heuristic_used': 'maxsim'
    }

    print("  Statistics computed")
    print()

    # -------------------------------------------------------------------------
    # Step 8: Export JSON files
    # -------------------------------------------------------------------------
    print("Step 8: Exporting JSON files...")

    # Export pano_patch_similarities.json
    output_file = OUTPUT_BASE / "pano_patch_similarities.json"
    print(f"  Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(pano_patch_similarities, f, indent=2)

    # Export similarity_statistics.json
    output_file = OUTPUT_BASE / "similarity_statistics.json"
    print(f"  Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(similarity_statistics, f, indent=2)

    # Export patch_index.json (simplified)
    patch_index = {
        str(idx): {
            'sat_filename': sat_metadata.iloc[idx]['path'].name,
            'lat': float(sat_metadata.iloc[idx]['lat']),
            'lon': float(sat_metadata.iloc[idx]['lon'])
        }
        for idx in range(len(sat_metadata))
    }

    output_file = OUTPUT_BASE / "patch_index.json"
    print(f"  Writing {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(patch_index, f, indent=2)

    print()
    print("=" * 80)
    print("Processing complete!")
    print("=" * 80)
    print(f"Output files written to: {OUTPUT_BASE}")
    print()
    print("Summary:")
    print(f"  - Recall@1: {ranking_statistics.get('recall_at_1', 0):.3f}")
    print(f"  - Recall@10: {ranking_statistics.get('recall_at_10', 0):.3f}")
    print(f"  - Median Rank: {ranking_statistics.get('median_rank_of_true_matches', 0):.1f}")
    print(f"  - MRR: {ranking_statistics.get('mrr', 0):.3f}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Compute panorama-satellite patch similarities")
    parser.add_argument('--max-panos', type=int, default=None,
                        help='Maximum number of panoramas to process (for testing)')
    args = parser.parse_args()

    main(max_panos=args.max_panos)
