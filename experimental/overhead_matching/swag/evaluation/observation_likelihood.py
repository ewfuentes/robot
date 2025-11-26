
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import geopandas as gpd
import pandas as pd
import itertools
import shapely
from common.gps import web_mercator
import math
import numpy as np
import hashlib
import json

from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_embeddings, custom_id_from_props, load_all_jsonl_from_folder,
    make_sentence_dict_from_json, make_sentence_dict_from_pano_jsons)


class LikelihoodMode(Enum):
    SAT_ONLY = "sat_only"
    OSM_ONLY = "osm_only"
    COMBINED = "combined"


@dataclass
class ObservationLikelihoodConfig:
    obs_likelihood_from_sat_similarity_sigma: float
    obs_likelihood_from_osm_similarity_sigma: float
    osm_similarity_scale: float = 10.0
    likelihood_mode: LikelihoodMode = LikelihoodMode.COMBINED


@dataclass
class PriorData:
    # The osm_geometry must have the following columns: [osm_id, geometry_px, osm_embedding_idx]
    osm_geometry: gpd.GeoDataFrame

    # The sat_geometry must have the following columns: [geometry_px, embedding_idx]
    # It is expected that the number of rows correspond directly to the rows of the
    # sat_embeddings tensor.
    sat_geometry: gpd.GeoDataFrame

    # This dataframe must have the following columns: [pano_id, pano_lm_idxs]
    pano_metadata: gpd.GeoDataFrame

    # This similarity matrix is num_panoramas x num_sat_patches
    pano_sat_similarity: torch.Tensor

    # This similarity matrix is num_panorama_landmarks x num_unique_osm_landmarks
    pano_osm_landmark_similarity: torch.Tensor


@dataclass
class QueryData:
    pano_ids: list[str]
    # A num_particles x 2 tensor that contains lat/lon locations
    particle_locs_deg: torch.Tensor


@dataclass
class Similarities:
    # num_panos x num_satellite_patches
    sat_patch: torch.Tensor
    # num_panos x num_pano_landmarks x num_osm_landmarks
    landmark: torch.Tensor
    # num_panos x num_pano_landmarks
    # True means that it is a valid landmark
    landmark_mask: torch.Tensor


def _get_similarities(prior_data: PriorData, pano_ids: list[str]) -> Similarities:
    if len(pano_ids) == 0:
        return Similarities(
            sat_patch=torch.zeros((0, prior_data.pano_sat_similarity.shape[1])),
            landmark=torch.zeros((0, 0, prior_data.pano_osm_landmark_similarity.shape[1])),
            landmark_mask=torch.zeros((0, 0), dtype=torch.bool)
        )

    pano_metadata = pd.concat([
        prior_data.pano_metadata[prior_data.pano_metadata.pano_id == p]
        for p in pano_ids
    ])
    assert len(pano_metadata) == len(pano_ids)

    sat_patch_similarities = prior_data.pano_sat_similarity[pano_metadata.index]

    num_panos = len(pano_ids)
    max_num_pano_landmarks = pano_metadata.pano_lm_idxs.apply(len).max()
    max_num_pano_landmarks = max_num_pano_landmarks if len(pano_metadata) > 0 else 0
    num_osm_landmarks = len(prior_data.osm_geometry)
    landmark_similarities = torch.zeros((num_panos, max_num_pano_landmarks, num_osm_landmarks),
                                        dtype=torch.float32)
    landmark_mask = torch.zeros((num_panos, max_num_pano_landmarks), dtype=torch.bool)

    for pano_idx, (_, row) in enumerate(pano_metadata.iterrows()):
        num_pano_lms = len(row.pano_lm_idxs)
        landmark_similarities[pano_idx, :num_pano_lms] = prior_data.pano_osm_landmark_similarity[:,prior_data.osm_geometry.osm_embedding_idx][row.pano_lm_idxs]
        landmark_mask[pano_idx, :num_pano_lms] = True

    return Similarities(
        sat_patch=sat_patch_similarities,
        landmark=landmark_similarities,
        landmark_mask=landmark_mask
    )


def _compute_pixel_locs_px(particle_locs_deg: torch.Tensor, zoom_level: int = 20):
    y_px, x_px = web_mercator.latlon_to_pixel_coords(
            particle_locs_deg[..., 0], particle_locs_deg[..., 1], zoom_level=zoom_level)
    return torch.stack([y_px, x_px], dim=-1)


def _build_sat_spatial_index(sat_geometry):
    """Build spatial index and centroids for satellite patches.

    Args:
        sat_geometry: GeoDataFrame with geometry_px column containing patch polygons

    Returns:
        sat_tree: STRtree for spatial queries, or None if no patches
        patch_centroids: Array of patch centroids, or None if no patches
    """
    if len(sat_geometry) == 0:
        return None, None

    sat_tree = shapely.STRtree(sat_geometry.geometry_px)
    patch_centroids = sat_geometry.geometry_px.apply(lambda x: x.centroid).values
    return sat_tree, patch_centroids


def _query_particle_patch_mapping(particle_locs_px: torch.Tensor, sat_tree, patch_centroids):
    """
    Query the spatial index to find which patch each particle belongs to.

    For particles that overlap multiple patches, returns the nearest patch
    by centroid distance.

    Args:
        particle_locs_px: Particle locations in pixels, shape (..., 2)
        sat_tree: STRtree built from satellite patch geometries
        patch_centroids: Array of patch centroids

    Returns:
        kept_particle_idxs: Indices of particles that matched a patch
        kept_patch_idxs: Corresponding patch indices for each matched particle
    """
    if sat_tree is None:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    query_pts = np.array([shapely.Point(x) for x in particle_locs_px.reshape(-1, 2)])
    pt_and_patch_idxs = sat_tree.query(query_pts).T

    if len(pt_and_patch_idxs) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    flat_particle_idxs = torch.tensor(pt_and_patch_idxs[:, 0])
    particle_match_idxs = torch.tensor(pt_and_patch_idxs[:, 1])

    if len(flat_particle_idxs) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # For particles matching multiple patches, find the closest by centroid distance
    particle_to_center_distances = torch.tensor(
            shapely.distance(patch_centroids[particle_match_idxs],
                             query_pts[flat_particle_idxs]))

    # Mark beginning of each run of repeated particle indices
    run_starts = torch.ones(
            len(flat_particle_idxs), dtype=torch.bool, device=flat_particle_idxs.device)
    if len(flat_particle_idxs) > 1:
        run_starts[1:] = flat_particle_idxs[1:] != flat_particle_idxs[:-1]

    # Compute segment IDs (which run each element belongs to)
    segment_ids = torch.cumsum(run_starts, dim=0) - 1

    # Find minimum distance per segment
    num_segments = segment_ids[-1].item() + 1 if len(segment_ids) > 0 else 0
    min_distances = torch.full((num_segments,), float('inf'),
                               device=particle_to_center_distances.device,
                               dtype=particle_to_center_distances.dtype)
    min_distances.scatter_reduce_(0, segment_ids, particle_to_center_distances, reduce='amin')

    # Mark which elements to keep (those with minimum distance in their segment)
    segment_min_distances = min_distances[segment_ids]
    keep_mask = particle_to_center_distances == segment_min_distances

    # Keep only the FIRST occurrence of minimum per segment
    first_keep_idx_per_segment = torch.full((num_segments,), -1,
                                            dtype=torch.long, device=segment_ids.device)

    # Get indices where keep_mask is True
    true_indices = torch.where(keep_mask)[0]
    if len(true_indices) > 0:
        true_segments = segment_ids[true_indices]
        # Use scatter_reduce with 'amin' to get minimum index per segment
        first_keep_idx_per_segment.scatter_reduce_(
                0, true_segments, true_indices, reduce='amin', include_self=False)

    # Create new keep_mask: only True at the first kept index per segment
    keep_mask = torch.zeros_like(keep_mask)
    valid_segments = first_keep_idx_per_segment >= 0
    if valid_segments.any():
        keep_mask[first_keep_idx_per_segment[valid_segments]] = True

    # Filter to get final results
    kept_particle_idxs = flat_particle_idxs[keep_mask]
    kept_patch_idxs = particle_match_idxs[keep_mask]

    return kept_particle_idxs, kept_patch_idxs


def _compute_sat_log_likelihood(similarities: torch.Tensor,
                                particle_locs_px: torch.Tensor,
                                sat_tree,
                                patch_centroids,
                                sigma_px: float = 0.1) -> torch.Tensor:
    """
    Compute satellite patch observation log likelihood.

    Args:
        similarities: (num_panos, num_sat_patches) similarity matrix
        particle_locs_px: (num_panos, num_particles, 2) particle locations in pixels
        sat_tree: STRtree built from satellite patch geometries
        patch_centroids: Array of patch centroids
        sigma_px: Sigma parameter for the Gaussian likelihood

    Returns:
        log_likelihood: (num_panos, num_particles) unnormalized log likelihood
    """
    assert similarities.shape[0] == particle_locs_px.shape[0]
    assert particle_locs_px.shape[-1] == 2

    num_patches = similarities.shape[1]
    if num_patches == 0:
        return torch.full((particle_locs_px.shape[:-1]), -torch.inf, dtype=torch.float32)

    max_similarities, _ = torch.max(similarities, -1, keepdim=True)
    obs_log_likelihood = (-torch.log(torch.tensor(math.sqrt(2 * torch.pi) * sigma_px)) +
                          -0.5 * torch.square((max_similarities - similarities) / sigma_px))

    # shapely uses a col, row ordering where as the particles use a row column ordering.
    # swap the order for the particles here
    swapped_particles = torch.roll(particle_locs_px, 1, dims=-1)
    kept_particle_idxs, kept_patch_idxs = _query_particle_patch_mapping(
        swapped_particles, sat_tree, patch_centroids)

    if len(kept_particle_idxs) == 0:
        return torch.full((particle_locs_px.shape[:-1]), -torch.inf, dtype=torch.float32)

    out = torch.full(particle_locs_px.shape[:-1], -torch.inf)
    particle_idxs = torch.unravel_index(kept_particle_idxs, particle_locs_px.shape[:-1])

    if len(particle_idxs) > 0:
        pano_idxs = particle_idxs[0]
        particle_likelihoods = obs_log_likelihood[pano_idxs, kept_patch_idxs]
        out[particle_idxs] = particle_likelihoods

    return out


def _compute_osm_log_likelihood(similarities, mask, osm_geometry, particle_locs_px, point_sigma_px: float, similarity_scale: float = 1.0, selected_pano_landmark_idx: int | None = None):
    # Similarities is a num_panos x num_pano_landmarks x num_osm_landmark tensor
    # Mask is a num_panos x num_pano_landmarks tensor where true means that it's a valid landmark
    # osm_geometry is a dataframe that contains the geometry for each osm landmark
    # particle_locs_px is a num_panos x num_particles x 2 tensor
    # similarity_scale: factor to multiply similarities by before adding to log-likelihood
    # selected_pano_landmark_idx: if provided, only compute likelihood for this pano landmark

    assert similarities.ndim == 3
    assert mask.ndim == 2
    assert particle_locs_px.ndim == 3
    assert similarities.shape[0] == mask.shape[0]
    assert similarities.shape[0] == particle_locs_px.shape[0]
    assert similarities.shape[1] == mask.shape[1]
    assert similarities.shape[2] == len(osm_geometry)

    is_point = osm_geometry.geometry_px.apply(lambda x: x.geom_type == "Point")
    # assert is_point.all()
    num_panos, num_particles = particle_locs_px.shape[:2]
    num_osm = len(osm_geometry)

    # For each particle, compute the distance to the OSM geometry
    # Shapely objects have the column, row ordering, so we swap the particle order here
    swapped_particles = torch.roll(particle_locs_px, 1, dims=-1)
    particles_flat = [shapely.Point(*x) for x in swapped_particles.reshape(-1, 2)]
    particles = np.array(particles_flat).reshape(num_panos, num_particles, 1)
    osm_landmarks = osm_geometry.geometry_px.values.reshape(1, 1, num_osm)
    distances = shapely.distance(particles, osm_geometry.geometry_px.values)
    # Handle the case where we may only have a single particle or a single osm landmark
    distances = distances.reshape(num_panos, num_particles, num_osm)

    # Compute the observation likelihood for each particle/geometry pair
    # dimensions: num_panos x num_particles x num_osm
    obs_log_likelihood_per_landmark = (
            -torch.log(torch.tensor(math.sqrt(2 * torch.pi) * point_sigma_px)) +
            -0.5 * torch.square(torch.tensor(distances) / point_sigma_px))
    obs_log_likelihood_per_landmark[obs_log_likelihood_per_landmark < -14.0] = -14.0

    # compute the weights
    # dimensions: num_panos x num_pano_landmarks x num_osm_landmarks
    weight = similarities * similarity_scale
    weight[~mask, :] = -torch.inf

    # We want both tensors to be num_panos x num_particles x num_pano_landmarks x num_osm
    # So we insert a singular dimension for the observation likelihoods, and
    # we insert all num_particles dimensions to the weights
    obs_log_likelihood_per_landmark = obs_log_likelihood_per_landmark.unsqueeze(-2)
    weight = weight.unsqueeze(1)
    obs_log_likelihood_per_landmark = weight + obs_log_likelihood_per_landmark

    # logsumexp over the osm landmarks
    # num_panos x num_particles x num_pano_landmarks
    per_pano_lm = torch.logsumexp(obs_log_likelihood_per_landmark, dim=-1)
    per_pano_lm = torch.where(mask.unsqueeze(1), per_pano_lm, torch.tensor(0.0))

    # If a specific landmark is selected, return only that landmark's contribution
    if selected_pano_landmark_idx is not None:
        # Validate index
        num_pano_landmarks = per_pano_lm.shape[2]
        if selected_pano_landmark_idx >= num_pano_landmarks:
            raise ValueError(f"selected_pano_landmark_idx {selected_pano_landmark_idx} >= num_pano_landmarks {num_pano_landmarks}")
        out = per_pano_lm[:, :, selected_pano_landmark_idx]
    else:
        # Sum over all pano landmarks
        out = torch.sum(per_pano_lm, dim=-1)

    return out


def _check_prior_data(prior_data: PriorData):
    assert len(prior_data.pano_metadata) == prior_data.pano_sat_similarity.shape[0]
    assert len(prior_data.sat_geometry) == prior_data.pano_sat_similarity.shape[1]

    max_osm_idx = prior_data.osm_geometry["osm_embedding_idx"].max()
    max_pano_idx = max(list(itertools.chain(*prior_data.pano_metadata.pano_lm_idxs)))
    assert max_pano_idx < prior_data.pano_osm_landmark_similarity.shape[0]
    assert max_osm_idx < prior_data.pano_osm_landmark_similarity.shape[1]


@dataclass
class LandmarkSimilarityData:
    """Pre-computed landmark similarity data that can be cached."""
    # GeoDataFrame with columns: [osm_id, geometry_px, osm_embedding_idx]
    osm_geometry: gpd.GeoDataFrame
    # GeoDataFrame with columns: [pano_id, pano_lm_idxs]
    pano_metadata: gpd.GeoDataFrame
    # Similarity matrix: (num_pano_landmarks, num_osm_landmarks)
    pano_osm_landmark_similarity: torch.Tensor


def compute_landmark_similarity_data(
    vigor_dataset,
    osm_semantic_path: Path,
    pano_semantic_path: Path,
    embedding_dim: int = 1536,
) -> LandmarkSimilarityData:
    """Compute landmark similarity data from embeddings.

    This is the expensive operation that loads embeddings and computes similarities.

    Args:
        vigor_dataset: VigorDataset instance with loaded landmarks
        osm_embedding_path: Path to directory containing OSM landmark embeddings
        pano_embedding_path: Path to directory containing panorama landmark embeddings
        embedding_dim: Dimension to truncate embeddings to

    Returns:
        LandmarkSimilarityData with OSM geometry, pano metadata, and similarity matrix
    """
    landmark_metadata = vigor_dataset._landmark_metadata

    # Load OSM embeddings
    # Determine the correct paths for embeddings and sentences
    # The osm_embedding_path could be either:
    # - /path/to/data/embeddings (pointing directly to embeddings dir)
    # - /path/to/data (pointing to parent with embeddings/ and sentences/ subdirs)

    # Check if this is the embeddings directory itself or the parent
    osm_embeddings_dir = (osm_semantic_path / "embeddings")
    osm_sentences_dir = (osm_semantic_path / "sentences")
    
    assert osm_embeddings_dir.exists()
    assert osm_sentences_dir.exists()
    osm_embeddings, osm_id_to_idx = load_embeddings(
        osm_embeddings_dir,
        output_dim=embedding_dim,
        normalize=True
    )

    # Load OSM sentences
    osm_sentence_jsonl = load_all_jsonl_from_folder(osm_sentences_dir)
    osm_sentences, _ = make_sentence_dict_from_json(osm_sentence_jsonl)

    # Filter to landmarks that have embeddings and build OSM geometry
    osm_geometry_rows = []
    num_total_embeddings = len(landmark_metadata)
    num_with_sentences = 0
    num_with_embeddings = 0
    for idx, row in landmark_metadata.iterrows():
        custom_id = custom_id_from_props(row.pruned_props)
        sentence = osm_sentences.get(custom_id, '')

        if sentence:
            num_with_sentences += 1
        else:
            continue

        if custom_id in osm_id_to_idx:
            num_with_embeddings += 1
        else:
            continue

        osm_geometry_rows.append({
            'osm_id': row.id,
            'geometry_px': row.geometry_px,
            'osm_embedding_idx': osm_id_to_idx[custom_id],
            'geometry': row.geometry,  # Add lat/lon geometry for visualization
            'pruned_props': row.pruned_props,  # Add OSM properties
            'sentence': sentence
        })

    osm_geometry = gpd.GeoDataFrame(osm_geometry_rows)

    # Load panorama embeddings
    # Pano embeddings are keyed by "{panorama_id}__landmark_{idx}" format
    # where panorama_id = "{pano_id},{lat},{lon},"
    pano_embeddings, pano_custom_id_to_idx = load_embeddings(
        pano_semantic_path / "embeddings",
        output_dim=embedding_dim,
        normalize=True
    )

    # Load panorama sentences
    pano_sentences = {}
    pano_sentences_dir = pano_semantic_path / "sentences"
    pano_sentence_jsonl = load_all_jsonl_from_folder(pano_sentences_dir)
    pano_sentences, pano_sentence_metadata, _ = make_sentence_dict_from_pano_jsons(pano_sentence_jsonl)

    # Load panorama landmark metadata to get landmark info per panorama
    pano_lm_metadata_path = pano_semantic_path / "embedding_requests" / "panorama_metadata.jsonl"
    pano_lm_jsons = []
    with open(pano_lm_metadata_path, 'r') as f:
        for line in f:
            pano_lm_jsons.append(json.loads(line))

    # Group landmarks by pano_id (without GPS coordinates)
    # panorama_id format is "{pano_id},{lat},{lon}," - extract just pano_id
    pano_lm_by_pano_id = {}
    for entry in pano_lm_jsons:
        panorama_id = entry['panorama_id']
        pano_id = panorama_id.split(',')[0]  # Extract just the pano_id
        if pano_id not in pano_lm_by_pano_id:
            pano_lm_by_pano_id[pano_id] = []
        pano_lm_by_pano_id[pano_id].append(entry)

    # Build pano metadata with landmark indices into pano_embeddings tensor
    pano_metadata = vigor_dataset._panorama_metadata
    pano_metadata_rows = []


    for pano_idx, row in pano_metadata.iterrows():
        # Extract pano_id without GPS coordinates
        pano_id = row["pano_id"]
        pano_lm_idxs = []
        pano_lm_sentences = []

        # Get landmarks for this panorama from metadata
        if pano_id in pano_lm_by_pano_id:
            # Sort by landmark_idx to ensure correct order
            landmarks = sorted(pano_lm_by_pano_id[pano_id], key=lambda x: x['landmark_idx'])

            for lm_entry in landmarks:
                custom_id = lm_entry['custom_id']
                if custom_id in pano_custom_id_to_idx:
                    pano_lm_idxs.append(pano_custom_id_to_idx[custom_id])
                pano_lm_sentences.append(pano_sentences[custom_id])

        pano_metadata_rows.append({
            'pano_id': pano_id,
            'pano_lm_idxs': pano_lm_idxs,
            'pano_lm_sentences': pano_lm_sentences
        })

    pano_metadata_df = gpd.GeoDataFrame(pano_metadata_rows)

    # Compute pano_osm_landmark_similarity
    # pano_embeddings is already the full tensor, compute similarity against all OSM embeddings
    # Compute in float16 to save memory (100k x 100k = 10B elements)
    cuda_pano_embeddings = pano_embeddings.cuda()
    cuda_osm_embeddings = osm_embeddings.cuda()

    pano_osm_similarity = torch.empty(
            (len(pano_embeddings), len(osm_embeddings)), dtype=torch.float16)
    BATCH_SIZE = 4096
    for start_idx in range(0, len(cuda_osm_embeddings), BATCH_SIZE):
        batch = cuda_osm_embeddings[start_idx:start_idx+BATCH_SIZE]
        num_items = batch.shape[0]
        pano_osm_similarity[:, start_idx:start_idx+num_items] = (
                cuda_pano_embeddings @ batch.T).cpu().half()

    return LandmarkSimilarityData(
        osm_geometry=osm_geometry,
        pano_metadata=pano_metadata_df,
        pano_osm_landmark_similarity=pano_osm_similarity
    )


def _compute_landmark_similarity_hash(
    vigor_dataset,
    osm_embedding_path: Path,
    pano_embedding_path: Path,
    embedding_dim: int
) -> str:
    """Compute a hash for caching landmark similarity data."""
    import struct
    m = hashlib.sha256()

    # Hash panorama metadata (pano_ids)
    for i, row in vigor_dataset._panorama_metadata.iterrows():
        m.update(i.to_bytes(length=4, byteorder='little', signed=True))
        m.update(row.path.stem.encode())  # pano_id

    # Hash landmark metadata
    landmark_metadata = vigor_dataset._landmark_metadata
    for i, row in landmark_metadata.iterrows():
        m.update(i.to_bytes(length=4, byteorder='little', signed=True))
        # Hash geometry bounds as a proxy for the geometry
        bounds = row.geometry_px.bounds
        m.update(struct.pack('<dddd', *bounds))

    # Hash embedding paths and dimension
    m.update(str(osm_embedding_path.resolve()).encode())
    m.update(str(pano_embedding_path.resolve()).encode())
    m.update(embedding_dim.to_bytes(length=4, byteorder='little'))

    return m.hexdigest()[:16]


def compute_cached_landmark_similarity_data(
    vigor_dataset,
    osm_embedding_path: Path,
    pano_embedding_path: Path,
    embedding_dim: int = 1536,
    use_cache: bool = True,
) -> LandmarkSimilarityData:
    """Compute landmark similarity data with disk caching.

    Args:
        vigor_dataset: VigorDataset instance with loaded landmarks
        osm_embedding_path: Path to directory containing OSM landmark embeddings
        pano_embedding_path: Path to directory containing panorama landmark embeddings
        embedding_dim: Dimension to truncate embeddings to
        use_cache: Whether to use disk caching

    Returns:
        LandmarkSimilarityData with OSM geometry, pano metadata, and similarity matrix
    """
    cache_path = None
    if use_cache:
        cache_hash = _compute_landmark_similarity_hash(
            vigor_dataset, osm_embedding_path, pano_embedding_path, embedding_dim)
        cache_path = Path(f"~/.cache/robot/overhead_matching/landmark_similarity/{cache_hash}.pt").expanduser()

        if cache_path.exists():
            print(f"Loading cached landmark similarity data from {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            return LandmarkSimilarityData(
                osm_geometry=cached['osm_geometry'],
                pano_metadata=cached['pano_metadata'],
                pano_osm_landmark_similarity=cached['pano_osm_landmark_similarity']
            )

    data = compute_landmark_similarity_data(
        vigor_dataset, osm_embedding_path, pano_embedding_path, embedding_dim)

    if use_cache and cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'osm_geometry': data.osm_geometry,
            'pano_metadata': data.pano_metadata,
            'pano_osm_landmark_similarity': data.pano_osm_landmark_similarity
        }, cache_path)
        print(f"Cached landmark similarity data to {cache_path}")

    return data


def build_prior_data_from_vigor(
    vigor_dataset,
    pano_sat_similarity: torch.Tensor,
    landmark_similarity_data: LandmarkSimilarityData,
) -> PriorData:
    """Build PriorData from a VIGOR dataset and pre-computed similarity data.

    Args:
        vigor_dataset: VigorDataset instance with loaded landmarks
        pano_sat_similarity: Pre-computed (num_panos, num_sat_patches) similarity matrix
        landmark_similarity_data: Pre-computed landmark similarity data

    Returns:
        PriorData instance ready for use with LandmarkObservationLikelihoodCalculator
    """
    # Build satellite geometry with bounding boxes
    sat_metadata = vigor_dataset._satellite_metadata
    patch_height, patch_width = vigor_dataset._original_satellite_patch_size

    sat_geometries = []
    for _, row in sat_metadata.iterrows():
        center_x = row.web_mercator_x
        center_y = row.web_mercator_y
        geom = shapely.box(
            xmin=center_x - patch_width // 2,
            xmax=center_x + patch_width // 2,
            ymin=center_y - patch_height // 2,
            ymax=center_y + patch_height // 2
        )
        sat_geometries.append(geom)

    sat_geometry = gpd.GeoDataFrame({
        'geometry_px': sat_geometries,
        'embedding_idx': range(len(sat_metadata))
    })

    return PriorData(
        osm_geometry=landmark_similarity_data.osm_geometry,
        sat_geometry=sat_geometry,
        pano_metadata=landmark_similarity_data.pano_metadata,
        pano_sat_similarity=pano_sat_similarity,
        pano_osm_landmark_similarity=landmark_similarity_data.pano_osm_landmark_similarity
    )


class LandmarkObservationLikelihoodCalculator:
    """Observation likelihood calculator using satellite patches and OSM landmarks.

    This implements the observation likelihood model that combines satellite patch
    similarities and OSM landmark similarities based on the configured mode.
    """

    def __init__(self,
                 prior_data: PriorData,
                 config: ObservationLikelihoodConfig,
                 device: torch.device):
        """
        Initialize the landmark observation likelihood calculator.

        Args:
            prior_data: Contains geometry and similarity matrices
            config: Configuration with sigma values and likelihood mode
            device: PyTorch device for computation
        """
        self.prior_data = prior_data
        self.config = config
        self.device = device

        # Build panorama ID to index mapping
        self.pano_id_to_idx = {
            pano_id: idx for idx, pano_id in enumerate(prior_data.pano_metadata.pano_id)
        }

        # Validate prior data
        _check_prior_data(prior_data)

        # Build spatial index for satellite patches
        self._sat_tree, self._patch_centroids = _build_sat_spatial_index(prior_data.sat_geometry)

    def compute_log_likelihoods(self, particles: torch.Tensor, panorama_ids: list[str], selected_pano_landmark_idx: int | None = None) -> torch.Tensor:
        """
        Compute unnormalized log likelihoods for particles given observations.

        Args:
            particles: (num_particles, 2) tensor of particle states (lat/lon)
            panorama_ids: List of identifiers for the observations/panoramas
            selected_pano_landmark_idx: If provided, only compute OSM likelihood for this specific
                panorama landmark index (applies to OSM_ONLY and COMBINED modes)

        Returns:
            log_likelihoods: (num_panoramas, num_particles) tensor of unnormalized log likelihoods
        """
        # Get similarities for the specified panoramas
        similarities = _get_similarities(self.prior_data, panorama_ids)

        # Convert lat/lon to pixel coordinates
        # Need to expand particles to (num_panoramas, num_particles, 2) for the likelihood functions
        num_panoramas = len(panorama_ids)
        particle_locs_px = _compute_pixel_locs_px(particles)
        # Expand to (num_panoramas, num_particles, 2)
        particle_locs_px = particle_locs_px.unsqueeze(0).expand(num_panoramas, -1, -1)

        mode = self.config.likelihood_mode

        if mode == LikelihoodMode.SAT_ONLY:
            log_likelihood = _compute_sat_log_likelihood(
                similarities.sat_patch,
                particle_locs_px,
                self._sat_tree,
                self._patch_centroids,
                sigma_px=self.config.obs_likelihood_from_sat_similarity_sigma
            )
        elif mode == LikelihoodMode.OSM_ONLY:
            log_likelihood = _compute_osm_log_likelihood(
                similarities.landmark,
                similarities.landmark_mask,
                self.prior_data.osm_geometry,
                particle_locs_px,
                point_sigma_px=self.config.obs_likelihood_from_osm_similarity_sigma,
                similarity_scale=self.config.osm_similarity_scale,
                selected_pano_landmark_idx=selected_pano_landmark_idx
            )
        else:  # COMBINED
            sat_log_likelihood = _compute_sat_log_likelihood(
                similarities.sat_patch,
                particle_locs_px,
                self._sat_tree,
                self._patch_centroids,
                sigma_px=self.config.obs_likelihood_from_sat_similarity_sigma
            )
            osm_log_likelihood = _compute_osm_log_likelihood(
                similarities.landmark,
                similarities.landmark_mask,
                self.prior_data.osm_geometry,
                particle_locs_px,
                point_sigma_px=self.config.obs_likelihood_from_osm_similarity_sigma,
                similarity_scale=self.config.osm_similarity_scale,
                selected_pano_landmark_idx=selected_pano_landmark_idx
            )
            log_likelihood = sat_log_likelihood + osm_log_likelihood

        return log_likelihood

    def sample_from_observation(self, num_particles: int, panorama_ids: list[str],
                                generator: torch.Generator) -> torch.Tensor:
        """
        Sample particles from the observation likelihood distribution.

        Args:
            num_particles: Number of particles to sample per panorama
            panorama_ids: List of identifiers for the observations/panoramas
            generator: Random generator for sampling

        Returns:
            particles: (num_panoramas, num_particles, 2) sampled particles (lat/lon)

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError("sample_from_observation is not yet implemented")

