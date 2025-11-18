
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass
from enum import Enum
import geopandas as gpd
import pandas as pd
import itertools
import shapely
from common.gps import web_mercator
import math
import numpy as np


class LikelihoodMode(Enum):
    SAT_ONLY = "sat_only"
    OSM_ONLY = "osm_only"
    COMBINED = "combined"


@dataclass
class ObservationLikelihoodConfig:
    obs_likelihood_from_sat_similarity_sigma: float
    obs_likelihood_from_osm_similarity_sigma: float
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
        landmark_similarities[pano_idx, :num_pano_lms] = prior_data.pano_osm_landmark_similarity[row.pano_lm_idxs]
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

    kept_particle_idxs, kept_patch_idxs = _query_particle_patch_mapping(
        particle_locs_px, sat_tree, patch_centroids)

    if len(kept_particle_idxs) == 0:
        return torch.full((particle_locs_px.shape[:-1]), -torch.inf, dtype=torch.float32)

    out = torch.full(particle_locs_px.shape[:-1], -torch.inf)
    particle_idxs = torch.unravel_index(kept_particle_idxs, particle_locs_px.shape[:-1])

    if len(particle_idxs) > 0:
        pano_idxs = particle_idxs[0]
        particle_likelihoods = obs_log_likelihood[pano_idxs, kept_patch_idxs]
        out[particle_idxs] = particle_likelihoods

    return out


def _compute_osm_log_likelihood(similarities, mask, osm_geometry, particle_locs_px, point_sigma_px: float):
    # Similarities is a num_panos x num_pano_landmarks x num_osm_landmark tensor
    # Mask is a num_panos x num_pano_landmarks tensor where true means that it's a valid landmark
    # osm_geometry is a dataframe that contains the geometry for each osm landmark
    # particle_locs_px is a num_panos x num_particles x 2 tensor

    assert similarities.ndim == 3
    assert mask.ndim == 2
    assert particle_locs_px.ndim == 3
    assert similarities.shape[0] == mask.shape[0]
    assert similarities.shape[0] == particle_locs_px.shape[0]
    assert similarities.shape[1] == mask.shape[1]
    assert similarities.shape[2] == len(osm_geometry)

    is_point = osm_geometry.geometry_px.apply(lambda x: x.geom_type == "Point")
    assert is_point.all()
    num_panos, num_particles = particle_locs_px.shape[:2]
    num_osm = len(osm_geometry)

    # For each particle, compute the distance to the OSM geometry
    particles_flat = [shapely.Point(*x) for x in particle_locs_px.reshape(-1, 2)]
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

    # compute the weights
    # dimensions: num_panos x num_pano_landmarks x num_osm_landmarks
    weight = torch.log(similarities)
    weight[~mask, :] = -torch.inf

    # We want both tensors to be num_panos x num_particles x num_pano_landmarks x num_osm
    # So we insert a singular dimension for the observation likelihoods, and
    # we insert all num_particles dimensions to the weights
    obs_log_likelihood_per_landmark = obs_log_likelihood_per_landmark.unsqueeze(-2)
    weight = weight.unsqueeze(1)
    obs_log_likelihood_per_landmark = weight + obs_log_likelihood_per_landmark

    # logsumexp over the panorama and osm landmarks
    # num_panos x num_particles
    out = torch.logsumexp(obs_log_likelihood_per_landmark, (-1, -2))
    return out


def _check_prior_data(prior_data: PriorData):
    assert len(prior_data.pano_metadata) == prior_data.pano_sat_similarity.shape[0]
    assert len(prior_data.sat_geometry) == prior_data.pano_sat_similarity.shape[1]

    max_osm_idx = prior_data.osm_geometry["osm_embedding_idx"].max()
    max_pano_idx = max(list(itertools.chain(*prior_data.pano_metadata.pano_lm_idxs)))
    assert max_pano_idx < prior_data.pano_osm_landmark_similarity.shape[0]
    assert max_osm_idx < prior_data.pano_osm_landmark_similarity.shape[1]


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

    def compute_log_likelihoods(self, particles: torch.Tensor, panorama_ids: list[str]) -> torch.Tensor:
        """
        Compute unnormalized log likelihoods for particles given observations.

        Args:
            particles: (num_particles, 2) tensor of particle states (lat/lon)
            panorama_ids: List of identifiers for the observations/panoramas

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
                point_sigma_px=self.config.obs_likelihood_from_osm_similarity_sigma
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
                point_sigma_px=self.config.obs_likelihood_from_osm_similarity_sigma
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

