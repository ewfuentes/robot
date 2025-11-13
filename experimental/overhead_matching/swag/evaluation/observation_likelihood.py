
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass
import geopandas as gpd
import pandas as pd
import itertools
import shapely
from common.gps import web_mercator
import math
import numpy as np


@dataclass
class ObservationLikelihoodConfig:
    obs_likelihood_from_sat_similarity_sigma: float
    obs_likelihood_from_osm_similarity_sigma: float


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


def _compute_sat_log_likelihood(similarities, sat_geometry, particle_locs_px, sigma_px=0.1):
    # Similarities is a n_panos x n_sat_patches similarity matrix
    # Sat geometry contains bounding box information for each satellite patch
    # particle_locs_px is a num_panos x num_particles x 2 (row, col) tensor of particle locations

    # This function returns a num_panos x num_particles shaped unnormalized log likelihood

    assert similarities.shape[0] == particle_locs_px.shape[0]
    assert particle_locs_px.shape[-1] == 2
    assert len(sat_geometry) == similarities.shape[1]

    if len(sat_geometry) == 0:
        return torch.full((particle_locs_px.shape[:-1]), -torch.inf, dtype=torch.float32)

    max_similarities, _ = torch.max(similarities, -1, keepdim=True)
    obs_log_likelihood = (-torch.log(torch.tensor(math.sqrt(2 * torch.pi) * sigma_px)) +
                          -0.5 * torch.square((max_similarities - similarities) / sigma_px))

    sat_tree = shapely.STRtree(sat_geometry.geometry_px)
    query_pts = [shapely.Point(x) for x in particle_locs_px.reshape(-1, 2)]
    pt_and_patch_idxs = sat_tree.query(query_pts).T
    particle_idxs = torch.unravel_index(torch.from_numpy(pt_and_patch_idxs[:, 0]), particle_locs_px.shape[:-1])
    particle_match_idxs = pt_and_patch_idxs[:, 1]

    out = torch.full(particle_locs_px.shape[:-1], -torch.inf)
    
    if pt_and_patch_idxs.shape[0] > 0:
        # some particles overlapped with a satellite patch, update the likelihoods to match
        pano_idxs = particle_idxs[0]
        particle_likelihoods = obs_log_likelihood[pano_idxs, particle_match_idxs]
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


def compute_log_observation_likelihood(prior_data, query, config):
    _check_prior_data(prior_data)
    # Given the panorama ids, get the relevant satellite patch
    # similarities and osm landmark similarities
    similarities = _get_similarities(prior_data, query.pano_ids)

    # Convert the particle lat/lon locations to pixels
    particle_locs_px = _compute_pixel_locs_px(query.particle_locs_deg)

    # Compute the log observation likelihood for each particle
    sat_log_likelihood = _compute_sat_log_likelihood(
            similarities.sat_patch, prior_data.sat_geometry, particle_locs_px)
    osm_log_likelihood = _compute_osm_log_likelihood(
            similarities.landmark,
            similarities.landmark_mask,
            prior_data.osm_geometry,
            particle_locs_px,
            point_sigma_px=config.obs_likelihood_from_osm_similarity_sigma)

    return sat_log_likelihood, osm_log_likelihood

