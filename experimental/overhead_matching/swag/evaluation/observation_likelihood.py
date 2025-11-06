
import common.torch.load_torch_deps
import torch
from dataclasses import dataclass
import geopandas as gpd
import itertools


@dataclass
class ObservationLikelihoodConfig:
    ...


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
    # A ... x 2 tensor that contains lat/lon locations
    particle_locs_deg: torch.Tensor


def _get_similarities(prior_data: PriorData, pano_ids: list[str]):
    mask = prior_data.pano_metadata.isin(pano_ids)
    requested_pano_metadata = prior_data.pano_metadata[mask]
    assert len(requested_pano_metadata) == len(pano_ids)
    return None, None


def _compute_pixel_locs_px(particle_locs_deg: torch.Tensor):
    return particle_locs_deg


def _compute_sat_log_likelihood(similarities, sat_geometry, particle_locs_px):
    return torch.full(particle_locs_px.shape[:-1], -torch.inf)


def _compute_osm_log_likelihood(similarities, osm_geometry, particle_locs_px):
    return torch.full(particle_locs_px.shape[:-1], -torch.inf)


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
    sat_patch_similarities, osm_similarities = _get_similarities(prior_data, query.pano_ids)

    # Convert the particle lat/lon locations to pixels
    particle_locs_px = _compute_pixel_locs_px(query.particle_locs_deg)

    # Compute the log observation likelihood for each particle
    sat_log_likelihood = _compute_sat_log_likelihood(
            sat_patch_similarities, prior_data.sat_geometry, particle_locs_px)
    osm_log_likelihood = _compute_osm_log_likelihood(
            osm_similarities, prior_data.osm_geometry, particle_locs_px)

    return sat_log_likelihood, osm_log_likelihood

