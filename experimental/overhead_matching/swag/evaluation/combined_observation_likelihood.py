"""Combined Observation Likelihood Calculator for satellite and OSM embedding fusion.

This module provides a likelihood calculator that combines satellite imagery and OSM
location embeddings via log-likelihood fusion. Both models produce location-level
embeddings at the same location grid, allowing efficient grid search over fusion
parameters (sigma values and weights).
"""

import common.torch.load_torch_deps
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import experimental.overhead_matching.swag.filter.particle_filter as pf


class CombinedLikelihoodMode(Enum):
    """Mode for combined observation likelihood calculation."""
    SAT_ONLY = "sat_only"
    OSM_ONLY = "osm_only"
    COMBINED = "combined"


@dataclass
class CombinedObservationLikelihoodConfig:
    """Configuration for combined observation likelihood calculator.

    These parameters are designed to be cheap to change during grid search,
    as they only affect the log-likelihood computation, not the precomputed
    similarity matrices.

    Attributes:
        mode: Which likelihood model(s) to use
        sat_sigma: Sigma for Gaussian log-likelihood conversion of satellite similarities
        osm_sigma: Sigma for Gaussian log-likelihood conversion of OSM similarities
        sat_weight: Weight for satellite log-likelihood in combined mode
        osm_weight: Weight for OSM log-likelihood in combined mode
    """
    mode: CombinedLikelihoodMode = CombinedLikelihoodMode.COMBINED
    sat_sigma: float = 0.1
    osm_sigma: float = 0.1
    sat_weight: float = 1.0
    osm_weight: float = 1.0


class CombinedObservationLikelihoodCalculator:
    """Observation likelihood calculator combining satellite and OSM embeddings.

    This calculator implements the ObservationLikelihoodCalculator protocol and
    combines satellite imagery embeddings with OSM location embeddings via
    log-likelihood fusion. Both models use the same location grid (satellite patch
    locations), enabling reuse of the patch_index_from_particle function.

    The fusion strategy is:
        combined_ll = sat_weight * sat_ll + osm_weight * osm_ll

    where sat_ll and osm_ll are computed using the WAG Gaussian log-likelihood
    formula based on similarity to the maximum similarity.

    Design for grid search:
        - Similarity matrices are precomputed once (expensive)
        - Config parameters (sigma, weights) can be changed cheaply
        - Creating a new calculator with different config is fast
    """

    def __init__(
        self,
        sat_similarity_matrix: torch.Tensor,
        osm_similarity_matrix: torch.Tensor,
        panorama_ids: list[str],
        satellite_patch_locations: torch.Tensor,
        patch_index_from_particle: Callable[[torch.Tensor], torch.Tensor],
        config: CombinedObservationLikelihoodConfig,
        device: torch.device,
    ):
        """Initialize combined observation likelihood calculator.

        Args:
            sat_similarity_matrix: (num_panoramas, num_patches) precomputed satellite
                similarities between panoramas and satellite patches
            osm_similarity_matrix: (num_panoramas, num_patches) precomputed OSM
                similarities between panoramas and OSM location embeddings.
                Must have the same shape as sat_similarity_matrix since both use
                the same location grid.
            panorama_ids: List of panorama IDs corresponding to rows of similarity matrices
            satellite_patch_locations: (num_patches, 2) lat/lon coordinates of patch centers
            patch_index_from_particle: Function mapping particle positions to patch indices.
                Used for both satellite and OSM since they share the same grid.
            config: Configuration for likelihood computation
            device: Device for torch operations
        """
        assert sat_similarity_matrix.shape == osm_similarity_matrix.shape, (
            f"Similarity matrices must have the same shape. "
            f"Got sat: {sat_similarity_matrix.shape}, osm: {osm_similarity_matrix.shape}"
        )

        self.sat_similarity_matrix = sat_similarity_matrix.to(device)
        self.osm_similarity_matrix = osm_similarity_matrix.to(device)
        self.pano_id_to_idx = {pano_id: i for i, pano_id in enumerate(panorama_ids)}
        self.satellite_patch_locations = satellite_patch_locations.to(device)
        self.patch_index_from_particle = patch_index_from_particle
        self.config = config
        self.device = device

    def compute_log_likelihoods(
        self, particles: torch.Tensor, panorama_ids: list[str]
    ) -> torch.Tensor:
        """Compute combined log likelihoods for particles given observations.

        Args:
            particles: (num_panoramas, num_particles, state_dim) tensor of particle states.
                Each panorama/trajectory has its own set of particles.
            panorama_ids: List of identifiers for the observations/panoramas, one per trajectory.

        Returns:
            log_likelihoods: (num_panoramas, num_particles) tensor of unnormalized
                log likelihoods. Each trajectory's particles are evaluated against
                that trajectory's observation.
        """
        assert particles.shape[0] == len(panorama_ids), (
            f"First dimension of particles ({particles.shape[0]}) must match "
            f"number of panorama_ids ({len(panorama_ids)})"
        )
        if self.config.mode == CombinedLikelihoodMode.SAT_ONLY:
            return self._compute_log_likelihood_for_matrix(
                particles, panorama_ids,
                self.sat_similarity_matrix,
                self.config.sat_sigma
            )
        elif self.config.mode == CombinedLikelihoodMode.OSM_ONLY:
            return self._compute_log_likelihood_for_matrix(
                particles, panorama_ids,
                self.osm_similarity_matrix,
                self.config.osm_sigma
            )
        else:  # COMBINED
            sat_ll = self._compute_log_likelihood_for_matrix(
                particles, panorama_ids,
                self.sat_similarity_matrix,
                self.config.sat_sigma
            )
            osm_ll = self._compute_log_likelihood_for_matrix(
                particles, panorama_ids,
                self.osm_similarity_matrix,
                self.config.osm_sigma
            )
            return self.config.sat_weight * sat_ll + self.config.osm_weight * osm_ll

    def _compute_log_likelihood_for_matrix(
        self,
        particles: torch.Tensor,
        panorama_ids: list[str],
        similarity_matrix: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Compute log likelihoods using a single similarity matrix.

        Args:
            particles: (num_panoramas, num_particles, state_dim) tensor of particle states.
                Each panorama/trajectory has its own set of particles.
            panorama_ids: List of panorama IDs, one per trajectory.
            similarity_matrix: (num_panoramas, num_patches) similarity matrix
            sigma: Sigma for Gaussian log-likelihood conversion

        Returns:
            (num_panoramas, num_particles) tensor of log likelihoods.
                Each trajectory's particles are evaluated against that trajectory's observation.
        """
        # Get similarity values for these panoramas
        pano_indices = [self.pano_id_to_idx[pano_id] for pano_id in panorama_ids]
        pano_similarities = similarity_matrix[pano_indices]  # (num_panoramas, num_patches)

        # Compute log likelihoods for each panorama
        particle_log_likelihoods_list = []
        for i in range(len(panorama_ids)):
            # Compute observation log likelihood from similarity for this panorama
            observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
                pano_similarities[i], sigma
            )  # (num_patches,)

            # Get particle likelihoods for this trajectory's particles
            particle_log_likelihoods = pf.wag_calculate_log_particle_weights(
                observation_log_likelihoods,
                particles[i],  # Use this trajectory's particles
                self.patch_index_from_particle
            )
            particle_log_likelihoods_list.append(particle_log_likelihoods)

        # Stack to get (num_panoramas, num_particles)
        return torch.stack(particle_log_likelihoods_list, dim=0)

    def sample_from_observation(
        self, num_particles: int, panorama_ids: list[str], generator: torch.Generator
    ) -> torch.Tensor:
        """Sample particles from observation likelihood distribution.

        Not implemented for combined calculator. For dual MCL, use
        WagObservationLikelihoodCalculator with a single fused embedding model.

        Raises:
            NotImplementedError: Always, as sampling is not supported.
        """
        raise NotImplementedError(
            "sample_from_observation is not implemented for CombinedObservationLikelihoodCalculator. "
            "For dual MCL sampling, use WagObservationLikelihoodCalculator with a fused embedding model."
        )
