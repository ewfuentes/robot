import common.torch.load_torch_deps
import torch
from typing import NamedTuple
import experimental.overhead_matching.swag.filter.particle_filter as pf
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
import dataclasses
from typing import Callable, Protocol


@dataclasses.dataclass
class WagMeasurementResult:
    # In a WAG observation step, the particles are weighted according to the observation likelihood
    # and then resampled according to this likelihood.
    log_particle_weights: torch.Tensor | None

    # The last `num_dual_particles` of `resampled_particles` are the dual particles
    resampled_particles: torch.Tensor
    num_dual_particles: int


def initialize_wag_particles(gt_start_position_lat_lon: torch.Tensor,
                            wag_config: WagConfig,
                            generator: torch.Generator):
    # TODO: compensate for non-uniformity of latitude radius (who defined these reference frames)
    sampled_mean = torch.normal(mean=gt_start_position_lat_lon,
                                std=wag_config.initial_particle_distribution_offset_std_deg,
                                generator=generator).to(gt_start_position_lat_lon.device)
    particles = torch.normal(mean=torch.zeros((wag_config.num_particles, 2),
                                device=gt_start_position_lat_lon.device),
                             std=wag_config.initial_particle_distribution_std_deg,
                             generator=generator).to(gt_start_position_lat_lon.device)
    particles += sampled_mean
    return particles


class BeliefWeighting:
    """Computes particle weights based on prior belief distribution.

    Used for dual MCL weighting in WAG particle filter. The belief weighting
    is independent of the observation model - it only depends on where the
    prior particles are located in space.
    """

    def __init__(self,
                 satellite_patch_locations: torch.Tensor,
                 patch_index_from_particle: Callable[[torch.Tensor], torch.Tensor],
                 phantom_counts_frac: float):
        """
        Initialize belief weighting.

        Args:
            satellite_patch_locations: (num_patches, 2) lat/lon coordinates
            patch_index_from_particle: Function mapping particles to patch indices
            phantom_counts_frac: Fraction of phantom counts to add (for smoothing)
        """
        self.satellite_patch_locations = satellite_patch_locations
        self.patch_index_from_particle = patch_index_from_particle
        self.phantom_counts_frac = phantom_counts_frac

    def compute_weights(self,
                       belief_particles: torch.Tensor,
                       particles_to_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute log weights for particles based on prior belief distribution.

        This computes how likely each particle in particles_to_weight is according
        to where the belief_particles are located. Particles in regions with many
        belief particles get higher weights.

        Args:
            belief_particles: (num_belief_particles, state_dim) representing prior belief
            particles_to_weight: (num_particles, state_dim) particles to compute weights for

        Returns:
            log_weights: (num_particles,) unnormalized log weights
        """
        # Compute belief log likelihood from particle distribution
        belief_log_likelihoods = pf.wag_belief_log_likelihood_from_particles(
            belief_particles,
            self.satellite_patch_locations,
            self.phantom_counts_frac,
            self.patch_index_from_particle
        )

        # Map particles to patches and get their belief likelihoods
        particle_log_weights = pf.wag_calculate_log_particle_weights(
            belief_log_likelihoods,
            particles_to_weight,
            self.patch_index_from_particle
        )

        return particle_log_weights


class ObservationLikelihoodCalculator(Protocol):
    """Protocol for computing observation likelihoods in WAG particle filter.

    Implementations should be initialized with a list of panorama IDs and any
    necessary context (e.g., satellite patch locations, observation data).
    """

    def compute_log_likelihoods(self, particles: torch.Tensor, panorama_id: str) -> torch.Tensor:
        """
        Compute unnormalized log likelihoods for particles given an observation.

        This computes p(z|x) for each particle x, where z is the observation
        identified by panorama_id.

        Args:
            particles: (num_particles, state_dim) tensor of particle states
            panorama_id: Identifier for the observation/panorama

        Returns:
            log_likelihoods: (num_particles,) tensor of unnormalized log likelihoods
        """
        ...

    def sample_from_observation(self, num_particles: int, panorama_id: str,
                                generator: torch.Generator) -> torch.Tensor:
        """
        Sample particles from the observation likelihood distribution.

        This samples from p(x|z), used for dual MCL initial sampling.
        The caller (measurement_wag) will weight these samples by the prior belief.

        Args:
            num_particles: Number of particles to sample
            panorama_id: Identifier for the observation/panorama
            generator: Random generator for sampling

        Returns:
            particles: (num_particles, state_dim) sampled particles
        """
        ...


class WagObservationLikelihoodCalculator:
    """WAG observation likelihood calculator using similarity matrices.

    This implements the observation likelihood model from the WAG paper,
    using precomputed similarity matrices between panoramas and satellite patches.
    """

    def __init__(self,
                 similarity_matrix: torch.Tensor,  # path_length x num_patches
                 panorama_ids: list[str],
                 satellite_patch_locations: torch.Tensor,  # num_patches x 2 (lat, lon)
                 patch_index_from_particle: Callable[[torch.Tensor], torch.Tensor],
                 wag_config: WagConfig,
                 device: torch.device):
        """
        Initialize WAG observation likelihood calculator.

        Args:
            similarity_matrix: (path_length, num_patches) precomputed similarities
            panorama_ids: List of panorama IDs corresponding to rows of similarity_matrix
            satellite_patch_locations: (num_patches, 2) lat/lon coordinates of patch centers
            patch_index_from_particle: Function mapping particle positions to patch indices
            wag_config: WAG configuration (sigma, etc.)
            device: Device for torch operations
        """
        self.similarity_matrix = similarity_matrix.to(device)
        self.pano_id_to_idx = {pano_id: i for i, pano_id in enumerate(panorama_ids)}
        self.satellite_patch_locations = satellite_patch_locations.to(device)
        self.patch_index_from_particle = patch_index_from_particle
        self.sigma = wag_config.sigma_obs_prob_from_sim
        self.device = device

    def compute_log_likelihoods(self, particles: torch.Tensor, panorama_id: str) -> torch.Tensor:
        """Compute unnormalized log likelihoods for particles."""
        # Get similarity values for this panorama
        pano_idx = self.pano_id_to_idx[panorama_id]
        similarity_vector = self.similarity_matrix[pano_idx]  # (num_patches,)

        # Compute observation log likelihood from similarity
        observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
            similarity_vector, self.sigma)

        # Map particles to patches and get their likelihoods
        particle_log_likelihoods = pf.wag_calculate_log_particle_weights(
            observation_log_likelihoods,
            particles,
            self.patch_index_from_particle)

        return particle_log_likelihoods

    def sample_from_observation(self, num_particles: int, panorama_id: str,
                                generator: torch.Generator) -> torch.Tensor:
        """Sample particles from observation likelihood distribution."""
        # Get similarity values for this panorama
        pano_idx = self.pano_id_to_idx[panorama_id]
        similarity_vector = self.similarity_matrix[pano_idx]

        # Compute observation log likelihood from similarity
        observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
            similarity_vector, self.sigma)

        # Normalize to get state likelihood
        normalizer = torch.logsumexp(observation_log_likelihoods, 0)
        state_log_likelihood = observation_log_likelihoods - normalizer

        # Sample from satellite patch locations weighted by state likelihood
        sampled_particles = pf.wag_multinomial_resampling(
            self.satellite_patch_locations,
            state_log_likelihood,
            generator,
            num_samples=num_particles)

        return sampled_particles


def measurement_wag(
        particles: torch.Tensor,  # N x state dimension
        obs_likelihood_calculator: ObservationLikelihoodCalculator,
        belief_weighting: BeliefWeighting,
        panorama_id: str,
        wag_config: WagConfig,
        generator: torch.Generator,
        return_past_particle_weights: bool = False
) -> WagMeasurementResult:
    """WAG measurement update step.

    Args:
        particles: Current particle states (num_particles, state_dim)
        obs_likelihood_calculator: Calculator for observation likelihoods
        belief_weighting: Calculator for belief-based weighting (dual MCL)
        panorama_id: Identifier for the current observation
        wag_config: WAG configuration parameters
        generator: Random generator for sampling
        return_past_particle_weights: If True, return log weights in result

    Returns:
        WagMeasurementResult with resampled particles and optional weights
    """

    num_dual_mcl_samples = int(wag_config.dual_mcl_frac * particles.shape[0])
    num_mcl_samples = particles.shape[0] - num_dual_mcl_samples

    # Create new MCL samples (primal particles)
    # Resample from prior particles weighted by observation likelihood
    if num_mcl_samples:
        # Compute observation log likelihoods for current particles
        observation_log_likelihoods = obs_likelihood_calculator.compute_log_likelihoods(
            particles, panorama_id)

        # Normalize to get particle weights
        primal_log_particle_weights = observation_log_likelihoods - observation_log_likelihoods.logsumexp(dim=0)

        # Resample particles according to observation likelihood
        resampled_primal_particles = pf.wag_multinomial_resampling(
                particles, primal_log_particle_weights, generator, num_samples=num_mcl_samples)
    else:
        resampled_primal_particles = torch.zeros((0, particles.shape[1]), device=particles.device)
        primal_log_particle_weights = torch.zeros((0), device=particles.device)

    # Create new Dual MCL samples
    # Sample from observation likelihood, then weight by prior belief
    if num_dual_mcl_samples:
        # Sample particles from observation likelihood distribution
        dual_mcl_particles = obs_likelihood_calculator.sample_from_observation(
            num_dual_mcl_samples, panorama_id, generator)

        # Weight dual particles by belief (where do prior particles think we are?)
        # This is independent of the observation model
        dual_log_particle_weights = belief_weighting.compute_weights(
            particles, dual_mcl_particles)

        # Normalize to get particle weights
        dual_log_particle_weights = dual_log_particle_weights - dual_log_particle_weights.logsumexp(dim=0)

        # Resample dual particles according to belief weights
        resampled_dual_particles = pf.wag_multinomial_resampling(
                dual_mcl_particles,
                dual_log_particle_weights,
                generator,
                num_samples=num_dual_mcl_samples)
    else:
        resampled_dual_particles = torch.zeros((0, particles.shape[1]), device=particles.device)
        dual_log_particle_weights = torch.zeros((0), device=particles.device)

    resampled_particles = torch.cat([resampled_primal_particles, resampled_dual_particles])
    log_particle_weights = torch.cat([primal_log_particle_weights, dual_log_particle_weights])

    if return_past_particle_weights:
        return WagMeasurementResult(
                resampled_particles=resampled_particles,
                num_dual_particles=num_dual_mcl_samples,
                log_particle_weights=log_particle_weights
        )
    return WagMeasurementResult(
            resampled_particles=resampled_particles,
            num_dual_particles=num_dual_mcl_samples,
            log_particle_weights=None
    )


def move_wag(
        particles: torch.Tensor,  # N x state dimension
        motion_delta: torch.Tensor,  # state dimension
        wag_config: WagConfig,
        generator: torch.Generator,
) -> torch.Tensor:
    moved_particles = pf.wag_motion_model(particles,
                                          motion_delta,
                                          wag_config.noise_percent_motion_model,
                                          generator)
    return moved_particles
