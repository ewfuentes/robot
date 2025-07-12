import common.torch.load_torch_deps
import torch
from typing import NamedTuple
import experimental.overhead_matching.swag.filter.particle_filter as pf
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
import dataclasses
from typing import Callable


@dataclasses.dataclass
class WagMeasurementResult:
    # In a WAG observation step, the particles are weighted according to the observation likelihood
    # and then resampled according to this likelihood.
    log_particle_weights: torch.Tensor | None

    dual_mcl_particles: torch.Tensor | None
    dual_log_particle_weights: torch.Tensor | None

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


def measurement_wag(
        particles: torch.Tensor,  # N x state dimension
        similarity_matrix: torch.Tensor,  # W
        patch_index_from_particle: Callable[[torch.Tensor], torch.Tensor],
        satellite_patch_locations: torch.Tensor,
        wag_config: WagConfig,
        generator: torch.Generator,
        return_past_particle_weights: bool = False
) -> torch.Tensor:  # particles

    num_dual_mcl_samples = int(wag_config.dual_mcl_frac * particles.shape[0])
    num_mcl_samples = particles.shape[0] - num_dual_mcl_samples
    # Create new MCL samples
    # calculate observation likelihoods
    if num_mcl_samples:
        observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
            similarity_matrix, wag_config.sigma_obs_prob_from_sim)
        log_particle_weights = pf.wag_calculate_log_particle_weights(
            observation_log_likelihoods,
            particles,
            patch_index_from_particle)

        # resample particles
        resampled_primal_particles = pf.wag_multinomial_resampling(
                particles, log_particle_weights, generator, num_samples=num_mcl_samples)
    else:
        resampled_primal_particles = torch.zeros((0, particles.shape[1]), device=particles.device)

    # Create new Dual MCL samples
    # compute the state likelihood by normalizing the observation likelihoods
    if num_dual_mcl_samples:
        observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
                similarity_matrix, wag_config.sigma_obs_prob_from_sim)
        normalizer = torch.logsumexp(observation_log_likelihoods, 0)
        state_log_likelihood = observation_log_likelihoods - normalizer

        # Sample particles based on the likelihood of being in a particular satellite patch
        dev_satellite_patch_locations = satellite_patch_locations.to(state_log_likelihood.device)
        dual_mcl_particles = pf.wag_multinomial_resampling(
                dev_satellite_patch_locations,
                state_log_likelihood,
                generator,
                num_samples=num_dual_mcl_samples)

        # Compute the log likelihood of a patch based on the prior belief
        belief_log_likelihoods = pf.wag_belief_log_likelihood_from_particles(
            particles,
            dev_satellite_patch_locations,
            wag_config.dual_mcl_belief_phantom_counts_frac,
            patch_index_from_particle)

        # Compute particle weights based on the computed belief log likelihoods
        dual_log_particle_weights = pf.wag_calculate_log_particle_weights(
                belief_log_likelihoods, dual_mcl_particles, patch_index_from_particle)

        # Sample new dual particles based on the likelihood coming from the belief
        resampled_dual_particles = pf.wag_multinomial_resampling(
                dual_mcl_particles,
                dual_log_particle_weights,
                generator,
                num_samples=num_dual_mcl_samples)
    else:
        resampled_dual_particles = torch.zeros((0, particles.shape[1]), device=particles.device)

    resampled_particles = torch.cat([resampled_primal_particles, resampled_dual_particles])
    if return_past_particle_weights:
        return WagMeasurementResult(
                resampled_particles=resampled_particles,
                num_dual_particles=num_dual_mcl_samples,
                log_particle_weights=log_particle_weights,
                dual_mcl_particles=dual_mcl_particles if num_dual_mcl_samples else None,
                dual_log_particle_weights=dual_log_particle_weights if num_dual_mcl_samples else None)
    return WagMeasurementResult(
            resampled_particles=resampled_particles,
            num_dual_particles=num_dual_mcl_samples,
            log_particle_weights=None,
            dual_mcl_particles=None,
            dual_log_particle_weights=None)


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
