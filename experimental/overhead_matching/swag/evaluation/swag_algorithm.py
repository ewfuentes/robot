import common.torch.load_torch_deps
import torch
from typing import NamedTuple
import experimental.overhead_matching.swag.filter.particle_filter as pf
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
from torch_kdtree.nn_distance import TorchKDTree
import dataclasses


@dataclasses.dataclass
class WagObservationResult:
    # In a WAG observation step, the particles are weighted according to the observation likelihood
    # and then resampled according to this likelihood.
    log_particle_weights: torch.Tensor | None
    resampled_particles: torch.Tensor


def initialize_wag_particles(gt_start_position_lat_lon: torch.Tensor,
                            wag_config: WagConfig,
                            generator: torch.Generator):
    # TODO: compensate for non-uniformity of latitude radius (who defined these reference frames)
    sampled_mean = torch.normal(mean=gt_start_position_lat_lon,
                                std=wag_config.initial_particle_distribution_offset_std_deg,
                                generator=generator).to(gt_start_position_lat_lon.device)
    particles = torch.normal(mean=0, std=wag_config.initial_particle_distribution_std_deg,
                             size=(wag_config.num_particles, 2)).to(gt_start_position_lat_lon.device)
    particles += sampled_mean
    return particles


def observe_wag(
        particles: torch.Tensor,  # N x state dimension
        similarity_matrix: torch.Tensor,  # W
        satellite_patch_kdtree: TorchKDTree,
        wag_config: WagConfig,
        generator: torch.Generator,
        return_past_particle_weights: bool = False
) -> torch.Tensor:  # particles

    # calculate observation likelihoods
    observation_log_likelihoods = pf.wag_observation_log_likelihood_from_similarity_matrix(
        similarity_matrix, wag_config.sigma_obs_prob_from_sim)
    log_particle_weights = pf.wag_calculate_log_particle_weights(
        observation_log_likelihoods,
        satellite_patch_kdtree,
        particles,
        wag_config.max_distance_to_patch_deg)

    # resample particles
    resampled_particles = pf.wag_multinomial_resampling(particles, log_particle_weights, generator)
    if return_past_particle_weights:
        return WagObservationResult(
                resampled_particles=resampled_particles, log_particle_weights=log_particle_weights)
    return WagObservationResult(resampled_particles=resampled_particles, log_particle_weights=None)


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
