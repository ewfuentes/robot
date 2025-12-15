import common.torch.load_torch_deps
import torch
import numpy as np
from typing import Callable


def wag_motion_model(particles: torch.Tensor,
                     motion_delta: torch.Tensor,
                     noise_percent: float,
                     generator: torch.Generator) -> torch.Tensor:
    """
    see section 3.2.5 in "City-scale Cross-view Geolocalization with Generalization to Unseen Environments" 

    particles: N x state dimension
    motion_delta: state dimenson
    noise_percentage: float
    generator: random generator

    """
    n_particles, state_dim = particles.shape
    assert state_dim == motion_delta.shape[0] and motion_delta.ndim == 1
    delta_t = torch.norm(motion_delta, dim=0)
    n_t = delta_t * noise_percent * torch.randn(size=(n_particles, state_dim), generator=generator, device=particles.device)
    return particles + motion_delta + n_t  ## NOTE: this differs from the thesis, but I think the thesis Equation 3.5 is wrong


def wag_observation_log_likelihood_from_similarity_matrix(
        similarity_matrix: torch.Tensor,
        sigma: float  # 0.1 in the paper
) -> torch.Tensor:
    return (
        -torch.log(torch.sqrt(torch.tensor(2 * torch.pi))) - torch.log(torch.tensor(sigma)) +
            -0.5 * torch.square(
                (similarity_matrix.max() - similarity_matrix) / sigma
            )
    )


def wag_calculate_log_particle_weights(observation_log_likelihood: torch.Tensor,
                                       particles: torch.Tensor,
                                       patch_index_from_pos: Callable[[torch.Tensor], torch.Tensor],
                                       no_patch_log_likelihood: float = np.log(1e-50)) -> torch.Tensor:
    """
    observation_likelihood_matrix: Matrix of length M, each cell contains log(p(z_t | x_t^j))
    particles: N_particles x state dimension: the particles
    patch_index_from_pos: A function that takes in `particles` and returns which index each
        particle corresponds to. If a particle does not correspond to any patch, then it returns
        len(observation_log_likelihood).
    no_patch_log_likelihood: ...this likelihood value

    Returns log weights of each particle
    """
    assert observation_log_likelihood.ndim == 1 and particles.ndim == 2

    # find which patches are closest to each particle
    particle_patch_indices = patch_index_from_pos(particles)


    valid_mask = particle_patch_indices < observation_log_likelihood.shape[0]
    invalid_mask = torch.logical_not(valid_mask)

    particle_log_likelihood = torch.empty(particles.shape[0], dtype=torch.float32, device=particles.device)
    particle_log_likelihood[valid_mask] = observation_log_likelihood[particle_patch_indices[valid_mask]]
    particle_log_likelihood[invalid_mask] = no_patch_log_likelihood
    # normalize
    particle_log_likelihood = particle_log_likelihood - particle_log_likelihood.logsumexp(dim=0)
    return particle_log_likelihood


def wag_multinomial_resampling(particles: torch.Tensor,
                               log_weights: torch.Tensor,
                               generator: torch.Generator,
                               num_samples: int | None = None):
    """
    particles: N_particles x state dimension: the particles
    weights: N_particles: the (natural) log weight of each particle
    generator: random generator
    num_samples: if none, sample the same number of samples as there are particles
    """
    num_particles, state_dim = particles.shape
    if num_samples is None:
        num_samples = num_particles
    log_weights_cpu = log_weights.cpu()
    indices = torch.multinomial(torch.exp(log_weights), num_samples, replacement=True, generator=generator)
    resampled_particles = particles[indices]
    return resampled_particles

def wag_belief_log_likelihood_from_particles(particles: torch.Tensor,
                                             satellite_patch_locations: torch.Tensor,
                                             phantom_counts_frac: float,
                                             patch_index_from_pos):
    particle_patch_indices = patch_index_from_pos(particles)

    num_phantom_counts = int(phantom_counts_frac * particles.shape[0])
    out_counts = torch.full((particles.shape[0],), num_phantom_counts, device=particle_patch_indices.device)

    patch_idxs, counts = torch.unique(particle_patch_indices, sorted=False, return_counts=True)
    out_counts[patch_idxs] += counts

    return torch.log(out_counts)

