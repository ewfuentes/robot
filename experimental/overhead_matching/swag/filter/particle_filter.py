import common.torch.load_torch_deps
import torch
import numpy as np
import torch_kdtree.nn_distance


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
                                       patch_kdtree: torch_kdtree.nn_distance.TorchKDTree,
                                       particles: torch.Tensor,
                                       max_patch_distance_deg: float,
                                       no_patch_log_likleihood: float = np.log(1e-6)) -> torch.Tensor:
    """
    observation_likelihood_matrix: Matrix of length M, each cell contains log(p(z_t | x_t^j))
    patch_kdtree: KDTree of the patch centers
    particles: N_particles x state dimension: the particles
    max_patch_distance_deg: particles that are not closer to any patch than this distance are assigned..
    no_patch_log_likleihood: ...this likleihood value

    Returns log weights of each particle
    """
    assert observation_log_likelihood.ndim == 1 and particles.ndim == 2

    # find which patches are closest to each particle
    distances_to_patch, particle_patch_indices = patch_kdtree.query(particles, nr_nns_searches=1)
    distances_to_patch = distances_to_patch.squeeze(1)
    particle_patch_indices = particle_patch_indices.squeeze(1)  # N_particles x 1 -> N_particles

    particle_log_likelihood = observation_log_likelihood[particle_patch_indices]
    lost_particles = distances_to_patch > max_patch_distance_deg
    particle_log_likelihood[lost_particles] = no_patch_log_likleihood
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
    indices = torch.multinomial(torch.exp(log_weights), num_samples, replacement=True, generator=generator)
    resampled_particles = particles[indices]
    return resampled_particles
