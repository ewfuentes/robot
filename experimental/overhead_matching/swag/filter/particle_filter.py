import common.torch.load_torch_deps
import torch


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
    n_t = delta_t * noise_percent * torch.normal(0, 1, size=(n_particles, state_dim), generator=generator)
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


def wag_calculate_log_particle_weights(observation_log_likelihood_matrix: torch.Tensor,
                                patch_positions: torch.Tensor,
                                particles: torch.Tensor) -> torch.Tensor:
    """
    observation_likelihood_matrix: Matrix of M x N, each cell contains p(z_t | x_t^j)
    patch_positions: Matrix of M x N x state dimension, where each cell in M x N matrix is position of that cell in the same frame as the particles
    particles: N_particles x state dimension: the particles

    Returns log weights of each particle
    """
    assert observation_log_likelihood_matrix.ndim == 2 and patch_positions.ndim == 3 and particles.ndim == 2
    assert observation_log_likelihood_matrix.shape == patch_positions.shape[:2]
    assert particles.shape[1] == patch_positions.shape[2]

    # TODO: this is slow, we can probably do this faster
    #  find closest patch to each particle and return likelihood of that patch
    diff = patch_positions.unsqueeze(0) - particles.unsqueeze(1).unsqueeze(1)
    dist = torch.norm(diff.to(torch.float32), dim=-1).reshape(particles.shape[0], -1)
    particle_log_likelihood = observation_log_likelihood_matrix.flatten()[torch.argmin(dist, dim=1)]
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
