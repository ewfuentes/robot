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
    n_particles, n_dim = particles.shape
    assert n_dim == motion_delta.shape[0] and motion_delta.ndim == 1
    delta_t = torch.norm(motion_delta)  # length of motion
    n_t = delta_t * noise_percent * torch.normal(0, 1, size=(n_particles,), generator=generator)
    return particles + delta_t.unsqueeze(0) * n_t


def wag_observation_liklihood_from_similarity_matrix(
        similarity_matrix: torch.Tensor,
        sigma: float  # 0.1 in the paper
) -> torch.Tensor:
    return (
        1 / (torch.sqrt(2 * torch.pi) * sigma) *
        torch.exp(
            -0.5 * torch.square(
                (similarity_matrix.max() - similarity_matrix) / sigma
            )
        )
    )


def wag_update_particle_weights(observation_liklihood_matrix: torch.Tensor,
                                patch_positions: torch.Tensor,
                                particles: torch.Tensor) -> torch.Tensor:
    """
    observation_liklihood_matrix: Matrix of M x N, each cell contains p(z_t | x_t^j)
    patch_positions: Matrix of M x N x state dimension, where each cell in M x N matrix is position of that cell in the same frame as the particles
    particles: N_particles x state dimension: the particles
    """
    assert observation_liklihood_matrix.ndim == 2 and patch_positions.ndim == 3 and particles.ndim == 2
    assert observation_liklihood_matrix.shape == patch_positions.shape[:2]
    assert particles.shape[1] == patch_positions[2]

    #  find closest patch to each particle and return liklihood of that patch
    diff = patch_positions.unsqueeze(0) - particles.unsqueeze(1).unsqueeze(1)
    dist = torch.norm(diff.to(torch.float32), dim=-1).reshape(particles.shape[0], -1)
    particle_liklihood = observation_liklihood_matrix.flatten()[torch.argmin(dist)]
    # normalize
    particle_liklihood = particle_liklihood / particle_liklihood.sum()
    return particle_liklihood


def wag_multinomial_resampling(particles: torch.Tensor,
                               weights: torch.Tensor,
                               generator: torch.Generator):
    """
    particles: N_particles x state dimension: the particles
    weights: N_particles: the weight of each particle
    generator: random generator
    """
    norm_cumsum = torch.cumsum(weights) / weights.sum()
    norm_cumsum = norm_cumsum - norm_cumsum[0]  # cumsum now spans 0 -> 1 - norm_cumsum[0]
    random_sample = torch.rand(particles.shape[0], generator=generator)
    comparison = random_sample.unsqueeze(1) > norm_cumsum.unsqueeze(0)
    indices = torch.argmax(comparison.to(torch.int), dim=1)

    resampled_particles = particles[indices]
    return resampled_particles
