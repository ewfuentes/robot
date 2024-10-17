

import matplotlib.pyplot as plt
import math
from common.torch import torch
import tqdm

from typing import NamedTuple


class RadialExp(NamedTuple):
    center: torch.Tensor
    scale: float

# The observation model returns the log probabilility of a particle
class ObservationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._components = [
            (1.0, RadialExp(center=torch.tensor([3, 4]), scale=1)),
            (0.5, RadialExp(center=torch.tensor([6, 7]), scale=1)),
            (1.05, RadialExp(center=torch.tensor([8, 4]), scale=1))
        ]

    def forward(self, x):
        # x is a b x d tensor
        b, d = x.shape
        assert d == 2
        log_prob_terms = torch.zeros(b, len(self._components))
        for i, (weight, comp) in enumerate(self._components):

            log_prob_terms[:, i] = (
                math.log(weight) -
                torch.sum(torch.abs(x - comp.center), dim=1) / comp.scale)
        return torch.logsumexp(log_prob_terms, 1)


def motion_model(particles, gen):
    deltas = 0.25 * torch.randn(particles.shape, generator=gen)
    return particles + deltas


def sample_with_replacement(log_probs, num_samples, gen):
    steps = torch.arange(0, 1, 1 / num_samples) 
    cum_sum = torch.cumsum(torch.exp(log_probs), dim=0)
    cum_sum = cum_sum / cum_sum[-1]
    sample_vals = torch.rand(1, generator=gen) / num_samples + steps
    out = torch.searchsorted(cum_sum, sample_vals)
    return out


def particle_filter_step(state, motion_model, observation_model, gen):
    # Apply the motion model
    new_particles = motion_model(state, gen)

    # Compute the weights
    unnorm_log_prob = observation_model(new_particles)

    # Resample
    sample_idxs = sample_with_replacement(unnorm_log_prob, unnorm_log_prob.shape[0], gen)

    return new_particles[sample_idxs]


def run_particle_filter(initial_particles, motion_model, observation_model, num_steps, gen):
    particles = [initial_particles]
    for i in tqdm.tqdm(range(num_steps)):
        particles.append(particle_filter_step(particles[-1], motion_model, observation_model, gen))
    return torch.stack(particles, axis=0)


def reverse_kl(samples, obs_model):
    unnorm_log_probs = obs_model(samples)
    return -math.log(samples.shape[0]) - torch.mean(unnorm_log_probs)


def svgd_step(state, obs_model, sigma, step_size):
    n, d = state.shape
    # The observation jacobian is an n x n x d tensor.
    # since the log probabilities are independent only the diagonal elements
    # in the first two dimensions will be non zero
    obs_jac = torch.diagonal(torch.autograd.functional.jacobian(obs_model, state))
    obs_jac = torch.transpose(obs_jac, 0, 1)

    def compute_kernel_dist(state):
        mahal_dist = torch.nn.functional.pdist(state) / (2 * sigma ** 2)
        dist = torch.exp(-mahal_dist)
        return dist

    kernel_dist = compute_kernel_dist(state)
    kernel_dist_jac = torch.autograd.functional.jacobian(compute_kernel_dist, state.cuda()).cpu()

    kernel_mat = torch.eye(n, n)
    kernel_jac = torch.zeros(n, n, n, 2)

    triu_idxs = torch.triu_indices(n, n, offset=1)

    diag_idxs = torch.arange(n)

    kernel_mat[triu_idxs[0], triu_idxs[1]] = kernel_dist
    kernel_mat[triu_idxs[1], triu_idxs[0]] = kernel_dist

    kernel_jac[triu_idxs[0], triu_idxs[1]] = kernel_dist_jac
    kernel_jac[triu_idxs[1], triu_idxs[0]] = kernel_dist_jac

    kernel_jac = torch.diagonal(kernel_jac, dim1=1, dim2=2)
    kernel_jac = torch.transpose(kernel_jac, 1, 2)

    direction = (kernel_mat @ obs_jac + torch.sum(kernel_jac, axis=1)) / n

    delta = direction * step_size
    
    return state + delta


def run_svgd(state, obs_model, num_steps, sigma=1.0, step_size=1.0):
    particles = [state]
    for i in tqdm.tqdm(range(num_steps)):
        particles.append(svgd_step(particles[-1], obs_model, sigma, step_size))
    return torch.stack(particles, axis=0)

def main():
    obs_model = ObservationModel()
    xs, ys = torch.meshgrid([torch.linspace(0, 10, 300)]*2, indexing='ij')
    eval_pts = torch.stack([xs, ys], dim=-1)
    eval_pts = eval_pts.reshape(-1, 2)
    log_probs = obs_model(eval_pts)
    log_probs = log_probs.reshape(*xs.shape)
    probs = torch.exp(log_probs)

    gen = torch.random.manual_seed(0)

    NUM_PARTICLES = 1000
    NUM_STEPS = 400
    particles = 10.0 * torch.rand(NUM_PARTICLES, 2, generator=gen)
    particle_steps = run_particle_filter(particles, motion_model, obs_model, NUM_STEPS, gen)

    NUM_SVGD_PARTICLES = 50
    sigma = 0.5
    step_size = 1.0
    # svgd_particles = 10.0 * torch.rand(NUM_SVGD_PARTICLES, 2, generator=gen)
    svgd_particles = 0.1 * torch.randn(NUM_SVGD_PARTICLES, 2)
    svgd_steps = run_svgd(svgd_particles, obs_model, NUM_STEPS, sigma, step_size)

    # for i in tqdm.tqdm(range(particle_steps.shape[0])):
    #     fig = plt.figure()
    #     plt.contour(xs, ys, torch.exp(log_probs))
    #     plt.colorbar()
    #     plt.title(f'Step {i} Rev KL: {reverse_kl(particle_steps[i], obs_model)}')
    #     plt.scatter(particle_steps[i, :, 0], particle_steps[i, :, 1])
    #     plt.xlim(0, 10)
    #     plt.ylim(0, 10)
    #     plt.savefig(f'/tmp/particle_filter_{i:05d}.png')

    #     plt.close()

    for i in tqdm.tqdm(range(svgd_steps.shape[0])):
        fig = plt.figure()
        plt.contour(xs, ys, torch.exp(log_probs))
        plt.colorbar()
        plt.title(f'Step {i} Rev KL: {reverse_kl(svgd_steps[i], obs_model)}')
        plt.scatter(svgd_steps[i, :, 0], svgd_steps[i, :, 1])
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(f'/tmp/svgd_{i:05d}.png')

        plt.close()
        


if __name__ == "__main__":
    main()
