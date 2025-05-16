import unittest
import common.torch.load_torch_deps
from torch_kdtree import build_kd_tree
import torch
import numpy as np
from experimental.overhead_matching.swag.filter.particle_filter import (
    wag_motion_model,
    wag_observation_log_likelihood_from_similarity_matrix,
    wag_calculate_log_particle_weights,
    wag_multinomial_resampling
)

import matplotlib.pyplot as plt


class TestParticleFilter(unittest.TestCase):

    def setUp(self):
        # Set random seed for reproducibility
        self.seed = 42
        torch.manual_seed(self.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        # Common test parameters
        self.n_particles = 100
        self.state_dim = 2  # 2D positions (x, y)

    def test_wag_motion_model(self):
        particles = torch.zeros((self.n_particles, self.state_dim))
        motion_delta = torch.tensor([1.0, 0.0])  # Move 1 unit along x-axis
        noise_percent = 0.1

        new_particles = wag_motion_model(
            particles, motion_delta, noise_percent, self.generator
        )

        # Check shape
        self.assertEqual(new_particles.shape, particles.shape)

        # Check that particles have moved in the x direction
        self.assertTrue(torch.all(new_particles[:, 0] != 0))

        # Check that noise is applied
        std_dev = torch.std(new_particles[:, 0])
        self.assertGreater(std_dev, 0)

    def test_wag_observation_likelihood_from_similarity_matrix(self):
        similarity_matrix = torch.tensor([[0.8, 0.3], [0.5, 0.9]])
        sigma = 0.1

        log_likelihood = wag_observation_log_likelihood_from_similarity_matrix(
            similarity_matrix, sigma
        )

        # Check shape
        self.assertEqual(log_likelihood.shape, similarity_matrix.shape)

        # Check that maximum similarity gives max likelihood
        max_sim_idx = torch.argmax(similarity_matrix)
        max_lik_idx = torch.argmax(log_likelihood)
        self.assertEqual(max_sim_idx, max_lik_idx)

    def test_wag_update_particle_weights(self):
        observation_log_likelihood_matrix = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        patch_positions = torch.zeros((4, 2))
        # Set patch positions at (0,0), (1,0), (0,1), (1,1)
        patch_positions[0] = torch.tensor([0.0, 0.0])
        patch_positions[1] = torch.tensor([1.0, 0.0])
        patch_positions[2] = torch.tensor([0.0, 1.0])
        patch_positions[3] = torch.tensor([1.0, 1.0])


        # Particles at various positions
        particles = torch.tensor([
            [0.1, 0.1],  # Close to (0,0)
            [0.9, 0.1],  # Close to (1,0)
            [0.1, 0.9],  # Close to (0,1)
            [0.9, 0.9]   # Close to (1,1)
        ])

        kd_tree = build_kd_tree(patch_positions)

        log_weights = wag_calculate_log_particle_weights(
            observation_log_likelihood_matrix, kd_tree, particles
        )

        # Check shape
        self.assertEqual(log_weights.shape, (particles.shape[0],))

        # Check that weights sum to 1
        self.assertTrue(torch.isclose(torch.sum(torch.exp(log_weights)), torch.tensor(1.0)))

    def test_wag_multinomial_resampling(self):
        particles = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Heavily weight one particle
        NUM_SAMPLES = 1000
        log_weights = torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1]))

        resampled = wag_multinomial_resampling(
            particles, log_weights, self.generator, NUM_SAMPLES
        )

        # Check shape
        self.assertEqual(resampled.shape, (NUM_SAMPLES, 2))

        # With high probability, the first particle should appear multiple times
        # due to its high weight, but we can't test this deterministically
        num_high_likelihood_particle = torch.isclose(resampled, particles[0:1]).prod(dim=1).sum()
        self.assertGreater(num_high_likelihood_particle, NUM_SAMPLES // 2)

    def test_particle_filter_path(self):
        # Initial particles scattered around origin
        n_particles = 500
        particles = torch.normal(
            mean=0.0, std=5.0,
            size=(n_particles, 2),
            generator=self.generator
        )

        # Create a map with ground truth at (100, 0)
        map_size = 400
        grid_size = 200
        x_grid = torch.linspace(0, map_size, grid_size)
        y_grid = torch.linspace(-map_size//2, map_size//2, grid_size)

        # Create patch positions
        patch_positions = torch.zeros((grid_size, grid_size, 2))
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                patch_positions[i, j, 0] = x
                patch_positions[i, j, 1] = y

        # Reshape to (grid_size^2, 2)
        patch_positions = patch_positions.view(-1, 2)
        kd_tree = build_kd_tree(patch_positions)

        # Create a path going straight right
        n_steps = 10
        position = torch.tensor([0.0, 0.0])
        motion_steps = torch.tensor([20.0, 0.0])

        # Parameters
        noise_percent = 0.1

        # Save positions for plotting
        all_particle_positions = [particles.clone()]
        ground_truth_positions = [position.clone()]

        for _ in range(n_steps):
            # Move ground truth
            position = position + motion_steps
            ground_truth_positions.append(position.clone())

            # Move particles according to motion model
            particles = wag_motion_model(
                particles, motion_steps, noise_percent, self.generator
            )

            # Generate similarity matrix (higher value = more similar)
            # Simulating similarity based on distance from ground truth
            obs_log_liklihood = torch.zeros((grid_size * grid_size))
            for i in range(grid_size**2):
                    patch_pos = patch_positions[i]
                    dist = torch.norm(patch_pos - position)
                    # Inverse relationship - closer patches have higher similarity
                    obs_log_liklihood[i] = -dist

            obs_log_liklihood = obs_log_liklihood.view(-1)

            # fig, ax = plt.subplots()
            # ax.imshow(obs_likelihood)
            # plt.savefig(f"/tmp/sim_matrix_{_}.png")

            # Update particle weights
            log_weights = wag_calculate_log_particle_weights(
                obs_log_liklihood, kd_tree, particles
            )
            # Resample particles
            particles = wag_multinomial_resampling(
                particles, log_weights, self.generator
            )

            # Save for plotting
            all_particle_positions.append(particles.clone())

        # Test convergence - check if particles are clustered near ground truth
        final_positions = all_particle_positions[-1]
        mean_position = torch.mean(final_positions, dim=0)
        ground_truth = ground_truth_positions[-1]

        # Verify that the mean of the particles is close to ground truth
        dist_to_truth = torch.norm(mean_position - ground_truth)
        self.assertLess(dist_to_truth, 20.0)

        # Plot if requested
        if hasattr(self, 'plot') and self.plot:
            self._plot_particle_path(all_particle_positions, ground_truth_positions)

    def _plot_particle_path(self, all_particle_positions, ground_truth_positions):
        plt.figure(figsize=(12, 8))

        # Plot particle evolution at different steps
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_particle_positions)))

        for i, (particles, color) in enumerate(zip(all_particle_positions, colors)):
            particles_np = particles.numpy()
            alpha = 0.3 if i < len(all_particle_positions) - 1 else 0.8
            plt.scatter(
                particles_np[:, 0],
                particles_np[:, 1],
                color=color,
                alpha=alpha,
                s=10,
                label=f'Step {i}' if i % 2 == 0 else None
            )

        # Plot ground truth path
        gt_np = torch.stack(ground_truth_positions).numpy()
        plt.plot(
            gt_np[:, 0],
            gt_np[:, 1],
            'k-o',
            linewidth=2,
            markersize=8,
            label='Ground Truth'
        )

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Particle Filter Path Tracking')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.axis('equal')

        plt.savefig('/tmp/particle_filter_path.png')
        plt.close()


if __name__ == '__main__':
    # To enable plotting, add this line before running the tests
    # TestParticleFilter.plot = True
    unittest.main()
