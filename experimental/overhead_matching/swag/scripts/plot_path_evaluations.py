import common.torch.load_torch_deps
import torch
import json 
import numpy as np
import tqdm
from pathlib import Path
import matplotlib.pyplot as plt 
from experimental.overhead_matching.swag.scripts.evaluate_model_on_paths import construct_path_eval_inputs_from_args
from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset


def plot_path_evaluation(
    vigor_dataset,
    path,
    particle_history,
):
    fig, ax = vigor_dataset.visualize(path=path)
    for i, particles in enumerate(particle_history):
        ax.scatter(particles[::100][:, 1], particles[::100][:, 0], color=plt.cm.viridis(i / len(particle_history)), s=1, alpha=0.5, label='particles', zorder=i)
    plt.show()


def plot_similarity(
    similarity: np.ndarray,
    vigor_dataset: VigorDataset,
    highlight_top_n: int = 0,
    ax: plt.Axes = None,
)-> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # plot similarity
    similarity_positions = vigor_dataset.get_patch_positions()
    if highlight_top_n > 0:
        top_n_indices = np.argsort(similarity)[-highlight_top_n:]
        ax.scatter(
            similarity_positions[top_n_indices, 1], similarity_positions[top_n_indices, 0],
            c='purple', s=5, alpha=0.5, label='top n', zorder=1000
        )
    ax.scatter(
        similarity_positions[:, 1], similarity_positions[:, 0],
        c=similarity, cmap='viridis', vmin=-1, vmax=1, s=5, alpha=0.8, label='similarity'
    )
    ax.set_title("Similarity")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax

def save_all_similarity_plots(
    similarity_for_path: torch.Tensor,
    path: list[int],
    vigor_dataset: VigorDataset,
    output_path: Path,
    particle_history: list[torch.Tensor] = None,
):
    output_path.mkdir(exist_ok=True, parents=True)
    path_positions = vigor_dataset.get_panorama_positions(path)
    assert similarity_for_path.shape[0] == len(path), "Similarity shape does not match path length"
    path_colors = [[1, 0, 0]] * len(path_positions)
    path_sizes = [2] * len(path_positions)
    similarity_for_path = similarity_for_path.cpu().numpy()
    for i in tqdm.tqdm(range(len(path))):

        fig, ax = plot_similarity(
            similarity=similarity_for_path[i],
            vigor_dataset=vigor_dataset,
            highlight_top_n=20,
        )
        # highlight current position
        path_colors[i] = [0, 1, 0]
        path_sizes[i] = 10
        ax.scatter(
            path_positions[:, 1], path_positions[:, 0], color=path_colors, s=path_sizes,
            label='path', zorder=10
        )
        # plot particles 
        if particle_history is not None:
            particles = particle_history[i].cpu().numpy()
            ax.scatter(particles[::100][:, 1], particles[::100][:, 0], color='blue', s=1, alpha=0.5, label='particles', zorder=20)

        path_colors[i] = path_colors[-1]
        path_sizes[i] = path_sizes[-1]
        fig.savefig(output_path / f"{i:07d}_similarity.png", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True,
                        help="Path to json file full of evaluation paths")
    
    args = parser.parse_args()

    with open(Path(args.eval_path) / 'args.json', 'r') as f:
        eval_args = json.load(f)

    vigor_dataset, sat_model, pano_model, wag_config, paths_data = construct_path_eval_inputs_from_args(
            sat_model_path=eval_args['sat_path'],
            pano_model_path=eval_args['pano_path'],
            dataset_path=eval_args['dataset_path'],
            paths_path=eval_args['paths_path'],
            panorama_neighbor_radius_deg=eval_args['panorama_neighbor_radius_deg'],
    )

    for path_eval_path in sorted(Path(args.eval_path).glob("*/")):
        print(f"Plotting path evaluation for {path_eval_path}")
        path = torch.load(path_eval_path / 'path.pt')
        particle_history = torch.load(path_eval_path / 'particle_history.pt')
        similarity = torch.load(path_eval_path / 'similarity.pt')
        save_all_similarity_plots(
            similarity_for_path=similarity,
            path=path,
            vigor_dataset=vigor_dataset,
            output_path=path_eval_path / "similarity_plots",
            particle_history=particle_history,
        )
        plot_path_evaluation(
            vigor_dataset=vigor_dataset,
            path=path,
            particle_history=particle_history,
        )
        break


    
