import argparse
import matplotlib.pyplot as plt
import json
import tqdm
import common.torch.load_torch_deps
import torch
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to Vigor dataset")
    parser.add_argument("--out", type=Path, required=True, help="Save json file path")
    parser.add_argument("--path-length-m", type=float, required=True, help="Path length meters")
    parser.add_argument("--num-paths", type=int, required=True, help="Number of paths to generate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--skip-plots", action="store_true", help="skip path visualization")
    parser.add_argument("--turn-temperature", type=float, default=0.1, help="Turn temperature. Higher value more turns")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005, help="Panorama neighbor radius deg")

    args = parser.parse_args()
    args.out = Path(args.out)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg)
    dataset = vd.VigorDataset(Path(args.dataset_path), dataset_config)
    paths = []

    for i in tqdm.tqdm(range(args.num_paths)):
        generator = torch.manual_seed(args.seed + i)
        paths.append(dataset.generate_random_path(
            generator, args.path_length_m, args.turn_temperature))
        if not args.skip_plots:
            fig, ax = dataset.visualize(path=paths[-1])
            plt.savefig(args.out.parent / f"{i:06d}.png")
            plt.close(fig)

    out = {
        "paths": paths,
        "dataset_hash": hash(dataset),
        "dataset_path": str(args.dataset_path),
        "args": {
            "path_length_m": args.path_length_m,
            "seed": args.seed,
            "turn_temperature": args.turn_temperature,
            "panorama_neighbor_radius_deg": args.panorama_neighbor_radius_deg
        },
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
