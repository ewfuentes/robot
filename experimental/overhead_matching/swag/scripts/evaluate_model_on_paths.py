import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
import common.torch.load_and_save_models as lsm
from pathlib import Path
import json
import math
import common.torch.load_torch_deps
from google.protobuf import text_format
import torch

def construct_path_eval_inputs_from_args(
        sat_model_path: str,
        pano_model_path: str,
        dataset_path: str,
        paths_path: str,
        panorama_neighbor_radius_deg: float,
):
    with open(paths_path, 'r') as f:
        paths_data = json.load(f)
    pano_model = lsm.load_model(pano_model_path)
    sat_model = lsm.load_model(sat_model_path)

    dataset_path = Path(dataset_path).expanduser()
    dataset_config = vd.VigorDatasetConfig(
        panorama_neighbor_radius=panorama_neighbor_radius_deg,
        satellite_patch_size=(320, 320),
        panorama_size=(320, 640),
        factor=1,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    return vigor_dataset, sat_model, pano_model, paths_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sat-path", type=str, required=True,
                        help="Model folder path for sat model")
    parser.add_argument("--pano-path", type=str, required=True,
                        help="Model folder path for pano model")
    parser.add_argument("--paths-path", type=str, required=True,
                        help="Path to json file full of evaluation paths")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the evaluation results")
    # parser.add_argument("--wag-config-path", type=str, required=True, help="Path to WAG config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float,
                        default=0.0005, help="Panorama neighbor radius deg")

    args = parser.parse_args()

    # torch.cuda.memory._record_memory_history(max_entries=100_000)

    DEVICE = "cuda:0"

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)


    args.sat_model_path = Path(args.sat_path).expanduser()
    args.pano_model_path = Path(args.pano_path).expanduser()
    args.output_path = Path(args.output_path).expanduser()

    vigor_dataset, sat_model, pano_model, paths_data = construct_path_eval_inputs_from_args(
        sat_model_path=args.sat_model_path,
        pano_model_path=args.pano_model_path,
        dataset_path=args.dataset_path,
        paths_path=args.paths_path,
        panorama_neighbor_radius_deg=args.panorama_neighbor_radius_deg,
    )

    EARTH_RADIUS_M = 6_371_000.0
    wag_config = WagConfig(noise_percent_motion_model=0.02,  # page 71 thesis
                           # offset was fixed at 1.3km in thesis (page 71)
                           initial_particle_distribution_offset_std_deg=1300.0 / EARTH_RADIUS_M * 180.0 / math.pi,  # 1300m to deg
                           initial_particle_distribution_std_deg=2970.0 / EARTH_RADIUS_M * 180.0 / math.pi,  # page 73 of thesis, 2970m to deg
                           num_particles=100_000,
                           sigma_obs_prob_from_sim=0.1,
                           max_distance_to_patch_deg=200.0 / EARTH_RADIUS_M * 180.0 / math.pi)

    with open(Path(args.output_path) / "wag_config.pb", "w") as f:
        f.write(text_format.MessageToString(wag_config))

    es.evaluate_model_on_paths(
        vigor_dataset=vigor_dataset,
        sat_model=sat_model,
        pano_model=pano_model,
        paths=paths_data['paths'],
        wag_config=wag_config,
        seed=args.seed,
        output_path=args.output_path,
        device=DEVICE,
    )
