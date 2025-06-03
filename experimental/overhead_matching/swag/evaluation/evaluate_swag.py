import common.torch.load_torch_deps
import torch
import json
import tqdm
from pathlib import Path
import pandas as pd
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa
from common.math.haversine import find_d_on_unit_circle
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
from torch_kdtree import build_kd_tree
from torch_kdtree.nn_distance import TorchKDTree


def evaluate_prediction_top_k(
    satellite_embedding_database: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device | str = "cuda:0",
    verbose: bool = False
) -> pd.DataFrame:
    model.eval()
    satellite_embedding_database = satellite_embedding_database.to(device)

    out_df = dict(
        panorama_index=[],  # index of the ego image
        # cosine similarity with every overhead patch, where index in the list is the patch index (db row)
        patch_cosine_similarity=[],
        k_value=[],  # the correct patch is the k'th highest similarity
    )

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, disable=not verbose):
            batch_embedding = model(batch.panorama.to(device))
            patch_cosine_distance = sed.calculate_cos_similarity_against_database(
                batch_embedding, satellite_embedding_database)  # B x sat_db_size

            panorama_indices = [x['index'] for x in batch.panorama_metadata]
            correct_overhead_patch_indices = [x['satellite_idx'] for x in batch.panorama_metadata]

            rankings = torch.argsort(patch_cosine_distance, dim=1, descending=True)
            for i in range(len(batch.panorama_metadata)):
                out_df['k_value'].append(torch.argwhere(
                    rankings[i] == correct_overhead_patch_indices[i]).item())
            out_df['panorama_index'].extend(panorama_indices)
            out_df['patch_cosine_similarity'].extend(patch_cosine_distance.tolist())

    df = pd.DataFrame.from_dict(out_df)
    df = df.set_index("panorama_index", drop=True)
    return df

def get_distance_error_between_pano_and_particles_meters(
    vigor_dataset: vd.VigorDataset,
    panorama_index: int | list[int], 
    particles: torch.Tensor,
)->torch.Tensor:
    """
    Calculate the distance in meters between the mean particle position (in lat-lon deg)
    and the panorama at index panorama_index

    If panorama_index is a list of length N, then particles should be of shape (N, num_particles, num_state_dim)

    """
    if isinstance(panorama_index, int):
        panorama_index = [panorama_index]
        assert particles.ndim == 2
        particles = particles.unsqueeze(0)
    
    true_latlong = vigor_dataset.get_panorama_positions(panorama_index)
    particle_latlong_estimate = particles.mean(dim=1)
    out = []
    for i in range(len(panorama_index)):
        distance_error_meters = vd.EARTH_RADIUS_M * find_d_on_unit_circle(true_latlong[i], particle_latlong_estimate[i])
        out.append(distance_error_meters)

    out = torch.tensor(out)
    if len(out) == 1:
        out = out[0]
    return out

def get_motion_deltas_from_path(vigor_dataset: vd.VigorDataset, path: list[int]):
    latlong = vigor_dataset.get_panorama_positions(path)
    motion_deltas = torch.diff(latlong, dim=0)

    return motion_deltas


def get_pano_embeddings_for_indices(vigor_dataset: vd.VigorDataset,
                                    pano_indices: list[int],
                                    model: torch.nn.Module,
                                    device: torch.device | str = "cuda:0",
                                    batch_size: int = 128) -> torch.Tensor:
    """ return_value[i] will be the embedding for vigor_dataset panorama[pano_indices[i]]"""
    model = model.eval().to(device)
    pano_view = vigor_dataset.get_pano_view()
    subset = torch.utils.data.Subset(pano_view, pano_indices)
    dataloader = vd.get_dataloader(subset, batch_size=batch_size, shuffle=False)
    pano_embeddings = sed.build_panorama_db(model, dataloader, device=device)
    return pano_embeddings


def pano_embeddings_and_motion_deltas_from_path(
        vigor_dataset: vd.VigorDataset,
        path: list[int],
        model: torch.nn.Module,
        device: torch.device | str = "cuda:0",
        batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:

    pano_embeddings_for_path = get_pano_embeddings_for_indices(
        vigor_dataset, path, model, device, batch_size)
    motion_deltas = get_motion_deltas_from_path(vigor_dataset, path).to(device)
    return pano_embeddings_for_path, motion_deltas


def run_inference_on_path(
    satellite_patch_kdtree: TorchKDTree,
    initial_particle_state: torch.Tensor,  # N x state dim
    motion_deltas: torch.Tensor,  # path_length - 1 x state dim
    patch_similarity_for_path: torch.Tensor,  # path_length x W
    wag_config: WagConfig,
    generator: torch.Generator,
) -> torch.Tensor:  # path_length x N x state dim

    particle_state = initial_particle_state.clone()
    # TODO: return history of similarities for visualization
    particle_history = []
    for likelihood_value, motion_delta in zip(patch_similarity_for_path[:-1], motion_deltas):
        particle_history.append(particle_state.cpu().clone())
        # observe
        particle_state = sa.observe_wag(particle_state,
                                        likelihood_value,
                                        satellite_patch_kdtree,
                                        wag_config,
                                        generator)
        # move
        particle_state = sa.move_wag(particle_state, motion_delta, wag_config, generator)

    # apply final observation
    particle_state = sa.observe_wag(particle_state,
                                    patch_similarity_for_path[-1],
                                    satellite_patch_kdtree,
                                    wag_config,
                                    generator)
    particle_history.append(particle_state.cpu().clone())
    return particle_history

def construct_inputs_and_evalulate_path(
    sat_patch_kdtree,
    vigor_dataset: vd.VigorDataset,
    path: list[int],
    path_similarity_values: torch.Tensor,
    generator_seed: int,
    device: str,
    wag_config: WagConfig,
):
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    motion_deltas = get_motion_deltas_from_path(vigor_dataset, path).to(device)
    gt_initial_position_lat_lon = vigor_dataset._panorama_metadata.loc[path[0]]
    gt_initial_position_lat_lon = torch.tensor(
        (gt_initial_position_lat_lon['lat'], gt_initial_position_lat_lon['lon']), device=device)
    initial_particle_state = sa.initialize_wag_particles(
        gt_initial_position_lat_lon, wag_config, generator).to(device)

    return run_inference_on_path(sat_patch_kdtree,
                                 initial_particle_state,
                                 motion_deltas,
                                 path_similarity_values,
                                 wag_config,
                                 generator)

def evaluate_model_on_paths(
    vigor_dataset: vd.VigorDataset,
    sat_model: torch.nn.Module,
    pano_model: torch.nn.Module,
    paths: list[list[int]],
    wag_config: WagConfig,
    seed: int,
    output_path: Path,
    device: torch.device = "cuda:0",
) -> None:
    all_final_particle_error_meters = []
    sat_data_view = vigor_dataset.get_sat_patch_view()
    sat_data_view_loader = vd.get_dataloader(sat_data_view, batch_size=64, num_workers=16)
    pano_data_view = vigor_dataset.get_pano_view()
    pano_data_view_loader = vd.get_dataloader(pano_data_view, batch_size=64, num_workers=16)
    print("building satellite embedding database")
    with torch.no_grad():
        sat_patch_positions = vigor_dataset.get_patch_positions().to(device)
        sat_patch_kdtree = build_kd_tree(sat_patch_positions)
        # sat_patch_kdtree = vigor_dataset._satellite_kdtree
        if Path("/tmp/similarity_db.pt").exists():
            all_similarity = torch.load("/tmp/similarity_db.pt").to(device)
            print("USING CACHED similarity db found at /tmp/similarity_db.pt")
        else:
            sat_embeddings = sed.build_satellite_db(
                sat_model, sat_data_view_loader, device=device)
            print("building satellite embedding database done")
            print("building panorama embedding database")
            pano_embeddings = sed.build_panorama_db(
                pano_model, pano_data_view_loader, device=device)
            print("building all similarity")
            all_similarity = sed.calculate_cos_similarity_against_database(
                pano_embeddings, sat_embeddings)  # pano_embeddings x sat_patches

            torch.save(all_similarity.cpu(), "/tmp/similarity_db.pt")

        print("starting iter over paths")
        for i, path in enumerate(tqdm.tqdm(paths)):
            path_similarity_values = all_similarity[path]
            generator_seed = seed * i

            particle_history = construct_inputs_and_evalulate_path(
                sat_patch_kdtree=sat_patch_kdtree,
                vigor_dataset=vigor_dataset,
                path=path,
                path_similarity_values=path_similarity_values,
                generator_seed=generator_seed,
                device=device,
                wag_config=wag_config
            )

            save_path = output_path / f"{i:07d}"
            save_path.mkdir(parents=True, exist_ok=True)
            particle_history = torch.stack(particle_history)
            error_meters_at_each_step = torch.tensor(get_distance_error_meters(vigor_dataset, path, particle_history))
            all_final_particle_error_meters.append(error_meters_at_each_step[-1])
            torch.save(error_meters_at_each_step, save_path / "error.pt")
            torch.save(path, save_path / "path.pt")
            torch.save(path_similarity_values, save_path / "similarity.pt")
            with open(save_path / "other_info.json", "w") as f:
                f.write(json.dumps({
                    "seed": generator_seed,
                }, indent=2))
        average_final_error_meters = torch.tensor(all_final_particle_error_meters).mean().item()
        with open(output_path / "summary_statistics.json", 'w') as f:
            f.write(json.dumps({
                "average_final_error": average_final_error_meters
            }, indent=2))

        print(f"Average final error meters is {average_final_error_meters}")
