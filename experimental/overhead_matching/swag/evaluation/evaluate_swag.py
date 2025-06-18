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
import hashlib


def hash_model(model: torch.nn.Module):
    m = hashlib.sha256()
    state_dict = model.state_dict()
    for k in sorted(model.state_dict()):
        m.update(k.encode())
        m.update(state_dict[k].cpu().numpy().tobytes())
    return m.digest()


def hash_dataset(dataset: vd.VigorDataset):
    import struct
    m = hashlib.sha256()

    for i, row in dataset._panorama_metadata.iterrows():
        m.update(i.to_bytes(length=4, byteorder='little', signed=True))
        m.update(row.pano_id.encode())

    for i, row in dataset._satellite_metadata.iterrows():
        m.update(i.to_bytes(length=4, byteorder='little', signed=True))
        m.update(struct.pack('<dd', row.lat, row.lon))

    return m.digest()


def compute_combined_hash(sat_model, pano_model, dataset):
    m = hashlib.sha256()
    m.update(hash_model(sat_model))
    m.update(hash_model(pano_model))
    m.update(hash_dataset(dataset))
    return m.hexdigest()


def compute_similarity_matrix(
        sat_model: torch.nn.Module,
        pano_model: torch.nn.Module,
        dataset: vd.VigorDataset,
        device: torch.device):
    sat_data_view = dataset.get_sat_patch_view()
    sat_data_view_loader = vd.get_dataloader(sat_data_view, batch_size=64, num_workers=16)
    pano_data_view = dataset.get_pano_view()
    pano_data_view_loader = vd.get_dataloader(pano_data_view, batch_size=64, num_workers=16)

    with torch.no_grad():
        print("building satellite embedding database")
        sat_embeddings = sed.build_satellite_db(
            sat_model, sat_data_view_loader, device=device)
        print("building panorama embedding database")
        pano_embeddings = sed.build_panorama_db(
            pano_model, pano_data_view_loader, device=device)
        print("building all similarity")
        out = sed.calculate_cos_similarity_against_database(
            pano_embeddings, sat_embeddings)  # pano_embeddings x sat_patches
    return out


def compute_cached_similarity_matrix(
        sat_model: torch.nn.Module,
        pano_model: torch.nn.Module,
        dataset: vd.VigorDataset,
        device: torch.device):
    combined_hash = compute_combined_hash(sat_model, pano_model, dataset)
    file_path = Path(f"~/.cache/robot/overhead_matching/similarity_matrix/{combined_hash}.pt").expanduser()
    if file_path.exists():
        all_similarity = torch.load(file_path).to(device)
        print(f"USING CACHED similarity db found at {file_path}")
    else:
        all_similarity = compute_similarity_matrix(
                sat_model=sat_model,
                pano_model=pano_model,
                dataset=dataset,
                device=device)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_similarity.cpu(), file_path)
    return all_similarity


def evaluate_prediction_top_k(
        sat_model: torch.nn.Module,
        pano_model: torch.nn.Module,
        dataset: vd.VigorDataset,
        device: torch.device = "cuda"):

    all_similarity = compute_cached_similarity_matrix(
        sat_model=sat_model,
        pano_model=pano_model,
        dataset=dataset,
        device=device)

    rankings = torch.argsort(all_similarity, dim=1, descending=True)

    out = []
    for i, row in dataset._panorama_metadata.iterrows():
        for sat_idx in row.positive_satellite_idxs:
            k_value = torch.argwhere(rankings[i] == sat_idx)
            out.append({
                "pano_idx": i,
                "sat_idx": sat_idx,
                "match_type": 'positive',
                "k_value": k_value.item()})

        for sat_idx in row.semipositive_satellite_idxs:
            k_value = torch.argwhere(rankings[i] == sat_idx)
            out.append({
                "pano_idx": i,
                "sat_idx": sat_idx,
                "match_type": 'semipositive',
                "k_value": k_value.item()})
    out = pd.DataFrame.from_records(out)

    return out, all_similarity


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
    with torch.no_grad():
        sat_patch_positions = vigor_dataset.get_patch_positions().to(device)
        sat_patch_kdtree = build_kd_tree(sat_patch_positions)

        all_similarity = compute_cached_similarity_matrix(
                sat_model=sat_model, pano_model=pano_model, dataset=vigor_dataset, device=device)

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
            error_meters_at_each_step = get_distance_error_between_pano_and_particles_meters(vigor_dataset, path, particle_history)
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
