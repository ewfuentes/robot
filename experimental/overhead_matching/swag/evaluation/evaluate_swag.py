import common.torch.load_torch_deps
import torch
import tqdm
from pathlib import Path
import pandas as pd
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa
from torch_kdtree import build_kd_tree
from torch_kdtree.nn_distance import TorchKDTree


def evaluate_prediction_top_k(
    satellite_embedding_database: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device | str = "cuda:0",
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
        for batch in dataloader:
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
    wag_config: sa.WagConfig,
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


def evaluate_model_on_paths(
    vigor_dataset: vd.VigorDataset,
    sat_model: torch.nn.Module,
    pano_model: torch.nn.Module,
    paths: list[list[int]],
    wag_config: sa.WagConfig,
    seed: int,
    output_path: Path,
    device: torch.device = "cuda:0",
) -> None:

    generator = torch.Generator(device=device).manual_seed(seed)
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
            motion_deltas = get_motion_deltas_from_path(vigor_dataset, path).to(device)
            gt_initial_position_lat_lon = vigor_dataset._panorama_metadata.loc[path[0]]
            gt_initial_position_lat_lon = torch.tensor(
                (gt_initial_position_lat_lon['lat'], gt_initial_position_lat_lon['lon']), device=device)
            generator.manual_seed(seed * i)
            initial_particle_state = sa.initialize_wag_particles(
                gt_initial_position_lat_lon, wag_config, generator).to(device)
            particle_history = run_inference_on_path(sat_patch_kdtree,
                                                     initial_particle_state,
                                                     motion_deltas,
                                                     path_similarity_values,
                                                     wag_config,
                                                     generator)
            # save particle history
            save_path = output_path / f"{i:07d}"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(particle_history, save_path / "particle_history.pt")
            torch.save(path, save_path / "path.pt")
            torch.save(path_similarity_values, save_path / "similarity.pt")
