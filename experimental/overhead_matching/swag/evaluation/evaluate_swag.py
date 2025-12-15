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
from common.gps import web_mercator
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import (
        WagConfig, SatellitePatchConfig)
from torch_kdtree import build_kd_tree
import hashlib
import dataclasses
from typing import Callable


@dataclasses.dataclass
class PathInferenceResult:
    # sequence is start -> particle_history[0] = log_particle_weights[0] ->
    #  observe_wag -> particle_history_pre_move[0] -> move_wag -> particle_history[1]
    particle_history: torch.Tensor  # path_length x num_particles x state_dim
    log_particle_weights: torch.Tensor | None   # path_length x num_particles
    particle_history_pre_move: torch.Tensor | None # path_length x num_particles x state_dim
    num_dual_particles: int | None # 
    # true when the path inference terminates early due to all particles having a -inf likelihood
    terminated_early: bool

    def get_dual_particle_history(self)->torch.Tensor | None:
        if self.num_dual_particles is None or self.num_dual_particles == 0:
            return None
        assert self.num_dual_particles <= self.particle_history.shape[1]
        return self.particle_history[:, -self.num_dual_particles:, :]
    def get_dual_log_particle_weights(self)->torch.Tensor | None:
        if self.num_dual_particles is None or self.log_particle_weights is None or self.num_dual_particles == 0:
            return None
        assert self.num_dual_particles <= self.log_particle_weights.shape[1]
        return self.log_particle_weights[:, -self.num_dual_particles:]
    def get_dual_particle_history_pre_move(self)->torch.Tensor | None:
        if self.num_dual_particles is None or self.particle_history_pre_move is None or self.num_dual_particles == 0:
            return None
        assert self.num_dual_particles <= self.particle_history_pre_move.shape[1]
        return self.particle_history_pre_move[:, -self.num_dual_particles:]


        



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
    sat_data_view_loader = vd.get_dataloader(sat_data_view, batch_size=96, num_workers=8)
    pano_data_view = dataset.get_pano_view()
    pano_data_view_loader = vd.get_dataloader(pano_data_view, batch_size=96, num_workers=8)

    with torch.no_grad():
        print("building satellite embedding database")
        sat_embeddings = sed.build_satellite_db(
            sat_model, sat_data_view_loader, device=device)
        print("building panorama embedding database")
        pano_embeddings = sed.build_panorama_db(
            pano_model, pano_data_view_loader, device=device)
        print("building all similarity")
        out = sed.calculate_cos_similarity_against_database(
            pano_embeddings.squeeze(), sat_embeddings.squeeze())  # pano_embeddings x sat_patches
    return out


def compute_cached_similarity_matrix(
        sat_model: torch.nn.Module,
        pano_model: torch.nn.Module,
        dataset: vd.VigorDataset,
        device: torch.device,
        use_cached_similarity: bool):

    cache_exists = False
    if use_cached_similarity:
        combined_hash = compute_combined_hash(sat_model, pano_model, dataset)
        file_path = Path(f"~/.cache/robot/overhead_matching/similarity_matrix/{combined_hash}.pt").expanduser()
        cache_exists = file_path.exists()

    if cache_exists:
        all_similarity = torch.load(file_path).to(device)
        print(f"USING CACHED similarity db found at {file_path}")
    else:
        all_similarity = compute_similarity_matrix(
                sat_model=sat_model,
                pano_model=pano_model,
                dataset=dataset,
                device=device)

    if use_cached_similarity:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_similarity.cpu(), file_path)
    return all_similarity


def evaluate_prediction_top_k(
        sat_model: torch.nn.Module,
        pano_model: torch.nn.Module,
        dataset: vd.VigorDataset,
        device: torch.device = "cuda",
        use_cached_similarity: bool = True):

    all_similarity = compute_cached_similarity_matrix(
        sat_model=sat_model,
        pano_model=pano_model,
        dataset=dataset,
        device=device,
        use_cached_similarity=use_cached_similarity)

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
    
    true_latlong = vigor_dataset.get_panorama_positions(panorama_index).to(device=particles.device)
    particle_latlong_estimate = particles.mean(dim=1)

    mean_deviation_m = vd.EARTH_RADIUS_M * find_d_on_unit_circle(
            particles, particle_latlong_estimate[:, None, :])
    var_sq_m = torch.mean(mean_deviation_m ** 2, -1)

    mean_error_m = []
    for i in range(particles.shape[0]):
        distance_error_meters = vd.EARTH_RADIUS_M * find_d_on_unit_circle(
                true_latlong[i], particle_latlong_estimate[i])
        mean_error_m.append(distance_error_meters)

    mean_error_m = torch.tensor(mean_error_m)
    if len(mean_error_m) == 1:
        mean_error_m = mean_error_m[0]
    return mean_error_m, var_sq_m


def get_motion_deltas_from_path(vigor_dataset: vd.VigorDataset, path: list[int]):
    latlong = vigor_dataset.get_panorama_positions(path)
    motion_deltas = torch.diff(latlong, dim=0)

    return motion_deltas


def build_patch_index_from_particle(
        dataset: vd.VigorDataset,
        satellite_patch_config: SatellitePatchConfig,
        device: torch.device):
    patch_positions_px = torch.tensor(
            dataset._satellite_metadata[["web_mercator_y", "web_mercator_x"]].values,
            device=device, dtype=torch.float32)
    sat_patch_kdtree = build_kd_tree(patch_positions_px)
    num_patches = patch_positions_px.shape[0]

    def __inner__(particles: torch.Tensor):
        K = 1
        particles_px = web_mercator.latlon_to_pixel_coords(
                particles[..., 0], particles[..., 1], satellite_patch_config.zoom_level)

        # particles_px is num_particles x 2
        particles_px = torch.stack(particles_px, dim=-1)
        # px_dist_sq and idxs are num_particles x 1
        px_dist_sq, idxs = sat_patch_kdtree.query(particles_px, nr_nns_searches=K)
        # selected_patch_positions is num_particles x 2
        selected_patch_positions = patch_positions_px[idxs, :].squeeze()
        # abs_deltas is num_particles x 2
        abs_deltas = torch.abs(particles_px - selected_patch_positions)
        is_row_out_of_bounds = abs_deltas[:, 0] > satellite_patch_config.patch_height_px / 2.0
        is_col_out_of_bounds = abs_deltas[:, 1] > satellite_patch_config.patch_width_px / 2.0
        is_too_far = torch.logical_or(is_row_out_of_bounds, is_col_out_of_bounds)

        idxs[is_too_far] = num_patches
        return idxs.squeeze()
    return __inner__


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
        obs_likelihood_calculator: sa.ObservationLikelihoodCalculator,
        belief_weighting: sa.BeliefWeighting,
        initial_particle_state: torch.Tensor,  # N x state dim
        motion_deltas: torch.Tensor,  # path_length - 1 x state dim
        panorama_ids: list[str],  # path_length panorama IDs
        wag_config: WagConfig,
        generator: torch.Generator,
        return_intermediates: bool = False) -> PathInferenceResult:

    particle_state = initial_particle_state.clone()

    particle_history = []
    log_particle_weights = []
    particle_history_pre_move = []  # the particle state history before move_wag but after observe_wag
    num_dual_particles = []
    terminated_early = False
    for panorama_id, motion_delta in tqdm.tqdm(zip(panorama_ids[:-1], motion_deltas)):
        particle_history.append(particle_state.cpu().clone())

        # Generate new particles based on the observation
        wag_observation_result = sa.measurement_wag(
                particle_state,
                obs_likelihood_calculator,
                belief_weighting,
                panorama_id,
                wag_config,
                generator,
                return_past_particle_weights=return_intermediates)

        if wag_observation_result is None:
            # This happens when all particles have a -inf likelihood
            terminated_early = True
            break

        particle_state = wag_observation_result.resampled_particles
        if return_intermediates:
            log_particle_weights.append(wag_observation_result.log_particle_weights.cpu().clone())
            particle_history_pre_move.append(particle_state.cpu().clone())
            num_dual_particles.append(wag_observation_result.num_dual_particles)

        # move
        particle_state = sa.move_wag(particle_state, motion_delta, wag_config, generator)
    else:
        # apply final observation if we didn't break early
        wag_observation_result = sa.measurement_wag(
                particle_state,
                obs_likelihood_calculator,
                belief_weighting,
                panorama_ids[-1],
                wag_config,
                generator,
                return_past_particle_weights=return_intermediates)

        if return_intermediates:
            log_particle_weights.append(wag_observation_result.log_particle_weights.cpu().clone())
            particle_history_pre_move.append(particle_state.cpu().clone())
            num_dual_particles.append(wag_observation_result.num_dual_particles)

        particle_history.append(wag_observation_result.resampled_particles.cpu().clone())

    if return_intermediates:
        if len(num_dual_particles) > 0:
            num_dual_particles = torch.tensor(num_dual_particles)
            assert torch.all(num_dual_particles[0] == num_dual_particles) 
            num_dual_particles = num_dual_particles[0]
        else:
            num_dual_particles=None

        return PathInferenceResult(
            particle_history=torch.stack(particle_history),  # N+1, +1 from final particle state
            log_particle_weights=torch.stack(log_particle_weights),  # N
            particle_history_pre_move=torch.stack(particle_history_pre_move),
            num_dual_particles=num_dual_particles,
            terminated_early=terminated_early
        )
    else:
        return PathInferenceResult(
            particle_history=torch.stack(particle_history),
            log_particle_weights=None,
            particle_history_pre_move=None,
            num_dual_particles=None,
            terminated_early=terminated_early)


def construct_inputs_and_evaluate_path(
    vigor_dataset: vd.VigorDataset,
    path: list[int],
    generator_seed: int,
    device: str,
    wag_config: WagConfig,
    obs_likelihood_calculator: sa.ObservationLikelihoodCalculator,
    return_intermediates: bool = False,
) -> PathInferenceResult:
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    motion_deltas = get_motion_deltas_from_path(vigor_dataset, path).to(device)

    # Convert path indices to panorama IDs
    panorama_ids = [vigor_dataset._panorama_metadata.iloc[idx].pano_id for idx in path]

    torch.save(
        dict(ground_truth = vigor_dataset._panorama_metadata.iloc[path]),
        '/tmp/ground_truth.pt'
    )


    # Get satellite patch locations
    satellite_patch_locations = vigor_dataset.get_patch_positions()

    # Build patch index function for spatial discretization
    # This is used by both observation likelihood and belief weighting
    patch_index_from_particle = build_patch_index_from_particle(
        vigor_dataset, wag_config.satellite_patch_config, device)

    # Create belief weighting (independent of observation model)
    belief_weighting = sa.BeliefWeighting(
        satellite_patch_locations=satellite_patch_locations,
        patch_index_from_particle=patch_index_from_particle,
        phantom_counts_frac=wag_config.dual_mcl_belief_phantom_counts_frac
    )

    # Initialize particles
    gt_initial_position_lat_lon = vigor_dataset._panorama_metadata.iloc[path[0]]
    gt_initial_position_lat_lon = torch.tensor(
        (gt_initial_position_lat_lon['lat'], gt_initial_position_lat_lon['lon']), device=device)
    initial_particle_state = sa.initialize_wag_particles(
        gt_initial_position_lat_lon, wag_config, generator).to(device)

    return run_inference_on_path(
        obs_likelihood_calculator=obs_likelihood_calculator,
        belief_weighting=belief_weighting,
        initial_particle_state=initial_particle_state,
        motion_deltas=motion_deltas,
        panorama_ids=panorama_ids,
        wag_config=wag_config,
        generator=generator,
        return_intermediates=return_intermediates)

def evaluate_model_on_paths(
    vigor_dataset: vd.VigorDataset,
    sat_model: torch.nn.Module,
    pano_model: torch.nn.Module,
    paths: list[list[int]],
    wag_config: WagConfig,
    seed: int,
    output_path: Path,
    device: torch.device = "cuda:0",
    use_cached_similarity: bool = True,
    save_intermediate_filter_states=False,
    obs_likelihood_calculator: sa.ObservationLikelihoodCalculator | None = None,
) -> None:
    all_final_particle_error_meters = []
    with torch.no_grad():
        all_similarity = compute_cached_similarity_matrix(
                sat_model=sat_model,
                pano_model=pano_model,
                dataset=vigor_dataset,
                device=device,
                use_cached_similarity=use_cached_similarity)

        # Build shared components for observation likelihood
        satellite_patch_locations = vigor_dataset.get_patch_positions()
        patch_index_from_particle = build_patch_index_from_particle(
            vigor_dataset, wag_config.satellite_patch_config, device)

        # Create observation likelihood calculator if not provided
        if obs_likelihood_calculator is None:
            # Get all panorama IDs
            all_panorama_ids = vigor_dataset._panorama_metadata['pano_id'].tolist()

            obs_likelihood_calculator = sa.WagObservationLikelihoodCalculator(
                similarity_matrix=all_similarity,
                panorama_ids=all_panorama_ids,
                satellite_patch_locations=satellite_patch_locations,
                patch_index_from_particle=patch_index_from_particle,
                sigma=wag_config.sigma_obs_prob_from_sim,
                device=device
            )

        print("starting iter over paths")
        for i, path in enumerate(tqdm.tqdm(paths)):
            generator_seed = seed * i

            path_inference_result = construct_inputs_and_evaluate_path(
                vigor_dataset=vigor_dataset,
                path=path,
                generator_seed=generator_seed,
                device=device,
                wag_config=wag_config,
                obs_likelihood_calculator=obs_likelihood_calculator,
                return_intermediates=save_intermediate_filter_states)

            particle_history = path_inference_result.particle_history
            save_path = output_path / f"{i:07d}"
            save_path.mkdir(parents=True, exist_ok=True)
            error_meters_at_each_step, var_sq_m_at_each_step = (
                    get_distance_error_between_pano_and_particles_meters(
                        vigor_dataset, path, particle_history.to(device))
                )
            all_final_particle_error_meters.append(error_meters_at_each_step[-1])
            torch.save(error_meters_at_each_step, save_path / "error.pt")
            torch.save(var_sq_m_at_each_step, save_path / "var.pt")
            torch.save(path, save_path / "path.pt")
            # torch.save(all_similarity[path], save_path / "similarity.pt")
            if save_intermediate_filter_states:
                pir = path_inference_result
                torch.save(pir.particle_history, save_path / "particle_history.pt")
                torch.save(pir.log_particle_weights, save_path / "log_particle_weights.pt")
                torch.save(pir.particle_history_pre_move, save_path / "particle_history_pre_move.pt")
                torch.save(pir.dual_mcl_particles, save_path / "dual_mcl_particles.pt")
                torch.save(pir.dual_log_particle_weights, save_path / "dual_log_particle_weights.pt")
                torch.save(pir.num_dual_particles, save_path / "num_dual_particles.pt")
            with open(save_path / "other_info.json", "w") as f:
                f.write(json.dumps({
                    "seed": generator_seed,
                }, indent=2))
        average_final_error_meters = torch.tensor(all_final_particle_error_meters).mean().item()
        with open(output_path / "summary_statistics.json", 'w') as f:
            f.write(json.dumps({
                "average_final_error": average_final_error_meters,
                "terminated_early": path_inference_result.terminated_early
            }, indent=2))

        print(f"Average final error meters is {average_final_error_meters}")
