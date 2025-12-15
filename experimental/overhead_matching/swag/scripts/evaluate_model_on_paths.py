import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig, SatellitePatchConfig
from experimental.overhead_matching.swag.evaluation.observation_likelihood import (
    compute_cached_landmark_similarity_data,
    build_prior_data_from_vigor,
    LandmarkObservationLikelihoodCalculator,
    ObservationLikelihoodConfig,
    LikelihoodMode,
)
import common.torch.load_and_save_models as lsm
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
from pathlib import Path
import msgspec
import json
import math
import common.torch.load_torch_deps
from google.protobuf import text_format
import torch


def load_model(path, device='cuda'):
    try:
        model = lsm.load_model(path, device=device)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        print("Failed to load model", e)
        training_config_path = path.parent / "config.json"
        training_config_json = json.loads(training_config_path.read_text())
        model_config_json = training_config_json["sat_model_config"] if 'satellite' in path.name else training_config_json["pano_model_config"]
        config = msgspec.json.decode(
                json.dumps(model_config_json),
                type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)

        model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
        model_type = patch_embedding.WagPatchEmbedding if isinstance(config, patch_embedding.WagPatchEmbeddingConfig) else swag_patch_embedding.SwagPatchEmbedding
        model = model_type(config)
        model.load_state_dict(model_weights)
        model = model.to(device)
    return model


def construct_path_eval_inputs_from_args(
        sat_model_path: str,
        pano_model_path: str,
        dataset_path: str,
        paths_path: str,
        panorama_neighbor_radius_deg: float,
        panorama_landmark_radius_px: int,
        device: torch.device,
        landmark_version: str
):
    with open(paths_path, 'r') as f:
        paths_data = json.load(f)
    pano_model = load_model(pano_model_path, device=device)
    sat_model = load_model(sat_model_path, device=device)

    dataset_path = Path(dataset_path).expanduser()
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="panorama",
            landmark_version=landmark_version,
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            extractor_info=pano_model.cache_info()),
        satellite_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="satellite",
            landmark_version=landmark_version,
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            extractor_info=sat_model.cache_info()),
        panorama_neighbor_radius=panorama_neighbor_radius_deg,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
        factor=0.3,
        landmark_version=landmark_version,
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
    parser.add_argument("--landmark_version", type=str, required=True)
    parser.add_argument("--save-intermediate-filter-states", action='store_true',
                        help="If intermediate filter states should be saved")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float,
                        default=0.0005, help="Panorama neighbor radius deg")
    parser.add_argument("--panorama-landmark-radius-px", type=int,
                        default=640, help="Panorama landmark radius in pixels")
    parser.add_argument("--sigma_obs_prob_from_sim", type=float, default=0.1)
    parser.add_argument("--dual_mcl_frac", type=float, default=0.0)
    parser.add_argument("--dual_mcl_phantom_counts_frac", type=float, default=1e-4)
    parser.add_argument("--osm-embedding-path", type=str, default=None,
                        help="Path to OSM landmark embeddings directory")
    parser.add_argument("--pano-embedding-path", type=str, default=None,
                        help="Path to panorama landmark embeddings directory")
    parser.add_argument("--use-landmark-likelihood", action='store_true',
                        help="Use LandmarkObservationLikelihoodCalculator instead of WagObservationLikelihoodCalculator")
    parser.add_argument("--landmark-sigma", type=float, default=100.0,
                        help="Sigma for OSM landmark observation likelihood")
    parser.add_argument("--embedding-dim", type=int, default=1536,
                        help="Embedding dimension for landmark embeddings")
    parser.add_argument("--osm-similarity-scale", type=float, default=10.0,
                        help="Scale factor for OSM similarity in log-likelihood computation")

    args = parser.parse_args()

    # torch.cuda.memory._record_memory_history(max_entries=100_000)

    DEVICE = "cuda:0"
    torch.set_deterministic_debug_mode('error')

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
        panorama_landmark_radius_px=args.panorama_landmark_radius_px,
        device=DEVICE,
        landmark_version=args.landmark_version,
    )

    def degrees_from_meters(dist_m):
        EARTH_RADIUS_M = 6_371_000.0
        return math.degrees(dist_m / EARTH_RADIUS_M)

    wag_config = WagConfig(noise_percent_motion_model=0.02,  # page 71 thesis
                           # offset was fixed at 1.3km in thesis (page 71)
                           initial_particle_distribution_offset_std_deg=degrees_from_meters(1300.0),
                           # page 73 of thesis
                           initial_particle_distribution_std_deg=degrees_from_meters(2970.0),
                           num_particles=10_000,
                           sigma_obs_prob_from_sim=args.sigma_obs_prob_from_sim,
                           satellite_patch_config=SatellitePatchConfig(
                               zoom_level=20,
                               patch_height_px=640,
                               patch_width_px=640),
                           dual_mcl_frac=args.dual_mcl_frac,
                           dual_mcl_belief_phantom_counts_frac=args.dual_mcl_phantom_counts_frac)

    with open(Path(args.output_path) / "wag_config.pbtxt", "w") as f:
        f.write(text_format.MessageToString(wag_config))

    # Build landmark observation likelihood calculator if requested
    landmark_obs_calculator = None
    if args.use_landmark_likelihood:
        if args.osm_embedding_path is None or args.pano_embedding_path is None:
            raise ValueError("--osm-embedding-path and --pano-embedding-path are required when using --use-landmark-likelihood")

        osm_embedding_path = Path(args.osm_embedding_path).expanduser()
        pano_embedding_path = Path(args.pano_embedding_path).expanduser()

        import ipdb
        print("Computing landmark similarity data...")
        landmark_sim_data = compute_cached_landmark_similarity_data(
            vigor_dataset,
            osm_embedding_path,
            pano_embedding_path,
            embedding_dim=args.embedding_dim,
            use_cache=True
        )

        # Get pano_sat_similarity from evaluate_swag
        all_similarity = es.compute_cached_similarity_matrix(
            sat_model=sat_model,
            pano_model=pano_model,
            dataset=vigor_dataset,
            device=DEVICE,
            use_cached_similarity=True
        )

        print("Building prior data...")
        prior_data = build_prior_data_from_vigor(
            vigor_dataset,
            all_similarity,
            landmark_sim_data
        )

        obs_config = ObservationLikelihoodConfig(
            obs_likelihood_from_sat_similarity_sigma=args.sigma_obs_prob_from_sim,
            obs_likelihood_from_osm_similarity_sigma=args.landmark_sigma,
            osm_similarity_scale=args.osm_similarity_scale,
            likelihood_mode=LikelihoodMode.COMBINED
        )

        landmark_obs_calculator = LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=obs_config,
            device=torch.device(DEVICE)
        )
        print("Landmark observation likelihood calculator ready")

    with ipdb.launch_ipdb_on_exception():
        es.evaluate_model_on_paths(
            vigor_dataset=vigor_dataset,
            sat_model=sat_model,
            pano_model=pano_model,
            paths=paths_data['paths'],
            wag_config=wag_config,
            seed=args.seed,
            output_path=args.output_path,
            device=DEVICE,
            save_intermediate_filter_states=args.save_intermediate_filter_states,
            obs_likelihood_calculator=landmark_obs_calculator
        )
