"""Grid search over combined observation likelihood parameters.

This script performs a grid search over the parameters of CombinedObservationLikelihoodCalculator
to find the best combination of sigma values and weights for fusing satellite and OSM embeddings.

The similarity matrices are precomputed once (expensive), then the grid search over
config parameters (sigma, weights) is cheap.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:grid_search_combined_likelihood -- \
        --sat-model-path /path/to/sat/model \
        --osm-model-path /path/to/osm/model \
        --sat-pano-model-path /path/to/sat/pano/model \
        --osm-pano-model-path /path/to/osm/pano/model \
        --dataset-path /path/to/vigor/dataset \
        --paths-path /path/to/paths.json \
        --output-path /path/to/output
"""

import common.torch.load_torch_deps
import torch
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
import tqdm

import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
import experimental.overhead_matching.swag.evaluation.swag_algorithm as sa
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig, SatellitePatchConfig
from experimental.overhead_matching.swag.evaluation.combined_observation_likelihood import (
    CombinedObservationLikelihoodCalculator,
    CombinedObservationLikelihoodConfig,
    CombinedLikelihoodMode,
)
import common.torch.load_and_save_models as lsm
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
import msgspec
import math


@dataclass
class GridSearchResult:
    """Result from a single grid search evaluation."""
    sat_sigma: float
    osm_sigma: float
    sat_weight: float
    osm_weight: float
    mode: str
    average_final_error_meters: float
    num_paths: int


def load_model(path, device='cuda'):
    """Load a model from a path."""
    try:
        model = lsm.load_model(path, device=device)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        print(f"Failed to load model with lsm.load_model: {e}")
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


def degrees_from_meters(dist_m):
    """Convert meters to degrees (approximate)."""
    EARTH_RADIUS_M = 6_371_000.0
    return math.degrees(dist_m / EARTH_RADIUS_M)


def evaluate_single_config(
    vigor_dataset: vd.VigorDataset,
    sat_similarity_matrix: torch.Tensor,
    osm_similarity_matrix: torch.Tensor,
    paths: list[list[int]],
    wag_config: WagConfig,
    config: CombinedObservationLikelihoodConfig,
    seed: int,
    device: torch.device,
    save_path: Path,
) -> float:
    """Evaluate a single config and return average final error in meters.

    Saves per-path results to save_path in the same format as evaluate_model_on_paths.
    """
    # Build shared components
    satellite_patch_locations = vigor_dataset.get_patch_positions()
    patch_index_from_particle = es.build_patch_index_from_particle(
        vigor_dataset, wag_config.satellite_patch_config, device)

    # Get all panorama IDs
    all_panorama_ids = vigor_dataset._panorama_metadata['pano_id'].tolist()

    # Create combined observation likelihood calculator
    obs_likelihood_calculator = CombinedObservationLikelihoodCalculator(
        sat_similarity_matrix=sat_similarity_matrix,
        osm_similarity_matrix=osm_similarity_matrix,
        panorama_ids=all_panorama_ids,
        satellite_patch_locations=satellite_patch_locations,
        patch_index_from_particle=patch_index_from_particle,
        config=config,
        device=device,
    )

    all_final_errors = []

    for i, path in enumerate(paths):
        generator_seed = seed + i

        # Use the existing evaluation infrastructure
        path_inference_result = es.construct_inputs_and_evaluate_path(
            vigor_dataset=vigor_dataset,
            path=path,
            generator_seed=generator_seed,
            device=device,
            wag_config=wag_config,
            obs_likelihood_calculator=obs_likelihood_calculator,
            return_intermediates=False,
        )

        # Get final particle positions and compute error
        particle_history = path_inference_result.particle_history.to(device)
        error_meters_at_each_step, var_sq_m_at_each_step = es.get_distance_error_between_pano_and_particles_meters(
            vigor_dataset, path, particle_history)
        all_final_errors.append(error_meters_at_each_step[-1].item())

        # Compute distance traveled
        distance_traveled_m = es.compute_distance_traveled(vigor_dataset, path)

        # Save per-path results
        path_save_dir = save_path / f"{i:07d}"
        path_save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(error_meters_at_each_step, path_save_dir / "error.pt")
        torch.save(var_sq_m_at_each_step, path_save_dir / "var.pt")
        torch.save(path, path_save_dir / "path.pt")
        torch.save(distance_traveled_m, path_save_dir / "distance_traveled_m.pt")

        with open(path_save_dir / "other_info.json", "w") as f:
            json.dump({
                "seed": generator_seed,
                "terminated_early": path_inference_result.terminated_early,
            }, f, indent=2)

    return sum(all_final_errors) / len(all_final_errors)


def run_grid_search(
    vigor_dataset: vd.VigorDataset,
    sat_similarity_matrix: torch.Tensor,
    osm_similarity_matrix: torch.Tensor,
    paths: list[list[int]],
    wag_config: WagConfig,
    seed: int,
    device: torch.device,
    output_path: Path,
    sat_sigma_values: list[float],
    osm_sigma_values: list[float],
    alpha_values: list[float],
) -> list[GridSearchResult]:
    """Run grid search over all parameter combinations.

    Args:
        alpha_values: Values for alpha where sat_weight=alpha, osm_weight=1-alpha.
            alpha=1.0 means SAT only, alpha=0.0 means OSM only.
        output_path: Base directory for saving per-path results.
    """
    results = []

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        sat_sigma_values, osm_sigma_values, alpha_values
    ))

    print(f"Running grid search over {len(param_combinations)} parameter combinations...")

    for sat_sigma, osm_sigma, alpha in tqdm.tqdm(param_combinations):
        sat_weight = alpha
        osm_weight = 1.0 - alpha

        config = CombinedObservationLikelihoodConfig(
            mode=CombinedLikelihoodMode.COMBINED,
            sat_sigma=sat_sigma,
            osm_sigma=osm_sigma,
            sat_weight=sat_weight,
            osm_weight=osm_weight,
        )

        # Create config directory name with explicit weight names
        config_dir_name = (
            f"sat_sigma_{sat_sigma:.2f}_osm_sigma_{osm_sigma:.2f}_"
            f"sat_weight_{sat_weight:.2f}_osm_weight_{osm_weight:.2f}"
        )
        config_save_path = output_path / config_dir_name

        avg_error = evaluate_single_config(
            vigor_dataset=vigor_dataset,
            sat_similarity_matrix=sat_similarity_matrix,
            osm_similarity_matrix=osm_similarity_matrix,
            paths=paths,
            wag_config=wag_config,
            config=config,
            seed=seed,
            device=device,
            save_path=config_save_path,
        )

        result = GridSearchResult(
            sat_sigma=sat_sigma,
            osm_sigma=osm_sigma,
            sat_weight=sat_weight,
            osm_weight=osm_weight,
            mode=config.mode.value,
            average_final_error_meters=avg_error,
            num_paths=len(paths),
        )
        results.append(result)

        print(f"  sat_sigma={sat_sigma:.3f}, osm_sigma={osm_sigma:.3f}, "
              f"alpha={alpha:.2f} (sat={sat_weight:.2f}, osm={osm_weight:.2f}) -> "
              f"avg_error={avg_error:.2f}m")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grid search over combined likelihood parameters")
    parser.add_argument("--sat-model-path", type=str, required=True,
                        help="Path to satellite model folder")
    parser.add_argument("--osm-model-path", type=str, required=True,
                        help="Path to OSM model folder")
    parser.add_argument("--sat-pano-model-path", type=str, required=True,
                        help="Path to satellite panorama model folder")
    parser.add_argument("--osm-pano-model-path", type=str, required=True,
                        help="Path to OSM panorama model folder")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VIGOR dataset")
    parser.add_argument("--paths-path", type=str, required=True,
                        help="Path to JSON file containing evaluation paths")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--landmark-version", type=str, default="v1",
                        help="Landmark version")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float, default=0.0005,
                        help="Panorama neighbor radius in degrees")
    parser.add_argument("--panorama-landmark-radius-px", type=int, default=640,
                        help="Panorama landmark radius in pixels")

    # Grid search parameters
    parser.add_argument("--sat-sigma-values", type=str, default="0.05,0.1,0.2",
                        help="Comma-separated list of sat_sigma values to search")
    parser.add_argument("--osm-sigma-values", type=str, default="0.05,0.1,0.2",
                        help="Comma-separated list of osm_sigma values to search")
    parser.add_argument("--alpha-values", type=str, default="0.0,0.25,0.5,0.75,1.0",
                        help="Comma-separated list of alpha values (sat_weight=alpha, osm_weight=1-alpha). "
                             "alpha=1.0 means SAT only, alpha=0.0 means OSM only.")

    args = parser.parse_args()

    DEVICE = "cuda:0"
    torch.set_deterministic_debug_mode('error')

    # Parse grid search parameter values
    sat_sigma_values = [float(x) for x in args.sat_sigma_values.split(",")]
    osm_sigma_values = [float(x) for x in args.osm_sigma_values.split(",")]
    alpha_values = [float(x) for x in args.alpha_values.split(",")]

    print(f"Grid search parameters:")
    print(f"  sat_sigma: {sat_sigma_values}")
    print(f"  osm_sigma: {osm_sigma_values}")
    print(f"  alpha: {alpha_values} (sat_weight=alpha, osm_weight=1-alpha)")

    # Create output directory
    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load paths
    with open(args.paths_path, 'r') as f:
        paths_data = json.load(f)
    # Check for old format
    paths = paths_data['paths']
    if paths and paths[0] and isinstance(paths[0][0], int):
        raise ValueError(
            f"Path file '{args.paths_path}' uses old index format (integers). "
            "Regenerate with create_evaluation_paths.py to get pano_id format (strings)."
        )
    factor = paths_data.get('args', {}).get('factor', 1.0)

    print(f"Loaded {len(paths)} evaluation paths")

    # Load models
    print("Loading models...")
    sat_model_path = Path(args.sat_model_path).expanduser()
    osm_model_path = Path(args.osm_model_path).expanduser()
    sat_pano_model_path = Path(args.sat_pano_model_path).expanduser()
    osm_pano_model_path = Path(args.osm_pano_model_path).expanduser()

    sat_model = load_model(sat_model_path, device=DEVICE)
    osm_model = load_model(osm_model_path, device=DEVICE)
    sat_pano_model = load_model(sat_pano_model_path, device=DEVICE)
    osm_pano_model = load_model(osm_pano_model_path, device=DEVICE)

    # Create dataset config
    dataset_path = Path(args.dataset_path).expanduser()
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="panorama",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=sat_pano_model.cache_info()),
        satellite_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="satellite",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=sat_model.cache_info()),
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=sat_pano_model.patch_dims,
        factor=factor,
        landmark_version=args.landmark_version,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    # Compute similarity matrices (cached)
    print("Computing satellite similarity matrix...")
    sat_similarity_matrix = es.compute_cached_similarity_matrix(
        sat_model=sat_model,
        pano_model=sat_pano_model,
        dataset=vigor_dataset,
        device=DEVICE,
        use_cached_similarity=True
    )
    print(f"  Satellite similarity matrix shape: {sat_similarity_matrix.shape}")

    print("Computing OSM similarity matrix...")
    # For the OSM model, we need to build a dataset config with the OSM model's cache info
    osm_dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="panorama",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=osm_pano_model.cache_info()),
        satellite_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="satellite",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=osm_model.cache_info()),
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg,
        satellite_patch_size=osm_model.patch_dims,
        panorama_size=osm_pano_model.patch_dims,
        factor=factor,
        landmark_version=args.landmark_version,
    )
    osm_vigor_dataset = vd.VigorDataset(dataset_path, osm_dataset_config)

    osm_similarity_matrix = es.compute_cached_similarity_matrix(
        sat_model=osm_model,
        pano_model=osm_pano_model,
        dataset=osm_vigor_dataset,
        device=DEVICE,
        use_cached_similarity=True
    )
    print(f"  OSM similarity matrix shape: {osm_similarity_matrix.shape}")

    # Verify shapes match
    assert sat_similarity_matrix.shape == osm_similarity_matrix.shape, (
        f"Similarity matrices must have same shape. "
        f"Got sat: {sat_similarity_matrix.shape}, osm: {osm_similarity_matrix.shape}"
    )

    # Create WAG config
    wag_config = WagConfig(
        noise_percent_motion_model=0.02,
        initial_particle_distribution_offset_std_deg=degrees_from_meters(1300.0),
        initial_particle_distribution_std_deg=degrees_from_meters(2970.0),
        num_particles=10_000,
        sigma_obs_prob_from_sim=0.1,  # This will be overridden by grid search
        satellite_patch_config=SatellitePatchConfig(
            zoom_level=20,
            patch_height_px=640,
            patch_width_px=640),
        dual_mcl_frac=0.0,
        dual_mcl_belief_phantom_counts_frac=1e-4
    )

    # Run grid search
    print("\nStarting grid search...")
    results = run_grid_search(
        vigor_dataset=vigor_dataset,
        sat_similarity_matrix=sat_similarity_matrix,
        osm_similarity_matrix=osm_similarity_matrix,
        paths=paths,
        wag_config=wag_config,
        seed=args.seed,
        device=DEVICE,
        output_path=output_path,
        sat_sigma_values=sat_sigma_values,
        osm_sigma_values=osm_sigma_values,
        alpha_values=alpha_values,
    )

    # Save results
    results_dicts = [asdict(r) for r in results]
    with open(output_path / "grid_search_results.json", "w") as f:
        json.dump(results_dicts, f, indent=2)

    # Find best result
    best_result = min(results, key=lambda r: r.average_final_error_meters)
    print(f"\nBest result:")
    print(f"  sat_sigma: {best_result.sat_sigma}")
    print(f"  osm_sigma: {best_result.osm_sigma}")
    print(f"  sat_weight: {best_result.sat_weight}")
    print(f"  osm_weight: {best_result.osm_weight}")
    print(f"  average_final_error_meters: {best_result.average_final_error_meters:.2f}")

    with open(output_path / "best_result.json", "w") as f:
        json.dump(asdict(best_result), f, indent=2)

    print(f"\nResults saved to {output_path}")
