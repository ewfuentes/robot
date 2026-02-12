import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import SatellitePatchConfig
from experimental.overhead_matching.swag.evaluation.convergence_metrics import (
    compute_probability_mass_within_radius,
    compute_convergence_cost,
)
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    CellToPatchMapping,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    ObservationLogLikelihoodAggregator,
    AggregatorConfig,
    aggregator_from_config,
    load_aggregator_config,
)
import common.torch.load_and_save_models as lsm
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
from experimental.overhead_matching.swag.evaluation.odometry_noise import (
    OdometryNoiseConfig,
    add_noise_to_motion_deltas,
)
from pathlib import Path
from common.gps import web_mercator
from common.math.haversine import find_d_on_unit_circle
import shutil
import msgspec
import json
import math
import common.torch.load_torch_deps
import torch
import tqdm
from dataclasses import dataclass, field


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


@dataclass
class HistogramFilterConfig:
    """Configuration for histogram filter evaluation."""
    noise_percent: float = 0.02  # Motion noise as fraction of motion magnitude
    subdivision_factor: int = 4  # Grid subdivision (4 = 160px cells at zoom 20)
    initial_std_deg: float = 0.0267  # ~2970m initial uncertainty
    initial_offset_std_deg: float = 0.0117  # ~1300m offset
    zoom_level: int = 20
    patch_size_px: int = 640
    odometry_noise: OdometryNoiseConfig | None = None  # Optional odometry noise config


@dataclass
class HistogramPathResult:
    """Result of running histogram filter on a single path."""
    mean_history: torch.Tensor  # (path_len + 1, 2) lat/lon estimates
    variance_history: torch.Tensor  # (path_len + 1, 2) variance in degrees squared
    final_belief: HistogramBelief
    # Convergence metrics: probability mass within radius at each step
    # Keys are radius in meters, values are (path_len + 1,) tensors
    prob_mass_by_radius: dict[int, torch.Tensor] = field(default_factory=dict)


def get_dataset_bounds(vigor_dataset: vd.VigorDataset) -> tuple[float, float, float, float]:
    """Get lat/lon bounds from satellite metadata."""
    sat_meta = vigor_dataset._satellite_metadata
    min_lat = sat_meta['lat'].min()
    max_lat = sat_meta['lat'].max()
    min_lon = sat_meta['lon'].min()
    max_lon = sat_meta['lon'].max()
    return min_lat, max_lat, min_lon, max_lon


def get_patch_positions_px(vigor_dataset: vd.VigorDataset, device: torch.device) -> torch.Tensor:
    """Get satellite patch centers in pixel coordinates."""
    patch_positions_px = torch.tensor(
        vigor_dataset._satellite_metadata[["web_mercator_y", "web_mercator_x"]].values,
        device=device, dtype=torch.float32)
    return patch_positions_px


def run_histogram_filter_on_path(
    belief: HistogramBelief,
    motion_deltas: torch.Tensor,
    path_pano_ids: list[str],
    log_likelihood_aggregator: ObservationLogLikelihoodAggregator,
    mapping: CellToPatchMapping,
    config: HistogramFilterConfig,
    true_latlons: torch.Tensor | None = None,
    convergence_radii: list[int] | None = None,
) -> HistogramPathResult:
    """Run histogram filter on a single path.

    Args:
        belief: Initial histogram belief
        motion_deltas: (path_len - 1, 2) motion deltas in lat/lon degrees
        path_pano_ids: List of pano_ids for the path
        log_likelihood_aggregator: Aggregator to compute observation log-likelihoods
        mapping: Cell-to-patch mapping
        config: Filter configuration
        true_latlons: (path_len, 2) ground truth positions for convergence metrics
        convergence_radii: List of radii in meters for convergence metrics

    Returns:
        HistogramPathResult with mean/variance history and convergence metrics
    """
    mean_history = [belief.get_mean_latlon()]
    variance_history = [belief.get_variance_deg_sq()]

    # Initialize convergence tracking
    track_convergence = true_latlons is not None and convergence_radii is not None
    prob_mass_by_radius: dict[int, list[float]] = {}
    if track_convergence:
        for radius in convergence_radii:
            prob_mass_by_radius[radius] = []
            # Record initial probability mass (before any observations)
            prob_mass = compute_probability_mass_within_radius(
                belief, true_latlons[0], float(radius)
            )
            prob_mass_by_radius[radius].append(prob_mass)
    else:
        if true_latlons is None:
            print("Not tracking convergence: true_latlons not provided")
        if convergence_radii is None:
            print("Not tracking convergence: convergence_radii not provided")

    path_len = len(path_pano_ids)

    for step_idx in range(path_len - 1):
        # Observation update
        obs_log_ll = log_likelihood_aggregator(path_pano_ids[step_idx])
        belief.apply_observation(obs_log_ll, mapping)

        mean_history.append(belief.get_mean_latlon())
        variance_history.append(belief.get_variance_deg_sq())

        # Track convergence after observation (before motion blurs the belief)
        if track_convergence:
            for radius in convergence_radii:
                prob_mass = compute_probability_mass_within_radius(
                    belief, true_latlons[step_idx + 1], float(radius)
                )
                prob_mass_by_radius[radius].append(prob_mass)

        # Motion prediction
        belief.apply_motion(motion_deltas[step_idx], config.noise_percent)

    # Final observation
    obs_log_ll = log_likelihood_aggregator(path_pano_ids[-1])
    belief.apply_observation(obs_log_ll, mapping)
    mean_history.append(belief.get_mean_latlon())
    variance_history.append(belief.get_variance_deg_sq())

    # Track convergence after final observation
    if track_convergence:
        for radius in convergence_radii:
            prob_mass = compute_probability_mass_within_radius(
                belief, true_latlons[-1], float(radius)
            )
            prob_mass_by_radius[radius].append(prob_mass)

    # Convert lists to tensors
    prob_mass_tensors = {
        radius: torch.tensor(masses) for radius, masses in prob_mass_by_radius.items()
    }

    return HistogramPathResult(
        mean_history=torch.stack(mean_history),
        variance_history=torch.stack(variance_history),
        final_belief=belief,
        prob_mass_by_radius=prob_mass_tensors,
    )


def get_distance_error_from_mean_history(
    vigor_dataset: vd.VigorDataset,
    path: list[str],
    mean_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distance error between mean estimates and ground truth.

    Args:
        vigor_dataset: Dataset with panorama positions
        path: List of panorama indices
        mean_history: (path_len + 1, 2) lat/lon estimates

    Returns:
        error_meters: (path_len + 1,) distance error in meters
        variance_sq_m: (path_len + 1,) variance in meters squared (placeholder)
    """
    true_latlon = vigor_dataset.get_panorama_positions(path).to(device=mean_history.device)

    # mean_history has len(path) + 1 entries (initial + after each observation)
    # But we only have len(path) ground truth positions
    # The history is: initial -> obs[0] -> move -> obs[1] -> move -> ... -> obs[-1]
    # We compare at each step, so we need to align properly
    # mean_history[0] = before first observation
    # mean_history[1] = after first observation + first move
    # ...
    # mean_history[-1] = after last observation

    # Actually looking at the particle filter code, particle_history has len(path) entries
    # But our mean_history has path_len + 1 entries because we track before first obs too
    # Let's match the particle filter: compare estimate at each position

    # For simplicity, use the last N entries of mean_history to match path length
    estimates = mean_history[-len(path):]

    error_meters = []
    for i in range(len(path)):
        d = vd.EARTH_RADIUS_M * find_d_on_unit_circle(true_latlon[i], estimates[i])
        error_meters.append(d)

    return torch.tensor(error_meters, device=mean_history.device)


def evaluate_histogram_on_paths(
    vigor_dataset: vd.VigorDataset,
    log_likelihood_aggregator: ObservationLogLikelihoodAggregator,
    paths: list[list[str]],
    config: HistogramFilterConfig,
    seed: int,
    output_path: Path,
    device: torch.device = "cuda:0",
    save_intermediate_states: bool = False,
    convergence_radii: list[int] | None = None,
) -> None:
    """Evaluate histogram filter on paths.

    Args:
        vigor_dataset: VIGOR dataset
        log_likelihood_aggregator: Aggregator to compute observation log-likelihoods
        paths: List of paths (each path is list of pano_ids)
        config: Histogram filter configuration
        seed: Random seed for initial offset
        output_path: Directory to save results
        device: Torch device
        save_intermediate_states: Whether to save belief history
        convergence_radii: List of radii in meters for convergence metrics
    """
    all_final_error_meters = []
    # Track convergence costs per radius
    convergence_costs_by_radius: dict[int, list[float]] = {}
    if convergence_radii:
        for radius in convergence_radii:
            convergence_costs_by_radius[radius] = []

    with torch.no_grad():
        # Build GridSpec from dataset bounds with buffer of half patch size
        min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
        cell_size_px = config.patch_size_px / config.subdivision_factor

        # Add buffer of half patch size (in pixels at zoom level)
        # Convert to degrees using web mercator at the center latitude
        patch_half_size_px = config.patch_size_px / 2.0
        center_lat = (min_lat + max_lat) / 2
        ref_y, ref_x = web_mercator.latlon_to_pixel_coords(center_lat, min_lon, config.zoom_level)
        buf_lat, _ = web_mercator.pixel_coords_to_latlon(ref_y - patch_half_size_px, ref_x, config.zoom_level)
        _, buf_lon = web_mercator.pixel_coords_to_latlon(ref_y, ref_x + patch_half_size_px, config.zoom_level)
        lat_buffer = buf_lat - center_lat
        lon_buffer = buf_lon - min_lon

        grid_spec = GridSpec.from_bounds_and_cell_size(
            min_lat=min_lat - lat_buffer,
            max_lat=max_lat + lat_buffer,
            min_lon=min_lon - lon_buffer,
            max_lon=max_lon + lon_buffer,
            zoom_level=config.zoom_level,
            cell_size_px=cell_size_px,
        )
        print(f"Grid size: {grid_spec.num_rows} x {grid_spec.num_cols} = {grid_spec.num_rows * grid_spec.num_cols} cells")

        # Get patch positions and build mapping
        patch_positions_px = get_patch_positions_px(vigor_dataset, device)
        patch_half_size_px = config.patch_size_px / 2.0
        mapping = build_cell_to_patch_mapping(
            grid_spec=grid_spec,
            patch_positions_px=patch_positions_px,
            patch_half_size_px=patch_half_size_px,
            device=device,
        )
        print(f"Built cell-to-patch mapping with {len(mapping.patch_indices)} overlaps")

        print("Starting iteration over paths")
        for i, path in enumerate(tqdm.tqdm(paths)):
            generator_seed = seed * (i + 1)
            generator = torch.Generator(device=device).manual_seed(generator_seed)

            # Initialize belief
            belief = HistogramBelief.from_uniform(
                grid_spec=grid_spec,
                device=device,
            )

            # Get motion deltas for this path
            motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path).to(device)

            # Apply odometry noise if configured
            if config.odometry_noise is not None:
                start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(device)
                noise_gen = torch.Generator(device='cpu').manual_seed(
                    config.odometry_noise.seed * (i + 1))
                motion_deltas = add_noise_to_motion_deltas(
                    motion_deltas.cpu(), start_latlon.cpu(), config.odometry_noise,
                    generator=noise_gen,
                ).to(device)

            # Get ground truth positions for convergence metrics (only if needed)
            true_latlons = (
                vigor_dataset.get_panorama_positions(path).to(device)
                if convergence_radii else None
            )

            # Run filter
            result = run_histogram_filter_on_path(
                belief=belief,
                motion_deltas=motion_deltas,
                path_pano_ids=path,
                log_likelihood_aggregator=log_likelihood_aggregator,
                mapping=mapping,
                config=config,
                true_latlons=true_latlons,
                convergence_radii=convergence_radii,
            )

            # Compute distance traveled
            distance_traveled_m = es.compute_distance_traveled(vigor_dataset, path)

            # Compute error
            error_meters = get_distance_error_from_mean_history(
                vigor_dataset, path, result.mean_history)

            # Variance in meters squared (convert from degrees)
            var_sq_m = result.variance_history[-len(path):].sum(dim=-1) * (web_mercator.METERS_PER_DEG_LAT ** 2)

            all_final_error_meters.append(error_meters[-1].item())

            # Save results
            save_path = output_path / f"{i:07d}"
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(error_meters, save_path / "error.pt")
            torch.save(var_sq_m, save_path / "var.pt")
            torch.save(path, save_path / "path.pt")
            torch.save(distance_traveled_m, save_path / "distance_traveled_m.pt")

            if save_intermediate_states:
                torch.save(result.mean_history.cpu(), save_path / "mean_history.pt")
                torch.save(result.variance_history.cpu(), save_path / "variance_history.pt")

            # Save convergence metrics
            if convergence_radii and result.prob_mass_by_radius:
                torch.save(result.prob_mass_by_radius, save_path / "prob_mass_by_radius.pt")
                # Compute and track convergence costs
                for radius in convergence_radii:
                    cost = compute_convergence_cost(
                        result.prob_mass_by_radius[radius],
                        distance_traveled_m,
                    )
                    convergence_costs_by_radius[radius].append(cost)

            with open(save_path / "other_info.json", "w") as f:
                f.write(json.dumps({
                    "seed": generator_seed,
                }, indent=2))

        # Summary statistics
        average_final_error = sum(all_final_error_meters) / len(all_final_error_meters)
        summary_stats = {
            "average_final_error": average_final_error,
            "filter_type": "histogram",
            "grid_rows": grid_spec.num_rows,
            "grid_cols": grid_spec.num_cols,
            "cell_size_px": cell_size_px,
        }

        # Add convergence metrics to summary
        if convergence_radii:
            for radius in convergence_radii:
                costs = convergence_costs_by_radius[radius]
                summary_stats[f"convergence_cost_{radius}m"] = costs
                summary_stats[f"mean_convergence_cost_{radius}m"] = (
                    sum(costs) / len(costs) if costs else 0.0
                )

        with open(output_path / "summary_statistics.json", "w") as f:
            f.write(json.dumps(summary_stats, indent=2))

        print(f"Average final error meters: {average_final_error:.2f}")
        if convergence_radii:
            for radius in convergence_radii:
                mean_cost = summary_stats[f"mean_convergence_cost_{radius}m"]
                print(f"Mean convergence cost ({radius}m): {mean_cost:.2f}")


def construct_path_eval_inputs_from_args(
        dataset_path: str,
        paths_path: str,
        panorama_neighbor_radius_deg: float,
        panorama_landmark_radius_px: int,
        device: torch.device,
        landmark_version: str,
        sat_model_path: str | None = None,
        pano_model_path: str | None = None,
):
    """Load dataset and optionally models for evaluation.

    Args:
        dataset_path: Path to VIGOR dataset
        paths_path: Path to JSON file with evaluation paths
        panorama_neighbor_radius_deg: Panorama neighbor radius in degrees
        panorama_landmark_radius_px: Panorama landmark radius in pixels
        device: Torch device
        landmark_version: Landmark version string
        sat_model_path: Optional path to satellite model (required if pano_model_path is set)
        pano_model_path: Optional path to panorama model (required if sat_model_path is set)

    Returns:
        Tuple of (vigor_dataset, sat_model, pano_model, paths_data)
        Models will be None if model paths are not provided.
    """
    with open(paths_path, 'r') as f:
        paths_data = json.load(f)
    # Check that paths use pano_id strings, not old integer indices
    paths = paths_data.get('paths', [])
    if paths and paths[0] and isinstance(paths[0][0], int):
        raise ValueError(
            f"Path file '{paths_path}' uses old index format (integers). "
            "Regenerate with create_evaluation_paths.py to get pano_id format (strings)."
        )
    factor = paths_data.get('args', {}).get('factor', 1.0)
    print(f"Dataset Factor: {factor}")

    dataset_path = Path(dataset_path).expanduser()

    # Check that both model paths are provided or neither
    if (sat_model_path is None) != (pano_model_path is None):
        raise ValueError("Both sat_model_path and pano_model_path must be provided together, or neither.")

    # Load models and set config values based on whether models are provided
    if sat_model_path and pano_model_path:
        pano_model = load_model(pano_model_path, device=device)
        sat_model = load_model(sat_model_path, device=device)
        panorama_tensor_cache_info = vd.TensorCacheInfo(
            dataset_keys=[dataset_path.name],
            model_type="panorama",
            landmark_version=landmark_version,
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=pano_model.cache_info())
        satellite_tensor_cache_info = vd.TensorCacheInfo(
            dataset_keys=[dataset_path.name],
            model_type="satellite",
            landmark_version=landmark_version,
            panorama_landmark_radius_px=panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=1.0,
            extractor_info=sat_model.cache_info())
        satellite_patch_size = sat_model.patch_dims
        panorama_size = pano_model.patch_dims
        should_load_images = True
        should_load_landmarks = True
    else:
        pano_model = None
        sat_model = None
        panorama_tensor_cache_info = None
        satellite_tensor_cache_info = None
        satellite_patch_size = None
        panorama_size = None
        should_load_images = False
        should_load_landmarks = False

    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=panorama_tensor_cache_info,
        satellite_tensor_cache_info=satellite_tensor_cache_info,
        panorama_neighbor_radius=panorama_neighbor_radius_deg,
        satellite_patch_size=satellite_patch_size,
        panorama_size=panorama_size,
        factor=factor,
        landmark_version=landmark_version,
        should_load_images=should_load_images,
        should_load_landmarks=should_load_landmarks,
    )

    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    return vigor_dataset, sat_model, pano_model, paths_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate histogram filter on paths")

    parser.add_argument("--aggregator-config", type=str, required=True,
                        help="Path to YAML config file for aggregator (see adaptive_aggregators.py)")

    parser.add_argument("--paths-path", type=str, required=True,
                        help="Path to json file full of evaluation paths")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--save-intermediate-filter-states", action='store_true',
                        help="Save intermediate filter states (mean/variance history)")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float,
                        default=0.0005, help="Panorama neighbor radius deg")
    parser.add_argument("--panorama-landmark-radius-px", type=int,
                        default=640, help="Panorama landmark radius in pixels")
    parser.add_argument("--noise-percent", type=float, default=0.02,
                        help="Motion noise as fraction of motion magnitude")
    parser.add_argument("--subdivision-factor", type=int, default=4,
                        help="Grid subdivision factor (4 = 160px cells)")
    parser.add_argument("--convergence-radii", type=str, default="25,50,100",
                        help="Comma-separated list of radii (meters) for convergence metrics")

    # Odometry noise arguments
    parser.add_argument("--odometry-noise-frac", type=float, default=None,
                        help="Noise std as fraction of step distance (isotropic north/east)")
    parser.add_argument("--odometry-noise-seed", type=int, default=7919,
                        help="Seed for odometry noise generation")

    args = parser.parse_args()

    # Parse convergence radii
    convergence_radii = [int(r.strip()) for r in args.convergence_radii.split(",")]

    DEVICE = "cuda:0"
    torch.set_deterministic_debug_mode('error')

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_path) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    args.output_path = Path(args.output_path).expanduser()

    # Copy aggregator config to output directory for reproducibility
    shutil.copy(args.aggregator_config, args.output_path / "aggregator_config.yaml")

    vigor_dataset, sat_model, pano_model, paths_data = construct_path_eval_inputs_from_args(
        dataset_path=args.dataset_path,
        paths_path=args.paths_path,
        panorama_neighbor_radius_deg=args.panorama_neighbor_radius_deg,
        panorama_landmark_radius_px=args.panorama_landmark_radius_px,
        device=DEVICE,
        landmark_version=args.landmark_version,
    )

    # Load aggregator config and create aggregator
    aggregator_config = load_aggregator_config(Path(args.aggregator_config))
    print(f"Loaded aggregator config: {type(aggregator_config).__name__}")
    log_likelihood_aggregator = aggregator_from_config(
        aggregator_config,
        vigor_dataset,
        DEVICE,
    )

    # Build config
    def degrees_from_meters(dist_m):
        EARTH_RADIUS_M = 6_371_000.0
        return math.degrees(dist_m / EARTH_RADIUS_M)

    # Build odometry noise config
    odometry_noise_config = None
    if args.odometry_noise_frac is not None:
        odometry_noise_config = OdometryNoiseConfig(
            sigma_noise_frac=args.odometry_noise_frac,
            seed=args.odometry_noise_seed,
        )
        print(f"Odometry noise enabled: sigma_frac={odometry_noise_config.sigma_noise_frac}, seed={odometry_noise_config.seed}")

    config = HistogramFilterConfig(
        noise_percent=args.noise_percent,
        subdivision_factor=args.subdivision_factor,
        initial_std_deg=degrees_from_meters(2970.0),
        initial_offset_std_deg=degrees_from_meters(1300.0),
        odometry_noise=odometry_noise_config,
    )

    histogram_config_dict = {
        "noise_percent": config.noise_percent,
        "subdivision_factor": config.subdivision_factor,
        "initial_std_deg": config.initial_std_deg,
        "initial_offset_std_deg": config.initial_offset_std_deg,
        "zoom_level": config.zoom_level,
        "patch_size_px": config.patch_size_px,
    }
    if odometry_noise_config is not None:
        histogram_config_dict["odometry_noise"] = {
            "sigma_noise_frac": odometry_noise_config.sigma_noise_frac,
            "seed": odometry_noise_config.seed,
        }

    with open(Path(args.output_path) / "histogram_config.json", "w") as f:
        json.dump(histogram_config_dict, f, indent=4)

    evaluate_histogram_on_paths(
        vigor_dataset=vigor_dataset,
        log_likelihood_aggregator=log_likelihood_aggregator,
        paths=paths_data['paths'],
        config=config,
        seed=args.seed,
        output_path=args.output_path,
        device=DEVICE,
        save_intermediate_states=args.save_intermediate_filter_states,
        convergence_radii=convergence_radii,
    )
