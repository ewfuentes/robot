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
from experimental.overhead_matching.swag.farfield import (
    artifact as farfield_artifact,
    paths as farfield_paths,
)
from experimental.overhead_matching.swag.farfield.localization import (
    metrics as farfield_metrics,
    run_io as farfield_run_io,
    structs as farfield_structs,
)
from pathlib import Path
from common.gps import web_mercator
from common.math.haversine import find_d_on_unit_circle
import csv
import hashlib
import shutil
import msgspec
import json
import math
import string
import common.torch.load_torch_deps
import torch
import tqdm
import warnings
from dataclasses import dataclass, field


APPLIED_MOTION_DELTAS_FILENAME = "applied_motion_deltas.pt"
DEFAULT_CONVERGENCE_RADII = tuple(
    int(radius) for radius in farfield_metrics.DEFAULT_POSITION_MASS_RADII_M)
# Farfield truth historically came from six-decimal panorama filenames while
# pano_id_mapping.csv retains fuller GPS precision. Recovering the anchor from
# one rounded endpoint leaves up to two endpoint-rounding errors in another.
FARFIELD_TRUTH_MAPPING_TOLERANCE_DEG = 1.1e-6


def load_applied_motion_deltas(
        path_dir: Path, path_len: int) -> torch.Tensor:
    """Load the exact controls persisted by histogram evaluation."""
    artifact_path = path_dir / APPLIED_MOTION_DELTAS_FILENAME
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} is missing; regenerate this evaluation rather "
            "than reconstructing its odometry from a seed")
    motion_deltas = torch.load(
        artifact_path, map_location="cpu", weights_only=True)
    expected_shape = (max(0, path_len - 1), 2)
    if (not isinstance(motion_deltas, torch.Tensor)
            or tuple(motion_deltas.shape) != expected_shape
            or not torch.is_floating_point(motion_deltas)
            or not torch.isfinite(motion_deltas).all()):
        raise ValueError(
            f"{artifact_path} must be a finite floating-point tensor with "
            f"shape {expected_shape}")
    return motion_deltas


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
    motion_noise_frac: float = 0.05  # Wiener noise intensity (m/√m) for filter blur
    subdivision_factor: int = 4  # Grid subdivision (4 = 160px cells at zoom 20)
    initial_std_deg: float = 0.0267  # ~2970m initial uncertainty
    initial_offset_std_deg: float = 0.0117  # ~1300m offset
    zoom_level: int = 20
    patch_size_px: int = 640
    # source_px: actual footprint of each satellite patch in zoom-level pixels.
    # When satellite images are resolution-normalized (cropped to source_px and
    # resized to patch_size_px), the ground footprint is source_px, not patch_size_px.
    # Read from satellite_bbox.json; defaults to patch_size_px (no normalization).
    source_px: int | float | None = None
    odometry_noise: OdometryNoiseConfig | None = None  # Optional odometry noise config
    max_chunk_gib: float = 2.0  # Peak GPU memory per chunk for cell-to-patch mapping

    @property
    def footprint_px(self) -> float:
        """Ground footprint of each satellite patch in zoom-level pixels."""
        return float(self.source_px if self.source_px is not None else self.patch_size_px)


@dataclass
class HistogramPathResult:
    """Result of running histogram filter on a single path."""
    mean_history: torch.Tensor  # (path_len + 1, 2) lat/lon — weighted-mean estimate
    mode_history: torch.Tensor  # (path_len + 1, 2) lat/lon — argmax-cell (MAP) estimate
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
    mode_history = [belief.get_mode_latlon()]
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
        mode_history.append(belief.get_mode_latlon())
        variance_history.append(belief.get_variance_deg_sq())

        # Track convergence after observation (before motion blurs the belief)
        if track_convergence:
            for radius in convergence_radii:
                prob_mass = compute_probability_mass_within_radius(
                    belief, true_latlons[step_idx], float(radius)
                )
                prob_mass_by_radius[radius].append(prob_mass)

        # Motion prediction
        belief.apply_motion(motion_deltas[step_idx], config.motion_noise_frac)

    # Final observation
    obs_log_ll = log_likelihood_aggregator(path_pano_ids[-1])
    belief.apply_observation(obs_log_ll, mapping)
    mean_history.append(belief.get_mean_latlon())
    mode_history.append(belief.get_mode_latlon())
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
        mode_history=torch.stack(mode_history),
        variance_history=torch.stack(variance_history),
        final_belief=belief,
        prob_mass_by_radius=prob_mass_tensors,
    )


def get_distance_error_from_estimate_history(
    vigor_dataset: vd.VigorDataset,
    path: list[str],
    estimate_history: torch.Tensor,
    true_latlon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute distance error between position estimates and ground truth.

    Works for any per-step lat/lon estimate (e.g. weighted mean or argmax
    cell / MAP).

    Args:
        vigor_dataset: Dataset with panorama positions
        path: List of panorama indices
        estimate_history: (path_len + 1, 2) lat/lon estimates
        true_latlon: Optional authoritative (path_len, 2) truth positions. If
            omitted, use the legacy coordinates encoded in panorama filenames.

    Returns:
        error_meters: (path_len,) distance error in meters
    """
    if true_latlon is None:
        true_latlon = vigor_dataset.get_panorama_positions(path)
    true_latlon = true_latlon.to(device=estimate_history.device)

    # history[0] is the prior; history[i + 1] is the posterior after
    # observation i and before the following motion prediction.
    estimates = estimate_history[-len(path):]

    error_meters = []
    for i in range(len(path)):
        d = vd.EARTH_RADIUS_M * find_d_on_unit_circle(true_latlon[i], estimates[i])
        error_meters.append(d)

    return torch.stack(error_meters)


def load_authoritative_panorama_positions(
        mapping_path: Path) -> dict[str, tuple[float, float]]:
    """Load full-precision evaluation truth keyed by panorama ID."""
    positions = {}
    with open(mapping_path, newline="") as source:
        reader = csv.DictReader(source)
        required = {"pano_id", "lat", "lon"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{mapping_path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            pano_id = row["pano_id"]
            if not pano_id:
                raise ValueError(f"{mapping_path}:{line_number} has an empty pano_id")
            if pano_id in positions:
                raise ValueError(
                    f"{mapping_path}:{line_number} duplicates pano_id {pano_id!r}")
            try:
                latlon = (float(row["lat"]), float(row["lon"]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{mapping_path}:{line_number} has invalid lat/lon") from error
            if (not all(math.isfinite(value) for value in latlon)
                    or not -90.0 <= latlon[0] <= 90.0
                    or not -180.0 <= latlon[1] <= 180.0):
                raise ValueError(
                    f"{mapping_path}:{line_number} has out-of-range lat/lon")
            positions[pano_id] = latlon
    if not positions:
        raise ValueError(f"{mapping_path} contains no panorama positions")
    return positions


def authoritative_positions_for_path(
        positions_by_id: dict[str, tuple[float, float]],
        path: list[str]) -> torch.Tensor:
    """Return path truth at CSV precision, preserving path order."""
    try:
        positions = [positions_by_id[pano_id] for pano_id in path]
    except KeyError as error:
        raise ValueError(
            f"pano_id {error.args[0]!r} is missing from pano_id_mapping.csv") from error
    return torch.tensor(positions, dtype=torch.float64)


def compute_distance_traveled_from_positions(
        true_latlon: torch.Tensor) -> torch.Tensor:
    """Cumulative great-circle distance for explicit trajectory truth."""
    if true_latlon.ndim != 2 or true_latlon.shape[1] != 2 or len(true_latlon) == 0:
        raise ValueError("true_latlon must have non-empty shape (N, 2)")
    distance_delta = vd.EARTH_RADIUS_M * find_d_on_unit_circle(
        true_latlon[:-1], true_latlon[1:])
    return torch.cat((true_latlon.new_zeros(1), torch.cumsum(distance_delta, 0)))


def farfield_odometry_to_latlon_deltas(
        odometry: list[farfield_structs.OdometryDelta],
        truth: list[farfield_structs.TruthPose],
        path_latlons: torch.Tensor,
        *,
        reverse_keyframe_ranges: list,
        displacement_gate_m: float) -> torch.Tensor:
    """Convert realized forward/left odometry into LOCI lat/lon controls."""
    n_keyframes = len(truth)
    if n_keyframes < 2 or tuple(path_latlons.shape) != (n_keyframes, 2):
        raise ValueError(
            "farfield truth and LOCI path must contain the same >=2 keyframes")
    if [pose.keyframe_idx for pose in truth] != list(range(n_keyframes)):
        raise ValueError("farfield truth keyframes must be contiguous 0..N-1")
    if [delta.keyframe_idx for delta in odometry] != list(
            range(1, n_keyframes)):
        raise ValueError("farfield odometry keyframes must be contiguous 1..N-1")

    for pose in truth:
        if (not all(math.isfinite(value) for value in (
                pose.east_m, pose.north_m, pose.course_world_cw_deg))
                or not 0.0 <= pose.course_world_cw_deg < 360.0):
            raise ValueError(
                f"farfield truth at keyframe {pose.keyframe_idx} is invalid")
    for delta in odometry:
        if (not all(math.isfinite(value) for value in (
                delta.forward_m, delta.left_m, delta.delta_yaw_cw_rad,
                delta.sigma_m, delta.sigma_yaw_rad))
                or delta.sigma_m <= 0.0 or delta.sigma_yaw_rad <= 0.0):
            raise ValueError(
                f"farfield odometry at keyframe {delta.keyframe_idx} is invalid")
    if (isinstance(displacement_gate_m, bool)
            or not isinstance(displacement_gate_m, (int, float))
            or not math.isfinite(displacement_gate_m)
            or displacement_gate_m <= 0.0):
        raise ValueError("farfield displacement_gate_m must be finite and positive")
    if not isinstance(reverse_keyframe_ranges, list):
        raise ValueError("farfield reverse_keyframe_ranges must be a list")
    reverse_keyframes = set()
    previous_end = 0
    for index, interval in enumerate(reverse_keyframe_ranges):
        if (not isinstance(interval, list) or len(interval) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in interval)):
            raise ValueError(
                f"farfield reverse_keyframe_ranges[{index}] must be [start, end] ints")
        start, end = interval
        if (start < 1 or end < start or end >= n_keyframes
                or start <= previous_end):
            raise ValueError(
                "farfield reverse_keyframe_ranges must be sorted, "
                "non-overlapping, and within 1..N-1")
        reverse_keyframes.update(range(start, end + 1))
        previous_end = end

    path_latlons = path_latlons.to(dtype=torch.float64, device="cpu")
    if (not torch.isfinite(path_latlons).all()
            or not torch.all(path_latlons[:, 0].abs() <= 90.0)
            or not torch.all(path_latlons[:, 1].abs() <= 180.0)):
        raise ValueError("LOCI path lat/lon coordinates are invalid")

    # Farfield truth is expressed in a fixed-anchor ENU frame. Recover that
    # anchor from the first shared position so the conversion uses the exact
    # same longitude scale, then verify that both artifacts describe one path.
    meters_per_degree = web_mercator.METERS_PER_DEG_LAT
    anchor_lat = path_latlons[0, 0].item() - truth[0].north_m / meters_per_degree
    meters_per_degree_lon = meters_per_degree * math.cos(math.radians(anchor_lat))
    if not math.isfinite(meters_per_degree_lon) or abs(meters_per_degree_lon) < 1e-9:
        raise ValueError("farfield truth implies an invalid ENU anchor latitude")
    anchor_lon = path_latlons[0, 1].item() - truth[0].east_m / meters_per_degree_lon
    for pose, latlon in zip(truth, path_latlons.tolist()):
        expected = (
            anchor_lat + pose.north_m / meters_per_degree,
            anchor_lon + pose.east_m / meters_per_degree_lon,
        )
        if max(abs(actual - wanted) for actual, wanted in zip(
                latlon, expected)) > FARFIELD_TRUTH_MAPPING_TOLERANCE_DEG:
            raise ValueError(
                "farfield truth positions do not match the LOCI trajectory")

    result = []
    for delta in odometry:
        previous = truth[delta.keyframe_idx - 1]
        current = truth[delta.keyframe_idx]
        travel_east_m = current.east_m - previous.east_m
        travel_north_m = current.north_m - previous.north_m
        travel_m = math.hypot(travel_east_m, travel_north_m)
        if travel_m < displacement_gate_m:
            if delta.forward_m != 0.0 or delta.left_m != 0.0:
                raise ValueError(
                    "farfield stationary odometry must have zero realized "
                    f"translation at keyframe {delta.keyframe_idx}")
            result.append((0.0, 0.0))
            continue

        # The serialized producer uses each raw truth chord as the body frame;
        # truth.course_world_cw_deg is smoothed diagnostic course. A reviewed
        # reverse step points body-forward opposite travel. LOCI assumes that
        # heading is known, so delta_yaw remains deliberately unused.
        course = math.atan2(travel_east_m, travel_north_m)
        if delta.keyframe_idx in reverse_keyframes:
            course += math.pi
        east_m = (delta.forward_m * math.sin(course)
                  - delta.left_m * math.cos(course))
        north_m = (delta.forward_m * math.cos(course)
                   + delta.left_m * math.sin(course))
        result.append((
            north_m / meters_per_degree,
            east_m / meters_per_degree_lon,
        ))
    return torch.tensor(result, dtype=torch.float64)


def load_farfield_motion_deltas(
        artifact_dir: Path,
        *,
        expected_dataset: str,
        paths: list[list[str]],
        positions_by_id: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, dict]:
    """Load one published farfield odometry realization for one forward path."""
    if len(paths) != 1 or paths[0] != list(positions_by_id):
        raise ValueError(
            "farfield odometry requires exactly one forward path in "
            "pano_id_mapping.csv order")
    reference = farfield_artifact.open_artifact(
        artifact_dir,
        expected_kind=farfield_paths.LOCALIZATION_INPUTS,
        expected_dataset=expected_dataset,
    )
    manifest = farfield_artifact.load_manifest(artifact_dir)
    config = manifest.config.get("localization_inputs")
    if not isinstance(config, dict):
        raise ValueError(
            "farfield manifest must record localization_inputs config")
    truth = farfield_run_io.read_jsonl(
        Path(artifact_dir) / "truth.jsonl", farfield_structs.TruthPose)
    odometry = farfield_run_io.read_jsonl(
        Path(artifact_dir) / "tier1_odometry.jsonl",
        farfield_structs.OdometryDelta,
    )
    path_latlons = authoritative_positions_for_path(positions_by_id, paths[0])
    motion_deltas = farfield_odometry_to_latlon_deltas(
        odometry,
        truth,
        path_latlons,
        reverse_keyframe_ranges=config.get("reverse_keyframe_ranges"),
        displacement_gate_m=config.get("displacement_gate_m"),
    )
    return motion_deltas, {
        "artifact": reference.to_dict(),
        "translation": "forward_left_rotated_by_truth_chord_body_heading",
        "delta_yaw": "ignored_known_heading",
    }


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
    evaluation_positions_by_id: dict[str, tuple[float, float]] | None = None,
    shared_motion_deltas: torch.Tensor | None = None,
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
        evaluation_positions_by_id: Optional authoritative truth coordinates.
            These affect scoring only; LOCI motion and odometry continue to use
            the panorama filename coordinates used by the released baseline.
        shared_motion_deltas: Optional exact realized controls from a farfield
            localization_inputs artifact. Only valid for one forward path.
    """
    if shared_motion_deltas is not None:
        if config.odometry_noise is not None:
            raise ValueError(
                "shared farfield odometry cannot be combined with LOCI "
                "odometry noise")
        if (len(paths) != 1
                or tuple(shared_motion_deltas.shape)
                != (len(paths[0]) - 1, 2)
                or not torch.is_floating_point(shared_motion_deltas)
                or not torch.isfinite(shared_motion_deltas).all()):
            raise ValueError(
                "shared farfield odometry must be a finite floating-point "
                "(path_len - 1, 2) tensor for exactly one path")
    all_final_error_meters = []        # using mean estimator
    all_final_mode_error_meters = []   # using argmax-cell (MAP / mode) estimator
    # Track convergence costs per radius
    convergence_costs_by_radius: dict[int, list[float]] = {}
    if convergence_radii:
        for radius in convergence_radii:
            convergence_costs_by_radius[radius] = []

    with torch.no_grad():
        # Build GridSpec from dataset bounds with buffer of half patch footprint
        min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
        footprint_px = config.footprint_px
        cell_size_px = footprint_px / config.subdivision_factor

        # Add buffer of half patch footprint (in pixels at zoom level)
        # Convert to degrees using web mercator at the center latitude
        patch_half_size_px = footprint_px / 2.0
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
        print(f"Grid size: {grid_spec.num_rows} x {grid_spec.num_cols} = {grid_spec.num_rows * grid_spec.num_cols} cells "
              f"(cell={cell_size_px:.0f}px, footprint={footprint_px:.0f}px)")

        # Get patch positions and build mapping
        patch_positions_px = get_patch_positions_px(vigor_dataset, device)
        mapping = build_cell_to_patch_mapping(
            grid_spec=grid_spec,
            patch_positions_px=patch_positions_px,
            patch_half_size_px=patch_half_size_px,
            device=device,
            max_chunk_bytes=int(config.max_chunk_gib * 1024**3),
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

            # Get motion deltas for this path.
            motion_deltas = (
                shared_motion_deltas.to(device)
                if shared_motion_deltas is not None
                else es.get_motion_deltas_from_path(vigor_dataset, path).to(device)
            )

            # Apply odometry noise if configured
            if config.odometry_noise is not None:
                start_latlon = vigor_dataset.get_panorama_positions(path)[0].to(device)
                noise_gen = torch.Generator(device='cpu').manual_seed(
                    config.odometry_noise.seed * (i + 1))
                motion_deltas = add_noise_to_motion_deltas(
                    motion_deltas.cpu(), start_latlon.cpu(), config.odometry_noise,
                    generator=noise_gen,
                ).to(device)

            evaluation_truth = (
                authoritative_positions_for_path(evaluation_positions_by_id, path)
                if evaluation_positions_by_id is not None
                else vigor_dataset.get_panorama_positions(path)
            )
            convergence_truth = (
                evaluation_truth.to(device) if convergence_radii else None)

            # Run filter
            result = run_histogram_filter_on_path(
                belief=belief,
                motion_deltas=motion_deltas,
                path_pano_ids=path,
                log_likelihood_aggregator=log_likelihood_aggregator,
                mapping=mapping,
                config=config,
                true_latlons=convergence_truth,
                convergence_radii=convergence_radii,
            )

            # Compute distance traveled
            distance_traveled_m = compute_distance_traveled_from_positions(
                evaluation_truth)

            # Compute error
            error_meters = get_distance_error_from_estimate_history(
                vigor_dataset, path, result.mean_history, evaluation_truth)
            mode_error_meters = get_distance_error_from_estimate_history(
                vigor_dataset, path, result.mode_history, evaluation_truth)

            # Variance in meters squared (convert from degrees)
            var_sq_m = result.variance_history[-len(path):].sum(dim=-1) * (web_mercator.METERS_PER_DEG_LAT ** 2)

            all_final_error_meters.append(error_meters[-1].item())
            all_final_mode_error_meters.append(mode_error_meters[-1].item())

            # Save results
            save_path = output_path / f"{i:07d}"
            save_path.mkdir(parents=True, exist_ok=True)

            torch.save(error_meters, save_path / "error.pt")
            torch.save(mode_error_meters, save_path / "mode_error.pt")
            torch.save(var_sq_m, save_path / "var.pt")
            torch.save(path, save_path / "path.pt")
            torch.save(distance_traveled_m, save_path / "distance_traveled_m.pt")
            torch.save(
                motion_deltas.detach().cpu(),
                save_path / APPLIED_MOTION_DELTAS_FILENAME,
            )

            if save_intermediate_states:
                torch.save(result.mean_history.cpu(), save_path / "mean_history.pt")
                torch.save(result.mode_history.cpu(), save_path / "mode_history.pt")
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
        average_final_mode_error = (
            sum(all_final_mode_error_meters) / len(all_final_mode_error_meters)
        )
        summary_stats = {
            "average_final_error": average_final_error,            # mean estimator
            "average_final_mode_error": average_final_mode_error,  # MAP / argmax-cell estimator
            "filter_type": "histogram",
            "grid_rows": grid_spec.num_rows,
            "grid_cols": grid_spec.num_cols,
            "cell_size_px": cell_size_px,
            "evaluation_truth_source": (
                "pano_id_mapping.csv:lat,lon:float64"
                if evaluation_positions_by_id is not None
                else "panorama_filename:lat,lon:float32"
            ),
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

        print(f"Average final error meters: {average_final_error:.2f} "
              f"(mean estimator)  vs  {average_final_mode_error:.2f} (MAP estimator)")
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
        satellite_dir: str | None = None,
        sat_model_path: str | None = None,
        pano_model_path: str | None = None,
        allow_legacy_path_identity: bool = False,
        require_path_identity: bool = False,
):
    """Load dataset and optionally models for evaluation.

    Args:
        dataset_path: Path to VIGOR dataset
        paths_path: Path to JSON file with evaluation paths
        panorama_neighbor_radius_deg: Panorama neighbor radius in degrees
        panorama_landmark_radius_px: Panorama landmark radius in pixels
        device: Torch device
        landmark_version: Landmark version string
        satellite_dir: Optional external satellite artifact payload directory
        sat_model_path: Optional path to satellite model (required if pano_model_path is set)
        pano_model_path: Optional path to panorama model (required if sat_model_path is set)
        allow_legacy_path_identity: Explicitly accept a path file without a
            cryptographic mapping identity
        require_path_identity: Fail unless the path file binds the live mapping

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
    dataset_path = Path(dataset_path).expanduser()
    validate_path_dataset_identity(
        paths_data,
        dataset_path,
        allow_legacy=allow_legacy_path_identity,
        require_identity=require_path_identity,
    )
    factor = paths_data.get('args', {}).get('factor', 1.0)
    print(f"Dataset Factor: {factor}")

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
        satellite_dir=(Path(satellite_dir).expanduser()
                       if satellite_dir is not None else None),
    )

    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    return vigor_dataset, sat_model, pano_model, paths_data


def validate_path_dataset_identity(
        paths_data: dict,
        dataset_path: Path,
        allow_legacy: bool = False,
        require_identity: bool = False) -> None:
    """Bind new path files to the exact live panorama trajectory mapping."""
    recorded = paths_data.get("dataset_hash")
    is_sha256 = (
        isinstance(recorded, str)
        and len(recorded) == 64
        and all(character in string.hexdigits for character in recorded)
    )
    if is_sha256:
        mapping_path = Path(dataset_path) / "pano_id_mapping.csv"
        expected = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
        if recorded.lower() != expected:
            raise ValueError(
                f"Path file dataset_hash does not match live mapping "
                f"{mapping_path}; the paths belong to a different dataset "
                "revision or leg.")
        return

    message = (
        "Path file has no cryptographic pano_id_mapping.csv identity. "
        "Regenerate it, or explicitly pass --allow-legacy-path-identity "
        "for an independently audited legacy artifact."
    )
    if require_identity and not allow_legacy:
        raise ValueError(message)
    if allow_legacy:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def read_satellite_source_px(
        dataset_path: Path, satellite_dir: Path | None, *,
        expected_zoom: int = 20,
        expected_patch_px: int = 640) -> int | float | None:
    """Read and validate the rendered patch footprint metadata."""
    external = satellite_dir is not None
    sat_bbox_path = (
        satellite_dir.resolve().parent
        if external
        else dataset_path
    ) / "satellite_bbox.json"
    if not sat_bbox_path.exists():
        if external:
            raise FileNotFoundError(
                "external satellite directory requires metadata at "
                f"{sat_bbox_path}")
        return None
    sat_bbox = json.loads(sat_bbox_path.read_text())
    if not isinstance(sat_bbox, dict):
        raise ValueError(f"{sat_bbox_path} must contain a JSON object")
    grid = sat_bbox.get("grid", {})
    if not isinstance(grid, dict):
        raise ValueError(f"{sat_bbox_path} grid must be a JSON object")

    def read_field(name: str):
        if name in sat_bbox and name in grid \
                and sat_bbox[name] != grid[name]:
            raise ValueError(
                f"{sat_bbox_path} has conflicting {name} metadata")
        return sat_bbox[name] if name in sat_bbox else grid.get(name)

    source_px = read_field("source_px")
    if (isinstance(source_px, bool)
            or not isinstance(source_px, (int, float))
            or not math.isfinite(source_px)
            or source_px <= 0):
        if source_px is not None or external:
            raise ValueError(
                f"{sat_bbox_path} source_px must be a positive number")
        return None

    for name, expected in (
            ("zoom", expected_zoom), ("patch_px", expected_patch_px)):
        actual = read_field(name)
        if actual is None and not external:
            continue
        if type(actual) is not int or actual != expected:
            raise ValueError(
                f"{sat_bbox_path} {name} must be {expected}, got {actual!r}")
    return source_px


def copy_aggregator_config(config_path: Path, output_path: Path) -> None:
    """Preserve the config unless it already lives at the destination."""
    destination = output_path / "aggregator_config.yaml"
    if config_path.resolve() != destination.resolve():
        shutil.copy(config_path, destination)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate histogram filter on paths")

    parser.add_argument("--aggregator-config", type=str, required=True,
                        help="Path to YAML config file for aggregator (see adaptive_aggregators.py)")

    parser.add_argument("--paths-path", type=str, required=True,
                        help="Path to json file full of evaluation paths")
    parser.add_argument(
        "--allow-legacy-path-identity", action="store_true",
        help="Explicitly use an audited legacy path file whose dataset_hash "
             "predates the cryptographic mapping identity",
    )
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save the evaluation results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--satellite-dir", type=str, default=None,
        help="External satellite artifact payload directory",
    )
    parser.add_argument("--landmark-version", type=str, required=True)
    parser.add_argument("--save-intermediate-filter-states", action='store_true',
                        help="Save intermediate filter states (mean/variance history)")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float,
                        default=0.0005, help="Panorama neighbor radius deg")
    parser.add_argument("--panorama-landmark-radius-px", type=int,
                        default=640, help="Panorama landmark radius in pixels")
    parser.add_argument("--motion-noise-frac", type=float, default=0.05,
                        help="Wiener noise intensity (m/√m) for the filter's "
                             "process-model blur. Per step, blur std = "
                             "motion_noise_frac × √step_distance_m. Same "
                             "units as --odometry-noise-frac so a calibrated "
                             "filter sets them equal.")
    parser.add_argument("--subdivision-factor", type=int, default=4,
                        help="Grid subdivision factor (4 = 160px cells)")
    parser.add_argument(
        "--convergence-radii", type=str,
        default=",".join(str(radius) for radius in DEFAULT_CONVERGENCE_RADII),
                        help="Comma-separated list of radii (meters) for convergence metrics")
    parser.add_argument("--max-chunk-gib", type=float, default=2.0,
                        help="Peak GPU memory per chunk in GiB for cell-to-patch mapping (default: 2)")

    # Odometry noise arguments
    parser.add_argument("--odometry-noise-frac", type=float, default=None,
                        help="Noise std as fraction of step distance (isotropic north/east)")
    parser.add_argument("--odometry-noise-seed", type=int, default=7919,
                        help="Seed for odometry noise generation")
    parser.add_argument(
        "--farfield-localization-inputs", type=str, default=None,
        help="Published farfield localization_inputs artifact whose exact "
             "realized translation odometry should drive one forward path",
    )

    args = parser.parse_args()
    if (args.farfield_localization_inputs is not None
            and args.odometry_noise_frac is not None):
        parser.error(
            "--farfield-localization-inputs cannot be combined with "
            "--odometry-noise-frac")

    # Parse convergence radii
    convergence_radii = [int(r.strip()) for r in args.convergence_radii.split(",")]

    DEVICE = "cuda:0"
    torch.set_deterministic_debug_mode('error')

    output_path = Path(args.output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    args.output_path = output_path

    # Copy aggregator config to output directory for reproducibility
    copy_aggregator_config(Path(args.aggregator_config), args.output_path)

    vigor_dataset, sat_model, pano_model, paths_data = construct_path_eval_inputs_from_args(
        dataset_path=args.dataset_path,
        paths_path=args.paths_path,
        panorama_neighbor_radius_deg=args.panorama_neighbor_radius_deg,
        panorama_landmark_radius_px=args.panorama_landmark_radius_px,
        device=DEVICE,
        landmark_version=args.landmark_version,
        satellite_dir=args.satellite_dir,
        allow_legacy_path_identity=args.allow_legacy_path_identity,
        require_path_identity=True,
    )

    # Load aggregator config and create aggregator
    aggregator_config = load_aggregator_config(Path(args.aggregator_config))
    print(f"Loaded aggregator config: {type(aggregator_config).__name__}")
    log_likelihood_aggregator = aggregator_from_config(
        aggregator_config,
        vigor_dataset,
        DEVICE,
        require_similarity_identity=True,
    )
    mapping_path = Path(args.dataset_path).expanduser() / "pano_id_mapping.csv"
    evaluation_positions_by_id = None
    if mapping_path.exists():
        evaluation_positions_by_id = load_authoritative_panorama_positions(
            mapping_path)
    else:
        warnings.warn(
            f"{mapping_path} is absent; evaluation truth falls back to rounded "
            "panorama filename coordinates",
            RuntimeWarning,
        )
    shared_motion_deltas = None
    farfield_odometry_source = None
    if args.farfield_localization_inputs is not None:
        if evaluation_positions_by_id is None:
            raise ValueError(
                "shared farfield odometry requires pano_id_mapping.csv")
        shared_motion_deltas, farfield_odometry_source = \
            load_farfield_motion_deltas(
                Path(args.farfield_localization_inputs).expanduser(),
                expected_dataset=Path(args.dataset_path).expanduser().name,
                paths=paths_data["paths"],
                positions_by_id=evaluation_positions_by_id,
            )
        print(
            "Using realized translation odometry from "
            f"{farfield_odometry_source['artifact']['version']}; "
            "delta_yaw is ignored because LOCI assumes known heading")

    with open(output_path / "args.json", "w") as f:
        args_record = {**vars(args), "output_path": str(output_path)}
        if farfield_odometry_source is not None:
            args_record["farfield_odometry_source"] = farfield_odometry_source
        json.dump(args_record, f, indent=4)

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

    # Read source_px from satellite_bbox.json if available
    source_px = read_satellite_source_px(
        Path(args.dataset_path).expanduser(),
        (Path(args.satellite_dir).expanduser()
         if args.satellite_dir is not None else None),
        expected_zoom=HistogramFilterConfig.zoom_level,
        expected_patch_px=HistogramFilterConfig.patch_size_px,
    )
    if source_px is not None and source_px != 640:
        print(f"Resolution-normalized dataset: source_px={source_px} (footprint {source_px}px, image 640px)")

    config = HistogramFilterConfig(
        motion_noise_frac=args.motion_noise_frac,
        subdivision_factor=args.subdivision_factor,
        initial_std_deg=degrees_from_meters(2970.0),
        initial_offset_std_deg=degrees_from_meters(1300.0),
        source_px=source_px,
        odometry_noise=odometry_noise_config,
        max_chunk_gib=args.max_chunk_gib,
    )

    histogram_config_dict = {
        "motion_noise_frac": config.motion_noise_frac,
        "subdivision_factor": config.subdivision_factor,
        "initial_std_deg": config.initial_std_deg,
        "initial_offset_std_deg": config.initial_offset_std_deg,
        "zoom_level": config.zoom_level,
        "patch_size_px": config.patch_size_px,
        "source_px": config.source_px,
    }
    if odometry_noise_config is not None:
        histogram_config_dict["odometry_noise"] = {
            "sigma_noise_frac": odometry_noise_config.sigma_noise_frac,
            "seed": odometry_noise_config.seed,
        }
    if farfield_odometry_source is not None:
        histogram_config_dict[
            "farfield_odometry_source"] = farfield_odometry_source

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
        evaluation_positions_by_id=evaluation_positions_by_id,
        shared_motion_deltas=shared_motion_deltas,
    )
