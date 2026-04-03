"""Web-based step-through visualization for histogram filter."""

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import json
import argparse
import io
import base64
import sys
from PIL import Image as PILImage

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    ObservationLogLikelihoodAggregator,
    load_aggregator_config,
    aggregator_from_config,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from common.gps import web_mercator


def tensor_to_base64(img_tensor: torch.Tensor) -> str:
    """Convert a (C, H, W) float32 tensor to a base64 JPEG data URI."""
    arr = (img_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = PILImage.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def get_dataset_bounds(vigor_dataset: vd.VigorDataset):
    sat_meta = vigor_dataset._satellite_metadata
    min_lat = sat_meta['lat'].min()
    max_lat = sat_meta['lat'].max()
    min_lon = sat_meta['lon'].min()
    max_lon = sat_meta['lon'].max()
    return min_lat, max_lat, min_lon, max_lon


def get_patch_positions_px(vigor_dataset: vd.VigorDataset, device: torch.device):
    patch_positions_px = torch.tensor(
        vigor_dataset._satellite_metadata[["web_mercator_y", "web_mercator_x"]].values,
        device=device, dtype=torch.float32)
    return patch_positions_px


def run_filter_with_history(
    grid_spec: GridSpec,
    mapping,
    initial_belief: HistogramBelief,
    motion_deltas: torch.Tensor,
    path_pano_ids: list[str],
    log_likelihood_aggregator: ObservationLogLikelihoodAggregator,
    noise_percent: float,
):
    """Run filter and save belief at each step."""
    belief = initial_belief.clone()

    history = []
    history.append({
        'stage': 'init',
        'log_belief': belief.get_log_belief().clone().cpu(),
        'mean': belief.get_mean_latlon().clone().cpu(),
        'gt_idx': 0,
        'pano_id': path_pano_ids[0],
        'obs_log_ll': None,
    })

    path_len = len(path_pano_ids)

    filter_device = belief.get_log_belief().device

    for step_idx in range(path_len - 1):
        # Observation update - get log-likelihoods from aggregator
        obs_log_ll = log_likelihood_aggregator(path_pano_ids[step_idx]).to(filter_device)
        belief.apply_observation(obs_log_ll, mapping)
        history.append({
            'stage': f'obs_{step_idx}',
            'log_belief': belief.get_log_belief().clone().cpu(),
            'mean': belief.get_mean_latlon().clone().cpu(),
            'gt_idx': step_idx,
            'pano_id': path_pano_ids[step_idx],
            'obs_log_ll': obs_log_ll.clone().cpu(),
        })

        # Motion prediction
        belief.apply_motion(motion_deltas[step_idx], noise_percent)
        history.append({
            'stage': f'motion_{step_idx}',
            'log_belief': belief.get_log_belief().clone().cpu(),
            'mean': belief.get_mean_latlon().clone().cpu(),
            'gt_idx': step_idx + 1,
            'pano_id': path_pano_ids[step_idx + 1],
            'obs_log_ll': None,
        })

    # Final observation - get log-likelihoods from aggregator
    obs_log_ll = log_likelihood_aggregator(path_pano_ids[-1]).to(filter_device)
    belief.apply_observation(obs_log_ll, mapping)
    history.append({
        'stage': f'obs_{path_len-1}',
        'log_belief': belief.get_log_belief().clone().cpu(),
        'mean': belief.get_mean_latlon().clone().cpu(),
        'gt_idx': path_len - 1,
        'pano_id': path_pano_ids[-1],
        'obs_log_ll': obs_log_ll.clone().cpu(),
    })

    return history


CHECKPOINT_INTERVAL = 50  # Save full belief every N steps


def run_filter_lightweight(
    grid_spec: GridSpec,
    mapping,
    initial_belief: HistogramBelief,
    motion_deltas: torch.Tensor,
    path_pano_ids: list[str],
    log_likelihood_aggregator: ObservationLogLikelihoodAggregator,
    noise_percent: float,
):
    """Run filter saving summaries per step and full belief at checkpoints.

    Full belief is saved every CHECKPOINT_INTERVAL steps. For other steps,
    use replay_filter_to_step() which replays from the nearest checkpoint.
    """
    belief = initial_belief.clone()

    history = []
    checkpoints = {}  # step_idx -> log_belief tensor (cpu)

    def save_step(stage, gt_idx, pano_id, obs_log_ll, step_num):
        entry = {
            'stage': stage,
            'mean': belief.get_mean_latlon().clone().cpu(),
            'gt_idx': gt_idx,
            'pano_id': pano_id,
        }
        history.append(entry)
        if step_num % CHECKPOINT_INTERVAL == 0:
            checkpoints[step_num] = belief.get_log_belief().clone().cpu()

    step_num = 0
    save_step('init', 0, path_pano_ids[0], None, step_num)

    path_len = len(path_pano_ids)
    filter_device = belief.get_log_belief().device

    for step_idx in range(path_len - 1):
        obs_log_ll = log_likelihood_aggregator(path_pano_ids[step_idx]).to(filter_device)
        belief.apply_observation(obs_log_ll, mapping)
        step_num += 1
        save_step(f'obs_{step_idx}', step_idx, path_pano_ids[step_idx], obs_log_ll, step_num)

        belief.apply_motion(motion_deltas[step_idx], noise_percent)
        step_num += 1
        save_step(f'motion_{step_idx}', step_idx + 1, path_pano_ids[step_idx + 1], None, step_num)

    obs_log_ll = log_likelihood_aggregator(path_pano_ids[-1]).to(filter_device)
    belief.apply_observation(obs_log_ll, mapping)
    step_num += 1
    save_step(f'obs_{path_len-1}', path_len - 1, path_pano_ids[-1], obs_log_ll, step_num)

    # Always checkpoint the last step
    checkpoints[step_num] = belief.get_log_belief().clone().cpu()

    return history, checkpoints


def replay_filter_to_step(
    grid_spec: GridSpec,
    mapping,
    filter_device: torch.device,
    motion_deltas: torch.Tensor,
    path_pano_ids: list[str],
    log_likelihood_aggregator: ObservationLogLikelihoodAggregator,
    noise_percent: float,
    target_step: int,
    checkpoints: dict[int, torch.Tensor],
):
    """Replay from the nearest checkpoint to target_step.

    Steps follow the same ordering as run_filter_lightweight:
      0: init
      1: obs_0, 2: motion_0, 3: obs_1, 4: motion_1, ...
    """
    # Find nearest checkpoint at or before target_step
    valid_checkpoints = [s for s in checkpoints if s <= target_step]
    if valid_checkpoints:
        start_step = max(valid_checkpoints)
        belief = HistogramBelief.from_uniform(grid_spec, filter_device)
        belief._log_belief = checkpoints[start_step].to(filter_device)
    else:
        start_step = 0
        belief = HistogramBelief.from_uniform(grid_spec, filter_device)

    if start_step == target_step:
        # Determine obs_log_ll: obs steps are odd (1, 3, 5, ...) -> step_idx = (step-1)//2
        obs_log_ll = None
        if target_step > 0 and target_step % 2 == 1:
            pano_idx = (target_step - 1) // 2
            obs_log_ll = log_likelihood_aggregator(path_pano_ids[pano_idx]).to(filter_device).clone().cpu()
        return belief.get_log_belief().clone().cpu(), obs_log_ll

    current_step = start_step
    obs_log_ll = None
    path_len = len(path_pano_ids)

    # Determine which step_idx in the loop corresponds to current_step
    # step 0=init, then pairs: (2*i+1=obs_i, 2*i+2=motion_i) for i in 0..path_len-2
    # final: 2*(path_len-1)+1 = obs_{path_len-1}
    for step_idx in range(path_len - 1):
        obs_step = 2 * step_idx + 1
        motion_step = 2 * step_idx + 2

        if obs_step > target_step:
            break
        if obs_step > current_step:
            obs_log_ll = log_likelihood_aggregator(path_pano_ids[step_idx]).to(filter_device)
            belief.apply_observation(obs_log_ll, mapping)
            current_step = obs_step
            if current_step == target_step:
                return belief.get_log_belief().clone().cpu(), obs_log_ll.clone().cpu()

        if motion_step > target_step:
            break
        if motion_step > current_step:
            belief.apply_motion(motion_deltas[step_idx], noise_percent)
            current_step = motion_step
            obs_log_ll = None
            if current_step == target_step:
                return belief.get_log_belief().clone().cpu(), None

    # Final observation
    final_obs_step = 2 * (path_len - 1) + 1
    if final_obs_step > current_step and final_obs_step <= target_step:
        obs_log_ll = log_likelihood_aggregator(path_pano_ids[-1]).to(filter_device)
        belief.apply_observation(obs_log_ll, mapping)
        current_step = final_obs_step
        if current_step == target_step:
            return belief.get_log_belief().clone().cpu(), obs_log_ll.clone().cpu()

    raise ValueError(f"target_step {target_step} not reached (got to {current_step})")


def create_belief_heatmap(log_belief, grid_spec, max_heatmap_dim: int | None = None):
    """Convert log belief to plotly heatmap data, optionally downsampled for rendering."""
    # Normalize for visualization
    belief = torch.exp(log_belief - log_belief.max())

    nr, nc = belief.shape
    row_stride = 1
    col_stride = 1

    if max_heatmap_dim is not None:
        row_stride = max(1, nr // max_heatmap_dim)
        col_stride = max(1, nc // max_heatmap_dim)
        if row_stride > 1 or col_stride > 1:
            nr_trim = (nr // row_stride) * row_stride
            nc_trim = (nc // col_stride) * col_stride
            belief_trimmed = belief[:nr_trim, :nc_trim]
            belief = belief_trimmed.reshape(nr_trim // row_stride, row_stride,
                                            nc_trim // col_stride, col_stride).amax(dim=(1, 3))

    belief_np = belief.numpy()

    row_indices = torch.arange(belief_np.shape[0]) * row_stride + row_stride / 2
    col_indices = torch.arange(belief_np.shape[1]) * col_stride + col_stride / 2

    lat_coords = []
    for row in row_indices:
        lat, _ = grid_spec.cell_indices_to_latlon(row, torch.tensor(0.0))
        lat_coords.append(lat.item())

    lon_coords = []
    for col in col_indices:
        _, lon = grid_spec.cell_indices_to_latlon(torch.tensor(0.0), col)
        lon_coords.append(lon.item())

    return belief_np, lat_coords, lon_coords


def load_path_statistics(eval_path: Path) -> list[dict]:
    """Load statistics for all evaluated paths."""
    stats = []

    # Find all path directories
    path_dirs = sorted([d for d in eval_path.iterdir() if d.is_dir() and d.name.isdigit()])

    for path_dir in path_dirs:
        path_idx = int(path_dir.name)
        try:
            path = torch.load(path_dir / "path.pt", map_location='cpu')
            # Check for old format
            if path and isinstance(path[0], int):
                raise ValueError(
                    f"path.pt in '{path_dir}' uses old index format (integers). "
                    "Re-run evaluation with new path files to get pano_id format (strings)."
                )
            path_len = len(path) if isinstance(path, (list, torch.Tensor)) else 1

            error = None
            final_error = None
            if (path_dir / "error.pt").exists():
                error = torch.load(path_dir / "error.pt", map_location='cpu')
                if hasattr(error, 'cpu'):
                    error = error.cpu()
                final_error = float(error[-1]) if len(error) > 0 else None

            distance = None
            if (path_dir / "distance_traveled_m.pt").exists():
                dist = torch.load(path_dir / "distance_traveled_m.pt", map_location='cpu')
                if hasattr(dist, 'cpu'):
                    dist = dist.cpu()
                distance = float(dist[-1]) if len(dist) > 0 else None

            stats.append({
                'path_idx': path_idx,
                'path_len': path_len,
                'final_error_m': final_error,
                'distance_m': distance,
            })
        except Exception as e:
            print(f"Failed to load path {path_idx}: {e}")
            continue

    return stats


def main():
    parser = argparse.ArgumentParser(description="Web-based histogram filter visualization")
    parser.add_argument("--eval-path", type=str, required=True,
                        help="Path to evaluation results directory")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VIGOR dataset")
    parser.add_argument("--aggregator-config", type=str, required=True,
                        help="Path to YAML config file for aggregator (see adaptive_aggregators.py)")
    parser.add_argument("--noise-percent", type=float, default=None)
    parser.add_argument("--simple", action="store_true",
                        help="Disable new features (pano/patch images, likelihood coloring)")
    parser.add_argument("--lightweight", action="store_true",
                        help="Don't store full belief history in RAM. Recomputes belief on demand per step. "
                             "Use for large grids (e.g., Norway) that would otherwise OOM.")
    parser.add_argument("--downsample-heatmap", type=int, default=None,
                        help="Max rows/cols for belief heatmap. If not set, no heatmap downsampling.")
    parser.add_argument("--downsample-scatter", type=int, default=None,
                        help="Max scatter points for obs likelihood overlay. If not set, no scatter downsampling.")
    parser.add_argument("--max-chunk-gib", type=float, default=None,
                        help="Max GiB per chunk for cell-to-patch mapping. Defaults to eval_args value or 2.0.")
    parser.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    eval_path = Path(args.eval_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()

    # Load config
    with open(eval_path / "args.json") as f:
        eval_args = json.load(f)

    simple_mode = args.simple
    lightweight_mode = args.lightweight
    noise_percent = args.noise_percent if args.noise_percent is not None else eval_args.get("noise_percent", 0.02)
    subdivision_factor = eval_args.get("subdivision_factor", 4)

    print(f"Config: noise_percent={noise_percent}, subdivision={subdivision_factor}")

    # Load path statistics
    print("Loading path statistics...")
    path_stats = load_path_statistics(eval_path)
    print(f"Found {len(path_stats)} paths")

    # Compute summary statistics
    final_errors = [s['final_error_m'] for s in path_stats if s['final_error_m'] is not None]
    if final_errors:
        avg_error = np.mean(final_errors)
        median_error = np.median(final_errors)
        min_error = np.min(final_errors)
        max_error = np.max(final_errors)
        print(f"Error stats: avg={avg_error:.1f}m, median={median_error:.1f}m, min={min_error:.1f}m, max={max_error:.1f}m")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    import time as _time
    _t0 = _time.time()
    print("Loading dataset...")
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=None,
        satellite_tensor_cache_info=None,
        panorama_neighbor_radius=0.0005,
        satellite_patch_size=(640, 640),
        panorama_size=(640, 640),
        factor=1.0,
        landmark_version=eval_args.get("landmark_version", "v4_202001"),
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)
    print(f"Dataset loaded in {_time.time() - _t0:.1f}s")

    # Load aggregator config and create aggregator
    _t0 = _time.time()
    print("Loading aggregator config...")
    aggregator_config = load_aggregator_config(Path(args.aggregator_config))
    print(f"Loaded aggregator config: {type(aggregator_config).__name__}")
    log_likelihood_aggregator = aggregator_from_config(
        aggregator_config,
        vigor_dataset,
        device,
    )
    print(f"Aggregator loaded in {_time.time() - _t0:.1f}s")

    # Use CPU for filter computations (visualization doesn't need GPU)
    filter_device = torch.device('cpu')

    # Build grid with buffer of half patch size (320 pixels)
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    cell_size_px = 640.0 / subdivision_factor

    # Add buffer of half patch size (320 pixels at zoom 20)
    center_lat = (min_lat + max_lat) / 2
    ref_y, ref_x = web_mercator.latlon_to_pixel_coords(center_lat, min_lon, 20)
    buf_lat, _ = web_mercator.pixel_coords_to_latlon(ref_y - 320, ref_x, 20)
    _, buf_lon = web_mercator.pixel_coords_to_latlon(ref_y, ref_x + 320, 20)
    lat_buffer = buf_lat - center_lat
    lon_buffer = buf_lon - min_lon

    grid_spec = GridSpec.from_bounds_and_cell_size(
        min_lat=min_lat - lat_buffer, max_lat=max_lat + lat_buffer,
        min_lon=min_lon - lon_buffer, max_lon=max_lon + lon_buffer,
        zoom_level=20, cell_size_px=cell_size_px,
    )
    print(f"Grid: {grid_spec.num_rows} x {grid_spec.num_cols}")

    # Build mapping on GPU (much faster), then move to CPU for filter operations
    patch_positions_px = get_patch_positions_px(vigor_dataset, device)
    _t0 = _time.time()
    max_chunk_gib = args.max_chunk_gib if args.max_chunk_gib is not None else eval_args.get("max_chunk_gib", 2.0)
    mapping = build_cell_to_patch_mapping(grid_spec, patch_positions_px, 320.0, device,
                                          max_chunk_bytes=int(max_chunk_gib * 1024**3))
    mapping = mapping.to(filter_device)
    print(f"Cell-to-patch mapping built in {_time.time() - _t0:.1f}s")

    sat_positions = vigor_dataset._satellite_metadata[['lat', 'lon']].values

    # Prepare table data (keep as numbers for proper sorting)
    table_data = []
    for s in path_stats:
        table_data.append({
            'Path': s['path_idx'],
            'Length': s['path_len'],
            'Final Error (m)': round(s['final_error_m'], 1) if s['final_error_m'] is not None else None,
            'Distance (m)': round(s['distance_m'], 0) if s['distance_m'] is not None else None,
        })

    # Server-side cache for filter history (avoid sending huge data to browser)
    cache = {
        'current_path_idx': None,
        'history': None,
        'checkpoints': {},
        'gt_positions': None,
        'motion_deltas': None,
        'path': None,
    }

    # Create Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Histogram Filter Visualization"),

        # Summary statistics
        html.Div([
            html.H3("Summary Statistics"),
            html.Div([
                html.Span(f"Paths: {len(path_stats)}", style={'marginRight': '30px'}),
                html.Span(f"Avg Error: {avg_error:.1f}m" if final_errors else "N/A", style={'marginRight': '30px'}),
                html.Span(f"Median Error: {median_error:.1f}m" if final_errors else "N/A", style={'marginRight': '30px'}),
                html.Span(f"Min: {min_error:.1f}m" if final_errors else "N/A", style={'marginRight': '30px'}),
                html.Span(f"Max: {max_error:.1f}m" if final_errors else "N/A"),
            ], style={'marginBottom': '20px'}),
        ]),

        # Path selection table
        html.Div([
            html.H3("Select Path (click a row)"),
            dash_table.DataTable(
                id='path-table',
                columns=[
                    {'name': 'Path', 'id': 'Path', 'type': 'numeric'},
                    {'name': 'Length', 'id': 'Length', 'type': 'numeric'},
                    {'name': 'Final Error (m)', 'id': 'Final Error (m)', 'type': 'numeric'},
                    {'name': 'Distance (m)', 'id': 'Distance (m)', 'type': 'numeric'},
                ],
                data=table_data,
                row_selectable='single',
                selected_rows=[0] if table_data else [],
                style_table={'height': '200px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'fontWeight': 'bold'},
                sort_action='native',
                filter_action='native',
            ),
        ], style={'marginBottom': '20px'}),

        # Step controls
        html.Div([
            html.Div([
                html.Label("Step:"),
                dcc.Slider(
                    id='step-slider',
                    min=0,
                    max=1,
                    value=0,
                    marks={},
                    step=1,
                ),
            ], style={'width': '80%', 'margin': '20px auto'}),
            html.Div([
                html.Button('Previous', id='prev-btn', n_clicks=0),
                html.Button('Next', id='next-btn', n_clicks=0),
                dcc.Checklist(
                    id='skip-motion-check',
                    options=[{'label': ' Skip motion steps', 'value': 'skip'}],
                    value=['skip'],
                    style={'display': 'inline-block', 'marginLeft': '20px'},
                ),
                html.Span(id='step-info', style={'marginLeft': '20px'}),
            ], style={'textAlign': 'center', 'margin': '10px'}),
        ], id='step-controls'),

        # Main plot
        dcc.Graph(id='main-plot', style={'height': '70vh'}),

        # Panorama and top satellite patches (hidden in simple mode)
        html.Div([
            # Left: panorama image
            html.Div([
                html.H4(id='pano-title', children='Panorama'),
                html.Img(id='pano-image', style={'width': '100%', 'maxHeight': '300px', 'objectFit': 'contain'}),
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

            # Right: top satellite patches
            html.Div([
                html.H4(id='top-patches-title', children='Top Satellite Patches by Observation Likelihood'),
                html.Div(id='top-patches-container', style={
                    'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px',
                }),
            ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        ], style={'marginTop': '20px', 'display': 'none' if simple_mode else 'block'}),

        # Minimal client-side state (path index, step count, and obs step indices)
        dcc.Store(id='path-metadata-store'),
    ])

    # Arrow key navigation via clientside callback
    app.clientside_callback(
        """
        function(id) {
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowLeft') {
                    document.getElementById('prev-btn').click();
                } else if (e.key === 'ArrowRight') {
                    document.getElementById('next-btn').click();
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('step-slider', 'id'),  # dummy output
        Input('step-slider', 'id'),   # triggers once on load
    )

    @app.callback(
        [Output('path-metadata-store', 'data'),
         Output('step-slider', 'max'),
         Output('step-slider', 'marks'),
         Output('step-slider', 'value')],
        [Input('path-table', 'selected_rows')],
        [State('path-table', 'data')]
    )
    def load_path(selected_rows, table_data):
        if not selected_rows or not table_data:
            return None, 1, {}, 0

        row = table_data[selected_rows[0]]
        path_idx = row['Path']

        print(f"Loading path {path_idx}...")

        # Load path
        path_dir = eval_path / f"{path_idx:07d}"
        path = torch.load(path_dir / "path.pt", map_location='cpu')
        # Check for old format
        if path and isinstance(path[0], int):
            raise ValueError(
                f"path.pt in '{path_dir}' uses old index format (integers). "
                "Re-run evaluation with new path files to get pano_id format (strings)."
            )

        # Get ground truth positions
        gt_positions = vigor_dataset.get_panorama_positions(path).numpy()

        # Get motion deltas
        motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path)

        # Initialize and run filter
        if lightweight_mode:
            history, checkpoints = run_filter_lightweight(
                grid_spec, mapping,
                HistogramBelief.from_uniform(grid_spec, filter_device),
                motion_deltas, path,
                log_likelihood_aggregator, noise_percent
            )
        else:
            history = run_filter_with_history(
                grid_spec, mapping,
                HistogramBelief.from_uniform(grid_spec, filter_device),
                motion_deltas, path,
                log_likelihood_aggregator, noise_percent
            )
            checkpoints = {}

        # Store in server-side cache (not sent to browser)
        cache['current_path_idx'] = path_idx
        cache['history'] = history
        cache['checkpoints'] = checkpoints
        cache['gt_positions'] = gt_positions
        cache['motion_deltas'] = motion_deltas
        cache['path'] = path

        num_steps = len(history)
        marks = {i: str(i) for i in range(0, num_steps, max(1, num_steps // 20))}

        print(f"Loaded path {path_idx} with {num_steps} steps")

        # Compute obs step indices (init + observation steps, not motion steps)
        obs_step_indices = []
        for i, h in enumerate(history):
            if h['stage'] == 'init' or h['stage'].startswith('obs_'):
                obs_step_indices.append(i)

        # Only send minimal metadata to browser
        return {
            'path_idx': path_idx,
            'num_steps': num_steps,
            'obs_step_indices': obs_step_indices,
        }, num_steps - 1, marks, 0

    @app.callback(
        Output('step-slider', 'value', allow_duplicate=True),
        [Input('prev-btn', 'n_clicks'),
         Input('next-btn', 'n_clicks')],
        [State('step-slider', 'value'),
         State('step-slider', 'max'),
         State('skip-motion-check', 'value'),
         State('path-metadata-store', 'data')],
        prevent_initial_call=True
    )
    def update_slider(prev_clicks, next_clicks, current_value, max_value,
                      skip_motion_value, metadata):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_value

        skip_motion = 'skip' in (skip_motion_value or [])
        obs_steps = None
        if skip_motion and metadata and 'obs_step_indices' in metadata:
            obs_steps = metadata['obs_step_indices']

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'prev-btn' and current_value > 0:
            if obs_steps:
                # Find the largest obs step index strictly less than current_value
                prev_obs = [s for s in obs_steps if s < current_value]
                return prev_obs[-1] if prev_obs else current_value
            return current_value - 1
        elif button_id == 'next-btn' and current_value < max_value:
            if obs_steps:
                # Find the smallest obs step index strictly greater than current_value
                next_obs = [s for s in obs_steps if s > current_value]
                return next_obs[0] if next_obs else current_value
            return current_value + 1
        return current_value

    @app.callback(
        [Output('main-plot', 'figure'),
         Output('step-info', 'children'),
         Output('pano-image', 'src'),
         Output('pano-title', 'children'),
         Output('top-patches-container', 'children'),
         Output('top-patches-title', 'children')],
        [Input('step-slider', 'value'),
         Input('path-metadata-store', 'data')]
    )
    def update_plot(step_idx, metadata):
        import time as _time
        _t_start = _time.time()

        empty_outputs = ("", "Panorama", [], "Top Satellite Patches by Observation Likelihood")
        if not metadata or cache['history'] is None:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="Select a path from the table above",
                height=600,
            )
            return fig, "No path selected", *empty_outputs

        # Get data from server-side cache
        path_idx = cache['current_path_idx']
        gt_positions = cache['gt_positions']
        step = cache['history'][step_idx]

        stage = step['stage']
        mean = step['mean'].numpy()
        gt_idx = step['gt_idx']
        pano_id = step['pano_id']

        if 'log_belief' in step:
            # Full history mode
            log_belief = step['log_belief']
            obs_log_ll = step['obs_log_ll']
        else:
            # Lightweight mode — replay from nearest checkpoint
            _t0 = _time.time()
            log_belief, obs_log_ll = replay_filter_to_step(
                grid_spec, mapping, filter_device,
                cache['motion_deltas'], cache['path'],
                log_likelihood_aggregator, noise_percent,
                step_idx, cache['checkpoints'],
            )
            print(f"  Replay: {_time.time() - _t0:.2f}s")

        # Create heatmap data
        _t0 = _time.time()
        belief_data, lat_coords, lon_coords = create_belief_heatmap(log_belief, grid_spec, args.downsample_heatmap)
        _t_belief = _time.time() - _t0

        # Compute zoom center and extent (used for zoomed panels)
        center_lat = (gt_positions[gt_idx, 0] + mean[0]) / 2
        center_lon = (gt_positions[gt_idx, 1] + mean[1]) / 2
        zoom_size = 0.015

        def crop_heatmap(z, lats, lons, clat, clon, size):
            """Crop heatmap data to zoom window, returns (z_crop, lats_crop, lons_crop)."""
            lats_arr = np.array(lats)
            lons_arr = np.array(lons)
            lat_mask = (lats_arr >= clat - size) & (lats_arr <= clat + size)
            lon_mask = (lons_arr >= clon - size) & (lons_arr <= clon + size)
            if not lat_mask.any() or not lon_mask.any():
                return z, lats, lons
            z_crop = z[np.ix_(lat_mask, lon_mask)] if isinstance(z, np.ndarray) else z
            return z_crop, lats_arr[lat_mask].tolist(), lons_arr[lon_mask].tolist()

        # Prepare obs likelihood heatmap data
        _t0 = _time.time()
        has_obs = not simple_mode and obs_log_ll is not None
        obs_heatmap_data = None
        obs_lat_coords = None
        obs_lon_coords = None
        if has_obs:
            ll_np = obs_log_ll.numpy()
            # Satellites are on a regular grid — reshape directly
            lats = sat_positions[:, 0]
            lons = sat_positions[:, 1]
            unique_lats = np.sort(np.unique(lats))
            unique_lons = np.sort(np.unique(lons))
            n_rows, n_cols = len(unique_lats), len(unique_lons)
            if n_rows * n_cols == len(ll_np):
                # Perfect grid — sort by (lat, lon) via lexsort to get row-major order
                sorted_idx = np.lexsort((lons, lats))
                obs_heatmap_data = ll_np[sorted_idx].reshape(n_rows, n_cols)
                obs_lat_coords = unique_lats.tolist()
                obs_lon_coords = unique_lons.tolist()
            else:
                # Irregular grid — fall back to binning
                grid_dim = max(n_rows, n_cols)
                lat_bins = np.linspace(lats.min(), lats.max(), grid_dim + 1)
                lon_bins = np.linspace(lons.min(), lons.max(), grid_dim + 1)
                lat_bin_idx = np.clip(np.digitize(lats, lat_bins) - 1, 0, grid_dim - 1)
                lon_bin_idx = np.clip(np.digitize(lons, lon_bins) - 1, 0, grid_dim - 1)
                obs_grid = np.full((grid_dim, grid_dim), np.nan)
                for i in range(len(ll_np)):
                    r, c = lat_bin_idx[i], lon_bin_idx[i]
                    if np.isnan(obs_grid[r, c]) or ll_np[i] > obs_grid[r, c]:
                        obs_grid[r, c] = ll_np[i]
                obs_heatmap_data = obs_grid
                obs_lat_coords = ((lat_bins[:-1] + lat_bins[1:]) / 2).tolist()
                obs_lon_coords = ((lon_bins[:-1] + lon_bins[1:]) / 2).tolist()

        # Create 2x2 subplots
        _t_obs = _time.time() - _t0

        _t0 = _time.time()
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Obs Likelihood (full)', 'Obs Likelihood (zoomed)',
                f'Belief - Path {path_idx} (full)', 'Belief (zoomed)',
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
        )

        # --- Top-left (1,1): Obs likelihood heatmap full view ---
        if has_obs:
            fig.add_trace(
                go.Heatmap(
                    z=obs_heatmap_data,
                    x=obs_lon_coords,
                    y=obs_lat_coords,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Log LL', x=0.45, len=0.45, y=0.78),
                    opacity=0.8,
                    hovertemplate='lat=%{y:.4f}<br>lon=%{x:.4f}<br>log LL=%{z:.2f}<extra></extra>',
                ),
                row=1, col=1
            )
        else:
            fig.add_annotation(
                text="No observation at this step",
                xref="x1", yref="y1",
                x=0.5, y=0.5, xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=14, color="gray"),
            )

        # --- Top-right (1,2): Obs likelihood heatmap zoomed ---
        if has_obs:
            obs_z_crop, obs_lat_crop, obs_lon_crop = crop_heatmap(
                obs_heatmap_data, obs_lat_coords, obs_lon_coords,
                center_lat, center_lon, zoom_size)
            fig.add_trace(
                go.Heatmap(
                    z=obs_z_crop,
                    x=obs_lon_crop,
                    y=obs_lat_crop,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.8,
                    hovertemplate='lat=%{y:.4f}<br>lon=%{x:.4f}<br>log LL=%{z:.2f}<extra></extra>',
                ),
                row=1, col=2
            )

        # Add GT path + markers to top row
        for col in [1, 2]:
            if has_obs or col == 1:
                fig.add_trace(
                    go.Scatter(
                        x=gt_positions[:gt_idx+1, 1], y=gt_positions[:gt_idx+1, 0],
                        mode='lines+markers', line=dict(color='green', width=2),
                        marker=dict(size=5), showlegend=(col == 1), name='GT path' if col == 1 else None,
                    ), row=1, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=[gt_positions[gt_idx, 1]], y=[gt_positions[gt_idx, 0]],
                        mode='markers', marker=dict(size=15, color='green', symbol='star'),
                        showlegend=(col == 1), name='GT current' if col == 1 else None,
                    ), row=1, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=[mean[1]], y=[mean[0]],
                        mode='markers', marker=dict(size=15, color='red', symbol='x'),
                        showlegend=(col == 1), name='Estimate' if col == 1 else None,
                    ), row=1, col=col
                )

        # --- Bottom-left (2,1): Belief heatmap full view ---
        fig.add_trace(
            go.Heatmap(
                z=belief_data, x=lon_coords, y=lat_coords,
                colorscale='Hot', showscale=False, opacity=0.7,
            ), row=2, col=1
        )

        # GT path on belief full view
        fig.add_trace(
            go.Scatter(
                x=gt_positions[:gt_idx+1, 1], y=gt_positions[:gt_idx+1, 0],
                mode='lines+markers', line=dict(color='green', width=2),
                marker=dict(size=5), showlegend=False,
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[gt_positions[gt_idx, 1]], y=[gt_positions[gt_idx, 0]],
                mode='markers', marker=dict(size=15, color='green', symbol='star'),
                showlegend=False,
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[mean[1]], y=[mean[0]],
                mode='markers', marker=dict(size=15, color='red', symbol='x'),
                showlegend=False,
            ), row=2, col=1
        )

        # --- Bottom-right (2,2): Belief heatmap zoomed ---
        bel_z_crop, bel_lat_crop, bel_lon_crop = crop_heatmap(
            belief_data, lat_coords, lon_coords,
            center_lat, center_lon, zoom_size)
        fig.add_trace(
            go.Heatmap(
                z=bel_z_crop, x=bel_lon_crop, y=bel_lat_crop,
                colorscale='Hot', showscale=True,
                colorbar=dict(title='Belief', x=1.0, len=0.45, y=0.22),
                opacity=0.7,
            ), row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=gt_positions[:gt_idx+1, 1], y=gt_positions[:gt_idx+1, 0],
                mode='lines+markers', line=dict(color='green', width=2),
                marker=dict(size=8), showlegend=False,
            ), row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[gt_positions[gt_idx, 1]], y=[gt_positions[gt_idx, 0]],
                mode='markers', marker=dict(size=20, color='green', symbol='star'),
                showlegend=False,
            ), row=2, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=[mean[1]],
                y=[mean[0]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='x'),
                showlegend=False,
            ),
            row=2, col=2
        )

        # Set zoom for right column (top-right and bottom-right)
        fig.update_xaxes(range=[center_lon - zoom_size, center_lon + zoom_size], row=1, col=2)
        fig.update_yaxes(range=[center_lat - zoom_size, center_lat + zoom_size], row=1, col=2)
        fig.update_xaxes(range=[center_lon - zoom_size, center_lon + zoom_size], row=2, col=2)
        fig.update_yaxes(range=[center_lat - zoom_size, center_lat + zoom_size], row=2, col=2)

        fig.update_layout(
            height=900,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        _t_fig = _time.time() - _t0

        # Compute error
        error_deg = np.sqrt((mean[0] - gt_positions[gt_idx, 0])**2 +
                           (mean[1] - gt_positions[gt_idx, 1])**2)
        error_m = error_deg * web_mercator.METERS_PER_DEG_LAT

        info = f"Path {path_idx} | Stage: {stage} | GT idx: {gt_idx} | Error: {error_m:.1f}m"

        _t0 = _time.time()
        # Load panorama and top satellite patches (skip in simple mode)
        if simple_mode:
            pano_src = ""
            pano_title = "Panorama"
            patch_children = []
            patches_title = "Top Satellite Patches by Observation Likelihood"
        else:
            pano_row = vigor_dataset._panorama_metadata[
                vigor_dataset._panorama_metadata['pano_id'] == pano_id
            ]
            if not pano_row.empty:
                pano_path = pano_row.iloc[0]['path']
                pano_img, _ = vd.load_image(pano_path, (320, 640))
                pano_src = tensor_to_base64(pano_img)
            else:
                pano_src = ""
            pano_title = f"Panorama: {pano_id}"

            if obs_log_ll is not None:
                top_k = min(10, len(obs_log_ll))
                top_indices = torch.argsort(obs_log_ll, descending=True)[:top_k]
                patch_children = []
                for rank, idx in enumerate(top_indices):
                    idx_int = idx.item()
                    ll_val = obs_log_ll[idx_int].item()
                    sat_path = vigor_dataset._satellite_metadata.iloc[idx_int]['path']
                    sat_img, _ = vd.load_image(sat_path, (160, 160))
                    sat_src = tensor_to_base64(sat_img)
                    patch_children.append(
                        html.Div([
                            html.Img(src=sat_src, style={'width': '140px', 'height': '140px', 'objectFit': 'cover'}),
                            html.Div(f"#{rank+1} LL:{ll_val:.2f}", style={'fontSize': '11px', 'textAlign': 'center'}),
                        ], style={'display': 'inline-block', 'textAlign': 'center'})
                    )
                patches_title = f"Top {top_k} Satellite Patches by Observation Likelihood"
            else:
                patch_children = [html.Div("No observation at this step", style={'color': 'gray', 'fontStyle': 'italic'})]
                patches_title = "Top Satellite Patches by Observation Likelihood"

        _t_images = _time.time() - _t0
        _t_total = _time.time() - _t_start
        print(f"  Timing: belief_heatmap={_t_belief:.2f}s obs_heatmap={_t_obs:.2f}s "
              f"fig_build={_t_fig:.2f}s images={_t_images:.2f}s total={_t_total:.2f}s",
              file=sys.stderr, flush=True)
        return fig, info, pano_src, pano_title, patch_children, patches_title

    print(f"Starting server at http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
