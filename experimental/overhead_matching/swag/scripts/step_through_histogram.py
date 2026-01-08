"""Web-based step-through visualization for histogram filter."""

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import json
import argparse
import math
import msgspec

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.histogram_belief import (
    GridSpec,
    HistogramBelief,
    build_cell_to_patch_mapping,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
import common.torch.load_and_save_models as lsm
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding


def load_model(path, device='cuda'):
    """Load a model from path."""
    try:
        model = lsm.load_model(path, device=device)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        print("Failed to load model via lsm, trying fallback:", e)
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
    path_similarity: torch.Tensor,
    sigma_obs: float,
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
    })

    path_len = path_similarity.shape[0]

    for step_idx in range(path_len - 1):
        # Observation update
        belief.apply_observation(path_similarity[step_idx], mapping, sigma_obs)
        history.append({
            'stage': f'obs_{step_idx}',
            'log_belief': belief.get_log_belief().clone().cpu(),
            'mean': belief.get_mean_latlon().clone().cpu(),
            'gt_idx': step_idx,
        })

        # Motion prediction
        belief.apply_motion(motion_deltas[step_idx], noise_percent)
        history.append({
            'stage': f'motion_{step_idx}',
            'log_belief': belief.get_log_belief().clone().cpu(),
            'mean': belief.get_mean_latlon().clone().cpu(),
            'gt_idx': step_idx + 1,
        })

    # Final observation
    belief.apply_observation(path_similarity[-1], mapping, sigma_obs)
    history.append({
        'stage': f'obs_{path_len-1}',
        'log_belief': belief.get_log_belief().clone().cpu(),
        'mean': belief.get_mean_latlon().clone().cpu(),
        'gt_idx': path_len - 1,
    })

    return history


def create_belief_heatmap(log_belief, grid_spec):
    """Convert log belief to plotly heatmap data."""
    # Normalize for visualization
    belief = torch.exp(log_belief - log_belief.max()).numpy()

    # Get lat/lon coordinates for each cell
    lat_coords = []
    lon_coords = []
    for row in range(grid_spec.num_rows):
        lat, lon = grid_spec.cell_indices_to_latlon(
            torch.tensor(float(row)), torch.tensor(0.0))
        lat_coords.append(lat.item())

    for col in range(grid_spec.num_cols):
        lat, lon = grid_spec.cell_indices_to_latlon(
            torch.tensor(0.0), torch.tensor(float(col)))
        lon_coords.append(lon.item())

    return belief, lat_coords, lon_coords


def main():
    parser = argparse.ArgumentParser(description="Web-based histogram filter visualization")
    parser.add_argument("--eval-path", type=str, required=True,
                        help="Path to evaluation results directory")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VIGOR dataset")
    parser.add_argument("--sat-path", type=str, required=True,
                        help="Path to satellite model")
    parser.add_argument("--pano-path", type=str, required=True,
                        help="Path to panorama model")
    parser.add_argument("--path-idx", type=int, required=True,
                        help="Path index to visualize")
    parser.add_argument("--sigma-obs", type=float, default=None)
    parser.add_argument("--noise-percent", type=float, default=None)
    parser.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    eval_path = Path(args.eval_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()
    sat_model_path = Path(args.sat_path).expanduser()
    pano_model_path = Path(args.pano_path).expanduser()

    # Load config
    with open(eval_path / "args.json") as f:
        eval_args = json.load(f)

    sigma_obs = args.sigma_obs if args.sigma_obs is not None else eval_args.get("sigma_obs_prob_from_sim", 0.1)
    noise_percent = args.noise_percent if args.noise_percent is not None else eval_args.get("noise_percent", 0.02)
    subdivision_factor = eval_args.get("subdivision_factor", 4)

    print(f"Config: sigma_obs={sigma_obs}, noise_percent={noise_percent}, subdivision={subdivision_factor}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    sat_model = load_model(sat_model_path, device=device)
    pano_model = load_model(pano_model_path, device=device)

    # Load dataset
    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=None,
        satellite_tensor_cache_info=None,
        panorama_neighbor_radius=0.0005,
        satellite_patch_size=(640, 640),
        panorama_size=(640, 640),
        factor=0.3,
        landmark_version=eval_args.get("landmark_version", "v4_202001"),
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    # Load path
    path_dir = eval_path / f"{args.path_idx:07d}"
    path = torch.load(path_dir / "path.pt", map_location='cpu')

    # Compute similarity matrix
    print("Computing similarity matrix (cached)...")
    all_similarity = es.compute_cached_similarity_matrix(
        sat_model=sat_model,
        pano_model=pano_model,
        dataset=vigor_dataset,
        device=device,
        use_cached_similarity=True,
    )
    path_similarity = all_similarity[path].cpu()

    # Use CPU for filter computations (visualization doesn't need GPU)
    filter_device = torch.device('cpu')

    # Build grid with buffer of half patch size (320 pixels)
    min_lat, max_lat, min_lon, max_lon = get_dataset_bounds(vigor_dataset)
    cell_size_px = 640.0 / subdivision_factor

    # Add buffer of half patch size (320 pixels at zoom 20)
    # Convert to degrees using web mercator at the center latitude
    center_lat = (min_lat + max_lat) / 2
    from common.gps import web_mercator
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

    patch_positions_px = get_patch_positions_px(vigor_dataset, filter_device)
    mapping = build_cell_to_patch_mapping(grid_spec, patch_positions_px, 320.0, filter_device)

    # Get positions
    gt_positions = vigor_dataset.get_panorama_positions(path).numpy()
    motion_deltas = es.get_motion_deltas_from_path(vigor_dataset, path)
    sat_positions = vigor_dataset._satellite_metadata[['lat', 'lon']].values

    # Initialize belief
    def degrees_from_meters(dist_m):
        return math.degrees(dist_m / 6_371_000.0)

    belief = HistogramBelief.from_uniform(grid_spec, filter_device)

    print("Running filter...")
    history = run_filter_with_history(
        grid_spec, mapping, belief,
        motion_deltas, path_similarity,
        sigma_obs, noise_percent
    )
    print(f"Generated {len(history)} steps")

    # Create Dash app
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1(f"Histogram Filter - Path {args.path_idx}"),
        html.Div([
            html.Label("Step:"),
            dcc.Slider(
                id='step-slider',
                min=0,
                max=len(history) - 1,
                value=0,
                marks={i: str(i) for i in range(0, len(history), max(1, len(history)//20))},
                step=1,
            ),
        ], style={'width': '80%', 'margin': '20px auto'}),
        html.Div([
            html.Button('Previous', id='prev-btn', n_clicks=0),
            html.Button('Next', id='next-btn', n_clicks=0),
            html.Span(id='step-info', style={'marginLeft': '20px'}),
        ], style={'textAlign': 'center', 'margin': '10px'}),
        dcc.Graph(id='main-plot', style={'height': '80vh'}),
    ])

    @app.callback(
        Output('step-slider', 'value'),
        [Input('prev-btn', 'n_clicks'),
         Input('next-btn', 'n_clicks')],
        [State('step-slider', 'value')]
    )
    def update_slider(prev_clicks, next_clicks, current_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_value
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'prev-btn' and current_value > 0:
            return current_value - 1
        elif button_id == 'next-btn' and current_value < len(history) - 1:
            return current_value + 1
        return current_value

    @app.callback(
        [Output('main-plot', 'figure'),
         Output('step-info', 'children')],
        [Input('step-slider', 'value')]
    )
    def update_plot(step_idx):
        step = history[step_idx]
        stage = step['stage']
        log_belief = step['log_belief']
        mean = step['mean']
        gt_idx = step['gt_idx']

        # Create heatmap data
        belief_data, lat_coords, lon_coords = create_belief_heatmap(log_belief, grid_spec)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Full View', 'Zoomed View'],
            horizontal_spacing=0.1,
        )

        # Heatmap for full view
        fig.add_trace(
            go.Heatmap(
                z=belief_data,
                x=lon_coords,
                y=lat_coords,
                colorscale='Hot',
                showscale=False,
                opacity=0.7,
            ),
            row=1, col=1
        )

        # Satellite patches (subsample for performance)
        step_size = max(1, len(sat_positions) // 500)
        fig.add_trace(
            go.Scatter(
                x=sat_positions[::step_size, 1],
                y=sat_positions[::step_size, 0],
                mode='markers',
                marker=dict(size=3, color='lightblue', opacity=0.3),
                name='Sat patches',
                showlegend=True,
            ),
            row=1, col=1
        )

        # GT path up to current position
        fig.add_trace(
            go.Scatter(
                x=gt_positions[:gt_idx+1, 1],
                y=gt_positions[:gt_idx+1, 0],
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=5),
                name='GT path',
            ),
            row=1, col=1
        )

        # Current GT position
        fig.add_trace(
            go.Scatter(
                x=[gt_positions[gt_idx, 1]],
                y=[gt_positions[gt_idx, 0]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='star'),
                name='GT current',
            ),
            row=1, col=1
        )

        # Estimate
        fig.add_trace(
            go.Scatter(
                x=[mean[1].item()],
                y=[mean[0].item()],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='Estimate',
            ),
            row=1, col=1
        )

        # Zoomed view (same traces)
        fig.add_trace(
            go.Heatmap(
                z=belief_data,
                x=lon_coords,
                y=lat_coords,
                colorscale='Hot',
                showscale=True,
                opacity=0.7,
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=sat_positions[::step_size, 1],
                y=sat_positions[::step_size, 0],
                mode='markers',
                marker=dict(size=5, color='lightblue', opacity=0.5),
                showlegend=False,
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=gt_positions[:gt_idx+1, 1],
                y=gt_positions[:gt_idx+1, 0],
                mode='lines+markers',
                line=dict(color='green', width=2),
                marker=dict(size=8),
                showlegend=False,
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=[gt_positions[gt_idx, 1]],
                y=[gt_positions[gt_idx, 0]],
                mode='markers',
                marker=dict(size=20, color='green', symbol='star'),
                showlegend=False,
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x=[mean[1].item()],
                y=[mean[0].item()],
                mode='markers',
                marker=dict(size=20, color='red', symbol='x'),
                showlegend=False,
            ),
            row=1, col=2
        )

        # Set zoom for right plot
        center_lat = (gt_positions[gt_idx, 0] + mean[0].item()) / 2
        center_lon = (gt_positions[gt_idx, 1] + mean[1].item()) / 2
        zoom_size = 0.015

        fig.update_xaxes(range=[center_lon - zoom_size, center_lon + zoom_size], row=1, col=2)
        fig.update_yaxes(range=[center_lat - zoom_size, center_lat + zoom_size], row=1, col=2)

        fig.update_layout(
            height=700,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        # Compute error
        error_deg = np.sqrt((mean[0].item() - gt_positions[gt_idx, 0])**2 +
                           (mean[1].item() - gt_positions[gt_idx, 1])**2)
        error_m = error_deg * 111000

        info = f"Stage: {stage} | GT idx: {gt_idx} | Error: {error_m:.1f}m"

        return fig, info

    print(f"Starting server at http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
