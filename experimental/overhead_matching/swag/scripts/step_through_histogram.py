"""Web-based step-through visualization for histogram filter."""

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import json
import argparse
import io
import base64
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
    parser.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    eval_path = Path(args.eval_path).expanduser()
    dataset_path = Path(args.dataset_path).expanduser()

    # Load config
    with open(eval_path / "args.json") as f:
        eval_args = json.load(f)

    simple_mode = args.simple
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

    # Load aggregator config and create aggregator
    print("Loading aggregator config...")
    aggregator_config = load_aggregator_config(Path(args.aggregator_config))
    print(f"Loaded aggregator config: {type(aggregator_config).__name__}")
    log_likelihood_aggregator = aggregator_from_config(
        aggregator_config,
        vigor_dataset,
        device,
    )

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

    patch_positions_px = get_patch_positions_px(vigor_dataset, filter_device)
    mapping = build_cell_to_patch_mapping(grid_spec, patch_positions_px, 320.0, filter_device)

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
        'gt_positions': None,
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

        # Minimal client-side state (just path index and step count)
        dcc.Store(id='path-metadata-store'),
    ])

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
        belief = HistogramBelief.from_uniform(grid_spec, filter_device)
        history = run_filter_with_history(
            grid_spec, mapping, belief,
            motion_deltas, path,
            log_likelihood_aggregator, noise_percent
        )

        # Store in server-side cache (not sent to browser)
        cache['current_path_idx'] = path_idx
        cache['history'] = history
        cache['gt_positions'] = gt_positions

        num_steps = len(history)
        marks = {i: str(i) for i in range(0, num_steps, max(1, num_steps // 20))}

        print(f"Loaded path {path_idx} with {num_steps} steps")

        # Only send minimal metadata to browser
        return {'path_idx': path_idx, 'num_steps': num_steps}, num_steps - 1, marks, 0

    @app.callback(
        Output('step-slider', 'value', allow_duplicate=True),
        [Input('prev-btn', 'n_clicks'),
         Input('next-btn', 'n_clicks')],
        [State('step-slider', 'value'),
         State('step-slider', 'max')],
        prevent_initial_call=True
    )
    def update_slider(prev_clicks, next_clicks, current_value, max_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_value
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'prev-btn' and current_value > 0:
            return current_value - 1
        elif button_id == 'next-btn' and current_value < max_value:
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
        log_belief = step['log_belief']
        mean = step['mean'].numpy()
        gt_idx = step['gt_idx']
        pano_id = step['pano_id']
        obs_log_ll = step['obs_log_ll']

        # Create heatmap data
        belief_data, lat_coords, lon_coords = create_belief_heatmap(log_belief, grid_spec)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f'Full View - Path {path_idx}', 'Zoomed View'],
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

        # Satellite patches — color by obs likelihood on obs steps (unless simple mode)
        if not simple_mode and obs_log_ll is not None:
            ll_np = obs_log_ll.numpy()
            for subplot_col, marker_size in [(1, 3), (2, 5)]:
                fig.add_trace(
                    go.Scatter(
                        x=sat_positions[:, 1],
                        y=sat_positions[:, 0],
                        mode='markers',
                        marker=dict(
                            size=marker_size,
                            color=ll_np,
                            colorscale='Viridis',
                            opacity=0.6,
                            showscale=(subplot_col == 2),
                            colorbar=dict(title='Log LL') if subplot_col == 2 else None,
                        ),
                        name='Obs likelihood' if subplot_col == 1 else None,
                        showlegend=(subplot_col == 1),
                    ),
                    row=1, col=subplot_col
                )
        else:
            step_size = max(1, len(sat_positions) // 500)
            for subplot_col, marker_size in [(1, 3), (2, 5)]:
                fig.add_trace(
                    go.Scatter(
                        x=sat_positions[::step_size, 1],
                        y=sat_positions[::step_size, 0],
                        mode='markers',
                        marker=dict(size=marker_size, color='lightblue', opacity=0.3 if subplot_col == 1 else 0.5),
                        name='Sat patches' if subplot_col == 1 else None,
                        showlegend=(subplot_col == 1),
                    ),
                    row=1, col=subplot_col
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
                x=[mean[1]],
                y=[mean[0]],
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
                x=[mean[1]],
                y=[mean[0]],
                mode='markers',
                marker=dict(size=20, color='red', symbol='x'),
                showlegend=False,
            ),
            row=1, col=2
        )

        # Set zoom for right plot
        center_lat = (gt_positions[gt_idx, 0] + mean[0]) / 2
        center_lon = (gt_positions[gt_idx, 1] + mean[1]) / 2
        zoom_size = 0.015

        fig.update_xaxes(range=[center_lon - zoom_size, center_lon + zoom_size], row=1, col=2)
        fig.update_yaxes(range=[center_lat - zoom_size, center_lat + zoom_size], row=1, col=2)

        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        # Compute error
        error_deg = np.sqrt((mean[0] - gt_positions[gt_idx, 0])**2 +
                           (mean[1] - gt_positions[gt_idx, 1])**2)
        error_m = error_deg * web_mercator.METERS_PER_DEG_LAT

        info = f"Path {path_idx} | Stage: {stage} | GT idx: {gt_idx} | Error: {error_m:.1f}m"

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

        return fig, info, pano_src, pano_title, patch_children, patches_title

    print(f"Starting server at http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
