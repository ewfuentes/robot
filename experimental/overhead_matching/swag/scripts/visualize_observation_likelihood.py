"""Web visualization tool for observation likelihood on VIGOR Chicago dataset.

This tool provides an interactive web interface to visualize:
- Satellite patch similarity heatmaps for selected panoramas
- OSM landmark similarities
- Combined observation likelihoods
- Geographic positions of panoramas, satellite patches, and landmarks

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:visualize_observation_likelihood -- \
        --sat-path /path/to/sat/model \
        --pano-path /path/to/pano/model \
        --dataset-path /data/overhead_matching/datasets/VIGOR/Chicago \
        --osm-embedding-path /data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses \
        --pano-embedding-path /data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1
"""

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import json
import math

from dash import Dash, dcc, html, Input, Output, State, callback_context
from plotly.graph_objects import Figure, Scattergl, Scatter, Heatmap, Scattermapbox
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from PIL import Image
import io

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.observation_likelihood import (
    compute_cached_landmark_similarity_data,
    build_prior_data_from_vigor,
    LandmarkObservationLikelihoodCalculator,
    ObservationLikelihoodConfig,
    LikelihoodMode,
    _compute_pixel_locs_px,
    _get_similarities,
)
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
import experimental.overhead_matching.swag.filter.particle_filter as pf
import common.torch.load_and_save_models as lsm
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
import msgspec

# Simple dict-based cache for likelihood computation results
# Cache key: (pano_idx, likelihood_mode, sigma_sat, sigma_osm, lat_min, lat_max, lon_min, lon_max)
# We'll manually limit size to 20 entries
likelihood_cache = {}
CACHE_MAX_SIZE = 20


def load_model(path, device='cuda'):
    """Load a trained model from a checkpoint directory."""
    try:
        model = lsm.load_model(path, device=device)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        print("Failed to load model", e)
        training_config_path = path.parent / "config.json"
        training_config_json = json.loads(training_config_path.read_text())
        model_config_json = (training_config_json["sat_model_config"]
                            if 'satellite' in path.name
                            else training_config_json["pano_model_config"])
        config = msgspec.json.decode(
                json.dumps(model_config_json),
                type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)

        model_weights = torch.load(path / 'model_weights.pt', weights_only=True)
        model_type = (patch_embedding.WagPatchEmbedding
                     if isinstance(config, patch_embedding.WagPatchEmbeddingConfig)
                     else swag_patch_embedding.SwagPatchEmbedding)
        model = model_type(config)
        model.load_state_dict(model_weights)
        model = model.to(device)
    return model


def create_app(vigor_dataset, all_similarity, landmark_obs_calculator, obs_config, device):
    """Create the Dash application for visualizing observation likelihood."""

    # Get dataset information
    panorama_positions = vigor_dataset.get_panorama_positions().numpy()
    satellite_positions = vigor_dataset.get_patch_positions().numpy()
    pano_ids = vigor_dataset._panorama_metadata['pano_id'].tolist()
    pano_id_stems = [Path(p).stem.split(',')[0] for p in vigor_dataset._panorama_metadata['path']]
    pano_paths = vigor_dataset._panorama_metadata['path'].tolist()  # Paths are already absolute Path objects

    # Create index mapping from pano_id_stem to index
    pano_stem_to_idx = {stem: idx for idx, stem in enumerate(pano_id_stems)}

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Observation Likelihood Visualizer - Chicago VIGOR"),

        html.Div([
            html.Div([
                html.H3("Select Panorama Location"),
                dcc.Graph(
                    id='panorama-map',
                    style={'height': '400px'},
                    config={
                        'scrollZoom': True,  # Enable scroll wheel zoom
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    }
                ),
                html.P(id='selected-pano-info', style={'marginTop': '10px'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.Div([
                    html.H3("Selected Panorama", id='panorama-title', style={'display': 'inline-block', 'marginRight': '10px'}),
                    html.Span(id='stale-indicator',
                             style={'display': 'inline-block', 'backgroundColor': '#ff4444',
                                   'color': 'white', 'padding': '4px 8px', 'borderRadius': '4px',
                                   'fontSize': '14px', 'fontWeight': 'bold', 'visibility': 'hidden'},
                             children='STALE'),
                ], style={'marginBottom': '10px'}),
                html.Img(id='panorama-image', style={'width': '100%', 'height': 'auto'}),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'}),
        ], style={'marginBottom': '20px'}),

        # Hidden stores
        dcc.Store(id='panorama-dropdown', data=0),
        dcc.Store(id='last-updated-params', data=None),

        html.Div([
            html.Div([
                html.Label("Likelihood Mode:"),
                dcc.RadioItems(
                    id='likelihood-mode',
                    options=[
                        {'label': 'Satellite Only', 'value': 'sat_only'},
                        {'label': 'OSM Only', 'value': 'osm_only'},
                        {'label': 'Combined', 'value': 'combined'},
                    ],
                    value='sat_only',
                    inline=True
                ),
            ], style={'width': '35%', 'display': 'inline-block', 'padding': '0 20px'}),

            html.Div([
                html.Label("Sigma (sat):"),
                dcc.Input(id='sigma-sat', type='number', value=0.1, step=0.01,
                         style={'width': '60px'}),
                html.Label(" Sigma (OSM):"),
                dcc.Input(id='sigma-osm', type='number', value=100.0, step=1.0,
                         style={'width': '60px'}),
                html.Label(" OSM Sim Scale:"),
                dcc.Input(id='osm-sim-scale', type='number', value=10.0, step=1.0,
                         style={'width': '60px'}),
            ], style={'width': '40%', 'display': 'inline-block'}),

            html.Div([
                dcc.Checklist(
                    id='enable-likelihood',
                    options=[{'label': ' Enable Likelihood Heatmap', 'value': 'enabled'}],
                    value=[],  # Unchecked by default
                    inline=True
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 20px'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("OSM View Mode:"),
                dcc.RadioItems(
                    id='osm-view-mode',
                    options=[
                        {'label': 'Overall Top Matches', 'value': 'overall'},
                        {'label': 'Per Panorama Landmark', 'value': 'per_landmark'},
                    ],
                    value='overall',
                    inline=True
                ),
            ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top'}),

            html.Div([
                html.Label("Panorama Landmark:"),
                dcc.Dropdown(
                    id='pano-landmark-dropdown',
                    options=[],  # Will be populated dynamically
                    value=None,
                    style={'width': '100%'},
                    disabled=True
                ),
            ], style={'width': '55%', 'display': 'inline-block', 'padding': '0 20px', 'verticalAlign': 'top'}),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Label("OSM Similarity Threshold:"),
            dcc.Slider(
                id='osm-similarity-threshold',
                min=0,
                max=1,
                step=0.01,
                value=0.5,
                marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Button('Update Visualization', id='submit-button', n_clicks=0,
                       style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer'})
        ], style={'marginBottom': '20px'}),

        dcc.Loading(
            id="loading",
            type="default",
            children=[html.Div(id="loading-output")],
            fullscreen=True,
        ),

        html.Div([
            html.Div([
                dcc.Graph(id='combined-map', style={'height': '600px'}),
            ]),
        ]),

        html.Div([
            html.H3("Statistics"),
            html.Div(id='stats-output')
        ], style={'marginTop': '20px'}),

        # Store for cached data
        dcc.Store(id='cached-data'),
    ])

    @app.callback(
        Output('panorama-map', 'figure'),
        [Input('panorama-dropdown', 'data')],
        [State('panorama-map', 'relayoutData')]
    )
    def render_panorama_map(selected_idx, relayout_data):
        """Render the map showing all panorama locations."""
        # Use Scattermapbox with WebGL for efficient rendering of 25k points
        fig = Figure()

        # Add all panoramas as small points
        fig.add_trace(Scattermapbox(
            lon=panorama_positions[:, 1],
            lat=panorama_positions[:, 0],
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
                opacity=0.6
            ),
            name='Panoramas',
            customdata=list(range(len(panorama_positions))),
            hovertemplate='Index: %{customdata}<br>Lat: %{lat:.6f}<br>Lon: %{lon:.6f}<extra></extra>',
            showlegend=False
        ))

        # Highlight selected panorama
        if selected_idx is not None and 0 <= selected_idx < len(panorama_positions):
            fig.add_trace(Scattermapbox(
                lon=[panorama_positions[selected_idx, 1]],
                lat=[panorama_positions[selected_idx, 0]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='circle'
                ),
                name='Selected',
                showlegend=False
            ))

        # Preserve current map viewport if it exists, otherwise center on Chicago
        if relayout_data and 'mapbox.center' in relayout_data:
            # User has interacted with the map - preserve their viewport
            center_lat = relayout_data['mapbox.center']['lat']
            center_lon = relayout_data['mapbox.center']['lon']
            zoom = relayout_data.get('mapbox.zoom', 12)
        else:
            # Initial load - center on selected panorama or Chicago
            center_lat = panorama_positions[selected_idx, 0] if selected_idx is not None else panorama_positions[:, 0].mean()
            center_lon = panorama_positions[selected_idx, 1] if selected_idx is not None else panorama_positions[:, 1].mean()
            zoom = 12

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            uirevision='constant',  # This prevents Plotly from resetting the view
            dragmode='pan'  # Enable pan mode by default (drag to pan, scroll to zoom)
        )

        return fig

    @app.callback(
        [Output('panorama-dropdown', 'data'),
         Output('selected-pano-info', 'children')],
        Input('panorama-map', 'clickData'),
        State('panorama-dropdown', 'data')
    )
    def update_selected_panorama(clickData, current_idx):
        """Update selected panorama when user clicks on the map."""
        if clickData is None:
            return current_idx, f"Panorama #{current_idx}: {pano_id_stems[current_idx]}" if current_idx is not None else "Click on a panorama marker"

        # Extract the clicked point data
        point_data = clickData['points'][0]

        # Check if user clicked on a panorama point (has customdata)
        if 'customdata' in point_data:
            clicked_idx = int(point_data['customdata'])
            return clicked_idx, f"Panorama #{clicked_idx}: {pano_id_stems[clicked_idx]}"

        # If clicked elsewhere, keep current selection
        return current_idx, f"Panorama #{current_idx}: {pano_id_stems[current_idx]}" if current_idx is not None else "Click on a panorama marker"

    @app.callback(
        [Output('panorama-image', 'src'),
         Output('panorama-title', 'children')],
        Input('panorama-dropdown', 'data')
    )
    def update_panorama_image(pano_idx):
        """Load and display the panorama image immediately when selected."""
        if pano_idx is None or pano_idx < 0 or pano_idx >= len(pano_paths):
            return "", "Selected Panorama"

        pano_path = pano_paths[pano_idx]
        pano_id_stem = pano_id_stems[pano_idx]

        try:
            # Load the image
            img = Image.open(pano_path)

            # Resize for web display (max width 800px to keep page responsive)
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.LANCZOS)

            # Convert to base64 for embedding in HTML
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}", f"Selected Panorama: {pano_id_stem}"
        except Exception as e:
            print(f"Error loading panorama image: {e}")
            return "", f"Selected Panorama: {pano_id_stem} (Image failed to load)"

    @app.callback(
        Output('stale-indicator', 'style'),
        [Input('last-updated-params', 'data'),
         Input('likelihood-mode', 'value'),
         Input('sigma-sat', 'value'),
         Input('sigma-osm', 'value'),
         Input('osm-sim-scale', 'value'),
         Input('osm-view-mode', 'value'),
         Input('pano-landmark-dropdown', 'value')]
    )
    def update_stale_indicator(last_params, likelihood_mode, sigma_sat, sigma_osm, osm_sim_scale, osm_view_mode, selected_pano_lm):
        """Show/hide the STALE indicator based on whether parameters have changed."""
        current_params = {
            'likelihood_mode': likelihood_mode,
            'sigma_sat': sigma_sat,
            'sigma_osm': sigma_osm,
            'osm_sim_scale': osm_sim_scale,
            'osm_view_mode': osm_view_mode,
            'selected_pano_lm': selected_pano_lm
        }

        if last_params is None:
            # No visualization rendered yet
            return {'display': 'inline-block', 'backgroundColor': '#ff4444', 'color': 'white',
                   'padding': '4px 8px', 'borderRadius': '4px', 'fontSize': '14px',
                   'fontWeight': 'bold', 'visibility': 'hidden'}

        # Check if parameters changed
        is_stale = last_params != current_params

        return {'display': 'inline-block', 'backgroundColor': '#ff4444', 'color': 'white',
               'padding': '4px 8px', 'borderRadius': '4px', 'fontSize': '14px',
               'fontWeight': 'bold', 'visibility': 'visible' if is_stale else 'hidden'}

    @app.callback(
        [Output('pano-landmark-dropdown', 'options'),
         Output('pano-landmark-dropdown', 'value'),
         Output('pano-landmark-dropdown', 'disabled')],
        [Input('panorama-dropdown', 'data'),
         Input('osm-view-mode', 'value')]
    )
    def update_pano_landmark_dropdown(pano_idx, osm_view_mode):
        if pano_idx is None or osm_view_mode != 'per_landmark':
            return [], None, True

        pano_id_stem = pano_id_stems[pano_idx]
        # Get pano metadata for this panorama
        pano_metadata = landmark_obs_calculator.prior_data.pano_metadata
        pano_row = pano_metadata[pano_metadata.pano_id == pano_id_stem]

        if len(pano_row) == 0:
            return [], None, True

        pano_row = pano_row.iloc[0]
        num_landmarks = len(pano_row.pano_lm_idxs)

        if num_landmarks == 0:
            return [], None, True

        options = []
        for i in range(num_landmarks):
            # Check if pano_lm_sentences exists and has this index
            if 'pano_lm_sentences' in pano_row and hasattr(pano_row['pano_lm_sentences'], '__len__') and i < len(pano_row['pano_lm_sentences']):
                sentence = pano_row['pano_lm_sentences'][i]
                label = f"Landmark {i}: {sentence[:60]}..." if len(sentence) > 60 else f"Landmark {i}: {sentence}"
            else:
                label = f"Landmark {i}"
            options.append({'label': label, 'value': i})

        return options, 0, False

    @app.callback(
        [Output('combined-map', 'figure'),
         Output('stats-output', 'children'),
         Output('loading-output', 'children'),
         Output('last-updated-params', 'data')],
        [Input('submit-button', 'n_clicks'),
         Input('osm-view-mode', 'value'),
         Input('pano-landmark-dropdown', 'value')],
        [State('panorama-dropdown', 'data'),
         State('likelihood-mode', 'value'),
         State('sigma-sat', 'value'),
         State('sigma-osm', 'value'),
         State('osm-sim-scale', 'value'),
         State('enable-likelihood', 'value'),
         State('combined-map', 'relayoutData'),
         State('osm-similarity-threshold', 'value')]
    )
    def update_visualization(n_clicks, osm_view_mode, selected_pano_lm, pano_idx, likelihood_mode, sigma_sat, sigma_osm, osm_sim_scale, enable_likelihood, combined_relayout_data, osm_threshold):
        import time
        from dash import callback_context, no_update

        timing_info = {}
        start_total = time.time()

        # Store current parameters for staleness tracking
        current_params = {
            'likelihood_mode': likelihood_mode,
            'sigma_sat': sigma_sat,
            'sigma_osm': sigma_osm,
            'osm_sim_scale': osm_sim_scale,
            'osm_view_mode': osm_view_mode,
            'selected_pano_lm': selected_pano_lm
        }

        if pano_idx is None:
            return Figure(), Figure(), "No panorama selected", "", None

        # Convert to int in case it comes as float from dcc.Store
        pano_idx = int(pano_idx)

        pano_id_stem = pano_id_stems[pano_idx]
        pano_pos = panorama_positions[pano_idx]

        # Extract viewport bounds from relayout data (or use full extent)
        start = time.time()
        if combined_relayout_data and 'xaxis.range[0]' in combined_relayout_data:
            lon_min = combined_relayout_data['xaxis.range[0]']
            lon_max = combined_relayout_data['xaxis.range[1]']
            lat_min = combined_relayout_data['yaxis.range[0]']
            lat_max = combined_relayout_data['yaxis.range[1]']
        else:
            # Fallback to full extent
            lat_min, lat_max = satellite_positions[:, 0].min(), satellite_positions[:, 0].max()
            lon_min, lon_max = satellite_positions[:, 1].min(), satellite_positions[:, 1].max()
        timing_info['Extract viewport bounds'] = time.time() - start

        # Always compute similarity heatmap (cheap operation)
        start = time.time()
        similarity_values = all_similarity[pano_idx].numpy()
        timing_info['Get similarity values'] = time.time() - start

        # Create combined figure with two subplots (side by side)
        start = time.time()
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Satellite Patch Similarity', 'Observation Log-Likelihood'),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.05
        )

        # Plot satellite patches colored by similarity (left subplot)
        fig.add_trace(Scattergl(
            x=satellite_positions[:, 1],  # Longitude
            y=satellite_positions[:, 0],  # Latitude
            mode='markers',
            marker=dict(
                color=similarity_values,
                colorscale='Viridis',
                colorbar=dict(title='Similarity', x=0.48),
                size=4,
                opacity=0.8
            ),
            name='Satellite Patches',
            hovertemplate='Similarity: %{customdata:.3f}<br>Lat: %{y:.6f}<br>Lon: %{x:.6f}',
            customdata=similarity_values
        ), row=1, col=1)

        # Mark top matches
        top_k = 20
        top_indices = np.argsort(similarity_values)[-top_k:]
        fig.add_trace(Scattergl(
            x=satellite_positions[top_indices, 1],
            y=satellite_positions[top_indices, 0],
            mode='markers',
            marker=dict(
                color='red',
                size=8,
                symbol='star'
            ),
            name=f'Top {top_k} Matches',
            hovertemplate='Rank: %{customdata}<br>Similarity: %{text:.3f}',
            customdata=list(range(top_k, 0, -1)),
            text=similarity_values[top_indices]
        ), row=1, col=1)

        # Mark selected panorama (on left subplot)
        fig.add_trace(Scattergl(
            x=[pano_pos[1]],
            y=[pano_pos[0]],
            mode='markers',
            marker=dict(
                color='lime',
                size=15,
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            name='Selected Panorama'
        ), row=1, col=1)

        timing_info['Create similarity subplot'] = time.time() - start

        # Create likelihood heatmap (with caching)
        start = time.time()
        likelihood_enabled = 'enabled' in enable_likelihood
        print(f"DEBUG: Likelihood enabled: {likelihood_enabled}, enable_likelihood value: {enable_likelihood}")

        if likelihood_enabled:
            # Round viewport bounds for cache key consistency
            cache_key = (
                pano_idx,
                likelihood_mode,
                float(sigma_sat),
                float(sigma_osm),
                float(osm_sim_scale),
                round(lat_min, 6),
                round(lat_max, 6),
                round(lon_min, 6),
                round(lon_max, 6),
                osm_view_mode,
                selected_pano_lm
            )

            # Check cache
            if cache_key in likelihood_cache:
                timing_info['Likelihood cache'] = 'HIT'
                log_likelihood_grid, lats, lons = likelihood_cache[cache_key]
                # Reconstruct flattened version for statistics
                log_likelihoods = log_likelihood_grid.flatten()
            else:
                timing_info['Likelihood cache'] = 'MISS'
                # Create grid of particles over viewport
                grid_size = 50  # 50x50 grid
                lats = np.linspace(lat_min, lat_max, grid_size)
                lons = np.linspace(lon_min, lon_max, grid_size)
                lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
                particles = torch.tensor(np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=1),
                                        dtype=torch.float32)

                # Compute log likelihoods
                mode = LikelihoodMode(likelihood_mode)

                # Update config with current sigma values
                current_config = ObservationLikelihoodConfig(
                    obs_likelihood_from_sat_similarity_sigma=float(sigma_sat),
                    obs_likelihood_from_osm_similarity_sigma=float(sigma_osm),
                    osm_similarity_scale=float(osm_sim_scale),
                    likelihood_mode=mode
                )

                # Create a temporary calculator with the updated config
                temp_calculator = LandmarkObservationLikelihoodCalculator(
                    prior_data=landmark_obs_calculator.prior_data,
                    config=current_config,
                    device=device
                )

                # Compute likelihoods for the grid
                timing_info['Setup likelihood grid'] = time.time() - start
                start = time.time()
                with torch.no_grad():
                    # Pass selected landmark index when in per_landmark mode
                    selected_landmark_idx = selected_pano_lm if osm_view_mode == 'per_landmark' else None
                    print(f"DEBUG: osm_view_mode={osm_view_mode}, selected_pano_lm={selected_pano_lm}, selected_landmark_idx={selected_landmark_idx}")
                    log_likelihoods = temp_calculator.compute_log_likelihoods(
                        particles, [pano_id_stem], selected_pano_landmark_idx=selected_landmark_idx)
                    log_likelihoods -= torch.logsumexp(log_likelihoods, (0, 1))
                    log_likelihoods = log_likelihoods.squeeze(0).numpy()
                timing_info['Compute log likelihoods'] = time.time() - start

                # Handle -inf values for visualization
                start = time.time()
                finite_mask = np.isfinite(log_likelihoods)
                num_finite = finite_mask.sum()
                print(f"DEBUG: Likelihood computation - {num_finite}/{len(log_likelihoods)} finite values, "
                      f"mode={likelihood_mode}, sigma_sat={sigma_sat}, sigma_osm={sigma_osm}")

                if finite_mask.any():
                    min_finite = log_likelihoods[finite_mask].min()
                    log_likelihoods[~finite_mask] = min_finite - 5  # Set -inf to below min
                else:
                    # All values are -inf, use a dummy value for visualization
                    print(f"WARNING: All likelihood values are -inf! Using dummy values for visualization.")
                    log_likelihoods = np.full_like(log_likelihoods, -100.0)

                # Reshape to grid for heatmap
                log_likelihood_grid = log_likelihoods.reshape(grid_size, grid_size)

                # Cache the result (implement simple size limiting)
                if len(likelihood_cache) >= CACHE_MAX_SIZE:
                    # Remove oldest entry (first key in dict)
                    oldest_key = next(iter(likelihood_cache))
                    del likelihood_cache[oldest_key]
                likelihood_cache[cache_key] = (log_likelihood_grid, lats, lons)

            # Debug: print heatmap info
            print(f"DEBUG: Creating likelihood heatmap - grid shape: {log_likelihood_grid.shape}, "
                  f"lat range: [{lats.min():.6f}, {lats.max():.6f}], "
                  f"lon range: [{lons.min():.6f}, {lons.max():.6f}], "
                  f"likelihood range: [{log_likelihood_grid.min():.2f}, {log_likelihood_grid.max():.2f}]")

            # Add heatmap (right subplot)
            fig.add_trace(Heatmap(
                z=log_likelihood_grid,
                x=lons,
                y=lats,
                colorscale='Viridis',
                colorbar=dict(title='Log Likelihood', x=1.02),
                hovertemplate='Lat: %{y:.6f}<br>Lon: %{x:.6f}<br>Log L: %{z:.2f}'
            ), row=1, col=2)

            # Mark selected panorama (right subplot)
            fig.add_trace(Scattergl(
                x=[pano_pos[1]],
                y=[pano_pos[0]],
                mode='markers',
                marker=dict(
                    color='lime',
                    size=15,
                    symbol='circle',
                    line=dict(color='black', width=2)
                ),
                name='Selected Panorama (Likelihood)',
                showlegend=False  # Already shown in left subplot
            ), row=1, col=2)

            timing_info['Create likelihood subplot'] = time.time() - start
        else:
            # Likelihood disabled - show placeholder text
            fig.add_annotation(
                text="Enable Likelihood Heatmap to compute",
                xref="x2 domain", yref="y2 domain",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray"),
                row=1, col=2
            )
            timing_info['Likelihood disabled'] = time.time() - start

        # Add OSM landmark visualization
        start = time.time()
        osm_lats, osm_lons, osm_similarities_to_show, osm_hover_text = [], [], [], []

        if likelihood_mode in ['osm_only', 'combined']:
            from experimental.overhead_matching.swag.evaluation.observation_likelihood import _get_similarities

            # Get similarities for this panorama
            similarities = _get_similarities(landmark_obs_calculator.prior_data, [pano_id_stem])
            pano_lm_similarities = similarities.landmark[0]  # (num_pano_landmarks, num_osm_landmarks)

            if osm_view_mode == 'overall':
                # Max similarity per OSM landmark across all panorama landmarks
                max_sim_per_osm = pano_lm_similarities.max(dim=0).values.numpy()
                # Filter by threshold instead of top-k
                above_threshold = max_sim_per_osm >= osm_threshold
                top_indices = np.where(above_threshold)[0]
                osm_similarities_to_show = max_sim_per_osm[top_indices]
                # Sort by similarity (descending)
                sort_order = np.argsort(osm_similarities_to_show)[::-1]
                top_indices = top_indices[sort_order]
                osm_similarities_to_show = osm_similarities_to_show[sort_order]

            else:  # per_landmark mode
                if selected_pano_lm is not None and selected_pano_lm < pano_lm_similarities.shape[0]:
                    # Get similarities for selected panorama landmark
                    pano_lm_sims = pano_lm_similarities[selected_pano_lm].numpy()
                    # Filter by threshold instead of top-k
                    above_threshold = pano_lm_sims >= osm_threshold
                    top_indices = np.where(above_threshold)[0]
                    osm_similarities_to_show = pano_lm_sims[top_indices]
                    # Sort by similarity (descending)
                    sort_order = np.argsort(osm_similarities_to_show)[::-1]
                    top_indices = top_indices[sort_order]
                    osm_similarities_to_show = osm_similarities_to_show[sort_order]
                else:
                    top_indices = np.array([])
                    osm_similarities_to_show = np.array([])

            # Build visualization data for top OSM landmarks
            if len(top_indices) > 0:
                osm_geometry_subset = landmark_obs_calculator.prior_data.osm_geometry.iloc[top_indices]

                # Debug: Print info about first landmark to help diagnose sentence issues
                if len(osm_geometry_subset) > 0:
                    first_row = osm_geometry_subset.iloc[0]
                    print(f"DEBUG: First OSM landmark columns: {list(osm_geometry_subset.columns)}")
                    print(f"DEBUG: Has 'sentence' column: {'sentence' in osm_geometry_subset.columns}")
                    if 'sentence' in osm_geometry_subset.columns:
                        first_sentence = first_row.get('sentence', '')
                        print(f"DEBUG: First landmark sentence length: {len(first_sentence) if first_sentence else 0}")
                        if first_sentence:
                            print(f"DEBUG: First 100 chars: {first_sentence[:100]}")

                for idx, (_, row) in enumerate(osm_geometry_subset.iterrows()):
                    # Convert to lat/lon
                    geom = row['geometry'] if 'geometry' in row else None
                    if geom is None:
                        # Fallback: skip this landmark if no geometry
                        continue
                    if geom.geom_type == 'Point':
                        osm_lons.append(geom.x)
                        osm_lats.append(geom.y)
                    else:
                        centroid = geom.centroid
                        osm_lons.append(centroid.x)
                        osm_lats.append(centroid.y)

                    # Build hover text
                    props = dict(row['pruned_props']) if 'pruned_props' in row and row['pruned_props'] else {}
                    hover_lines = [f"<b>{props.get('name', 'Unnamed OSM Landmark')}</b>"]
                    hover_lines.append(f"<b>Similarity: {osm_similarities_to_show[idx]:.3f}</b>")
                    hover_lines.append("")

                    # Add sentence description
                    sentence = row.get('sentence', '') if isinstance(row, dict) else (row['sentence'] if 'sentence' in row else '')
                    if sentence and len(sentence.strip()) > 0:
                        # Truncate if too long (increased limit for better readability)
                        if len(sentence) > 300:
                            sentence = sentence[:300] + "..."
                        hover_lines.append(f"<b>Description:</b>")
                        hover_lines.append(f"<i>{sentence}</i>")
                        hover_lines.append("")
                    else:
                        # Debug: indicate if sentence is missing
                        if 'sentence' not in row:
                            hover_lines.append("<i>[No sentence field in data]</i>")
                        else:
                            hover_lines.append("<i>[No description available]</i>")
                        hover_lines.append("")

                    # Add OSM tags
                    if 'amenity' in props:
                        hover_lines.append(f"Amenity: {props['amenity']}")
                    if 'building' in props:
                        hover_lines.append(f"Building: {props['building']}")

                    # Add up to 5 more properties
                    other_props = {k: v for k, v in props.items()
                                  if k not in ['name', 'amenity', 'building']}
                    for key, val in list(other_props.items())[:5]:
                        hover_lines.append(f"{key}: {val}")

                    osm_hover_text.append("<br>".join(hover_lines))

        # Add OSM landmarks to left subplot only
        # (Don't add to likelihood figure - would obscure the heatmap)
        if len(osm_lats) > 0:
            fig.add_trace(Scattergl(
                x=osm_lons,
                y=osm_lats,
                mode='markers',
                marker=dict(
                    color=osm_similarities_to_show,
                    colorscale='Plasma',
                    colorbar=dict(title='OSM Similarity', x=0.48, len=0.4, y=0.25),
                    size=10,
                    symbol='diamond',
                    line=dict(color='white', width=1.5)
                ),
                name='OSM Landmarks',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=osm_hover_text
            ), row=1, col=1)

        timing_info['OSM landmark visualization'] = time.time() - start

        start = time.time()
        max_sim = similarity_values.max()
        min_sim = similarity_values.min()
        mean_sim = similarity_values.mean()

        # Compute likelihood statistics only if likelihood was enabled
        if likelihood_enabled:
            finite_ll = log_likelihoods[np.isfinite(log_likelihoods)]
            max_ll = finite_ll.max() if len(finite_ll) > 0 else float('-inf')
            mean_ll = finite_ll.mean() if len(finite_ll) > 0 else float('-inf')
        else:
            max_ll = mean_ll = None

        # Find best match location
        best_patch_idx = np.argmax(similarity_values)
        best_patch_pos = satellite_positions[best_patch_idx]
        distance_to_best = np.sqrt(
            ((pano_pos[0] - best_patch_pos[0]) * 111000) ** 2 +  # Approximate m/deg
            ((pano_pos[1] - best_patch_pos[1]) * 85000) ** 2   # Approximate m/deg at ~40°N
        )

        stats_children = [
            html.P(f"Panorama ID: {pano_id_stem}"),
            html.P(f"Position: ({pano_pos[0]:.6f}, {pano_pos[1]:.6f})"),
        ]

        # Add selected panorama landmark info if in per-landmark mode
        if osm_view_mode == 'per_landmark' and selected_pano_lm is not None:
            pano_metadata = landmark_obs_calculator.prior_data.pano_metadata
            pano_row = pano_metadata[pano_metadata.pano_id == pano_id_stem]
            if len(pano_row) > 0:
                pano_row = pano_row.iloc[0]
                if 'pano_lm_sentences' in pano_row and hasattr(pano_row['pano_lm_sentences'], '__len__') and selected_pano_lm < len(pano_row['pano_lm_sentences']):
                    selected_lm_desc = pano_row['pano_lm_sentences'][selected_pano_lm]
                    stats_children.append(html.P(f"Selected Landmark {selected_pano_lm}: {selected_lm_desc}"))

        stats_children.extend([
            html.Hr(),
            html.P(f"Similarity - Max: {max_sim:.4f}, Min: {min_sim:.4f}, Mean: {mean_sim:.4f}"),
        ])

        if likelihood_enabled:
            stats_children.append(html.P(f"Log Likelihood - Max: {max_ll:.2f}, Mean: {mean_ll:.2f}"))
        else:
            stats_children.append(html.P("Log Likelihood - (disabled)"))

        stats_children.extend([
            html.P(f"Best match patch index: {best_patch_idx}"),
            html.P(f"Distance to best match: {distance_to_best:.1f} m"),
        ])

        if len(osm_lats) > 0:
            stats_children.append(html.P(f"Showing {len(osm_lats)} top OSM landmarks"))

        # Add timing information
        timing_info['Build statistics'] = time.time() - start
        total_time = time.time() - start_total
        timing_info['TOTAL'] = total_time

        stats_children.extend([
            html.Hr(),
            html.H4("Performance Profile"),
        ])

        # Separate numeric and string values, sort numeric by time descending
        numeric_timing = {k: v for k, v in timing_info.items() if isinstance(v, (int, float))}
        string_timing = {k: v for k, v in timing_info.items() if isinstance(v, str)}

        # Show mode first if present
        for key, value in string_timing.items():
            stats_children.append(html.P(f"{key}: {value}"))

        # Then show numeric timings sorted by duration
        sorted_timing = sorted(numeric_timing.items(), key=lambda x: x[1], reverse=True)
        for operation, duration in sorted_timing:
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            stats_children.append(
                html.P(f"{operation}: {duration:.3f}s ({percentage:.1f}%)")
            )

        stats_html = html.Div(stats_children)

        # Update layout for both subplots with shared axes
        fig.update_xaxes(title_text="Longitude", row=1, col=1)
        fig.update_xaxes(title_text="Longitude", row=1, col=2)
        fig.update_yaxes(title_text="Latitude", row=1, col=1)
        fig.update_yaxes(title_text="Latitude", row=1, col=2)

        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(x=0, y=1, xanchor='left', yanchor='top'),
            margin=dict(l=50, r=50, t=50, b=50),
            uirevision='combined'  # Maintain view state
        )

        return fig, stats_html, "", current_params

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize observation likelihood for VIGOR dataset")
    parser.add_argument("--sat-path", type=str, required=True,
                       help="Model folder path for satellite model")
    parser.add_argument("--pano-path", type=str, required=True,
                       help="Model folder path for panorama model")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to VIGOR dataset (e.g., /data/overhead_matching/datasets/VIGOR/Chicago)")
    parser.add_argument("--osm-embedding-path", type=str, default=None,
                       help="Path to OSM landmark embeddings directory")
    parser.add_argument("--pano-embedding-path", type=str, default=None,
                       help="Path to panorama landmark embeddings directory")
    parser.add_argument("--embedding-dim", type=int, default=1536,
                       help="Embedding dimension for landmark embeddings")
    parser.add_argument("--panorama-neighbor-radius-deg", type=float,
                       default=0.0005, help="Panorama neighbor radius in degrees")
    parser.add_argument("--panorama-landmark-radius-px", type=int,
                       default=640, help="Panorama landmark radius in pixels")
    parser.add_argument("--landmark-version", type=str, default="v2",
                       help="Landmark version")
    parser.add_argument("--port", type=int, default=8050,
                       help="Port to run the web server on")
    parser.add_argument("--debug", action='store_true',
                       help="Run in debug mode")
    parser.add_argument("--no-cache", action='store_true',
                       help="Disable caching and regenerate landmark similarity data (use this if sentences are missing)")

    args = parser.parse_args()

    # DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    # Load models
    sat_model_path = Path(args.sat_path).expanduser()
    pano_model_path = Path(args.pano_path).expanduser()

    print("Loading models...")
    sat_model = load_model(sat_model_path, device=DEVICE)
    pano_model = load_model(pano_model_path, device=DEVICE)

    # Load dataset
    dataset_path = Path(args.dataset_path).expanduser()
    print(f"Loading dataset from {dataset_path}...")

    dataset_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="panorama",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            extractor_info=pano_model.cache_info()),
        satellite_tensor_cache_info=vd.TensorCacheInfo(
            dataset_key=dataset_path.name,
            model_type="satellite",
            landmark_version=args.landmark_version,
            panorama_landmark_radius_px=args.panorama_landmark_radius_px,
            extractor_info=sat_model.cache_info()),
        panorama_neighbor_radius=args.panorama_neighbor_radius_deg,
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
        factor=1,
        landmark_version=args.landmark_version,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)

    # Compute or load cached similarity matrix
    print("Computing/loading similarity matrix...")
    all_similarity = es.compute_cached_similarity_matrix(
        sat_model=sat_model,
        pano_model=pano_model,
        dataset=vigor_dataset,
        device=DEVICE,
        use_cached_similarity=True
    )

    # Initialize observation likelihood config
    obs_config = ObservationLikelihoodConfig(
        obs_likelihood_from_sat_similarity_sigma=0.1,
        obs_likelihood_from_osm_similarity_sigma=100.0,
        likelihood_mode=LikelihoodMode.COMBINED
    )

    # Build landmark observation likelihood calculator if paths provided
    if args.osm_embedding_path and args.pano_embedding_path:
        osm_embedding_path = Path(args.osm_embedding_path).expanduser()
        pano_embedding_path = Path(args.pano_embedding_path).expanduser()

        print("Computing landmark similarity data...")
        if args.no_cache:
            print("WARNING: Cache disabled - regenerating all landmark similarity data from scratch")
        landmark_sim_data = compute_cached_landmark_similarity_data(
            vigor_dataset,
            osm_embedding_path,
            pano_embedding_path,
            embedding_dim=args.embedding_dim,
            use_cache=not args.no_cache
        )

        print("Building prior data...")
        prior_data = build_prior_data_from_vigor(
            vigor_dataset,
            all_similarity,
            landmark_sim_data
        )

        landmark_obs_calculator = LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=obs_config,
            device=torch.device(DEVICE)
        )
        print("Landmark observation likelihood calculator ready")

        # Check if sentences are loaded
        if 'sentence' in prior_data.osm_geometry.columns:
            num_with_sentences = (prior_data.osm_geometry['sentence'].str.len() > 0).sum()
            print(f"OSM geometry has {num_with_sentences}/{len(prior_data.osm_geometry)} landmarks with sentences")
        else:
            print("WARNING: OSM geometry does not have 'sentence' column - tooltips will not show descriptions")
    else:
        # Create a minimal calculator for satellite-only mode
        print("OSM embeddings not provided - satellite-only mode")

        # We need to create PriorData with just satellite info for the calculator
        import geopandas as gpd
        import shapely

        sat_metadata = vigor_dataset._satellite_metadata
        patch_height, patch_width = vigor_dataset._original_satellite_patch_size

        sat_geometries = []
        for _, row in sat_metadata.iterrows():
            center_x = row.web_mercator_x
            center_y = row.web_mercator_y
            geom = shapely.box(
                xmin=center_x - patch_width // 2,
                xmax=center_x + patch_width // 2,
                ymin=center_y - patch_height // 2,
                ymax=center_y + patch_height // 2
            )
            sat_geometries.append(geom)

        sat_geometry = gpd.GeoDataFrame({
            'geometry_px': sat_geometries,
            'embedding_idx': range(len(sat_metadata))
        })

        # Create empty OSM and pano metadata
        osm_geometry = gpd.GeoDataFrame({
            'osm_id': [],
            'geometry_px': [],
            'osm_embedding_idx': []
        })

        pano_id_stems = [Path(p).stem.split(',')[0]
                       for p in vigor_dataset._panorama_metadata['path']]
        pano_metadata = gpd.GeoDataFrame({
            'pano_id': pano_id_stems,
            'pano_lm_idxs': [[] for _ in range(len(pano_id_stems))]
        })

        from experimental.overhead_matching.swag.evaluation.observation_likelihood import PriorData
        prior_data = PriorData(
            osm_geometry=osm_geometry,
            sat_geometry=sat_geometry,
            pano_metadata=pano_metadata,
            pano_sat_similarity=all_similarity,
            pano_osm_landmark_similarity=torch.zeros((0, 0))
        )

        landmark_obs_calculator = LandmarkObservationLikelihoodCalculator(
            prior_data=prior_data,
            config=obs_config,
            device=torch.device(DEVICE)
        )

    # Create and run the app
    print(f"Starting web server on port {args.port}...")
    app = create_app(vigor_dataset, all_similarity, landmark_obs_calculator, obs_config, DEVICE)
    app.run(debug=args.debug, port=args.port, host='0.0.0.0')
