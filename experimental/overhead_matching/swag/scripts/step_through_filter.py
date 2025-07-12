"""
Given a single path evaluation, we want to be able to see...
-What is the observation?
-What areas of the satellite image are marked as likly (laid over rgb)
-Where are the particles, and how likely are they? (e.g., be able to filter by a minimum/maximum likleihood)
    - They get their liklihood from the closest patch. Hovering over a patch should give its liklihood, same with hovering over a particle
    - Need to be able to turn on/off patches/particels
-Need to be able to step through liklihood calculation, movement, and resampling

Questions to answer:
- Why do the similarity patterns update as they do? Can we tell what the model is latching onto, or not 
- Why do the particles act as they do. Why were they creating a line? 
    - why do they get the liklihoods they do (based on the patches)
    - why do they move the way they do


First version:
Understand why the particles do what they do.
Workflow: 
- See generally how the particles move across the map. Find a transition that we want to focus on
- For that transition, activiate a particular feature we want to look at
    - particle liklihood
    - Particle motion
    - 

Step 1: visualize dataset
    - 
Step 2: visualize particles during observation step
Step 3: visualize particles during move step
"""

from dash import Dash, dcc, html, Input, Output, State, callback_context
import common.torch.load_torch_deps
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import base64
import argparse
import cv2
import json
from experimental.overhead_matching.swag.scripts.evaluate_model_on_paths import construct_path_eval_inputs_from_args
from experimental.overhead_matching.swag.evaluation.evaluate_swag import construct_inputs_and_evaluate_path
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
from google.protobuf import text_format
from torch_kdtree import build_kd_tree

DEVICE="cuda:0"

# Add this near the top of the file with other global variables
SHOW_LIKLIHOOD_WITH_OPACITY = False
SHOW_SATELLITE_PATCHES = False


# CLI args
parser = argparse.ArgumentParser(description='Dataset visualizer for radial bins on costmap')
parser.add_argument('--path-eval-path', type=str, required=True, help='Path to path eval you would like to visualize')
args = parser.parse_args()
args.path_eval_path = Path(args.path_eval_path)

with open(args.path_eval_path.parent / "args.json", 'r') as f:
    path_eval_args = json.load(f)
with open(args.path_eval_path / "other_info.json", 'r') as f:
    aux_info = json.load(f)

gt_path_pano_indices = torch.load(args.path_eval_path / 'path.pt', weights_only=True)

vigor_dataset, sat_model, pano_model, paths_data = construct_path_eval_inputs_from_args(
    sat_model_path=path_eval_args['sat_path'],
    pano_model_path=path_eval_args['pano_path'],
    dataset_path=path_eval_args['dataset_path'],
    paths_path=path_eval_args['paths_path'],
    panorama_neighbor_radius_deg=path_eval_args['panorama_neighbor_radius_deg'],
    device="cuda:0"
)
with open(args.path_eval_path.parent / "wag_config.pbtxt", 'r') as f:
    wag_config = WagConfig()
    wag_config = text_format.Parse(f.read(), wag_config)

sat_data_view = vigor_dataset.get_sat_patch_view()
sat_data_view_loader = vd.get_dataloader(sat_data_view, batch_size=64, num_workers=16)
pano_data_view = vigor_dataset.get_pano_view()
pano_data_view_loader = vd.get_dataloader(pano_data_view, batch_size=64, num_workers=16)

sat_patch_positions = vigor_dataset.get_patch_positions().to(DEVICE)
sat_patch_kdtree = build_kd_tree(sat_patch_positions)

path_similarity_values = torch.load(args.path_eval_path / "similarity.pt", weights_only=True)
print("starting constructing particle histories")
inference_result = construct_inputs_and_evaluate_path(
    device=DEVICE,
    generator_seed=aux_info['seed'],
    path=gt_path_pano_indices,
    vigor_dataset=vigor_dataset,
    path_similarity_values=path_similarity_values,
    wag_config=wag_config,
    return_intermediates=True
)
print("finished calculating particle histories")
gt_path_latlong = vigor_dataset.get_panorama_positions(gt_path_pano_indices).cpu().numpy()
particle_histories = inference_result.particle_history.numpy()
log_particle_weights = inference_result.log_particle_weights.numpy()
particle_histories_pre_move = inference_result.particle_history_pre_move.numpy()
print(f"Calculated histories with shapes: {particle_histories.shape=}, {log_particle_weights.shape}, {particle_histories_pre_move.shape}")


# Build figure
from plotly.graph_objects import Figure, Image, Scattergl, Scatter

# Setup Dash
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div(
            dcc.Dropdown(
                id='view-mode-dropdown',
                options=[
                    {'label': 'View particles', 'value': 'particles'},
                    {'label': 'View resampling (TODO)', 'value': 'resampling'},
                    {'label': 'View motion (TODO)', 'value': 'motion'}
                ],
                value='particles',
                clearable=False
            ),
            style={'width':'20%', 'display':'inline-block'}
        ),
        html.Div(
            [
                html.Div([
                    dcc.Checklist(
                        id='weight-opacity-toggle',
                        options=[{'label': 'View particle weight as opacity', 'value': 'enabled'}],
                        value=[]
                    ),
                    dcc.Checklist(
                        id='satellite-patches-toggle',
                        options=[{'label': 'Show satellite patches', 'value': 'enabled'}],
                        value=[]
                    )
                ], id='weight-opacity-container', style={'margin-top': '10px'})
            ],
            style={'width':'20%', 'display':'inline-block'}
        ),
        html.Div([
            html.Button('Step -1', id='step-back-button', n_clicks=0, style={'margin': '5px'}),
            html.Button('Play', id='play-button', n_clicks=0, style={'margin': '5px'}),
            html.Button('Step +1', id='step-forward-button', n_clicks=0, style={'margin': '5px'}),
            html.Div(id='animation-status', children='Paused', style={'display': 'inline-block', 'margin': '10px'}),
            dcc.Slider(
                id='frame-slider',
                min=0,
                max=100,  # Will be updated dynamically
                step=1,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            dcc.Input(
                id='step-size',
                type='number',
                min=1,
                max=20,
                step=1,
                value=1,
                style={'width': '60px', 'margin': '10px'}
            ),
            html.Label('Step Size', style={'margin-left': '5px'})
        ], style={'width': '70%', 'margin': '10px'}),
        dcc.Graph(id='graph', style={'height':'80vh', 'width': '70%', 'display': 'inline-block'}),
        html.Div([
            html.Div(id='image-panel', style={'padding': '10px'}),
            html.Div(id='satellite-image-container', style={'padding': '10px'})
        ], style={'width': '28%', 'display': 'inline-block', 'vertical-align': 'top'}),
        # Interval component for animation
        dcc.Interval(
            id='animation-interval',
            interval=500,  # in milliseconds
            n_intervals=0,
            disabled=True
        ),
        # Store components to maintain state
        dcc.Store(id='animation-state', data={'playing': False, 'current_index': 0}),
        dcc.Store(id='view-mode-data', data={'current_view_mode': "particles", 'num_points': 0})
    ])
])

# Callback to update topic data when dropdown changes
@app.callback(
    Output('view-mode-data', 'data'),
    Output('frame-slider', 'max'),
    Output('frame-slider', 'value'),
    Input('view-mode-dropdown', 'value'),
)
def update_view_mode_data(view_mode):
    # Reset animation when view mode changes
    return {'current_view_mode': view_mode, 'num_points': particle_histories.shape[0]}, particle_histories.shape[0], 0

# Callback for play/pause button
@app.callback(
    Output('animation-interval', 'disabled'),
    Output('animation-status', 'children'),
    Output('play-button', 'children'),
    Output('animation-state', 'data'),
    Input('play-button', 'n_clicks'),
    State('animation-state', 'data'),
    State('frame-slider', 'value'),
)
def toggle_animation(n_clicks, animation_state, current_slider_value):
    if n_clicks == 0:
        return True, 'Paused', 'Play', {'playing': False, 'current_index': current_slider_value}
   
    # Toggle playing state
    playing = not animation_state['playing']
    
    if playing:
        return False, 'Playing', 'Pause', {'playing': True, 'current_index': current_slider_value}
    else:
        return True, 'Paused', 'Play', {'playing': False, 'current_index': current_slider_value}

# Callback for animation interval
@app.callback(
    Output('frame-slider', 'value', allow_duplicate=True),
    Input('animation-interval', 'n_intervals'),
    State('animation-state', 'data'),
    State('frame-slider', 'value'),
    State('frame-slider', 'max'),
    State('step-size', 'value'),
    prevent_initial_call=True
)
def update_frame_on_interval(n_intervals, animation_state, current_value, max_value, step_size):
    if not animation_state['playing']:
        return current_value
    
    # Calculate next frame value
    next_value = current_value + step_size
    
    # Loop back to the beginning if we reach the end
    if next_value > max_value:
        next_value = 0
    
    return next_value

# Add a callback to toggle the opacity checkbox visibility
@app.callback(
    Output('weight-opacity-container', 'style'),
    Input('view-mode-dropdown', 'value')
)
def toggle_opacity_checkbox(view_mode):
    if view_mode == 'particles':
        return {'margin-top': '10px', 'display': 'block'}
    else:
        return {'margin-top': '10px', 'display': 'none'}

# Add a callback to update the SHOW_LIKLIHOOD_WITH_OPACITY variable
@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('weight-opacity-toggle', 'value'),
    State('view-mode-dropdown', 'value'),
    State('frame-slider', 'value'),
    prevent_initial_call=True
)
def update_opacity_setting(toggle_value, view_mode, slider_value):
    global SHOW_LIKLIHOOD_WITH_OPACITY
    SHOW_LIKLIHOOD_WITH_OPACITY = 'enabled' in toggle_value
    
    # Trigger a redraw of the graph
    return update_graph(view_mode, None, slider_value, None, {'current_view_mode': view_mode, 'num_points': particle_histories.shape[0]})

# Add a callback for the satellite patches toggle
@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('satellite-patches-toggle', 'value'),
    State('view-mode-dropdown', 'value'),
    State('frame-slider', 'value'),
    prevent_initial_call=True
)
def update_satellite_patches_setting(toggle_value, view_mode, slider_value):
    global SHOW_SATELLITE_PATCHES
    SHOW_SATELLITE_PATCHES = 'enabled' in toggle_value
    
    # Trigger a redraw of the graph
    return update_graph(view_mode, None, slider_value, None, {'current_view_mode': view_mode, 'num_points': particle_histories.shape[0]})

# Update the callback to display satellite image when a patch is clicked
@app.callback(
    Output('satellite-image-container', 'children'),
    Input('graph', 'clickData'),
    State('view-mode-dropdown', 'value')
)
def display_satellite_image(clickData, view_mode):
    if not clickData or not SHOW_SATELLITE_PATCHES:
        return html.Div("Click on a satellite patch to view the image")
    
    # The debug output shows that satellite patches are in curveNumber 1, not 2
    curve_number = clickData['points'][0]['curveNumber']
    point_index = clickData['points'][0]['pointIndex']
    
    # Check if this is a satellite patch click (curveNumber 1)
    if curve_number == 1:
        try:
            # Get the satellite index from the point index
            sat_idx = point_index
            
            # Load satellite image
            sat_metadata = vigor_dataset._satellite_metadata.iloc[sat_idx]
            sat_img = vd.load_image(sat_metadata.path, vigor_dataset._satellite_patch_size).permute(1, 2, 0).cpu().numpy()
            
            # Convert to base64 for display
            _, buffer = cv2.imencode('.png', (sat_img * 255).astype(np.uint8)[..., ::-1])  # BGR to RGB for display
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            return html.Div([
                html.H4(f"Satellite Patch #{sat_idx}"),
                html.Img(src=f'data:image/png;base64,{img_str}', style={'width': '100%'}),
                html.P(f"Lat: {sat_metadata.lat:.6f}, Lon: {sat_metadata.lon:.6f}")
            ])
        except Exception as e:
            return html.Div([
                html.H4("Error Loading Image"),
                html.P(f"Error: {str(e)}"),
                html.P(f"Satellite index: {point_index}, Curve number: {curve_number}")
            ])
    
    return html.Div("Click on a satellite patch to view the image")

# Single callback to update figure based on topic, click, and slider
@app.callback(
    Output('graph', 'figure'),
    Input('view-mode-dropdown', 'value'),
    Input('graph', 'clickData'),
    Input('frame-slider', 'value'),
    State('graph', 'figure'),
    State('view-mode-data', 'data')
)
def update_graph(view_mode, clickData, slider_value, current_fig, view_mode_data):
    # Create global variables to track the selected point and its bins
    global selected_point_idx, current_view_mode
    
    # Determine what triggered the callback
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Base figure
    fig = Figure()
    # Plot true path: most points grey, current point highlighted
    num_points = gt_path_latlong.shape[0]

    # All points except current: grey and semi-transparent
    fig.add_trace(Scattergl(
        x=gt_path_latlong[:, 1],
        y=gt_path_latlong[:, 0],
        mode='markers+lines',
        marker=dict(color='rgba(150,150,150,0.5)', size=8, line=dict(width=0)),
        line=dict(color='rgba(180,180,180,0.3)', width=2),
        name='True Path',
        hoverinfo='skip',
        showlegend=True
    ))

    # Add satellite patches if enabled
    if SHOW_SATELLITE_PATCHES:
        # Get satellite patch positions
        sat_positions = vigor_dataset.get_patch_positions().cpu().numpy()
        
        # Create all satellite patches with a single Scattergl trace
        fig.add_trace(Scattergl(
            x=sat_positions[:, 1],  # Longitude
            y=sat_positions[:, 0],  # Latitude
            mode='markers',
            marker=dict(
                symbol='square',
                size=2.56,
                color='rgba(100,100,255,0.2)',
                line=dict(color='rgba(100,100,255,0.5)', width=1)
            ),
            name='Satellite Patches',
            hovertemplate='Satellite #%{pointIndex}<br>(%{x:.6f}, %{y:.6f})',
            showlegend=True
        ))

    if SHOW_LIKLIHOOD_WITH_OPACITY:
        # Get particle positions and weights for current frame
        particles_x = particle_histories[slider_value, :, 1]
        particles_y = particle_histories[slider_value, :, 0]
        
        # Process log weights to get opacity values
        weights = log_particle_weights[slider_value, :]
        
        # Normalize log weights for colorscale and opacity
        # Get min/max for scaling
        min_log_weight = np.min(weights)
        max_log_weight = np.max(weights)
        weight_range = max_log_weight - min_log_weight if max_log_weight > min_log_weight else 1.0
        
        # Normalize to [0,1] for colorscale
        normalized_weights = (weights - min_log_weight) / weight_range if weight_range > 0 else np.zeros_like(weights)
        
        # Compute opacities directly from log weights (normalized)
        min_opacity = 0.1
        opacities = min_opacity + (1-min_opacity) * normalized_weights
        
        # Add weight information to hover data
        hover_data = [f"{w:.4f}" for w in weights]
        
        # Plot particles with both color and opacity based on log weights
        fig.add_trace(Scattergl(
            x=particles_x, 
            y=particles_y, 
            mode='markers', 
            marker=dict(
                color=weights,  # Use log weights directly 
                colorscale='Viridis',  # Use a colorscale
                colorbar=dict(
                    title='Log Weight',
                    x=1.05,  # Position colorbar further to the right
                    xpad=20,  # Add padding
                    len=0.8,  # Make it slightly shorter
                    y=0.5,   # Center it vertically
                    yanchor='middle'
                ),
                size=6,
                opacity=opacities
            ),
            customdata=hover_data,
            name='Particles', 
            hovertemplate='Log Weight: %{customdata}<br>(%{x:.4f}, %{y:.4f})'
        ))
    else:
        fig.add_trace(Scattergl(
            x=particle_histories[slider_value, :, 1], 
            y=particle_histories[slider_value, :, 0], 
            mode='markers', 
            marker=dict(color='red', size=6),
            name='Particles',
            hovertemplate='(%{x:.2f},%{y:.2f})'
        ))
    
    
    # Highlight current point: bright color, larger
    if 0 <= slider_value < num_points:
        fig.add_trace(Scattergl(
            x=[gt_path_latlong[slider_value, 1]],
            y=[gt_path_latlong[slider_value, 0]],
            mode='markers',
            marker=dict(color='orange', size=18, line=dict(color='black', width=3)),
            name='Current True Position',
            hovertemplate='True Path<br>(%{x:.2f},%{y:.2f})',
            showlegend=True
        ))

    fig.update_layout(
        uirevision='constant',  # Keep view state consistent
        title=f'View Mode: {view_mode} - Frame: {slider_value + 1}/{view_mode_data["num_points"]}', 
        clickmode='event+select',
        xaxis=dict(title='Longitude'),
        yaxis=dict(title='Latitude', scaleanchor='x', autorange=True),
        # Position legend to avoid overlap with colorbar
        legend=dict(
            x=0,        # Place legend on the left side
            y=1,        # Place at the top
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.7)'  # Semi-transparent background
        ),
        # Ensure enough margin on the right for colorbar
        margin=dict(r=80)
    )
    
    return fig

# Add a callback for the step buttons
@app.callback(
    Output('frame-slider', 'value', allow_duplicate=True),
    [Input('step-back-button', 'n_clicks'),
     Input('step-forward-button', 'n_clicks')],
    [State('frame-slider', 'value'),
     State('frame-slider', 'min'),
     State('frame-slider', 'max')],
    prevent_initial_call=True
)
def step_frame(back_clicks, forward_clicks, current_value, min_value, max_value):
    ctx = callback_context
    if not ctx.triggered:
        return current_value
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'step-back-button':
        new_value = max(min_value, current_value - 1)
        return new_value
    elif button_id == 'step-forward-button':
        new_value = min(max_value, current_value + 1)
        return new_value
    
    return current_value

if __name__=='__main__':
    print("Starting server...")
    app.run(debug=True, port=8050)
