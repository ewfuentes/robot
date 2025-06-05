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
import pickle
import math
import base64
import argparse
import cv2
import json
import enum
from experimental.overhead_matching.swag.scripts.evaluate_model_on_paths import construct_path_eval_inputs_from_args
from experimental.overhead_matching.swag.evaluation.evaluate_swag import construct_inputs_and_evalulate_path
import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.wag_config_pb2 import WagConfig
from google.protobuf import text_format
from torch_kdtree import build_kd_tree

DEVICE="cuda:0"


# CLI args
parser = argparse.ArgumentParser(description='Dataset visualizer for radial bins on costmap')
parser.add_argument('--path-eval-path', type=str, required=True, help='Path to path eval you would like to visualize')
args = parser.parse_args()
args.path_eval_path = Path(args.path_eval_path)

with open(args.path_eval_path.parent / "args.json", 'r') as f:
    path_eval_args = json.load(f)
with open(args.path_eval_path / "other_info.json", 'r') as f:
    aux_info = json.load(f)

path = torch.load(args.path_eval_path / 'path.pt', weights_only=True)

vigor_dataset, sat_model, pano_model, paths_data = construct_path_eval_inputs_from_args(
    sat_model_path=path_eval_args['sat_path'],
    pano_model_path=path_eval_args['pano_path'],
    dataset_path=path_eval_args['dataset_path'],
    paths_path=path_eval_args['paths_path'],
    panorama_neighbor_radius_deg=path_eval_args['panorama_neighbor_radius_deg'],
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

particle_histories = construct_inputs_and_evalulate_path(
    device=DEVICE,
    generator_seed=aux_info['seed'],
    path=path,
    sat_patch_kdtree=sat_patch_kdtree,
    vigor_dataset=vigor_dataset,
    path_similarity_values=path_similarity_values,
    wag_config=wag_config
)

# # Crop to valid region (data>=0)
# data = grid.data
# # flip data upside down
# data = np.flipud(data)
# mask = data >= 0
# rows = np.any(mask,axis=1)
# cols = np.any(mask,axis=0)
# min_r, max_r = np.where(rows)[0][[0,-1]]
# min_c, max_c = np.where(cols)[0][[0,-1]]
# buf_px = int(np.ceil(args.buffer / grid.resolution))
# min_r = max(min_r-buf_px,0)
# max_r = min(max_r+buf_px,data.shape[0]-1)
# min_c = max(min_c-buf_px,0)
# max_c = min(max_c+buf_px,data.shape[1]-1)
# cropped = data[min_r:max_r+1, min_c:max_c+1]
# h,w = cropped.shape
# # compute spatial origin
# x0 = grid.x_lim_meters[0] + min_c*grid.resolution
# y0 = grid.y_lim_meters[0] + min_r*grid.resolution
# print(f"Cropped region: shape={cropped.shape}, x0={x0}, y1={y0}")

# # Scale speeds to max_speed
# speed_map = 1 / cropped
# actual_max = np.nanmax(speed_map) + 0.5
# scale = min(1.0, args.max_speed/actual_max) if actual_max>0 else 1.0
# if actual_max>args.max_speed:
#     print(f"Warning: actual_max {actual_max:.2f} > max_speed {args.max_speed:.2f}, scaling by {scale:.3f}")
# speed_map = (speed_map + 0.5) * scale
# speed_map[speed_map < 0] = 0

# # Build greyscale image
# grey = np.clip(speed_map/args.max_speed,0,1)
# img = (255*(1-grey)).astype(np.uint8)
# # convert single channel to BGR for PNG
# png = cv2.imencode('.png', img)[1].tobytes()
# b64 = base64.b64encode(png).decode('ascii')
# data_uri = f"data:image/png;base64,{b64}"

# # Build figure
# from plotly.graph_objects import Figure, Image, Scattergl, Scatter

# # Setup Dash
# dd_options = [{'label':t, 'value':t} for t in topics]
# app = Dash(__name__)
# app.layout = html.Div([
#     html.Div([
#         html.Div(
#             dcc.Dropdown(id='topic-dropdown', options=dd_options, value=topics[0], clearable=False),
#             style={'width':'20%', 'display':'inline-block'}
#         ),
#         html.Div([
#             html.Button('Play', id='play-button', n_clicks=0, style={'margin': '5px'}),
#             html.Div(id='animation-status', children='Paused', style={'display': 'inline-block', 'margin': '10px'}),
#             dcc.Slider(
#                 id='frame-slider',
#                 min=0,
#                 max=100,  # Will be updated dynamically
#                 step=1,
#                 value=0,
#                 marks=None,
#                 tooltip={"placement": "bottom", "always_visible": True}
#             ),
#             dcc.Input(
#                 id='step-size',
#                 type='number',
#                 min=1,
#                 max=20,
#                 step=1,
#                 value=5,
#                 style={'width': '60px', 'margin': '10px'}
#             ),
#             html.Label('Step Size', style={'margin-left': '5px'})
#         ], style={'width': '70%', 'margin': '10px'}),
#         dcc.Graph(id='graph', style={'height':'80vh', 'width': '70%', 'display': 'inline-block'}),
#         html.Div(id='image-panel', style={'width': '28%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
#         # Interval component for animation
#         dcc.Interval(
#             id='animation-interval',
#             interval=500,  # in milliseconds
#             n_intervals=0,
#             disabled=True
#         ),
#         # Store components to maintain state
#         dcc.Store(id='animation-state', data={'playing': False, 'current_index': 0}),
#         dcc.Store(id='topic-data', data={'current_topic': topics[0], 'num_points': 0})
#     ])
# ])

# # Callback to update topic data when dropdown changes
# @app.callback(
#     Output('topic-data', 'data'),
#     Output('frame-slider', 'max'),
#     Output('frame-slider', 'value'),
#     Input('topic-dropdown', 'value'),
# )
# def update_topic_data(topic):
#     # Reset animation when topic changes
#     sub_image_df = tbl[tbl['camera_name']==topic]
#     num_points = len(sub_image_df)
#     return {'current_topic': topic, 'num_points': num_points}, num_points-1, 0

# # Callback for play/pause button
# @app.callback(
#     Output('animation-interval', 'disabled'),
#     Output('animation-status', 'children'),
#     Output('play-button', 'children'),
#     Output('animation-state', 'data'),
#     Input('play-button', 'n_clicks'),
#     State('animation-state', 'data'),
#     State('frame-slider', 'value'),
# )
# def toggle_animation(n_clicks, animation_state, current_slider_value):
#     if n_clicks == 0:
#         return True, 'Paused', 'Play', {'playing': False, 'current_index': current_slider_value}
    
#     # Toggle playing state
#     playing = not animation_state['playing']
    
#     if playing:
#         return False, 'Playing', 'Pause', {'playing': True, 'current_index': current_slider_value}
#     else:
#         return True, 'Paused', 'Play', {'playing': False, 'current_index': current_slider_value}

# # Callback for animation interval
# @app.callback(
#     Output('frame-slider', 'value', allow_duplicate=True),
#     Input('animation-interval', 'n_intervals'),
#     State('animation-state', 'data'),
#     State('frame-slider', 'value'),
#     State('frame-slider', 'max'),
#     State('step-size', 'value'),
#     prevent_initial_call=True
# )
# def update_frame_on_interval(n_intervals, animation_state, current_value, max_value, step_size):
#     if not animation_state['playing']:
#         return current_value
    
#     # Calculate next frame value
#     next_value = current_value + step_size
    
#     # Loop back to the beginning if we reach the end
#     if next_value > max_value:
#         next_value = 0
    
#     return next_value

# # Single callback to update figure based on topic, click, and slider
# @app.callback(
#     Output('graph', 'figure'),
#     Input('topic-dropdown', 'value'),
#     Input('graph', 'clickData'),
#     Input('frame-slider', 'value'),
#     State('graph', 'figure'),
#     State('topic-data', 'data')
# )
# def update_graph(topic, clickData, slider_value, current_fig, topic_data):
#     # Create global variables to track the selected point and its bins
#     global selected_point_idx, current_topic
    
#     # Determine what triggered the callback
#     ctx = callback_context
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
#     # If topic changed, reset selection
#     if current_topic != topic:
#         current_topic = topic
#         if 'selected_point_idx' in globals():
#             del selected_point_idx
    
#     # Base figure
#     fig = Figure()
#     fig.add_trace(Image(source=data_uri, x0=x0+grid.resolution/2, y0=y0+grid.resolution/2,
#                         dx=grid.resolution, dy=grid.resolution))
    
#     # Filter points by camera topic
#     sub_image_df = tbl[tbl['camera_name']==current_topic]
#     fig.add_trace(Scattergl(
#         x=sub_image_df['x'], y=sub_image_df['y'], mode='markers', marker=dict(color='red', size=6),
#         customdata=sub_image_df.index,
#         name='pos', hovertemplate='idx:%{customdata}<br>(%{x:.2f},%{y:.2f})'
#     ))
    
#     # Handle point selection from different sources
#     if trigger_id == 'graph' and clickData and clickData['points'][0]['curveNumber'] == 1:
#         # Selection via click
#         idx = clickData['points'][0]['pointIndex']
#         selected_point_idx = sub_image_df.index[idx]
#         print(f"Selected via click: index {selected_point_idx}")
#     elif trigger_id == 'frame-slider' and len(sub_image_df) > 0:
#         # Selection via slider
#         if slider_value < len(sub_image_df):
#             selected_point_idx = sub_image_df.index[slider_value]
#             print(f"Selected via slider: index {selected_point_idx}, slider value {slider_value}")
    
#     # Add bins if a point is selected
#     if 'selected_point_idx' in globals():
#         subset = df[df['location_idx'] == selected_point_idx]
#         print(f"Selected point: {selected_point_idx}, subset size: {len(subset)}")
#         for idx, row in subset.iterrows():
#             x0r, y0r = row['x'], row['y']
#             yl, yr = row['yaw_left'], row['yaw_right']
#             if yl < yr:
#                 yl += 2*math.pi
#             thetas = np.linspace(yl, yr, 30)
            
#             for j in range(len(ranges)-1):
#                 cov = row[f'coverage_{j}']
#                 lbl = row[f'label_{j}']
#                 r0, r1 = ranges[j], ranges[j+1]
#                 xs_out = x0r + r1*np.cos(thetas)
#                 ys_out = y0r + r1*np.sin(thetas)
#                 xs_in  = x0r + r0*np.cos(thetas[::-1])
#                 ys_in  = y0r + r0*np.sin(thetas[::-1])
#                 xv = np.concatenate([xs_out, xs_in])
#                 yv = np.concatenate([ys_out, ys_in])

#                 # Style zero-coverage differently
#                 if cov <= 0:
#                     fillcol = 'lightgray'
#                     opacity = 0.2
#                     line_style = dict(color='gray', width=1, dash='dash')
#                 elif "output_0" in row: # if there are model predictions present, color based on them
#                     correct = row[f'output_{j}'] == lbl
#                     fillcol = 'rgba(100, 255, 0, 0.3)' if correct else 'rgba(255, 100, 0, 0.3)'
#                     opacity = 0.5
#                     line_style = dict(width=1, color='green' if correct else 'red')
#                 else:
#                     if lbl == 0:
#                         fillcol = 'rgba(255, 0, 0, 0.3)'
#                         opacity = 0.5
#                         line_style = dict(width=1, color='red')
#                     elif lbl == 1:
#                         fillcol = 'rgba(0, 255, 0, 0.3)'
#                         opacity = 0.5
#                         line_style = dict(width=1, color='green')
#                     else:
#                         raise ValueError(f"Unknown label {lbl} for bin {j} at index {idx}")
#                 name = f'cov {cov:.2f} l{lbl}'
#                 if "output_0" in row:
#                     name += f' o{row[f"output_{j}"]}'
#                 fig.add_trace(Scatter(
#                     x=xv, y=yv,
#                     fill='toself',
#                     fillcolor=fillcol,
#                     opacity=opacity,
#                     line=line_style,
#                     name=name,
#                     customdata=[[selected_point_idx, j, lbl, cov, idx] for _ in range(len(xv))],
#                     mode='lines',
#                     hoverinfo='text',
#                     hoveron='fills',
#                     hovertemplate=(
#                         'bin: %{customdata[1]}<br>'
#                         'parent: %{customdata[0]}<br>'
#                         'label: %{customdata[2]}<br>'
#                         'coverage: %{customdata[3]}<extra></extra>'
#                     )
#                 ))

#     # Highlight the currently selected point
#     if 'selected_point_idx' in globals():
#         # Find the position of the selected point
#         selected_row = sub_image_df[sub_image_df.index == selected_point_idx]
#         if not selected_row.empty:
#             # Add a highlight circle
#             fig.add_trace(Scatter(
#                 x=[selected_row['x'].values[0]],
#                 y=[selected_row['y'].values[0]],
#                 mode='markers',
#                 marker=dict(color='yellow', size=12, symbol='circle-open', line=dict(width=3)),
#                 name='Selected',
#                 hoverinfo='none'
#             ))

#     fig.update_layout(
#         uirevision='constant',  # Keep view state consistent
#         title=f'Topic: {topic} - Frame: {slider_value + 1}/{topic_data["num_points"]}', 
#         clickmode='event+select',
#         xaxis=dict(title='X (m)'),
#         yaxis=dict(title='Y (m)', scaleanchor='x', autorange=True)
#     )
    
#     return fig

# # Update the image panel callback to also work with slider selection
# @app.callback(
#     Output('image-panel', 'children'),
#     Input('graph', 'clickData'),
#     Input('frame-slider', 'value'),
#     State('topic-data', 'data')
# )
# def display_images(clickData, slider_value, topic_data):
#     ctx = callback_context
#     trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
#     # For slider selection, just show info about the point
#     if trigger_id == 'frame-slider' and 'selected_point_idx' in globals():
#         return [html.Div([
#             html.H3(f"Point Information:"),
#             html.P(f"Point Index: {selected_point_idx}"),
#             html.P(f"Frame: {slider_value + 1} of {topic_data['num_points']}"),
#             html.P("Click on a radial bin to view images")
#         ])]
    
#     # Original behavior for bin clicks
#     if not clickData:
#         return [html.Div("Click on a position point then a radial bin to view images")]
    
#     pt = clickData['points'][0]
    
#     # For bin clicks (curve numbers > 1)
#     if pt['curveNumber'] > 1:
#         try:
#             # Extract data from the clicked bin
#             parent_idx, bin_idx, lbl, cov, dataset_idx = pt['customdata'][0]

#             # Get dataset image and full image
#             row = df.loc[dataset_idx]
#             location_idx = row['location_idx']
            
#             # Create a header with bin information
#             header = html.Div([
#                 html.H3(f"Bin {bin_idx} Info:"),
#                 html.P(f"Parent Point: {parent_idx}"),
#                 html.P(f"Label: {lbl}"),
#                 html.P(f"Coverage: {cov}")
#             ])
            
#             # Get the dataset_image path
#             if 'img_path' not in row:
#                 return [header, html.Div("Dataset image path not available in data")]
#             dataset_image_path = Path(args.dataset_csv).parent / row['img_path']
#             if not dataset_image_path or pd.isna(dataset_image_path):
#                 return [header, html.Div("Dataset image path not available")]
            
#             # Get the full image from tbl
#             full_image_row = tbl.loc[location_idx] if location_idx < len(tbl) else None
#             if full_image_row is None:
#                 return [header, html.Div(f"Full image not found for location index {location_idx}")]
            
#             full_image_path = image_df_path.parent / full_image_row['img_path']
#             if not full_image_path or pd.isna(full_image_path):
#                 return [header, html.Div("Full image path not available")]
            
#             # Create image elements
#             image_elements = [header]
            
#             # Add dataset image
#             try:
#                 with open(dataset_image_path, 'rb') as f:
#                     dataset_img_data = base64.b64encode(f.read()).decode('utf-8')
#                 image_elements.append(html.Div([
#                     html.H4("Dataset Image"),
#                     html.Img(src=f'data:image/png;base64,{dataset_img_data}', 
#                              style={'max-width': '100%', 'margin': '5px'})
#                 ]))
#             except Exception as e:
#                 image_elements.append(html.Div(f"Error loading dataset image: {str(e)}"))
            
#             # Add full image
#             try:
#                 with open(full_image_path, 'rb') as f:
#                     full_img_data = base64.b64encode(f.read()).decode('utf-8')
#                 image_elements.append(html.Div([
#                     html.H4("Full Image"),
#                     html.Img(src=f'data:image/png;base64,{full_img_data}', 
#                              style={'max-width': '100%', 'margin': '5px'})
#                 ]))
#             except Exception as e:
#                 image_elements.append(html.Div(f"Error loading full image: {str(e)}"))
            
#             return image_elements
            
#         except Exception as e:
#             return [html.Div(f"Error processing bin click: {str(e)}")]
    
#     # For clicks on position points
#     elif pt['curveNumber'] == 1:
#         return [html.Div("Position point selected. Now click on a radial bin to view images.")]
    
#     return [html.Div("Click on a position point then a radial bin to view images")]

# if __name__=='__main__':
#     print("Starting server...")
#     app.run(debug=True, port=8050)
