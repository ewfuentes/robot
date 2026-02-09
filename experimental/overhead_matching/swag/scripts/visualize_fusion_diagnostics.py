"""Interactive Dash web app for fusion diagnostics visualization.

Allows comparing image-based vs landmark-based similarity matrices
with click-to-inspect panorama details and side-by-side analysis.
"""

import common.torch.load_torch_deps
import torch
import numpy as np
from pathlib import Path
import argparse
import base64
import io
import json
import glob

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

import experimental.overhead_matching.swag.data.vigor_dataset as vd


SIGMA = 0.25


def load_similarity_matrix(path: Path) -> torch.Tensor:
    """Load a similarity matrix from a file.
    Handles both raw tensor format and dict format (with 'similarity' key).
    """
    data = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(data, dict):
        if "similarity" in data:
            return data["similarity"]
        raise ValueError(
            f"Similarity matrix file {path} is a dict but has no 'similarity' key. "
            f"Available keys: {list(data.keys())}"
        )
    return data


def _compute_rank(sim_row: torch.Tensor, true_idxs: list[int]) -> tuple[float, float]:
    """Compute rank of best true patch, breaking ties against the true patch.

    If all similarities are identical (e.g., all -inf or all 0), the true patch
    gets rank = num_patches (dead last).

    Returns (rank, best_true_sim).
    """
    if not true_idxs:
        return float("nan"), float("nan")

    true_sims = sim_row[true_idxs]
    finite_true = true_sims[torch.isfinite(true_sims)]
    if len(finite_true) == 0:
        # All true patch sims are non-finite => dead last
        return float(sim_row.shape[0]), float("nan")

    best_true_sim = finite_true.max().item()

    # Check if all values are the same (no signal)
    finite_all = sim_row[torch.isfinite(sim_row)]
    if len(finite_all) == 0:
        return float(sim_row.shape[0]), best_true_sim

    if finite_all.max().item() == finite_all.min().item():
        # All similarities identical => no signal => dead last
        return float(sim_row.shape[0]), best_true_sim

    # Count how many are strictly better (lower rank = better)
    num_strictly_better = (sim_row > best_true_sim).sum().item()
    # Count how many are tied (same value)
    num_tied = (sim_row == best_true_sim).sum().item()
    # Break ties against: rank = num_better + num_tied (worst case within tie group)
    rank = num_strictly_better + num_tied
    return rank, best_true_sim


def compute_per_pano_diagnostics(sim_matrix, pano_metadata, sigma=SIGMA):
    """Compute per-panorama diagnostic metrics for a similarity matrix."""
    num_panos = sim_matrix.shape[0]
    records = []

    for pano_idx in range(num_panos):
        sim_row = sim_matrix[pano_idx]
        finite_mask = torch.isfinite(sim_row)

        if finite_mask.sum() == 0:
            row = pano_metadata.iloc[pano_idx]
            true_idxs = list(row["positive_satellite_idxs"]) + list(row["semipositive_satellite_idxs"])
            records.append({
                "pano_idx": pano_idx, "entropy": float("nan"),
                "peak_sharpness": float("nan"), "max_sim": float("nan"),
                "true_rank": float(sim_row.shape[0]), "best_true_sim": float("nan"),
                "has_data": False,
            })
            continue

        finite_sims = sim_row[finite_mask]

        # Check if all values are the same (no signal)
        all_same = finite_sims.max().item() == finite_sims.min().item()

        # Entropy of softmax(sim / sigma)
        if all_same:
            entropy = float(np.log(finite_sims.shape[0]))  # max entropy
        else:
            logits = finite_sims / sigma
            log_probs = logits - torch.logsumexp(logits, dim=0)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum().item()

        # Peak sharpness (top-2 gap)
        finite_sorted = torch.sort(finite_sims, descending=True).values
        peak_sharpness = (finite_sorted[0] - finite_sorted[1]).item() if len(finite_sorted) >= 2 else 0.0

        max_sim = finite_sorted[0].item()

        # Rank of true patch
        row = pano_metadata.iloc[pano_idx]
        true_idxs = list(row["positive_satellite_idxs"]) + list(row["semipositive_satellite_idxs"])
        rank, best_true_sim = _compute_rank(sim_row, true_idxs)

        records.append({
            "pano_idx": pano_idx, "entropy": entropy,
            "peak_sharpness": peak_sharpness, "max_sim": max_sim,
            "true_rank": rank, "best_true_sim": best_true_sim,
            "has_data": True,
        })

    return records


def get_top_k_patches(sim_row, k=5):
    """Return top-k patch indices and their similarity scores."""
    finite_mask = torch.isfinite(sim_row)
    if finite_mask.sum() == 0:
        return [], []
    sorted_vals, sorted_idxs = torch.sort(sim_row, descending=True)
    finite_mask_sorted = torch.isfinite(sorted_vals)
    sorted_vals = sorted_vals[finite_mask_sorted][:k]
    sorted_idxs = sorted_idxs[finite_mask_sorted][:k]
    return sorted_idxs.tolist(), sorted_vals.tolist()


def load_image_as_base64(path, max_width=600):
    """Load an image from disk and return as base64-encoded JPEG."""
    path = Path(path) if not isinstance(path, Path) else path
    if not path.exists():
        print(f"Image not found: {path}")
        return ""
    try:
        img = Image.open(str(path))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return ""


def load_gemini_pano_landmarks(dataset_path: Path) -> dict[str, dict]:
    """Load Gemini panorama landmark extractions.

    Returns dict mapping pano_id -> {"location_type": str, "landmarks": [...]}.
    """
    city_name = dataset_path.name
    gemini_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini")
    city_dir = gemini_base / city_name / "sentences"

    if not city_dir.exists():
        print(f"No Gemini landmarks found at {city_dir}")
        return {}

    pred_files = sorted(glob.glob(str(city_dir / "*/*/prediction-*/predictions.jsonl")))
    if not pred_files:
        # Try alternate structure: city_out/panorama_request_*/prediction-*/predictions.jsonl
        pred_files = sorted(glob.glob(str(city_dir / "*_out/*/prediction-*/predictions.jsonl")))
    if not pred_files:
        print(f"No prediction files found under {city_dir}")
        return {}

    print(f"Loading Gemini landmarks from {len(pred_files)} files...")
    pano_landmarks = {}
    errors = 0
    for pf in pred_files:
        with open(pf) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    key = d["key"]
                    pano_id = key.split(",")[0]
                    text = d["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    landmarks = json.loads(text)
                    pano_landmarks[pano_id] = landmarks
                except (json.JSONDecodeError, KeyError, IndexError):
                    errors += 1
    print(f"Loaded {len(pano_landmarks)} panorama landmark extractions ({errors} errors)")
    return pano_landmarks


def build_sat_osm_mapping(sat_metadata, landmark_metadata):
    """Build mapping from satellite patch index to nearby OSM landmark props.

    Uses the dataset's pre-computed spatial correspondences (Shapely STRtree)
    via sat_metadata["landmark_idxs"] and landmark_metadata["pruned_props"].
    """
    if landmark_metadata is None or "landmark_idxs" not in sat_metadata.columns:
        return {}

    sat_idx_to_osm = {}
    for sat_idx in range(len(sat_metadata)):
        lm_idxs = sat_metadata.iloc[sat_idx]["landmark_idxs"]
        if not lm_idxs:
            continue
        props_list = []
        for lm_idx in lm_idxs:
            pruned = landmark_metadata.iloc[lm_idx]["pruned_props"]
            # pruned_props is a frozenset of (key, value) tuples
            props_list.append(dict(pruned))
        sat_idx_to_osm[sat_idx] = props_list
    print(f"Built OSM mapping: {len(sat_idx_to_osm)}/{len(sat_metadata)} satellite patches have OSM landmarks")
    return sat_idx_to_osm


# Allowlist of OSM keys that are semantically meaningful for understanding a location.
_OSM_DISPLAY_KEYS = {
    "name", "alt_name", "short_name", "official_name", "brand", "operator",
    "amenity", "shop", "cuisine", "sport", "leisure", "tourism", "office",
    "building", "highway", "railway", "natural", "man_made", "emergency",
    "religion", "description", "crossing",
}


def format_osm_props(props: dict, show_all: bool = False) -> str:
    """Format OSM properties into a readable string.

    If show_all is False, only shows semantically meaningful tags (allowlist).
    If show_all is True, shows all tags from pruned_props.
    """
    if show_all:
        parts = []
        for k, v in sorted(props.items()):
            parts.append(f"{k}={v}")
        return ", ".join(parts)

    parts = []
    if "name" in props:
        parts.append(props["name"])
    for k in ("amenity", "shop", "cuisine", "brand", "tourism", "leisure",
              "office", "railway", "highway", "building", "natural",
              "man_made", "emergency", "sport", "religion", "crossing",
              "description", "operator", "alt_name", "official_name"):
        if k in props and k != "name":
            parts.append(f"{k}={props[k]}")
    return ", ".join(parts)


def _osm_semantic_key(props: dict) -> str:
    """Return a dedup key based on the semantic identity of an OSM feature.

    E.g., 10 different bike racks all map to "bicycle_parking" so they collapse to one entry.
    """
    name = props.get("name", "")
    kind = props.get("amenity") or props.get("shop") or props.get("tourism") or \
           props.get("leisure") or props.get("office") or props.get("highway") or \
           props.get("railway") or props.get("building") or props.get("natural") or \
           props.get("man_made") or props.get("emergency") or ""
    return f"{name}|{kind}"


def main():
    parser = argparse.ArgumentParser(description="Interactive fusion diagnostics visualizer")
    parser.add_argument("--img-sim-path", type=str, required=True,
                        help="Path to image similarity matrix (.pt)")
    parser.add_argument("--lm-sim-path", type=str, required=True,
                        help="Path to landmark similarity matrix (.pt)")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to VIGOR dataset directory (city level, e.g. .../VIGOR/Seattle)")
    parser.add_argument("--landmark-version", type=str, default="v4_202001",
                        help="Landmark version for dataset config")
    parser.add_argument("--port", type=int, default=8051,
                        help="Port for Dash server")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser()

    # Load dataset metadata (with landmarks for proper OSM association)
    print("Loading dataset...")
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=args.landmark_version,
    )
    vigor_dataset = vd.VigorDataset(dataset_path, dataset_config)
    pano_metadata = vigor_dataset._panorama_metadata
    sat_metadata = vigor_dataset._satellite_metadata
    landmark_metadata = vigor_dataset._landmark_metadata
    print(f"Dataset: {len(pano_metadata)} panoramas, {len(sat_metadata)} satellite patches, "
          f"{len(landmark_metadata)} landmarks")

    # Load similarity matrices
    print("Loading similarity matrices...")
    img_sim = load_similarity_matrix(Path(args.img_sim_path).expanduser())
    lm_sim = load_similarity_matrix(Path(args.lm_sim_path).expanduser())
    print(f"Image sim: {img_sim.shape}, Landmark sim: {lm_sim.shape}")
    assert img_sim.shape[0] == lm_sim.shape[0], "Panorama count mismatch"

    # Load Gemini panorama landmarks
    pano_gemini_landmarks = load_gemini_pano_landmarks(dataset_path)

    # Build sat patch -> OSM landmark mapping from dataset correspondences
    sat_osm_mapping = build_sat_osm_mapping(sat_metadata, landmark_metadata)

    # Compute diagnostics
    print("Computing diagnostics...")
    img_diag = compute_per_pano_diagnostics(img_sim, pano_metadata)
    lm_diag = compute_per_pano_diagnostics(lm_sim, pano_metadata)

    # Build combined table data
    table_data = []
    for i in range(len(img_diag)):
        img_rank = img_diag[i]["true_rank"]
        lm_rank = lm_diag[i]["true_rank"]
        if np.isfinite(img_rank) and np.isfinite(lm_rank):
            winner = "image" if img_rank < lm_rank else ("landmark" if lm_rank < img_rank else "tie")
        else:
            winner = "N/A"
        table_data.append({
            "pano_idx": i,
            "pano_id": pano_metadata.iloc[i]["pano_id"],
            "img_rank": int(img_rank) if np.isfinite(img_rank) else None,
            "lm_rank": int(lm_rank) if np.isfinite(lm_rank) else None,
            "img_entropy": round(img_diag[i]["entropy"], 3) if np.isfinite(img_diag[i]["entropy"]) else None,
            "lm_entropy": round(lm_diag[i]["entropy"], 3) if np.isfinite(lm_diag[i]["entropy"]) else None,
            "img_peak_sharp": round(img_diag[i]["peak_sharpness"], 4) if np.isfinite(img_diag[i]["peak_sharpness"]) else None,
            "lm_peak_sharp": round(lm_diag[i]["peak_sharpness"], 4) if np.isfinite(lm_diag[i]["peak_sharpness"]) else None,
            "img_max_sim": round(img_diag[i]["max_sim"], 4) if np.isfinite(img_diag[i]["max_sim"]) else None,
            "lm_max_sim": round(lm_diag[i]["max_sim"], 4) if np.isfinite(lm_diag[i]["max_sim"]) else None,
            "winner": winner,
        })

    # Filter to panoramas with valid data from both sources for the scatter
    scatter_data = [d for d in table_data if d["img_rank"] is not None and d["lm_rank"] is not None]
    print(f"Valid panoramas (both sources): {len(scatter_data)}")
    scatter_img_ranks = [d["img_rank"] for d in scatter_data]
    scatter_lm_ranks = [d["lm_rank"] for d in scatter_data]
    scatter_winners = [d["winner"] for d in scatter_data]
    scatter_pano_idxs = [d["pano_idx"] for d in scatter_data]

    # Image paths (stored as PosixPath in metadata)
    sat_paths = [str(p) for p in sat_metadata["path"].tolist()]
    pano_paths = [str(p) for p in pano_metadata["path"].tolist()]

    # Server-side state
    cache = {"selected_pano_idx": None}

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Fusion Diagnostics: Image vs Landmark Similarity"),

        # Scatter plot
        html.Div([
            html.H3("Scatter: Image Rank vs Landmark Rank (click to inspect)"),
            dcc.Graph(id='scatter-plot', style={'height': '60vh'}),
        ]),

        # Data table
        html.Div([
            html.H3("All Panoramas (sortable/filterable)"),
            dash_table.DataTable(
                id='pano-table',
                columns=[
                    {'name': 'Pano Idx', 'id': 'pano_idx', 'type': 'numeric'},
                    {'name': 'Pano ID', 'id': 'pano_id', 'type': 'text'},
                    {'name': 'Img Rank', 'id': 'img_rank', 'type': 'numeric'},
                    {'name': 'Lm Rank', 'id': 'lm_rank', 'type': 'numeric'},
                    {'name': 'Img Entropy', 'id': 'img_entropy', 'type': 'numeric'},
                    {'name': 'Lm Entropy', 'id': 'lm_entropy', 'type': 'numeric'},
                    {'name': 'Img Peak Sharp', 'id': 'img_peak_sharp', 'type': 'numeric'},
                    {'name': 'Lm Peak Sharp', 'id': 'lm_peak_sharp', 'type': 'numeric'},
                    {'name': 'Img Max Sim', 'id': 'img_max_sim', 'type': 'numeric'},
                    {'name': 'Lm Max Sim', 'id': 'lm_max_sim', 'type': 'numeric'},
                    {'name': 'Winner', 'id': 'winner', 'type': 'text'},
                ],
                data=table_data,
                row_selectable='single',
                selected_rows=[],
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px', 'fontSize': '12px'},
                style_header={'fontWeight': 'bold'},
                sort_action='native',
                filter_action='native',
                page_size=50,
            ),
        ], style={'marginBottom': '20px'}),

        # Inspection panel
        html.Div([
            html.H3("Inspection Panel"),
            html.Div(id='inspection-info', style={'marginBottom': '10px'}),

            # Panorama image + landmarks side by side
            html.Div([
                html.Div([
                    html.H4("Panorama Image"),
                    html.Img(id='pano-image', style={'maxWidth': '100%', 'height': 'auto'}),
                ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.H4("Panorama Landmarks (Gemini)"),
                    html.Div(id='pano-landmarks'),
                ], style={'width': '38%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'marginLeft': '2%'}),
            ], style={'marginBottom': '20px'}),

            # OSM display toggle
            dcc.Checklist(
                id='osm-show-all',
                options=[{'label': ' Show all OSM tags (unfiltered)', 'value': 'all'}],
                value=[],
                style={'marginBottom': '10px'},
            ),

            # Top-5 satellite patches side by side
            html.Div([
                html.Div([
                    html.H4("Top-5 Satellite Patches (Image Source)"),
                    html.Div(id='img-top5-patches'),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.H4("Top-5 Satellite Patches (Landmark Source)"),
                    html.Div(id='lm-top5-patches'),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top',
                          'marginLeft': '2%'}),
            ]),

            # Similarity distributions
            html.Div([
                html.H4("Similarity Distributions"),
                dcc.Graph(id='sim-distribution-plot', style={'height': '40vh'}),
            ]),
        ], id='inspection-panel'),

        dcc.Store(id='selected-pano-store'),
    ])

    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('scatter-plot', 'id')
    )
    def build_scatter(_):
        fig = go.Figure()
        color_map = {"image": "blue", "landmark": "orange", "tie": "gray"}
        for winner_val, color in color_map.items():
            mask = [i for i, w in enumerate(scatter_winners) if w == winner_val]
            if not mask:
                continue
            fig.add_trace(go.Scattergl(
                x=[scatter_img_ranks[i] for i in mask],
                y=[scatter_lm_ranks[i] for i in mask],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.5),
                name=winner_val,
                customdata=[scatter_pano_idxs[i] for i in mask],
                text=[f"pano_idx={scatter_pano_idxs[i]}<br>img_rank={scatter_img_ranks[i]}<br>lm_rank={scatter_lm_ranks[i]}"
                      for i in mask],
                hovertemplate="%{text}<extra></extra>",
            ))

        max_rank = max(max(scatter_img_ranks, default=1), max(scatter_lm_ranks, default=1))
        fig.add_trace(go.Scattergl(
            x=[1, max_rank], y=[1, max_rank],
            mode='lines', line=dict(dash='dash', color='black', width=1),
            showlegend=False,
        ))
        fig.update_layout(
            xaxis=dict(title="Image Rank (lower=better)", type="log"),
            yaxis=dict(title="Landmark Rank (lower=better)", type="log"),
            clickmode='event+select',
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    @app.callback(
        Output('selected-pano-store', 'data'),
        [Input('scatter-plot', 'clickData'),
         Input('pano-table', 'selected_rows')],
        State('pano-table', 'data'),
    )
    def select_pano(click_data, selected_rows, current_table_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger == 'scatter-plot' and click_data:
            point = click_data['points'][0]
            pano_idx = point.get('customdata')
            if pano_idx is not None:
                cache['selected_pano_idx'] = pano_idx
                return pano_idx
        if trigger == 'pano-table' and selected_rows and current_table_data:
            row = current_table_data[selected_rows[0]]
            pano_idx = row['pano_idx']
            cache['selected_pano_idx'] = pano_idx
            return pano_idx
        return cache.get('selected_pano_idx')

    @app.callback(
        [Output('inspection-info', 'children'),
         Output('pano-image', 'src'),
         Output('pano-landmarks', 'children'),
         Output('img-top5-patches', 'children'),
         Output('lm-top5-patches', 'children'),
         Output('sim-distribution-plot', 'figure')],
        [Input('selected-pano-store', 'data'),
         Input('osm-show-all', 'value')],
    )
    def update_inspection(pano_idx, osm_show_all):
        show_all_osm = 'all' in (osm_show_all or [])
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Select a panorama to inspect")

        if pano_idx is None:
            return "Click on a point or select a row to inspect", "", [], [], [], empty_fig

        pano_idx = int(pano_idx)
        pano_row = pano_metadata.iloc[pano_idx]
        pano_id = pano_row["pano_id"]
        true_idxs = list(pano_row["positive_satellite_idxs"]) + list(pano_row["semipositive_satellite_idxs"])

        # Info text
        img_r = img_diag[pano_idx]["true_rank"]
        lm_r = lm_diag[pano_idx]["true_rank"]
        pano_lat = pano_row["lat"]
        pano_lon = pano_row["lon"]
        gmaps_url = f"https://www.google.com/maps/@{pano_lat},{pano_lon},18z"
        info_parts = [
            html.P([
                f"Panorama: {pano_id} (idx={pano_idx}) ",
                html.A("Google Maps", href=gmaps_url, target="_blank",
                       style={'color': '#1a73e8', 'textDecoration': 'underline'}),
            ]),
            html.P(f"Image Rank: {int(img_r)}, Landmark Rank: {int(lm_r)}"
                   if np.isfinite(img_r) and np.isfinite(lm_r) else "Rank: N/A"),
            html.P(f"True satellite patches: {true_idxs}"),
            html.P(f"Image entropy: {img_diag[pano_idx]['entropy']:.3f}, "
                   f"Landmark entropy: {lm_diag[pano_idx]['entropy']:.3f}"
                   if np.isfinite(img_diag[pano_idx]['entropy']) else "Entropy: N/A"),
        ]
        info = html.Div(info_parts)

        # Panorama image
        pano_src = load_image_as_base64(pano_paths[pano_idx], max_width=800)

        # Gemini panorama landmarks
        gemini_data = pano_gemini_landmarks.get(pano_id, {})
        landmark_children = []
        if gemini_data:
            loc_type = gemini_data.get("location_type", "unknown")
            landmark_children.append(
                html.P(f"Location type: {loc_type}",
                       style={'fontWeight': 'bold', 'marginBottom': '5px'}))
            for lm in gemini_data.get("landmarks", []):
                desc = lm.get("description", "")
                proper_nouns = lm.get("proper_nouns", [])
                noun_str = f" [{', '.join(proper_nouns)}]" if proper_nouns else ""
                landmark_children.append(
                    html.Div([
                        html.P(f"{desc}{noun_str}",
                               style={'fontSize': '11px', 'margin': '2px 0',
                                      'padding': '3px',
                                      'backgroundColor': '#f0f0f0' if proper_nouns else '#fff',
                                      'borderLeft': '3px solid #e67e22' if proper_nouns else '3px solid #ccc'}),
                    ]))
        else:
            landmark_children.append(html.P("No Gemini landmarks available", style={'color': 'gray'}))

        # Top-5 patches for each source
        def make_patch_display(sim_row, source_name):
            top_idxs, top_sims = get_top_k_patches(sim_row, k=5)
            children = []
            for rank_pos, (sat_idx, sim_val) in enumerate(zip(top_idxs, top_sims)):
                is_true = sat_idx in true_idxs
                border_color = "green" if is_true else "red"
                sat_src = load_image_as_base64(sat_paths[sat_idx], max_width=200)

                # OSM tags for this sat patch (semantically deduplicated)
                osm_tags = sat_osm_mapping.get(sat_idx, [])
                seen_keys = set()
                osm_strs = []
                for p in osm_tags:
                    sk = _osm_semantic_key(p)
                    if sk in seen_keys:
                        continue
                    seen_keys.add(sk)
                    s = format_osm_props(p, show_all=show_all_osm)
                    if s:
                        osm_strs.append(s)
                osm_display = [html.P(s, style={'fontSize': '9px', 'margin': '0', 'color': '#555'})
                               for s in osm_strs]

                patch_content = [
                    html.Img(src=sat_src, style={
                        'width': '180px', 'height': '180px', 'objectFit': 'cover',
                        'border': f'3px solid {border_color}', 'marginRight': '5px',
                    }) if sat_src else html.Div("Image not found",
                                                style={'width': '180px', 'height': '180px',
                                                       'border': f'3px solid {border_color}',
                                                       'display': 'flex', 'alignItems': 'center',
                                                       'justifyContent': 'center', 'color': 'gray'}),
                    html.P(f"#{rank_pos+1} idx={sat_idx} sim={sim_val:.4f}"
                           + (" [TRUE]" if is_true else ""),
                           style={'fontSize': '11px', 'margin': '2px', 'fontWeight': 'bold' if is_true else 'normal'}),
                ] + osm_display

                children.append(html.Div(patch_content,
                    style={'display': 'inline-block', 'verticalAlign': 'top',
                           'marginBottom': '10px', 'marginRight': '8px', 'maxWidth': '190px'}))
            return children

        img_patches = make_patch_display(img_sim[pano_idx], "Image")
        lm_patches = make_patch_display(lm_sim[pano_idx], "Landmark")

        # Side-by-side similarity distributions
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Image Similarity (sorted)", "Landmark Similarity (sorted)"])

        for col, (sim_row, name) in enumerate([(img_sim[pano_idx], "Image"),
                                                 (lm_sim[pano_idx], "Landmark")], 1):
            finite_mask = torch.isfinite(sim_row)
            finite_sims = sim_row[finite_mask]
            sorted_sims = torch.sort(finite_sims, descending=True).values.numpy()

            show_n = min(200, len(sorted_sims))
            fig.add_trace(go.Bar(
                x=list(range(show_n)),
                y=sorted_sims[:show_n].tolist(),
                name=f"{name} top-{show_n}",
                marker_color="blue" if name == "Image" else "orange",
            ), row=1, col=col)

            if true_idxs:
                true_sims_vals = sim_row[true_idxs]
                finite_true = true_sims_vals[torch.isfinite(true_sims_vals)]
                if len(finite_true) > 0:
                    best_true = finite_true.max().item()
                    fig.add_hline(y=best_true, line_dash="dash", line_color="green",
                                  annotation_text="best true", row=1, col=col)

        fig.update_layout(height=350, showlegend=True)

        return info, pano_src, landmark_children, img_patches, lm_patches, fig

    print(f"Starting server at http://localhost:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
