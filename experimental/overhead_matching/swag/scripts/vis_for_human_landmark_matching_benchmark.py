#!/usr/bin/env python3
"""
Landmark-based cross-view geolocalization visualizer.

Allows exploring panoramas and satellite patches with semantic landmark filtering.
Supports two modes:
  - Panorama mode: Select a panorama, filter satellite patches by landmarks
  - Satellite mode: Select a satellite patch, filter panoramas by landmarks

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:landmark_visualizer -- \
        --pickle-path /tmp/visualizer_information.pkl \
        --panorama-dir /data/overhead_matching/datasets/VIGOR/Chicago/panorama \
        --port 5000
"""

import argparse
import pickle
from pathlib import Path
from flask import Flask, render_template_string, send_file, jsonify, request
import numpy as np
from typing import Optional

# Torch modules will be imported lazily when needed
TORCH_AVAILABLE = None  # Will be set on first import attempt

# Global data
VISUALIZER_DATA: Optional[dict] = None
PANORAMA_PATH_MAP: dict[str, Path] = {}
SIMILARITY_MATRIX: Optional[np.ndarray] = None
PANORAMA_DIR: Optional[Path] = None

# Model inference state (lazy loaded)
MODELS_LOADED = False
PANO_MODEL = None
SAT_MODEL = None
OPENAI_CLIENT = None
PANO_EMBEDDINGS = None  # Cached embeddings for all panoramas
SAT_EMBEDDINGS = None  # Cached embeddings for all satellites
EDITED_STATE = {}  # {mode_idx: {sentences, custom_embedding, similarities}}

app = Flask(__name__)


def load_visualizer_data(pickle_path: Path) -> dict:
    """Load and process the visualizer data from pickle file."""
    global PANO_EMBEDDINGS, SAT_EMBEDDINGS

    print(f"Loading data from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['panorama_id'])} panoramas and {len(data['sat_loc'])} satellites")
    print(f"Similarity matrix has {len(data['similarity_matrix'])} elements")

    # Reshape similarity matrix
    n_pano = len(data['panorama_id'])
    n_sat = len(data['sat_loc'])
    expected_size = n_pano * n_sat

    if len(data['similarity_matrix']) != expected_size:
        raise ValueError(f"Similarity matrix size mismatch: {len(data['similarity_matrix'])} != {expected_size}")

    print(f"Reshaping similarity matrix to ({n_pano}, {n_sat})...")
    data['similarity_matrix'] = np.array(data['similarity_matrix'], dtype=np.float32).reshape(n_pano, n_sat)

    # Load pre-computed embeddings if available
    if 'panorama_embeddings' in data and 'sat_embeddings' in data:
        print("Loading pre-computed embeddings...")
        # Convert to numpy arrays for now, will convert to torch when models are loaded
        data['panorama_embeddings_array'] = np.array(data['panorama_embeddings'], dtype=np.float32)
        data['sat_embeddings_array'] = np.array(data['sat_embeddings'], dtype=np.float32)
        print(f"  Panorama embeddings: {data['panorama_embeddings_array'].shape}")
        print(f"  Satellite embeddings: {data['sat_embeddings_array'].shape}")
    else:
        print("Warning: Pre-computed embeddings not found in pickle file.")
        print("Landmark editing will require computing embeddings on-demand.")
        data['panorama_embeddings_array'] = None
        data['sat_embeddings_array'] = None

    print("Data loaded successfully!")
    return data


def build_panorama_path_map(panorama_dir: Path, panorama_ids: list[str]) -> dict[str, Path]:
    """Build mapping from panorama ID to file path."""
    print(f"Building panorama path map from {panorama_dir}...")

    # Scan directory for all panorama files
    all_files = list(panorama_dir.glob("*.jpg")) + list(panorama_dir.glob("*.png"))
    print(f"Found {len(all_files)} panorama image files")

    # Build map: pano_id -> file_path
    path_map = {}
    for file_path in all_files:
        # Extract pano_id from filename (format: {pano_id},{lat},{lon},.jpg)
        filename = file_path.stem  # Remove extension
        pano_id = filename.split(',')[0]
        path_map[pano_id] = file_path

    # Check coverage
    missing = [pid for pid in panorama_ids if pid not in path_map]
    if missing:
        print(f"Warning: {len(missing)} panorama IDs not found in image directory")
        if len(missing) <= 10:
            print(f"Missing IDs: {missing}")

    print(f"Mapped {len(path_map)} panorama IDs to image files")
    return path_map


def safe_eval_filter(expression: str, sentences_list: list[list[str]], mode: str) -> list[int]:
    """
    Safely evaluate Python expression on landmark sentences.

    Args:
        expression: Python expression (e.g., '("United Center" in x) and ("tree" in x.lower())')
        sentences_list: List of sentence lists for each item
        mode: 'panorama' or 'satellite' (for error messages)

    Returns:
        List of indices that match the filter
    """
    # Restricted namespace - only safe builtins
    safe_globals = {
        '__builtins__': {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        }
    }

    filtered_indices = []
    errors = []

    for idx, sentences in enumerate(sentences_list):
        x = '\n'.join(sentences)
        try:
            # Evaluate expression with restricted namespace
            result = eval(expression, safe_globals, {'x': x})
            if result:
                filtered_indices.append(idx)
        except Exception as e:
            # Collect errors but don't stop processing
            if len(errors) < 5:  # Only store first 5 errors
                errors.append(str(e))

    if errors:
        print(f"Filter evaluation errors (showing first {len(errors)}): {errors}")

    return filtered_indices


# Flask routes

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/init')
def get_init_data():
    """Get initial metadata about the dataset."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    return jsonify({
        'n_panoramas': len(VISUALIZER_DATA['panorama_id']),
        'n_satellites': len(VISUALIZER_DATA['sat_loc']),
        'has_data': True
    })


@app.route('/api/panorama/<int:pano_idx>')
def get_panorama_data(pano_idx: int):
    """Get data for a specific panorama."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    if pano_idx < 0 or pano_idx >= len(VISUALIZER_DATA['panorama_id']):
        return jsonify({'error': 'Invalid panorama index'}), 400

    # Get similarities for all satellites
    similarities = SIMILARITY_MATRIX[pano_idx, :].tolist()

    # Get positive satellite indices (convert numpy int64 to Python int)
    positive_sat_indices = [int(x) for x in VISUALIZER_DATA['pano_to_positive_sat_index_map'][pano_idx]]

    # Get location (convert numpy float64 to Python float)
    loc = VISUALIZER_DATA['panorama_loc'][pano_idx]
    loc_native = [float(loc[0]), float(loc[1])]

    pano_id = VISUALIZER_DATA['panorama_id'][pano_idx]
    has_image = pano_id in PANORAMA_PATH_MAP
    image_filename = None
    if has_image:
        image_filename = PANORAMA_PATH_MAP[pano_id].name

    return jsonify({
        'pano_id': pano_id,
        'loc': loc_native,
        'sentences': VISUALIZER_DATA['panorama_sentences'][pano_idx],
        'similarities': similarities,
        'positive_indices': positive_sat_indices,
        'has_image': has_image,
        'image_filename': image_filename
    })


@app.route('/api/satellite/<int:sat_idx>')
def get_satellite_data(sat_idx: int):
    """Get data for a specific satellite patch."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    if sat_idx < 0 or sat_idx >= len(VISUALIZER_DATA['sat_loc']):
        return jsonify({'error': 'Invalid satellite index'}), 400

    # Get similarities for all panoramas
    similarities = SIMILARITY_MATRIX[:, sat_idx].tolist()

    # Find which panoramas have this satellite as a positive match (reverse lookup)
    positive_pano_indices = []
    for pano_idx, pos_sats in enumerate(VISUALIZER_DATA['pano_to_positive_sat_index_map']):
        if sat_idx in pos_sats:
            positive_pano_indices.append(pano_idx)

    # Get location (convert numpy float64 to Python float)
    loc = VISUALIZER_DATA['sat_loc'][sat_idx]
    loc_native = [float(loc[0]), float(loc[1])]

    return jsonify({
        'sat_idx': sat_idx,
        'loc': loc_native,
        'sentences': VISUALIZER_DATA['sat_sentences'][sat_idx],
        'similarities': similarities,
        'positive_indices': positive_pano_indices
    })


@app.route('/api/filter', methods=['POST'])
def filter_landmarks():
    """Apply Python expression filter to landmarks."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.json
    expression = data.get('expression', '')
    mode = data.get('mode', 'panorama')

    if not expression.strip():
        # Empty filter - return all indices
        if mode == 'panorama':
            n_items = len(VISUALIZER_DATA['sat_loc'])
        else:
            n_items = len(VISUALIZER_DATA['panorama_id'])
        return jsonify({'filtered_indices': list(range(n_items))})

    try:
        # Select appropriate sentences list based on mode
        if mode == 'panorama':
            sentences_list = VISUALIZER_DATA['sat_sentences']
        else:
            sentences_list = VISUALIZER_DATA['panorama_sentences']

        filtered_indices = safe_eval_filter(expression, sentences_list, mode)

        return jsonify({
            'filtered_indices': filtered_indices,
            'count': len(filtered_indices)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/image/panorama/<int:pano_idx>')
def serve_panorama_image(pano_idx: int):
    """Return the direct file path for panorama image."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    if pano_idx < 0 or pano_idx >= len(VISUALIZER_DATA['panorama_id']):
        return jsonify({'error': 'Invalid panorama index'}), 400

    pano_id = VISUALIZER_DATA['panorama_id'][pano_idx]

    if pano_id not in PANORAMA_PATH_MAP:
        return jsonify({'error': 'Image not found'}), 404

    # Return the absolute file path for direct access
    image_path = str(PANORAMA_PATH_MAP[pano_id])
    return jsonify({'image_path': image_path})


@app.route('/static/panorama/<path:filename>')
def serve_panorama_static(filename: str):
    """Serve panorama images as static files."""
    if PANORAMA_DIR is None:
        return jsonify({'error': 'Panorama directory not configured'}), 500

    return send_file(PANORAMA_DIR / filename, mimetype='image/jpeg')


@app.route('/api/find_panorama', methods=['POST'])
def find_panorama():
    """Find panorama index by panorama ID."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.json
    pano_id = data.get('pano_id', '').strip()

    if not pano_id:
        return jsonify({'error': 'No panorama ID provided'}), 400

    # Search for the panorama ID
    try:
        idx = VISUALIZER_DATA['panorama_id'].index(pano_id)
        return jsonify({'index': idx})
    except ValueError:
        return jsonify({'error': f'Panorama ID not found: {pano_id}'}), 404


@app.route('/api/batch_items', methods=['POST'])
def get_batch_items():
    """Get data for multiple items at once (for results list)."""
    if VISUALIZER_DATA is None:
        return jsonify({'error': 'Data not loaded'}), 500

    data = request.json
    indices = data.get('indices', [])
    mode = data.get('mode', 'panorama')

    if not indices:
        return jsonify({'items': []})

    # No limit - allow fetching all requested items
    # (Map may request thousands of items)

    items = []
    if mode == 'panorama':
        # Fetching satellite data
        for idx in indices:
            if 0 <= idx < len(VISUALIZER_DATA['sat_loc']):
                loc = VISUALIZER_DATA['sat_loc'][idx]
                items.append({
                    'idx': idx,
                    'loc': [float(loc[0]), float(loc[1])],
                    'sentences': VISUALIZER_DATA['sat_sentences'][idx]
                })
    else:
        # Fetching panorama data (for satellite mode)
        for idx in indices:
            if 0 <= idx < len(VISUALIZER_DATA['panorama_id']):
                loc = VISUALIZER_DATA['panorama_loc'][idx]
                items.append({
                    'idx': idx,
                    'pano_id': VISUALIZER_DATA['panorama_id'][idx],
                    'loc': [float(loc[0]), float(loc[1])],
                    'sentences': VISUALIZER_DATA['panorama_sentences'][idx]
                })

    return jsonify({'items': items})


# ============================================================================
# Landmark Editing API Endpoints
# ============================================================================

@app.route('/api/edit/landmarks', methods=['POST'])
def api_edit_landmarks():
    """Save edited landmarks to session state."""
    data = request.json
    item_idx = data.get('item_idx')
    mode = data.get('mode')  # 'panorama' or 'satellite'
    sentences = data.get('sentences', [])

    if item_idx is None or mode is None:
        return jsonify({'error': 'Missing item_idx or mode'}), 400

    # Store edited landmarks in global state
    state_key = f"{mode}_{item_idx}"
    EDITED_STATE[state_key] = {
        'sentences': sentences,
        'mode': mode,
        'item_idx': item_idx
    }

    return jsonify({'success': True})


@app.route('/api/edit/recompute', methods=['POST'])
def api_edit_recompute():
    """Recompute similarities using edited landmarks."""
    data = request.json
    item_idx = data.get('item_idx')
    mode = data.get('mode')

    if item_idx is None or mode is None:
        return jsonify({'error': 'Missing item_idx or mode'}), 400

    state_key = f"{mode}_{item_idx}"
    if state_key not in EDITED_STATE:
        return jsonify({'error': 'No edited landmarks found'}), 400

    try:
        # Get custom sentences
        custom_sentences = EDITED_STATE[state_key]['sentences']

        # Compute new similarities
        similarities = compute_similarity_with_custom_landmarks(
            item_idx, mode, custom_sentences
        )

        # Store in state for retrieval
        EDITED_STATE[state_key]['similarities'] = similarities

        return jsonify({
            'success': True,
            'similarities': similarities
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error computing similarities: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/edit/reset', methods=['POST'])
def api_edit_reset():
    """Clear edits and restore original data."""
    data = request.json
    item_idx = data.get('item_idx')
    mode = data.get('mode')

    if item_idx is None or mode is None:
        return jsonify({'error': 'Missing item_idx or mode'}), 400

    state_key = f"{mode}_{item_idx}"
    if state_key in EDITED_STATE:
        del EDITED_STATE[state_key]

    return jsonify({'success': True})


@app.route('/api/edit/status', methods=['GET'])
def api_edit_status():
    """Check if current item has edits."""
    item_idx = request.args.get('item_idx', type=int)
    mode = request.args.get('mode')

    if item_idx is None or mode is None:
        return jsonify({'error': 'Missing item_idx or mode'}), 400

    state_key = f"{mode}_{item_idx}"
    has_edits = state_key in EDITED_STATE
    has_similarities = has_edits and 'similarities' in EDITED_STATE.get(state_key, {})

    response = {
        'has_edits': has_edits,
        'has_similarities': has_similarities
    }

    if has_edits:
        response['sentences'] = EDITED_STATE[state_key]['sentences']
    if has_similarities:
        response['similarities'] = EDITED_STATE[state_key]['similarities']

    return jsonify(response)


@app.route('/api/test/validate_embeddings', methods=['POST'])
def api_test_validate_embeddings():
    """
    Test endpoint to validate that recomputing embeddings for unedited landmarks
    produces the same similarities as the cached data.
    """
    data = request.json
    num_test_items = data.get('num_test_items', 5)
    mode = data.get('mode', 'panorama')

    try:
        # Ensure models are loaded
        load_models_lazy()

        torch = globals()['torch']

        results = []

        # Test a few items
        if mode == 'panorama':
            test_indices = list(range(min(num_test_items, len(VISUALIZER_DATA['panorama_id']))))
            num_opposite = len(VISUALIZER_DATA['sat_loc'])
        else:
            test_indices = list(range(min(num_test_items, len(VISUALIZER_DATA['sat_loc']))))
            num_opposite = len(VISUALIZER_DATA['panorama_id'])

        for idx in test_indices:
            # Get original sentences
            if mode == 'panorama':
                original_sentences = VISUALIZER_DATA['panorama_sentences'][idx]
                # Get cached similarities from similarity matrix
                # Matrix shape is (npano, nsat), so extract row idx
                cached_similarities = SIMILARITY_MATRIX[idx, :].tolist()
            else:
                original_sentences = VISUALIZER_DATA['sat_sentences'][idx]
                # For satellite mode, similarities are in columns
                # Matrix shape is (npano, nsat), so extract column idx
                cached_similarities = SIMILARITY_MATRIX[:, idx].tolist()

            # Recompute similarities using our implementation
            recomputed_similarities = compute_similarity_with_custom_landmarks(
                idx, mode, original_sentences
            )

            # Compare
            cached_tensor = torch.tensor(cached_similarities, dtype=torch.float32)
            recomputed_tensor = torch.tensor(recomputed_similarities, dtype=torch.float32)

            max_diff = (cached_tensor - recomputed_tensor).abs().max().item()
            mean_diff = (cached_tensor - recomputed_tensor).abs().mean().item()
            correlation = torch.corrcoef(torch.stack([cached_tensor, recomputed_tensor]))[0, 1].item()

            results.append({
                'idx': idx,
                'num_sentences': len(original_sentences),
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'correlation': float(correlation),
                'cached_min': float(cached_tensor.min()),
                'cached_max': float(cached_tensor.max()),
                'cached_mean': float(cached_tensor.mean()),
                'recomputed_min': float(recomputed_tensor.min()),
                'recomputed_max': float(recomputed_tensor.max()),
                'recomputed_mean': float(recomputed_tensor.mean()),
            })

        return jsonify({
            'success': True,
            'mode': mode,
            'num_tested': len(results),
            'results': results
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Validation error: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Landmark Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f0f2f5;
            padding: 20px;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
        }

        .header h1 {
            font-size: 24px;
            margin-bottom: 10px;
        }

        .mode-toggle {
            display: inline-flex;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 4px;
            margin-top: 10px;
        }

        .mode-btn {
            padding: 8px 20px;
            border: none;
            background: transparent;
            color: white;
            cursor: pointer;
            border-radius: 6px;
            font-size: 14px;
            transition: background 0.2s;
        }

        .mode-btn.active {
            background: rgba(255,255,255,0.3);
            font-weight: 600;
        }

        .controls {
            padding: 20px 30px;
            background: #fafbfc;
            border-bottom: 1px solid #e1e4e8;
        }

        .nav-controls {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }

        .nav-btn {
            padding: 8px 16px;
            background: white;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }

        .nav-btn:hover:not(:disabled) {
            background: #f3f4f6;
            border-color: #9ca3af;
        }

        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .current-info {
            flex: 1;
            font-size: 16px;
            font-weight: 500;
            color: #374151;
        }

        .filter-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .filter-label {
            font-size: 14px;
            font-weight: 500;
            color: #374151;
        }

        .filter-input {
            flex: 1;
            padding: 10px 15px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            font-family: 'Monaco', 'Courier New', monospace;
        }

        .filter-btn {
            padding: 10px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .filter-btn:hover {
            background: #5568d3;
        }

        .filter-count {
            font-size: 14px;
            color: #6b7280;
            padding: 10px 0;
        }

        .filter-count strong {
            color: #374151;
        }

        .main-content {
            display: grid;
            grid-template-columns: 500px 1fr;
            gap: 0;
            min-height: calc(100vh - 280px);
        }

        .left-panel {
            border-right: 1px solid #e1e4e8;
            padding: 20px;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .panorama-image-container {
            margin-bottom: 20px;
        }

        .panorama-image {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
        }

        .no-image {
            padding: 60px 20px;
            text-align: center;
            background: #f9fafb;
            border: 1px dashed #d1d5db;
            border-radius: 8px;
            color: #6b7280;
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 12px;
        }

        .landmarks-list {
            list-style: none;
        }

        .landmark-item {
            padding: 8px 12px;
            margin-bottom: 6px;
            background: #f9fafb;
            border-left: 3px solid #667eea;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.5;
            color: #374151;
        }

        .landmark-edit-row {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
            align-items: flex-start;
        }

        .landmark-edit-input {
            flex: 1;
            padding: 8px 12px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 4px;
            font-size: 14px;
            line-height: 1.5;
            font-family: inherit;
            resize: vertical;
            min-height: 42px;
        }

        .landmark-edit-input:focus {
            outline: none;
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .landmark-remove-btn {
            padding: 6px 10px;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            line-height: 1;
            transition: background 0.2s;
        }

        .landmark-remove-btn:hover {
            background: #dc2626;
        }

        .landmark-add-btn {
            padding: 8px 16px;
            background: #10b981;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: background 0.2s;
        }

        .landmark-add-btn:hover {
            background: #059669;
        }

        .ground-truth-card {
            background: #fefce8;
            border: 2px solid #facc15;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
        }

        .ground-truth-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid #fde047;
        }

        .ground-truth-title {
            font-weight: 600;
            color: #854d0e;
            font-size: 13px;
        }

        .ground-truth-links {
            display: flex;
            gap: 8px;
        }

        .map-link {
            padding: 3px 8px;
            background: #fbbf24;
            color: #78350f;
            text-decoration: none;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }

        .map-link:hover {
            background: #f59e0b;
        }

        .ground-truth-landmarks {
            font-size: 13px;
        }

        .ground-truth-landmarks .landmark-item {
            background: #fffbeb;
            border-left-color: #fbbf24;
        }

        .item-nav-link {
            margin-left: auto;
            padding: 4px 12px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }

        .item-nav-link:hover {
            background: #5568d3;
        }

        .histogram-container {
            padding: 20px;
            border-bottom: 1px solid #e1e4e8;
            background: white;
        }

        #histogram {
            width: 100%;
            height: 300px;
        }

        .results-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e1e4e8;
        }

        .results-title {
            font-size: 16px;
            font-weight: 600;
            color: #111827;
        }

        .sim-range-info {
            font-size: 13px;
            color: #6b7280;
        }

        .item-card {
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            transition: all 0.2s;
        }

        .item-card:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102,126,234,0.15);
        }

        .item-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .sim-badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 13px;
        }

        .sim-badge.high {
            background: #dcfce7;
            color: #166534;
        }

        .sim-badge.medium {
            background: #fef9c3;
            color: #854d0e;
        }

        .sim-badge.low {
            background: #fee2e2;
            color: #991b1b;
        }

        .item-index {
            font-size: 13px;
            color: #6b7280;
        }

        .maps-link {
            margin-left: auto;
            padding: 4px 12px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .maps-link:hover {
            background: #5568d3;
        }

        .item-landmarks {
            font-size: 13px;
            line-height: 1.6;
            color: #374151;
        }

        .item-landmark {
            padding: 4px 0;
            padding-left: 16px;
            position: relative;
        }

        .item-landmark:before {
            content: "•";
            position: absolute;
            left: 4px;
            color: #667eea;
            font-weight: bold;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }

        .error {
            background: #fee2e2;
            color: #991b1b;
            padding: 12px 20px;
            border-radius: 6px;
            margin: 20px;
        }

        .true-match-badge {
            background: #fecaca;
            color: #991b1b;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Landmark-Based Geolocalization Visualizer</h1>
            <div class="mode-toggle">
                <button class="mode-btn active" onclick="switchMode('panorama')" id="pano-mode-btn">
                    📷 Panorama Mode
                </button>
                <button class="mode-btn" onclick="switchMode('satellite')" id="sat-mode-btn">
                    🛰️ Satellite Mode
                </button>
            </div>
        </div>

        <div class="controls">
            <div class="nav-controls">
                <button class="nav-btn" onclick="navigate(-1)" id="prev-btn">← Previous</button>
                <button class="nav-btn" onclick="navigate(1)" id="next-btn">Next →</button>
                <div class="current-info" id="current-info">
                    Loading...
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <input
                        type="text"
                        id="seek-input"
                        placeholder="Index or Pano ID"
                        style="padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 13px; width: 150px;"
                        onkeypress="if(event.key === 'Enter') seekToItem()"
                    >
                    <button class="nav-btn" onclick="seekToItem()">Go</button>
                </div>
            </div>

            <div class="filter-controls">
                <span class="filter-label">Python Filter:</span>
                <input
                    type="text"
                    class="filter-input"
                    id="filter-input"
                    placeholder='e.g., ("United Center" in x) and ("tree" in x.lower())'
                    onkeypress="if(event.key === 'Enter') applyFilter()"
                >
                <button class="filter-btn" onclick="applyFilter()">🔍 Filter</button>
            </div>

            <div class="filter-count" id="filter-count">
                No filter applied
            </div>
        </div>

        <div class="main-content">
            <div class="left-panel">
                <div id="panorama-section" style="display: block;">
                    <div class="panorama-image-container" id="image-container">
                        <div class="no-image">No image available</div>
                    </div>
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div class="section-title" style="margin-bottom: 0;">Landmarks</div>
                    <div style="display: flex; gap: 12px;">
                        <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="show-ground-truth" onchange="toggleGroundTruth()" style="cursor: pointer;">
                            <span>Show Ground Truth</span>
                        </label>
                        <label style="display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer;">
                            <input type="checkbox" id="show-map" onchange="toggleMap()" style="cursor: pointer;">
                            <span>Show Map</span>
                        </label>
                    </div>
                </div>

                <!-- Edit controls -->
                <div style="display: flex; gap: 8px; margin-bottom: 12px; align-items: center;">
                    <button id="edit-landmarks-btn" onclick="toggleEditMode()"
                            style="padding: 6px 12px; font-size: 13px; border: 1px solid #d1d5db; background: white; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
                        ✏️ Edit
                    </button>
                    <button id="recompute-btn" onclick="recomputeSimilarities()"
                            style="padding: 6px 12px; font-size: 13px; border: 1px solid #3b82f6; background: #3b82f6; color: white; border-radius: 4px; cursor: pointer; display: none; align-items: center; gap: 4px;">
                        🔄 Recompute
                    </button>
                    <button id="reset-btn" onclick="resetEdits()"
                            style="padding: 6px 12px; font-size: 13px; border: 1px solid #ef4444; background: #ef4444; color: white; border-radius: 4px; cursor: pointer; display: none; align-items: center; gap: 4px;">
                        ↩️ Reset
                    </button>
                    <div id="edit-status" style="font-size: 12px; color: #6b7280; margin-left: auto; display: none;">
                        <span id="edit-status-text"></span>
                    </div>
                </div>

                <!-- Edit indicator -->
                <div id="edit-indicator" style="display: none; padding: 8px 12px; margin-bottom: 12px; background: #fef3c7; border-left: 3px solid #f59e0b; border-radius: 4px; font-size: 13px; color: #92400e;">
                    ⚠️ Viewing edited landmarks. Click "Recompute" to update similarities with your changes.
                </div>

                <ul class="landmarks-list" id="landmarks-list">
                    <li class="landmark-item">Loading...</li>
                </ul>
                <div id="ground-truth-section" style="display: none; margin-top: 20px; padding-top: 20px; border-top: 2px solid #e1e4e8;">
                    <div class="section-title">Ground Truth Matches</div>
                    <div id="ground-truth-content">
                        Loading...
                    </div>
                </div>

                <div id="map-section" style="display: none; margin-top: 20px; padding-top: 20px; border-top: 2px solid #e1e4e8;">
                    <div class="section-title">Interactive Map</div>
                    <div id="map-info" style="font-size: 12px; margin-bottom: 8px; padding: 8px; background: #f3f4f6; border-radius: 4px; border: 1px solid #e5e7eb; color: #374151;">
                        Showing <span id="map-count-shown">0</span> markers<span id="map-perf-warning" style="display: none; color: #f59e0b; font-weight: 500;"> ⚠️ Large dataset may affect performance</span>
                    </div>
                    <div id="map-container" style="width: 100%; height: 400px; border: 1px solid #e1e4e8; border-radius: 8px;"></div>
                </div>
            </div>

            <div class="right-panel">
                <div class="histogram-container">
                    <div class="section-title">Similarity Distribution</div>
                    <div id="histogram"></div>
                </div>

                <div class="results-container">
                    <div class="results-header">
                        <div class="results-title">Filtered Results</div>
                        <div class="sim-range-info" id="sim-range-info">All similarities</div>
                    </div>
                    <div id="results-list">
                        <div class="loading">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        const state = {
            mode: 'panorama',
            currentIndex: 0,
            maxIndex: 0,
            currentData: null,
            allSimilarities: [],
            filteredIndices: [],
            similarityRange: null,
            filterExpression: '',
            displayedResultsCount: 50,  // Start with 50 results
            maxDisplayedResults: 50,    // Increase when "Load More" clicked
            queryPhrases: [],  // Extracted phrases from Python expression
            showGroundTruth: false,
            showMap: false,
            mapInitialized: false,
            // Edit mode state
            isEditing: false,
            hasUnsavedEdits: false,
            hasComputedEdits: false,
            editedLandmarks: null
        };

        // Initialize on page load
        async function init() {
            try {
                const response = await fetch('/api/init');
                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    return;
                }

                if (state.mode === 'panorama') {
                    state.maxIndex = data.n_panoramas - 1;
                } else {
                    state.maxIndex = data.n_satellites - 1;
                }

                // Load first item
                await loadItem(0);

                // Setup keyboard navigation
                document.addEventListener('keydown', handleKeyboard);

            } catch (error) {
                showError('Failed to initialize: ' + error.message);
            }
        }

        async function loadItem(index) {
            state.currentIndex = index;

            try {
                const endpoint = state.mode === 'panorama'
                    ? `/api/panorama/${index}`
                    : `/api/satellite/${index}`;

                const response = await fetch(endpoint);
                const data = await response.json();

                if (data.error) {
                    showError(data.error);
                    return;
                }

                state.currentData = data;
                state.allSimilarities = data.similarities;

                // Cache global data for results rendering
                if (state.mode === 'panorama') {
                    // Need sat data for results
                    if (!state.sat_sentences) {
                        // Will fetch as needed
                    }
                } else {
                    // Need pano data for results
                    if (!state.pano_sentences) {
                        // Will fetch as needed
                    }
                }

                // Reset filter
                state.filteredIndices = Array.from({length: state.allSimilarities.length}, (_, i) => i);
                state.similarityRange = null;

                updateUI();

            } catch (error) {
                showError('Failed to load data: ' + error.message);
            }
        }

        function updateUI() {
            updateCurrentInfo();
            updateImage();
            updateLandmarks();
            updateGroundTruth();
            updateHistogram();
            updateResultsList();
            updateNavigationButtons();
            updateMap();
        }

        function updateCurrentInfo() {
            const info = document.getElementById('current-info');
            if (state.mode === 'panorama') {
                info.textContent = `Viewing Panorama #${state.currentIndex} of ${state.maxIndex}`;
            } else {
                info.textContent = `Viewing Satellite #${state.currentIndex} of ${state.maxIndex}`;
            }
        }

        function updateImage() {
            const container = document.getElementById('image-container');
            const section = document.getElementById('panorama-section');

            if (state.mode === 'panorama' && state.currentData.has_image) {
                section.style.display = 'block';
                // Use static route for faster loading
                const imageUrl = `/static/panorama/${state.currentData.image_filename}`;
                container.innerHTML = `
                    <img class="panorama-image" src="${imageUrl}" alt="Panorama">
                `;
            } else if (state.mode === 'panorama') {
                section.style.display = 'block';
                container.innerHTML = '<div class="no-image">Image not available</div>';
            } else {
                section.style.display = 'none';
            }
        }

        function deduplicateSentences(sentences) {
            // Remove exact duplicates while preserving order
            const seen = new Set();
            return sentences.filter(s => {
                if (seen.has(s)) return false;
                seen.add(s);
                return true;
            });
        }

        function updateLandmarks() {
            if (state.isEditing) {
                return; // Don't update while in edit mode
            }

            const list = document.getElementById('landmarks-list');
            const sentences = state.editedLandmarks || deduplicateSentences(state.currentData.sentences);

            if (sentences.length === 0) {
                list.innerHTML = '<li class="landmark-item">No landmarks</li>';
                return;
            }

            const duplicateCount = state.currentData.sentences.length - sentences.length;
            const duplicateNote = duplicateCount > 0
                ? `<div style="color: #9ca3af; font-size: 12px; margin-top: 8px;">${duplicateCount} duplicate${duplicateCount > 1 ? 's' : ''} removed</div>`
                : '';

            list.innerHTML = sentences.map(s =>
                `<li class="landmark-item">${escapeHtml(s)}</li>`
            ).join('') + duplicateNote;
        }

        // Edit mode functions
        function toggleEditMode() {
            if (state.mode !== 'panorama') {
                alert('Landmark editing is only supported for panorama mode.');
                return;
            }

            if (state.isEditing) {
                exitEditMode();
            } else {
                enterEditMode();
            }
        }

        function enterEditMode() {
            state.isEditing = true;
            const sentences = state.editedLandmarks || deduplicateSentences(state.currentData.sentences);
            const list = document.getElementById('landmarks-list');

            const editHTML = sentences.map((s, idx) => `
                <div class="landmark-edit-row">
                    <textarea class="landmark-edit-input" data-idx="${idx}">${escapeHtml(s)}</textarea>
                    <button class="landmark-remove-btn" onclick="removeLandmark(${idx})">×</button>
                </div>
            `).join('');

            const addButtonHTML = `
                <button class="landmark-add-btn" onclick="addLandmark()">
                    + Add Landmark
                </button>
            `;

            list.innerHTML = editHTML + addButtonHTML;

            // Update button states
            document.getElementById('edit-landmarks-btn').textContent = '💾 Save';
            document.getElementById('recompute-btn').style.display = 'none';
            document.getElementById('reset-btn').style.display = 'none';
        }

        async function exitEditMode() {
            // Collect edited landmarks
            const textareas = document.querySelectorAll('.landmark-edit-input');
            const editedSentences = Array.from(textareas)
                .map(ta => ta.value.trim())
                .filter(s => s.length > 0);

            if (editedSentences.length === 0) {
                alert('Cannot save with no landmarks. Add at least one landmark.');
                return;
            }

            // Save to client state
            state.editedLandmarks = editedSentences;
            state.isEditing = false;

            // Save to server
            try {
                const response = await fetch('/api/edit/landmarks', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        item_idx: state.currentIndex,
                        mode: state.mode,
                        sentences: editedSentences
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Failed to save landmarks');
                }

                state.hasUnsavedEdits = true;

                // Update UI
                updateLandmarks();
                document.getElementById('edit-landmarks-btn').textContent = '✏️ Edit';
                document.getElementById('recompute-btn').style.display = 'inline-flex';
                document.getElementById('reset-btn').style.display = 'inline-flex';
                document.getElementById('edit-indicator').style.display = 'block';

            } catch (error) {
                alert(`Failed to save landmarks: ${error.message}`);
                // Re-enter edit mode so user doesn't lose their changes
                state.isEditing = true;
            }
        }

        function addLandmark() {
            const list = document.getElementById('landmarks-list');
            const existingRows = list.querySelectorAll('.landmark-edit-row');
            const newIdx = existingRows.length;

            const newRowHTML = `
                <div class="landmark-edit-row">
                    <textarea class="landmark-edit-input" data-idx="${newIdx}" placeholder="Enter landmark description..."></textarea>
                    <button class="landmark-remove-btn" onclick="removeLandmark(${newIdx})">×</button>
                </div>
            `;

            const addButton = list.querySelector('.landmark-add-btn');
            addButton.insertAdjacentHTML('beforebegin', newRowHTML);

            // Focus on the new textarea
            list.querySelectorAll('.landmark-edit-input')[newIdx].focus();
        }

        function removeLandmark(idx) {
            const list = document.getElementById('landmarks-list');
            const rows = list.querySelectorAll('.landmark-edit-row');
            if (rows.length > 1) {  // Keep at least one landmark
                rows[idx].remove();
            } else {
                alert('Cannot remove the last landmark. Add a new one first if you want to replace it.');
            }
        }

        async function recomputeSimilarities() {
            if (!state.editedLandmarks) {
                alert('No edits to recompute.');
                return;
            }

            const recomputeBtn = document.getElementById('recompute-btn');
            const statusText = document.getElementById('edit-status-text');
            const statusDiv = document.getElementById('edit-status');

            recomputeBtn.disabled = true;
            recomputeBtn.textContent = '⏳ Computing...';
            statusDiv.style.display = 'block';
            statusText.textContent = 'Generating embeddings and computing similarities...';

            try {
                const response = await fetch('/api/edit/recompute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        item_idx: state.currentIndex,
                        mode: state.mode,
                        sentences: state.editedLandmarks
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Recomputation failed');
                }

                // Update similarities
                state.allSimilarities = data.similarities;
                state.hasComputedEdits = true;
                state.hasUnsavedEdits = false;

                // Update UI
                updateHistogram();
                updateResultsList();
                if (state.showMap) {
                    updateMap();
                }

                statusText.textContent = '✓ Similarities updated successfully!';
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);

            } catch (error) {
                statusText.textContent = `Error: ${error.message}`;
                statusText.style.color = '#ef4444';
                alert(`Failed to recompute similarities: ${error.message}`);
            } finally {
                recomputeBtn.disabled = false;
                recomputeBtn.textContent = '🔄 Recompute';
            }
        }

        async function resetEdits() {
            if (!confirm('Reset all edits and restore original landmarks?')) {
                return;
            }

            try {
                const response = await fetch('/api/edit/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        item_idx: state.currentIndex,
                        mode: state.mode
                    })
                });

                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || 'Reset failed');
                }

                // Clear edit state
                state.editedLandmarks = null;
                state.hasUnsavedEdits = false;
                state.hasComputedEdits = false;
                state.isEditing = false;

                // Reload original data
                await loadItem(state.currentIndex);

                // Update UI
                document.getElementById('edit-landmarks-btn').textContent = '✏️ Edit';
                document.getElementById('recompute-btn').style.display = 'none';
                document.getElementById('reset-btn').style.display = 'none';
                document.getElementById('edit-indicator').style.display = 'none';

            } catch (error) {
                alert(`Failed to reset edits: ${error.message}`);
            }
        }

        async function toggleGroundTruth() {
            state.showGroundTruth = document.getElementById('show-ground-truth').checked;
            updateGroundTruth();
        }

        async function updateGroundTruth() {
            const section = document.getElementById('ground-truth-section');
            const content = document.getElementById('ground-truth-content');

            if (!state.showGroundTruth) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';

            const positiveIndices = state.currentData.positive_indices;

            if (!positiveIndices || positiveIndices.length === 0) {
                content.innerHTML = '<div style="color: #9ca3af; font-size: 13px;">No ground truth matches</div>';
                return;
            }

            content.innerHTML = '<div style="color: #9ca3af; font-size: 13px;">Loading ground truth...</div>';

            try {
                // Fetch data for ground truth items
                const response = await fetch('/api/batch_items', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        indices: positiveIndices,
                        mode: state.mode
                    })
                });

                const data = await response.json();

                if (data.error) {
                    content.innerHTML = '<div style="color: #ef4444;">Error loading ground truth</div>';
                    return;
                }

                // Render ground truth items
                const html = data.items.map(item => {
                    const similarity = state.allSimilarities[item.idx];
                    const dedupedSentences = deduplicateSentences(item.sentences);

                    const mapsUrl = `https://www.google.com/maps?q=${item.loc[0]},${item.loc[1]}`;
                    const osmUrl = `https://www.openstreetmap.org/?mlat=${item.loc[0]}&mlon=${item.loc[1]}&zoom=18`;

                    const itemType = state.mode === 'panorama' ? 'Satellite' : 'Panorama';
                    const itemLabel = state.mode === 'satellite' && item.pano_id
                        ? `${item.pano_id}`
                        : `#${item.idx}`;

                    return `
                        <div class="ground-truth-card">
                            <div class="ground-truth-header">
                                <div class="ground-truth-title">
                                    ${itemType} ${itemLabel} • Similarity: ${similarity.toFixed(3)}
                                </div>
                                <div class="ground-truth-links">
                                    <a href="${mapsUrl}" target="_blank" class="map-link">Maps</a>
                                    <a href="${osmUrl}" target="_blank" class="map-link">OSM</a>
                                </div>
                            </div>
                            <div class="ground-truth-landmarks">
                                ${dedupedSentences.length === 0
                                    ? '<div style="color: #9ca3af; font-size: 12px;">No landmarks</div>'
                                    : dedupedSentences.map(s =>
                                        `<div class="landmark-item">${escapeHtml(s)}</div>`
                                    ).join('')}
                            </div>
                        </div>
                    `;
                }).join('');

                content.innerHTML = html;

            } catch (error) {
                content.innerHTML = '<div style="color: #ef4444;">Error: ' + escapeHtml(error.message) + '</div>';
            }
        }

        async function toggleMap() {
            state.showMap = document.getElementById('show-map').checked;
            const mapSection = document.getElementById('map-section');

            if (!state.showMap) {
                mapSection.style.display = 'none';
                return;
            }

            mapSection.style.display = 'block';
            await updateMap();
        }

        async function updateMap() {
            if (!state.showMap) {
                return;
            }

            const mapContainer = document.getElementById('map-container');
            const mapCountShown = document.getElementById('map-count-shown');
            const mapPerfWarning = document.getElementById('map-perf-warning');

            // Get filtered items (apply both Python filter and histogram range)
            const filteredItems = state.filteredIndices
                .map(idx => ({
                    idx: idx,
                    similarity: state.allSimilarities[idx]
                }))
                .filter(item => {
                    if (!state.similarityRange) return true;
                    return item.similarity >= state.similarityRange[0] &&
                           item.similarity <= state.similarityRange[1];
                });

            // Show all filtered items (no limit)
            const itemsToShow = filteredItems;
            const totalCount = itemsToShow.length;

            // Update info display
            mapCountShown.textContent = totalCount;

            // Show performance warning for very large datasets
            if (totalCount > 10000) {
                mapPerfWarning.style.display = 'inline';
            } else {
                mapPerfWarning.style.display = 'none';
            }

            try {
                // Always render the current location marker, even if no filtered items
                let filteredMarkers = [];

                if (itemsToShow.length > 0) {
                    const response = await fetch('/api/batch_items', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            indices: itemsToShow.map(item => item.idx),
                            mode: state.mode
                        })
                    });

                    const data = await response.json();

                    if (data.error) {
                        console.error('Error loading map data:', data.error);
                    } else {
                        // Build marker data
                        filteredMarkers = data.items.map((item, i) => ({
                            lat: item.loc[0],
                            lon: item.loc[1],
                            idx: item.idx,
                            similarity: itemsToShow[i].similarity,
                            isTrueMatch: false,
                            isCurrent: false
                        }));
                    }
                }

                // Add true match markers
                const trueMatchIndices = state.currentData.positive_indices || [];
                const trueMatchResponse = await fetch('/api/batch_items', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        indices: trueMatchIndices,
                        mode: state.mode
                    })
                });

                const trueMatchData = await trueMatchResponse.json();
                const trueMatchMarkers = trueMatchData.items ? trueMatchData.items.map(item => ({
                    lat: item.loc[0],
                    lon: item.loc[1],
                    idx: item.idx,
                    similarity: state.allSimilarities[item.idx],
                    isTrueMatch: true,
                    isCurrent: false
                })) : [];

                // Add current location marker
                const currentMarker = {
                    lat: state.currentData.loc[0],
                    lon: state.currentData.loc[1],
                    idx: state.currentIndex,
                    similarity: null,
                    isTrueMatch: false,
                    isCurrent: true
                };

                // Create Plotly traces
                const traces = [];

                // Filtered results trace (blue, colored by similarity)
                if (filteredMarkers.length > 0) {
                    const viridisColors = filteredMarkers.map(m => {
                        const normalized = (m.similarity - 0) / (1 - 0);
                        return normalized;
                    });

                    traces.push({
                        type: 'scattermapbox',
                        mode: 'markers',
                        lon: filteredMarkers.map(m => m.lon),
                        lat: filteredMarkers.map(m => m.lat),
                        marker: {
                            size: 6,
                            color: viridisColors,
                            colorscale: 'Viridis',
                            showscale: true,
                            colorbar: {
                                title: 'Similarity',
                                thickness: 10,
                                len: 0.5
                            }
                        },
                        text: filteredMarkers.map(m => `#${m.idx}: ${m.similarity.toFixed(3)}`),
                        hoverinfo: 'text',
                        customdata: filteredMarkers.map(m => m.idx),
                        name: 'Filtered Results'
                    });
                }

                // True matches trace (green)
                if (trueMatchMarkers.length > 0) {
                    traces.push({
                        type: 'scattermapbox',
                        mode: 'markers',
                        lon: trueMatchMarkers.map(m => m.lon),
                        lat: trueMatchMarkers.map(m => m.lat),
                        marker: {
                            size: 10,
                            color: '#10b981',
                            symbol: 'circle'
                        },
                        text: trueMatchMarkers.map(m => `TRUE MATCH #${m.idx}: ${m.similarity.toFixed(3)}`),
                        hoverinfo: 'text',
                        customdata: trueMatchMarkers.map(m => m.idx),
                        name: 'True Matches'
                    });
                }

                // Current location trace (red circle with border)
                traces.push({
                    type: 'scattermapbox',
                    mode: 'markers',
                    lon: [currentMarker.lon],
                    lat: [currentMarker.lat],
                    marker: {
                        size: 16,
                        color: '#ef4444',
                        opacity: 0.9,
                        symbol: 'circle'
                    },
                    text: [`Current: ${state.mode === 'panorama' ? 'Panorama' : 'Satellite'} #${currentMarker.idx}`],
                    hoverinfo: 'text',
                    customdata: [null],  // Don't navigate to self
                    name: 'Current Location'
                });

                // Calculate map center
                const allLats = [...filteredMarkers.map(m => m.lat), ...trueMatchMarkers.map(m => m.lat), currentMarker.lat];
                const allLons = [...filteredMarkers.map(m => m.lon), ...trueMatchMarkers.map(m => m.lon), currentMarker.lon];
                const centerLat = allLats.reduce((a, b) => a + b, 0) / allLats.length;
                const centerLon = allLons.reduce((a, b) => a + b, 0) / allLons.length;

                const layout = {
                    mapbox: {
                        style: 'open-street-map',
                        center: {lat: centerLat, lon: centerLon},
                        zoom: 12
                    },
                    showlegend: true,
                    legend: {
                        x: 0,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.8)'
                    },
                    margin: {l: 0, r: 0, t: 0, b: 0},
                    dragmode: 'pan'
                };

                const config = {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                };

                if (state.mapInitialized) {
                    Plotly.react('map-container', traces, layout, config);
                } else {
                    Plotly.newPlot('map-container', traces, layout, config);
                    state.mapInitialized = true;

                    // Attach click handler
                    mapContainer.on('plotly_click', handleMapClick);
                }

            } catch (error) {
                console.error('Error updating map:', error);
                // Render empty map on error
                if (state.mapInitialized) {
                    Plotly.react('map-container', [], {
                        mapbox: {style: 'open-street-map', center: {lat: 0, lon: 0}, zoom: 1},
                        margin: {l: 0, r: 0, t: 0, b: 0}
                    }, {responsive: true});
                }
            }
        }

        function handleMapClick(eventData) {
            if (!eventData || !eventData.points || eventData.points.length === 0) return;

            const point = eventData.points[0];
            if (point.customdata !== undefined && point.customdata !== null) {
                const clickedIdx = point.customdata;
                if (clickedIdx !== state.currentIndex) {
                    navigateToItem(clickedIdx);
                }
            }
        }

        function updateHistogram() {
            const filteredSimilarities = state.filteredIndices.map(i => state.allSimilarities[i]);

            // Get true match similarities
            const trueMatchIndices = state.currentData.positive_indices;
            const trueMatchSims = trueMatchIndices.map(i => state.allSimilarities[i]);

            // Create traces
            const traces = [
                {
                    x: state.allSimilarities,
                    type: 'histogram',
                    name: 'All',
                    opacity: 0.3,
                    marker: {color: 'lightgray'},
                    nbinsx: 50
                },
                {
                    x: filteredSimilarities,
                    type: 'histogram',
                    name: 'Filtered',
                    opacity: 0.7,
                    marker: {color: 'steelblue'},
                    nbinsx: 50
                }
            ];

            // Add vertical lines for true matches
            if (trueMatchSims.length > 0) {
                const maxY = Math.max(...state.allSimilarities) * state.allSimilarities.length * 0.1;

                trueMatchSims.forEach((sim, idx) => {
                    traces.push({
                        x: [sim, sim],
                        y: [0, maxY],
                        type: 'scatter',
                        mode: 'lines',
                        line: {color: 'red', width: 2, dash: 'dash'},
                        name: idx === 0 ? 'True Match' : '',
                        showlegend: idx === 0,
                        hoverinfo: 'x'
                    });
                });
            }

            const layout = {
                title: '',
                xaxis: {title: 'Similarity Score'},
                yaxis: {title: 'Count'},
                margin: {l: 50, r: 30, t: 10, b: 50},
                hovermode: 'closest',
                showlegend: true,
                legend: {x: 0.7, y: 1}
            };

            const config = {
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                dragmode: 'zoom'
            };

            Plotly.newPlot('histogram', traces, layout, config);

            // Listen for range selection
            const histDiv = document.getElementById('histogram');
            histDiv.on('plotly_relayout', (eventData) => {
                if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
                    const minSim = eventData['xaxis.range[0]'];
                    const maxSim = eventData['xaxis.range[1]'];
                    filterBySimilarityRange(minSim, maxSim);
                } else if (eventData['xaxis.autorange']) {
                    // Reset range
                    state.similarityRange = null;
                    updateResultsList();
                    updateSimRangeInfo();
                }
            });
        }

        function filterBySimilarityRange(minSim, maxSim) {
            state.similarityRange = [minSim, maxSim];
            updateResultsList();
            updateSimRangeInfo();
            updateMap();
        }

        function updateSimRangeInfo() {
            const info = document.getElementById('sim-range-info');
            if (state.similarityRange) {
                info.textContent = `Similarity range: [${state.similarityRange[0].toFixed(3)}, ${state.similarityRange[1].toFixed(3)}]`;
            } else {
                info.textContent = 'All similarities';
            }
        }

        function extractQueryPhrases(expression) {
            // Extract string literals from Python expression
            const phrases = [];
            // Match both single and double quoted strings
            const regex = /["']([^"']+)["']/g;
            let match;
            while ((match = regex.exec(expression)) !== null) {
                phrases.push(match[1]);
            }
            return phrases;
        }

        function highlightText(text, phrases) {
            if (!phrases || phrases.length === 0) return escapeHtml(text);

            let result = escapeHtml(text);
            // Sort phrases by length (longest first) to avoid partial matches
            const sortedPhrases = [...phrases].sort((a, b) => b.length - a.length);

            for (const phrase of sortedPhrases) {
                const escapedPhrase = escapeHtml(phrase);
                // Case-insensitive highlighting
                const regex = new RegExp(`(${escapedPhrase.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')})`, 'gi');
                result = result.replace(regex, '<mark style="background-color: #fef08a; padding: 1px 2px;">$1</mark>');
            }
            return result;
        }

        async function updateResultsList() {
            const container = document.getElementById('results-list');

            // Get items to display
            let items = state.filteredIndices.map(idx => ({
                idx: idx,
                similarity: state.allSimilarities[idx]
            }));

            // Filter by similarity range if set
            if (state.similarityRange) {
                items = items.filter(item =>
                    item.similarity >= state.similarityRange[0] &&
                    item.similarity <= state.similarityRange[1]
                );
            }

            // Sort by similarity (descending)
            items.sort((a, b) => b.similarity - a.similarity);

            // Get true match indices for highlighting
            const trueMatchSet = new Set(state.currentData.positive_indices);

            // Show only top N items
            const displayItems = items.slice(0, state.maxDisplayedResults);

            if (displayItems.length === 0) {
                container.innerHTML = '<div class="loading">No results match the current filter</div>';
                return;
            }

            const html = displayItems.map(item => {
                const simClass = item.similarity > 0.7 ? 'high' : item.similarity > 0.4 ? 'medium' : 'low';
                const isTrueMatch = trueMatchSet.has(item.idx);

                return `
                    <div class="item-card" data-idx="${item.idx}">
                        <div class="item-header">
                            <span class="sim-badge ${simClass}">${item.similarity.toFixed(3)}</span>
                            <span class="item-index">#${item.idx}</span>
                            ${isTrueMatch ? '<span class="true-match-badge">TRUE MATCH</span>' : ''}
                            <a href="#" class="item-nav-link" onclick="navigateToItem(${item.idx}); return false;">
                                View →
                            </a>
                        </div>
                        <div class="item-landmarks" id="landmarks-${item.idx}">
                            Loading landmarks...
                        </div>
                    </div>
                `;
            }).join('');

            // Add "Load More" button if there are more results
            const loadMoreBtn = items.length > state.maxDisplayedResults
                ? `<div style="text-align: center; padding: 20px;">
                    <button class="filter-btn" onclick="loadMoreResults()">
                        Load More (${items.length - state.maxDisplayedResults} remaining)
                    </button>
                   </div>`
                : '';

            container.innerHTML = html + loadMoreBtn;

            // Lazy load landmarks for visible items
            loadLandmarksForVisibleItems(displayItems);
        }

        function loadMoreResults() {
            state.maxDisplayedResults += 50;
            updateResultsList();
        }

        async function loadLandmarksForVisibleItems(items) {
            // Batch fetch data for all items
            const indices = items.map(item => item.idx);

            try {
                const response = await fetch('/api/batch_items', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        indices: indices,
                        mode: state.mode
                    })
                });

                const data = await response.json();

                if (data.error) {
                    console.error('Failed to load batch items:', data.error);
                    return;
                }

                // Update landmarks for each item
                data.items.forEach(item => {
                    const elem = document.getElementById(`landmarks-${item.idx}`);
                    if (elem) {
                        // Deduplicate sentences
                        const dedupedSentences = deduplicateSentences(item.sentences);

                        // In satellite mode, show panorama ID and index
                        let headerHtml = '';
                        if (state.mode === 'satellite' && item.pano_id) {
                            headerHtml = `<div style="color: #667eea; font-weight: 600; font-size: 12px; margin-bottom: 6px; padding-left: 16px;">
                                Panorama: ${escapeHtml(item.pano_id)} (Index: ${item.idx})
                            </div>`;
                        }

                        if (dedupedSentences.length === 0) {
                            elem.innerHTML = headerHtml + '<div class="item-landmark" style="color: #9ca3af;">No landmarks</div>';
                        } else {
                            // Show ALL landmarks with highlighting
                            const landmarksHtml = dedupedSentences
                                .map(s => `<div class="item-landmark">${highlightText(s, state.queryPhrases)}</div>`)
                                .join('');

                            const duplicateCount = item.sentences.length - dedupedSentences.length;
                            const duplicateNote = duplicateCount > 0
                                ? `<div style="color: #9ca3af; font-size: 11px; margin-top: 4px; padding-left: 16px;">(${duplicateCount} duplicate${duplicateCount > 1 ? 's' : ''} removed)</div>`
                                : '';

                            elem.innerHTML = headerHtml + landmarksHtml + duplicateNote;
                        }
                    }

                    // Store location data for Maps links
                    const card = document.querySelector(`[data-idx="${item.idx}"]`);
                    if (card) {
                        card.dataset.lat = item.loc[0];
                        card.dataset.lon = item.loc[1];
                    }
                });

            } catch (error) {
                console.error('Failed to fetch batch items:', error);
            }
        }

        function openMaps(idx, event) {
            event.preventDefault();
            // Get location from card data attributes
            const card = document.querySelector(`[data-idx="${idx}"]`);
            if (card && card.dataset.lat && card.dataset.lon) {
                const mapsUrl = `https://www.google.com/maps?q=${card.dataset.lat},${card.dataset.lon}`;
                window.open(mapsUrl, '_blank');
            } else {
                alert('Location data not yet loaded');
            }
        }

        function navigateToItem(idx) {
            // Navigate to the clicked item
            if (state.mode === 'panorama') {
                // In panorama mode, clicked item is a satellite
                // Switch to satellite mode and show that satellite
                state.mode = 'satellite';
                updateModeButtons();
                loadItem(idx);
            } else {
                // In satellite mode, clicked item is a panorama
                // Switch to panorama mode and show that panorama
                state.mode = 'panorama';
                updateModeButtons();
                loadItem(idx);
            }
        }

        async function seekToItem() {
            const input = document.getElementById('seek-input').value.trim();
            if (!input) return;

            // Try to parse as index first
            const indexNum = parseInt(input);
            if (!isNaN(indexNum)) {
                // It's a number - treat as index
                if (indexNum >= 0 && indexNum <= state.maxIndex) {
                    loadItem(indexNum);
                    document.getElementById('seek-input').value = '';
                } else {
                    alert(`Index must be between 0 and ${state.maxIndex}`);
                }
                return;
            }

            // If in panorama mode, try to find panorama by ID
            if (state.mode === 'panorama') {
                try {
                    const response = await fetch('/api/find_panorama', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({pano_id: input})
                    });

                    const data = await response.json();

                    if (data.error) {
                        alert(`Panorama ID not found: ${input}`);
                    } else {
                        loadItem(data.index);
                        document.getElementById('seek-input').value = '';
                    }
                } catch (error) {
                    alert('Error searching for panorama: ' + error.message);
                }
            } else {
                alert('Panorama ID search only works in panorama mode. Use index number for satellite mode.');
            }
        }

        function updateModeButtons() {
            document.getElementById('pano-mode-btn').classList.toggle('active', state.mode === 'panorama');
            document.getElementById('sat-mode-btn').classList.toggle('active', state.mode === 'satellite');
        }

        async function applyFilter() {
            const expression = document.getElementById('filter-input').value.trim();
            state.filterExpression = expression;

            // Extract query phrases for highlighting
            state.queryPhrases = extractQueryPhrases(expression);

            if (!expression) {
                // Reset filter
                state.filteredIndices = Array.from({length: state.allSimilarities.length}, (_, i) => i);
                state.queryPhrases = [];
                state.maxDisplayedResults = 50;  // Reset pagination
                document.getElementById('filter-count').innerHTML = 'No filter applied';
                updateHistogram();
                updateResultsList();
                updateMap();
                return;
            }

            try {
                const response = await fetch('/api/filter', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        expression: expression,
                        mode: state.mode
                    })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Filter error: ' + data.error);
                    return;
                }

                state.filteredIndices = data.filtered_indices;
                state.maxDisplayedResults = 50;  // Reset pagination

                const totalCount = state.allSimilarities.length;
                document.getElementById('filter-count').innerHTML =
                    `Filtered: <strong>${data.count}</strong> / ${totalCount} items`;

                updateHistogram();
                updateResultsList();
                updateMap();

            } catch (error) {
                alert('Failed to apply filter: ' + error.message);
            }
        }

        function switchMode(newMode) {
            if (state.mode === newMode) return;

            state.mode = newMode;
            state.currentIndex = 0;
            state.filterExpression = '';
            document.getElementById('filter-input').value = '';
            document.getElementById('seek-input').value = '';

            // Update mode buttons
            updateModeButtons();

            // Reload
            init();
        }

        function navigate(delta) {
            const newIndex = state.currentIndex + delta;
            if (newIndex >= 0 && newIndex <= state.maxIndex) {
                loadItem(newIndex);
            }
        }

        function updateNavigationButtons() {
            document.getElementById('prev-btn').disabled = state.currentIndex === 0;
            document.getElementById('next-btn').disabled = state.currentIndex === state.maxIndex;
        }

        function handleKeyboard(event) {
            if (event.target.tagName === 'INPUT') return;

            if (event.key === 'ArrowLeft') {
                navigate(-1);
            } else if (event.key === 'ArrowRight') {
                navigate(1);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function showError(message) {
            document.querySelector('.container').innerHTML =
                `<div class="error">Error: ${escapeHtml(message)}</div>`;
        }

        // Start
        init();
    </script>
</body>
</html>
'''


# ============================================================================
# Model Loading and Inference Functions
# ============================================================================

def load_models_lazy():
    """Lazy load PyTorch models and pre-compute all embeddings on first use."""
    global MODELS_LOADED, PANO_MODEL, SAT_MODEL, OPENAI_CLIENT
    global PANO_EMBEDDINGS, SAT_EMBEDDINGS, TORCH_AVAILABLE

    if MODELS_LOADED:
        return

    # Try to import torch modules if not already attempted
    if TORCH_AVAILABLE is None:
        try:
            # CRITICAL: Import load_torch_deps BEFORE torch to enable CUDA
            import common.torch.load_torch_deps
            import torch
            import torch.nn.functional as F
            import common.torch.load_and_save_models as lsm
            from openai import OpenAI

            # Make modules globally available
            globals()['torch'] = torch
            globals()['F'] = F
            globals()['lsm'] = lsm
            globals()['OpenAI'] = OpenAI

            TORCH_AVAILABLE = True
            print("Successfully imported PyTorch with CUDA support")
        except ImportError as e:
            TORCH_AVAILABLE = False
            raise RuntimeError(f"Could not import PyTorch dependencies: {e}")

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch dependencies not available. Cannot load models.")

    print("\n" + "="*80)
    print("Loading PyTorch models and computing embeddings...")
    print("="*80)

    try:
        # Import modules from globals (set above)
        torch = globals()['torch']
        lsm = globals()['lsm']
        OpenAI = globals()['OpenAI']

        # Load models from same paths as understand_model_performance.py
        model_base = Path("/data/overhead_matching/training_outputs/"
                         "251205_09000000_model_size_experiments/"
                         "landmark_base_performance/"
                         "panorama_semantic_landmark_embeddings/")

        print(f"Loading panorama model from {model_base / '0099_panorama'}...")
        PANO_MODEL = lsm.load_model(model_base / "0099_panorama", device="cuda:0")

        print(f"Loading satellite model from {model_base / '0099_satellite'}...")
        SAT_MODEL = lsm.load_model(model_base / "0099_satellite", device="cuda:0")

        # Debug: Print available extractors
        print(f"\nPanorama model extractors: {list(PANO_MODEL._config.extractor_config_by_name.keys())}")
        print(f"Satellite model extractors: {list(SAT_MODEL._config.extractor_config_by_name.keys())}")

        # Initialize OpenAI client
        print("Initializing OpenAI client...")
        OPENAI_CLIENT = OpenAI()

        # Load pre-computed embeddings from pickle file
        if VISUALIZER_DATA.get('panorama_embeddings_array') is not None:
            print("\nLoading pre-computed embeddings from pickle file...")
            PANO_EMBEDDINGS = torch.from_numpy(VISUALIZER_DATA['panorama_embeddings_array']).to("cuda:0")
            SAT_EMBEDDINGS = torch.from_numpy(VISUALIZER_DATA['sat_embeddings_array']).to("cuda:0")
            print(f"  ✓ Loaded {PANO_EMBEDDINGS.shape[0]} panorama embeddings (shape: {PANO_EMBEDDINGS.shape})")
            print(f"  ✓ Loaded {SAT_EMBEDDINGS.shape[0]} satellite embeddings (shape: {SAT_EMBEDDINGS.shape})")
        else:
            print("\nWarning: No pre-computed embeddings available.")
            print("Embeddings will be computed on-demand (slower, requires OpenAI API calls).")

        MODELS_LOADED = True
        print("\n" + "="*80)
        print("Models loaded successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nERROR loading models: {e}")
        print("Landmark editing will be disabled.")
        import traceback
        traceback.print_exc()
        raise


def compute_item_embedding(sentences: list[str], mode: str):
    """
    Compute embedding for a single panorama given its landmark sentences.

    Full pipeline:
    1. Generate OpenAI embeddings for sentences (1536-dim, full length)
    2. Normalize embeddings
    3. Project to model output dimension (no position embeddings - NullPositionEmbedding behavior)
    4. Run through aggregator transformer
    5. Extract CLS token output (final embedding)

    Args:
        sentences: List of landmark description strings
        mode: 'panorama' only (satellite mode not supported)

    Returns:
        torch.Tensor of shape (output_dim,) - normalized embedding vector
    """
    if mode != 'panorama':
        raise ValueError("Only panorama mode is supported for landmark editing")

    torch = globals()['torch']
    F = globals()['F']

    model = PANO_MODEL
    extractor_name = 'panorama_semantic_landmark_extractor'
    output_dim = model._output_dim

    if not sentences or len(sentences) == 0:
        # No landmarks - return zero embedding
        return torch.zeros(output_dim, device=model._cls_token.device)

    # Step 1: Generate OpenAI embeddings (1536-dim, full length - no truncation)
    raw_embeddings = generate_embeddings_for_sentences(sentences)  # (N_sentences, 1536)
    raw_embeddings = raw_embeddings.to(model._cls_token.device)

    # Step 2: Normalize
    raw_embeddings = F.normalize(raw_embeddings, dim=-1)

    # Step 3: Project to model output dimension
    # NullPositionEmbedding behavior: no position features appended
    projection = model._projection_by_name[extractor_name]
    projected_tokens = projection(raw_embeddings)  # (N_sentences, output_dim)

    # Add token marker
    token_marker = model._token_marker_by_name[extractor_name]

    # Ensure token_marker is 1D (output_dim,) by squeezing out batch/sequence dimensions
    token_marker_squeezed = token_marker.squeeze()
    tokens_with_marker = projected_tokens + token_marker_squeezed  # (N_sentences, output_dim)

    # Add batch dimension
    tokens_with_marker = tokens_with_marker.unsqueeze(0)  # (1, N_sentences, output_dim)

    # Step 4: Add CLS token (expand to batch size like in swag_patch_embedding.py:506)
    batch_size = tokens_with_marker.shape[0]  # Should be 1
    cls_token = model._cls_token.expand(batch_size, -1, -1)  # (batch_size, num_embeddings, output_dim)
    all_tokens = torch.cat([cls_token, tokens_with_marker], dim=1)  # (batch_size, num_embeddings + N_sentences, output_dim)

    # CRITICAL: Normalize all input tokens before aggregator (swag_patch_embedding.py:508-509)
    if model._normalize_embeddings:
        all_tokens = F.normalize(all_tokens, dim=-1)

    # Create mask (all tokens are valid)
    token_mask = torch.zeros(all_tokens.shape[0], all_tokens.shape[1], dtype=torch.bool, device=all_tokens.device)

    # Step 5: Run through aggregator transformer
    aggregated = model._aggregator_model(all_tokens, token_mask)  # (1, num_embeddings + N_sentences, output_dim)

    # Extract CLS token output (first num_embeddings tokens, then average)
    cls_output = aggregated[:, :model._config.num_embeddings, :].mean(dim=1).squeeze(0)  # (output_dim,)

    # Final normalization if configured
    if model._normalize_embeddings:
        cls_output = F.normalize(cls_output, dim=-1)

    return cls_output


def generate_embeddings_for_sentences(sentences: list[str]):
    """
    Call OpenAI API to generate embeddings for a list of sentences.

    Args:
        sentences: List of text descriptions

    Returns:
        torch.Tensor of shape (len(sentences), 1536)
    """
    torch = globals()['torch']

    if not sentences:
        return torch.zeros((0, 1536))

    try:
        response = OPENAI_CLIENT.embeddings.create(
            model="text-embedding-3-small",
            input=sentences
        )

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return torch.tensor(embeddings, dtype=torch.float32)

    except Exception as e:
        print(f"ERROR calling OpenAI API: {e}")
        raise


def compute_similarity_with_custom_landmarks(
    item_idx: int,
    mode: str,
    custom_sentences: list[str]
) -> list[float]:
    """
    Recompute similarities using custom landmark descriptions.
    Only supports panorama mode.

    Steps:
    1. Ensure models and pre-computed embeddings are loaded
    2. Generate embedding for custom panorama landmarks (OpenAI API + full model pipeline)
    3. Compute dot product with pre-computed satellite embeddings

    Args:
        item_idx: Index of panorama item
        mode: 'panorama' only (satellite mode not supported)
        custom_sentences: New landmark descriptions

    Returns:
        List of similarity scores for all satellites
    """
    if mode != 'panorama':
        raise ValueError("Only panorama mode is supported for landmark editing. Cannot edit satellite landmarks.")

    # Ensure models and embeddings are loaded
    load_models_lazy()

    torch = globals()['torch']

    if PANO_EMBEDDINGS is None or SAT_EMBEDDINGS is None:
        raise RuntimeError("Pre-computed embeddings not available. Please regenerate the pickle file with embeddings.")

    # Compute embedding for custom panorama landmarks (runs full model pipeline)
    custom_embedding = compute_item_embedding(custom_sentences, mode)  # (output_dim,)

    # Get all pre-computed satellite embeddings
    sat_embeddings = SAT_EMBEDDINGS  # (N_sat, output_dim)

    # Compute dot product similarities
    similarities = torch.matmul(
        sat_embeddings,  # (N_sat, output_dim)
        custom_embedding.unsqueeze(-1)  # (output_dim, 1)
    ).squeeze(-1)  # (N_sat,)

    return similarities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description='Landmark-based geolocalization visualizer')
    parser.add_argument('--pickle-path', type=Path, required=True,
                        help='Path to visualizer_information.pkl')
    parser.add_argument('--panorama-dir', type=Path, required=True,
                        help='Directory containing panorama images')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')

    args = parser.parse_args()

    # Load data
    global VISUALIZER_DATA, SIMILARITY_MATRIX, PANORAMA_PATH_MAP, PANORAMA_DIR

    VISUALIZER_DATA = load_visualizer_data(args.pickle_path)
    SIMILARITY_MATRIX = VISUALIZER_DATA['similarity_matrix']
    PANORAMA_DIR = args.panorama_dir
    PANORAMA_PATH_MAP = build_panorama_path_map(args.panorama_dir, VISUALIZER_DATA['panorama_id'])

    print(f"\n{'='*60}")
    print(f"🚀 Starting Landmark Visualizer")
    print(f"{'='*60}")
    print(f"📊 Loaded {len(VISUALIZER_DATA['panorama_id'])} panoramas")
    print(f"🛰️  Loaded {len(VISUALIZER_DATA['sat_loc'])} satellites")
    print(f"🖼️  Found {len(PANORAMA_PATH_MAP)} panorama images")
    print(f"{'='*60}")
    print(f"\n🌐 Server starting at http://{args.host}:{args.port}")
    print(f"   Open in your browser to begin exploring!\n")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
