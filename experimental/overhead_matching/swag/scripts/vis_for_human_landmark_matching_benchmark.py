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

# Global data
VISUALIZER_DATA: Optional[dict] = None
PANORAMA_PATH_MAP: dict[str, Path] = {}
SIMILARITY_MATRIX: Optional[np.ndarray] = None
PANORAMA_DIR: Optional[Path] = None

app = Flask(__name__)


def load_visualizer_data(pickle_path: Path) -> dict:
    """Load and process the visualizer data from pickle file."""
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

    # Limit to 100 items for performance
    indices = indices[:100]

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
            overflow-y: auto;
            max-height: calc(100vh - 280px);
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
            cursor: pointer;
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

                <div class="section-title">Landmarks</div>
                <ul class="landmarks-list" id="landmarks-list">
                    <li class="landmark-item">Loading...</li>
                </ul>
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
            queryPhrases: []  // Extracted phrases from Python expression
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
            updateHistogram();
            updateResultsList();
            updateNavigationButtons();
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
            const list = document.getElementById('landmarks-list');
            const sentences = deduplicateSentences(state.currentData.sentences);

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
                    <div class="item-card" onclick="navigateToItem(${item.idx})" data-idx="${item.idx}">
                        <div class="item-header">
                            <span class="sim-badge ${simClass}">${item.similarity.toFixed(3)}</span>
                            <span class="item-index">#${item.idx}</span>
                            ${isTrueMatch ? '<span class="true-match-badge">TRUE MATCH</span>' : ''}
                            <a href="#" class="maps-link" onclick="event.stopPropagation(); openMaps(${item.idx}, event)">
                                📍 Maps
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
