#!/usr/bin/env python3
"""
Web-based inspector for captured model input batches.

This tool allows visualization of captured batch data including:
- Panorama and satellite semantic class histograms
- Positive/semipositive/negative pairings
- Raw feature matrices and tensors

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:inspect_captured_batches -- \
        --captures_dir /tmp/osm_and_pano_semantic_class/debug_captures \
        --semantic_classes /data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/semantic_class_grouping.json \
        --port 5001
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import sys
import common.torch.load_torch_deps
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experimental.overhead_matching.swag.scripts.model_inspector import load_captured_batch
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_sentence_dict_from_pano_jsons,
    make_sentence_dict_from_json, prune_landmark)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import _custom_id_from_props


class BatchInspector:
    """Manages batch data loading and processing for inspection."""

    def __init__(self, captures_dir: Path, semantic_classes_path: Path, satellite_sentences_dir: Path = None):
        self.captures_dir = Path(captures_dir)
        self.semantic_classes_path = Path(semantic_classes_path)
        self.satellite_sentences_dir = Path(satellite_sentences_dir) if satellite_sentences_dir else None

        # Load semantic class names
        with open(semantic_classes_path, 'r') as f:
            class_data = json.load(f)
            # Handle different JSON formats
            if 'ontology' in class_data:
                self.semantic_class_names = class_data['ontology']
            elif 'semantic_groups' in class_data:
                # Extract group names (keys) as the semantic class names
                self.semantic_class_names = list(class_data['semantic_groups'].keys())
            else:
                raise ValueError(f"Unknown semantic class JSON format in {semantic_classes_path}")

        # Cache for loaded batches
        self.batch_cache = {}

        # Get available batch files
        self.batch_files = sorted(self.captures_dir.glob("batch_*.pt*"))

        # Load sentences from train config
        self.panorama_sentences = {}  # pano_id -> sentence
        self.panorama_metadata = {}    # custom_id -> metadata
        self.satellite_sentences = {}  # landmark_id -> sentence
        self._load_sentence_data()

    def _load_sentence_data(self):
        """Load panorama and satellite sentences from train config."""
        # Find train config in parent directory of captures_dir
        train_config_path = self.captures_dir.parent / "train_config.yaml"
        if not train_config_path.exists():
            print(f"Warning: train config not found at {train_config_path}, landmark descriptions will not be available")
            return

        # Load train config
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)

        # Load panorama sentences if panorama extractor is configured
        pano_config = train_config.get('pano_model_config', {})
        pano_extractor_configs = pano_config.get('extractor_config_by_name', {})
        if 'panorama_semantic_landmark_extractor' in pano_extractor_configs:
            pano_aux_info = pano_config.get('auxiliary_info', {})
            semantic_embedding_base_path = pano_aux_info.get('semantic_embedding_base_path')
            pano_extractor_config = pano_extractor_configs['panorama_semantic_landmark_extractor']
            embedding_version = pano_extractor_config.get('embedding_version', 'pano_v1')

            if semantic_embedding_base_path:
                base_path = Path(semantic_embedding_base_path) / embedding_version
                if base_path.exists():
                    # Load sentences from all city directories
                    city_dirs = [d for d in base_path.iterdir() if d.is_dir()]
                    for city_dir in city_dirs:
                        city_name = city_dir.name
                        sentence_dir = city_dir / "sentences"
                        if sentence_dir.exists():
                            print(f"Loading panorama sentences for city: {city_name}")
                            city_sentences, city_metadata, _ = make_sentence_dict_from_pano_jsons(
                                load_all_jsonl_from_folder(sentence_dir))
                            self.panorama_sentences.update(city_sentences)
                            self.panorama_metadata.update(city_metadata)
                            print(f"  Loaded {len(city_sentences)} sentences")

        # Load satellite sentences from CLI-provided directory
        if self.satellite_sentences_dir and self.satellite_sentences_dir.exists():
            print(f"Loading satellite landmark sentences from {self.satellite_sentences_dir}")
            satellite_sentences_dict, _ = make_sentence_dict_from_json(
                load_all_jsonl_from_folder(self.satellite_sentences_dir))
            self.satellite_sentences.update(satellite_sentences_dict)
            print(f"  Loaded {len(satellite_sentences_dict)} satellite landmark sentences")

    def get_batch_list(self):
        """Get list of available batch files."""
        return [f.name for f in self.batch_files]

    def load_batch(self, batch_name: str):
        """Load a batch from disk, using cache if available."""
        if batch_name in self.batch_cache:
            return self.batch_cache[batch_name]

        batch_path = self.captures_dir / batch_name
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_path}")

        data = load_captured_batch(batch_path)
        self.batch_cache[batch_name] = data
        return data

    def compute_histogram(self, features: torch.Tensor, mask: torch.Tensor) -> list:
        """Compute semantic class histogram from features.

        Args:
            features: Tensor of shape (num_items, num_classes) - one-hot encoded
            mask: Boolean tensor of shape (num_items,) - True means masked/invalid

        Returns:
            List of counts for each of the 37 semantic classes
        """
        # Filter out masked items
        valid_features = features[~mask]

        if len(valid_features) == 0:
            return [0] * len(self.semantic_class_names)

        # Get class indices from one-hot encoding
        class_indices = valid_features.argmax(dim=1)

        # Compute histogram
        histogram = torch.bincount(class_indices, minlength=len(self.semantic_class_names))

        return histogram.tolist()

    def get_pano_data(self, batch_name: str, pano_idx: int):
        """Get detailed data for a specific panorama."""
        data = self.load_batch(batch_name)

        if pano_idx < 0 or pano_idx >= len(data['pano_input']['metadata']):
            raise ValueError(f"Invalid pano_idx: {pano_idx}")

        pano_meta = data['pano_input']['metadata'][pano_idx]
        pano_extractor = list(data['pano_extractor_outputs'].keys())[0]

        pano_features = data['pano_extractor_outputs'][pano_extractor]['features'][pano_idx]
        pano_mask = data['pano_extractor_outputs'][pano_extractor]['mask'][pano_idx]
        pano_positions = data['pano_extractor_outputs'][pano_extractor]['positions'][pano_idx]

        # Extract and decode sentences from debug tensor if available
        pano_sentences = []
        if 'debug' in data['pano_extractor_outputs'][pano_extractor]:
            debug_dict = data['pano_extractor_outputs'][pano_extractor]['debug']
            if 'sentences' in debug_dict:
                sentences_tensor = debug_dict['sentences'][pano_idx]  # Shape: [num_landmarks, max_length]
                for landmark_idx in range(len(pano_features)):
                    if not pano_mask[landmark_idx]:  # Only decode valid landmarks
                        sentence_bytes = sentences_tensor[landmark_idx].cpu().numpy()
                        # Find the end of the string (first zero byte)
                        sentence_bytes = sentence_bytes[sentence_bytes != 0]
                        if len(sentence_bytes) > 0:
                            sentence = sentence_bytes.tobytes().decode('utf-8', errors='ignore')
                            semantic_class_idx = int(pano_features[landmark_idx].argmax())
                            semantic_class = self.semantic_class_names[semantic_class_idx]
                            pano_sentences.append({
                                'landmark_idx': landmark_idx,
                                'sentence': sentence,
                                'semantic_class': semantic_class
                            })

        # Compute histogram - features are already [num_items, num_classes], mask is [num_items]
        pano_histogram = self.compute_histogram(pano_features, pano_mask)

        # Get related satellite indices from pairing_data
        pairing_data = data['pairing_data']
        positive_sat_idxs = []
        semipositive_sat_idxs = []

        # Check if pairing_data is Pairs or PositiveAnchorSets
        if hasattr(pairing_data, 'positive_pairs'):
            # It's a Pairs object
            positive_sat_idxs = [sat_idx for pano_idx_in_pair, sat_idx in pairing_data.positive_pairs if pano_idx_in_pair == pano_idx]
            semipositive_sat_idxs = [sat_idx for pano_idx_in_pair, sat_idx in pairing_data.semipositive_pairs if pano_idx_in_pair == pano_idx]
        elif hasattr(pairing_data, 'anchor'):
            # It's a PositiveAnchorSets object
            if pano_idx in pairing_data.anchor:
                anchor_position = pairing_data.anchor.index(pano_idx)
                positive_sat_idxs = list(pairing_data.positive[anchor_position])
                semipositive_sat_idxs = list(pairing_data.semipositive[anchor_position])

        # Map global satellite index to batch-local index for associated_sat_idx
        associated_sat_global = pano_meta.get('satellite_idx')
        associated_sat_local = None
        if associated_sat_global is not None:
            # Build a mapping from global sat index to batch-local index
            for local_idx, sat_meta in enumerate(data['sat_input']['metadata']):
                if sat_meta.get('index') == associated_sat_global:
                    associated_sat_local = local_idx
                    break

        # Build result
        result = {
            'pano_id': pano_meta['pano_id'],
            'lat': float(pano_meta['lat']),
            'lon': float(pano_meta['lon']),
            'num_valid_landmarks': int((~pano_mask).sum()),
            'histogram': pano_histogram,
            'positive_sat_idxs': positive_sat_idxs,
            'semipositive_sat_idxs': semipositive_sat_idxs,
            'associated_sat_idx': associated_sat_local,
            'features_shape': list(pano_features.shape),
            'positions_shape': list(pano_positions.shape),
            'raw_features': pano_features.tolist(),
            'raw_positions': pano_positions.tolist(),
            'raw_mask': pano_mask.tolist(),
            'sentences': pano_sentences,
        }

        return result

    def get_sat_data(self, batch_name: str, sat_idx: int):
        """Get detailed data for a specific satellite patch."""
        print(f"Loading satellite {sat_idx} from {batch_name}", flush=True)
        data = self.load_batch(batch_name)

        if sat_idx < 0 or sat_idx >= len(data['sat_input']['metadata']):
            raise ValueError(f"Invalid sat_idx: {sat_idx}")

        sat_meta = data['sat_input']['metadata'][sat_idx]
        sat_extractor = list(data['sat_extractor_outputs'].keys())[0]

        sat_features = data['sat_extractor_outputs'][sat_extractor]['features'][sat_idx]
        sat_mask = data['sat_extractor_outputs'][sat_extractor]['mask'][sat_idx]
        sat_positions = data['sat_extractor_outputs'][sat_extractor]['positions'][sat_idx]

        # Match satellite landmarks to sentences using unique IDs
        sat_sentences = []

        # Get landmarks from metadata
        landmarks = sat_meta.get('landmarks', [])

        for landmark_idx, landmark in enumerate(landmarks):
            # Create unique ID using prune_landmark and hash
            pruned_props = prune_landmark(landmark)
            landmark_id = _custom_id_from_props(pruned_props)

            # Look up sentence
            if landmark_id in self.satellite_sentences:
                sentence = self.satellite_sentences[landmark_id]
            else:
                continue

            # Determine semantic class from satellite features
            # Features are one-hot encoded semantic classes
            # Find the token index for this landmark (skipping masked ones)
            valid_token_idx = 0
            token_idx = None
            for i, is_masked in enumerate(sat_mask):
                if not is_masked:
                    if valid_token_idx == landmark_idx:
                        token_idx = i
                        break
                    valid_token_idx += 1

            if token_idx is not None:
                semantic_class_idx = int(sat_features[token_idx].argmax())
                semantic_class = self.semantic_class_names[semantic_class_idx]
            else:
                semantic_class_idx = len(self.semantic_class_names)  # Put unknowns at the end
                semantic_class = "unknown"

            # Convert pruned properties frozenset to dict for JSON serialization
            osm_tags = {k: v for k, v in sorted(pruned_props)}

            sat_sentences.append({
                'landmark_idx': landmark_idx,
                'sentence': sentence,
                'semantic_class': semantic_class,
                'semantic_class_idx': semantic_class_idx,
                'osm_tags': osm_tags
            })

        # Sort sentences by semantic class index to match histogram order
        sat_sentences.sort(key=lambda x: x['semantic_class_idx'])

        # Compute histogram - features are already [num_items, num_classes], mask is [num_items]
        sat_histogram = self.compute_histogram(sat_features, sat_mask)

        result = {
            'sat_idx': sat_idx,
            'lat': float(sat_meta['lat']),
            'lon': float(sat_meta['lon']),
            'num_valid_tokens': int((~sat_mask).sum()),
            'histogram': sat_histogram,
            'features_shape': list(sat_features.shape),
            'positions_shape': list(sat_positions.shape),
            'raw_features': sat_features.tolist(),
            'raw_positions': sat_positions.tolist(),
            'raw_mask': sat_mask.tolist(),
            'sentences': sat_sentences,
        }

        print(f"Finished loading satellite {sat_idx}", flush=True)
        return result

    def get_batch_metadata(self, batch_name: str):
        """Get metadata about a batch."""
        data = self.load_batch(batch_name)

        return {
            'batch_idx': data['metadata']['batch_idx'],
            'epoch_idx': data['metadata']['epoch_idx'],
            'total_batches': data['metadata']['total_batches'],
            'num_panos': len(data['pano_input']['metadata']),
            'num_sats': len(data['sat_input']['metadata']),
            'pano_extractor': list(data['pano_extractor_outputs'].keys())[0],
            'sat_extractor': list(data['sat_extractor_outputs'].keys())[0],
        }


# Create Flask app
app = Flask(__name__)
inspector = None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Batch Inspector</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        h1 {
            margin: 0;
            color: #333;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-weight: 600;
            margin-bottom: 5px;
            color: #555;
            font-size: 14px;
        }
        select, button {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            border: none;
            font-weight: 600;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .pano-info {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        .pano-info h2 {
            margin-top: 0;
            color: #1976d2;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .info-item {
            display: flex;
            flex-direction: column;
        }
        .info-label {
            font-weight: 600;
            color: #555;
            font-size: 13px;
        }
        .info-value {
            font-size: 16px;
            color: #333;
            margin-top: 5px;
        }
        .pano-id {
            font-family: monospace;
            background: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            display: inline-block;
        }
        .pano-id:hover {
            background: #f0f0f0;
        }
        .histogram-container {
            margin: 20px 0;
            background: white;
            border-radius: 6px;
            padding: 15px;
            border: 1px solid #e0e0e0;
        }
        .sat-list {
            margin-top: 30px;
        }
        .sat-item {
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
        }
        .sat-item:hover {
            border-color: #4CAF50;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sat-item.positive {
            border-left: 5px solid #4CAF50;
        }
        .sat-item.semipositive {
            border-left: 5px solid #FFC107;
        }
        .sat-item.negative {
            border-left: 5px solid #999;
        }
        .sat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        .badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            color: white;
        }
        .badge.positive {
            background-color: #4CAF50;
        }
        .badge.semipositive {
            background-color: #FFC107;
            color: #333;
        }
        .badge.negative {
            background-color: #999;
        }
        .sat-details {
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }
        .sat-item.expanded .sat-details {
            display: block;
        }
        .expandable-section {
            margin-top: 30px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }
        .section-header {
            background: #f8f9fa;
            padding: 15px;
            cursor: pointer;
            font-weight: 600;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .section-header:hover {
            background: #e9ecef;
        }
        .section-content {
            display: none;
            padding: 20px;
        }
        .expandable-section.expanded .section-content {
            display: block;
        }
        .expandable-section.expanded .section-header::after {
            content: '▼';
        }
        .section-header::after {
            content: '▶';
            font-size: 12px;
        }
        .raw-data {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .error {
            background-color: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .comparison-panel {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
        }
        .comparison-panel h3 {
            margin-top: 0;
        }
        .metrics {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        .sentence-item {
            background: #f8f9fa;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 4px;
            border-left: 3px solid #4CAF50;
        }
        .sentence-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            cursor: pointer;
        }
        .sentence-header:hover {
            background: #e9ecef;
        }
        .sentence-text {
            color: #333;
            line-height: 1.4;
            font-size: 13px;
            flex: 1;
        }
        .sentence-class {
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            flex-shrink: 0;
        }
        .no-sentences {
            color: #999;
            font-style: italic;
            padding: 10px;
            text-align: center;
            font-size: 13px;
        }
        .osm-tags {
            display: none;
            margin-top: 8px;
            padding: 8px;
            background: #fff;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
        .osm-tags.expanded {
            display: block;
        }
        .osm-tag {
            display: inline-block;
            margin: 2px 4px 2px 0;
            padding: 2px 6px;
            background: #e9ecef;
            border-radius: 3px;
            font-size: 11px;
            font-family: monospace;
        }
        .osm-tag-key {
            color: #0066cc;
            font-weight: 600;
        }
        .osm-tag-value {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Batch Inspector</h1>
            <div id="batchInfo"></div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label for="batchSelect">Batch:</label>
                <select id="batchSelect" onchange="loadBatch()">
                    <option value="">Loading...</option>
                </select>
            </div>
            <div class="control-group">
                <label for="panoSelect">Panorama:</label>
                <select id="panoSelect" onchange="loadPano()">
                    <option value="">Select batch first</option>
                </select>
            </div>
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="prevPano()">← Previous</button>
            </div>
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="nextPano()">Next →</button>
            </div>
        </div>

        <div id="content">
            <div class="loading">Select a batch and panorama to begin</div>
        </div>
    </div>

    <script>
        let currentBatch = null;
        let currentPano = null;
        let batchMetadata = null;
        let panoData = null;
        let classNames = {{ class_names | tojson }};

        // Load batch list on page load
        fetch('/api/batches')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('batchSelect');
                select.innerHTML = '<option value="">Select batch...</option>';
                data.batches.forEach(batch => {
                    const option = document.createElement('option');
                    option.value = batch;
                    option.textContent = batch;
                    select.appendChild(option);
                });
            });

        function loadBatch() {
            const batchSelect = document.getElementById('batchSelect');
            currentBatch = batchSelect.value;

            if (!currentBatch) return;

            fetch(`/api/batch/${currentBatch}`)
                .then(r => r.json())
                .then(data => {
                    batchMetadata = data;
                    const panoSelect = document.getElementById('panoSelect');
                    panoSelect.innerHTML = '';
                    for (let i = 0; i < data.num_panos; i++) {
                        const option = document.createElement('option');
                        option.value = i;
                        option.textContent = `Pano ${i}`;
                        panoSelect.appendChild(option);
                    }
                    document.getElementById('batchInfo').textContent =
                        `Epoch ${data.epoch_idx}, Batch ${data.batch_idx}`;
                    panoSelect.selectedIndex = 0;
                    loadPano();
                });
        }

        function loadPano() {
            const panoSelect = document.getElementById('panoSelect');
            currentPano = parseInt(panoSelect.value);

            if (currentPano === null || isNaN(currentPano)) return;

            document.getElementById('content').innerHTML = '<div class="loading">Loading...</div>';

            fetch(`/api/pano/${currentBatch}/${currentPano}`)
                .then(r => r.json())
                .then(data => {
                    panoData = data;
                    renderPanoView(data);
                })
                .catch(err => {
                    document.getElementById('content').innerHTML =
                        `<div class="error">Error loading data: ${err}</div>`;
                });
        }

        function renderPanoView(data) {
            let html = `
                <div class="pano-info">
                    <h2>Panorama ${currentPano}</h2>
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="info-label">Pano ID</span>
                            <span class="info-value">
                                <span class="pano-id" onclick="copyToClipboard('${data.pano_id}')" title="Click to copy">
                                    ${data.pano_id}
                                </span>
                            </span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Location</span>
                            <span class="info-value">${data.lat.toFixed(6)}, ${data.lon.toFixed(6)}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Valid Landmarks</span>
                            <span class="info-value">${data.num_valid_landmarks}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Positive Matches</span>
                            <span class="info-value">${data.positive_sat_idxs.length}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Semipositive Matches</span>
                            <span class="info-value">${data.semipositive_sat_idxs.length}</span>
                        </div>
                    </div>
                </div>

                <div class="histogram-container">
                    <h3>Panorama Semantic Class Histogram</h3>
                    <div id="panoHistogram"></div>
                </div>

                <div class="histogram-container">
                    <h3>Panorama Landmark Descriptions</h3>
                    <div id="panoSentences"></div>
                </div>

                <div class="sat-list">
                    <h2>Related Satellite Patches</h2>
                    <div id="satList"></div>
                </div>

                <div class="expandable-section" id="rawDataSection">
                    <div class="section-header" onclick="toggleSection('rawDataSection')">
                        Raw Panorama Data
                    </div>
                    <div class="section-content">
                        <h4>Features (shape: ${JSON.stringify(data.features_shape)})</h4>
                        <div class="raw-data">${formatFeatures(data.raw_features, data.raw_mask)}</div>
                        <h4>Positions (shape: ${JSON.stringify(data.positions_shape)})</h4>
                        <div class="raw-data">${formatPositions(data.raw_positions, data.raw_mask)}</div>
                        <h4>Mask</h4>
                        <div class="raw-data">${JSON.stringify(data.raw_mask)}</div>
                    </div>
                </div>
            `;

            document.getElementById('content').innerHTML = html;

            // Plot panorama histogram
            plotHistogram('panoHistogram', data.histogram, 'Panorama Landmarks');

            // Render panorama sentences
            renderSentences('panoSentences', data.sentences, 'Landmark');

            // Load and render satellite patches
            renderSatelliteList(data);
        }

        function renderSatelliteList(panoData) {
            const satList = document.getElementById('satList');

            // Determine which satellites to show
            const satIdxs = new Set();
            if (panoData.associated_sat_idx !== null) {
                satIdxs.add(panoData.associated_sat_idx);
            }
            panoData.positive_sat_idxs.forEach(idx => satIdxs.add(idx));
            panoData.semipositive_sat_idxs.forEach(idx => satIdxs.add(idx));

            // Limit to first 20 to avoid overwhelming the UI
            const satIdxArray = Array.from(satIdxs).slice(0, 20);

            satList.innerHTML = `<div class="loading">Loading 0 of ${satIdxArray.length} satellites...</div>`;

            // Load satellites sequentially with progress updates
            let loadedCount = 0;
            const satDataList = [];

            const loadNext = (idx) => {
                if (idx >= satIdxArray.length) {
                    // All loaded, render the list
                    renderSatelliteItems(satDataList, satIdxArray, panoData);
                    return;
                }

                const satIdx = satIdxArray[idx];
                fetch(`/api/sat/${currentBatch}/${satIdx}`)
                    .then(r => r.json())
                    .then(data => {
                        satDataList.push(data);
                        loadedCount++;
                        satList.innerHTML = `<div class="loading">Loading ${loadedCount} of ${satIdxArray.length} satellites...</div>`;
                        loadNext(idx + 1);
                    })
                    .catch(err => {
                        console.error(`Error loading satellite ${satIdx}:`, err);
                        loadNext(idx + 1);
                    });
            };

            loadNext(0);
        }

        function renderSatelliteItems(satDataList, satIdxArray, panoData) {
            Promise.resolve().then(() => {
                let html = '';
                satDataList.forEach((satData, i) => {
                    const idx = satIdxArray[i];
                    let matchType = 'negative';
                    let matchLabel = 'Negative';

                    if (panoData.positive_sat_idxs.includes(idx)) {
                        matchType = 'positive';
                        matchLabel = '✓ Positive';
                    } else if (panoData.semipositive_sat_idxs.includes(idx)) {
                        matchType = 'semipositive';
                        matchLabel = '~ Semipositive';
                    }

                    html += `
                        <div class="sat-item ${matchType} expanded">
                            <div class="sat-header" onclick="toggleSat(this.parentElement)">
                                <div>
                                    <strong>Satellite ${idx}</strong>
                                    <span class="badge ${matchType}">${matchLabel}</span>
                                    <br>
                                    <small>${satData.lat.toFixed(6)}, ${satData.lon.toFixed(6)}</small>
                                    <small>| ${satData.num_valid_tokens} tokens</small>
                                </div>
                            </div>
                            <div class="sat-details">
                                <div id="satHist${idx}"></div>
                                <div style="margin-top: 20px;">
                                    <h4>Landmark Descriptions</h4>
                                    <div id="satSentences${idx}"></div>
                                </div>
                                <div class="expandable-section" id="satRaw${idx}">
                                    <div class="section-header" onclick="event.stopPropagation(); toggleSection('satRaw${idx}')">
                                        Raw Satellite Data
                                    </div>
                                    <div class="section-content">
                                        <h4>Features (shape: ${JSON.stringify(satData.features_shape)})</h4>
                                        <div class="raw-data">${formatFeatures(satData.raw_features, satData.raw_mask)}</div>
                                        <h4>Positions (shape: ${JSON.stringify(satData.positions_shape)})</h4>
                                        <div class="raw-data">${formatPositions(satData.raw_positions, satData.raw_mask)}</div>
                                        <h4>Mask</h4>
                                        <div class="raw-data">${JSON.stringify(satData.raw_mask)}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                satList.innerHTML = html;

                // Plot satellite histograms and render sentences
                satDataList.forEach((satData, i) => {
                    const idx = satIdxArray[i];
                    plotHistogram(`satHist${idx}`, satData.histogram, `Satellite ${idx} Tokens`);
                    renderSentences(`satSentences${idx}`, satData.sentences, 'Token');
                });
            });
        }

        function renderSentences(divId, sentences, itemType) {
            const container = document.getElementById(divId);
            if (!sentences || sentences.length === 0) {
                container.innerHTML = '<div class="no-sentences">No descriptions available</div>';
                return;
            }

            let html = '';
            sentences.forEach((item, idx) => {
                // Find the class index to match histogram colors
                const classIdx = classNames.indexOf(item.semantic_class);
                const color = classIdx >= 0 ? `hsl(${classIdx * 10}, 70%, 50%)` : '#999';

                // Generate OSM tags HTML if available
                let tagsHtml = '';
                if (item.osm_tags && Object.keys(item.osm_tags).length > 0) {
                    tagsHtml = '<div class="osm-tags" id="tags-' + idx + '">';
                    for (const [key, value] of Object.entries(item.osm_tags)) {
                        tagsHtml += `<span class="osm-tag"><span class="osm-tag-key">${key}</span>=<span class="osm-tag-value">${value}</span></span>`;
                    }
                    tagsHtml += '</div>';
                }

                html += `
                    <div class="sentence-item">
                        <div class="sentence-header" onclick="toggleTags(${idx})">
                            <div class="sentence-text">${item.sentence}</div>
                            <span class="sentence-class" style="background-color: ${color}">${item.semantic_class}</span>
                        </div>
                        ${tagsHtml}
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function toggleTags(idx) {
            const tagsElement = document.getElementById('tags-' + idx);
            if (tagsElement) {
                tagsElement.classList.toggle('expanded');
            }
        }

        function plotHistogram(divId, histogram, title) {
            const nonZeroIndices = histogram.map((val, idx) => val > 0 ? idx : -1).filter(idx => idx >= 0);

            const trace = {
                x: nonZeroIndices.map(idx => classNames[idx]),
                y: nonZeroIndices.map(idx => histogram[idx]),
                type: 'bar',
                marker: {
                    color: nonZeroIndices.map(idx => `hsl(${idx * 10}, 70%, 50%)`)
                }
            };

            const layout = {
                title: title,
                xaxis: { title: 'Semantic Class' },
                yaxis: { title: 'Count' },
                margin: { t: 40, b: 120, l: 60, r: 20 },
                height: 350
            };

            Plotly.newPlot(divId, [trace], layout);
        }

        function toggleSat(element) {
            element.classList.toggle('expanded');
        }

        function toggleSection(sectionId) {
            document.getElementById(sectionId).classList.toggle('expanded');
        }

        function formatFeatures(features, mask) {
            if (!Array.isArray(features) || features.length === 0) {
                return JSON.stringify(features);
            }
            // Show all rows with color coding based on mask
            let html = '<div style="font-family: monospace; white-space: pre-wrap;">';
            features.forEach((row, idx) => {
                const isInvalid = mask[idx];
                const style = isInvalid ? 'color: #999; background-color: #f5f5f5;' : '';
                html += `<div style="${style}">${JSON.stringify(row)}</div>`;
            });
            html += '</div>';
            return html;
        }

        function formatPositions(positions, mask) {
            if (!Array.isArray(positions) || positions.length === 0) {
                return JSON.stringify(positions);
            }
            // For each position, show first dimension on its line, rest flattened
            let html = '<div style="font-family: monospace; white-space: pre-wrap;">';
            positions.forEach((pos, idx) => {
                const isInvalid = mask[idx];
                const style = isInvalid ? 'color: #999; background-color: #f5f5f5;' : '';
                if (Array.isArray(pos) && pos.length > 0) {
                    // First dimension
                    html += `<div style="${style}">[${pos[0]}]`;
                    // Rest flattened
                    if (pos.length > 1) {
                        const rest = pos.slice(1).flat(Infinity);
                        html += ` ${JSON.stringify(rest)}`;
                    }
                    html += '</div>';
                } else {
                    html += `<div style="${style}">${JSON.stringify(pos)}</div>`;
                }
            });
            html += '</div>';
            return html;
        }

        function copyToClipboard(text) {
            navigator.clipboard.writeText(text);
        }

        function prevPano() {
            if (currentPano > 0) {
                document.getElementById('panoSelect').selectedIndex = currentPano - 1;
                loadPano();
            }
        }

        function nextPano() {
            const select = document.getElementById('panoSelect');
            if (currentPano < select.options.length - 1) {
                select.selectedIndex = currentPano + 1;
                loadPano();
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                prevPano();
            } else if (e.key === 'ArrowRight') {
                nextPano();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE, class_names=inspector.semantic_class_names)


@app.route('/api/batches')
def api_batches():
    """Get list of available batches."""
    return jsonify({
        'batches': inspector.get_batch_list()
    })


@app.route('/api/batch/<batch_name>')
def api_batch_metadata(batch_name):
    """Get metadata for a specific batch."""
    try:
        metadata = inspector.get_batch_metadata(batch_name)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/pano/<batch_name>/<int:pano_idx>')
def api_pano_data(batch_name, pano_idx):
    """Get data for a specific panorama."""
    try:
        data = inspector.get_pano_data(batch_name, pano_idx)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sat/<batch_name>/<int:sat_idx>')
def api_sat_data(batch_name, sat_idx):
    """Get data for a specific satellite patch."""
    try:
        data = inspector.get_sat_data(batch_name, sat_idx)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def main():
    parser = argparse.ArgumentParser(description='Inspect captured model input batches')
    parser.add_argument('--captures_dir', type=str, required=True,
                        help='Directory containing captured batch files')
    parser.add_argument('--semantic_classes', type=str, required=True,
                        help='Path to semantic class grouping JSON file')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the web server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind the web server to')
    parser.add_argument('--satellite_sentences_dir', type=str, default=None,
                        help='Directory containing satellite landmark sentence JSONL files')

    args = parser.parse_args()

    # Initialize inspector
    global inspector
    satellite_dir = Path(args.satellite_sentences_dir) if args.satellite_sentences_dir else None
    inspector = BatchInspector(Path(args.captures_dir), Path(args.semantic_classes), satellite_dir)

    print(f"Starting Batch Inspector on http://{args.host}:{args.port}")
    print(f"Captures directory: {args.captures_dir}")
    print(f"Found {len(inspector.batch_files)} batch files")
    print(f"Loaded {len(inspector.semantic_class_names)} semantic classes")

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
