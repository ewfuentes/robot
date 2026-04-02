#!/usr/bin/env python3
"""
Simple web app to view panoramas, pinhole images, and landmark sentences with similarity comparison.

Usage:
    python panorama_viewer.py \
        --panorama_dir /data/overhead_matching/datasets/VIGOR/Chicago/panorama \
        --pinhole_dir /data/overhead_matching/datasets/pinhole_images/Chicagojpg \
        --pano_landmarks_dir /data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/Chicago \
        --osm_landmarks_dir /data/overhead_matching/datasets/semantic_landmark_embeddings/v2 \
        --osm_landmarks_geojson /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v3.geojson

Note: pano_landmarks_dir and osm_landmarks_dir should contain 'sentences/' and 'embeddings/' subdirectories.
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template_string, send_file, jsonify, request
import base64
import hashlib
import pandas as pd
import pickle
import time
from dataclasses import dataclass
from typing import Optional
import common.torch.load_torch_deps
import torch

# Import from existing modules
from experimental.overhead_matching.swag.data.vigor_dataset import load_landmark_geojson
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark, custom_id_from_props)
from common.gps import web_mercator

# Correspondence model imports (lazy-used when --correspondence_model_path provided)
from experimental.overhead_matching.swag.evaluation.correspondence_matching import (
    compute_cost_matrix, match_and_aggregate, MatchingMethod, AggregationMode,
)
from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    load_text_embeddings,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier, CorrespondenceClassifierConfig, TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    extract_panorama_data_across_cities,
)
from experimental.overhead_matching.swag.scripts.landmark_pairing_cli import (
    extract_tags_from_pano_data,
)


@dataclass
class PanoramaLandmark:
    """Represents a landmark visible in a panorama."""
    description: str
    landmark_id: tuple[str, int]  # (panorama_id, landmark_index)
    yaws: list[int]  # Yaw angles where this landmark is visible
    panorama_lat: float
    panorama_lon: float
    primary_tag: Optional[dict] = None  # v2: {"key": "highway", "value": "secondary"}
    additional_tags: list[dict] = None  # v2: [{"key": ..., "value": ...}, ...]


@dataclass
class OSMLandmark:
    """Represents an OpenStreetMap landmark."""
    custom_id: str  # Base64 hash identifier
    description: str
    properties: dict  # OSM tags (name, amenity, etc.)
    lat: float
    lon: float


app = Flask(__name__)

# Global data
PANORAMA_DATA = []

# Panorama landmark data
PANO_SENTENCES: dict[str, list[PanoramaLandmark]] = {}  # pano_id -> landmarks
PANO_EMBEDDINGS: Optional[torch.Tensor] = None
PANO_EMBEDDING_INDEX: dict[tuple[str, int], int] = {}  # landmark_id -> tensor row
PANO_INDEX_REVERSE: list[tuple[str, int]] = []  # tensor row -> landmark_id

# OSM landmark data (independent, no duplication)
OSM_LANDMARKS: dict[str, OSMLandmark] = {}  # custom_id -> landmark
OSM_EMBEDDINGS: Optional[torch.Tensor] = None
OSM_EMBEDDING_INDEX: dict[str, int] = {}  # custom_id -> tensor row
OSM_INDEX_REVERSE: list[str] = []  # tensor row -> custom_id

# Pre-computed associations (stores indices only, no data duplication)
PANO_TO_OSM: dict[str, list[str]] = {}  # pano_id -> list of OSM custom_ids

# Tag-weight similarity data (loaded when --similarity_matrix is provided)
SIMILARITY_MATRIX: Optional[torch.Tensor] = None  # (num_panos, num_sats)
VIGOR_DATASET = None  # VigorDataset instance
PANO_OSM_MATCHES = None  # polars DataFrame
SAT_OSM_TABLE = None  # polars DataFrame
PANO_ID_TO_VIGOR_IDX: dict[str, int] = {}  # pano_id -> index in VigorDataset
MRR_RANKING: list[dict] = []  # sorted list of {pano_idx, pano_id, mrr, best_rank}

# Correspondence model data (loaded when --correspondence_model_path is provided)
CORRESPONDENCE_MODEL = None
CORRESPONDENCE_TEXT_EMBEDDINGS = None
CORRESPONDENCE_TEXT_INPUT_DIM = None
CORRESPONDENCE_DEVICE = None
PANO_TAGS_FROM_PANO_ID: dict[str, list[dict]] = {}

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Panorama Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .nav-buttons button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            margin: 0 5px;
        }
        .nav-buttons button:hover {
            background: #0056b3;
        }
        .panorama-section {
            margin-bottom: 30px;
        }
        .panorama-section h2 {
            color: #333;
            margin-bottom: 15px;
        }
        .panorama-img {
            width: 100%;
            max-width: 1600px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .pinhole-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        .pinhole-item {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
        }
        .pinhole-item img {
            width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        .pinhole-item h3 {
            margin: 0 0 15px 0;
            color: #555;
            font-size: 16px;
            text-align: center;
        }
        .pinhole-sentences {
            margin-top: 10px;
        }
        .pinhole-sentences ul {
            margin: 0;
            padding-left: 20px;
            font-size: 13px;
        }
        .pinhole-sentences li {
            margin-bottom: 5px;
            line-height: 1.4;
        }
        .landmark-all-mode {
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 3px;
            border-left: 4px solid;
        }
        .panorama-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #eee;
        }
        .info-text {
            color: #666;
            font-size: 14px;
        }
        .keyboard-hint {
            color: #999;
            font-size: 12px;
            margin-top: 10px;
        }
        .similarity-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px solid #eee;
        }
        .similarity-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }
        .landmark-column {
            background: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
        }
        .landmark-column h3 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .landmark-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .landmark-item {
            padding: 12px;
            margin-bottom: 8px;
            background: white;
            border-radius: 4px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s;
        }
        .landmark-item:hover {
            border-color: #007bff;
            box-shadow: 0 2px 4px rgba(0,123,255,0.1);
        }
        .landmark-item.selected {
            border-color: #28a745;
            background: #e8f5e9;
        }
        .landmark-desc {
            font-size: 14px;
            color: #333;
            margin-bottom: 6px;
        }
        .landmark-meta {
            font-size: 12px;
            color: #666;
        }
        .similarity-badge {
            display: inline-block;
            background: #ff9800;
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }
        .similarity-badge.high {
            background: #4caf50;
        }
        .similarity-badge.medium {
            background: #ff9800;
        }
        .similarity-badge.low {
            background: #9e9e9e;
        }
        /* Satellite ranking styles */
        .sat-ranking-header {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        .mrr-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            color: white;
        }
        .mrr-badge.good { background: #4caf50; }
        .mrr-badge.medium { background: #ff9800; }
        .mrr-badge.bad { background: #f44336; }
        .sat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        .sat-card {
            position: relative;
            border: 3px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sat-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .sat-card.positive { border-color: #4caf50; }
        .sat-card.negative { border-color: #f44336; }
        .sat-card img {
            width: 100%;
            display: block;
        }
        .sat-card-info {
            padding: 6px 8px;
            font-size: 11px;
            background: #f5f5f5;
        }
        .sat-card-rank {
            position: absolute;
            top: 4px;
            left: 4px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
        }
        .sat-detail-panel {
            margin-top: 10px;
            padding: 12px;
            background: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #ddd;
            display: none;
        }
        .sat-detail-panel table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .sat-detail-panel th, .sat-detail-panel td {
            padding: 4px 8px;
            border-bottom: 1px solid #eee;
            text-align: left;
        }
        .sat-detail-panel th { font-weight: 600; color: #555; }
        /* Landmark search styles */
        .search-section {
            margin-bottom: 20px;
            padding: 15px;
            background: #f0f8ff;
            border-radius: 6px;
            border: 1px solid #d0e8ff;
        }
        .search-section.collapsed .search-results { display: none; }
        .search-input-row {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .search-input-row input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        .search-results {
            margin-top: 10px;
            max-height: 300px;
            overflow-y: auto;
        }
        .search-results table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .search-results th, .search-results td {
            padding: 6px 10px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        .search-results th { background: #e8f0fe; font-weight: 600; }
        /* Navigation mode selector */
        .nav-mode-selector {
            margin-top: 8px;
            font-size: 13px;
        }
        .nav-mode-selector select {
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Panorama Viewer</h1>
                <div class="info-text">
                    <span id="pano-name">Loading...</span>
                    (<span id="pano-index">0</span> of <span id="pano-total">0</span>)
                    <span id="pano-location"></span>
                </div>
                <div class="keyboard-hint">Use ← → arrow keys to navigate</div>
            </div>
            <div class="nav-buttons">
                <button onclick="navigate(-1)">← Previous</button>
                <button onclick="navigate(1)">Next →</button>
                <div style="margin-top: 10px; display: flex; gap: 8px; align-items: center;">
                    <input type="text" id="pano-search-input" placeholder="Panorama ID..."
                           style="padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; width: 250px;"
                           onkeypress="if(event.key === 'Enter') seekToPanorama()">
                    <button onclick="seekToPanorama()" style="white-space: nowrap;">Go to Panorama</button>
                    <span id="seek-status" style="font-size: 12px; color: #dc3545;"></span>
                </div>
                <div class="nav-mode-selector" id="nav-mode-container" style="display:none;">
                    Navigate: <select id="nav-mode" onchange="changeNavMode(this.value)">
                        <option value="sequential">Sequential</option>
                        <option value="best_mrr">Best MRR first</option>
                        <option value="worst_mrr">Worst MRR first</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- OSM Landmark Search (only when similarity matrix loaded) -->
        <div class="search-section" id="landmark-search-section" style="display:none;">
            <div class="search-input-row">
                <strong>Search OSM Landmarks:</strong>
                <input type="text" id="landmark-search-input" placeholder="e.g. street, building, restaurant..."
                       onkeypress="if(event.key === 'Enter') searchLandmarks()">
                <button onclick="searchLandmarks()">Search</button>
            </div>
            <div class="search-results" id="landmark-search-results"></div>
        </div>

        <div>
            <h2>Pinhole Views</h2>
            <div style="margin-bottom: 15px; padding: 10px; background: #f0f8ff; border-radius: 4px; font-size: 13px;">
                <strong>Legend:</strong>
                <span style="margin-left: 10px;">
                    Landmarks with matching colors and yaw badges (e.g., <span style="background:#ddd;padding:2px 4px;border-radius:2px;">90°</span>) appear in multiple views.
                </span>
            </div>
            <div class="pinhole-grid" id="pinhole-grid">
                <!-- Pinhole images will be inserted here -->
            </div>
        </div>

        <div class="panorama-section">
            <h2>Nearby OSM Landmarks</h2>
            <div id="osm-landmarks-container" style="columns: 2; column-gap: 20px;">
                <!-- OSM landmarks will be inserted here -->
            </div>
        </div>

        <!-- Satellite Ranking (only when similarity matrix loaded) -->
        <div class="panorama-section" id="sat-ranking-section" style="display:none;">
            <div class="sat-ranking-header">
                <h2>Satellite Ranking (Tag-Weight Similarity)</h2>
                <span class="mrr-badge" id="mrr-badge"></span>
                <span id="mrr-detail" style="font-size:13px;color:#666;"></span>
            </div>
            <div style="display:flex;gap:30px;flex-wrap:wrap;">
                <div style="flex:1;min-width:300px;">
                    <h3 style="margin:8px 0;color:#555;font-size:14px;">Top Ranked</h3>
                    <div class="sat-grid" id="sat-grid">
                        <!-- Top satellite cards inserted by JS -->
                    </div>
                </div>
                <div id="positive-sat-section" style="flex:1;min-width:300px;display:none;">
                    <h3 style="margin:8px 0;color:#4caf50;font-size:14px;">Ground Truth Positives</h3>
                    <div class="sat-grid" id="positive-sat-grid">
                        <!-- Positive satellite cards inserted by JS -->
                    </div>
                </div>
            </div>
            <div class="sat-detail-panel" id="sat-detail-panel">
                <h4 id="sat-detail-title"></h4>
                <table id="sat-detail-table">
                    <thead><tr><th>Tag Key</th><th>Pano Value</th><th>Sat Value</th></tr></thead>
                    <tbody></tbody>
                </table>
                <div id="sat-osm-landmarks-section"></div>
                <div id="sat-correspondence-section"></div>
            </div>

            <!-- Similarity histogram -->
            <div id="sim-histogram-container" style="margin-top:15px;">
                <h3 style="margin:8px 0;color:#555;font-size:14px;">Observation Likelihood Distribution</h3>
                <canvas id="sim-histogram-canvas" style="width:100%;height:250px;border:1px solid #ddd;border-radius:6px;cursor:crosshair;"></canvas>
                <div id="sim-histogram-tooltip" style="display:none;position:absolute;background:white;border:1px solid #ccc;border-radius:6px;padding:10px;box-shadow:0 2px 8px rgba(0,0,0,0.15);max-width:350px;z-index:1000;font-size:12px;pointer-events:none;"></div>
            </div>

            <!-- Map of satellite match locations -->
            <div id="sat-map-container" style="margin-top:15px;">
                <h3 style="margin:8px 0;color:#555;font-size:14px;">Top-50 Satellite Match Locations</h3>
                <div id="sat-map" style="height:400px;border:1px solid #ddd;border-radius:6px;"></div>
            </div>
        </div>

        <div class="panorama-section">
            <h2>Landmark Similarity Comparison</h2>
            <div class="similarity-section">
                <div class="similarity-container">
                    <div class="landmark-column">
                        <h3>Panorama Landmarks <span id="pano-local-label">(Local)</span></h3>
                        <ul class="landmark-list" id="pano-comparison-list">
                            <!-- Populated by JavaScript -->
                        </ul>
                    </div>
                    <div class="landmark-column">
                        <h3>OSM Landmarks <span id="osm-local-label">(Local)</span></h3>
                        <ul class="landmark-list" id="osm-comparison-list">
                            <!-- Populated by JavaScript -->
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Global matches section -->
            <div id="global-matches-section" style="margin-top: 20px; display: none;">
                <h3 id="global-matches-title">Global Matches</h3>
                <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: #fafafa;">
                    <ul class="landmark-list" id="global-matches-list">
                        <!-- Populated by JavaScript -->
                    </ul>
                </div>
            </div>
        </div>

        <div class="panorama-section">
            <h2>Full Panorama</h2>
            <img id="panorama-img" class="panorama-img" src="" alt="Panorama">
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let totalPanoramas = 0;
        let selectedLandmark = null; // {type: 'pano'|'osm', key: string}
        let currentSimilarityData = null; // Store current similarity data
        let globalMatches = null; // Store global matches for selected landmark
        let panoIdToIndex = {}; // Map panorama ID to index for navigation
        let hasSimilarityMatrix = false; // Whether similarity matrix is loaded
        let hasCorrespondenceModel = false; // Whether correspondence model is loaded
        let navMode = 'sequential'; // 'sequential', 'best_mrr', 'worst_mrr'
        let mrrRanking = null; // Sorted panorama indices by MRR
        let mrrRankingReverse = null; // Reverse sorted

        // Color palette for landmarks (distinct colors)
        const LANDMARK_COLORS = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84',
            '#6C5B7B', '#355C7D', '#F67280', '#C8D6AF', '#8E7AB5'
        ];

        function getLandmarkColor(landmarkId) {
            return LANDMARK_COLORS[landmarkId % LANDMARK_COLORS.length];
        }

        function getSimilarityClass(score) {
            if (score >= 0.8) return 'high';
            if (score >= 0.6) return 'medium';
            return 'low';
        }

        function navigateToPanoramaById(panoId) {
            const index = panoIdToIndex[panoId];
            if (index !== undefined) {
                loadPanorama(index);
            } else {
                console.warn('Panorama not found:', panoId);
            }
        }

        // Make function globally accessible
        window.navigateToPanoramaById = navigateToPanoramaById;

        function seekToPanorama() {
            const input = document.getElementById('pano-search-input');
            const statusElement = document.getElementById('seek-status');
            const panoId = input.value.trim();

            if (!panoId) {
                statusElement.textContent = 'Please enter a panorama ID';
                statusElement.style.color = '#dc3545';
                return;
            }

            const index = panoIdToIndex[panoId];
            if (index !== undefined) {
                loadPanorama(index);
                statusElement.textContent = '✓ Found';
                statusElement.style.color = '#28a745';
                input.value = ''; // Clear input on success
                setTimeout(() => { statusElement.textContent = ''; }, 2000);
            } else {
                statusElement.textContent = '✗ Panorama not found';
                statusElement.style.color = '#dc3545';
            }
        }

        // Make function globally accessible
        window.seekToPanorama = seekToPanorama;

        function makeClickablePanoId(panoId) {
            // Escape single quotes in panoId for onclick string
            const escapedId = panoId.replace(/'/g, "\\'");
            return `<a href="#" onclick="navigateToPanoramaById('${escapedId}'); return false;" style="color:#007bff;text-decoration:underline;cursor:pointer;">${panoId}</a>`;
        }

        function updateGlobalMatches() {
            const section = document.getElementById('global-matches-section');
            const title = document.getElementById('global-matches-title');
            const list = document.getElementById('global-matches-list');

            if (!selectedLandmark || !globalMatches || globalMatches.length === 0) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';

            if (selectedLandmark.type === 'osm') {
                title.textContent = 'Global Matches: Panorama Landmarks Most Similar to Selected OSM';
                list.innerHTML = '';

                globalMatches.forEach(match => {
                    const li = document.createElement('li');
                    li.className = 'landmark-item';
                    li.style.cursor = 'default';

                    const simBadge = `<span class="similarity-badge ${getSimilarityClass(match.similarity)}">${(match.similarity * 100).toFixed(1)}%</span>`;
                    const panoInfo = `<span style="color: #666; font-size: 12px; margin-left: 8px;">(${makeClickablePanoId(match.pano_id)})</span>`;

                    li.innerHTML = `${match.description} ${simBadge} ${panoInfo}`;
                    list.appendChild(li);
                });
            } else {
                title.textContent = 'Global Matches: OSM Landmarks Most Similar to Selected Panorama Landmark';
                list.innerHTML = '';

                globalMatches.forEach(match => {
                    const li = document.createElement('li');
                    li.className = 'landmark-item';
                    li.style.cursor = 'default';

                    const simBadge = `<span class="similarity-badge ${getSimilarityClass(match.similarity)}">${(match.similarity * 100).toFixed(1)}%</span>`;

                    // Show nearby panoramas if available
                    let panoInfo = '';
                    if (match.nearby_panos && match.nearby_panos.length > 0) {
                        const clickablePanos = match.nearby_panos.slice(0, 3).map(id => makeClickablePanoId(id)).join(', ');
                        const moreCount = match.nearby_panos.length > 3 ? ` +${match.nearby_panos.length - 3} more` : '';
                        panoInfo = `<span style="color: #666; font-size: 12px; margin-left: 8px;">(Near: ${clickablePanos}${moreCount})</span>`;
                    }

                    li.innerHTML = `${match.description} ${simBadge} ${panoInfo}`;
                    list.appendChild(li);
                });
            }
        }

        function updateComparisonLists(similarityData, selectedType = null, selectedKey = null) {
            const panoList = document.getElementById('pano-comparison-list');
            const osmList = document.getElementById('osm-comparison-list');

            if (!similarityData || !similarityData.pano_landmarks || !similarityData.osm_landmarks) {
                panoList.innerHTML = '<li style="color:#999;font-style:italic;">No data available</li>';
                osmList.innerHTML = '<li style="color:#999;font-style:italic;">No data available</li>';
                return;
            }

            // Update panorama landmarks list
            panoList.innerHTML = '';

            // Build list of pano landmarks with their scores
            const panoItems = similarityData.pano_landmarks.map(pano => {
                const panoKey = pano.key;  // Already in format "pano_id:idx"
                let similarityScore = 0;

                if (selectedType === 'osm' && selectedKey) {
                    // Show similarity to selected OSM landmark
                    const osmToPano = similarityData.osm_to_pano[selectedKey] || [];
                    const match = osmToPano.find(m => m.pano_key === panoKey);
                    if (match) {
                        similarityScore = match.similarity;
                    }
                } else {
                    // Show highest similarity to any OSM landmark
                    const panoToOsm = similarityData.pano_to_osm[panoKey] || [];
                    if (panoToOsm.length > 0) {
                        similarityScore = panoToOsm[0].similarity;
                    }
                }

                return {pano, panoKey, similarityScore};
            });

            // Sort by similarity if a landmark is selected
            if (selectedType === 'osm' && selectedKey) {
                panoItems.sort((a, b) => b.similarityScore - a.similarityScore);
            }

            // Render pano landmarks
            panoItems.forEach(({pano, panoKey, similarityScore}) => {
                const li = document.createElement('li');
                li.className = 'landmark-item';
                if (selectedType === 'pano' && selectedKey === panoKey) {
                    li.classList.add('selected');
                }

                let matchInfo = '';
                if (similarityScore > 0) {
                    matchInfo = `<span class="similarity-badge ${getSimilarityClass(similarityScore)}">${(similarityScore * 100).toFixed(1)}%</span>`;
                }

                li.innerHTML = `${pano.description} ${matchInfo}`;
                li.dataset.type = 'pano';
                li.dataset.key = panoKey;
                li.addEventListener('click', () => handleLandmarkClick('pano', panoKey));
                panoList.appendChild(li);
            });

            // Update OSM landmarks list
            osmList.innerHTML = '';

            // Build list of OSM landmarks with their scores
            const osmItems = similarityData.osm_landmarks.map(osm => {
                const osmKey = osm.key;  // OSM custom_id
                let similarityScore = 0;

                if (selectedType === 'pano' && selectedKey) {
                    // Show similarity to selected pano landmark
                    const panoToOsm = similarityData.pano_to_osm[selectedKey] || [];
                    const match = panoToOsm.find(m => m.osm_key === osmKey);
                    if (match) {
                        similarityScore = match.similarity;
                    }
                } else {
                    // Show highest similarity to any pano landmark
                    const osmToPano = similarityData.osm_to_pano[osmKey] || [];
                    if (osmToPano.length > 0) {
                        similarityScore = osmToPano[0].similarity;
                    }
                }

                return {osm, osmKey, similarityScore};
            });

            // Sort by similarity if a landmark is selected
            if (selectedType === 'pano' && selectedKey) {
                osmItems.sort((a, b) => b.similarityScore - a.similarityScore);
            }

            // Render OSM landmarks
            osmItems.forEach(({osm, osmKey, similarityScore}) => {
                const li = document.createElement('li');
                li.className = 'landmark-item';
                if (selectedType === 'osm' && selectedKey === osmKey) {
                    li.classList.add('selected');
                }

                let matchInfo = '';
                if (similarityScore > 0) {
                    matchInfo = `<span class="similarity-badge ${getSimilarityClass(similarityScore)}">${(similarityScore * 100).toFixed(1)}%</span>`;
                }

                li.innerHTML = `${osm.description} ${matchInfo}`;
                li.dataset.type = 'osm';
                li.dataset.key = osmKey;
                li.addEventListener('click', () => handleLandmarkClick('osm', osmKey));
                osmList.appendChild(li);
            });

            // Update global matches section
            updateGlobalMatches();
        }

        function handleLandmarkClick(type, key) {
            // If clicking the same landmark, deselect it
            if (selectedLandmark && selectedLandmark.type === type && selectedLandmark.key === key) {
                selectedLandmark = null;
                globalMatches = null;
                updateComparisonLists(currentSimilarityData);
            } else {
                // Select the new landmark
                selectedLandmark = {type, key};

                // Fetch global matches
                fetch(`/api/global_matches/${type}/${encodeURIComponent(key)}`)
                    .then(r => r.json())
                    .then(data => {
                        globalMatches = data.matches;
                        updateComparisonLists(currentSimilarityData, type, key);
                    })
                    .catch(err => {
                        console.error('Error fetching global matches:', err);
                        globalMatches = [];
                        updateComparisonLists(currentSimilarityData, type, key);
                    });
            }
        }

        function changeNavMode(mode) {
            navMode = mode;
            if (mode === 'sequential') {
                loadPanorama(0);
            } else if (!mrrRanking) {
                // Fetch MRR ranking, then navigate to first in that order
                fetch('/api/panorama_mrr_ranking')
                    .then(r => r.json())
                    .then(data => {
                        mrrRanking = data.ranking; // best MRR first
                        mrrRankingReverse = [...mrrRanking].reverse(); // worst MRR first
                        const order = getNavigationOrder();
                        if (order && order.length > 0) loadPanorama(order[0].pano_idx);
                    });
            } else {
                // Already have ranking, jump to first
                const order = getNavigationOrder();
                if (order && order.length > 0) loadPanorama(order[0].pano_idx);
            }
        }

        function getNavigationOrder() {
            if (navMode === 'best_mrr' && mrrRanking) return mrrRanking;
            if (navMode === 'worst_mrr' && mrrRankingReverse) return mrrRankingReverse;
            return null; // sequential
        }

        function loadSatelliteRanking(index) {
            const section = document.getElementById('sat-ranking-section');
            if (!hasSimilarityMatrix) {
                section.style.display = 'none';
                return;
            }
            section.style.display = '';

            fetch('/api/satellite_ranking/' + index)
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        section.style.display = 'none';
                        return;
                    }

                    // Update MRR badge
                    const badge = document.getElementById('mrr-badge');
                    const detail = document.getElementById('mrr-detail');
                    if (data.mrr !== null) {
                        const mrr = data.mrr;
                        badge.textContent = 'MRR: ' + mrr.toFixed(4);
                        badge.className = 'mrr-badge ' + (mrr >= 0.5 ? 'good' : mrr >= 0.1 ? 'medium' : 'bad');
                        detail.textContent = `Best positive rank: ${data.best_rank + 1} / ${data.num_sats} (${data.num_positives} positives)`;
                    } else {
                        badge.textContent = 'No positives';
                        badge.className = 'mrr-badge bad';
                        detail.textContent = '';
                    }

                    // Build top-ranked satellite grid
                    const grid = document.getElementById('sat-grid');
                    grid.innerHTML = '';

                    data.satellites.forEach((sat, i) => {
                        const card = document.createElement('div');
                        card.className = 'sat-card ' + (sat.is_positive ? 'positive' : 'negative');
                        card.innerHTML = `
                            <span class="sat-card-rank">#${i + 1}</span>
                            <img src="/api/image/satellite/${sat.sat_idx}" alt="Satellite ${sat.sat_idx}" loading="lazy">
                            <div class="sat-card-info">
                                Score: ${sat.similarity_score.toFixed(4)}
                                ${sat.is_positive ? '<span style="color:#4caf50;font-weight:600;"> ✓</span>' : ''}
                            </div>
                        `;
                        card.addEventListener('click', () => showSatelliteDetail(sat, i + 1));
                        grid.appendChild(card);
                    });

                    // Build ground-truth positive satellites grid
                    const posSection = document.getElementById('positive-sat-section');
                    const posGrid = document.getElementById('positive-sat-grid');
                    posGrid.innerHTML = '';

                    // Include positives already in top-k + those not in top-k
                    const topPositives = data.satellites.filter(s => s.is_positive);
                    const extraPositives = data.positive_satellites || [];
                    const allPositives = [...topPositives, ...extraPositives];

                    if (allPositives.length > 0) {
                        posSection.style.display = '';
                        allPositives.forEach(sat => {
                            const rankLabel = sat.rank_label ? `Rank #${sat.rank_label}` : 'In top-k';
                            const card = document.createElement('div');
                            card.className = 'sat-card positive';
                            card.innerHTML = `
                                <span class="sat-card-rank" style="background:#4caf50;">${rankLabel}</span>
                                <img src="/api/image/satellite/${sat.sat_idx}" alt="Satellite ${sat.sat_idx}" loading="lazy">
                                <div class="sat-card-info">
                                    Score: ${sat.similarity_score.toFixed(4)}
                                </div>
                            `;
                            card.addEventListener('click', () => showSatelliteDetail(sat, rankLabel));
                            posGrid.appendChild(card);
                        });
                    } else {
                        posSection.style.display = 'none';
                    }
                })
                .catch(err => console.error('Error loading satellite ranking:', err));

            // Also load map data and histogram
            loadSatelliteMap(index);
            loadSimilarityHistogram(index);
        }

        let satMap = null;
        let satMapMarkers = [];

        function loadSatelliteMap(index) {
            const container = document.getElementById('sat-map-container');
            if (!hasSimilarityMatrix) {
                container.style.display = 'none';
                return;
            }
            container.style.display = '';

            fetch('/api/satellite_map/' + index)
                .then(r => r.json())
                .then(data => {
                    if (data.error) return;

                    // Initialize or reset map
                    if (!satMap) {
                        satMap = L.map('sat-map');
                        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            attribution: '&copy; OpenStreetMap',
                            maxZoom: 19,
                        }).addTo(satMap);
                    }

                    // Clear old markers
                    satMapMarkers.forEach(m => satMap.removeLayer(m));
                    satMapMarkers = [];

                    // Add panorama marker (blue)
                    const panoMarker = L.circleMarker([data.pano_lat, data.pano_lon], {
                        radius: 10, color: '#1976d2', fillColor: '#1976d2', fillOpacity: 0.9, weight: 2,
                    }).addTo(satMap).bindPopup('<b>Panorama</b>');
                    satMapMarkers.push(panoMarker);

                    // Add satellite markers
                    const bounds = L.latLngBounds([[data.pano_lat, data.pano_lon]]);
                    data.markers.forEach(m => {
                        const color = m.is_positive ? '#4caf50' : '#e53935';
                        const radius = m.is_positive ? 8 : 5;
                        const opacity = Math.max(0.3, 1.0 - (m.rank - 1) / 50);
                        const marker = L.circleMarker([m.lat, m.lon], {
                            radius: radius, color: color, fillColor: color,
                            fillOpacity: opacity, weight: m.is_positive ? 3 : 1,
                        }).addTo(satMap);

                        let popup = `<b>Rank #${m.rank}</b> (score: ${m.score.toFixed(4)})<br>`;
                        popup += m.is_positive ? '<span style="color:#4caf50;font-weight:600;">Ground Truth</span><br>' : '';
                        popup += `<img src="/api/image/satellite/${m.sat_idx}" style="width:150px;margin:4px 0;"><br>`;
                        if (m.landmarks.length > 0) {
                            popup += '<b>OSM Landmarks:</b><br>';
                            m.landmarks.forEach(l => { popup += `<small>${l}</small><br>`; });
                        }
                        marker.bindPopup(popup, {maxWidth: 300});
                        satMapMarkers.push(marker);
                        bounds.extend([m.lat, m.lon]);
                    });

                    satMap.fitBounds(bounds, {padding: [30, 30]});
                })
                .catch(err => console.error('Error loading satellite map:', err));
        }

        let histogramData = null;

        function loadSimilarityHistogram(index) {
            const container = document.getElementById('sim-histogram-container');
            if (!hasSimilarityMatrix) {
                container.style.display = 'none';
                return;
            }
            container.style.display = '';

            fetch('/api/similarity_histogram/' + index)
                .then(r => r.json())
                .then(data => {
                    if (data.error) return;
                    histogramData = data;
                    drawHistogram(data);
                })
                .catch(err => console.error('Error loading histogram:', err));
        }

        function drawHistogram(data) {
            const canvas = document.getElementById('sim-histogram-canvas');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = 250;
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = {top: 15, right: 15, bottom: 40, left: 55};
            const plotW = W - pad.left - pad.right;
            const plotH = H - pad.top - pad.bottom;

            ctx.clearRect(0, 0, W, H);

            const numBins = data.counts.length;
            const barW = plotW / numBins;

            // Log scale: map count -> pixel height via log10(count+1)
            const logMax = Math.log10(Math.max(...data.counts) + 1);
            function countToH(c) {
                return logMax > 0 ? (Math.log10(c + 1) / logMax) * plotH : 0;
            }

            // Draw bars
            for (let i = 0; i < numBins; i++) {
                const barH = countToH(data.counts[i]);
                const x = pad.left + i * barW;
                const y = pad.top + plotH - barH;

                // Main bar
                ctx.fillStyle = '#90caf9';
                ctx.fillRect(x, y, barW - 1, barH);

                // Positive overlay
                if (data.positive_counts[i] > 0) {
                    const posH = countToH(data.positive_counts[i]);
                    ctx.fillStyle = '#4caf50';
                    ctx.fillRect(x, pad.top + plotH - posH, barW - 1, posH);
                }
            }

            // Axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top);
            ctx.lineTo(pad.left, pad.top + plotH);
            ctx.lineTo(pad.left + plotW, pad.top + plotH);
            ctx.stroke();

            // X-axis labels
            ctx.fillStyle = '#333';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            const nXLabels = 6;
            for (let i = 0; i <= nXLabels; i++) {
                const frac = i / nXLabels;
                const val = data.bin_edges[0] + frac * (data.bin_edges[data.bin_edges.length - 1] - data.bin_edges[0]);
                const x = pad.left + frac * plotW;
                ctx.fillText(val.toFixed(2), x, pad.top + plotH + 15);
            }
            ctx.textAlign = 'center';
            ctx.fillText('Similarity Score', pad.left + plotW / 2, H - 3);

            // Y-axis labels (log scale ticks: 1, 10, 100, 1000, ...)
            ctx.textAlign = 'right';
            const maxCount = Math.max(...data.counts);
            const maxPow = Math.ceil(Math.log10(maxCount + 1));
            for (let p = 0; p <= maxPow; p++) {
                const val = Math.pow(10, p);
                if (val > maxCount * 1.5) break;
                const frac = Math.log10(val + 1) / logMax;
                const y = pad.top + plotH - frac * plotH;
                ctx.fillText(val.toLocaleString(), pad.left - 5, y + 4);
            }
            ctx.save();
            ctx.translate(12, pad.top + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillText('Count (log)', 0, 0);
            ctx.restore();

            // Legend
            ctx.fillStyle = '#90caf9';
            ctx.fillRect(W - 160, 8, 12, 12);
            ctx.fillStyle = '#333';
            ctx.textAlign = 'left';
            ctx.fillText('All patches', W - 144, 18);
            ctx.fillStyle = '#4caf50';
            ctx.fillRect(W - 160, 24, 12, 12);
            ctx.fillStyle = '#333';
            ctx.fillText('Ground truth', W - 144, 34);

            // Store layout for hover
            canvas._histLayout = {pad, plotW, plotH, barW, numBins, data};
        }

        // Histogram hover handler
        (function() {
            const canvas = document.getElementById('sim-histogram-canvas');
            const tooltip = document.getElementById('sim-histogram-tooltip');

            canvas.addEventListener('mousemove', function(e) {
                if (!canvas._histLayout) return;
                const {pad, plotW, plotH, barW, numBins, data} = canvas._histLayout;
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;

                const binIdx = Math.floor((mx - pad.left) / barW);
                if (binIdx < 0 || binIdx >= numBins || mx < pad.left || mx > pad.left + plotW ||
                    my < pad.top || my > pad.top + plotH) {
                    tooltip.style.display = 'none';
                    return;
                }

                const lo = data.bin_edges[binIdx].toFixed(3);
                const hi = data.bin_edges[binIdx + 1].toFixed(3);
                const count = data.counts[binIdx];
                const posCount = data.positive_counts[binIdx];
                const sample = data.bin_samples[binIdx];

                let html = `<b>[${lo}, ${hi})</b><br>${count} patches`;
                if (posCount > 0) html += ` <span style="color:#4caf50;">(${posCount} positive)</span>`;

                if (sample) {
                    html += `<hr style="margin:4px 0;">`;
                    html += `<div style="display:flex;gap:8px;align-items:flex-start;">`;
                    html += `<img src="/api/image/satellite/${sample.sat_idx}" style="width:80px;height:80px;border-radius:4px;flex-shrink:0;">`;
                    html += `<div>`;
                    html += `<b>Sat ${sample.sat_idx}</b> (${sample.score.toFixed(4)})`;
                    if (sample.is_positive) html += ` <span style="color:#4caf50;">✓</span>`;
                    if (sample.shared_landmarks.length > 0) {
                        html += `<br><b>Matched tags:</b>`;
                        sample.shared_landmarks.forEach(lm => {
                            html += `<br><span style="color:#1976d2;">${lm.tag_key}</span>: ${lm.pano_value}`;
                            if (lm.sat_value && lm.sat_value !== lm.pano_value)
                                html += ` → ${lm.sat_value}`;
                        });
                    } else {
                        html += `<br><i style="color:#999;">No tag matches</i>`;
                    }
                    html += `</div></div>`;
                }

                tooltip.innerHTML = html;
                tooltip.style.display = 'block';
                const tipW = tooltip.offsetWidth;
                const tipH = tooltip.offsetHeight;
                let left = e.pageX + 15;
                let top = e.pageY - 10;
                if (left + tipW > window.scrollX + window.innerWidth)
                    left = e.pageX - tipW - 15;
                if (top + tipH > window.scrollY + window.innerHeight)
                    top = e.pageY - tipH - 10;
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
            });

            canvas.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });
        })();

        function showSatelliteDetail(sat, rank) {
            const panel = document.getElementById('sat-detail-panel');
            const title = document.getElementById('sat-detail-title');
            const tbody = document.querySelector('#sat-detail-table tbody');

            title.textContent = `Rank #${rank} - Satellite ${sat.sat_idx} (score: ${sat.similarity_score.toFixed(4)})${sat.is_positive ? ' ✓ Positive' : ''}`;
            tbody.innerHTML = '';

            if (sat.shared_landmarks && sat.shared_landmarks.length > 0) {
                sat.shared_landmarks.forEach(lm => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>${lm.tag_key}</td><td>${lm.pano_lm_value}</td><td>${lm.sat_lm_value || '-'}</td>`;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="3" style="color:#999;font-style:italic;">No shared tag matches from parquet tables</td></tr>';
            }

            // Show OSM landmarks on this satellite from .feather data
            const osmSection = document.getElementById('sat-osm-landmarks-section');
            osmSection.innerHTML = '';
            if (sat.sat_osm_landmarks && sat.sat_osm_landmarks.length > 0) {
                osmSection.innerHTML = '<h4 style="margin:12px 0 4px 0;color:#555;">OSM Landmarks on Satellite Patch (' + sat.sat_osm_landmarks.length + ')</h4>';
                const osmTable = document.createElement('table');
                osmTable.style.width = '100%';
                osmTable.innerHTML = '<thead><tr><th>Tag</th><th>Value</th></tr></thead>';
                const osmTbody = document.createElement('tbody');
                sat.sat_osm_landmarks.forEach(lm => {
                    Object.entries(lm).forEach(([key, val]) => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `<td>${key}</td><td>${val}</td>`;
                        osmTbody.appendChild(tr);
                    });
                    // Separator between landmarks
                    const sep = document.createElement('tr');
                    sep.innerHTML = '<td colspan="2" style="border-bottom:1px solid #eee;"></td>';
                    osmTbody.appendChild(sep);
                });
                osmTable.appendChild(osmTbody);
                osmSection.appendChild(osmTable);
            }

            // Correspondence details (if model is loaded)
            const corrSection = document.getElementById('sat-correspondence-section');
            if (corrSection) corrSection.innerHTML = '';
            if (hasCorrespondenceModel && corrSection) {
                corrSection.innerHTML = '<p style="color:#999;font-style:italic;">Loading correspondence details...</p>';
                fetch(`/api/correspondence_details/${currentIndex}/${sat.sat_idx}`)
                    .then(r => r.json())
                    .then(corrData => {
                        if (corrData.error && !corrData.matches) {
                            corrSection.innerHTML = `<p style="color:#c00;">${corrData.error}</p>`;
                            return;
                        }
                        let html = '<h4 style="margin:12px 0 4px 0;color:#555;">Correspondence Matching';
                        html += ` <span style="font-size:12px;color:#666;">(score: ${corrData.similarity_score})</span></h4>`;

                        if (corrData.matches && corrData.matches.length > 0) {
                            html += '<table style="width:100%;border-collapse:collapse;font-size:13px;">';
                            html += '<thead><tr><th style="padding:4px 8px;border-bottom:1px solid #eee;">P(match)</th>';
                            html += '<th style="padding:4px 8px;border-bottom:1px solid #eee;">Pano Landmark</th>';
                            html += '<th style="padding:4px 8px;border-bottom:1px solid #eee;">OSM Landmark</th></tr></thead><tbody>';
                            corrData.matches.forEach(m => {
                                const probPct = (m.prob * 100).toFixed(1);
                                const cls = m.prob >= 0.8 ? 'high' : m.prob >= 0.5 ? 'medium' : 'low';
                                html += `<tr>`;
                                html += `<td style="padding:4px 8px;border-bottom:1px solid #eee;"><span class="similarity-badge ${cls}">${probPct}%</span></td>`;
                                html += `<td style="padding:4px 8px;border-bottom:1px solid #eee;font-size:12px;">${m.pano_tags}</td>`;
                                html += `<td style="padding:4px 8px;border-bottom:1px solid #eee;font-size:12px;">${m.osm_tags}</td>`;
                                html += `</tr>`;
                            });
                            html += '</tbody></table>';
                        } else {
                            html += '<p style="color:#999;font-style:italic;">No landmark matches above threshold</p>';
                        }
                        corrSection.innerHTML = html;
                    })
                    .catch(err => {
                        console.error('Error loading correspondence details:', err);
                        corrSection.innerHTML = '<p style="color:#c00;">Failed to load correspondence details</p>';
                    });
            }

            panel.style.display = 'block';
        }

        function searchLandmarks() {
            const input = document.getElementById('landmark-search-input');
            const results = document.getElementById('landmark-search-results');
            const query = input.value.trim();
            if (!query) return;

            fetch('/api/search_landmarks?q=' + encodeURIComponent(query))
                .then(r => r.json())
                .then(data => {
                    if (!data.results || data.results.length === 0) {
                        results.innerHTML = '<p style="color:#999;font-style:italic;">No matches found.</p>';
                        return;
                    }

                    let html = '<table><thead><tr><th>Tag Key</th><th>Pano Value</th><th>Sat Value</th><th>Count</th><th>Panoramas</th></tr></thead><tbody>';
                    data.results.forEach(r => {
                        const panoLinks = r.pano_ids.slice(0, 5).map(id => makeClickablePanoId(id)).join(', ');
                        const more = r.pano_ids.length > 5 ? ` +${r.pano_ids.length - 5} more` : '';
                        html += `<tr><td>${r.tag_key}</td><td>${r.pano_lm_value}</td><td>${r.sat_lm_value || '-'}</td><td>${r.count}</td><td>${panoLinks}${more}</td></tr>`;
                    });
                    html += '</tbody></table>';
                    results.innerHTML = html;
                })
                .catch(err => console.error('Error searching landmarks:', err));
        }

        function loadPanorama(index) {
            // Load panorama data and similarity data in parallel
            Promise.all([
                fetch('/api/panorama/' + index).then(r => r.json()),
                fetch('/api/similarity/' + index).then(r => r.json())
            ]).then(([data, similarityData]) => {
                currentIndex = index;
                totalPanoramas = data.total;

                // Update header
                document.getElementById('pano-name').textContent = data.name;
                document.getElementById('pano-index').textContent = index + 1;
                document.getElementById('pano-total').textContent = totalPanoramas;

                    // Update location links
                    const locationElem = document.getElementById('pano-location');
                    if (data.lat && data.lon) {
                        const googleMapsUrl = `https://www.google.com/maps?q=${data.lat},${data.lon}`;
                        const osmUrl = `https://www.openstreetmap.org/?mlat=${data.lat}&mlon=${data.lon}#map=19/${data.lat}/${data.lon}`;
                        const osmNearbyUrl = `https://www.openstreetmap.org/query?lat=${data.lat}&lon=${data.lon}`;
                        locationElem.innerHTML = ` - 📍 ${data.lat.toFixed(6)}, ${data.lon.toFixed(6)} ` +
                            `[<a href="${googleMapsUrl}" target="_blank" style="color:#007bff;text-decoration:none;">Google</a> | ` +
                            `<a href="${osmUrl}" target="_blank" style="color:#007bff;text-decoration:none;">OSM</a> | ` +
                            `<a href="${osmNearbyUrl}" target="_blank" style="color:#007bff;text-decoration:none;">OSM Nearby</a>]`;
                    } else {
                        locationElem.textContent = '';
                    }

                // Update panorama
                document.getElementById('panorama-img').src = '/api/image/panorama/' + index;

                // Update pinhole images with sentences underneath
                const pinholeGrid = document.getElementById('pinhole-grid');
                pinholeGrid.innerHTML = '';

                // Iterate over yaw data from API
                data.yaw_data.forEach(yawItem => {
                    const yaw = yawItem.yaw;
                    const landmarks = yawItem.landmarks;

                    const div = document.createElement('div');
                    div.className = 'pinhole-item';

                    let html = `
                        <h3>Yaw ${yaw}°</h3>
                        <img src="/api/image/pinhole/${index}/${yaw}" alt="Yaw ${yaw}">
                        <div class="pinhole-sentences">
                    `;

                    // Add landmarks for this yaw
                    if (landmarks && landmarks.length > 0) {
                        html += `<ul>`;
                        landmarks.forEach(lm => {
                            const desc = lm.description;
                            const count = lm.count || 1;
                            const countBadge = count > 1 ? `<span style="background:#28a745;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;font-weight:600;">x${count}</span>` : '';

                            // landmark_id is a tuple [pano_id, idx]
                            const landmarkIdx = lm.landmark_id[1];
                            const allYaws = lm.all_yaws;
                            const color = getLandmarkColor(landmarkIdx);
                            const yawBadges = allYaws.map(y => `<span style="background:#ddd;padding:2px 4px;border-radius:2px;font-size:11px;margin-left:4px;">${y}°</span>`).join('');

                            // v2 tag badges
                            let tagBadges = '';
                            if (lm.primary_tag) {
                                tagBadges += `<span style="background:#1976d2;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;">${lm.primary_tag.key}=${lm.primary_tag.value}</span>`;
                            }
                            if (lm.additional_tags) {
                                lm.additional_tags.forEach(t => {
                                    tagBadges += `<span style="background:#7b1fa2;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;">${t.key}=${t.value}</span>`;
                                });
                            }

                            html += `<li class="landmark-all-mode" style="border-color:${color};background-color:${color}15;">${desc}${tagBadges}${yawBadges}${countBadge}</li>`;
                        });
                        html += `</ul>`;
                    }

                    html += `</div>`;
                    div.innerHTML = html;
                    pinholeGrid.appendChild(div);
                });

                // Update OSM landmarks
                const osmContainer = document.getElementById('osm-landmarks-container');
                let osmHtml = '';

                // Old-style OSM landmarks (from geojson)
                if (data.osm_landmarks && data.osm_landmarks.length > 0) {
                    osmHtml += '<ul style="margin:0;padding-left:20px;">' +
                        data.osm_landmarks.map(lm => {
                            const count = lm.count || 1;
                            const countBadge = count > 1 ? `<span style="background:#28a745;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;font-weight:600;">x${count}</span>` : '';
                            return `<li style="margin-bottom:8px;break-inside:avoid;">${lm.text}${countBadge}</li>`;
                        }).join('') +
                        '</ul>';
                }

                // Tag-weight OSM matches (from parquet tables)
                if (data.osm_tag_matches && data.osm_tag_matches.length > 0) {
                    osmHtml += '<h4 style="margin-top:15px;">OSM Tag Matches (from landmark tables)</h4>';
                    osmHtml += '<table style="width:100%;border-collapse:collapse;font-size:13px;">';
                    osmHtml += '<thead><tr style="background:#e8f0fe;"><th style="padding:6px 10px;text-align:left;">Tag Key</th><th style="padding:6px 10px;text-align:left;">Pano Value</th><th style="padding:6px 10px;text-align:left;">OSM Matches</th><th style="padding:6px 10px;text-align:left;">Sat Values</th></tr></thead><tbody>';
                    data.osm_tag_matches.forEach(m => {
                        const satVals = m.sat_values.length > 0 ? m.sat_values.join(', ') : '-';
                        osmHtml += `<tr style="border-bottom:1px solid #eee;"><td style="padding:4px 10px;">${m.tag_key}</td><td style="padding:4px 10px;">${m.pano_value}</td><td style="padding:4px 10px;">${m.osm_count}</td><td style="padding:4px 10px;">${satVals}</td></tr>`;
                    });
                    osmHtml += '</tbody></table>';
                }

                if (osmHtml) {
                    osmContainer.innerHTML = osmHtml;
                } else {
                    osmContainer.innerHTML = '<p style="color:#999;font-style:italic;">No OSM landmarks found near this panorama.</p>';
                }

                // Update comparison lists with similarity data
                currentSimilarityData = similarityData;
                selectedLandmark = null; // Reset selection when changing panorama
                globalMatches = null; // Reset global matches when changing panorama
                updateComparisonLists(similarityData);

                // Load satellite ranking if available
                loadSatelliteRanking(index);

                // Hide detail panel
                document.getElementById('sat-detail-panel').style.display = 'none';
            }).catch(err => {
                console.error('Error loading panorama:', err);
            });
        }

        function navigate(delta) {
            const order = getNavigationOrder();
            if (order) {
                // Find current position in the ranking
                let pos = order.findIndex(r => r.pano_idx === currentIndex);
                if (pos === -1) pos = 0;
                pos += delta;
                if (pos < 0) pos = order.length - 1;
                if (pos >= order.length) pos = 0;
                loadPanorama(order[pos].pano_idx);
            } else {
                let newIndex = currentIndex + delta;
                if (newIndex < 0) newIndex = totalPanoramas - 1;
                if (newIndex >= totalPanoramas) newIndex = 0;
                loadPanorama(newIndex);
            }
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
        });

        // Load panorama list for ID-to-index mapping, then load first panorama
        fetch('/api/panorama_list')
            .then(r => r.json())
            .then(data => {
                // Build panorama ID to index mapping
                data.panoramas.forEach(p => {
                    panoIdToIndex[p.id] = p.index;
                });
                hasSimilarityMatrix = data.has_similarity_matrix || false;
                hasCorrespondenceModel = data.has_correspondence_model || false;

                // Show/hide UI elements based on similarity matrix
                if (hasSimilarityMatrix) {
                    document.getElementById('nav-mode-container').style.display = '';
                    document.getElementById('landmark-search-section').style.display = '';
                }

                // Now load the first panorama
                loadPanorama(0);
            })
            .catch(err => {
                console.error('Error loading panorama list:', err);
                // Still try to load first panorama even if mapping fails
                loadPanorama(0);
            });
    </script>
</body>
</html>
'''


def parse_vigor_filename(filename):
    """Parse VIGOR format filename to extract panorama ID."""
    # Format: {id},{lat},{lon},.jpg
    parts = filename.replace('.jpg', '').split(',')
    if len(parts) >= 1:
        return parts[0]
    return filename


def load_osm_sentences_for_landmarks(sentence_dir, landmark_custom_ids):
    """Load OSM landmark sentences only for specific custom_ids (efficient)."""
    sentence_path = Path(sentence_dir)
    if not sentence_path.exists():
        print(f"Warning: OSM sentence directory not found: {sentence_dir}")
        return {}

    print(f"  Loading sentences for {len(landmark_custom_ids)} landmarks...")
    needed_ids = set(landmark_custom_ids)
    sentences = {}

    # Read JSONL files and only keep entries we need
    files_processed = 0
    for jsonl_file in sentence_path.glob("*"):
        files_processed += 1
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    custom_id = entry['custom_id']

                    # Only process if we need this landmark
                    if custom_id in needed_ids:
                        if 'response' in entry and 'body' in entry['response']:
                            body = entry['response']['body']
                            if 'choices' in body and len(body['choices']) > 0:
                                content = body['choices'][0]['message']['content']
                                sentences[custom_id] = content
                                needed_ids.remove(custom_id)

                                # Early exit if we found everything
                                if not needed_ids:
                                    print(f"  Found all sentences after {files_processed} files")
                                    return sentences
                except Exception:
                    pass  # Skip malformed entries

    return sentences


def compute_panorama_to_osm_associations(panorama_dir, osm_landmarks, landmarks_geojson_path, zoom_level=20):
    """
    Compute which OSM landmarks are near each panorama using spatial queries.
    Uses caching to avoid recomputing if inputs haven't changed.

    Args:
        panorama_dir: Directory containing panorama images
        osm_landmarks: dict[str, OSMLandmark] - pre-loaded OSM landmarks
        landmarks_geojson_path: Path to OSM landmarks GeoJSON
        zoom_level: Web Mercator zoom level for spatial queries

    Returns:
        dict[str, list[str]]: Maps panorama_id -> list of OSM custom_ids (indices only)
    """
    import pandas as pd
    import shapely
    from common.gps import web_mercator

    # Compute cache key based on input file mtimes
    geojson_path = Path(landmarks_geojson_path)
    pano_path = Path(panorama_dir)

    cache_key_parts = [
        f"geojson:{geojson_path.stat().st_mtime}",
        f"panoramas:{pano_path.stat().st_mtime}",
        f"zoom:{zoom_level}",
        f"osm_count:{len(osm_landmarks)}"
    ]
    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/osm_pano_assoc_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading OSM→panorama associations from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"  Loaded cached associations for {len(cached_data)} panoramas")
            return cached_data
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    print("Computing OSM→panorama associations (no cache found)...")
    start = time.time()
    print("Loading OSM landmarks GeoJSON...")
    landmarks_df = load_landmark_geojson(geojson_path, zoom_level)
    print(f"  Loaded {len(landmarks_df)} landmarks in {time.time()-start:.1f}s")

    print("Loading panorama metadata...")
    panorama_path = Path(panorama_dir)
    pano_files = list(panorama_path.glob('*.jpg'))

    # Build panorama metadata dataframe
    pano_data = []
    for pano_file in pano_files:
        parts = pano_file.stem.split(',')
        if len(parts) >= 3:
            pano_id = parts[0]
            try:
                lat = float(parts[1])
                lon = float(parts[2])
                y, x = web_mercator.latlon_to_pixel_coords(lat, lon, zoom_level)
                pano_data.append({
                    'id': pano_id,
                    'web_mercator_x': x,
                    'web_mercator_y': y
                })
            except ValueError:
                continue

    pano_df = pd.DataFrame(pano_data)
    print(f"  Loaded {len(pano_df)} panoramas")

    print("Computing panorama→OSM associations...")
    start = time.time()

    # Use spatial query (same as vigor_dataset.py)
    strtree = shapely.STRtree(landmarks_df.geometry_px)
    MAX_DIST_PX = 640  # Same as in vigor_dataset.py

    queries = []
    for _, pano in pano_df.iterrows():
        center_y, center_x = pano["web_mercator_y"], pano["web_mercator_x"]
        queries.append(shapely.box(
            xmin=center_x - MAX_DIST_PX//2,
            xmax=center_x + MAX_DIST_PX//2,
            ymin=center_y - MAX_DIST_PX//2,
            ymax=center_y + MAX_DIST_PX//2))

    # Batch query - much faster!
    results = strtree.query(queries, predicate='intersects')

    print(f"  Computed associations in {time.time()-start:.1f}s")

    # Build associations: pano_id -> list of custom_ids
    print("Building association dictionary...")
    start = time.time()
    pano_to_osm = {}
    osm_custom_ids_set = set(osm_landmarks.keys())

    for pano_idx, landmark_idx in results.T:
        pano_id = pano_df.iloc[pano_idx]['id']
        landmark = landmarks_df.iloc[landmark_idx]

        # Compute custom_id
        props = landmark['pruned_props']
        custom_id = custom_id_from_props(props)

        # Only include if we have this OSM landmark loaded
        if custom_id in osm_custom_ids_set:
            if pano_id not in pano_to_osm:
                pano_to_osm[pano_id] = []
            if custom_id not in pano_to_osm[pano_id]:  # Avoid duplicates
                pano_to_osm[pano_id].append(custom_id)

    print(f"  Built associations in {time.time()-start:.1f}s")
    print(f"  Found associations for {len(pano_to_osm)} panoramas")

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(pano_to_osm, f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return pano_to_osm


def load_osm_landmarks(geojson_path, sentences_dir, embeddings_dir=None):
    """
    Load OSM landmarks from GeoJSON, sentences, and optionally embeddings.
    Uses caching to avoid recomputing if inputs haven't changed.

    Args:
        geojson_path: Path to OSM landmarks GeoJSON file
        sentences_dir: Directory containing OSM landmark sentence files
        embeddings_dir: Optional directory containing OSM landmark embeddings

    Returns:
        Tuple of (OSM_LANDMARKS dict, OSM_EMBEDDINGS tensor, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE)
        If embeddings_dir is None, returns (dict, None, {}, [])
    """
    # Compute cache key based on input file mtimes
    geojson_path_obj = Path(geojson_path)
    sentences_path_obj = Path(sentences_dir)

    cache_key_parts = [
        f"geojson:{geojson_path_obj.stat().st_mtime}",
        f"sentences:{sentences_path_obj.stat().st_mtime}"
    ]
    if embeddings_dir:
        embeddings_path_obj = Path(embeddings_dir)
        if embeddings_path_obj.exists():
            cache_key_parts.append(f"embeddings:{embeddings_path_obj.stat().st_mtime}")

    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/osm_landmarks_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading OSM landmarks from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            osm_landmarks, osm_embeddings, osm_embedding_index, osm_index_reverse = cached_data
            print(f"  Loaded {len(osm_landmarks)} OSM landmarks from cache")
            if osm_embeddings is not None:
                print(f"  Loaded {osm_embeddings.shape[0]} OSM embeddings from cache")
            return cached_data
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    print("Loading OSM landmarks (no cache found)...")
    start = time.time()

    # Load GeoJSON to get landmark properties
    landmarks_df = load_landmark_geojson(geojson_path, zoom_level=20)
    print(f"  Loaded {len(landmarks_df)} landmarks from GeoJSON in {time.time()-start:.1f}s")

    # Build custom_id for each landmark
    print("  Computing custom IDs...")
    landmark_custom_ids = []
    for _, landmark in landmarks_df.iterrows():
        props = landmark['pruned_props']
        custom_id = custom_id_from_props(props)
        landmark_custom_ids.append((custom_id, props, landmark))

    # Load sentences for these landmarks
    print(f"  Loading sentences for {len(landmark_custom_ids)} landmarks...")
    custom_ids_needed = {cid for cid, _, _ in landmark_custom_ids}
    osm_sentences = load_osm_sentences_for_landmarks(sentences_dir, custom_ids_needed)
    print(f"  Loaded {len(osm_sentences)} sentences")

    # Build OSMLandmark objects
    print("  Building OSMLandmark objects...")
    osm_landmarks_dict = {}
    for custom_id, props, landmark_row in landmark_custom_ids:
        if custom_id in osm_sentences:
            # Extract lat/lon from landmark geometry
            geometry = landmark_row.get('geometry')
            if geometry:
                centroid = geometry.centroid
                lat = centroid.y
                lon = centroid.x
            else:
                lat, lon = 0.0, 0.0

            osm_landmarks_dict[custom_id] = OSMLandmark(
                custom_id=custom_id,
                description=osm_sentences[custom_id],
                properties=dict(props),
                lat=lat,
                lon=lon
            )

    print(f"  Created {len(osm_landmarks_dict)} OSMLandmark objects")
    print(f"Loaded OSM landmarks in {time.time()-start:.1f}s")

    # Load embeddings if embeddings_dir is provided
    if embeddings_dir:
        print("\n  Loading OSM embeddings...")
        emb_start = time.time()
        embeddings_path = Path(embeddings_dir)

        if not embeddings_path.exists():
            print(f"  Warning: Embeddings directory not found: {embeddings_dir}")
            osm_embeddings = None
            osm_embedding_index = {}
            osm_index_reverse = []
        else:
            pkl_file = embeddings_path / "embeddings.pkl"
            osm_embeddings = None  # Initialize before trying to load

            # Try to load from .pkl file first (fast path, same as semantic_landmark_extractor)
            if pkl_file.exists():
                print(f"    Loading from {pkl_file}...")
                try:
                    with open(pkl_file, 'rb') as f:
                        pkl_data = pickle.load(f)

                    # Detect format: v2 is a dict with "description_embeddings", v1 is a tuple
                    if isinstance(pkl_data, dict) and "description_embeddings" in pkl_data:
                        osm_embeddings = pkl_data["description_embeddings"]
                        osm_embedding_index = pkl_data["description_id_to_idx"]
                    else:
                        osm_embeddings, osm_embedding_index = pkl_data

                    # Build reverse index
                    osm_index_reverse = [None] * len(osm_embedding_index)
                    for custom_id, idx in osm_embedding_index.items():
                        osm_index_reverse[idx] = custom_id

                    print(f"    Loaded {osm_embeddings.shape[0]} OSM embeddings from .pkl in {time.time()-emb_start:.1f}s")
                except Exception as e:
                    print(f"    Failed to load .pkl file: {e}, falling back to JSONL...")
                    osm_embeddings = None  # Reset on failure

            # Fall back to JSONL loading if .pkl doesn't exist or failed
            if osm_embeddings is None:
                # Collect embeddings for our loaded landmarks
                embeddings_data = []  # List of (custom_id, embedding_vector)
                custom_ids_set = set(osm_landmarks_dict.keys())

                print(f"    Reading embedding files for {len(custom_ids_set)} landmarks...")
                files_processed = 0
                for jsonl_file in embeddings_path.glob('*'):
                    if not jsonl_file.is_file() or jsonl_file.suffix == '.pkl':
                        continue
                    files_processed += 1
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                custom_id = entry['custom_id']

                                # Only load if we have this landmark
                                if custom_id not in custom_ids_set:
                                    continue

                                # Extract embedding from response
                                if 'response' not in entry or 'body' not in entry['response']:
                                    continue

                                body = entry['response']['body']
                                if 'data' not in body or len(body['data']) == 0:
                                    continue

                                embedding = body['data'][0]['embedding']
                                embeddings_data.append((custom_id, embedding))

                            except Exception as e:
                                pass  # Skip malformed entries

                print(f"    Processed {files_processed} files")
                print(f"    Found {len(embeddings_data)} embeddings")

                if len(embeddings_data) > 0:
                    # Build tensor and indices
                    print("    Building embedding tensor...")
                    embeddings_data.sort(key=lambda x: x[0])  # Sort by custom_id

                    osm_embedding_index = {}
                    osm_index_reverse = []
                    embedding_vectors = []

                    for row_idx, (custom_id, embedding) in enumerate(embeddings_data):
                        osm_embedding_index[custom_id] = row_idx
                        osm_index_reverse.append(custom_id)
                        embedding_vectors.append(embedding)

                    # Convert to tensor
                    osm_embeddings = torch.tensor(embedding_vectors, dtype=torch.float32)
                    print(f"    Created tensor with shape {osm_embeddings.shape}")
                else:
                    print("    No embeddings found!")
                    osm_embeddings = None
                    osm_embedding_index = {}
                    osm_index_reverse = []

                print(f"  Loaded OSM embeddings in {time.time()-emb_start:.1f}s")
    else:
        osm_embeddings = None
        osm_embedding_index = {}
        osm_index_reverse = []

    # Save to cache
    result = (osm_landmarks_dict, osm_embeddings, osm_embedding_index, osm_index_reverse)
    print(f"Saving OSM landmarks to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return result


def load_pano_embeddings(embeddings_dir, pano_sentences):
    """
    Load panorama landmark embeddings from .pkl file if available, otherwise from JSONL files.
    Uses caching to avoid recomputing if inputs haven't changed.

    Args:
        embeddings_dir: Directory containing panorama embedding files (.pkl or JSONL)
        pano_sentences: dict[str, list[PanoramaLandmark]] to ensure consistency

    Returns:
        Tuple of (PANO_EMBEDDINGS tensor, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE)
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        print(f"Warning: Embeddings directory not found: {embeddings_dir}")
        return None, {}, []

    pkl_file = embeddings_path / "embeddings.pkl"

    # Try to load from .pkl file first (fast path, same as semantic_landmark_extractor)
    if pkl_file.exists():
        print(f"Loading panorama embeddings from {pkl_file}...")
        start = time.time()
        try:
            with open(pkl_file, 'rb') as f:
                pkl_data = pickle.load(f)

            # Detect format: v2 is a dict with "description_embeddings", v1 is a tuple
            if isinstance(pkl_data, dict) and "description_embeddings" in pkl_data:
                embeddings_tensor = pkl_data["description_embeddings"]
                embedding_index_raw = pkl_data["description_id_to_idx"]
            else:
                embeddings_tensor, embedding_index_raw = pkl_data

            # Convert string landmark IDs to tuples
            # The pkl file stores keys as strings like "pano_id,lat,lon,__landmark_N"
            # But we need tuples like (pano_id, landmark_idx)
            embedding_index = {}
            index_reverse = [None] * len(embedding_index_raw)

            for string_id, idx in embedding_index_raw.items():
                # Parse custom_id: "pano_id,lat,lon,__landmark_N"
                parts = string_id.split(',')
                if len(parts) >= 4 and parts[3].startswith('__landmark_'):
                    pano_id = parts[0]
                    try:
                        landmark_idx = int(parts[3].replace('__landmark_', ''))
                        landmark_id = (pano_id, landmark_idx)
                        embedding_index[landmark_id] = idx
                        index_reverse[idx] = landmark_id
                    except ValueError:
                        continue

            print(f"  Loaded {embeddings_tensor.shape[0]} panorama embeddings in {time.time()-start:.1f}s")
            print(f"  Converted {len(embedding_index)} landmark IDs to tuple format")
            return embeddings_tensor, embedding_index, index_reverse
        except Exception as e:
            print(f"  Failed to load .pkl file: {e}, falling back to JSONL...")

    # Fall back to JSONL loading with caching
    # Compute cache key based on embeddings dir mtime and pano_sentences structure
    cache_key_parts = [
        f"embeddings:{embeddings_path.stat().st_mtime}",
        f"pano_count:{len(pano_sentences)}",
        f"landmark_count:{sum(len(lms) for lms in pano_sentences.values())}"
    ]
    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/pano_embeddings_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading panorama embeddings from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            pano_embeddings, pano_embedding_index, pano_index_reverse = cached_data
            if pano_embeddings is not None:
                print(f"  Loaded {pano_embeddings.shape[0]} panorama embeddings from cache")
            return cached_data
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    print("Loading panorama embeddings from JSONL (no cache found)...")
    start = time.time()

    # Collect all embeddings
    embeddings_data = []  # List of (landmark_id, embedding_vector)

    print("  Reading embedding files...")
    files_processed = 0
    for jsonl_file in embeddings_path.rglob('*'):
        if not jsonl_file.is_file() or jsonl_file.suffix == '.pkl':
            continue

        files_processed += 1
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # v1 uses 'custom_id', v2 uses 'key'
                    custom_id = entry.get('custom_id') or entry.get('key')
                    if not custom_id:
                        continue

                    # Parse custom_id: "pano_id,lat,lon,__landmark_N"
                    parts = custom_id.split(',')
                    if len(parts) < 4:
                        continue

                    # Check if the last part is the landmark marker
                    landmark_part = parts[3]
                    if not landmark_part.startswith('__landmark_'):
                        continue

                    pano_id = parts[0]
                    try:
                        landmark_idx = int(landmark_part.replace('__landmark_', ''))
                    except ValueError:
                        continue

                    # Extract embedding from response
                    if 'response' not in entry or 'body' not in entry['response']:
                        continue

                    body = entry['response']['body']
                    if 'data' not in body or len(body['data']) == 0:
                        continue

                    embedding = body['data'][0]['embedding']

                    # Verify this landmark exists in our loaded data
                    landmark_id = (pano_id, landmark_idx)
                    if pano_id in pano_sentences:
                        # Check if this landmark_idx exists
                        if any(lm.landmark_id == landmark_id for lm in pano_sentences[pano_id]):
                            embeddings_data.append((landmark_id, embedding))

                except Exception as e:
                    pass  # Skip malformed entries

    print(f"  Processed {files_processed} files")
    print(f"  Found {len(embeddings_data)} embeddings")

    if len(embeddings_data) == 0:
        print("  No embeddings found!")
        return None, {}, []

    # Build tensor and indices
    print("  Building embedding tensor...")
    embeddings_data.sort(key=lambda x: x[0])  # Sort by landmark_id for consistency

    embedding_index = {}
    index_reverse = []
    embedding_vectors = []

    for row_idx, (landmark_id, embedding) in enumerate(embeddings_data):
        embedding_index[landmark_id] = row_idx
        index_reverse.append(landmark_id)
        embedding_vectors.append(embedding)

    # Convert to tensor
    embeddings_tensor = torch.tensor(embedding_vectors, dtype=torch.float32)

    print(f"  Created tensor with shape {embeddings_tensor.shape}")
    print(f"Loaded panorama embeddings in {time.time()-start:.1f}s")

    # Save to cache
    result = (embeddings_tensor, embedding_index, index_reverse)
    print(f"Saving panorama embeddings to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return result


def load_sentence_data(sentence_dirs):
    """
    Load and parse JSONL sentence files from multiple directories.
    Uses caching to avoid recomputing if inputs haven't changed.

    Returns:
        dict[str, list[PanoramaLandmark]]: Maps panorama_id to list of landmarks
    """
    # Compute cache key based on all sentence directory mtimes
    cache_key_parts = []
    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if sentence_path.exists():
            cache_key_parts.append(f"{sentence_path.name}:{sentence_path.stat().st_mtime}")

    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/pano_sentences_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading panorama sentences from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"  Loaded landmarks for {len(cached_data)} panoramas from cache")
            return cached_data
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    print("Loading panorama sentences (no cache found)...")

    # Collect all landmarks by panorama_id
    pano_landmarks: dict[str, list[PanoramaLandmark]] = {}

    print(f"  Processing {len(sentence_dirs)} sentence directories...")
    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if not sentence_path.exists():
            print(f"Warning: Sentence directory not found: {sentence_dir}")
            continue

        print(f"    Loading from {sentence_path.name}...")
        dir_start = time.time()
        file_count = 0
        entry_count = 0

        # Load all files in this directory (JSONL format)
        # Use rglob to handle v2's deep nesting (sentences/results/panorama_request_*/predictions.jsonl)
        for jsonl_file in sentence_path.rglob('*'):
            if not jsonl_file.is_file() or jsonl_file.suffix == '.pkl':
                continue

            file_count += 1
            with open(jsonl_file, 'r') as f:
                for line in f:
                    entry_count += 1
                    try:
                        entry = json.loads(line)

                        # Determine format: v1 uses 'custom_id', v2 uses 'key'
                        custom_id = entry.get('custom_id') or entry.get('key')
                        if not custom_id:
                            continue

                        # Only process "all mode" entries (format: "pano_id,lat,lon,")
                        # Skip "individual mode" entries (format: "pano_id_yaw_N")
                        if ',' not in custom_id:
                            continue

                        parts = custom_id.split(',')
                        if len(parts) < 3:
                            continue

                        pano_id = parts[0]
                        try:
                            pano_lat = float(parts[1])
                            pano_lon = float(parts[2])
                        except (ValueError, IndexError):
                            print(f"Warning: Could not parse lat/lon from {custom_id}")
                            continue

                        # Try v1 format first: response.body.choices[0].message.content
                        content = None
                        if 'response' in entry and 'body' in entry['response']:
                            body = entry['response']['body']
                            if 'choices' in body and len(body['choices']) > 0:
                                content = body['choices'][0]['message']['content']

                        # Try v2 (Gemini) format: response.candidates[0].content.parts[0].text
                        if content is None and 'response' in entry:
                            resp = entry['response']
                            candidates = resp.get('candidates', [])
                            if candidates:
                                parts_list = candidates[0].get('content', {}).get('parts', [])
                                if parts_list:
                                    content = parts_list[0].get('text')

                        if content is None:
                            continue

                        try:
                            # Strip markdown code fences if present
                            text = content.strip()
                            if text.startswith('```'):
                                lines = text.split('\n')
                                # Remove first line (```json) and last line (```)
                                lines = [l for l in lines if not l.strip().startswith('```')]
                                text = '\n'.join(lines)

                            landmarks_data = json.loads(text)
                            landmarks = landmarks_data.get('landmarks', [])

                            for lm_idx, lm in enumerate(landmarks):
                                description = lm['description']

                                # v1: yaw_angles is a list of ints
                                yaw_angles = lm.get('yaw_angles', [])

                                # v2: yaws are in bounding_boxes[].yaw_angle (string)
                                if not yaw_angles and 'bounding_boxes' in lm:
                                    yaw_set = set()
                                    for bb in lm['bounding_boxes']:
                                        try:
                                            yaw_set.add(int(float(bb.get('yaw_angle', 0))))
                                        except (ValueError, TypeError):
                                            pass
                                    yaw_angles = sorted(yaw_set)

                                # v2 tags
                                primary_tag = lm.get('primary_tag')
                                additional_tags = lm.get('additional_tags', [])

                                landmark = PanoramaLandmark(
                                    description=description,
                                    landmark_id=(pano_id, lm_idx),
                                    yaws=yaw_angles,
                                    panorama_lat=pano_lat,
                                    panorama_lon=pano_lon,
                                    primary_tag=primary_tag,
                                    additional_tags=additional_tags or [],
                                )

                                if pano_id not in pano_landmarks:
                                    pano_landmarks[pano_id] = []
                                pano_landmarks[pano_id].append(landmark)

                        except json.JSONDecodeError:
                            print(f"Warning: Failed to parse JSON content for {custom_id}")

                    except Exception as e:
                        print(f"Warning: Error parsing line in {jsonl_file}: {e}")

        dir_time = time.time() - dir_start
        print(f"    Processed {file_count} files with {entry_count} entries in {dir_time:.1f}s")

    print(f"  Loaded landmarks for {len(pano_landmarks)} panoramas")

    # Save to cache
    print(f"Saving panorama sentences to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(pano_landmarks, f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return pano_landmarks


def compare_pano_to_nearby_osm(pano_id):
    """
    Compare panorama landmarks to nearby OSM landmarks using embeddings.
    Returns bidirectional similarity data for interactive visualization.

    Args:
        pano_id: Panorama ID to compare

    Returns:
        dict with keys:
            - 'pano_landmarks': list of pano landmark info
            - 'osm_landmarks': list of nearby OSM landmark info
            - 'pano_to_osm': dict mapping pano_key -> list of {osm_key, similarity}
            - 'osm_to_pano': dict mapping osm_key -> list of {pano_key, similarity}
        Returns None if embeddings are not loaded or panorama not found
    """
    global PANO_SENTENCES, PANO_EMBEDDINGS, PANO_EMBEDDING_INDEX
    global PANO_TO_OSM, OSM_LANDMARKS, OSM_EMBEDDINGS, OSM_EMBEDDING_INDEX

    # Check if embeddings are loaded
    if PANO_EMBEDDINGS is None or OSM_EMBEDDINGS is None:
        return None

    # Check if panorama exists
    if pano_id not in PANO_SENTENCES:
        return None

    # Get nearby OSM landmarks
    if pano_id not in PANO_TO_OSM:
        return {
            'pano_landmarks': [],
            'osm_landmarks': [],
            'pano_to_osm': {},
            'osm_to_pano': {}
        }

    nearby_osm_ids = PANO_TO_OSM[pano_id]

    # Filter to only OSM landmarks with embeddings
    nearby_osm_ids = [cid for cid in nearby_osm_ids if cid in OSM_EMBEDDING_INDEX]

    if len(nearby_osm_ids) == 0:
        return {
            'pano_landmarks': [],
            'osm_landmarks': [],
            'pano_to_osm': {},
            'osm_to_pano': {}
        }

    # Get panorama landmarks with embeddings
    pano_landmarks = [lm for lm in PANO_SENTENCES[pano_id]
                     if lm.landmark_id in PANO_EMBEDDING_INDEX]

    if len(pano_landmarks) == 0:
        return {
            'pano_landmarks': [],
            'osm_landmarks': [],
            'pano_to_osm': {},
            'osm_to_pano': {}
        }

    # Build embedding matrices
    pano_indices = [PANO_EMBEDDING_INDEX[lm.landmark_id] for lm in pano_landmarks]
    pano_embeddings = PANO_EMBEDDINGS[pano_indices]  # shape: (num_pano, embedding_dim)

    osm_indices = [OSM_EMBEDDING_INDEX[cid] for cid in nearby_osm_ids]
    osm_embeddings = OSM_EMBEDDINGS[osm_indices]  # shape: (num_osm, embedding_dim)

    # Normalize embeddings
    pano_embs_norm = pano_embeddings / torch.norm(pano_embeddings, dim=1, keepdim=True)
    osm_embs_norm = osm_embeddings / torch.norm(osm_embeddings, dim=1, keepdim=True)

    # Compute full similarity matrix: pano x osm
    similarity_matrix = torch.matmul(pano_embs_norm, osm_embs_norm.T)  # shape: (num_pano, num_osm)

    # Build result dictionaries
    pano_to_osm = {}
    osm_to_pano = {}

    # For each pano landmark, get all OSM similarities
    for i, pano_landmark in enumerate(pano_landmarks):
        pano_key = f"{pano_landmark.landmark_id[0]}:{pano_landmark.landmark_id[1]}"
        similarities = similarity_matrix[i].tolist()

        pano_to_osm[pano_key] = [
            {'osm_key': osm_id, 'similarity': float(sim)}
            for osm_id, sim in zip(nearby_osm_ids, similarities)
        ]
        # Sort by similarity descending
        pano_to_osm[pano_key].sort(key=lambda x: x['similarity'], reverse=True)

    # For each OSM landmark, get all pano similarities
    for j, osm_id in enumerate(nearby_osm_ids):
        similarities = similarity_matrix[:, j].tolist()

        osm_to_pano[osm_id] = [
            {'pano_key': f"{lm.landmark_id[0]}:{lm.landmark_id[1]}", 'similarity': float(sim)}
            for lm, sim in zip(pano_landmarks, similarities)
        ]
        # Sort by similarity descending
        osm_to_pano[osm_id].sort(key=lambda x: x['similarity'], reverse=True)

    # Build landmark info lists
    pano_info = [
        {
            'key': f"{lm.landmark_id[0]}:{lm.landmark_id[1]}",
            'description': lm.description,
            'yaws': lm.yaws
        }
        for lm in pano_landmarks
    ]

    osm_info = [
        {
            'key': osm_id,
            'description': OSM_LANDMARKS[osm_id].description if osm_id in OSM_LANDMARKS else "Unknown"
        }
        for osm_id in nearby_osm_ids
    ]

    return {
        'pano_landmarks': pano_info,
        'osm_landmarks': osm_info,
        'pano_to_osm': pano_to_osm,
        'osm_to_pano': osm_to_pano
    }


def get_global_similarity_matches(landmark_type, landmark_key, top_k=50):
    """
    Get global similarity matches for a given landmark across all panoramas/OSM landmarks.

    Args:
        landmark_type: 'pano' or 'osm'
        landmark_key: For pano: "pano_id:idx", for osm: custom_id
        top_k: Number of top matches to return

    Returns:
        List of matches with similarity scores and metadata
    """
    global PANO_SENTENCES, PANO_EMBEDDINGS, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE
    global OSM_LANDMARKS, OSM_EMBEDDINGS, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE
    global PANO_TO_OSM

    if PANO_EMBEDDINGS is None or OSM_EMBEDDINGS is None:
        return []

    if landmark_type == 'osm':
        # OSM landmark selected: find top K panorama landmarks globally
        if landmark_key not in OSM_EMBEDDING_INDEX:
            return []

        osm_idx = OSM_EMBEDDING_INDEX[landmark_key]
        osm_emb = OSM_EMBEDDINGS[osm_idx:osm_idx+1]  # shape: (1, dim)

        # Normalize embeddings
        osm_emb_norm = osm_emb / torch.norm(osm_emb, dim=1, keepdim=True)
        pano_embs_norm = PANO_EMBEDDINGS / torch.norm(PANO_EMBEDDINGS, dim=1, keepdim=True)

        # Compute similarities to all pano landmarks
        similarities = torch.matmul(osm_emb_norm, pano_embs_norm.T).squeeze().tolist()

        # Build list of matches with metadata
        matches = []
        for idx, sim in enumerate(similarities):
            pano_id, lm_idx = PANO_INDEX_REVERSE[idx]

            # Find the landmark object
            if pano_id in PANO_SENTENCES:
                for lm in PANO_SENTENCES[pano_id]:
                    if lm.landmark_id == (pano_id, lm_idx):
                        matches.append({
                            'key': f"{pano_id}:{lm_idx}",
                            'description': lm.description,
                            'similarity': float(sim),
                            'pano_id': pano_id,
                            'pano_lat': lm.panorama_lat,
                            'pano_lon': lm.panorama_lon
                        })
                        break

        # Sort by similarity and return top K
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]

    else:  # landmark_type == 'pano'
        # Pano landmark selected: find top K OSM landmarks globally
        # Parse the key (format: "pano_id:landmark_idx")
        # Use rsplit to split from right in case pano_id contains colons
        parts = landmark_key.rsplit(':', 1)
        if len(parts) != 2:
            return []
        pano_id, lm_idx_str = parts
        try:
            lm_idx = int(lm_idx_str)
        except ValueError:
            return []

        landmark_id = (pano_id, lm_idx)
        if landmark_id not in PANO_EMBEDDING_INDEX:
            return []

        pano_idx = PANO_EMBEDDING_INDEX[landmark_id]
        pano_emb = PANO_EMBEDDINGS[pano_idx:pano_idx+1]  # shape: (1, dim)

        # Normalize embeddings
        pano_emb_norm = pano_emb / torch.norm(pano_emb, dim=1, keepdim=True)
        osm_embs_norm = OSM_EMBEDDINGS / torch.norm(OSM_EMBEDDINGS, dim=1, keepdim=True)

        # Compute similarities to all OSM landmarks
        similarities = torch.matmul(pano_emb_norm, osm_embs_norm.T).squeeze().tolist()

        # Build reverse mapping: osm_id -> list of pano_ids
        osm_to_panos = {}
        for pano_id, osm_ids in PANO_TO_OSM.items():
            for osm_id in osm_ids:
                if osm_id not in osm_to_panos:
                    osm_to_panos[osm_id] = []
                osm_to_panos[osm_id].append(pano_id)

        # Build list of matches with metadata
        matches = []
        for idx, sim in enumerate(similarities):
            osm_id = OSM_INDEX_REVERSE[idx]

            if osm_id in OSM_LANDMARKS:
                osm_lm = OSM_LANDMARKS[osm_id]
                pano_ids = osm_to_panos.get(osm_id, [])

                matches.append({
                    'key': osm_id,
                    'description': osm_lm.description,
                    'similarity': float(sim),
                    'osm_lat': osm_lm.lat,
                    'osm_lon': osm_lm.lon,
                    'nearby_panos': pano_ids[:5]  # Limit to first 5 panos
                })

        # Sort by similarity and return top K
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]


def find_common_panoramas(panorama_dir, pinhole_dir, pano_sentences, dataset_path=None):
    """
    Find panoramas that exist in all required locations.

    Args:
        panorama_dir: Directory containing panorama images
        pinhole_dir: Directory containing pinhole image subdirectories
        pano_sentences: dict[str, list[PanoramaLandmark]] from load_sentence_data()
        dataset_path: Optional path to VIGOR city dir; if pano_id_mapping.csv exists,
                      uses its order for sequential navigation.

    Returns:
        List of panorama data dicts
    """
    panorama_path = Path(panorama_dir)
    pinhole_path = Path(pinhole_dir)

    # Get panorama IDs from panorama directory
    t1 = time.time()
    pano_files = {parse_vigor_filename(f.name): f for f in panorama_path.glob('*.jpg')}
    print(f"  Scanned {len(pano_files)} panorama files in {time.time()-t1:.1f}s")

    # Get panorama IDs from pinhole directory
    t2 = time.time()
    pinhole_dirs = {d.name.split(',')[0]: d for d in pinhole_path.iterdir() if d.is_dir()}
    print(f"  Scanned {len(pinhole_dirs)} pinhole directories in {time.time()-t2:.1f}s")

    # Find intersection — require panorama image + pinhole dir, but not sentence data
    common_ids = set(pano_files.keys()) & set(pinhole_dirs.keys())
    with_landmarks = common_ids & set(pano_sentences.keys())
    print(f"  Found {len(common_ids)} panoramas present in all locations "
          f"({len(with_landmarks)} with landmarks, {len(common_ids) - len(with_landmarks)} without)")

    # Determine ordering: use pano_id_mapping.csv if available (sequential capture order)
    ordered_ids = None
    if dataset_path:
        mapping_csv = Path(dataset_path) / "pano_id_mapping.csv"
        if mapping_csv.exists():
            import csv
            with open(mapping_csv) as f:
                reader = csv.DictReader(f)
                ordered_ids = [row['pano_id'] for row in reader if row['pano_id'] in common_ids]
            # Add any common_ids not in the mapping (shouldn't happen, but be safe)
            remaining = common_ids - set(ordered_ids)
            ordered_ids.extend(sorted(remaining))
            print(f"  Using pano_id_mapping.csv for sequential order ({len(ordered_ids)} panos)")

    if ordered_ids is None:
        ordered_ids = sorted(common_ids)

    # Build panorama data
    t3 = time.time()
    panorama_data = []
    for pano_id in ordered_ids:
        # Get yaw angles available in pinhole dir
        pinhole_subdir = pinhole_dirs[pano_id]
        yaw_angles = []
        for yaw in [0, 90, 180, 270]:
            jpg_path = pinhole_subdir / f'yaw_{yaw:03d}.jpg'
            png_path = pinhole_subdir / f'yaw_{yaw:03d}.png'
            if jpg_path.exists() or png_path.exists():
                yaw_angles.append(yaw)

        if len(yaw_angles) == 4:  # Only include if all 4 views exist
            panorama_data.append({
                'id': pano_id,
                'panorama_file': pano_files[pano_id],
                'pinhole_dir': pinhole_subdir,
                'yaw_angles': yaw_angles
            })
    print(f"  Built panorama data in {time.time()-t3:.1f}s")

    return panorama_data


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


def collapse_duplicate_sentences(sentences):
    """Collapse duplicate sentences and add count multipliers."""
    from collections import Counter

    # For sentences that are dicts (with mode info), use description as key
    # For plain strings, use the string directly
    sentence_counts = Counter()
    sentence_objects = {}  # Maps description -> first occurrence object

    for sent in sentences:
        if isinstance(sent, dict):
            desc = sent['description']
            sentence_counts[desc] += 1
            if desc not in sentence_objects:
                sentence_objects[desc] = sent
        else:
            sentence_counts[sent] += 1
            if sent not in sentence_objects:
                sentence_objects[sent] = sent

    # Build result with counts
    result = []
    for key, count in sentence_counts.items():
        obj = sentence_objects[key]
        if isinstance(obj, dict):
            result.append({**obj, 'count': count})
        else:
            result.append({'description': obj, 'count': count, 'mode': 'plain'})

    return result


def collapse_string_list(strings):
    """Collapse duplicate strings in a list and add count multipliers."""
    from collections import Counter
    counts = Counter(strings)
    return [{'text': text, 'count': count} for text, count in counts.items()]


@app.route('/api/panorama_list')
def get_panorama_list():
    """Return a list of all panorama IDs with their indices."""
    global PANORAMA_DATA, SIMILARITY_MATRIX
    return jsonify({
        'panoramas': [{'id': pano['id'], 'index': i} for i, pano in enumerate(PANORAMA_DATA)],
        'has_similarity_matrix': SIMILARITY_MATRIX is not None,
        'has_correspondence_model': CORRESPONDENCE_MODEL is not None,
    })


@app.route('/api/panorama/<int:index>')
def get_panorama_data(index):
    global PANORAMA_DATA, PANO_SENTENCES, PANO_TO_OSM, OSM_LANDMARKS

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano = PANORAMA_DATA[index]
    pano_id = pano['id']

    # Get panorama landmarks and group by yaw
    landmarks_by_yaw = {}
    if pano_id in PANO_SENTENCES:
        for landmark in PANO_SENTENCES[pano_id]:
            for yaw in landmark.yaws:
                if yaw not in landmarks_by_yaw:
                    landmarks_by_yaw[yaw] = []

                # Convert landmark to dict for JSON
                lm_dict = {
                    'description': landmark.description,
                    'landmark_id': landmark.landmark_id,  # (pano_id, idx)
                    'all_yaws': landmark.yaws,
                }
                if landmark.primary_tag:
                    lm_dict['primary_tag'] = landmark.primary_tag
                if landmark.additional_tags:
                    lm_dict['additional_tags'] = landmark.additional_tags
                landmarks_by_yaw[yaw].append(lm_dict)

    # Collapse duplicates per yaw
    for yaw in landmarks_by_yaw:
        landmarks_by_yaw[yaw] = collapse_duplicate_sentences(landmarks_by_yaw[yaw])

    # Format for frontend (list of yaw data)
    yaw_data = []
    for yaw in pano['yaw_angles']:
        yaw_data.append({
            'yaw': yaw,
            'landmarks': landmarks_by_yaw.get(yaw, [])
        })

    # Resolve OSM landmarks — prefer .feather data from VigorDataset when available
    osm_landmarks_data = []
    if VIGOR_DATASET is not None and pano_id in PANO_ID_TO_VIGOR_IDX:
        vigor_idx = PANO_ID_TO_VIGOR_IDX[pano_id]
        pano_row = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]
        landmark_idxs = pano_row.get('landmark_idxs', [])
        raw_strings = []
        if landmark_idxs is not None and len(landmark_idxs) > 0:
            for lm_idx in landmark_idxs:
                lm_row = VIGOR_DATASET._landmark_metadata.iloc[lm_idx]
                pruned = lm_row.get('pruned_props', frozenset())
                if pruned:
                    tag_str = ", ".join(f"{k}={v}" for k, v in sorted(pruned))
                    raw_strings.append(tag_str)
        osm_landmarks_data = collapse_string_list(raw_strings)
    elif pano_id in PANO_TO_OSM:
        osm_custom_ids = PANO_TO_OSM[pano_id]
        osm_landmarks_raw = [OSM_LANDMARKS[cid].description
                             for cid in osm_custom_ids
                             if cid in OSM_LANDMARKS]
        osm_landmarks_data = collapse_string_list(osm_landmarks_raw)

    # If parquet tables are loaded, also show OSM tag matches for this pano
    osm_tag_matches = []
    if PANO_OSM_MATCHES is not None:
        import polars as pl
        pano_matches = PANO_OSM_MATCHES.filter(
            pl.col("pano_id").cast(pl.Utf8) == pano_id
        )
        if len(pano_matches) > 0:
            # Group by tag_key + pano_lm_value, show unique values
            grouped = (
                pano_matches
                .with_columns([
                    pl.col("tag_key").cast(pl.Utf8),
                    pl.col("pano_lm_value").cast(pl.Utf8),
                    pl.col("sat_lm_value").cast(pl.Utf8),
                ])
                .group_by("tag_key", "pano_lm_value")
                .agg([
                    pl.col("osm_idx").n_unique().alias("osm_count"),
                    pl.col("sat_lm_value").unique().alias("sat_values"),
                ])
                .sort("osm_count", descending=True)
                .head(30)
            )
            for row in grouped.iter_rows(named=True):
                sat_vals = [v for v in row['sat_values'] if v]
                osm_tag_matches.append({
                    'tag_key': row['tag_key'],
                    'pano_value': row['pano_lm_value'],
                    'osm_count': row['osm_count'],
                    'sat_values': sat_vals[:5],
                })

    # Get coordinates — prefer VigorDataset, fall back to sentence landmarks
    lat, lon = None, None
    if VIGOR_DATASET is not None and pano_id in PANO_ID_TO_VIGOR_IDX:
        vigor_idx = PANO_ID_TO_VIGOR_IDX[pano_id]
        pano_row = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]
        lat = float(pano_row['lat'])
        lon = float(pano_row['lon'])
    elif pano_id in PANO_SENTENCES and len(PANO_SENTENCES[pano_id]) > 0:
        first_landmark = PANO_SENTENCES[pano_id][0]
        lat = first_landmark.panorama_lat
        lon = first_landmark.panorama_lon

    return jsonify({
        'name': pano_id,
        'total': len(PANORAMA_DATA),
        'yaw_angles': pano['yaw_angles'],
        'yaw_data': yaw_data,
        'osm_landmarks': osm_landmarks_data,
        'osm_tag_matches': osm_tag_matches,
        'lat': lat,
        'lon': lon
    })


@app.route('/api/similarity/<int:index>')
def get_similarity_data(index):
    """Get bidirectional similarity data between panorama landmarks and nearby OSM landmarks."""
    global PANORAMA_DATA

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano = PANORAMA_DATA[index]
    pano_id = pano['id']

    # Compute similarities
    similarity_results = compare_pano_to_nearby_osm(pano_id)

    if similarity_results is None:
        return jsonify({
            'error': 'Embeddings not loaded',
            'pano_id': pano_id,
            'pano_landmarks': [],
            'osm_landmarks': [],
            'pano_to_osm': {},
            'osm_to_pano': {}
        })

    return jsonify({
        'pano_id': pano_id,
        **similarity_results
    })


@app.route('/api/global_matches/<string:landmark_type>/<path:landmark_key>')
def get_global_matches(landmark_type, landmark_key):
    """Get global similarity matches for a landmark across all panoramas/OSM landmarks."""
    if landmark_type not in ['pano', 'osm']:
        return jsonify({'error': 'Invalid landmark_type'}), 400

    matches = get_global_similarity_matches(landmark_type, landmark_key, top_k=50)
    return jsonify({
        'landmark_type': landmark_type,
        'landmark_key': landmark_key,
        'matches': matches
    })


@app.route('/api/image/panorama/<int:index>')
def get_panorama_image(index):
    global PANORAMA_DATA

    if index < 0 or index >= len(PANORAMA_DATA):
        return 'Invalid index', 404

    return send_file(PANORAMA_DATA[index]['panorama_file'])


@app.route('/api/image/pinhole/<int:index>/<int:yaw>')
def get_pinhole_image(index, yaw):
    global PANORAMA_DATA

    if index < 0 or index >= len(PANORAMA_DATA):
        return 'Invalid index', 404

    pinhole_dir = PANORAMA_DATA[index]['pinhole_dir']

    # Try jpg first, then png
    jpg_path = pinhole_dir / f'yaw_{yaw:03d}.jpg'
    png_path = pinhole_dir / f'yaw_{yaw:03d}.png'

    if jpg_path.exists():
        return send_file(jpg_path)
    elif png_path.exists():
        return send_file(png_path)
    else:
        return 'Image not found', 404


@app.route('/api/image/satellite/<int:sat_idx>')
def get_satellite_image(sat_idx):
    """Serve satellite image from VigorDataset metadata."""
    global VIGOR_DATASET
    if VIGOR_DATASET is None:
        return 'Similarity matrix not loaded', 404
    if sat_idx < 0 or sat_idx >= len(VIGOR_DATASET._satellite_metadata):
        return 'Invalid satellite index', 404
    sat_path = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]['path']
    return send_file(sat_path)


@app.route('/api/satellite_ranking/<int:index>')
def get_satellite_ranking(index):
    """Get top-k satellite patches for a panorama from the similarity matrix."""
    global PANORAMA_DATA, SIMILARITY_MATRIX, VIGOR_DATASET, PANO_ID_TO_VIGOR_IDX
    global PANO_OSM_MATCHES, SAT_OSM_TABLE

    if SIMILARITY_MATRIX is None or VIGOR_DATASET is None:
        return jsonify({'error': 'Similarity matrix not loaded'}), 404

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = PANORAMA_DATA[index]['id']
    pano_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
    if pano_idx is None:
        return jsonify({'error': f'Panorama {pano_id} not found in VigorDataset'}), 404

    # Get similarity row and top-k
    sim_row = SIMILARITY_MATRIX[pano_idx]
    top_k = min(4, sim_row.shape[0])
    top_vals, top_idxs = torch.topk(sim_row, top_k)

    # Get positive satellite set
    pano_row = VIGOR_DATASET._panorama_metadata.iloc[pano_idx]
    positive_set = set(pano_row['positive_satellite_idxs']) | set(pano_row.get('semipositive_satellite_idxs', []))

    # Find best rank among positives
    if positive_set:
        rankings = torch.argsort(sim_row, descending=True)
        best_rank = sim_row.shape[0]
        for pos_idx in positive_set:
            rank = (rankings == pos_idx).nonzero(as_tuple=True)[0].item()
            best_rank = min(best_rank, rank)
        mrr = 1.0 / (best_rank + 1)
    else:
        best_rank = -1
        mrr = None

    # Helper to build satellite info dict
    import polars as pl
    # Pre-filter pano matches once (used by all satellites)
    pano_matches_filtered = None
    if PANO_OSM_MATCHES is not None and SAT_OSM_TABLE is not None:
        try:
            pano_matches_filtered = PANO_OSM_MATCHES.filter(
                pl.col("pano_id").cast(pl.Utf8) == pano_id
            )
            if len(pano_matches_filtered) == 0:
                pano_matches_filtered = None
        except Exception:
            pass

    def build_sat_info(sat_idx, score, rank_label=None):
        shared_landmarks = []
        if pano_matches_filtered is not None:
            try:
                sat_osm_idxs = SAT_OSM_TABLE.filter(
                    pl.col("sat_idx") == sat_idx
                ).select("osm_idx")
                if len(sat_osm_idxs) > 0:
                    joined = pano_matches_filtered.join(sat_osm_idxs, on="osm_idx", how="inner")
                    joined_str = joined.with_columns([
                        pl.col("tag_key").cast(pl.Utf8),
                        pl.col("pano_lm_value").cast(pl.Utf8),
                        pl.col("sat_lm_value").cast(pl.Utf8),
                    ])
                    for row in joined_str.head(20).iter_rows(named=True):
                        shared_landmarks.append({
                            'tag_key': row.get('tag_key', ''),
                            'pano_lm_value': row.get('pano_lm_value', ''),
                            'sat_lm_value': row.get('sat_lm_value', ''),
                        })
            except Exception:
                pass

        sat_osm_landmarks = []
        try:
            sat_meta = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]
            sat_lm_idxs = sat_meta.get('landmark_idxs', [])
            if sat_lm_idxs is not None:
                for lm_idx in sat_lm_idxs[:30]:
                    lm_row = VIGOR_DATASET._landmark_metadata.iloc[lm_idx]
                    pruned = lm_row.get('pruned_props', frozenset())
                    if pruned:
                        sat_osm_landmarks.append(
                            {k: str(v) for k, v in sorted(pruned)})
        except Exception:
            pass

        info = {
            'sat_idx': sat_idx,
            'similarity_score': score,
            'is_positive': sat_idx in positive_set,
            'shared_landmarks': shared_landmarks,
            'sat_osm_landmarks': sat_osm_landmarks,
        }
        if rank_label is not None:
            info['rank_label'] = rank_label
        return info

    # Build top-k satellites
    satellites = []
    top_k_set = set()
    for i in range(top_k):
        sat_idx = top_idxs[i].item()
        top_k_set.add(sat_idx)
        satellites.append(build_sat_info(sat_idx, top_vals[i].item()))

    # Build ground-truth positive satellites (excluding any already in top-k)
    positive_satellites = []
    if positive_set:
        for pos_idx in sorted(positive_set):
            pos_idx = int(pos_idx)
            if pos_idx in top_k_set:
                continue
            score = sim_row[pos_idx].item()
            rank = (rankings == pos_idx).nonzero(as_tuple=True)[0].item()
            info = build_sat_info(pos_idx, score, rank_label=int(rank) + 1)
            positive_satellites.append(info)

    return jsonify({
        'pano_id': pano_id,
        'mrr': mrr,
        'best_rank': best_rank,
        'num_positives': len(positive_set),
        'num_sats': sim_row.shape[0],
        'satellites': satellites,
        'positive_satellites': positive_satellites,
    })


@app.route('/api/correspondence_details/<int:index>/<int:sat_idx>')
def get_correspondence_details(index, sat_idx):
    """Compute on-the-fly correspondence matching between a panorama and satellite."""
    global PANORAMA_DATA, VIGOR_DATASET, PANO_ID_TO_VIGOR_IDX
    global CORRESPONDENCE_MODEL, CORRESPONDENCE_TEXT_EMBEDDINGS
    global CORRESPONDENCE_TEXT_INPUT_DIM, CORRESPONDENCE_DEVICE, PANO_TAGS_FROM_PANO_ID

    if CORRESPONDENCE_MODEL is None or VIGOR_DATASET is None:
        return jsonify({'error': 'Correspondence model not loaded'}), 404

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = PANORAMA_DATA[index]['id']
    pano_landmarks = PANO_TAGS_FROM_PANO_ID.get(pano_id)
    if pano_landmarks is None:
        return jsonify({'error': f'No pano_v2 tags for {pano_id}', 'matches': [],
                        'similarity_score': 0.0})

    pano_tags_list = [dict(lm["tags"]) for lm in pano_landmarks]
    if not pano_tags_list:
        return jsonify({'error': 'No pano tags', 'matches': [],
                        'similarity_score': 0.0})

    # Get OSM landmarks on this satellite
    sat_meta = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]
    lm_idxs = sat_meta.get('landmark_idxs', [])
    if lm_idxs is None or len(lm_idxs) == 0:
        return jsonify({'matches': [], 'similarity_score': 0.0,
                        'pano_landmarks': ['; '.join(f'{k}={v}' for k, v in lm['tags']) for lm in pano_landmarks],
                        'osm_landmarks': []})

    osm_tags_list = []
    osm_tag_strs = []
    for lm_idx in lm_idxs:
        lm_row = VIGOR_DATASET._landmark_metadata.iloc[lm_idx]
        pruned = lm_row.get('pruned_props', frozenset())
        if not pruned:
            continue
        tags = dict(pruned)
        osm_tags_list.append(tags)
        osm_tag_strs.append('; '.join(f'{k}={v}' for k, v in sorted(tags.items())))

    if not osm_tags_list:
        return jsonify({'matches': [], 'similarity_score': 0.0,
                        'pano_landmarks': ['; '.join(f'{k}={v}' for k, v in lm['tags']) for lm in pano_landmarks],
                        'osm_landmarks': []})

    try:
        cost_matrix = compute_cost_matrix(
            pano_tags_list, osm_tags_list, CORRESPONDENCE_MODEL,
            CORRESPONDENCE_TEXT_EMBEDDINGS, CORRESPONDENCE_TEXT_INPUT_DIM,
            CORRESPONDENCE_DEVICE,
        )
        result = match_and_aggregate(cost_matrix, MatchingMethod.HUNGARIAN,
                                     AggregationMode.SUM)
    except Exception as e:
        return jsonify({'error': str(e), 'matches': [], 'similarity_score': 0.0})

    pano_tag_strs = ['; '.join(f'{k}={v}' for k, v in lm['tags']) for lm in pano_landmarks]

    matches = []
    for pi, oi, prob in zip(result.pano_lm_indices, result.osm_lm_indices,
                            result.match_probs):
        matches.append({
            'pano_lm_idx': pi,
            'osm_lm_idx': oi,
            'pano_tags': pano_tag_strs[pi],
            'osm_tags': osm_tag_strs[oi],
            'prob': round(prob, 4),
        })

    return jsonify({
        'matches': matches,
        'similarity_score': round(result.similarity_score, 4),
        'pano_landmarks': pano_tag_strs,
        'osm_landmarks': osm_tag_strs,
        'cost_matrix': cost_matrix.tolist(),
    })


@app.route('/api/similarity_histogram/<int:index>')
def get_similarity_histogram(index):
    """Return histogram of similarity scores with a sample satellite per bin."""
    global PANORAMA_DATA, SIMILARITY_MATRIX, VIGOR_DATASET, PANO_ID_TO_VIGOR_IDX
    global PANO_OSM_MATCHES, SAT_OSM_TABLE
    import numpy as np

    if SIMILARITY_MATRIX is None or VIGOR_DATASET is None:
        return jsonify({'error': 'Similarity matrix not loaded'}), 404
    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = PANORAMA_DATA[index]['id']
    pano_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
    if pano_idx is None:
        return jsonify({'error': f'Panorama {pano_id} not found in VigorDataset'}), 404

    sim_row = SIMILARITY_MATRIX[pano_idx].numpy()

    # Get positive set
    pano_row = VIGOR_DATASET._panorama_metadata.iloc[pano_idx]
    positive_set = set(pano_row['positive_satellite_idxs']) | set(pano_row.get('semipositive_satellite_idxs', []))

    # Build histogram
    num_bins = int(request.args.get('bins', 50))
    counts, bin_edges = np.histogram(sim_row, bins=num_bins)

    # Pre-filter pano matches once
    import polars as pl
    pano_matches_filtered = None
    if PANO_OSM_MATCHES is not None and SAT_OSM_TABLE is not None:
        try:
            pano_matches_filtered = PANO_OSM_MATCHES.filter(
                pl.col("pano_id").cast(pl.Utf8) == pano_id
            )
            if len(pano_matches_filtered) == 0:
                pano_matches_filtered = None
        except Exception:
            pass

    # For each bin, pick the satellite closest to the bin center and get its tag matches
    bin_samples = []
    digitized = np.digitize(sim_row, bin_edges[1:-1])  # bin index per satellite (0-based)
    for bin_idx in range(num_bins):
        if counts[bin_idx] == 0:
            bin_samples.append(None)
            continue

        # Find satellites in this bin
        mask = digitized == bin_idx
        bin_sat_idxs = np.where(mask)[0]
        bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        distances = np.abs(sim_row[bin_sat_idxs] - bin_center)
        sample_sat_idx = int(bin_sat_idxs[np.argmin(distances)])

        # Get tag matches for this sample satellite
        shared_landmarks = []
        if pano_matches_filtered is not None:
            try:
                sat_osm_idxs = SAT_OSM_TABLE.filter(
                    pl.col("sat_idx") == sample_sat_idx
                ).select("osm_idx")
                if len(sat_osm_idxs) > 0:
                    joined = pano_matches_filtered.join(sat_osm_idxs, on="osm_idx", how="inner")
                    joined_str = joined.with_columns([
                        pl.col("tag_key").cast(pl.Utf8),
                        pl.col("pano_lm_value").cast(pl.Utf8),
                        pl.col("sat_lm_value").cast(pl.Utf8),
                    ])
                    for row in joined_str.head(20).iter_rows(named=True):
                        shared_landmarks.append({
                            'tag_key': row.get('tag_key', ''),
                            'pano_value': row.get('pano_lm_value', ''),
                            'sat_value': row.get('sat_lm_value', ''),
                        })
            except Exception:
                pass

        bin_samples.append({
            'sat_idx': sample_sat_idx,
            'score': float(sim_row[sample_sat_idx]),
            'is_positive': sample_sat_idx in positive_set,
            'shared_landmarks': shared_landmarks,
        })

    # Also mark which bins contain positives
    positive_counts = np.zeros(num_bins, dtype=int)
    for pos_idx in positive_set:
        pos_idx = int(pos_idx)
        b = min(digitized[pos_idx], num_bins - 1)
        positive_counts[b] += 1

    return jsonify({
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist(),
        'positive_counts': positive_counts.tolist(),
        'bin_samples': bin_samples,
        'num_sats': len(sim_row),
    })


@app.route('/api/panorama_mrr_ranking')
def get_panorama_mrr_ranking():
    """Return panoramas sorted by MRR (best first)."""
    global MRR_RANKING
    return jsonify({'ranking': MRR_RANKING})


@app.route('/api/satellite_map/<int:index>')
def get_satellite_map_data(index):
    """Get top-50 satellite locations + positives for map display."""
    global PANORAMA_DATA, SIMILARITY_MATRIX, VIGOR_DATASET, PANO_ID_TO_VIGOR_IDX

    if SIMILARITY_MATRIX is None or VIGOR_DATASET is None:
        return jsonify({'error': 'Similarity matrix not loaded'}), 404

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = PANORAMA_DATA[index]['id']
    pano_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
    if pano_idx is None:
        return jsonify({'error': f'Panorama {pano_id} not found'}), 404

    sim_row = SIMILARITY_MATRIX[pano_idx]
    top_k = min(50, sim_row.shape[0])
    top_vals, top_idxs = torch.topk(sim_row, top_k)

    pano_row = VIGOR_DATASET._panorama_metadata.iloc[pano_idx]
    positive_set = set(pano_row['positive_satellite_idxs']) | set(pano_row.get('semipositive_satellite_idxs', []))
    pano_lat = float(pano_row['lat'])
    pano_lon = float(pano_row['lon'])

    markers = []
    for i in range(top_k):
        sat_idx = top_idxs[i].item()
        sat_meta = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]
        # Get landmark summary
        lm_tags = []
        sat_lm_idxs = sat_meta.get('landmark_idxs', [])
        if sat_lm_idxs is not None:
            for lm_idx in sat_lm_idxs[:10]:
                pruned = VIGOR_DATASET._landmark_metadata.iloc[lm_idx].get('pruned_props', frozenset())
                if pruned:
                    lm_tags.append(", ".join(f"{k}={v}" for k, v in sorted(pruned)))
        markers.append({
            'sat_idx': sat_idx,
            'lat': float(sat_meta['lat']),
            'lon': float(sat_meta['lon']),
            'rank': i + 1,
            'score': top_vals[i].item(),
            'is_positive': sat_idx in positive_set,
            'landmarks': lm_tags[:5],
        })

    # Also add positives not in top-k
    rankings = torch.argsort(sim_row, descending=True)
    top_k_idx_set = set(top_idxs.tolist())
    for pos_idx in sorted(positive_set):
        pos_idx = int(pos_idx)
        if pos_idx in top_k_idx_set:
            continue
        sat_meta = VIGOR_DATASET._satellite_metadata.iloc[pos_idx]
        rank = int((rankings == pos_idx).nonzero(as_tuple=True)[0].item())
        lm_tags = []
        sat_lm_idxs = sat_meta.get('landmark_idxs', [])
        if sat_lm_idxs is not None:
            for lm_idx in sat_lm_idxs[:10]:
                pruned = VIGOR_DATASET._landmark_metadata.iloc[lm_idx].get('pruned_props', frozenset())
                if pruned:
                    lm_tags.append(", ".join(f"{k}={v}" for k, v in sorted(pruned)))
        markers.append({
            'sat_idx': pos_idx,
            'lat': float(sat_meta['lat']),
            'lon': float(sat_meta['lon']),
            'rank': rank + 1,
            'score': float(sim_row[pos_idx].item()),
            'is_positive': True,
            'landmarks': lm_tags[:5],
        })

    return jsonify({
        'pano_lat': pano_lat,
        'pano_lon': pano_lon,
        'markers': markers,
    })


@app.route('/api/search_landmarks')
def search_landmarks():
    """Search OSM landmarks from .feather data by tag key or value substring."""
    global VIGOR_DATASET, PANO_ID_TO_VIGOR_IDX

    if VIGOR_DATASET is None or VIGOR_DATASET._landmark_metadata is None:
        return jsonify({'results': []})

    query = request.args.get('q', '').strip().lower()
    if not query:
        return jsonify({'results': []})

    # Search through pruned_props of each landmark for matching tag keys/values
    from collections import defaultdict
    # Group: (tag_key, tag_value) -> list of landmark indices
    matches_by_tag = defaultdict(list)
    for lm_idx, lm_row in VIGOR_DATASET._landmark_metadata.iterrows():
        pruned = lm_row.get('pruned_props', frozenset())
        if not pruned:
            continue
        for k, v in pruned:
            if query in str(k).lower() or query in str(v).lower():
                matches_by_tag[(str(k), str(v))].append(lm_idx)

    if not matches_by_tag:
        return jsonify({'results': []})

    # For each tag match, find associated panorama IDs
    pano_id_from_idx = {idx: row['pano_id']
                        for idx, (_, row) in enumerate(VIGOR_DATASET._panorama_metadata.iterrows())}
    results = []
    for (tag_key, tag_value), lm_idxs in sorted(matches_by_tag.items(), key=lambda x: -len(x[1]))[:50]:
        # Find panoramas that have any of these landmarks
        pano_ids = set()
        for lm_idx in lm_idxs:
            pano_idxs_list = VIGOR_DATASET._landmark_metadata.iloc[lm_idx].get('panorama_idxs', [])
            if pano_idxs_list:
                for pi in pano_idxs_list:
                    pid = pano_id_from_idx.get(pi)
                    if pid:
                        pano_ids.add(pid)
        results.append({
            'tag_key': tag_key,
            'pano_lm_value': tag_value,
            'sat_lm_value': '',
            'count': len(lm_idxs),
            'pano_ids': list(pano_ids)[:20],
        })

    return jsonify({'results': results})


def main():
    global PANORAMA_DATA, PANO_SENTENCES, OSM_LANDMARKS, PANO_TO_OSM
    global PANO_EMBEDDINGS, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE
    global OSM_EMBEDDINGS, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE
    global SIMILARITY_MATRIX, VIGOR_DATASET, PANO_OSM_MATCHES, SAT_OSM_TABLE
    global PANO_ID_TO_VIGOR_IDX, MRR_RANKING
    global CORRESPONDENCE_MODEL, CORRESPONDENCE_TEXT_EMBEDDINGS
    global CORRESPONDENCE_TEXT_INPUT_DIM, CORRESPONDENCE_DEVICE, PANO_TAGS_FROM_PANO_ID

    parser = argparse.ArgumentParser(description='Panorama viewer web app')
    parser.add_argument('--panorama_dir', type=str, required=True,
                       help='Directory containing panorama images (VIGOR format)')
    parser.add_argument('--pinhole_dir', type=str, required=True,
                       help='Directory containing pinhole image subdirectories')
    parser.add_argument('--pano_landmarks_dir', type=str, required=True,
                       help='Directory containing panorama landmarks (expects sentences/ and embeddings/ subdirs)')
    parser.add_argument('--osm_landmarks_dir', type=str, default=None,
                       help='Directory containing OSM landmarks (expects sentences/ and embeddings/ subdirs)')
    parser.add_argument('--osm_landmarks_geojson', type=str, default=None,
                       help='Path to OSM landmarks GeoJSON file')
    parser.add_argument('--similarity_matrix', type=str, default=None,
                       help='Path to .pt similarity matrix file')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to VIGOR city directory (needed for similarity matrix mode)')
    parser.add_argument('--landmark_version', type=str, default=None,
                       help='Landmark version string (default: auto-detect)')
    parser.add_argument('--landmark_tables_dir', type=str, default=None,
                       help='Path to dir with parquet match tables')
    parser.add_argument('--city_name', type=str, default=None,
                       help='City name for table filenames (default: inferred from dataset_path)')
    parser.add_argument('--correspondence_model_path', type=str, default=None,
                       help='Path to trained CorrespondenceClassifier .pt file')
    parser.add_argument('--correspondence_text_embeddings', type=str, default=None,
                       help='Path to text embeddings pickle for correspondence model')
    parser.add_argument('--pano_v2_base', type=str, default=None,
                       help='Base path for pano_v2 embeddings (contains city subdirs)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    startup_start = time.time()

    # Construct paths for panorama landmarks
    pano_sentences_dirs = [str(Path(args.pano_landmarks_dir) / 'sentences')]
    pano_embeddings_dir = str(Path(args.pano_landmarks_dir) / 'embeddings')

    # Construct paths for OSM landmarks (if provided)
    if args.osm_landmarks_dir:
        osm_sentences_dir = Path(args.osm_landmarks_dir) / 'sentences'
        osm_embeddings_dir = Path(args.osm_landmarks_dir) / 'embeddings'
    else:
        osm_sentences_dir = None
        osm_embeddings_dir = None

    # Step 1: Load OSM landmarks (independent, no duplication)
    if args.osm_landmarks_geojson and osm_sentences_dir:
        print("\n" + "="*60)
        print("STEP 1: Loading OSM landmarks")
        print("="*60)
        osm_data = load_osm_landmarks(
            Path(args.osm_landmarks_geojson),
            osm_sentences_dir,
            osm_embeddings_dir
        )
        OSM_LANDMARKS, OSM_EMBEDDINGS, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE = osm_data
        print(f"Loaded {len(OSM_LANDMARKS)} OSM landmarks")
        if OSM_EMBEDDINGS is not None:
            print(f"  Loaded {OSM_EMBEDDINGS.shape[0]} OSM embeddings")

        # Step 2: Pre-compute panorama→OSM associations (indices only)
        print("\n" + "="*60)
        print("STEP 2: Computing panorama→OSM associations")
        print("="*60)
        step2_start = time.time()
        PANO_TO_OSM = compute_panorama_to_osm_associations(
            args.panorama_dir,
            OSM_LANDMARKS,
            args.osm_landmarks_geojson
        )
        step2_time = time.time() - step2_start
        print(f"Computed associations for {len(PANO_TO_OSM)} panoramas in {step2_time:.1f}s")
    else:
        print("\nSkipping OSM landmarks (no OSM data provided)")
        OSM_LANDMARKS = {}
        OSM_EMBEDDINGS = None
        OSM_EMBEDDING_INDEX = {}
        OSM_INDEX_REVERSE = []
        PANO_TO_OSM = {}

    # Step 3: Load panorama landmark sentences
    print("\n" + "="*60)
    print("STEP 3: Loading panorama landmark sentences")
    print("="*60)
    step3_start = time.time()
    PANO_SENTENCES = load_sentence_data(pano_sentences_dirs)
    step3_time = time.time() - step3_start
    print(f"Loaded landmarks for {len(PANO_SENTENCES)} panoramas in {step3_time:.1f}s")

    # Step 3.5: Load panorama embeddings
    print("\n" + "="*60)
    print("STEP 3.5: Loading panorama embeddings")
    print("="*60)
    PANO_EMBEDDINGS, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE = load_pano_embeddings(
        pano_embeddings_dir,
        PANO_SENTENCES
    )
    if PANO_EMBEDDINGS is not None:
        print(f"  Loaded {PANO_EMBEDDINGS.shape[0]} panorama embeddings")
    else:
        print("  No panorama embeddings found")

    # Step 4: Find common panoramas
    print("\n" + "="*60)
    print("STEP 4: Finding common panoramas")
    print("="*60)
    step4_start = time.time()
    PANORAMA_DATA = find_common_panoramas(
        args.panorama_dir,
        args.pinhole_dir,
        PANO_SENTENCES,
        dataset_path=args.dataset_path if hasattr(args, 'dataset_path') else None,
    )
    step4_time = time.time() - step4_start
    print(f"Found {len(PANORAMA_DATA)} panoramas with complete data in {step4_time:.1f}s")

    if len(PANORAMA_DATA) == 0:
        print("ERROR: No panoramas found with complete data!")
        return

    # Step 5: Load similarity matrix (optional)
    if args.similarity_matrix and args.dataset_path:
        print("\n" + "="*60)
        print("STEP 5: Loading tag-weight similarity matrix")
        print("="*60)
        step5_start = time.time()

        import polars as pl
        from experimental.overhead_matching.swag.data import vigor_dataset as vd

        dataset_path = Path(args.dataset_path)
        city_name = args.city_name or dataset_path.name.lower()

        # Auto-detect landmark version
        landmark_version = args.landmark_version
        if landmark_version is None:
            landmarks_dir = dataset_path / "landmarks"
            if landmarks_dir.exists():
                feather_files = list(landmarks_dir.glob("*.feather"))
                if len(feather_files) == 1:
                    landmark_version = feather_files[0].stem
                    print(f"  Auto-detected landmark version: {landmark_version}")
                elif len(feather_files) == 0:
                    print("  Warning: No .feather files in landmarks/ dir")
                else:
                    print(f"  Warning: Multiple .feather files, specify --landmark_version")

        # Load VigorDataset
        print(f"  Loading VigorDataset from {dataset_path}")
        config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=True,
            landmark_version=landmark_version,
        )
        VIGOR_DATASET = vd.VigorDataset(dataset_path, config)
        print(f"  {len(VIGOR_DATASET._panorama_metadata)} panos, "
              f"{len(VIGOR_DATASET._satellite_metadata)} sats")

        # Build pano_id -> vigor_idx mapping
        for idx, (_, row) in enumerate(VIGOR_DATASET._panorama_metadata.iterrows()):
            PANO_ID_TO_VIGOR_IDX[row['pano_id']] = idx

        # Load similarity matrix
        print(f"  Loading similarity matrix from {args.similarity_matrix}")
        SIMILARITY_MATRIX = torch.load(args.similarity_matrix, weights_only=False)
        print(f"  Similarity matrix shape: {SIMILARITY_MATRIX.shape}")

        # Load parquet match tables
        tables_dir = Path(args.landmark_tables_dir) if args.landmark_tables_dir else dataset_path / "landmark_tables"
        matches_path = tables_dir / f"{city_name}_pano_osm_matches.parquet"
        sat_osm_path = tables_dir / f"{city_name}_sat_osm_table.parquet"

        if matches_path.exists() and sat_osm_path.exists():
            print(f"  Loading match tables from {tables_dir}")
            PANO_OSM_MATCHES = pl.read_parquet(str(matches_path))
            SAT_OSM_TABLE = pl.read_parquet(str(sat_osm_path))
            print(f"  Loaded {len(PANO_OSM_MATCHES)} pano-osm matches, {len(SAT_OSM_TABLE)} sat-osm entries")
        else:
            print(f"  Warning: Match tables not found at {tables_dir}")

        # Compute per-panorama MRR for navigation
        print("  Computing per-panorama MRR ranking...")
        rankings = torch.argsort(SIMILARITY_MATRIX, dim=1, descending=True)
        pano_id_to_data_idx = {pano['id']: i for i, pano in enumerate(PANORAMA_DATA)}

        for pano_data_idx, pano in enumerate(PANORAMA_DATA):
            pano_id = pano['id']
            vigor_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
            if vigor_idx is None:
                continue
            vigor_row = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]
            positive_set = set(vigor_row['positive_satellite_idxs']) | set(vigor_row.get('semipositive_satellite_idxs', []))
            if not positive_set:
                MRR_RANKING.append({'pano_idx': pano_data_idx, 'pano_id': pano_id, 'mrr': 0.0, 'best_rank': -1})
                continue
            best_rank = SIMILARITY_MATRIX.shape[1]
            for pos_idx in positive_set:
                rank = (rankings[vigor_idx] == pos_idx).nonzero(as_tuple=True)[0].item()
                best_rank = min(best_rank, rank)
            MRR_RANKING.append({
                'pano_idx': pano_data_idx,
                'pano_id': pano_id,
                'mrr': 1.0 / (best_rank + 1),
                'best_rank': best_rank,
            })

        MRR_RANKING.sort(key=lambda x: x['mrr'], reverse=True)

        step5_time = time.time() - step5_start
        print(f"  Loaded similarity data in {step5_time:.1f}s")
        avg_mrr = sum(r['mrr'] for r in MRR_RANKING) / len(MRR_RANKING) if MRR_RANKING else 0
        print(f"  Average MRR: {avg_mrr:.4f}, best: {MRR_RANKING[0]['mrr']:.4f}, worst: {MRR_RANKING[-1]['mrr']:.4f}")

    # Step 6: Load correspondence model (optional)
    if args.correspondence_model_path and args.correspondence_text_embeddings and args.pano_v2_base:
        print("\n" + "="*60)
        print("STEP 6: Loading correspondence model")
        print("="*60)
        step6_start = time.time()

        CORRESPONDENCE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load text embeddings
        print(f"  Loading text embeddings from {args.correspondence_text_embeddings}")
        CORRESPONDENCE_TEXT_EMBEDDINGS = load_text_embeddings(Path(args.correspondence_text_embeddings))
        CORRESPONDENCE_TEXT_INPUT_DIM = next(iter(CORRESPONDENCE_TEXT_EMBEDDINGS.values())).shape[0]
        print(f"  {len(CORRESPONDENCE_TEXT_EMBEDDINGS)} entries, dim={CORRESPONDENCE_TEXT_INPUT_DIM}")

        # Load model
        print(f"  Loading model from {args.correspondence_model_path}")
        encoder_config = TagBundleEncoderConfig(
            text_input_dim=CORRESPONDENCE_TEXT_INPUT_DIM, text_proj_dim=128)
        classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
        CORRESPONDENCE_MODEL = CorrespondenceClassifier(classifier_config).to(CORRESPONDENCE_DEVICE)
        CORRESPONDENCE_MODEL.load_state_dict(
            torch.load(args.correspondence_model_path, map_location=CORRESPONDENCE_DEVICE,
                       weights_only=True))
        CORRESPONDENCE_MODEL.eval()
        print(f"  Model loaded, device={CORRESPONDENCE_DEVICE}")

        # Load VigorDataset if not already loaded in step 5
        if VIGOR_DATASET is None and args.dataset_path:
            from experimental.overhead_matching.swag.data import vigor_dataset as vd

            dataset_path = Path(args.dataset_path)
            landmark_version = args.landmark_version
            if landmark_version is None:
                landmarks_dir = dataset_path / "landmarks"
                if landmarks_dir.exists():
                    feather_files = list(landmarks_dir.glob("*.feather"))
                    if len(feather_files) == 1:
                        landmark_version = feather_files[0].stem
                        print(f"  Auto-detected landmark version: {landmark_version}")

            print(f"  Loading VigorDataset from {dataset_path}")
            config = vd.VigorDatasetConfig(
                satellite_tensor_cache_info=None,
                panorama_tensor_cache_info=None,
                should_load_images=False,
                should_load_landmarks=True,
                landmark_version=landmark_version,
            )
            VIGOR_DATASET = vd.VigorDataset(dataset_path, config)
            print(f"  {len(VIGOR_DATASET._panorama_metadata)} panos, "
                  f"{len(VIGOR_DATASET._satellite_metadata)} sats")

        # Load pano_v2 tags
        print(f"  Loading pano_v2 tags from {args.pano_v2_base}")
        PANO_TAGS_FROM_PANO_ID = extract_panorama_data_across_cities(
            Path(args.pano_v2_base), extract_tags_from_pano_data,
        )
        print(f"  Loaded tags for {len(PANO_TAGS_FROM_PANO_ID)} panoramas")

        step6_time = time.time() - step6_start
        print(f"  Loaded correspondence model in {step6_time:.1f}s")

    startup_time = time.time() - startup_start
    print("\n" + "="*60)
    print(f"TOTAL STARTUP TIME: {startup_time:.1f}s")
    print("="*60)
    print(f"Starting web server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, use_reloader=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
