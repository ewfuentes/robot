#!/usr/bin/env python3
"""
Simple web app to view panoramas, pinhole images, and landmark sentences.

Usage:
    python panorama_viewer.py \
        --panorama_dir /data/overhead_matching/datasets/VIGOR/Chicago/panorama \
        --pinhole_dir /tmp/pinhole_images/Chicagojpg \
        --sentence_dirs /tmp/pano_sentences/source1 /tmp/pano_sentences/source2
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template_string, send_file, jsonify
import base64
import hashlib
import pandas as pd
import pickle
import time
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import numpy as np

# Import from existing modules
from experimental.overhead_matching.swag.data.vigor_dataset import load_landmark_geojson
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark)
from common.gps import web_mercator

app = Flask(__name__)

# Global data
PANORAMA_DATA = []
CURRENT_INDEX = 0
OSM_LANDMARKS = {}  # Maps panorama_id -> list of landmark sentences
OSM_EMBEDDINGS = {}  # Maps custom_id -> torch.Tensor (embedding vector)
PANORAMA_EMBEDDINGS = {}  # Maps custom_id -> torch.Tensor (embedding vector)
OSM_CUSTOM_IDS = {}  # Maps panorama_id -> list of custom_ids
PANO_CUSTOM_IDS = {}  # Maps (source_name, pano_id, yaw, landmark_idx) -> custom_id
GLOBAL_BEST_MATCHES = None  # Aggregated best matches across all panoramas
CACHED_PANO_EMBEDDINGS_LIST = None  # Cached list of all pano embeddings with metadata
CACHED_OSM_EMBEDDINGS_LIST = None  # Cached list of all OSM embeddings with metadata
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Panorama Viewer</title>
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
        .source-sentences {
            margin-bottom: 15px;
        }
        .source-sentences h4 {
            color: #007bff;
            margin: 0 0 8px 0;
            font-size: 14px;
            font-weight: 600;
        }
        .source-sentences ul {
            margin: 0;
            padding-left: 20px;
            font-size: 13px;
        }
        .source-sentences li {
            margin-bottom: 5px;
            line-height: 1.4;
        }
        .landmark-all-mode {
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 3px;
            border-left: 4px solid;
        }
        .landmark-individual-mode {
            /* No special styling for individual mode */
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
        .similarity-match {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .similarity-match h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 14px;
        }
        .match-list {
            margin: 5px 0;
        }
        .match-item {
            padding: 8px;
            margin: 5px 0;
            border-radius: 3px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .match-score {
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }
        .score-high {
            background: #d4edda;
            color: #155724;
        }
        .score-medium {
            background: #fff3cd;
            color: #856404;
        }
        .score-low {
            background: #f8d7da;
            color: #721c24;
        }
        .landmark-item {
            padding: 8px 10px;
            margin: 4px 0;
            border-radius: 4px;
            border: 2px solid transparent;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .landmark-item:hover {
            background: #e9ecef;
            border-color: #007bff;
        }
        .landmark-item.selected {
            background: #d4edff;
            border-color: #007bff;
        }
        .landmark-item.highlighted {
            background: #fff3cd;
            border-color: #ffc107;
        }
        .landmark-match-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 6px;
            font-weight: 600;
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
            </div>
        </div>

        <div>
            <h2>Pinhole Views</h2>
            <div style="margin-bottom: 15px; padding: 10px; background: #f0f8ff; border-radius: 4px; font-size: 13px;">
                <strong>Legend:</strong>
                <span style="margin-left: 10px;">Landmarks with matching colors and badges (e.g., <span style="background:#ddd;padding:2px 4px;border-radius:2px;">90°</span>) appear in multiple views (from "all" mode sources).</span>
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

        <div class="panorama-section" id="similarity-section" style="display:none;">
            <h2>Embedding Similarity Analysis</h2>
            <p style="color:#666;font-size:14px;margin-bottom:10px;">
                Click on any landmark to see its top-3 matches. Color intensity shows similarity strength.
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                    <h3 style="font-size:14px;margin:0 0 10px 0;color:#333;">OSM Landmarks</h3>
                    <div id="osm-similarity-list" style="max-height:400px;overflow-y:auto;">
                        <!-- OSM landmarks will be inserted here -->
                    </div>
                </div>
                <div>
                    <h3 style="font-size:14px;margin:0 0 10px 0;color:#333;">Panorama Landmarks</h3>
                    <div id="pano-similarity-list" style="max-height:400px;overflow-y:auto;">
                        <!-- Panorama landmarks will be inserted here -->
                    </div>
                </div>
            </div>
        </div>

        <div class="panorama-section" id="global-matches-section" style="display:none;">
            <h2>Global Best Matches</h2>
            <p style="color:#666;font-size:14px;margin-bottom:10px;">
                Top matches across all panoramas. Click to navigate to the panorama containing the match.
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                <div>
                    <h3 style="font-size:14px;margin:0 0 10px 0;color:#333;">OSM → Panorama (Top 25)</h3>
                    <div id="global-osm-to-pano-list" style="max-height:500px;overflow-y:auto;">
                        <!-- Global OSM->Pano matches will be inserted here -->
                    </div>
                </div>
                <div>
                    <h3 style="font-size:14px;margin:0 0 10px 0;color:#333;">Panorama → OSM (Top 25)</h3>
                    <div id="global-pano-to-osm-list" style="max-height:500px;overflow-y:auto;">
                        <!-- Global Pano->OSM matches will be inserted here -->
                    </div>
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
        let globalMatchesTimeout = null;
        let currentGlobalMatchesController = null;

        // Color palette for landmarks (distinct colors)
        const LANDMARK_COLORS = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84',
            '#6C5B7B', '#355C7D', '#F67280', '#C8D6AF', '#8E7AB5'
        ];

        function getLandmarkColor(landmarkId) {
            return LANDMARK_COLORS[landmarkId % LANDMARK_COLORS.length];
        }

        // Global functions for similarity interaction
        function selectOsmLandmarkGroup(osmIndicesStr) {
            const osmIndices = osmIndicesStr.split(',').map(s => parseInt(s));
            const osmItem = document.querySelector(`[data-osm-indices="${osmIndicesStr}"]`);

            // If already selected, deselect (return to original view and unsort)
            if (osmItem && osmItem.classList.contains('selected')) {
                document.querySelectorAll('.landmark-item').forEach(el => {
                    el.classList.remove('selected', 'highlighted');
                    // Reset badge to original best score
                    const badge = el.querySelector('.landmark-match-badge');
                    if (badge && el.dataset.bestScore) {
                        const bestScore = parseFloat(el.dataset.bestScore);
                        const scorePercent = (bestScore * 100).toFixed(0);
                        const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';
                        badge.className = `landmark-match-badge ${scoreClass}`;
                        badge.textContent = `${scorePercent}%`;
                    }
                });
                // Restore original order
                sortPanoramaLandmarksForOsmGroup(null);
                return;
            }

            // Clear all selections and reset badges
            document.querySelectorAll('.landmark-item').forEach(el => {
                el.classList.remove('selected', 'highlighted');
                // Reset badge to original best score
                const badge = el.querySelector('.landmark-match-badge');
                if (badge && el.dataset.bestScore) {
                    const bestScore = parseFloat(el.dataset.bestScore);
                    const scorePercent = (bestScore * 100).toFixed(0);
                    const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `${scorePercent}%`;
                }
            });

            // Select this OSM landmark group
            if (osmItem) osmItem.classList.add('selected');

            // Collect and merge matches from all OSM indices in the group
            // Create a map of pano_idx -> best score across all OSM instances
            const panoScoreMap = {};
            osmIndices.forEach(osmIdx => {
                const matches = window.currentSimilarityData.osm_to_pano[osmIdx].matches;
                matches.forEach(match => {
                    if (!panoScoreMap[match.pano_idx] || match.score > panoScoreMap[match.pano_idx]) {
                        panoScoreMap[match.pano_idx] = match.score;
                    }
                });
            });

            const matchedPanoIndices = new Set(Object.keys(panoScoreMap).map(k => parseInt(k)));

            // Find panorama groups that contain any of the matched indices and calculate their best scores
            const panoGroupScores = [];
            document.querySelectorAll('#pano-similarity-list .landmark-item').forEach(item => {
                const groupIndices = item.dataset.panoIndices.split(',').map(s => parseInt(s));
                let bestScore = 0;
                groupIndices.forEach(idx => {
                    if (matchedPanoIndices.has(idx) && panoScoreMap[idx] > bestScore) {
                        bestScore = panoScoreMap[idx];
                    }
                });
                if (bestScore > 0) {
                    panoGroupScores.push({ item: item, score: bestScore });
                }
            });

            // Sort by score and take top 3 for highlighting
            panoGroupScores.sort((a, b) => b.score - a.score);
            const top3 = new Set(panoGroupScores.slice(0, 3).map(g => g.item));

            // Update badges for all matched items
            panoGroupScores.forEach((groupData, index) => {
                const scorePercent = (groupData.score * 100).toFixed(0);
                const scoreClass = groupData.score > 0.8 ? 'score-high' : groupData.score > 0.6 ? 'score-medium' : 'score-low';
                const badge = groupData.item.querySelector('.landmark-match-badge');

                // Highlight only top 3
                if (top3.has(groupData.item)) {
                    groupData.item.classList.add('highlighted');
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `#${index + 1}: ${scorePercent}%`;
                } else {
                    // Show score but don't highlight
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `${scorePercent}%`;
                }
            });

            // Reset badges for items with no match
            document.querySelectorAll('#pano-similarity-list .landmark-item').forEach(item => {
                if (!panoGroupScores.find(g => g.item === item)) {
                    const badge = item.querySelector('.landmark-match-badge');
                    if (badge) {
                        badge.className = 'landmark-match-badge score-low';
                        badge.textContent = '—';
                    }
                }
            });

            // Sort panorama landmarks by match score to this OSM landmark group
            sortPanoramaLandmarksForOsmGroup(osmIndices);
        }

        function selectPanoLandmarkGroup(panoIndicesStr) {
            const panoIndices = panoIndicesStr.split(',').map(s => parseInt(s));
            const panoItem = document.querySelector(`[data-pano-indices="${panoIndicesStr}"]`);

            // If already selected, deselect (return to original view)
            if (panoItem && panoItem.classList.contains('selected')) {
                document.querySelectorAll('.landmark-item').forEach(el => {
                    el.classList.remove('selected', 'highlighted');
                    // Reset badge to original best score
                    const badge = el.querySelector('.landmark-match-badge');
                    if (badge && el.dataset.bestScore) {
                        const bestScore = parseFloat(el.dataset.bestScore);
                        const scorePercent = (bestScore * 100).toFixed(0);
                        const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';
                        badge.className = `landmark-match-badge ${scoreClass}`;
                        badge.textContent = `${scorePercent}%`;
                    }
                });
                // Restore original order
                sortOsmLandmarks(null);
                return;
            }

            // Clear all selections and reset badges
            document.querySelectorAll('.landmark-item').forEach(el => {
                el.classList.remove('selected', 'highlighted');
                // Reset badge to original best score
                const badge = el.querySelector('.landmark-match-badge');
                if (badge && el.dataset.bestScore) {
                    const bestScore = parseFloat(el.dataset.bestScore);
                    const scorePercent = (bestScore * 100).toFixed(0);
                    const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `${scorePercent}%`;
                }
            });

            // Select this panorama landmark group
            if (panoItem) panoItem.classList.add('selected');

            // Collect and merge matches from all pano indices in the group
            // Create a map of osm_idx -> best score across all pano instances
            const osmScoreMap = {};
            panoIndices.forEach(panoIdx => {
                const matches = window.currentSimilarityData.pano_to_osm[panoIdx].matches;
                matches.forEach(match => {
                    if (!osmScoreMap[match.osm_idx] || match.score > osmScoreMap[match.osm_idx]) {
                        osmScoreMap[match.osm_idx] = match.score;
                    }
                });
            });

            const matchedOsmIndices = new Set(Object.keys(osmScoreMap).map(k => parseInt(k)));

            // Find OSM groups that contain any of the matched indices and calculate their best scores
            const osmGroupScores = [];
            document.querySelectorAll('#osm-similarity-list .landmark-item').forEach(item => {
                const groupIndices = item.dataset.osmIndices.split(',').map(s => parseInt(s));
                let bestScore = 0;
                groupIndices.forEach(idx => {
                    if (matchedOsmIndices.has(idx) && osmScoreMap[idx] > bestScore) {
                        bestScore = osmScoreMap[idx];
                    }
                });
                if (bestScore > 0) {
                    osmGroupScores.push({ item: item, score: bestScore });
                }
            });

            // Sort by score and take top 3 for highlighting
            osmGroupScores.sort((a, b) => b.score - a.score);
            const top3 = new Set(osmGroupScores.slice(0, 3).map(g => g.item));

            // Update badges for all matched items
            osmGroupScores.forEach((groupData, index) => {
                const scorePercent = (groupData.score * 100).toFixed(0);
                const scoreClass = groupData.score > 0.8 ? 'score-high' : groupData.score > 0.6 ? 'score-medium' : 'score-low';
                const badge = groupData.item.querySelector('.landmark-match-badge');

                // Highlight only top 3
                if (top3.has(groupData.item)) {
                    groupData.item.classList.add('highlighted');
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `#${index + 1}: ${scorePercent}%`;
                } else {
                    // Show score but don't highlight
                    badge.className = `landmark-match-badge ${scoreClass}`;
                    badge.textContent = `${scorePercent}%`;
                }
            });

            // Reset badges for OSM items with no match
            document.querySelectorAll('#osm-similarity-list .landmark-item').forEach(item => {
                if (!osmGroupScores.find(g => g.item === item)) {
                    const badge = item.querySelector('.landmark-match-badge');
                    if (badge) {
                        badge.className = 'landmark-match-badge score-low';
                        badge.textContent = '—';
                    }
                }
            });

            // Sort OSM landmarks by match score to this panorama landmark group
            sortOsmLandmarksForPanoGroup(panoIndices);
        }

        function sortPanoramaLandmarksForOsmGroup(osmIndices) {
            const container = document.getElementById('pano-similarity-list');
            if (!container) return;

            const items = Array.from(container.querySelectorAll('.landmark-item'));

            if (osmIndices === null) {
                // Restore original order using data-original-index
                items.sort((a, b) => {
                    const aIdx = parseInt(a.dataset.originalIndex || '0');
                    const bIdx = parseInt(b.dataset.originalIndex || '0');
                    return aIdx - bIdx;
                });
            } else {
                // Sort by match score to selected OSM landmark group
                // Collect best scores from all OSM indices in the group
                const panoScoreMap = {};
                osmIndices.forEach(osmIdx => {
                    const matches = window.currentSimilarityData.osm_to_pano[osmIdx].matches;
                    matches.forEach(match => {
                        if (!panoScoreMap[match.pano_idx] || match.score > panoScoreMap[match.pano_idx]) {
                            panoScoreMap[match.pano_idx] = match.score;
                        }
                    });
                });

                items.sort((a, b) => {
                    const aGroupIndices = a.dataset.panoIndices.split(',').map(s => parseInt(s));
                    const bGroupIndices = b.dataset.panoIndices.split(',').map(s => parseInt(s));

                    let aScore = -1;
                    aGroupIndices.forEach(idx => {
                        if (panoScoreMap[idx] && panoScoreMap[idx] > aScore) {
                            aScore = panoScoreMap[idx];
                        }
                    });

                    let bScore = -1;
                    bGroupIndices.forEach(idx => {
                        if (panoScoreMap[idx] && panoScoreMap[idx] > bScore) {
                            bScore = panoScoreMap[idx];
                        }
                    });

                    return bScore - aScore; // Descending order (highest score first)
                });
            }

            // Re-append in sorted order
            items.forEach(item => container.appendChild(item));
        }

        function sortOsmLandmarksForPanoGroup(panoIndices) {
            const container = document.getElementById('osm-similarity-list');
            if (!container) return;

            const items = Array.from(container.querySelectorAll('.landmark-item'));

            if (panoIndices === null) {
                // Restore original order using data-original-index
                items.sort((a, b) => {
                    const aIdx = parseInt(a.dataset.originalIndex || '0');
                    const bIdx = parseInt(b.dataset.originalIndex || '0');
                    return aIdx - bIdx;
                });
            } else {
                // Collect best scores from all pano indices in the group
                const scoreMap = {};
                panoIndices.forEach(panoIdx => {
                    const matches = window.currentSimilarityData.pano_to_osm[panoIdx].matches;
                    matches.forEach(match => {
                        if (!scoreMap[match.osm_idx] || match.score > scoreMap[match.osm_idx]) {
                            scoreMap[match.osm_idx] = match.score;
                        }
                    });
                });

                items.sort((a, b) => {
                    const aGroupIndices = a.dataset.osmIndices.split(',').map(s => parseInt(s));
                    const bGroupIndices = b.dataset.osmIndices.split(',').map(s => parseInt(s));

                    let aScore = -1;
                    aGroupIndices.forEach(idx => {
                        if (scoreMap[idx] && scoreMap[idx] > aScore) {
                            aScore = scoreMap[idx];
                        }
                    });

                    let bScore = -1;
                    bGroupIndices.forEach(idx => {
                        if (scoreMap[idx] && scoreMap[idx] > bScore) {
                            bScore = scoreMap[idx];
                        }
                    });

                    return bScore - aScore; // Descending order (highest score first)
                });
            }

            // Re-append in sorted order
            items.forEach(item => container.appendChild(item));
        }

        function loadPanorama(index) {
            fetch('/api/panorama/' + index)
                .then(r => r.json())
                .then(data => {
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

                    data.yaw_angles.forEach(yaw => {
                        const div = document.createElement('div');
                        div.className = 'pinhole-item';

                        let html = `
                            <h3>Yaw ${yaw}°</h3>
                            <img src="/api/image/pinhole/${index}/${yaw}" alt="Yaw ${yaw}">
                            <div class="pinhole-sentences">
                        `;

                        // Add sentences from each source for this yaw
                        data.sources.forEach(source => {
                            const yawData = source.yaw_data.find(yd => yd.yaw === yaw);
                            if (yawData && yawData.sentences.length > 0) {
                                html += `<div class="source-sentences">
                                    <h4>${source.name}</h4>
                                    <ul>`;
                                yawData.sentences.forEach(sentenceObj => {
                                    const desc = sentenceObj.description;
                                    const mode = sentenceObj.mode;
                                    const count = sentenceObj.count || 1;
                                    const countBadge = count > 1 ? `<span style="background:#28a745;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;font-weight:600;">x${count}</span>` : '';

                                    if (mode === 'all') {
                                        // Color-code by landmark_id and show which yaws it appears in
                                        const landmarkId = sentenceObj.landmark_id;
                                        const allYaws = sentenceObj.all_yaws;
                                        const color = getLandmarkColor(landmarkId);
                                        const yawBadges = allYaws.map(y => `<span style="background:#ddd;padding:2px 4px;border-radius:2px;font-size:11px;margin-left:4px;">${y}°</span>`).join('');
                                        html += `<li class="landmark-all-mode" style="border-color:${color};background-color:${color}15;">${desc}${yawBadges}${countBadge}</li>`;
                                    } else {
                                        // Individual mode - plain text
                                        html += `<li class="landmark-individual-mode">${desc}${countBadge}</li>`;
                                    }
                                });
                                html += `</ul></div>`;
                            }
                        });

                        html += `</div>`;
                        div.innerHTML = html;
                        pinholeGrid.appendChild(div);
                    });

                    // Update OSM landmarks
                    const osmContainer = document.getElementById('osm-landmarks-container');
                    if (data.osm_landmarks && data.osm_landmarks.length > 0) {
                        osmContainer.innerHTML = '<ul style="margin:0;padding-left:20px;">' +
                            data.osm_landmarks.map(lm => {
                                const count = lm.count || 1;
                                const countBadge = count > 1 ? `<span style="background:#28a745;color:white;padding:2px 6px;border-radius:2px;font-size:11px;margin-left:4px;font-weight:600;">x${count}</span>` : '';
                                return `<li style="margin-bottom:8px;break-inside:avoid;">${lm.text}${countBadge}</li>`;
                            }).join('') +
                            '</ul>';
                    } else {
                        osmContainer.innerHTML = '<p style="color:#999;font-style:italic;">No OSM landmarks found near this panorama.</p>';
                    }

                    // Update similarity data
                    const similaritySection = document.getElementById('similarity-section');
                    if (data.similarity_data && data.similarity_data.osm_to_pano) {
                        similaritySection.style.display = 'block';

                        // Store similarity data globally for interaction
                        window.currentSimilarityData = data.similarity_data;
                        window.currentSourcesData = data.sources;
                        window.currentOsmLandmarks = data.osm_landmarks_uncollapsed;

                        // Render OSM landmarks - group by description text
                        const osmList = document.getElementById('osm-similarity-list');
                        let osmHtml = '';
                        let osmRenderIdx = 0;

                        // Group OSM landmarks by description text
                        const osmByText = {};
                        Object.keys(data.similarity_data.osm_to_pano).forEach(osmIdxStr => {
                            const osmIdx = parseInt(osmIdxStr);
                            const osmText = data.osm_landmarks_uncollapsed[osmIdx]?.text || 'OSM Landmark';
                            const matches = data.similarity_data.osm_to_pano[osmIdxStr].matches;
                            const bestScore = matches.length > 0 ? matches[0].score : 0;

                            if (!osmByText[osmText]) {
                                osmByText[osmText] = {
                                    indices: [],
                                    bestScore: bestScore,
                                    allMatches: []
                                };
                            }
                            osmByText[osmText].indices.push(osmIdx);
                            osmByText[osmText].allMatches.push(matches);
                            // Track the overall best score across all instances
                            if (bestScore > osmByText[osmText].bestScore) {
                                osmByText[osmText].bestScore = bestScore;
                            }
                        });

                        // Render consolidated OSM landmarks
                        Object.entries(osmByText).forEach(([osmText, data]) => {
                            const bestScore = data.bestScore;
                            const scorePercent = (bestScore * 100).toFixed(0);
                            const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';

                            // Store all osm indices in data attribute (comma-separated)
                            const osmIndices = data.indices.join(',');

                            osmHtml += `<div class="landmark-item" data-osm-indices="${osmIndices}" data-best-score="${bestScore}" data-original-index="${osmRenderIdx}" onclick="selectOsmLandmarkGroup('${osmIndices}')" title="${osmText.replace(/"/g, '&quot;')}">`;
                            osmHtml += `<span class="landmark-match-badge ${scoreClass}">${scorePercent}%</span> `;
                            osmHtml += osmText.length > 80 ? osmText.substring(0, 80) + '...' : osmText;
                            osmHtml += `</div>`;
                            osmRenderIdx++;
                        });
                        osmList.innerHTML = osmHtml;

                        // Render panorama landmarks - group by description text
                        const panoList = document.getElementById('pano-similarity-list');
                        let panoHtml = '';
                        let panoRenderIdx = 0;

                        // Group panorama landmarks by description text
                        const panoByText = {};
                        Object.keys(data.similarity_data.pano_to_osm).forEach(panoIdxStr => {
                            const panoIdx = parseInt(panoIdxStr);
                            const panoData = data.similarity_data.pano_to_osm[panoIdxStr];
                            const source = data.sources.find(s => s.name === panoData.source);
                            if (source) {
                                const yawData = source.yaw_data.find(yd => yd.yaw === panoData.yaw);
                                if (yawData && yawData.sentences[panoData.landmark_idx]) {
                                    const panoText = yawData.sentences[panoData.landmark_idx].description;
                                    const matches = panoData.matches;
                                    const bestScore = matches.length > 0 ? matches[0].score : 0;

                                    if (!panoByText[panoText]) {
                                        panoByText[panoText] = {
                                            indices: [],
                                            yaws: [],
                                            bestScore: bestScore,
                                            allMatches: []
                                        };
                                    }
                                    panoByText[panoText].indices.push(panoIdx);
                                    panoByText[panoText].yaws.push(panoData.yaw);
                                    panoByText[panoText].allMatches.push(matches);
                                    // Track the overall best score across all yaws
                                    if (bestScore > panoByText[panoText].bestScore) {
                                        panoByText[panoText].bestScore = bestScore;
                                    }
                                }
                            }
                        });

                        // Render consolidated panorama landmarks
                        Object.entries(panoByText).forEach(([panoText, data]) => {
                            const bestScore = data.bestScore;
                            const scorePercent = (bestScore * 100).toFixed(0);
                            const scoreClass = bestScore > 0.8 ? 'score-high' : bestScore > 0.6 ? 'score-medium' : 'score-low';
                            const yawBadges = data.yaws.map(y => `${y}°`).join(', ');

                            // Store all pano indices in data attribute (comma-separated)
                            const panoIndices = data.indices.join(',');

                            panoHtml += `<div class="landmark-item" data-pano-indices="${panoIndices}" data-best-score="${bestScore}" data-original-index="${panoRenderIdx}" onclick="selectPanoLandmarkGroup('${panoIndices}')" title="${panoText.replace(/"/g, '&quot;')}">`;
                            panoHtml += `<span class="landmark-match-badge ${scoreClass}">${scorePercent}%</span> `;
                            panoHtml += `<span style="color:#666;font-size:11px;">[${yawBadges}]</span> `;
                            panoHtml += panoText.length > 70 ? panoText.substring(0, 70) + '...' : panoText;
                            panoHtml += `</div>`;
                            panoRenderIdx++;
                        });
                        panoList.innerHTML = panoHtml;
                    } else {
                        similaritySection.style.display = 'none';
                    }

                    // Load global matches for this panorama (non-blocking)
                    loadGlobalBestMatchesForPanorama(index);
                });
        }

        function navigate(delta) {
            let newIndex = currentIndex + delta;
            if (newIndex < 0) newIndex = totalPanoramas - 1;
            if (newIndex >= totalPanoramas) newIndex = 0;
            loadPanorama(newIndex);
        }

        // Load and display global best matches for current panorama
        function loadGlobalBestMatchesForPanorama(index) {
            // Cancel pending timeout
            if (globalMatchesTimeout) {
                clearTimeout(globalMatchesTimeout);
            }

            // Cancel in-flight request
            if (currentGlobalMatchesController) {
                currentGlobalMatchesController.abort();
            }

            // Show loading state immediately
            const globalSection = document.getElementById('global-matches-section');
            globalSection.style.display = 'block';
            const osmToPanoList = document.getElementById('global-osm-to-pano-list');
            const panoToOsmList = document.getElementById('global-pano-to-osm-list');
            osmToPanoList.innerHTML = '<p style="color:#666;padding:20px;">Computing global matches...</p>';
            panoToOsmList.innerHTML = '<p style="color:#666;padding:20px;">Computing global matches...</p>';

            // Wait 200ms before actually fetching (debounce)
            globalMatchesTimeout = setTimeout(() => {
                currentGlobalMatchesController = new AbortController();

                fetch(`/api/global_best_matches_for_panorama/${index}`, {
                    signal: currentGlobalMatchesController.signal
                })
                    .then(r => r.json())
                    .then(data => {
                        if (data.error) {
                            osmToPanoList.innerHTML = `<p style="color:#999;padding:20px;">${data.error}</p>`;
                            panoToOsmList.innerHTML = `<p style="color:#999;padding:20px;">${data.error}</p>`;
                            currentGlobalMatchesController = null;
                            return;
                        }

                        // Render OSM->Pano matches
                        let osmToPanoHtml = '<div style="font-size:13px;">';
                        data.osm_to_pano.slice(0, 20).forEach((match, index) => {
                            const scorePercent = (match.score * 100).toFixed(0);
                            const scoreClass = match.score > 0.8 ? 'score-high' : match.score > 0.6 ? 'score-medium' : 'score-low';

                            osmToPanoHtml += `<div style="padding:8px;margin:4px 0;border:1px solid #ddd;border-radius:4px;cursor:pointer;background:#fafafa;" onclick="loadPanorama(${match.pano_index})" title="Click to view this panorama">`;
                            osmToPanoHtml += `<div style="margin-bottom:4px;"><span class="landmark-match-badge ${scoreClass}">#${index + 1}: ${scorePercent}%</span></div>`;
                            osmToPanoHtml += `<div style="margin-bottom:4px;color:#0066cc;font-weight:600;">OSM: ${match.osm_text.length > 100 ? match.osm_text.substring(0, 100) + '...' : match.osm_text}</div>`;
                            osmToPanoHtml += `<div style="margin-bottom:4px;color:#333;">Pano: ${match.pano_text.length > 100 ? match.pano_text.substring(0, 100) + '...' : match.pano_text}</div>`;
                            osmToPanoHtml += `<div style="font-size:11px;color:#666;">📍 Panorama #${match.pano_index + 1} | ${match.source} | ${match.yaw}°</div>`;
                            osmToPanoHtml += `</div>`;
                        });
                        osmToPanoHtml += '</div>';
                        osmToPanoList.innerHTML = osmToPanoHtml;

                        // Render Pano->OSM matches
                        let panoToOsmHtml = '<div style="font-size:13px;">';
                        data.pano_to_osm.slice(0, 20).forEach((match, index) => {
                            const scorePercent = (match.score * 100).toFixed(0);
                            const scoreClass = match.score > 0.8 ? 'score-high' : match.score > 0.6 ? 'score-medium' : 'score-low';

                            panoToOsmHtml += `<div style="padding:8px;margin:4px 0;border:1px solid #ddd;border-radius:4px;cursor:pointer;background:#fafafa;" onclick="loadPanorama(${match.pano_index})" title="Click to view this panorama">`;
                            panoToOsmHtml += `<div style="margin-bottom:4px;"><span class="landmark-match-badge ${scoreClass}">#${index + 1}: ${scorePercent}%</span></div>`;
                            panoToOsmHtml += `<div style="margin-bottom:4px;color:#cc6600;font-weight:600;">Pano: ${match.pano_text.length > 100 ? match.pano_text.substring(0, 100) + '...' : match.pano_text}</div>`;
                            panoToOsmHtml += `<div style="margin-bottom:4px;color:#333;">OSM: ${match.osm_text.length > 100 ? match.osm_text.substring(0, 100) + '...' : match.osm_text}</div>`;
                            panoToOsmHtml += `<div style="font-size:11px;color:#666;">📍 Panorama #${match.pano_index + 1} | ${match.source} | ${match.yaw}°</div>`;
                            panoToOsmHtml += `</div>`;
                        });
                        panoToOsmHtml += '</div>';
                        panoToOsmList.innerHTML = panoToOsmHtml;

                        currentGlobalMatchesController = null;
                    })
                    .catch(err => {
                        if (err.name === 'AbortError') {
                            console.log('Global matches request was cancelled');
                        } else {
                            console.error('Error loading global best matches:', err);
                            osmToPanoList.innerHTML = '<p style="color:red;padding:20px;">Failed to load</p>';
                            panoToOsmList.innerHTML = '<p style="color:red;padding:20px;">Failed to load</p>';
                        }
                        currentGlobalMatchesController = null;
                    });
            }, 200); // 200ms debounce
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
        });

        // Load first panorama on page load
        loadPanorama(0);
        // Global matches will be loaded by loadPanorama -> loadGlobalBestMatchesForPanorama
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


def custom_id_from_props(props):
    """Generate custom_id from landmark properties (copied from semantic_landmark_extractor.py)."""
    json_props = json.dumps(dict(props), sort_keys=True)
    custom_id = base64.b64encode(hashlib.sha256(
        json_props.encode('utf-8')).digest()).decode('utf-8')
    return custom_id


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


def load_osm_embeddings_for_landmarks(embedding_dir, landmark_custom_ids):
    """Load OSM landmark embeddings only for specific custom_ids (efficient)."""
    embedding_path = Path(embedding_dir)
    if not embedding_path.exists():
        print(f"Warning: OSM embedding directory not found: {embedding_dir}")
        return {}

    print(f"  Loading embeddings for {len(landmark_custom_ids)} landmarks...")
    needed_ids = set(landmark_custom_ids)
    embeddings = {}

    # Read JSONL files and only keep entries we need
    files_processed = 0
    for jsonl_file in embedding_path.glob("*"):
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
                            if 'data' in body and len(body['data']) > 0:
                                embedding = body['data'][0]['embedding']
                                # Convert to torch tensor on GPU
                                embeddings[custom_id] = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
                                needed_ids.remove(custom_id)

                                # Early exit if we found everything
                                if not needed_ids:
                                    print(f"  Found all embeddings after {files_processed} files")
                                    return embeddings
                except Exception as e:
                    pass  # Skip malformed entries

    print(f"  Loaded {len(embeddings)} embeddings (missing {len(needed_ids)})")
    return embeddings


def load_panorama_embeddings(embedding_dirs):
    """Load panorama landmark embeddings from multiple directories."""
    embeddings = {}

    print(f"  Loading panorama embeddings from {len(embedding_dirs)} directories...")
    for embedding_dir in embedding_dirs:
        embedding_path = Path(embedding_dir)
        if not embedding_path.exists():
            print(f"Warning: Panorama embedding directory not found: {embedding_dir}")
            continue

        for jsonl_file in embedding_path.glob("*"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        custom_id = entry['custom_id']

                        if 'response' in entry and 'body' in entry['response']:
                            body = entry['response']['body']
                            if 'data' in body and len(body['data']) > 0:
                                embedding = body['data'][0]['embedding']
                                # Convert to torch tensor on GPU
                                embeddings[custom_id] = torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
                    except Exception as e:
                        pass  # Skip malformed entries

    print(f"  Loaded {len(embeddings)} panorama embeddings")
    return embeddings


def compute_embedding_similarities_gpu(osm_embeddings_dict, osm_custom_ids, pano_embeddings_dict, pano_custom_ids, top_k=3):
    """
    Compute cosine similarity between OSM and panorama embeddings using PyTorch on GPU.

    Args:
        osm_embeddings_dict: Dict mapping custom_id -> embedding tensor
        osm_custom_ids: List of OSM custom_ids to compare
        pano_embeddings_dict: Dict mapping custom_id -> embedding tensor
        pano_custom_ids: List of panorama custom_ids to compare
        top_k: Number of top matches to return for each landmark

    Returns:
        osm_to_pano_matches: Dict mapping osm_idx -> [(pano_idx, similarity_score), ...]
        pano_to_osm_matches: Dict mapping pano_idx -> [(osm_idx, similarity_score), ...]
    """
    if not osm_custom_ids or not pano_custom_ids:
        return {}, {}

    # Filter to only embeddings that exist
    valid_osm_ids = [cid for cid in osm_custom_ids if cid in osm_embeddings_dict]
    valid_pano_ids = [cid for cid in pano_custom_ids if cid in pano_embeddings_dict]

    if not valid_osm_ids or not valid_pano_ids:
        return {}, {}

    # Stack embeddings into matrices
    osm_embeddings = torch.stack([osm_embeddings_dict[cid] for cid in valid_osm_ids])  # [N_osm, dim]
    pano_embeddings = torch.stack([pano_embeddings_dict[cid] for cid in valid_pano_ids])  # [N_pano, dim]

    with torch.no_grad():
        # L2 normalize embeddings
        osm_norm = F.normalize(osm_embeddings, dim=1)
        pano_norm = F.normalize(pano_embeddings, dim=1)

        # Compute similarity matrix: [N_osm, N_pano]
        similarity_matrix = osm_norm @ pano_norm.T

        # Find top-k matches for each OSM landmark
        osm_to_pano_matches = {}
        if top_k > 0:
            topk_vals, topk_idxs = torch.topk(similarity_matrix, min(top_k, similarity_matrix.shape[1]), dim=1)
            for osm_idx in range(len(valid_osm_ids)):
                matches = []
                for k in range(topk_idxs.shape[1]):
                    pano_idx = topk_idxs[osm_idx, k].item()
                    score = topk_vals[osm_idx, k].item()
                    matches.append((pano_idx, float(score)))
                osm_to_pano_matches[osm_idx] = matches

        # Find top-k matches for each panorama landmark
        pano_to_osm_matches = {}
        if top_k > 0:
            topk_vals, topk_idxs = torch.topk(similarity_matrix.T, min(top_k, similarity_matrix.shape[0]), dim=1)
            for pano_idx in range(len(valid_pano_ids)):
                matches = []
                for k in range(topk_idxs.shape[1]):
                    osm_idx = topk_idxs[pano_idx, k].item()
                    score = topk_vals[pano_idx, k].item()
                    matches.append((osm_idx, float(score)))
                pano_to_osm_matches[pano_idx] = matches

    return osm_to_pano_matches, pano_to_osm_matches


def compute_panorama_to_landmarks(panorama_dir, landmarks_geojson_path, osm_sentences_dir, osm_embeddings_dir=None, zoom_level=20):
    """
    Compute which OSM landmarks are near each panorama using VigorDataset approach.
    Uses caching to avoid recomputing if inputs haven't changed.

    Returns:
        pano_to_landmarks: dict mapping panorama_id -> list of landmark sentences
        pano_to_custom_ids: dict mapping panorama_id -> list of custom_ids
        osm_embeddings: dict mapping custom_id -> torch.Tensor
    """
    import os
    import pandas as pd
    import shapely
    from common.gps import web_mercator

    # Compute cache key based on input file mtimes
    geojson_path = Path(landmarks_geojson_path)
    sentences_path = Path(osm_sentences_dir)
    pano_path = Path(panorama_dir)

    cache_key_parts = [
        f"geojson:{geojson_path.stat().st_mtime}",
        f"sentences:{sentences_path.stat().st_mtime}",
        f"panoramas:{pano_path.stat().st_mtime}",
        f"zoom:{zoom_level}"
    ]

    # Include embeddings dir in cache key (unconditional)
    if osm_embeddings_dir:
        embeddings_path = Path(osm_embeddings_dir)
        if embeddings_path.exists():
            cache_key_parts.append(f"embeddings:{embeddings_path.stat().st_mtime}")
        else:
            cache_key_parts.append("embeddings:None")
    else:
        cache_key_parts.append("embeddings:None")

    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/osm_panorama_cache_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading OSM landmarks from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Handle different cache formats
            if isinstance(cached_data, tuple):
                if len(cached_data) == 3:
                    # New format with embeddings
                    print(f"  Loaded cached data for {len(cached_data[0])} panoramas (with embeddings)")
                    return cached_data
                elif len(cached_data) == 2:
                    # Old format without embeddings
                    print(f"  Loaded cached data (old format) for {len(cached_data[0])} panoramas")
                    return cached_data[0], cached_data[1], {}
            else:
                # Very old cache format (just dict)
                print(f"  Loaded cached data (very old format) for {len(cached_data)} panoramas")
                return cached_data, {}, {}
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    print("Computing OSM landmarks (no cache found)...")
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

    print("Computing panorama→landmark associations...")
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

    # First, identify which landmarks are near panoramas
    print("Identifying nearby landmarks...")
    start = time.time()
    landmark_custom_ids_needed = set()
    pano_to_landmark_indices = {}

    for pano_idx, landmark_idx in results.T:
        pano_id = pano_df.iloc[pano_idx]['id']
        landmark = landmarks_df.iloc[landmark_idx]

        # Compute custom_id
        props = prune_landmark(landmark.dropna().to_dict())
        custom_id = custom_id_from_props(props)
        landmark_custom_ids_needed.add(custom_id)

        if pano_id not in pano_to_landmark_indices:
            pano_to_landmark_indices[pano_id] = []
        pano_to_landmark_indices[pano_id].append((landmark_idx, custom_id))

    print(f"  Found {len(landmark_custom_ids_needed)} unique landmarks in {time.time()-start:.1f}s")

    # Load only the sentences we need
    osm_sentences = load_osm_sentences_for_landmarks(osm_sentences_dir, landmark_custom_ids_needed)
    print(f"  Loaded {len(osm_sentences)} sentences")

    # Load embeddings if directory provided
    osm_embeddings = {}
    if osm_embeddings_dir:
        print(f"  Loading OSM embeddings...")
        start_embed = time.time()
        osm_embeddings = load_osm_embeddings_for_landmarks(osm_embeddings_dir, landmark_custom_ids_needed)
        print(f"  Loaded {len(osm_embeddings)} embeddings in {time.time()-start_embed:.1f}s")

    # Build final result dictionary
    print("Building result dictionary...")
    start = time.time()
    pano_to_landmarks = {}
    pano_to_custom_ids = {}

    for pano_id, landmark_list in pano_to_landmark_indices.items():
        for landmark_idx, custom_id in landmark_list:
            if custom_id in osm_sentences:
                if pano_id not in pano_to_landmarks:
                    pano_to_landmarks[pano_id] = []
                    pano_to_custom_ids[pano_id] = []
                pano_to_landmarks[pano_id].append(osm_sentences[custom_id])
                pano_to_custom_ids[pano_id].append(custom_id)

    print(f"  Built dictionary in {time.time()-start:.1f}s")
    print(f"Found OSM landmarks for {len(pano_to_landmarks)} panoramas")

    # Save to cache (including embeddings)
    print(f"Saving to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((pano_to_landmarks, pano_to_custom_ids, osm_embeddings), f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return pano_to_landmarks, pano_to_custom_ids, osm_embeddings


def load_sentence_data(sentence_dirs, panorama_embeddings_dirs=None):
    """
    Load and parse JSONL sentence files from multiple directories.

    Returns:
        sources: Dict mapping source_name -> pano_id -> yaw -> list of landmark dicts
        pano_custom_ids: Dict mapping (source_name, pano_id, yaw, landmark_idx) -> custom_id
        pano_embeddings: Dict mapping custom_id -> torch.Tensor
    """
    # Compute cache key based on input directories
    cache_key_parts = []
    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if sentence_path.exists():
            cache_key_parts.append(f"sentences:{sentence_path}:{sentence_path.stat().st_mtime}")

    # Include embedding directories in cache key (unconditional)
    if panorama_embeddings_dirs:
        for embed_dir in panorama_embeddings_dirs:
            embed_path = Path(embed_dir)
            if embed_path.exists():
                cache_key_parts.append(f"embeddings:{embed_path}:{embed_path.stat().st_mtime}")
            else:
                cache_key_parts.append(f"embeddings:{embed_path}:None")
    else:
        cache_key_parts.append("embeddings:None")

    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/pano_sentence_cache_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading panorama sentence data from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # Handle different cache formats
            if isinstance(cached_data, tuple):
                if len(cached_data) == 3:
                    # New format with embeddings
                    print(f"  Loaded cached data (with embeddings)")
                    return cached_data
                elif len(cached_data) == 2:
                    # Old format without embeddings
                    print(f"  Loaded cached data (old format)")
                    return cached_data[0], cached_data[1], {}
            else:
                # Very old format
                print(f"  Loaded cached data (very old format)")
                return cached_data, {}, {}
        except Exception as e:
            print(f"  Cache load failed: {e}, recomputing...")

    sources = {}
    pano_custom_ids = {}

    print(f"  Processing {len(sentence_dirs)} sentence directories...")
    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if not sentence_path.exists():
            print(f"Warning: Sentence directory not found: {sentence_dir}")
            continue

        source_name = sentence_path.name
        source_data = {}

        # Load all files in this directory (JSONL format, may or may not have .jsonl extension)
        for jsonl_file in sentence_path.glob('*'):
            if not jsonl_file.is_file():
                continue
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        custom_id = entry['custom_id']

                        # Detect mode based on custom_id format
                        if '_yaw_' in custom_id:
                            # Individual mode: {pano_id}_yaw_{angle}
                            pano_id, yaw_part = custom_id.rsplit('_yaw_', 1)
                            yaw = int(yaw_part)

                            # Extract just the base ID (first part before comma if present)
                            base_id = pano_id.split(',')[0]

                            # Extract landmarks from response
                            if 'response' in entry and 'body' in entry['response']:
                                body = entry['response']['body']
                                if 'choices' in body and len(body['choices']) > 0:
                                    content = body['choices'][0]['message']['content']
                                    try:
                                        landmarks_data = json.loads(content)
                                        landmarks = landmarks_data.get('landmarks', [])

                                        if base_id not in source_data:
                                            source_data[base_id] = {}
                                        if yaw not in source_data[base_id]:
                                            source_data[base_id][yaw] = []

                                        # For individual mode, store sentences and track custom_ids
                                        for lm_idx, lm in enumerate(landmarks):
                                            landmark_idx = len(source_data[base_id][yaw])
                                            source_data[base_id][yaw].append({
                                                'description': lm['description'],
                                                'mode': 'individual'
                                            })
                                            # Store custom_id for this landmark's embedding
                                            # Format: {pano_id}_yaw_{angle}__landmark_{idx}
                                            landmark_custom_id = f"{custom_id}__landmark_{lm_idx}"
                                            pano_custom_ids[(source_name, base_id, yaw, landmark_idx)] = landmark_custom_id

                                    except json.JSONDecodeError:
                                        print(f"Warning: Failed to parse JSON content for {custom_id}")
                        else:
                            # All mode: custom_id is the full path to pinhole directory
                            path_str = custom_id
                            base_id = Path(path_str).name.split(',')[0]

                            # Extract landmarks from response
                            if 'response' in entry and 'body' in entry['response']:
                                body = entry['response']['body']
                                if 'choices' in body and len(body['choices']) > 0:
                                    content = body['choices'][0]['message']['content']
                                    try:
                                        landmarks_data = json.loads(content)
                                        landmarks = landmarks_data.get('landmarks', [])

                                        if base_id not in source_data:
                                            source_data[base_id] = {}

                                        # Process landmarks with yaw_angles
                                        for lm_idx, lm in enumerate(landmarks):
                                            description = lm['description']
                                            yaw_angles = lm.get('yaw_angles', [])

                                            # Add this landmark to each yaw it appears in
                                            for yaw in yaw_angles:
                                                if yaw not in source_data[base_id]:
                                                    source_data[base_id][yaw] = []
                                                landmark_idx = len(source_data[base_id][yaw])
                                                source_data[base_id][yaw].append({
                                                    'description': description,
                                                    'landmark_id': lm_idx,  # Use index for grouping
                                                    'all_yaws': yaw_angles,
                                                    'mode': 'all'
                                                })
                                                # Store custom_id for this landmark's embedding
                                                # Format: {pano_path}__landmark_{idx}
                                                landmark_custom_id = f"{custom_id}__landmark_{lm_idx}"
                                                pano_custom_ids[(source_name, base_id, yaw, landmark_idx)] = landmark_custom_id

                                    except json.JSONDecodeError:
                                        print(f"Warning: Failed to parse JSON content for {custom_id}")
                    except Exception as e:
                        print(f"Warning: Error parsing line in {jsonl_file}: {e}")

        if source_data:
            sources[source_name] = source_data

    # Load panorama embeddings if directories provided
    pano_embeddings = {}
    if panorama_embeddings_dirs:
        print(f"  Loading panorama embeddings...")
        start_embed = time.time()
        pano_embeddings = load_panorama_embeddings(panorama_embeddings_dirs)
        print(f"  Loaded {len(pano_embeddings)} panorama embeddings in {time.time()-start_embed:.1f}s")

    # Save to cache
    print(f"Saving panorama data to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((sources, pano_custom_ids, pano_embeddings), f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return sources, pano_custom_ids, pano_embeddings


def find_common_panoramas(panorama_dir, pinhole_dir, sentence_sources):
    """Find panoramas that exist in all required locations."""
    panorama_path = Path(panorama_dir)
    pinhole_path = Path(pinhole_dir)

    # Get panorama IDs from panorama directory
    pano_files = {parse_vigor_filename(f.name): f for f in panorama_path.glob('*.jpg')}

    # Get panorama IDs from pinhole directory
    pinhole_dirs = {d.name.split(',')[0]: d for d in pinhole_path.iterdir() if d.is_dir()}

    # Find intersection
    common_ids = set(pano_files.keys()) & set(pinhole_dirs.keys())

    # Filter by sentence availability (ALL sources must have data for this panorama)
    if sentence_sources:
        # Start with IDs from first source
        ids_in_all_sources = None
        for source_data in sentence_sources.values():
            source_ids = set(source_data.keys())
            if ids_in_all_sources is None:
                ids_in_all_sources = source_ids
            else:
                ids_in_all_sources &= source_ids

        if ids_in_all_sources:
            common_ids &= ids_in_all_sources
        else:
            common_ids = set()  # No panoramas in all sources

    # Build panorama data
    panorama_data = []
    for pano_id in sorted(common_ids):
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


@app.route('/api/panorama/<int:index>')
def get_panorama_data(index):
    global PANORAMA_DATA, SENTENCE_SOURCES, OSM_LANDMARKS, OSM_EMBEDDINGS, PANORAMA_EMBEDDINGS, OSM_CUSTOM_IDS, PANO_CUSTOM_IDS

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid index'}), 404

    pano = PANORAMA_DATA[index]

    # Get sentence data for this panorama
    sources_data = []
    for source_name, source_data in SENTENCE_SOURCES.items():
        if pano['id'] in source_data:
            yaw_data = []
            for yaw in pano['yaw_angles']:
                if yaw in source_data[pano['id']]:
                    # Collapse duplicates
                    collapsed = collapse_duplicate_sentences(source_data[pano['id']][yaw])
                    yaw_data.append({
                        'yaw': yaw,
                        'sentences': collapsed
                    })
            if yaw_data:
                sources_data.append({
                    'name': source_name,
                    'yaw_data': yaw_data
                })

    # Get OSM landmark sentences for this panorama
    osm_sentences_raw = OSM_LANDMARKS.get(pano['id'], [])
    osm_sentences = collapse_string_list(osm_sentences_raw)  # Collapsed for display
    osm_sentences_uncollapsed = [{'text': s} for s in osm_sentences_raw]  # Uncollapsed for similarity indexing

    # Compute embedding similarities if embeddings are available
    similarity_data = None

    # Debug: Log embedding availability
    print(f"Debug: OSM_EMBEDDINGS: {len(OSM_EMBEDDINGS) if OSM_EMBEDDINGS else 0}, "
          f"PANORAMA_EMBEDDINGS: {len(PANORAMA_EMBEDDINGS) if PANORAMA_EMBEDDINGS else 0}, "
          f"pano_id in OSM_CUSTOM_IDS: {pano['id'] in OSM_CUSTOM_IDS}")

    if OSM_EMBEDDINGS and PANORAMA_EMBEDDINGS and pano['id'] in OSM_CUSTOM_IDS:
        try:
            # Get OSM custom IDs for this panorama
            osm_custom_ids = OSM_CUSTOM_IDS[pano['id']]
            print(f"Debug: Found {len(osm_custom_ids)} OSM custom_ids for panorama {pano['id']}")

            # Get panorama custom IDs for this panorama (collect from all sources and yaws)
            pano_custom_ids_list = []
            for source_name, source_data in SENTENCE_SOURCES.items():
                if pano['id'] in source_data:
                    for yaw in pano['yaw_angles']:
                        if yaw in source_data[pano['id']]:
                            num_landmarks = len(source_data[pano['id']][yaw])
                            for lm_idx in range(num_landmarks):
                                key = (source_name, pano['id'], yaw, lm_idx)
                                if key in PANO_CUSTOM_IDS:
                                    pano_custom_ids_list.append((key, PANO_CUSTOM_IDS[key]))

            print(f"Debug: Found {len(pano_custom_ids_list)} panorama custom_ids")

            # Compute similarities
            if osm_custom_ids and pano_custom_ids_list:
                pano_cids_only = [cid for _, cid in pano_custom_ids_list]
                osm_to_pano, pano_to_osm = compute_embedding_similarities_gpu(
                    OSM_EMBEDDINGS, osm_custom_ids,
                    PANORAMA_EMBEDDINGS, pano_cids_only,
                    top_k=20
                )

                # Format the results for JSON
                similarity_data = {
                    'osm_to_pano': {},
                    'pano_to_osm': {}
                }

                # Map OSM indices to OSM custom_ids (for frontend lookup)
                for osm_idx, matches in osm_to_pano.items():
                    osm_cid = osm_custom_ids[osm_idx]
                    similarity_data['osm_to_pano'][osm_idx] = {
                        'custom_id': osm_cid,
                        'matches': [{'pano_idx': p_idx, 'score': score} for p_idx, score in matches]
                    }

                # Map panorama indices back to (source, yaw, landmark_idx)
                for pano_idx, matches in pano_to_osm.items():
                    key, pano_cid = pano_custom_ids_list[pano_idx]
                    source_name, _, yaw, lm_idx = key
                    similarity_data['pano_to_osm'][pano_idx] = {
                        'source': source_name,
                        'yaw': yaw,
                        'landmark_idx': lm_idx,
                        'custom_id': pano_cid,
                        'matches': [{'osm_idx': o_idx, 'score': score} for o_idx, score in matches]
                    }

        except Exception as e:
            print(f"Error computing similarities: {e}")
            similarity_data = {'error': str(e)}

    # Extract coordinates from panorama filename
    lat, lon = None, None
    pano_file = pano['panorama_file']
    parts = pano_file.stem.split(',')
    if len(parts) >= 3:
        try:
            lat = float(parts[1])
            lon = float(parts[2])
        except ValueError:
            pass

    return jsonify({
        'name': pano['id'],
        'total': len(PANORAMA_DATA),
        'yaw_angles': pano['yaw_angles'],
        'sources': sources_data,
        'osm_landmarks': osm_sentences,
        'osm_landmarks_uncollapsed': osm_sentences_uncollapsed,  # For similarity indexing
        'similarity_data': similarity_data,
        'lat': lat,
        'lon': lon
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


@app.route('/api/global_best_matches_for_panorama/<int:index>')
def get_global_best_matches_for_panorama(index):
    """
    Compute global best matches for a specific panorama on-demand.

    This compares the current panorama's landmarks against ALL other landmarks:
    - OSM → Pano: This pano's OSM landmarks vs ALL pano landmarks
    - Pano → OSM: This pano's pano landmarks vs ALL OSM landmarks
    """
    global PANORAMA_DATA, OSM_LANDMARKS, OSM_EMBEDDINGS, PANORAMA_EMBEDDINGS, OSM_CUSTOM_IDS, PANO_CUSTOM_IDS, SENTENCE_SOURCES

    if index < 0 or index >= len(PANORAMA_DATA):
        return jsonify({'error': 'Invalid panorama index'}), 404

    pano = PANORAMA_DATA[index]
    pano_id = pano['id']

    # Check if we have embeddings for this panorama
    if not OSM_EMBEDDINGS or not PANORAMA_EMBEDDINGS:
        return jsonify({'error': 'Embeddings not available'}), 404

    if pano_id not in OSM_CUSTOM_IDS:
        return jsonify({'error': 'No OSM data for this panorama'}), 404

    # Get OSM custom IDs for this panorama
    osm_custom_ids_for_pano = OSM_CUSTOM_IDS[pano_id]

    # Get pano custom IDs for this panorama
    pano_custom_ids_for_pano = []
    for key, cid in PANO_CUSTOM_IDS.items():
        source_name, pid, yaw, lm_idx = key
        if pid == pano_id and cid in PANORAMA_EMBEDDINGS:
            pano_custom_ids_for_pano.append((key, cid))

    # Compute global matches
    result = compute_global_matches_for_panorama(
        pano_id, index, osm_custom_ids_for_pano, pano_custom_ids_for_pano,
        OSM_EMBEDDINGS, PANORAMA_EMBEDDINGS, PANO_CUSTOM_IDS,
        SENTENCE_SOURCES, PANORAMA_DATA, top_k=20
    )

    return jsonify(result)


def compute_global_matches_for_panorama(pano_id, pano_index, osm_custom_ids_for_pano,
                                         pano_custom_ids_for_pano, osm_embeddings,
                                         panorama_embeddings, all_pano_custom_ids,
                                         sentence_sources, panorama_data, top_k=20):
    """
    Compute global best matches for ONE panorama's landmarks.

    - OSM → Pano: Compare this pano's OSM landmarks against ALL panorama embeddings
    - Pano → OSM: Compare this pano's pano landmarks against ALL OSM embeddings

    This is much faster than computing for all panoramas since we only compute
    for ~10 OSM landmarks and ~50 pano landmarks.
    """
    global CACHED_PANO_EMBEDDINGS_LIST, CACHED_OSM_EMBEDDINGS_LIST

    # Build cached list of all pano embeddings with metadata (only once)
    if CACHED_PANO_EMBEDDINGS_LIST is None:
        print("Building cache of all panorama embeddings (one-time cost)...")
        all_pano_embeddings_list = []
        all_pano_metadata = []

        for key, cid in all_pano_custom_ids.items():
            if cid in panorama_embeddings:
                source_name, pid, yaw, lm_idx = key

                # Get text for this panorama landmark
                source_data = sentence_sources.get(source_name, {})
                if pid in source_data and yaw in source_data[pid]:
                    sentences = source_data[pid][yaw]
                    if lm_idx < len(sentences):
                        pano_text = sentences[lm_idx].get('description', 'Unknown')

                        # Find panorama index
                        pano_idx = next((i for i, p in enumerate(panorama_data) if p['id'] == pid), -1)

                        all_pano_embeddings_list.append(panorama_embeddings[cid])
                        all_pano_metadata.append({
                            'text': pano_text,
                            'pano_id': pid,
                            'pano_index': pano_idx,
                            'yaw': yaw,
                            'source': source_name
                        })

        CACHED_PANO_EMBEDDINGS_LIST = (all_pano_embeddings_list, all_pano_metadata)
        print(f"Cached {len(all_pano_embeddings_list)} panorama embeddings")
    else:
        all_pano_embeddings_list, all_pano_metadata = CACHED_PANO_EMBEDDINGS_LIST

    # OSM → Pano: Compare this pano's OSM landmarks against ALL pano embeddings
    osm_to_pano_results = []
    if osm_custom_ids_for_pano and all_pano_embeddings_list:
        osm_embeddings_list = [osm_embeddings[cid] for cid in osm_custom_ids_for_pano if cid in osm_embeddings]
        osm_texts = [OSM_LANDMARKS[pano_id][i] for i in range(len(osm_custom_ids_for_pano))]

        if osm_embeddings_list:
            osm_matrix = torch.stack(osm_embeddings_list)
            pano_matrix = torch.stack(all_pano_embeddings_list)

            with torch.no_grad():
                osm_norm = F.normalize(osm_matrix, dim=1)
                pano_norm = F.normalize(pano_matrix, dim=1)
                similarity_matrix = osm_norm @ pano_norm.T  # (N_osm, N_all_pano)

                # Get top-k for each OSM landmark
                topk_vals, topk_idxs = torch.topk(similarity_matrix, min(top_k, similarity_matrix.shape[1]), dim=1)

                for osm_idx in range(len(osm_embeddings_list)):
                    for k in range(topk_idxs.shape[1]):
                        pano_idx = topk_idxs[osm_idx, k].item()
                        score = topk_vals[osm_idx, k].item()
                        osm_to_pano_results.append({
                            'osm_text': osm_texts[osm_idx],
                            'pano_text': all_pano_metadata[pano_idx]['text'],
                            'score': float(score),
                            'pano_id': all_pano_metadata[pano_idx]['pano_id'],
                            'pano_index': all_pano_metadata[pano_idx]['pano_index'],
                            'yaw': all_pano_metadata[pano_idx]['yaw'],
                            'source': all_pano_metadata[pano_idx]['source']
                        })

    # Pano → OSM: Compare this pano's pano landmarks against ALL OSM embeddings
    pano_to_osm_results = []
    if pano_custom_ids_for_pano and osm_embeddings:
        # Get embeddings for this panorama's landmarks
        pano_embeddings_list = []
        pano_texts = []

        for key, cid in pano_custom_ids_for_pano:
            if cid in panorama_embeddings:
                source_name, pid, yaw, lm_idx = key
                source_data = sentence_sources.get(source_name, {})
                if pid in source_data and yaw in source_data[pid]:
                    sentences = source_data[pid][yaw]
                    if lm_idx < len(sentences):
                        pano_embeddings_list.append(panorama_embeddings[cid])
                        pano_texts.append(sentences[lm_idx].get('description', 'Unknown'))

        # Build cached list of all OSM embeddings with metadata (only once)
        if CACHED_OSM_EMBEDDINGS_LIST is None:
            print("Building cache of all OSM embeddings (one-time cost)...")
            all_osm_embeddings_list = []
            all_osm_metadata = []

            for pano_id_key, osm_custom_ids_list in OSM_CUSTOM_IDS.items():
                osm_landmarks_list = OSM_LANDMARKS[pano_id_key]
                for i, cid in enumerate(osm_custom_ids_list):
                    if cid in osm_embeddings:
                        all_osm_embeddings_list.append(osm_embeddings[cid])
                        all_osm_metadata.append({
                            'text': osm_landmarks_list[i] if i < len(osm_landmarks_list) else 'Unknown'
                        })

            CACHED_OSM_EMBEDDINGS_LIST = (all_osm_embeddings_list, all_osm_metadata)
            print(f"Cached {len(all_osm_embeddings_list)} OSM embeddings")
        else:
            all_osm_embeddings_list, all_osm_metadata = CACHED_OSM_EMBEDDINGS_LIST

        if pano_embeddings_list and all_osm_embeddings_list:
            pano_matrix = torch.stack(pano_embeddings_list)
            osm_matrix = torch.stack(all_osm_embeddings_list)

            with torch.no_grad():
                pano_norm = F.normalize(pano_matrix, dim=1)
                osm_norm = F.normalize(osm_matrix, dim=1)
                similarity_matrix = pano_norm @ osm_norm.T  # (N_pano_local, N_all_osm)

                # Get top-k for each pano landmark
                topk_vals, topk_idxs = torch.topk(similarity_matrix, min(top_k, similarity_matrix.shape[1]), dim=1)

                for pano_idx in range(len(pano_embeddings_list)):
                    for k in range(topk_idxs.shape[1]):
                        osm_idx = topk_idxs[pano_idx, k].item()
                        score = topk_vals[pano_idx, k].item()
                        pano_to_osm_results.append({
                            'pano_text': pano_texts[pano_idx],
                            'osm_text': all_osm_metadata[osm_idx]['text'],
                            'score': float(score),
                            'pano_id': pano_id,
                            'pano_index': pano_index,
                            'yaw': pano_custom_ids_for_pano[pano_idx][0][2],  # Get yaw from key
                            'source': pano_custom_ids_for_pano[pano_idx][0][0]  # Get source from key
                        })

    # Sort by score
    osm_to_pano_results.sort(key=lambda x: x['score'], reverse=True)
    pano_to_osm_results.sort(key=lambda x: x['score'], reverse=True)

    return {
        'osm_to_pano': osm_to_pano_results[:top_k],
        'pano_to_osm': pano_to_osm_results[:top_k]
    }


def main():
    global PANORAMA_DATA, SENTENCE_SOURCES, OSM_LANDMARKS, OSM_EMBEDDINGS, PANORAMA_EMBEDDINGS, OSM_CUSTOM_IDS, PANO_CUSTOM_IDS

    parser = argparse.ArgumentParser(description='Panorama viewer web app')
    parser.add_argument('--panorama_dir', type=str, required=True,
                       help='Directory containing panorama images (VIGOR format)')
    parser.add_argument('--pinhole_dir', type=str, required=True,
                       help='Directory containing pinhole image subdirectories')
    parser.add_argument('--sentence_dirs', type=str, nargs='+', required=True,
                       help='Directories containing JSONL sentence files (multiple sources)')
    parser.add_argument('--osm_landmarks_geojson', type=str, default=None,
                       help='Path to OSM landmarks GeoJSON file')
    parser.add_argument('--osm_sentences_dir', type=str, default=None,
                       help='Directory containing OSM landmark sentence files')
    parser.add_argument('--osm_embeddings_dir', type=str, default=None,
                       help='Directory containing OSM landmark embeddings')
    parser.add_argument('--panorama_embeddings_dirs', type=str, nargs='+', default=None,
                       help='Directories containing panorama landmark embeddings')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print("Loading sentence data...")
    SENTENCE_SOURCES, PANO_CUSTOM_IDS, PANORAMA_EMBEDDINGS = load_sentence_data(
        args.sentence_dirs,
        args.panorama_embeddings_dirs
    )
    print(f"Loaded {len(SENTENCE_SOURCES)} sentence sources")

    # Load OSM landmarks if provided
    if args.osm_landmarks_geojson and args.osm_sentences_dir:
        print("\nComputing OSM landmarks...")
        OSM_LANDMARKS, OSM_CUSTOM_IDS, OSM_EMBEDDINGS = compute_panorama_to_landmarks(
            args.panorama_dir,
            args.osm_landmarks_geojson,
            args.osm_sentences_dir,
            args.osm_embeddings_dir
        )
    else:
        print("\nSkipping OSM landmarks (no geojson or sentences dir provided)")
        OSM_LANDMARKS = {}
        OSM_CUSTOM_IDS = {}
        OSM_EMBEDDINGS = {}

    print("\nFinding common panoramas...")
    PANORAMA_DATA = find_common_panoramas(args.panorama_dir, args.pinhole_dir, SENTENCE_SOURCES)
    print(f"Found {len(PANORAMA_DATA)} panoramas with complete data")

    if len(PANORAMA_DATA) == 0:
        print("ERROR: No panoramas found with complete data!")
        return

    print(f"\nStarting web server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    app.run(debug=True, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
