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
from flask import Flask, render_template_string, send_file, jsonify
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
    prune_landmark)
from common.gps import web_mercator


@dataclass
class PanoramaLandmark:
    """Represents a landmark visible in a panorama."""
    description: str
    landmark_id: tuple[str, int]  # (panorama_id, landmark_index)
    yaws: list[int]  # Yaw angles where this landmark is visible
    panorama_lat: float
    panorama_lon: float


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

                            html += `<li class="landmark-all-mode" style="border-color:${color};background-color:${color}15;">${desc}${yawBadges}${countBadge}</li>`;
                        });
                        html += `</ul>`;
                    }

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

                // Update comparison lists with similarity data
                currentSimilarityData = similarityData;
                selectedLandmark = null; // Reset selection when changing panorama
                globalMatches = null; // Reset global matches when changing panorama
                updateComparisonLists(similarityData);
            }).catch(err => {
                console.error('Error loading panorama:', err);
            });
        }

        function navigate(delta) {
            let newIndex = currentIndex + delta;
            if (newIndex < 0) newIndex = totalPanoramas - 1;
            if (newIndex >= totalPanoramas) newIndex = 0;
            loadPanorama(newIndex);
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
            # Collect embeddings for our loaded landmarks
            embeddings_data = []  # List of (custom_id, embedding_vector)
            custom_ids_set = set(osm_landmarks_dict.keys())

            print(f"    Reading embedding files for {len(custom_ids_set)} landmarks...")
            files_processed = 0
            for jsonl_file in embeddings_path.glob('*'):
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
    Load panorama landmark embeddings from JSONL files.
    Uses caching to avoid recomputing if inputs haven't changed.

    Args:
        embeddings_dir: Directory containing panorama embedding JSONL files
        pano_sentences: dict[str, list[PanoramaLandmark]] to ensure consistency

    Returns:
        Tuple of (PANO_EMBEDDINGS tensor, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE)
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        print(f"Warning: Embeddings directory not found: {embeddings_dir}")
        return None, {}, []

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

    print("Loading panorama embeddings (no cache found)...")
    start = time.time()

    # Collect all embeddings
    embeddings_data = []  # List of (landmark_id, embedding_vector)

    print("  Reading embedding files...")
    files_processed = 0
    for jsonl_file in embeddings_path.glob('*'):
        if not jsonl_file.is_file():
            continue

        files_processed += 1
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    custom_id = entry['custom_id']

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
        for jsonl_file in sentence_path.glob('*'):
            if not jsonl_file.is_file():
                continue

            file_count += 1
            with open(jsonl_file, 'r') as f:
                for line in f:
                    entry_count += 1
                    try:
                        entry = json.loads(line)
                        custom_id = entry['custom_id']

                        # Only process "all mode" entries (custom_id format: "pano_id,lat,lon,")
                        # Skip "individual mode" entries (format: "pano_id_yaw_N")
                        # All mode has commas, individual mode doesn't
                        if ',' not in custom_id:
                            continue

                        # Parse custom_id to extract panorama_id, lat, lon
                        # Format: "panorama_id,latitude,longitude,"
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

                        # Extract landmarks from response
                        if 'response' not in entry or 'body' not in entry['response']:
                            continue

                        body = entry['response']['body']
                        if 'choices' not in body or len(body['choices']) == 0:
                            continue

                        content = body['choices'][0]['message']['content']
                        try:
                            landmarks_data = json.loads(content)
                            landmarks = landmarks_data.get('landmarks', [])

                            # Create PanoramaLandmark objects
                            for lm_idx, lm in enumerate(landmarks):
                                description = lm['description']
                                yaw_angles = lm.get('yaw_angles', [])

                                landmark = PanoramaLandmark(
                                    description=description,
                                    landmark_id=(pano_id, lm_idx),
                                    yaws=yaw_angles,
                                    panorama_lat=pano_lat,
                                    panorama_lon=pano_lon
                                )

                                # Add to collection
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


def find_common_panoramas(panorama_dir, pinhole_dir, pano_sentences):
    """
    Find panoramas that exist in all required locations.

    Args:
        panorama_dir: Directory containing panorama images
        pinhole_dir: Directory containing pinhole image subdirectories
        pano_sentences: dict[str, list[PanoramaLandmark]] from load_sentence_data()

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

    # Find intersection with sentence data
    common_ids = set(pano_files.keys()) & set(pinhole_dirs.keys()) & set(pano_sentences.keys())
    print(f"  Found {len(common_ids)} panoramas present in all locations")

    # Build panorama data
    t3 = time.time()
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
    global PANORAMA_DATA
    return jsonify({
        'panoramas': [{'id': pano['id'], 'index': i} for i, pano in enumerate(PANORAMA_DATA)]
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
                landmarks_by_yaw[yaw].append({
                    'description': landmark.description,
                    'landmark_id': landmark.landmark_id,  # (pano_id, idx)
                    'all_yaws': landmark.yaws
                })

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

    # Resolve OSM landmarks (using indices, not duplicated data)
    osm_landmarks_data = []
    if pano_id in PANO_TO_OSM:
        osm_custom_ids = PANO_TO_OSM[pano_id]
        osm_landmarks_raw = [OSM_LANDMARKS[cid].description
                             for cid in osm_custom_ids
                             if cid in OSM_LANDMARKS]
        osm_landmarks_data = collapse_string_list(osm_landmarks_raw)

    # Get coordinates from first landmark (all should have same coords)
    lat, lon = None, None
    if pano_id in PANO_SENTENCES and len(PANO_SENTENCES[pano_id]) > 0:
        first_landmark = PANO_SENTENCES[pano_id][0]
        lat = first_landmark.panorama_lat
        lon = first_landmark.panorama_lon

    return jsonify({
        'name': pano_id,
        'total': len(PANORAMA_DATA),
        'yaw_angles': pano['yaw_angles'],
        'yaw_data': yaw_data,
        'osm_landmarks': osm_landmarks_data,
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


def main():
    global PANORAMA_DATA, PANO_SENTENCES, OSM_LANDMARKS, PANO_TO_OSM
    global PANO_EMBEDDINGS, PANO_EMBEDDING_INDEX, PANO_INDEX_REVERSE
    global OSM_EMBEDDINGS, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE

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
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    startup_start = time.time()

    # Construct paths for panorama landmarks
    pano_sentences_dirs = [str(Path(args.pano_landmarks_dir) / 'sentences')]
    pano_embeddings_dir = str(Path(args.pano_landmarks_dir) / 'embeddings')

    # Construct paths for OSM landmarks (if provided)
    if args.osm_landmarks_dir:
        osm_sentences_dir = str(Path(args.osm_landmarks_dir) / 'sentences')
        osm_embeddings_dir = str(Path(args.osm_landmarks_dir) / 'embeddings')
    else:
        osm_sentences_dir = None
        osm_embeddings_dir = None

    # Step 1: Load OSM landmarks (independent, no duplication)
    if args.osm_landmarks_geojson and osm_sentences_dir:
        print("\n" + "="*60)
        print("STEP 1: Loading OSM landmarks")
        print("="*60)
        osm_data = load_osm_landmarks(
            args.osm_landmarks_geojson,
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
        PANO_SENTENCES
    )
    step4_time = time.time() - step4_start
    print(f"Found {len(PANORAMA_DATA)} panoramas with complete data in {step4_time:.1f}s")

    if len(PANORAMA_DATA) == 0:
        print("ERROR: No panoramas found with complete data!")
        return

    startup_time = time.time() - startup_start
    print("\n" + "="*60)
    print(f"TOTAL STARTUP TIME: {startup_time:.1f}s")
    print("="*60)
    print(f"Starting web server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
