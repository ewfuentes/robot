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
CURRENT_INDEX = 0

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
                <span style="margin-left: 10px;">Landmarks with matching colors and yaw badges (e.g., <span style="background:#ddd;padding:2px 4px;border-radius:2px;">90°</span>) appear in multiple views.</span>
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
            <h2>Full Panorama</h2>
            <img id="panorama-img" class="panorama-img" src="" alt="Panorama">
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let totalPanoramas = 0;

        // Color palette for landmarks (distinct colors)
        const LANDMARK_COLORS = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#C06C84',
            '#6C5B7B', '#355C7D', '#F67280', '#C8D6AF', '#8E7AB5'
        ];

        function getLandmarkColor(landmarkId) {
            return LANDMARK_COLORS[landmarkId % LANDMARK_COLORS.length];
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

        // Load first panorama on page load
        loadPanorama(0);
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
        props = prune_landmark(landmark.dropna().to_dict())
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

    Args:
        geojson_path: Path to OSM landmarks GeoJSON file
        sentences_dir: Directory containing OSM landmark sentence files
        embeddings_dir: Optional directory containing OSM landmark embeddings

    Returns:
        Tuple of (OSM_LANDMARKS dict, OSM_EMBEDDINGS tensor, OSM_EMBEDDING_INDEX, OSM_INDEX_REVERSE)
        If embeddings_dir is None, returns (dict, None, {}, [])
    """
    print("Loading OSM landmarks...")
    start = time.time()

    # Load GeoJSON to get landmark properties
    landmarks_df = load_landmark_geojson(geojson_path, zoom_level=20)
    print(f"  Loaded {len(landmarks_df)} landmarks from GeoJSON in {time.time()-start:.1f}s")

    # Build custom_id for each landmark
    print("  Computing custom IDs...")
    landmark_custom_ids = []
    for _, landmark in landmarks_df.iterrows():
        props = prune_landmark(landmark.dropna().to_dict())
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

    # TODO: Load embeddings if embeddings_dir is provided
    osm_embeddings = None
    osm_embedding_index = {}
    osm_index_reverse = []

    return osm_landmarks_dict, osm_embeddings, osm_embedding_index, osm_index_reverse


def load_sentence_data(sentence_dirs):
    """
    Load and parse JSONL sentence files from multiple directories.

    Returns:
        dict[str, list[PanoramaLandmark]]: Maps panorama_id to list of landmarks
    """
    # Collect all landmarks by panorama_id
    pano_landmarks: dict[str, list[PanoramaLandmark]] = {}

    print(f"  Processing {len(sentence_dirs)} sentence directories...")
    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if not sentence_path.exists():
            print(f"Warning: Sentence directory not found: {sentence_dir}")
            continue

        print(f"    Loading from {sentence_path.name}...")

        # Load all files in this directory (JSONL format)
        for jsonl_file in sentence_path.glob('*'):
            if not jsonl_file.is_file():
                continue

            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        custom_id = entry['custom_id']

                        # Only process "all mode" entries (custom_id format: "pano_id,lat,lon,")
                        # Skip "individual mode" entries (format: "pano_id_yaw_N")
                        if '_yaw_' in custom_id:
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

    print(f"  Loaded landmarks for {len(pano_landmarks)} panoramas")
    return pano_landmarks


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
    pano_files = {parse_vigor_filename(f.name): f for f in panorama_path.glob('*.jpg')}

    # Get panorama IDs from pinhole directory
    pinhole_dirs = {d.name.split(',')[0]: d for d in pinhole_path.iterdir() if d.is_dir()}

    # Find intersection with sentence data
    common_ids = set(pano_files.keys()) & set(pinhole_dirs.keys()) & set(pano_sentences.keys())

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

    parser = argparse.ArgumentParser(description='Panorama viewer web app')
    parser.add_argument('--panorama_dir', type=str, required=True,
                       help='Directory containing panorama images (VIGOR format)')
    parser.add_argument('--pinhole_dir', type=str, required=True,
                       help='Directory containing pinhole image subdirectories')
    parser.add_argument('--sentence_dirs', type=str, nargs='+', required=True,
                       help='Directories containing JSONL sentence files')
    parser.add_argument('--osm_landmarks_geojson', type=str, default=None,
                       help='Path to OSM landmarks GeoJSON file')
    parser.add_argument('--osm_sentences_dir', type=str, default=None,
                       help='Directory containing OSM landmark sentence files')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    # Step 1: Load OSM landmarks (independent, no duplication)
    if args.osm_landmarks_geojson and args.osm_sentences_dir:
        print("\n" + "="*60)
        print("STEP 1: Loading OSM landmarks")
        print("="*60)
        osm_data = load_osm_landmarks(
            args.osm_landmarks_geojson,
            args.osm_sentences_dir
        )
        OSM_LANDMARKS, _, _, _ = osm_data
        print(f"Loaded {len(OSM_LANDMARKS)} OSM landmarks")

        # Step 2: Pre-compute panorama→OSM associations (indices only)
        print("\n" + "="*60)
        print("STEP 2: Computing panorama→OSM associations")
        print("="*60)
        PANO_TO_OSM = compute_panorama_to_osm_associations(
            args.panorama_dir,
            OSM_LANDMARKS,
            args.osm_landmarks_geojson
        )
        print(f"Computed associations for {len(PANO_TO_OSM)} panoramas")
    else:
        print("\nSkipping OSM landmarks (no geojson or sentences dir provided)")
        OSM_LANDMARKS = {}
        PANO_TO_OSM = {}

    # Step 3: Load panorama landmark sentences
    print("\n" + "="*60)
    print("STEP 3: Loading panorama landmark sentences")
    print("="*60)
    PANO_SENTENCES = load_sentence_data(args.sentence_dirs)
    print(f"Loaded landmarks for {len(PANO_SENTENCES)} panoramas")

    # Step 4: Find common panoramas
    print("\n" + "="*60)
    print("STEP 4: Finding common panoramas")
    print("="*60)
    PANORAMA_DATA = find_common_panoramas(
        args.panorama_dir,
        args.pinhole_dir,
        PANO_SENTENCES
    )
    print(f"Found {len(PANORAMA_DATA)} panoramas with complete data")

    if len(PANORAMA_DATA) == 0:
        print("ERROR: No panoramas found with complete data!")
        return

    print("\n" + "="*60)
    print(f"Starting web server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
