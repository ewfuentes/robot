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

# Import from existing modules
from experimental.overhead_matching.swag.data.vigor_dataset import load_landmark_geojson
from common.gps import web_mercator

app = Flask(__name__)

# Global data
PANORAMA_DATA = []
CURRENT_INDEX = 0
OSM_LANDMARKS = {}  # Maps panorama_id -> list of landmark sentences

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

                    // Update location link
                    const locationElem = document.getElementById('pano-location');
                    if (data.lat && data.lon) {
                        const mapsUrl = `https://www.google.com/maps?q=${data.lat},${data.lon}`;
                        locationElem.innerHTML = ` - <a href="${mapsUrl}" target="_blank" style="color:#007bff;text-decoration:none;">📍 ${data.lat.toFixed(6)}, ${data.lon.toFixed(6)}</a>`;
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


def prune_landmark(props):
    """Prune landmark properties (copied from semantic_landmark_extractor.py)."""
    to_drop = [
        "index", "web_mercator", "panorama_idxs", "satellite_idxs",
        "landmark_type", "element", "id", "geometry",
        "opening_hours", "website", "addr:city", "addr:state",
        'check_date', 'checked_exists', 'opening_date',
        'chicago:building_id', 'survey:date', 'payment',
        'disused', 'time', 'end_date']
    out = set()
    for (k, v) in props.items():
        should_add = True
        for prefix in to_drop:
            if k.startswith(prefix):
                should_add = False
                break
        if not should_add:
            continue
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            continue
        out.add((k, v))
    return frozenset(out)


def custom_id_from_props(props):
    """Generate custom_id from landmark properties (copied from semantic_landmark_extractor.py)."""
    json_props = json.dumps(dict(props), sort_keys=True)
    custom_id = base64.b64encode(hashlib.sha256(
        json_props.encode('utf-8')).digest()).decode('utf-8')
    return custom_id


def load_all_jsonl_from_folder(folder):
    """Load all JSONL files from a folder (copied from semantic_landmark_extractor.py)."""
    all_json_objs = []
    for file in folder.glob("*"):
        with open(file, 'r') as f:
            for line in f:
                all_json_objs.append(json.loads(line))
    return all_json_objs


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


def compute_panorama_to_landmarks(panorama_dir, landmarks_geojson_path, osm_sentences_dir, zoom_level=20):
    """
    Compute which OSM landmarks are near each panorama using VigorDataset approach.
    Uses caching to avoid recomputing if inputs haven't changed.

    Returns dict mapping panorama_id -> list of landmark sentences
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
    cache_key = hashlib.sha256("_".join(cache_key_parts).encode()).hexdigest()[:16]
    cache_file = Path(f"/tmp/osm_panorama_cache_{cache_key}.pkl")

    # Try to load from cache
    if cache_file.exists():
        print(f"Loading OSM landmarks from cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"  Loaded cached data for {len(cached_data)} panoramas")
            return cached_data
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

    # Build final result dictionary
    print("Building result dictionary...")
    start = time.time()
    pano_to_landmarks = {}

    for pano_id, landmark_list in pano_to_landmark_indices.items():
        for landmark_idx, custom_id in landmark_list:
            if custom_id in osm_sentences:
                if pano_id not in pano_to_landmarks:
                    pano_to_landmarks[pano_id] = []
                pano_to_landmarks[pano_id].append(osm_sentences[custom_id])

    print(f"  Built dictionary in {time.time()-start:.1f}s")
    print(f"Found OSM landmarks for {len(pano_to_landmarks)} panoramas")

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(pano_to_landmarks, f)
        print("  Cache saved successfully")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")

    return pano_to_landmarks


def load_sentence_data(sentence_dirs):
    """Load and parse JSONL sentence files from multiple directories."""
    sources = {}

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

                                        # For individual mode, store simple strings
                                        sentences = [{'description': lm['description'], 'mode': 'individual'}
                                                    for lm in landmarks]

                                        if base_id not in source_data:
                                            source_data[base_id] = {}
                                        if yaw not in source_data[base_id]:
                                            source_data[base_id][yaw] = []
                                        source_data[base_id][yaw].extend(sentences)
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
                                                source_data[base_id][yaw].append({
                                                    'description': description,
                                                    'landmark_id': lm_idx,  # Use index for grouping
                                                    'all_yaws': yaw_angles,
                                                    'mode': 'all'
                                                })
                                    except json.JSONDecodeError:
                                        print(f"Warning: Failed to parse JSON content for {custom_id}")
                    except Exception as e:
                        print(f"Warning: Error parsing line in {jsonl_file}: {e}")

        if source_data:
            sources[source_name] = source_data

    return sources


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
    global PANORAMA_DATA, SENTENCE_SOURCES, OSM_LANDMARKS

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

    # Get OSM landmark sentences for this panorama (collapse duplicates)
    osm_sentences_raw = OSM_LANDMARKS.get(pano['id'], [])
    osm_sentences = collapse_string_list(osm_sentences_raw)

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
    global PANORAMA_DATA, SENTENCE_SOURCES, OSM_LANDMARKS

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
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    print("Loading sentence data...")
    SENTENCE_SOURCES = load_sentence_data(args.sentence_dirs)
    print(f"Loaded {len(SENTENCE_SOURCES)} sentence sources")

    # Load OSM landmarks if provided
    if args.osm_landmarks_geojson and args.osm_sentences_dir:
        print("\nComputing OSM landmarks...")
        OSM_LANDMARKS = compute_panorama_to_landmarks(
            args.panorama_dir,
            args.osm_landmarks_geojson,
            args.osm_sentences_dir
        )
    else:
        print("\nSkipping OSM landmarks (no geojson or sentences dir provided)")
        OSM_LANDMARKS = {}

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
