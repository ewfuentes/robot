#!/usr/bin/env python3
"""
Geometric filter viewer for pano_v2 landmarks.

Visualizes heading cones from panorama landmark bounding boxes and shows
which OSM landmarks fall within each cone. Single-file Flask app.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:geometric_filter_viewer -- \
        --pinhole_dir /data/overhead_matching/datasets/pinhole_images/Chicago \
        --pano_pickle /data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/Chicago/embeddings/embeddings.pkl \
        --osm_feather /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather \
        --osm_sentences_dir /data/overhead_matching/datasets/semantic_landmark_embeddings/v5_202001_w_addresses/sentences \
        --port 5002
"""

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import shapely.geometry
import shapely.ops
import geopandas as gpd
from flask import Flask, jsonify, send_file, render_template_string

from common.gps import web_mercator
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark, custom_id_from_props)

ZOOM_LEVEL = 20
CONE_RADIUS_PX = 640

app = Flask(__name__)

# Global state populated at startup
PANORAMAS = []       # list of panorama dicts
PINHOLE_DIR = None   # Path to pinhole images
OSM_DF = None        # geopandas DataFrame with geometry_px + custom_id
STRTREE = None       # shapely.STRtree on geometry_px
NEARBY_OSM = {}      # pano_index -> list of OSM df row indices
OSM_SENTENCES = {}   # custom_id -> sentence string


# ---- Data loading ----

def load_pano_v2_data(pickle_path):
    """Load pano_v2 panorama data from pickle file."""
    print(f"Loading pano_v2 data from {pickle_path}...")
    start = time.time()
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    panoramas = []
    for key, pano_data in data['panoramas'].items():
        parts = key.split(',')
        pano_id = parts[0]
        lat = float(parts[1])
        lon = float(parts[2])
        wm_y, wm_x = web_mercator.latlon_to_pixel_coords(lat, lon, ZOOM_LEVEL)

        panoramas.append({
            'pano_id': pano_id,
            'pano_key': key,
            'lat': lat,
            'lon': lon,
            'wm_x': float(wm_x),
            'wm_y': float(wm_y),
            'location_type': pano_data.get('location_type', ''),
            'landmarks': pano_data.get('landmarks', []),
        })

    panoramas.sort(key=lambda p: p['pano_id'])
    print(f"  Loaded {len(panoramas)} panoramas in {time.time()-start:.1f}s")
    return panoramas


def load_osm_data(feather_path):
    """Load OSM landmarks from feather, compute geometry_px and custom_id."""
    print(f"Loading OSM landmarks from {feather_path}...")
    start = time.time()
    df = gpd.read_feather(feather_path)

    def convert_geometry_to_pixels(geometry):
        def coord_transform(lon, lat):
            y, x = web_mercator.latlon_to_pixel_coords(lat, lon, ZOOM_LEVEL)
            return (x, y)
        return shapely.ops.transform(coord_transform, geometry)

    df['geometry_px'] = df['geometry'].apply(convert_geometry_to_pixels)

    # Compute custom_id for each row (same hash used by sentence generation pipeline)
    print("  Computing custom_ids...")
    t = time.time()
    df['custom_id'] = df.apply(
        lambda row: custom_id_from_props(prune_landmark(row.dropna().to_dict())), axis=1)
    print(f"  Computed {len(df)} custom_ids in {time.time()-t:.1f}s")

    print(f"  Loaded {len(df)} OSM landmarks in {time.time()-start:.1f}s")
    return df


def load_osm_sentences(sentences_dir):
    """Load OSM landmark sentences from JSONL files.

    Each JSONL entry has a custom_id (SHA256 hash of pruned OSM props) and
    a GPT-generated natural language description in response.body.choices[0].message.content.

    Returns dict mapping custom_id -> sentence string.
    """
    sentences_path = Path(sentences_dir)
    if not sentences_path.exists():
        print(f"  Warning: sentences directory not found: {sentences_dir}")
        return {}

    print(f"Loading OSM sentences from {sentences_dir}...")
    start = time.time()
    sentences = {}
    files_processed = 0

    for jsonl_file in sorted(sentences_path.iterdir()):
        if not jsonl_file.is_file():
            continue
        files_processed += 1
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    cid = entry['custom_id']
                    body = entry.get('response', {}).get('body', {})
                    choices = body.get('choices', [])
                    if choices:
                        sentences[cid] = choices[0]['message']['content']
                except Exception:
                    pass

    print(f"  Loaded {len(sentences)} sentences from {files_processed} files in {time.time()-start:.1f}s")
    return sentences


def precompute_nearby_osm(panoramas, osm_df, max_dist_px=CONE_RADIUS_PX):
    """Precompute nearby OSM landmarks for each panorama using STRtree."""
    print("Building spatial index and precomputing nearby OSM landmarks...")
    start = time.time()

    strtree = shapely.STRtree(osm_df.geometry_px.values)

    queries = []
    for pano in panoramas:
        cx, cy = pano['wm_x'], pano['wm_y']
        queries.append(shapely.box(
            xmin=cx - max_dist_px,
            xmax=cx + max_dist_px,
            ymin=cy - max_dist_px,
            ymax=cy + max_dist_px))

    results = strtree.query(queries, predicate='intersects')

    nearby = {}
    for pano_idx, osm_idx in results.T:
        pano_idx = int(pano_idx)
        osm_idx = int(osm_idx)
        if pano_idx not in nearby:
            nearby[pano_idx] = []
        nearby[pano_idx].append(osm_idx)

    total_pairs = sum(len(v) for v in nearby.values())
    print(f"  {total_pairs} panorama-OSM pairs for {len(nearby)} panoramas in {time.time()-start:.1f}s")
    return strtree, nearby


# ---- Heading / cone computation ----

def bbox_x_to_heading_rad(yaw_deg, x_norm, fov_deg=90):
    """Convert bounding box x coordinate to heading in radians.

    From panorama_to_pinhole.py: col_frac = linspace(1, -1, W), so
    col_frac = 1 - 2*(x_norm/1000). The offset angle from the view center
    is atan2(col_frac, fx) where fx = 1/tan(fov/2).
    """
    fx = 1.0 / math.tan(math.radians(fov_deg / 2))
    col_frac = 1.0 - 2.0 * (x_norm / 1000.0)
    offset_rad = math.atan2(col_frac, fx)
    return math.radians(yaw_deg) + offset_rad


def make_cone_polygon(cx, cy, h_min, h_max, radius=CONE_RADIUS_PX):
    """Build a wedge polygon in web mercator pixel space.

    Heading convention: 0 = north (-y in web mercator).
    dx_wm = -r * sin(h), dy_wm = -r * cos(h).
    """
    if h_max < h_min:
        h_max += 2 * math.pi

    n_arc = 30
    points = [(cx, cy)]
    for i in range(n_arc + 1):
        h = h_min + (h_max - h_min) * i / n_arc
        dx = -radius * math.sin(h)
        dy = -radius * math.cos(h)
        points.append((cx + dx, cy + dy))
    points.append((cx, cy))
    return shapely.Polygon(points)


def cone_to_latlon(cone_poly):
    """Convert cone polygon vertices from web mercator px to [lat, lon] list."""
    coords = list(cone_poly.exterior.coords)
    latlon = []
    for x_wm, y_wm in coords:
        lat, lon = web_mercator.pixel_coords_to_latlon(float(y_wm), float(x_wm), ZOOM_LEVEL)
        latlon.append([float(lat), float(lon)])
    return latlon


def compute_cones_for_landmark(pano, landmark):
    """Compute heading cones for a landmark's bounding boxes.

    Returns list of (cone_polygon, cone_latlon) tuples.
    """
    cones = []
    for bbox in landmark.get('bounding_boxes', []):
        yaw_deg = int(bbox['yaw_angle'])
        xmin = bbox['xmin']
        xmax = bbox['xmax']

        # xmin → left edge → larger col_frac → larger heading
        heading_max = bbox_x_to_heading_rad(yaw_deg, xmin)
        # xmax → right edge → smaller col_frac → smaller heading
        heading_min = bbox_x_to_heading_rad(yaw_deg, xmax)

        cone_poly = make_cone_polygon(pano['wm_x'], pano['wm_y'], heading_min, heading_max)
        cones.append((cone_poly, cone_to_latlon(cone_poly)))
    return cones


def get_osm_tags(row):
    """Extract OSM tags from a dataframe row, excluding geometry columns."""
    skip = {'geometry', 'geometry_px', 'custom_id'}
    tags = {}
    for col in row.index:
        if col in skip:
            continue
        val = row[col]
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            tags[col] = str(val)
    return tags


# ---- Flask routes ----

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/count')
def get_count():
    return jsonify({'count': len(PANORAMAS)})


@app.route('/api/panorama/<int:idx>')
def get_panorama(idx):
    if idx < 0 or idx >= len(PANORAMAS):
        return jsonify({'error': 'Invalid index'}), 404

    pano = PANORAMAS[idx]
    osm_idxs = NEARBY_OSM.get(idx, [])

    # Compute cones for each pano landmark and collect for intersection testing
    landmark_results = []
    all_cones = []  # (landmark_idx, cone_polygon)

    for lm in pano['landmarks']:
        lm_idx = lm['landmark_idx']
        cones = compute_cones_for_landmark(pano, lm)

        for cone_poly, _ in cones:
            all_cones.append((lm_idx, cone_poly))

        primary = lm.get('primary_tag', {})
        tag_str = f"{primary.get('key', '')}={primary.get('value', '')}"
        add_tags = [f"{t['key']}={t['value']}" for t in lm.get('additional_tags', [])]

        landmark_results.append({
            'idx': lm_idx,
            'primary_tag': tag_str,
            'additional_tags': add_tags,
            'confidence': lm.get('confidence', ''),
            'description': lm.get('description', ''),
            'bounding_boxes': [
                {'yaw': int(bb['yaw_angle']),
                 'xmin': bb['xmin'], 'xmax': bb['xmax'],
                 'ymin': bb['ymin'], 'ymax': bb['ymax']}
                for bb in lm.get('bounding_boxes', [])
            ],
            'cones_latlon': [c[1] for c in cones],
        })

    # Test intersection of each nearby OSM landmark with each pano landmark's cones
    osm_results = []
    for osm_idx in osm_idxs:
        osm_row = OSM_DF.iloc[osm_idx]
        geom_px = osm_row['geometry_px']
        if geom_px is None or geom_px.is_empty:
            continue

        matches = []
        for lm in pano['landmarks']:
            lm_idx = lm['landmark_idx']
            matched = any(
                cone_lm_idx == lm_idx and cone_poly.intersects(geom_px)
                for cone_lm_idx, cone_poly in all_cones
            )
            matches.append(matched)

        centroid = osm_row['geometry'].centroid
        cid = osm_row.get('custom_id', '')
        sentence = OSM_SENTENCES.get(cid, '')
        osm_results.append({
            'tags': get_osm_tags(osm_row),
            'geojson': shapely.geometry.mapping(osm_row['geometry']),
            'centroid_lat': float(centroid.y),
            'centroid_lon': float(centroid.x),
            'matches': matches,
            'sentence': sentence,
        })

    return jsonify({
        'pano_id': pano['pano_id'],
        'pano_key': pano['pano_key'],
        'lat': pano['lat'],
        'lon': pano['lon'],
        'location_type': pano['location_type'],
        'total': len(PANORAMAS),
        'landmarks': landmark_results,
        'osm_landmarks': osm_results,
    })


@app.route('/api/image/<int:idx>/<int:yaw>')
def get_image(idx, yaw):
    if idx < 0 or idx >= len(PANORAMAS):
        return 'Invalid index', 404

    pano = PANORAMAS[idx]
    subdir = Path(PINHOLE_DIR) / pano['pano_key']
    jpg_path = subdir / f'yaw_{yaw:03d}.jpg'
    png_path = subdir / f'yaw_{yaw:03d}.png'

    if jpg_path.exists():
        return send_file(jpg_path)
    elif png_path.exists():
        return send_file(png_path)
    else:
        return 'Image not found', 404


# ---- HTML template ----

HTML_TEMPLATE = r'''
<!DOCTYPE html>
<html>
<head>
    <title>Geometric Filter Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #f0f0f0; }

        .header {
            background: #1a1a2e; color: white; padding: 10px 20px;
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
        }
        .header h1 { font-size: 18px; font-weight: 600; white-space: nowrap; }
        .nav-btn {
            background: #16213e; border: 1px solid #0f3460; color: white;
            padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
        }
        .nav-btn:hover { background: #0f3460; }
        .header input {
            padding: 5px 8px; border-radius: 4px; border: 1px solid #0f3460;
            background: #16213e; color: white; font-size: 13px; width: 200px;
        }
        .header input::placeholder { color: #777; }
        .info { color: #aaa; font-size: 13px; white-space: nowrap; }

        .main { display: flex; height: calc(100vh - 48px); }
        .left-panel { width: 55%; overflow-y: auto; padding: 10px; }
        .right-panel { width: 45%; display: flex; flex-direction: column; }

        .pinhole-grid {
            display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
            margin-bottom: 10px;
        }
        .pinhole-cell {
            position: relative; background: #222; border-radius: 4px; overflow: hidden;
            cursor: crosshair;
        }
        .pinhole-cell img { width: 100%; display: block; }
        .pinhole-cell canvas {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        }
        .pinhole-label {
            position: absolute; top: 4px; left: 4px; background: rgba(0,0,0,0.65);
            color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;
            pointer-events: none;
        }

        .landmark-panel {
            background: white; border-radius: 6px; padding: 10px;
        }
        .landmark-panel h3 {
            font-size: 14px; color: #333; margin-bottom: 8px;
            border-bottom: 1px solid #eee; padding-bottom: 5px;
        }
        .lm-item {
            padding: 7px 8px; margin-bottom: 3px; border-radius: 4px; cursor: pointer;
            border-left: 4px solid transparent; font-size: 13px; transition: background 0.12s;
        }
        .lm-item:hover { background: #f0f7ff; }
        .lm-item.selected { background: #e3f2fd; }
        .lm-tag { font-weight: 600; }
        .lm-conf {
            font-size: 10px; padding: 1px 5px; border-radius: 2px;
            margin-left: 5px; text-transform: uppercase;
        }
        .lm-conf.high { background: #c8e6c9; color: #2e7d32; }
        .lm-conf.medium { background: #fff9c4; color: #f57f17; }
        .lm-conf.low { background: #ffcdd2; color: #c62828; }
        .lm-desc { color: #666; font-size: 12px; margin-top: 2px; }
        .yaw-badge {
            display: inline-block; background: #e0e0e0; padding: 1px 4px;
            border-radius: 2px; margin-left: 2px; font-size: 10px;
        }
        .lm-match-count {
            font-size: 11px; color: #4caf50; margin-left: 6px; font-weight: 600;
        }

        #map { flex: 1; min-height: 0; }
        .osm-panel {
            max-height: 45vh; overflow-y: auto; background: white; padding: 8px 10px;
            border-top: 2px solid #ddd; font-size: 12px;
        }
        .osm-panel h3 { font-size: 13px; margin-bottom: 5px; }
        .osm-item {
            padding: 4px 6px; border-radius: 3px; margin-bottom: 2px;
            cursor: pointer; transition: background 0.1s;
        }
        .osm-item:hover { filter: brightness(0.95); }
        .osm-match { background: #e8f5e9; }
        .osm-nomatch { background: #f5f5f5; color: #999; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Geometric Filter Viewer</h1>
        <button class="nav-btn" onclick="navigate(-1)">&#9664; Prev</button>
        <span class="info">
            <span id="pano-idx">0</span> / <span id="pano-total">0</span>
            &mdash; <b id="pano-id">...</b>
        </span>
        <button class="nav-btn" onclick="navigate(1)">Next &#9654;</button>
        <input type="text" id="search-input" placeholder="Search pano ID..."
               onkeypress="if(event.key==='Enter')searchPano()">
        <button class="nav-btn" onclick="searchPano()">Go</button>
        <span class="info" id="location-info"></span>
    </div>

    <div class="main">
        <div class="left-panel">
            <div class="pinhole-grid" id="pinhole-grid"></div>
            <div class="landmark-panel">
                <h3>Panorama Landmarks <span id="match-summary" style="font-weight:normal; color:#666;"></span></h3>
                <div id="landmarks"></div>
            </div>
        </div>
        <div class="right-panel">
            <div id="map"></div>
            <div class="osm-panel">
                <h3>Nearby OSM Landmarks (<span id="osm-count">0</span>)</h3>
                <div id="osm-list"></div>
            </div>
        </div>
    </div>

    <script>
        let currentIdx = 0;
        let totalPanos = 0;
        let currentData = null;
        let selectedLandmark = null;
        let panoIdMap = {};

        const COLORS = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
            '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
        ];

        // Leaflet map
        const map = L.map('map').setView([41.88, -87.63], 17);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 21, attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        let panoMarker = null;
        let coneLayers = [];
        let osmGeoLayers = [];

        function clearMapOverlays() {
            coneLayers.forEach(l => map.removeLayer(l));
            coneLayers = [];
            osmGeoLayers.forEach(l => map.removeLayer(l));
            osmGeoLayers = [];
            if (panoMarker) { map.removeLayer(panoMarker); panoMarker = null; }
        }

        function navigate(delta) {
            let n = currentIdx + delta;
            if (n < 0) n = totalPanos - 1;
            if (n >= totalPanos) n = 0;
            loadPanorama(n);
        }

        function searchPano() {
            const q = document.getElementById('search-input').value.trim();
            if (q in panoIdMap) loadPanorama(panoIdMap[q]);
        }

        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
        });

        function loadPanorama(idx) {
            currentIdx = idx;
            selectedLandmark = null;
            fetch('/api/panorama/' + idx)
                .then(r => r.json())
                .then(data => {
                    currentData = data;
                    totalPanos = data.total;
                    panoIdMap[data.pano_id] = idx;

                    document.getElementById('pano-idx').textContent = idx + 1;
                    document.getElementById('pano-total').textContent = data.total;
                    document.getElementById('pano-id').textContent = data.pano_id;
                    document.getElementById('location-info').textContent =
                        data.lat.toFixed(6) + ', ' + data.lon.toFixed(6) + ' (' + data.location_type + ')';

                    renderPinholes(data);
                    renderLandmarkList(data);
                    renderMap(data);
                    renderOsmPanel(data);
                });
        }

        // ---- Pinhole images with bbox overlay ----
        function renderPinholes(data) {
            const grid = document.getElementById('pinhole-grid');
            grid.innerHTML = '';

            [0, 90, 180, 270].forEach(yaw => {
                const cell = document.createElement('div');
                cell.className = 'pinhole-cell';
                cell.dataset.yaw = yaw;

                const img = document.createElement('img');
                img.src = '/api/image/' + currentIdx + '/' + yaw;
                img.onload = () => drawBboxes(cell, yaw, data);

                const canvas = document.createElement('canvas');
                const label = document.createElement('div');
                label.className = 'pinhole-label';
                label.textContent = yaw + '\u00B0';

                cell.appendChild(img);
                cell.appendChild(canvas);
                cell.appendChild(label);
                grid.appendChild(cell);

                // Click on canvas to select landmark by bbox
                canvas.addEventListener('click', e => {
                    const rect = canvas.getBoundingClientRect();
                    const sx = canvas.width / rect.width;
                    const sy = canvas.height / rect.height;
                    const cx = (e.clientX - rect.left) * sx;
                    const cy = (e.clientY - rect.top) * sy;
                    const imgSx = canvas.width / 1000;
                    const imgSy = canvas.height / 1000;

                    for (let li = data.landmarks.length - 1; li >= 0; li--) {
                        for (const bb of data.landmarks[li].bounding_boxes) {
                            if (bb.yaw !== yaw) continue;
                            const bx = bb.xmin * imgSx, by = bb.ymin * imgSy;
                            const bw = (bb.xmax - bb.xmin) * imgSx;
                            const bh = (bb.ymax - bb.ymin) * imgSy;
                            if (cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh) {
                                selectLandmark(li);
                                return;
                            }
                        }
                    }
                });
            });
        }

        function drawBboxes(cell, yaw, data) {
            const img = cell.querySelector('img');
            const canvas = cell.querySelector('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const sx = img.naturalWidth / 1000;
            const sy = img.naturalHeight / 1000;

            data.landmarks.forEach((lm, li) => {
                lm.bounding_boxes.forEach(bb => {
                    if (bb.yaw !== yaw) return;
                    const color = COLORS[li % COLORS.length];
                    const isSel = selectedLandmark === li;

                    const x = bb.xmin * sx, y = bb.ymin * sy;
                    const w = (bb.xmax - bb.xmin) * sx;
                    const h = (bb.ymax - bb.ymin) * sy;

                    ctx.strokeStyle = color;
                    ctx.lineWidth = isSel ? 3 : 2;
                    ctx.globalAlpha = isSel ? 1.0 : 0.5;
                    ctx.strokeRect(x, y, w, h);

                    ctx.fillStyle = color;
                    ctx.globalAlpha = isSel ? 0.18 : 0.05;
                    ctx.fillRect(x, y, w, h);

                    // Tag label at top of bbox
                    ctx.globalAlpha = isSel ? 1.0 : 0.65;
                    ctx.fillStyle = color;
                    ctx.font = 'bold 13px sans-serif';
                    ctx.fillText(lm.primary_tag, x + 2, y + 13);
                });
            });
            ctx.globalAlpha = 1.0;
        }

        // ---- Landmark list ----
        function renderLandmarkList(data) {
            const el = document.getElementById('landmarks');
            el.innerHTML = '';
            let summaryParts = [];

            data.landmarks.forEach((lm, li) => {
                const div = document.createElement('div');
                div.className = 'lm-item' + (selectedLandmark === li ? ' selected' : '');
                div.style.borderLeftColor = COLORS[li % COLORS.length];

                const yawBadges = lm.bounding_boxes
                    .map(bb => '<span class="yaw-badge">' + bb.yaw + '\u00B0</span>')
                    .join('');

                // Count matching OSM landmarks for this pano landmark
                let matchCount = 0;
                if (data.osm_landmarks) {
                    matchCount = data.osm_landmarks.filter(o => o.matches[li]).length;
                }
                const matchSpan = matchCount > 0
                    ? '<span class="lm-match-count">' + matchCount + ' OSM match' + (matchCount > 1 ? 'es' : '') + '</span>'
                    : '';

                div.innerHTML =
                    '<div><span class="lm-tag">' + lm.primary_tag + '</span>' +
                    '<span class="lm-conf ' + lm.confidence + '">' + lm.confidence + '</span> ' +
                    yawBadges + matchSpan + '</div>' +
                    '<div class="lm-desc">' + lm.description + '</div>' +
                    (lm.additional_tags.length > 0
                        ? '<div class="lm-desc" style="color:#999;">' + lm.additional_tags.join(', ') + '</div>'
                        : '');

                div.addEventListener('click', () => selectLandmark(li));
                el.appendChild(div);
            });

            // Summary
            if (selectedLandmark !== null && data.osm_landmarks) {
                const mc = data.osm_landmarks.filter(o => o.matches[selectedLandmark]).length;
                document.getElementById('match-summary').textContent =
                    '— ' + mc + ' / ' + data.osm_landmarks.length + ' OSM matches';
            } else {
                document.getElementById('match-summary').textContent = '';
            }
        }

        function selectLandmark(li) {
            selectedLandmark = selectedLandmark === li ? null : li;
            if (!currentData) return;
            renderLandmarkList(currentData);
            renderMap(currentData);
            renderOsmPanel(currentData);
            document.querySelectorAll('.pinhole-cell').forEach(cell => {
                drawBboxes(cell, parseInt(cell.dataset.yaw), currentData);
            });
        }

        // ---- Leaflet map ----
        function renderMap(data) {
            clearMapOverlays();

            // Panorama marker
            panoMarker = L.circleMarker([data.lat, data.lon], {
                radius: 8, fillColor: '#2196F3', fillOpacity: 1,
                color: '#fff', weight: 2
            }).addTo(map).bindPopup(
                '<b>' + data.pano_id + '</b><br>' +
                data.lat.toFixed(6) + ', ' + data.lon.toFixed(6)
            );
            map.setView([data.lat, data.lon], 18);

            // Draw cones for selected landmark
            if (selectedLandmark !== null && selectedLandmark < data.landmarks.length) {
                const lm = data.landmarks[selectedLandmark];
                const color = COLORS[selectedLandmark % COLORS.length];
                lm.cones_latlon.forEach(cone => {
                    const layer = L.polygon(cone, {
                        color: color, fillColor: color, fillOpacity: 0.2,
                        weight: 2, opacity: 0.8
                    }).addTo(map);
                    coneLayers.push(layer);
                });
            }

            // Draw OSM landmarks
            data.osm_landmarks.forEach((osm, oi) => {
                const isMatch = selectedLandmark !== null && osm.matches[selectedLandmark];
                const anyMatch = selectedLandmark === null;
                const color = isMatch ? '#4caf50' : (anyMatch ? '#5c6bc0' : '#bdbdbd');
                const opacity = isMatch ? 0.85 : (anyMatch ? 0.5 : 0.3);

                const layer = L.geoJSON(osm.geojson, {
                    style: {
                        color: color, fillColor: color, fillOpacity: opacity * 0.4,
                        weight: isMatch ? 3 : 1, opacity: opacity
                    },
                    pointToLayer: (f, ll) => L.circleMarker(ll, {
                        radius: isMatch ? 7 : 4,
                        fillColor: color, fillOpacity: opacity,
                        color: '#fff', weight: 1
                    })
                }).addTo(map);

                // Popup with sentence + tags
                let popupHtml = '';
                if (osm.sentence) {
                    popupHtml += '<div style="font-style:italic;margin-bottom:6px;color:#333;">' +
                        osm.sentence + '</div><hr style="margin:4px 0;">';
                }
                const tagLines = Object.entries(osm.tags)
                    .filter(([k]) => k !== 'osmid' && k !== 'custom_id')
                    .slice(0, 15)
                    .map(([k, v]) => '<b>' + k + '</b>: ' + v);
                popupHtml += tagLines.join('<br>');
                layer.bindPopup(popupHtml, {maxWidth: 350});

                osmGeoLayers.push(layer);
            });
        }

        // ---- OSM panel ----
        function renderOsmPanel(data) {
            document.getElementById('osm-count').textContent = data.osm_landmarks.length;
            const list = document.getElementById('osm-list');
            list.innerHTML = '';

            // Sort: matches first when a landmark is selected
            let items = data.osm_landmarks.map((o, i) => ({...o, _i: i}));
            if (selectedLandmark !== null) {
                items.sort((a, b) => (b.matches[selectedLandmark] ? 1 : 0) - (a.matches[selectedLandmark] ? 1 : 0));
            }

            items.forEach(osm => {
                const isMatch = selectedLandmark !== null && osm.matches[selectedLandmark];
                const div = document.createElement('div');
                div.className = 'osm-item ' + (selectedLandmark !== null ? (isMatch ? 'osm-match' : 'osm-nomatch') : '');

                // Show key tags
                const priority = ['name', 'amenity', 'building', 'shop', 'leisure',
                                  'highway', 'natural', 'landuse', 'tourism', 'railway',
                                  'man_made', 'barrier', 'emergency', 'public_transport'];
                const parts = [];
                for (const k of priority) {
                    if (osm.tags[k]) parts.push('<b>' + k + '</b>=' + osm.tags[k]);
                }
                if (parts.length === 0) {
                    for (const [k, v] of Object.entries(osm.tags).slice(0, 3)) {
                        if (k === 'osmid') continue;
                        parts.push('<b>' + k + '</b>=' + v);
                    }
                }

                let html = parts.join(' | ') || '<i>no tags</i>';
                if (osm.sentence) {
                    html += '<div style="font-style:italic;color:#555;font-size:11px;margin-top:2px;">' +
                        osm.sentence + '</div>';
                }
                div.innerHTML = html;
                div.addEventListener('click', () => {
                    map.setView([osm.centroid_lat, osm.centroid_lon], 19);
                    if (osmGeoLayers[osm._i]) osmGeoLayers[osm._i].openPopup();
                });
                list.appendChild(div);
            });
        }

        // Init
        fetch('/api/count').then(r => r.json()).then(d => {
            totalPanos = d.count;
            loadPanorama(0);
        });
    </script>
</body>
</html>
'''


# ---- Main ----

def main():
    global PANORAMAS, PINHOLE_DIR, OSM_DF, STRTREE, NEARBY_OSM, OSM_SENTENCES

    parser = argparse.ArgumentParser(description='Geometric filter viewer for pano_v2 landmarks')
    parser.add_argument('--pinhole_dir', type=str, required=True,
                        help='Directory containing pinhole images')
    parser.add_argument('--pano_pickle', type=str, required=True,
                        help='Path to pano_v2 embeddings pickle')
    parser.add_argument('--osm_feather', type=str, required=True,
                        help='Path to OSM landmarks feather file')
    parser.add_argument('--osm_sentences_dir', type=str, default=None,
                        help='Directory containing OSM landmark sentence JSONL files')
    parser.add_argument('--port', type=int, default=5002)

    args = parser.parse_args()
    PINHOLE_DIR = args.pinhole_dir
    startup_start = time.time()

    # Step 1: Load pano_v2 data
    print("=" * 60)
    print("STEP 1: Loading pano_v2 data")
    print("=" * 60)
    PANORAMAS = load_pano_v2_data(args.pano_pickle)

    # Step 2: Filter to panoramas with pinhole images
    print("\n" + "=" * 60)
    print("STEP 2: Filtering to panoramas with pinhole images")
    print("=" * 60)
    pinhole_path = Path(args.pinhole_dir)
    pinhole_subdirs = {d.name for d in pinhole_path.iterdir() if d.is_dir()}
    filtered = []
    for pano in PANORAMAS:
        if pano['pano_key'] not in pinhole_subdirs:
            continue
        subdir = pinhole_path / pano['pano_key']
        if all(
            (subdir / f'yaw_{yaw:03d}.jpg').exists() or
            (subdir / f'yaw_{yaw:03d}.png').exists()
            for yaw in [0, 90, 180, 270]
        ):
            filtered.append(pano)
    PANORAMAS = filtered
    print(f"  {len(PANORAMAS)} panoramas with pinhole images")

    # Step 3: Load OSM data
    print("\n" + "=" * 60)
    print("STEP 3: Loading OSM landmarks")
    print("=" * 60)
    OSM_DF = load_osm_data(args.osm_feather)

    # Step 4: Load OSM sentences (optional)
    if args.osm_sentences_dir:
        print("\n" + "=" * 60)
        print("STEP 4: Loading OSM sentences")
        print("=" * 60)
        OSM_SENTENCES = load_osm_sentences(args.osm_sentences_dir)
    else:
        print("\n  Skipping OSM sentences (no --osm_sentences_dir provided)")

    # Step 5: Spatial index + precompute nearby
    print("\n" + "=" * 60)
    print("STEP 5: Spatial index + nearby precomputation")
    print("=" * 60)
    STRTREE, NEARBY_OSM = precompute_nearby_osm(PANORAMAS, OSM_DF)

    print("\n" + "=" * 60)
    print(f"STARTUP COMPLETE in {time.time() - startup_start:.1f}s")
    print(f"  Panoramas: {len(PANORAMAS)}")
    print(f"  OSM landmarks: {len(OSM_DF)}")
    print(f"  http://localhost:{args.port}")
    print("=" * 60)

    app.run(debug=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
