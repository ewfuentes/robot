#!/usr/bin/env python3
"""
Web viewer to compare panorama landmark extraction across different prompting approaches.

This viewer is approach-agnostic and can handle any number of response files (>= 1).
Approach names are automatically derived from the JSONL filenames.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:compare_panorama_prompts_viewer -- \
        --pinhole_dir /data/overhead_matching/datasets/VIGOR/Chicago/pinhole_panorama_crops \
        --response_files /path/to/approach1.jsonl /path/to/approach2.jsonl [...] \
        --port 5001
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set

from flask import Flask, jsonify, send_file

app = Flask(__name__)

# Global data stores
PINHOLE_DIR = None
PANORAMA_DIR = None
APPROACHES = []  # List of (approach_name, response_data, token_stats) tuples
COMMON_PANO_IDS = []  # List of panorama IDs that exist in all approaches
YAW_ANGLES = [0, 90, 180, 270]


def derive_approach_name(filepath: str) -> str:
    """Derive human-readable approach name from filename.

    Examples:
        "with_vibe_responses.jsonl" -> "With Vibe"
        "bounding_boxes_responses.jsonl" -> "Bounding Boxes"
        "/path/to/robotics_responses.jsonl" -> "Robotics"
    """
    basename = os.path.basename(filepath)
    # Remove .jsonl extension
    name = basename.replace('.jsonl', '')
    # Remove common suffixes
    name = name.replace('_responses', '').replace('_response', '')
    # Convert underscores to spaces and title case
    name = name.replace('_', ' ').title()
    return name


def parse_text_response(text: str) -> Dict:
    """Fallback parser for non-JSON responses.

    Since all responses should now be JSON, this is just a fallback
    that returns empty data with a warning.
    """
    print(f"Warning: Failed to parse response as JSON. Response text: {text[:200]}...")
    return {
        'general_vibe': '',
        'landmarks': []
    }


def parse_response(response_obj: Any) -> Dict:
    """Parse response from JSONL, handling different formats.

    The response might be:
    - A JSON string
    - A dict with 'response' key containing JSON string or text
    - A dict with the content directly
    """
    response_text = None

    if isinstance(response_obj, str):
        response_text = response_obj
    elif 'response' in response_obj:
        response_text = response_obj['response']
    else:
        # Already parsed dict
        return response_obj

    # Try to parse as JSON first
    if response_text:
        try:
            content = json.loads(response_text)
            return content
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, try text parsing
            return parse_text_response(response_text)

    return {'general_vibe': '', 'landmarks': []}


def parse_custom_id(custom_id: str) -> str:
    """Extract panorama ID from custom_id.

    custom_id format: "pano_id,lat,lon,_yaw_270"
    Returns: "pano_id,lat,lon," (including coordinates and trailing comma)
    """
    # Remove the yaw part
    if '_yaw_' in custom_id:
        # Split at _yaw_ and keep everything before it (which already has trailing comma)
        pano_id = custom_id.rsplit('_yaw_', 1)[0]
        # Ensure trailing comma
        if not pano_id.endswith(','):
            pano_id += ','
    else:
        pano_id = custom_id
    return pano_id


def detect_format(record: Dict) -> str:
    """Detect which format a response record uses.

    Returns: 'current', 'gemini_batch', or 'openai_batch'
    """
    # Check for new Gemini batch format (has 'key' field and nested response)
    if 'key' in record and isinstance(record.get('response'), dict):
        if 'candidates' in record['response']:
            return 'gemini_batch'

    # Check for old OpenAI batch format (has 'body' in response)
    if 'response' in record and isinstance(record['response'], dict):
        if 'body' in record['response']:
            return 'openai_batch'

    # Current format (custom_id + string response)
    return 'current'


def extract_response_data(record: Dict, format_type: str) -> tuple:
    """Extract panorama_id, landmarks, and token_usage from a record.

    Returns: (pano_id, parsed_content, token_usage)
    """
    if format_type == 'current':
        pano_id = parse_custom_id(record['custom_id'])
        content = parse_response(record['response'])
        tokens = {
            'prompt': record.get('usage_metadata', {}).get('prompt_token_count', 0),
            'completion': record.get('usage_metadata', {}).get('candidates_token_count', 0),
            'total': record.get('usage_metadata', {}).get('total_token_count', 0)
        }

    elif format_type == 'gemini_batch':
        pano_id = parse_custom_id(record['key'])
        text = record['response']['candidates'][0]['content']['parts'][0]['text']
        content = parse_response(text)
        usage = record['response'].get('usageMetadata', {})
        tokens = {
            'prompt': usage.get('promptTokenCount', 0),
            'completion': usage.get('candidatesTokenCount', 0),
            'total': usage.get('totalTokenCount', 0)
        }

    elif format_type == 'openai_batch':
        pano_id = parse_custom_id(record['custom_id'])
        text = record['response']['body']['choices'][0]['message']['content']
        content = parse_response(text)
        usage = record['response']['body'].get('usage', {})
        tokens = {
            'prompt': usage.get('prompt_tokens', 0),
            'completion': usage.get('completion_tokens', 0),
            'total': usage.get('total_tokens', 0)
        }
        # Old format doesn't have general_vibe or location_type - add empty string
        if 'general_vibe' not in content and 'location_type' not in content:
            content['general_vibe'] = ''

    return pano_id, content, tokens


def load_response_directory(directory_path: str) -> tuple:
    """Load all JSONL files from a directory and combine them.

    Used for old format with 89 separate files.

    Returns: (data_dict, token_stats_dict)
    """
    combined_data = {}
    combined_tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    dir_path = Path(directory_path)
    files = sorted(dir_path.glob('file-*'))

    print(f"  Loading {len(files)} files from directory...")

    for file_path in files:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                format_type = detect_format(record)
                pano_id, content, tokens = extract_response_data(record, format_type)

                # Initialize pano_data for this panorama if needed
                if pano_id not in combined_data:
                    combined_data[pano_id] = {
                        'general_vibe': content.get('general_vibe') or content.get('location_type', ''),
                        'landmarks': content.get('landmarks', []),
                        'yaw_data': {}
                    }

                combined_tokens['prompt'] += tokens['prompt']
                combined_tokens['completion'] += tokens['completion']
                combined_tokens['total'] += tokens['total']

    print(f"  Loaded {len(combined_data)} panoramas from directory")
    return combined_data, combined_tokens


def load_response_data(response_file_path: str) -> tuple:
    """Load response data from JSONL file or directory.

    Returns: (data_dict, token_stats_dict)
    """
    print(f"Loading {response_file_path}...")

    # Check if it's a directory (old format)
    if os.path.isdir(response_file_path):
        return load_response_directory(response_file_path)

    # Load JSONL file
    pano_data = {}
    tokens = {'prompt': 0, 'completion': 0, 'total': 0}

    with open(response_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            format_type = detect_format(record)

            try:
                pano_id, content, token_usage = extract_response_data(record, format_type)
            except Exception as e:
                print(f"Error parsing response: {e}")
                continue

            # Initialize pano_data for this panorama if needed
            if pano_id not in pano_data:
                pano_data[pano_id] = {
                    'general_vibe': content.get('general_vibe') or content.get('location_type', ''),
                    'landmarks': content.get('landmarks', []),
                    'yaw_data': {}
                }

            # Aggregate tokens
            tokens['prompt'] += token_usage['prompt']
            tokens['completion'] += token_usage['completion']
            tokens['total'] += token_usage['total']

    print(f"  Loaded {len(pano_data)} panoramas")
    if len(pano_data) > 0:
        first_id = list(pano_data.keys())[0]
        first_data = pano_data[first_id]
        print(f"    First pano ID: {first_id}")
        print(f"    General vibe: {first_data.get('general_vibe', '')[:50]}")
        print(f"    Landmarks count: {len(first_data.get('landmarks', []))}")

    return pano_data, tokens


def load_all_approaches(response_files: List[str]) -> List[tuple]:
    """Load all approach data.

    Returns:
        List of (approach_name, response_data, token_stats) tuples
    """
    approaches = []
    for filepath in response_files:
        approach_name = derive_approach_name(filepath)
        response_data, token_stats = load_response_data(filepath)
        approaches.append((approach_name, response_data, token_stats))
        print(f"  Loaded approach: {approach_name}")

    return approaches


def find_common_panoramas(approaches: List[tuple], pinhole_dir: str) -> List[str]:
    """Find panoramas that exist in all approaches and have all 4 pinhole images.

    Returns:
        List of panorama IDs
    """
    # Find intersection of all approach panorama sets
    pano_sets = [set(response_data.keys()) for _, response_data, _ in approaches]
    common_panos = set.intersection(*pano_sets) if pano_sets else set()

    print(f"Found {len(common_panos)} panoramas common to all approaches")

    # Filter to only those with all 4 pinhole images
    filtered_panos = []
    pinhole_path = Path(pinhole_dir)

    for pano_id in common_panos:
        # Check if all 4 yaw images exist
        has_all_yaws = True
        for yaw in YAW_ANGLES:
            # Check common pinhole image path patterns
            possible_paths = [
                pinhole_path / pano_id / f"yaw_{yaw:03d}.jpg",  # yaw_000.jpg
                pinhole_path / pano_id / f"yaw_{yaw}.jpg",
                pinhole_path / pano_id / f"yaw_{yaw}.png",
                pinhole_path / f"{pano_id}_yaw_{yaw:03d}.jpg",
                pinhole_path / f"{pano_id}_yaw_{yaw}.jpg",
                pinhole_path / f"{pano_id}_yaw_{yaw}.png",
            ]

            if not any(p.exists() for p in possible_paths):
                has_all_yaws = False
                break

        if has_all_yaws:
            filtered_panos.append(pano_id)

    filtered_panos.sort()
    print(f"Filtered to {len(filtered_panos)} panoramas with all 4 yaw images")

    return filtered_panos


def get_approach_color(index: int, total: int) -> str:
    """Generate distinguishable colors for approaches."""
    palette = [
        '#e3f2fd',  # Light blue
        '#e8f5e9',  # Light green
        '#fff9c4',  # Light yellow
        '#fce4ec',  # Light pink
        '#f3e5f5',  # Light purple
        '#e0f2f1',  # Light teal
    ]
    return palette[index % len(palette)]


def filter_landmarks_for_yaw(landmarks: List[Dict], yaw: int) -> List[Dict]:
    """Filter landmarks that are visible at the given yaw angle.

    Automatically detects whether to use bounding_boxes or yaw_angles
    based on the landmark structure.
    """
    filtered = []
    yaw_str = str(yaw)
    yaw_int = int(yaw)

    for lm in landmarks:
        # Check if landmark has bounding boxes
        if 'bounding_boxes' in lm and lm['bounding_boxes']:
            # Filter by bboxes with matching yaw_angle (handle both string and int)
            if any(bbox.get('yaw_angle') == yaw_str or bbox.get('yaw_angle') == yaw_int for bbox in lm['bounding_boxes']):
                filtered.append(lm)
        # Otherwise check yaw_angles list (handle both string and int arrays)
        elif 'yaw_angles' in lm and (yaw_str in lm['yaw_angles'] or yaw_int in lm['yaw_angles']):
            filtered.append(lm)

    return filtered


def get_pinhole_image_path(pano_id: str, yaw: int) -> Path:
    """Find the pinhole image path for a given panorama and yaw."""
    pinhole_path = Path(PINHOLE_DIR)

    # Try different path patterns
    possible_paths = [
        pinhole_path / pano_id / f"yaw_{yaw:03d}.jpg",  # yaw_000.jpg, yaw_090.jpg, etc.
        pinhole_path / pano_id / f"yaw_{yaw}.jpg",
        pinhole_path / pano_id / f"yaw_{yaw}.png",
        pinhole_path / f"{pano_id}_yaw_{yaw:03d}.jpg",
        pinhole_path / f"{pano_id}_yaw_{yaw}.jpg",
        pinhole_path / f"{pano_id}_yaw_{yaw}.png",
    ]

    for p in possible_paths:
        if p.exists():
            return p

    raise FileNotFoundError(f"No pinhole image found for {pano_id} yaw {yaw}")


# Flask routes

@app.route('/')
def index():
    """Serve main HTML page."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Panorama Prompt Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
        }
        .nav-buttons button {
            margin: 0 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
        }
        .nav-buttons button:hover {
            background-color: #45a049;
        }
        .pinhole-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .pinhole-item {
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #fafafa;
        }
        .pinhole-item h3 {
            margin-top: 0;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }
        .pinhole-item img {
            width: 100%;
            height: auto;
            display: block;
            margin-bottom: 10px;
            border: 1px solid #ccc;
        }
        .image-container {
            position: relative;
            width: 100%;
            margin-bottom: 10px;
            max-width: 350px;
            margin-left: auto;
            margin-right: auto;
        }
        .image-container img {
            display: block;
            width: 100%;
            height: auto;
            border: 1px solid #ccc;
        }
        .image-container canvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        .landmark-color-marker {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 2px;
            margin-right: 4px;
            border: 1px solid rgba(0,0,0,0.3);
            vertical-align: middle;
        }
        .landmark-item {
            cursor: pointer;
            padding: 2px;
            border-radius: 2px;
            transition: background-color 0.2s;
        }
        .landmark-item:hover {
            background-color: rgba(0,0,0,0.05);
        }
        .landmark-item.highlighted {
            background-color: rgba(255,255,0,0.3);
        }
        .bbox-coords {
            font-size: 8px;
            color: #666;
            margin-left: 4px;
            font-family: monospace;
        }
        .approach-results {
            margin-top: 10px;
        }
        .approach-section {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 3px;
            border: 1px solid #ccc;
        }
        .approach-section h4 {
            margin: 0 0 5px 0;
            color: #333;
            font-size: 12px;
        }
        .general-vibe {
            font-weight: bold;
            font-size: 11px;
            margin-bottom: 6px;
            padding: 4px;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 2px;
        }
        .landmarks {
            list-style-type: none;
            padding: 0;
            margin: 0;
            font-size: 10px;
        }
        .landmarks li {
            margin-bottom: 4px;
            padding-left: 12px;
            position: relative;
            line-height: 1.3;
        }
        .landmarks li:before {
            content: "•";
            position: absolute;
            left: 0;
        }
        .proper-nouns {
            color: #0066cc;
            font-weight: 500;
            font-size: 9px;
        }
        .bbox-info {
            color: #666;
            font-size: 9px;
            margin-left: 8px;
        }
        .panorama-section {
            margin-top: 30px;
            border-top: 2px solid #333;
            padding-top: 20px;
        }
        .panorama-section h2 {
            margin-top: 0;
        }
        #panorama-img {
            width: 100%;
            max-width: 1200px;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #ccc;
        }
        .pano-info {
            text-align: center;
            color: #666;
            margin-top: 10px;
        }
        .token-summary {
            background-color: #f5f5f5;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        .token-summary h2 {
            margin: 0 0 10px 0;
            font-size: 16px;
        }
        .token-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .token-box {
            border: 1px solid #ddd;
            padding: 12px;
            border-radius: 5px;
            background-color: white;
        }
        .token-box h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
        }
        .token-stat {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 12px;
        }
        .token-value {
            font-weight: bold;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Panorama Prompt Comparison</h1>
                <div class="pano-info" id="pano-info"></div>
            </div>
            <div class="nav-buttons">
                <button onclick="navigate(-1)">← Previous</button>
                <button onclick="navigate(1)">Next →</button>
            </div>
        </div>

        <div class="token-summary">
            <h2>Token Usage Summary</h2>
            <div class="token-grid" id="token-grid">
                <!-- Populated by JavaScript -->
            </div>
        </div>

        <div class="pinhole-grid" id="pinhole-grid">
            <!-- Will be populated by JavaScript -->
        </div>

        <div class="panorama-section" id="panorama-section" style="display: none;">
            <h2>Full Panorama</h2>
            <img id="panorama-img" src="" alt="Panorama">
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let totalPanoramas = 0;

        // Color palette for bounding boxes
        const LANDMARK_COLORS = [
            '#FF6B6B', // Red
            '#4ECDC4', // Teal
            '#45B7D1', // Blue
            '#FFA07A', // Light Salmon
            '#98D8C8', // Mint
            '#F7DC6F', // Yellow
            '#BB8FCE', // Purple
            '#85C1E2', // Sky Blue
            '#F8B739', // Orange
            '#52C45A', // Green
        ];

        function getLandmarkColor(index) {
            return LANDMARK_COLORS[index % LANDMARK_COLORS.length];
        }

        function drawBoundingBoxes(canvas, bboxes, imageWidth, imageHeight, highlightedIndex = -1) {
            const ctx = canvas.getContext('2d');
            canvas.width = imageWidth;
            canvas.height = imageHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            bboxes.forEach(({bbox, color, landmarkIndex}, idx) => {
                // Convert from 0-1000 normalized coords to actual pixels
                const x1 = (bbox.xmin / 1000) * imageWidth;
                const y1 = (bbox.ymin / 1000) * imageHeight;
                const x2 = (bbox.xmax / 1000) * imageWidth;
                const y2 = (bbox.ymax / 1000) * imageHeight;

                const isHighlighted = (highlightedIndex >= 0 && landmarkIndex === highlightedIndex);

                // Draw rectangle
                ctx.strokeStyle = color;
                ctx.lineWidth = isHighlighted ? 5 : 3;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Draw semi-transparent fill
                ctx.fillStyle = color + (isHighlighted ? '60' : '30');  // More opaque when highlighted
                ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

                // Draw coordinates if highlighted
                if (isHighlighted) {
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = 'black';
                    ctx.lineWidth = 3;
                    ctx.font = '14px monospace';
                    const text = `(${bbox.xmin},${bbox.ymin})-(${bbox.xmax},${bbox.ymax})`;
                    const textWidth = ctx.measureText(text).width;
                    const textX = x1 + 5;
                    const textY = y1 + 20;

                    // Draw background
                    ctx.strokeRect(textX - 2, textY - 15, textWidth + 4, 20);
                    ctx.fillStyle = color;
                    ctx.fillRect(textX - 2, textY - 15, textWidth + 4, 20);

                    // Draw text
                    ctx.fillStyle = 'white';
                    ctx.fillText(text, textX, textY);
                }
            });
        }

        function loadPanorama(index) {
            fetch(`/api/panorama/${index}`)
                .then(response => response.json())
                .then(data => {
                    currentIndex = index;
                    totalPanoramas = data.total;

                    // Update info
                    document.getElementById('pano-info').textContent =
                        `Panorama ${index + 1} of ${totalPanoramas} (ID: ${data.pano_id})`;

                    // Update token summary
                    const tokenGrid = document.getElementById('token-grid');
                    tokenGrid.innerHTML = '';
                    data.approaches.forEach(approach => {
                        const box = document.createElement('div');
                        box.className = 'token-box';
                        box.style.backgroundColor = approach.color;
                        box.innerHTML = `
                            <h3>${approach.name}</h3>
                            <div class="token-stat">
                                <span>Prompt tokens:</span>
                                <span class="token-value">${approach.tokens.prompt.toLocaleString()}</span>
                            </div>
                            <div class="token-stat">
                                <span>Completion tokens:</span>
                                <span class="token-value">${approach.tokens.completion.toLocaleString()}</span>
                            </div>
                            <div class="token-stat">
                                <span>Total tokens:</span>
                                <span class="token-value">${approach.tokens.total.toLocaleString()}</span>
                            </div>
                        `;
                        tokenGrid.appendChild(box);
                    });

                    // Update pinhole grid
                    const grid = document.getElementById('pinhole-grid');
                    grid.innerHTML = '';

                    data.yaw_angles.forEach(yaw => {
                        const item = document.createElement('div');
                        item.className = 'pinhole-item';

                        let html = `<h3>Yaw ${yaw}°</h3>`;

                        // Create image container with canvas overlay
                        const imageId = `image-${index}-${yaw}`;
                        const canvasId = `canvas-${index}-${yaw}`;
                        html += '<div class="image-container">';
                        html += `<img id="${imageId}" src="/api/image/pinhole/${index}/${yaw}" alt="Yaw ${yaw}">`;
                        html += `<canvas id="${canvasId}"></canvas>`;
                        html += '</div>';

                        html += '<div class="approach-results">';

                        // Collect all bboxes for this yaw across all approaches
                        const allBboxesForYaw = [];
                        let landmarkIndex = 0;

                        data.approaches.forEach((approach, i) => {
                            html += `<div class="approach-section" style="background-color: ${approach.color}">`;
                            html += `<h4>${approach.name}</h4>`;
                            html += `<div class="general-vibe">${approach.general_vibe || 'N/A'}</div>`;

                            const landmarks = filterLandmarksForYaw(approach.landmarks, yaw);
                            if (landmarks.length > 0) {
                                html += '<ul class="landmarks">';
                                landmarks.forEach(lm => {
                                    const lmColor = getLandmarkColor(landmarkIndex);
                                    const currentLandmarkIndex = landmarkIndex;

                                    html += `<li class="landmark-item" data-landmark-index="${currentLandmarkIndex}" data-canvas-id="${canvasId}">`;
                                    html += `<span class="landmark-color-marker" style="background-color: ${lmColor}"></span>`;
                                    html += lm.description;

                                    // Show proper nouns if available
                                    if (lm.proper_nouns && lm.proper_nouns.length > 0) {
                                        html += ` <span class="proper-nouns">[${lm.proper_nouns.join(', ')}]</span>`;
                                    }

                                    // Collect bounding boxes for drawing and show coordinates
                                    if (lm.bounding_boxes) {
                                        const yawBboxes = lm.bounding_boxes.filter(b => b.yaw_angle == yaw);
                                        if (yawBboxes.length > 0) {
                                            yawBboxes.forEach(bbox => {
                                                allBboxesForYaw.push({bbox, color: lmColor, landmarkIndex: currentLandmarkIndex});
                                                html += ` <span class="bbox-coords">[${bbox.xmin},${bbox.ymin} → ${bbox.xmax},${bbox.ymax}]</span>`;
                                            });
                                        }
                                    }

                                    html += '</li>';
                                    landmarkIndex++;
                                });
                                html += '</ul>';
                            } else {
                                html += '<div style="color: #999; font-size: 12px;">No landmarks detected at this yaw</div>';
                            }

                            html += '</div>';
                        });

                        html += '</div>';
                        item.innerHTML = html;
                        item.dataset.bboxes = JSON.stringify(allBboxesForYaw);
                        item.dataset.imageId = imageId;
                        item.dataset.canvasId = canvasId;
                        grid.appendChild(item);
                    });

                    // Draw bounding boxes after images load
                    setTimeout(() => {
                        document.querySelectorAll('.pinhole-item').forEach(item => {
                            const imageId = item.dataset.imageId;
                            const canvasId = item.dataset.canvasId;
                            const bboxes = JSON.parse(item.dataset.bboxes || '[]');

                            if (bboxes.length > 0) {
                                const img = document.getElementById(imageId);
                                const canvas = document.getElementById(canvasId);

                                const redrawBoxes = (highlightedIndex = -1) => {
                                    // Use getBoundingClientRect to get the actual rendered size
                                    const rect = img.getBoundingClientRect();
                                    drawBoundingBoxes(canvas, bboxes, rect.width, rect.height, highlightedIndex);
                                };

                                if (img && canvas && img.complete) {
                                    redrawBoxes();
                                } else if (img && canvas) {
                                    img.onload = () => redrawBoxes();
                                }

                                // Store redraw function for hover handlers
                                item.dataset.redrawFunc = 'redrawBoxes';
                                item.redrawBoxes = redrawBoxes;
                            }
                        });

                        // Add hover handlers to landmark items
                        document.querySelectorAll('.landmark-item').forEach(landmarkItem => {
                            const landmarkIndex = parseInt(landmarkItem.dataset.landmarkIndex);
                            const canvasId = landmarkItem.dataset.canvasId;

                            landmarkItem.addEventListener('mouseenter', () => {
                                // Find the pinhole item that contains this canvas
                                const pinholeItem = document.querySelector(`[data-canvas-id="${canvasId}"]`);
                                if (pinholeItem && pinholeItem.redrawBoxes) {
                                    pinholeItem.redrawBoxes(landmarkIndex);
                                }
                                landmarkItem.classList.add('highlighted');
                            });

                            landmarkItem.addEventListener('mouseleave', () => {
                                const pinholeItem = document.querySelector(`[data-canvas-id="${canvasId}"]`);
                                if (pinholeItem && pinholeItem.redrawBoxes) {
                                    pinholeItem.redrawBoxes(-1);
                                }
                                landmarkItem.classList.remove('highlighted');
                            });
                        });
                    }, 100);

                    // Show panorama section if panorama image available
                    // For now, hide it since we may not have panorama images
                    document.getElementById('panorama-section').style.display = 'none';
                })
                .catch(error => {
                    console.error('Error loading panorama:', error);
                    alert('Error loading panorama data');
                });
        }

        function filterLandmarksForYaw(landmarks, yaw) {
            const yawStr = String(yaw);
            const yawNum = Number(yaw);
            return landmarks.filter(lm => {
                // Check bounding boxes
                if (lm.bounding_boxes && lm.bounding_boxes.length > 0) {
                    return lm.bounding_boxes.some(bbox => bbox.yaw_angle == yawStr || bbox.yaw_angle == yawNum);
                }
                // Check yaw_angles (handle both string and number arrays)
                if (lm.yaw_angles) {
                    return lm.yaw_angles.includes(yawStr) || lm.yaw_angles.includes(yawNum);
                }
                return false;
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
    """
    return html


@app.route('/api/panorama/<int:index>')
def get_panorama_data(index: int):
    """Get all data for a panorama including all approaches."""
    if index < 0 or index >= len(COMMON_PANO_IDS):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = COMMON_PANO_IDS[index]

    # Gather data from all approaches
    approach_data = []
    for i, (approach_name, response_data, token_stats) in enumerate(APPROACHES):
        pano_data = response_data.get(pano_id, {})
        landmarks = pano_data.get('landmarks', [])

        approach_data.append({
            'name': approach_name,
            'general_vibe': pano_data.get('general_vibe', ''),
            'landmarks': landmarks,
            'color': get_approach_color(i, len(APPROACHES)),
            'tokens': token_stats  # Include token statistics
        })

    return jsonify({
        'pano_id': pano_id,
        'total': len(COMMON_PANO_IDS),
        'yaw_angles': YAW_ANGLES,
        'approaches': approach_data
    })


@app.route('/api/image/pinhole/<int:index>/<int:yaw>')
def get_pinhole_image(index: int, yaw: int):
    """Serve pinhole image for specific yaw."""
    if index < 0 or index >= len(COMMON_PANO_IDS):
        return jsonify({'error': 'Invalid index'}), 404

    pano_id = COMMON_PANO_IDS[index]

    try:
        image_path = get_pinhole_image_path(pano_id, yaw)
        return send_file(image_path, mimetype='image/jpeg')
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404


def main():
    parser = argparse.ArgumentParser(
        description='Web viewer to compare panorama landmark extraction across different prompting approaches'
    )
    parser.add_argument(
        '--pinhole_dir',
        type=str,
        required=True,
        help='Directory containing pinhole image subdirectories'
    )
    parser.add_argument(
        '--panorama_dir',
        type=str,
        required=False,
        help='Directory containing full panorama images (optional, not currently used)'
    )
    parser.add_argument(
        '--response_files',
        nargs='+',
        required=True,
        help='One or more response JSONL files to compare'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run web server on (default: 5000)'
    )

    args = parser.parse_args()

    # Set global variables
    global PINHOLE_DIR, PANORAMA_DIR, APPROACHES, COMMON_PANO_IDS

    PINHOLE_DIR = args.pinhole_dir
    PANORAMA_DIR = args.panorama_dir

    # Load all approaches
    print(f"\nLoading {len(args.response_files)} approach(es)...")
    APPROACHES = load_all_approaches(args.response_files)

    # Find common panoramas
    print("\nFinding common panoramas...")
    COMMON_PANO_IDS = find_common_panoramas(APPROACHES, PINHOLE_DIR)

    if len(COMMON_PANO_IDS) == 0:
        print("\nERROR: No panoramas found that exist in all approaches with all 4 yaw images!")
        print("Please check:")
        print("  1. Response files have overlapping panorama IDs")
        print("  2. Pinhole directory contains the expected images")
        return

    print(f"\nReady to compare {len(APPROACHES)} approach(es) on {len(COMMON_PANO_IDS)} panoramas")
    print(f"\nStarting server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    app.run(host='0.0.0.0', port=args.port, debug=True)


if __name__ == '__main__':
    main()
