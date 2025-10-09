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

app = Flask(__name__)

# Global data
PANORAMA_DATA = []
CURRENT_INDEX = 0

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
            <div class="pinhole-grid" id="pinhole-grid">
                <!-- Pinhole images will be inserted here -->
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
                                yawData.sentences.forEach(sentence => {
                                    html += `<li>${sentence}</li>`;
                                });
                                html += `</ul></div>`;
                            }
                        });

                        html += `</div>`;
                        div.innerHTML = html;
                        pinholeGrid.appendChild(div);
                    });
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


def load_sentence_data(sentence_dirs):
    """Load and parse JSONL sentence files from multiple directories."""
    sources = {}

    for sentence_dir in sentence_dirs:
        sentence_path = Path(sentence_dir)
        if not sentence_path.exists():
            print(f"Warning: Sentence directory not found: {sentence_dir}")
            continue

        source_name = sentence_path.name
        source_data = {}

        # Load all JSONL files in this directory
        for jsonl_file in sentence_path.glob('*.jsonl'):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        custom_id = entry['custom_id']

                        # Parse custom_id: {pano_id}_yaw_{angle}
                        # Note: pano_id might be just the ID or ID with partial/full coordinates
                        if '_yaw_' in custom_id:
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
                                        sentences = [lm['description'] for lm in landmarks]

                                        # Use base_id as the key for matching with panorama files
                                        if base_id not in source_data:
                                            source_data[base_id] = {}
                                        source_data[base_id][yaw] = sentences
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


@app.route('/api/panorama/<int:index>')
def get_panorama_data(index):
    global PANORAMA_DATA, SENTENCE_SOURCES

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
                    yaw_data.append({
                        'yaw': yaw,
                        'sentences': source_data[pano['id']][yaw]
                    })
            if yaw_data:
                sources_data.append({
                    'name': source_name,
                    'yaw_data': yaw_data
                })

    return jsonify({
        'name': pano['id'],
        'total': len(PANORAMA_DATA),
        'yaw_angles': pano['yaw_angles'],
        'sources': sources_data
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
    global PANORAMA_DATA, SENTENCE_SOURCES

    parser = argparse.ArgumentParser(description='Panorama viewer web app')
    parser.add_argument('--panorama_dir', type=str, required=True,
                       help='Directory containing panorama images (VIGOR format)')
    parser.add_argument('--pinhole_dir', type=str, required=True,
                       help='Directory containing pinhole image subdirectories')
    parser.add_argument('--sentence_dirs', type=str, nargs='+', required=True,
                       help='Directories containing JSONL sentence files (multiple sources)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the web server on')

    args = parser.parse_args()

    print("Loading sentence data...")
    SENTENCE_SOURCES = load_sentence_data(args.sentence_dirs)
    print(f"Loaded {len(SENTENCE_SOURCES)} sentence sources")

    print("Finding common panoramas...")
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
