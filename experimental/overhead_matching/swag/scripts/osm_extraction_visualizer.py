"""
Web visualizer for OSM tag extraction results.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:osm_extraction_visualizer -- \
        --predictions_file /path/to/predictions.jsonl \
        --images_dir /path/to/pinhole_images \
        --port 8080
"""

import http.server
import socketserver
import json
import urllib.parse
from pathlib import Path
import argparse
import base64
import re

# Global config set by CLI args
CONFIG = {
    'predictions_file': None,
    'images_dir': None,
    'predictions': [],
}


def load_predictions(predictions_file: Path) -> list[dict]:
    """Load and parse predictions from JSONL file."""
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            key = data['key']

            # Extract the response text (Gemini format)
            try:
                response_text = data['response']['candidates'][0]['content']['parts'][0]['text']
                parsed_response = json.loads(response_text)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to parse response for {key}: {e}")
                continue

            predictions.append({
                'key': key,
                'location_type': parsed_response.get('location_type', 'unknown'),
                'landmarks': parsed_response.get('landmarks', []),
            })

    return predictions


def get_confidence_color(confidence: str) -> str:
    """Get color for confidence level."""
    colors = {
        'high': '#22c55e',    # green
        'medium': '#eab308',  # yellow
        'low': '#ef4444',     # red
    }
    return colors.get(confidence.lower(), '#6b7280')


def parse_pano_key(key: str) -> tuple[str, float, float]:
    """Parse panorama key into (pano_id, lat, lon)."""
    parts = key.rstrip(',').split(',')
    pano_id = parts[0]
    lat = float(parts[1]) if len(parts) > 1 else 0.0
    lon = float(parts[2]) if len(parts) > 2 else 0.0
    return pano_id, lat, lon


def generate_index_html() -> str:
    """Generate the main index page with map and search."""
    predictions = CONFIG['predictions']

    # Build markers data for JavaScript
    markers_data = []
    for idx, pred in enumerate(predictions):
        pano_id, lat, lon = parse_pano_key(pred['key'])
        landmark_count = len(pred['landmarks'])
        high_conf = sum(1 for lm in pred['landmarks'] if lm.get('confidence', '').lower() == 'high')
        med_conf = sum(1 for lm in pred['landmarks'] if lm.get('confidence', '').lower() == 'medium')
        low_conf = sum(1 for lm in pred['landmarks'] if lm.get('confidence', '').lower() == 'low')

        markers_data.append({
            'idx': idx,
            'pano_id': pano_id,
            'key': pred['key'],
            'lat': lat,
            'lon': lon,
            'location_type': pred['location_type'],
            'landmark_count': landmark_count,
            'high_conf': high_conf,
            'med_conf': med_conf,
            'low_conf': low_conf,
        })

    markers_json = json.dumps(markers_data)

    # Calculate center of all panoramas
    if markers_data:
        center_lat = sum(m['lat'] for m in markers_data) / len(markers_data)
        center_lon = sum(m['lon'] for m in markers_data) / len(markers_data)
    else:
        center_lat, center_lon = 41.88, -87.63  # Default to Chicago

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSM Extraction Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #map {{ height: calc(100vh - 180px); min-height: 500px; }}
        .search-result {{
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid #e5e7eb;
        }}
        .search-result:hover {{
            background-color: #f3f4f6;
        }}
        .search-results {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #d1d5db;
            border-top: none;
            border-radius: 0 0 8px 8px;
            max-height: 300px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }}
        .search-container {{
            position: relative;
        }}
    </style>
</head>
<body class="bg-gray-100">
    <div class="bg-white shadow-sm border-b">
        <div class="container mx-auto px-4 py-4">
            <div class="flex justify-between items-center flex-wrap gap-4">
                <div>
                    <h1 class="text-2xl font-bold">OSM Tag Extraction Results</h1>
                    <p class="text-gray-600">{len(predictions)} panoramas</p>
                </div>
                <div class="search-container w-full md:w-96">
                    <input type="text" id="search-input"
                           placeholder="Search by panorama ID..."
                           class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <div id="search-results" class="search-results"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="container mx-auto px-4 py-4">
        <div class="bg-white rounded-lg shadow overflow-hidden">
            <div id="map"></div>
        </div>
        <div class="mt-2 text-sm text-gray-500">
            Click a marker to view panorama details. Use scroll to zoom, drag to pan.
        </div>
    </div>

    <script>
        const markersData = {markers_json};

        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);

        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        // Create markers
        const markers = {{}};
        const markerGroup = L.featureGroup();

        markersData.forEach(data => {{
            const color = data.high_conf > data.med_conf && data.high_conf > data.low_conf ? '#22c55e' :
                          data.med_conf > data.low_conf ? '#eab308' : '#ef4444';

            const marker = L.circleMarker([data.lat, data.lon], {{
                radius: 8,
                fillColor: color,
                color: '#ffffff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }});

            marker.bindPopup(`
                <div style="min-width: 200px;">
                    <div style="font-weight: bold; margin-bottom: 4px;">${{data.pano_id.substring(0, 20)}}...</div>
                    <div style="font-size: 12px; color: #666; margin-bottom: 8px;">${{data.location_type}}</div>
                    <div style="font-size: 12px; margin-bottom: 8px;">
                        <strong>${{data.landmark_count}}</strong> landmarks
                        (<span style="color: #22c55e;">${{data.high_conf}}H</span> /
                        <span style="color: #eab308;">${{data.med_conf}}M</span> /
                        <span style="color: #ef4444;">${{data.low_conf}}L</span>)
                    </div>
                    <a href="/view/${{data.idx}}"
                       style="display: inline-block; padding: 6px 12px; background: #3b82f6; color: white; text-decoration: none; border-radius: 4px; font-size: 12px;">
                        View Details
                    </a>
                </div>
            `);

            marker.addTo(markerGroup);
            markers[data.pano_id] = {{ marker, data }};
        }});

        markerGroup.addTo(map);

        // Fit map to markers
        if (markersData.length > 0) {{
            map.fitBounds(markerGroup.getBounds(), {{ padding: [20, 20] }});
        }}

        // Search functionality
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase().trim();

            if (query.length < 2) {{
                searchResults.style.display = 'none';
                return;
            }}

            const matches = markersData.filter(d =>
                d.pano_id.toLowerCase().includes(query) ||
                d.key.toLowerCase().includes(query)
            ).slice(0, 10);

            if (matches.length === 0) {{
                searchResults.innerHTML = '<div class="search-result text-gray-500">No results found</div>';
            }} else {{
                searchResults.innerHTML = matches.map(m => `
                    <div class="search-result" data-idx="${{m.idx}}" data-pano-id="${{m.pano_id}}">
                        <div class="font-medium text-sm">${{m.pano_id}}</div>
                        <div class="text-xs text-gray-500">${{m.location_type}} - ${{m.landmark_count}} landmarks</div>
                    </div>
                `).join('');
            }}

            searchResults.style.display = 'block';
        }});

        searchResults.addEventListener('click', (e) => {{
            const result = e.target.closest('.search-result');
            if (result && result.dataset.panoId) {{
                const panoId = result.dataset.panoId;
                const markerInfo = markers[panoId];

                if (markerInfo) {{
                    map.setView(markerInfo.marker.getLatLng(), 18);
                    markerInfo.marker.openPopup();
                }}

                searchResults.style.display = 'none';
                searchInput.value = panoId;
            }}
        }});

        // Hide search results when clicking outside
        document.addEventListener('click', (e) => {{
            if (!e.target.closest('.search-container')) {{
                searchResults.style.display = 'none';
            }}
        }});

        // Handle Enter key in search
        searchInput.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') {{
                const firstResult = searchResults.querySelector('.search-result[data-pano-id]');
                if (firstResult) {{
                    firstResult.click();
                }}
            }}
        }});
    </script>
</body>
</html>'''


def generate_view_html(idx: int) -> str:
    """Generate the detailed view page for a single panorama."""
    predictions = CONFIG['predictions']
    images_dir = CONFIG['images_dir']

    if idx < 0 or idx >= len(predictions):
        return "<h1>Not found</h1>"

    pred = predictions[idx]
    pano_key = pred['key']

    # Parse lat/lon from key
    pano_id, lat, lon = parse_pano_key(pano_key)
    google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"

    # Build image paths
    pano_folder = images_dir / pano_key
    yaw_images = {
        0: pano_folder / 'yaw_000.jpg',
        90: pano_folder / 'yaw_090.jpg',
        180: pano_folder / 'yaw_180.jpg',
        270: pano_folder / 'yaw_270.jpg',
    }

    # Generate landmark cards
    landmark_cards = []
    for lm_idx, lm in enumerate(pred['landmarks']):
        primary_tag = lm.get('primary_tag', {})
        additional_tags = lm.get('additional_tags', [])
        confidence = lm.get('confidence', 'unknown')
        description = lm.get('description', '')
        bboxes = lm.get('bounding_boxes', [])

        # Format tags
        primary_tag_str = f"{primary_tag.get('key', '?')}={primary_tag.get('value', '?')}"
        additional_tags_html = ''.join([
            f'<span class="inline-block bg-gray-200 rounded px-2 py-1 text-xs mr-1 mb-1">{t.get("key")}={t.get("value")}</span>'
            for t in additional_tags
        ])

        # Format bounding boxes
        bbox_info = ', '.join([f"yaw {bb.get('yaw_angle', '?')}" for bb in bboxes])

        conf_color = get_confidence_color(confidence)

        landmark_cards.append(f'''
            <div class="bg-white rounded-lg shadow p-4 landmark-card" data-landmark-idx="{lm_idx}">
                <div class="flex justify-between items-start mb-2">
                    <span class="inline-block bg-blue-100 text-blue-800 rounded px-2 py-1 text-sm font-medium">
                        {primary_tag_str}
                    </span>
                    <span class="inline-block rounded px-2 py-1 text-xs text-white" style="background-color: {conf_color}">
                        {confidence}
                    </span>
                </div>
                <p class="text-sm text-gray-600 mb-2">{description}</p>
                <div class="mb-2">{additional_tags_html}</div>
                <p class="text-xs text-gray-400">Visible in: {bbox_info}</p>
                <button class="mt-2 text-xs text-blue-600 hover:underline show-boxes-btn" data-landmark-idx="{lm_idx}">
                    Show bounding boxes
                </button>
            </div>
        ''')

    # Serialize landmarks to JSON for JavaScript
    landmarks_json = json.dumps(pred['landmarks'])

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panorama {idx} - OSM Extraction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #detail-map {{ height: 350px; }}
        .image-container {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
        }}
        .bbox-overlay {{
            position: absolute;
            border: 3px solid;
            pointer-events: none;
            box-sizing: border-box;
        }}
        .bbox-label {{
            position: absolute;
            top: -20px;
            left: 0;
            font-size: 10px;
            padding: 1px 4px;
            color: white;
            white-space: nowrap;
        }}
        .landmark-card {{
            transition: all 0.2s;
        }}
        .landmark-card.highlighted {{
            ring: 2px;
            ring-color: blue;
            transform: scale(1.02);
        }}
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .modal.active {{
            display: flex;
        }}
        .modal-content {{
            position: relative;
            max-width: 90%;
            max-height: 90%;
        }}
        .modal-content img {{
            max-width: 100%;
            max-height: 90vh;
            object-fit: contain;
        }}
        .modal-label {{
            position: absolute;
            top: -30px;
            left: 0;
            color: white;
            font-size: 14px;
        }}
        .modal-close {{
            position: absolute;
            top: 10px;
            right: 20px;
            color: white;
            font-size: 30px;
            cursor: pointer;
        }}
        .modal-nav {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-size: 40px;
            cursor: pointer;
            padding: 20px;
            user-select: none;
        }}
        .modal-nav:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .modal-prev {{ left: 10px; }}
        .modal-next {{ right: 10px; }}
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <div class="flex justify-between items-start mb-6 gap-4">
            <div class="flex-1">
                <a href="/" class="text-blue-600 hover:underline mb-2 inline-block">&larr; Back to map</a>
                <h1 class="text-2xl font-bold">Panorama {idx}</h1>
                <p class="text-gray-600 font-mono text-sm">{pano_key}</p>
                <p class="text-gray-500">Location type: <span class="font-medium">{pred['location_type']}</span></p>
                <a href="{google_maps_url}" target="_blank" class="inline-flex items-center gap-1 text-blue-600 hover:underline text-sm mt-1">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    Open in Google Maps
                </a>
            </div>
            <div class="bg-white rounded-lg shadow overflow-hidden" style="width: 500px;">
                <div id="detail-map"></div>
            </div>
            <div class="flex flex-col gap-2">
                <a href="/view/{idx-1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {'opacity-50 pointer-events-none' if idx == 0 else ''}">&larr; Prev</a>
                <a href="/view/{idx+1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {'opacity-50 pointer-events-none' if idx >= len(predictions)-1 else ''}">&rarr; Next</a>
            </div>
        </div>

        <div class="mb-4">
            <label class="inline-flex items-center">
                <input type="checkbox" id="show-all-boxes" class="form-checkbox h-5 w-5 text-blue-600">
                <span class="ml-2">Show all bounding boxes</span>
            </label>
        </div>

        <!-- Images Row -->
        <div class="grid grid-cols-4 gap-2 mb-8">
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">0° (N)</p>
                <div class="image-container" data-yaw="0">
                    <img src="/image/{idx}/0" alt="Yaw 0">
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">90° (W)</p>
                <div class="image-container" data-yaw="90">
                    <img src="/image/{idx}/90" alt="Yaw 90">
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">180° (S)</p>
                <div class="image-container" data-yaw="180">
                    <img src="/image/{idx}/180" alt="Yaw 180">
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">270° (E)</p>
                <div class="image-container" data-yaw="270">
                    <img src="/image/{idx}/270" alt="Yaw 270">
                </div>
            </div>
        </div>

    <!-- Image Modal -->
    <div id="image-modal" class="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <span class="modal-nav modal-prev" onclick="navigateModal(-1)">&#10094;</span>
        <div class="modal-content">
            <div class="modal-label" id="modal-label"></div>
            <div class="image-container" id="modal-image-container">
                <img id="modal-image" src="" alt="Enlarged view">
            </div>
        </div>
        <span class="modal-nav modal-next" onclick="navigateModal(1)">&#10095;</span>
    </div>

        <!-- Landmarks -->
        <h2 class="text-xl font-bold mb-4">Landmarks ({len(pred['landmarks'])})</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {''.join(landmark_cards)}
        </div>
    </div>

    <script>
        const landmarks = {landmarks_json};
        const colors = [
            '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
            '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#06b6d4'
        ];

        // Initialize mini map
        const detailMap = L.map('detail-map').setView([{lat}, {lon}], 17);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OSM'
        }}).addTo(detailMap);
        L.marker([{lat}, {lon}]).addTo(detailMap);

        function clearBoxes() {{
            document.querySelectorAll('.bbox-overlay').forEach(el => el.remove());
        }}

        function drawBoxesForLandmark(landmarkIdx) {{
            clearBoxes();
            const landmark = landmarks[landmarkIdx];
            const color = colors[landmarkIdx % colors.length];

            landmark.bounding_boxes.forEach(bbox => {{
                const yaw = bbox.yaw_angle;
                const container = document.querySelector(`.image-container[data-yaw="${{yaw}}"]`);
                if (!container) return;

                const img = container.querySelector('img');
                const imgWidth = img.offsetWidth;
                const imgHeight = img.offsetHeight;

                // Convert normalized coords (0-1000) to pixels
                const left = (bbox.xmin / 1000) * imgWidth;
                const top = (bbox.ymin / 1000) * imgHeight;
                const width = ((bbox.xmax - bbox.xmin) / 1000) * imgWidth;
                const height = ((bbox.ymax - bbox.ymin) / 1000) * imgHeight;

                const overlay = document.createElement('div');
                overlay.className = 'bbox-overlay';
                overlay.style.left = left + 'px';
                overlay.style.top = top + 'px';
                overlay.style.width = width + 'px';
                overlay.style.height = height + 'px';
                overlay.style.borderColor = color;

                const label = document.createElement('div');
                label.className = 'bbox-label';
                label.style.backgroundColor = color;
                label.textContent = `${{landmark.primary_tag.key}}=${{landmark.primary_tag.value}}`;
                overlay.appendChild(label);

                container.appendChild(overlay);
            }});

            // Highlight the card
            document.querySelectorAll('.landmark-card').forEach(card => {{
                card.classList.remove('highlighted');
            }});
            document.querySelector(`.landmark-card[data-landmark-idx="${{landmarkIdx}}"]`)?.classList.add('highlighted');
        }}

        function drawAllBoxes() {{
            clearBoxes();
            landmarks.forEach((landmark, landmarkIdx) => {{
                const color = colors[landmarkIdx % colors.length];

                landmark.bounding_boxes.forEach(bbox => {{
                    const yaw = bbox.yaw_angle;
                    const container = document.querySelector(`.image-container[data-yaw="${{yaw}}"]`);
                    if (!container) return;

                    const img = container.querySelector('img');
                    const imgWidth = img.offsetWidth;
                    const imgHeight = img.offsetHeight;

                    const left = (bbox.xmin / 1000) * imgWidth;
                    const top = (bbox.ymin / 1000) * imgHeight;
                    const width = ((bbox.xmax - bbox.xmin) / 1000) * imgWidth;
                    const height = ((bbox.ymax - bbox.ymin) / 1000) * imgHeight;

                    const overlay = document.createElement('div');
                    overlay.className = 'bbox-overlay';
                    overlay.style.left = left + 'px';
                    overlay.style.top = top + 'px';
                    overlay.style.width = width + 'px';
                    overlay.style.height = height + 'px';
                    overlay.style.borderColor = color;

                    const label = document.createElement('div');
                    label.className = 'bbox-label';
                    label.style.backgroundColor = color;
                    label.textContent = `${{landmark.primary_tag.key}}=${{landmark.primary_tag.value}}`;
                    overlay.appendChild(label);

                    container.appendChild(overlay);
                }});
            }});
        }}

        // Event listeners
        document.querySelectorAll('.show-boxes-btn').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                const idx = parseInt(e.target.dataset.landmarkIdx);
                document.getElementById('show-all-boxes').checked = false;
                drawBoxesForLandmark(idx);
            }});
        }});

        document.querySelectorAll('.landmark-card').forEach(card => {{
            card.addEventListener('click', (e) => {{
                if (e.target.classList.contains('show-boxes-btn')) return;
                const idx = parseInt(card.dataset.landmarkIdx);
                document.getElementById('show-all-boxes').checked = false;
                drawBoxesForLandmark(idx);
            }});
        }});

        document.getElementById('show-all-boxes').addEventListener('change', (e) => {{
            if (e.target.checked) {{
                drawAllBoxes();
            }} else {{
                clearBoxes();
            }}
        }});

        // Redraw boxes on window resize
        window.addEventListener('resize', () => {{
            if (document.getElementById('show-all-boxes').checked) {{
                drawAllBoxes();
            }}
        }});

        // Modal functionality
        const yawOrder = [0, 90, 180, 270];
        const yawLabels = {{'0': '0° (North)', '90': '90° (West)', '180': '180° (South)', '270': '270° (East)'}};
        let currentModalYaw = 0;
        let currentHighlightedLandmark = null;

        function openModal(yaw) {{
            currentModalYaw = parseInt(yaw);
            const modal = document.getElementById('image-modal');
            const modalImg = document.getElementById('modal-image');
            const modalLabel = document.getElementById('modal-label');
            const modalContainer = document.getElementById('modal-image-container');

            modalImg.src = `/image/{idx}/${{currentModalYaw}}`;
            modalLabel.textContent = yawLabels[currentModalYaw];
            modalContainer.dataset.yaw = currentModalYaw;

            modal.classList.add('active');

            // Draw boxes in modal if any landmark is highlighted or show-all is checked
            setTimeout(() => {{
                if (document.getElementById('show-all-boxes').checked) {{
                    drawBoxesInModal();
                }} else if (currentHighlightedLandmark !== null) {{
                    drawBoxesInModalForLandmark(currentHighlightedLandmark);
                }}
            }}, 100);
        }}

        function closeModal() {{
            document.getElementById('image-modal').classList.remove('active');
            // Clear modal boxes
            document.querySelectorAll('#modal-image-container .bbox-overlay').forEach(el => el.remove());
        }}

        function navigateModal(direction) {{
            const currentIndex = yawOrder.indexOf(currentModalYaw);
            const newIndex = (currentIndex + direction + 4) % 4;
            openModal(yawOrder[newIndex]);
        }}

        function drawBoxesInModal() {{
            const container = document.getElementById('modal-image-container');
            container.querySelectorAll('.bbox-overlay').forEach(el => el.remove());

            const img = container.querySelector('img');
            const imgWidth = img.offsetWidth;
            const imgHeight = img.offsetHeight;

            landmarks.forEach((landmark, landmarkIdx) => {{
                const color = colors[landmarkIdx % colors.length];

                landmark.bounding_boxes.forEach(bbox => {{
                    if (bbox.yaw_angle != currentModalYaw) return;

                    const left = (bbox.xmin / 1000) * imgWidth;
                    const top = (bbox.ymin / 1000) * imgHeight;
                    const width = ((bbox.xmax - bbox.xmin) / 1000) * imgWidth;
                    const height = ((bbox.ymax - bbox.ymin) / 1000) * imgHeight;

                    const overlay = document.createElement('div');
                    overlay.className = 'bbox-overlay';
                    overlay.style.left = left + 'px';
                    overlay.style.top = top + 'px';
                    overlay.style.width = width + 'px';
                    overlay.style.height = height + 'px';
                    overlay.style.borderColor = color;

                    const label = document.createElement('div');
                    label.className = 'bbox-label';
                    label.style.backgroundColor = color;
                    label.textContent = `${{landmark.primary_tag.key}}=${{landmark.primary_tag.value}}`;
                    overlay.appendChild(label);

                    container.appendChild(overlay);
                }});
            }});
        }}

        function drawBoxesInModalForLandmark(landmarkIdx) {{
            const container = document.getElementById('modal-image-container');
            container.querySelectorAll('.bbox-overlay').forEach(el => el.remove());

            const img = container.querySelector('img');
            const imgWidth = img.offsetWidth;
            const imgHeight = img.offsetHeight;

            const landmark = landmarks[landmarkIdx];
            const color = colors[landmarkIdx % colors.length];

            landmark.bounding_boxes.forEach(bbox => {{
                if (bbox.yaw_angle != currentModalYaw) return;

                const left = (bbox.xmin / 1000) * imgWidth;
                const top = (bbox.ymin / 1000) * imgHeight;
                const width = ((bbox.xmax - bbox.xmin) / 1000) * imgWidth;
                const height = ((bbox.ymax - bbox.ymin) / 1000) * imgHeight;

                const overlay = document.createElement('div');
                overlay.className = 'bbox-overlay';
                overlay.style.left = left + 'px';
                overlay.style.top = top + 'px';
                overlay.style.width = width + 'px';
                overlay.style.height = height + 'px';
                overlay.style.borderColor = color;

                const label = document.createElement('div');
                label.className = 'bbox-label';
                label.style.backgroundColor = color;
                label.textContent = `${{landmark.primary_tag.key}}=${{landmark.primary_tag.value}}`;
                overlay.appendChild(label);

                container.appendChild(overlay);
            }});
        }}

        // Track which landmark is highlighted
        const originalDrawBoxesForLandmark = drawBoxesForLandmark;
        drawBoxesForLandmark = function(landmarkIdx) {{
            currentHighlightedLandmark = landmarkIdx;
            originalDrawBoxesForLandmark(landmarkIdx);
        }};

        // Click on image to open modal
        document.querySelectorAll('.image-container[data-yaw]').forEach(container => {{
            if (container.id === 'modal-image-container') return;
            container.addEventListener('click', (e) => {{
                // Don't open modal if clicking on a bounding box label area
                if (e.target.classList.contains('bbox-overlay') || e.target.classList.contains('bbox-label')) return;
                openModal(container.dataset.yaw);
            }});
        }});

        // Close modal on escape key
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
            if (e.key === 'ArrowLeft') navigateModal(-1);
            if (e.key === 'ArrowRight') navigateModal(1);
        }});

        // Close modal when clicking outside the image
        document.getElementById('image-modal').addEventListener('click', (e) => {{
            if (e.target.id === 'image-modal') closeModal();
        }});
    </script>
</body>
</html>'''


class OSMVisualizerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the visualizer."""

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(generate_index_html().encode())

        elif path.startswith('/view/'):
            try:
                idx = int(path.split('/')[2])
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(generate_view_html(idx).encode())
            except (ValueError, IndexError):
                self.send_error(404, 'Not found')

        elif path.startswith('/image/'):
            # Serve images: /image/{pano_idx}/{yaw}
            try:
                parts = path.split('/')
                pano_idx = int(parts[2])
                yaw = int(parts[3])

                pred = CONFIG['predictions'][pano_idx]
                pano_key = pred['key']

                yaw_map = {0: 'yaw_000.jpg', 90: 'yaw_090.jpg', 180: 'yaw_180.jpg', 270: 'yaw_270.jpg'}
                image_path = CONFIG['images_dir'] / pano_key / yaw_map[yaw]

                if image_path.exists():
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.end_headers()
                    with open(image_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, f'Image not found: {image_path}')
            except (ValueError, IndexError, KeyError) as e:
                self.send_error(404, f'Not found: {e}')
        else:
            self.send_error(404, 'Not found')

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    parser = argparse.ArgumentParser(description='Web visualizer for OSM tag extraction results')
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to predictions.jsonl file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to directory containing panorama folders with pinhole images')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the server on')

    args = parser.parse_args()

    # Set global config
    CONFIG['predictions_file'] = Path(args.predictions_file)
    CONFIG['images_dir'] = Path(args.images_dir)

    # Load predictions
    print(f"Loading predictions from {CONFIG['predictions_file']}...")
    CONFIG['predictions'] = load_predictions(CONFIG['predictions_file'])
    print(f"Loaded {len(CONFIG['predictions'])} predictions")

    # Start server
    with socketserver.TCPServer(("", args.port), OSMVisualizerHandler) as httpd:
        print(f"Server running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == '__main__':
    main()
