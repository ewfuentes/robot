"""
Web visualizer for OSM tag extraction results.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:osm_extraction_visualizer -- \
        --predictions_dir /path/to/predictions_directory \
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
    'predictions_dir': None,
    'images_dir': None,
    'predictions': [],
    'vigor_dataset': None,
    'pano_id_to_pred_idx': {},
}


def load_predictions(predictions_dir: Path) -> list[dict]:
    """Load and parse predictions from all JSONL files in a directory."""
    predictions = []
    jsonl_files = sorted(predictions_dir.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files in {predictions_dir}")
    for jsonl_file in jsonl_files:
        for line in jsonl_file.read_text().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            key = data['key']

            # Extract the response text (Gemini format)
            try:
                response_text = data['response']['candidates'][0]['content']['parts'][0]['text']
                parsed_response = json.loads(response_text)
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Warning: Failed to parse response for {key} in {jsonl_file}: {e}")
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


LANDMARK_COLORS = [
    '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
    '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#06b6d4',
]


def geometry_to_svg_elements(geom_px, left, top, width, height, color):
    """Convert a shapely geometry in pixel coords to SVG elements.

    Normalizes coordinates to a 0-100 range for a viewBox="0 0 100 100" SVG.
    """
    import shapely.geometry as sg

    elements = []

    def nx(x):
        return (x - left) / width * 100

    def ny(y):
        return (y - top) / height * 100

    if isinstance(geom_px, sg.Point):
        elements.append(
            f'<circle cx="{nx(geom_px.x):.2f}" cy="{ny(geom_px.y):.2f}" r="1.5" '
            f'fill="{color}" opacity="0.7" stroke="white" stroke-width="0.3"/>'
        )
    elif isinstance(geom_px, sg.Polygon):
        coords = geom_px.exterior.coords
        points = ' '.join(f'{nx(x):.2f},{ny(y):.2f}' for x, y in coords)
        elements.append(
            f'<polygon points="{points}" fill="{color}" opacity="0.25" '
            f'stroke="{color}" stroke-width="0.5"/>'
        )
    elif isinstance(geom_px, sg.LineString):
        coords = geom_px.coords
        points = ' '.join(f'{nx(x):.2f},{ny(y):.2f}' for x, y in coords)
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="0.8" opacity="0.7"/>'
        )
    elif isinstance(geom_px, sg.MultiPolygon):
        for poly in geom_px.geoms:
            elements.extend(geometry_to_svg_elements(poly, left, top, width, height, color))
    elif isinstance(geom_px, sg.MultiLineString):
        for line in geom_px.geoms:
            elements.extend(geometry_to_svg_elements(line, left, top, width, height, color))
    elif isinstance(geom_px, sg.MultiPoint):
        for pt in geom_px.geoms:
            elements.extend(geometry_to_svg_elements(pt, left, top, width, height, color))

    return elements


def get_satellite_landmark_data(sat_idx):
    """Get satellite patch data with landmarks for rendering.

    Returns dict with satellite metadata and landmark data prepared for
    SVG overlay and Leaflet GeoJSON rendering.
    """
    import shapely.geometry

    dataset = CONFIG['vigor_dataset']
    sat_meta = dataset._satellite_metadata.iloc[sat_idx]
    patch_h, patch_w = dataset._original_satellite_patch_size
    center_x = sat_meta['web_mercator_x']
    center_y = sat_meta['web_mercator_y']
    left = center_x - patch_w / 2
    top = center_y - patch_h / 2

    landmark_idxs = sat_meta.get('landmark_idxs', [])
    if landmark_idxs is None:
        landmark_idxs = []

    # Collect per-landmark data with area for sorting
    svg_groups = []  # (area, index, svg_string)
    geojson_features = []  # (area, index, feature_dict)
    landmark_cards_data = []

    for i, lm_idx in enumerate(landmark_idxs):
        lm = dataset._landmark_metadata.iloc[lm_idx]
        geom_px = lm['geometry_px']
        geom_latlon = lm['geometry']
        pruned = lm['pruned_props']
        color = LANDMARK_COLORS[i % len(LANDMARK_COLORS)]
        area = geom_px.area

        # Tags for card display and tooltips
        tags = [{'key': k, 'value': str(v)} for k, v in sorted(pruned)]
        tooltip_lines = [f'{t["key"]}={t["value"]}' for t in tags]
        svg_tooltip = '&#10;'.join(tooltip_lines) or f'Landmark {i}'
        leaflet_tooltip = '<br>'.join(tooltip_lines) or f'Landmark {i}'

        # SVG elements wrapped in <g> with data-tooltip for JS hover tooltip
        lm_svg = geometry_to_svg_elements(geom_px, left, top, patch_w, patch_h, color)
        if lm_svg:
            escaped_tooltip = leaflet_tooltip.replace('"', '&quot;').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('&amp;quot;', '&quot;')
            group = f'<g class="svg-landmark" data-tooltip="{escaped_tooltip}" style="pointer-events: all; cursor: pointer;">\n'
            group += '\n'.join(lm_svg)
            group += '\n</g>'
            svg_groups.append((area, i, group))

        # GeoJSON feature for Leaflet
        geojson_features.append((area, i, {
            'type': 'Feature',
            'geometry': shapely.geometry.mapping(geom_latlon),
            'properties': {
                'color': color,
                'index': i,
                'tooltip': leaflet_tooltip,
            },
        }))
        landmark_cards_data.append({
            'tags': tags,
            'color': color,
        })

    # Sort by area descending: largest drawn first (behind), smallest last (on top)
    svg_groups.sort(key=lambda x: (-x[0], x[1]))
    svg_content = '\n'.join(g[2] for g in svg_groups)

    geojson_features.sort(key=lambda x: (-x[0], x[1]))
    geojson_collection = {
        'type': 'FeatureCollection',
        'features': [g[2] for g in geojson_features],
    }

    return {
        'lat': float(sat_meta['lat']),
        'lon': float(sat_meta['lon']),
        'path': str(sat_meta['path']),
        'svg_content': svg_content,
        'geojson': geojson_collection,
        'landmark_cards': landmark_cards_data,
        'num_satellites': len(dataset._satellite_metadata),
    }


SVG_TOOLTIP_ASSETS = '''
    <style>
        .svg-tooltip {
            display: none;
            position: fixed;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.5;
            pointer-events: none;
            z-index: 10000;
            max-width: 350px;
        }
    </style>
    <div class="svg-tooltip" id="svg-tooltip"></div>
    <script>
        (function() {
            const tooltip = document.getElementById('svg-tooltip');
            document.addEventListener('mouseover', function(e) {
                const g = e.target.closest('.svg-landmark');
                if (g) {
                    tooltip.innerHTML = g.dataset.tooltip;
                    tooltip.style.display = 'block';
                }
            });
            document.addEventListener('mousemove', function(e) {
                if (tooltip.style.display === 'block') {
                    tooltip.style.left = (e.clientX + 12) + 'px';
                    tooltip.style.top = (e.clientY - 10) + 'px';
                }
            });
            document.addEventListener('mouseout', function(e) {
                const g = e.target.closest('.svg-landmark');
                if (g) {
                    tooltip.style.display = 'none';
                }
            });
        })();
    </script>
'''


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
                    {'<a href="/satellite/0" class="text-blue-600 hover:underline text-sm">Satellite Patches &rarr;</a>' if CONFIG['vigor_dataset'] is not None else ''}
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
                    <a href="/view/${{encodeURIComponent(data.pano_id)}}"
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


def _generate_satellite_input_section(pano_idx: int, pano_id: str) -> str:
    """Generate the satellite index input field and matching satellite buttons."""
    dataset = CONFIG['vigor_dataset']

    # Find this panorama in the VIGOR dataset to get satellite matches
    matches = dataset._panorama_metadata[dataset._panorama_metadata['pano_id'] == pano_id]

    sat_buttons_html = ''
    if len(matches) > 0:
        pano_row = matches.iloc[0]
        pos_sat_idxs = pano_row.get('positive_satellite_idxs', []) or []
        semipos_sat_idxs = pano_row.get('semipositive_satellite_idxs', []) or []

        def _sat_buttons(sat_idxs, css_class):
            buttons = []
            for s_idx in sat_idxs:
                buttons.append(
                    f'<a href="/view/{pano_idx}?sat_idx={s_idx}" '
                    f'class="inline-block px-3 py-1.5 rounded text-xs {css_class}">'
                    f'Sat {s_idx}</a>'
                )
            return buttons

        pos_buttons = _sat_buttons(pos_sat_idxs, 'bg-green-100 text-green-800 hover:bg-green-200')
        semipos_buttons = _sat_buttons(semipos_sat_idxs, 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200')

        if pos_buttons or semipos_buttons:
            pos_html = ' '.join(pos_buttons) if pos_buttons else '<span class="text-xs text-gray-400">None</span>'
            semipos_html = ' '.join(semipos_buttons) if semipos_buttons else '<span class="text-xs text-gray-400">None</span>'
            sat_buttons_html = f'''
                <div class="mt-3">
                    <h3 class="text-sm font-bold mb-2">Matching Satellites</h3>
                    <div class="mb-2">
                        <span class="text-xs font-medium text-gray-500 mr-2">Positive ({len(pos_sat_idxs)}):</span>
                        <div class="inline-flex flex-wrap gap-1">{pos_html}</div>
                    </div>
                    <div>
                        <span class="text-xs font-medium text-gray-500 mr-2">Semi-positive ({len(semipos_sat_idxs)}):</span>
                        <div class="inline-flex flex-wrap gap-1">{semipos_html}</div>
                    </div>
                </div>
            '''

    return f'''
        <div class="mt-8 mb-4 bg-white rounded-lg shadow p-4">
            <h2 class="text-lg font-bold mb-2">Compare with Satellite Patch</h2>
            <div class="flex gap-2 items-center">
                <input type="number" id="sat-idx-input" min="0"
                       max="{len(dataset._satellite_metadata) - 1}"
                       placeholder="Satellite index..."
                       class="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm w-48">
                <button onclick="window.location.href='/view/{pano_idx}?sat_idx=' + document.getElementById('sat-idx-input').value"
                        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm">
                    Compare
                </button>
            </div>
            {sat_buttons_html}
        </div>
    '''


def _generate_satellite_comparison_section(pano_idx: int, sat_idx: int) -> str:
    """Generate the satellite comparison section shown on the panorama view page."""
    dataset = CONFIG['vigor_dataset']
    if sat_idx < 0 or sat_idx >= len(dataset._satellite_metadata):
        return '<div class="mt-4 p-4 bg-red-100 text-red-700 rounded">Invalid satellite index.</div>'

    sat_data = get_satellite_landmark_data(sat_idx)
    geojson_json = json.dumps(sat_data['geojson'])
    sat_lat = sat_data['lat']
    sat_lon = sat_data['lon']

    # Build landmark cards HTML
    lm_cards = []
    for i, lm in enumerate(sat_data['landmark_cards']):
        tags_html = ''.join(
            f'<span class="inline-block bg-gray-200 rounded px-2 py-1 text-xs mr-1 mb-1">{t["key"]}={t["value"]}</span>'
            for t in lm['tags']
        )
        lm_cards.append(f'''
            <div class="bg-white rounded-lg shadow p-3 border-l-4" style="border-left-color: {lm['color']}">
                <div class="text-xs font-medium text-gray-500 mb-1">Landmark {i}</div>
                <div>{tags_html if tags_html else '<span class="text-xs text-gray-400">No tags</span>'}</div>
            </div>
        ''')

    return f'''
        <div class="mt-8 bg-gray-50 rounded-lg shadow p-4 border border-gray-200">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-xl font-bold">Satellite Patch {sat_idx}</h2>
                <a href="/satellite/{sat_idx}" class="text-blue-600 hover:underline text-sm">
                    Open standalone &rarr;
                </a>
            </div>
            <p class="text-sm text-gray-600 mb-4">
                Lat: {sat_lat:.6f}, Lon: {sat_lon:.6f} |
                {len(sat_data['landmark_cards'])} OSM landmarks
            </p>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
                <!-- Satellite image with SVG overlay -->
                <div class="bg-white rounded-lg shadow p-2">
                    <p class="text-center text-sm text-gray-500 mb-1">Satellite Image + OSM Landmarks</p>
                    <div style="position: relative; display: inline-block; width: 100%;">
                        <img src="/satellite_image/{sat_idx}" alt="Satellite {sat_idx}" style="width: 100%; height: auto; display: block;">
                        <svg viewBox="0 0 100 100" preserveAspectRatio="none"
                             style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
                            {sat_data['svg_content']}
                        </svg>
                    </div>
                </div>

                <!-- Leaflet map with satellite location and landmarks -->
                <div class="bg-white rounded-lg shadow p-2">
                    <p class="text-center text-sm text-gray-500 mb-1">Map View</p>
                    <div id="sat-comparison-map" style="height: 400px;"></div>
                </div>
            </div>

            <!-- Satellite landmark cards -->
            <h3 class="text-lg font-bold mb-2">OSM Landmarks ({len(sat_data['landmark_cards'])})</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {''.join(lm_cards)}
            </div>
        </div>

        <script>
            (function() {{
                const satMap = L.map('sat-comparison-map').setView([{sat_lat}, {sat_lon}], 17);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '&copy; OSM'
                }}).addTo(satMap);

                // Satellite center marker
                L.circleMarker([{sat_lat}, {sat_lon}], {{
                    radius: 6, fillColor: '#3b82f6', color: '#fff', weight: 2, fillOpacity: 0.8
                }}).bindPopup('Satellite patch center').addTo(satMap);

                // Landmark GeoJSON
                const geojsonData = {geojson_json};
                L.geoJSON(geojsonData, {{
                    style: function(feature) {{
                        return {{
                            color: feature.properties.color,
                            weight: 2,
                            opacity: 0.7,
                            fillColor: feature.properties.color,
                            fillOpacity: 0.3,
                        }};
                    }},
                    pointToLayer: function(feature, latlng) {{
                        return L.circleMarker(latlng, {{
                            radius: 6,
                            fillColor: feature.properties.color,
                            color: '#fff',
                            weight: 1,
                            fillOpacity: 0.7,
                        }});
                    }},
                    onEachFeature: function(feature, layer) {{
                        if (feature.properties.tooltip) {{
                            layer.bindTooltip(feature.properties.tooltip, {{sticky: true, direction: 'top'}});
                        }}
                    }},
                }}).addTo(satMap);
            }})();
        </script>
        {SVG_TOOLTIP_ASSETS}
    '''


def generate_view_html(idx: int, sat_idx: int | None = None) -> str:
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

    # Build search data for panorama search on this page
    search_data = []
    for search_idx, search_pred in enumerate(predictions):
        search_pano_id, _, _ = parse_pano_key(search_pred['key'])
        search_data.append({
            'idx': search_idx,
            'pano_id': search_pano_id,
            'key': search_pred['key'],
            'location_type': search_pred['location_type'],
            'landmark_count': len(search_pred['landmarks']),
        })
    search_data_json = json.dumps(search_data)

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
        .search-container {{
            position: relative;
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
        .search-result {{
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid #e5e7eb;
        }}
        .search-result:hover {{
            background-color: #f3f4f6;
        }}
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
            <div class="flex flex-col gap-2 items-end">
                <div class="search-container w-64">
                    <input type="text" id="search-input"
                           placeholder="Search by panorama ID..."
                           class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm">
                    <div id="search-results" class="search-results"></div>
                </div>
                <div class="flex gap-2">
                    <a href="/view/{idx-1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {'opacity-50 pointer-events-none' if idx == 0 else ''}">&larr; Prev</a>
                    <a href="/view/{idx+1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {'opacity-50 pointer-events-none' if idx >= len(predictions)-1 else ''}">&rarr; Next</a>
                </div>
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

        {_generate_satellite_input_section(idx, pano_id) if CONFIG['vigor_dataset'] is not None else ''}
        {_generate_satellite_comparison_section(idx, sat_idx) if CONFIG['vigor_dataset'] is not None and sat_idx is not None else ''}
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

        // Panorama search functionality
        const searchData = {search_data_json};
        const searchInput = document.getElementById('search-input');
        const searchResultsEl = document.getElementById('search-results');

        searchInput.addEventListener('input', (e) => {{
            const query = e.target.value.toLowerCase().trim();

            if (query.length < 2) {{
                searchResultsEl.style.display = 'none';
                return;
            }}

            const matches = searchData.filter(d =>
                d.pano_id.toLowerCase().includes(query) ||
                d.key.toLowerCase().includes(query)
            ).slice(0, 10);

            if (matches.length === 0) {{
                searchResultsEl.innerHTML = '<div class="search-result text-gray-500">No results found</div>';
            }} else {{
                searchResultsEl.innerHTML = matches.map(m => `
                    <div class="search-result" data-pano-id="${{m.pano_id}}">
                        <div class="font-medium text-sm">${{m.pano_id}}</div>
                        <div class="text-xs text-gray-500">${{m.location_type}} - ${{m.landmark_count}} landmarks</div>
                    </div>
                `).join('');
            }}

            searchResultsEl.style.display = 'block';
        }});

        searchResultsEl.addEventListener('click', (e) => {{
            const result = e.target.closest('.search-result');
            if (result && result.dataset.panoId) {{
                window.location.href = '/view/' + encodeURIComponent(result.dataset.panoId);
            }}
        }});

        document.addEventListener('click', (e) => {{
            if (!e.target.closest('.search-container')) {{
                searchResultsEl.style.display = 'none';
            }}
        }});

        searchInput.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') {{
                const firstResult = searchResultsEl.querySelector('.search-result[data-pano-id]');
                if (firstResult) {{
                    firstResult.click();
                }}
            }}
        }});
    </script>
</body>
</html>'''


def generate_satellite_view_html(sat_idx: int, pano_id: str | None = None) -> str:
    """Generate standalone satellite detail page."""
    dataset = CONFIG['vigor_dataset']
    num_sats = len(dataset._satellite_metadata)

    if sat_idx < 0 or sat_idx >= num_sats:
        return "<h1>Not found</h1>"

    sat_data = get_satellite_landmark_data(sat_idx)
    sat_lat = sat_data['lat']
    sat_lon = sat_data['lon']
    google_maps_url = f"https://www.google.com/maps?q={sat_lat},{sat_lon}"
    geojson_json = json.dumps(sat_data['geojson'])

    # Build positive/semipositive panorama buttons
    sat_meta = dataset._satellite_metadata.iloc[sat_idx]
    pano_meta = dataset._panorama_metadata

    def _pano_buttons(pano_dataset_idxs, css_class):
        buttons = []
        for p_idx in pano_dataset_idxs:
            p_id = pano_meta.iloc[p_idx]['pano_id']
            pred_idx = CONFIG['pano_id_to_pred_idx'].get(p_id)
            if pred_idx is not None:
                buttons.append(
                    f'<a href="/view/{pred_idx}?sat_idx={sat_idx}" '
                    f'class="inline-block px-3 py-1.5 rounded text-xs font-mono {css_class}">'
                    f'{p_id[:16]}...</a>'
                )
            else:
                buttons.append(
                    f'<span class="inline-block px-3 py-1.5 rounded text-xs font-mono bg-gray-100 text-gray-400">'
                    f'{p_id[:16]}... (no pred)</span>'
                )
        return buttons

    pos_pano_idxs = sat_meta.get('positive_panorama_idxs', []) or []
    semipos_pano_idxs = sat_meta.get('semipositive_panorama_idxs', []) or []
    pos_buttons = _pano_buttons(pos_pano_idxs, 'bg-green-100 text-green-800 hover:bg-green-200')
    semipos_buttons = _pano_buttons(semipos_pano_idxs, 'bg-yellow-100 text-yellow-800 hover:bg-yellow-200')

    pano_matches_section = ''
    if pos_buttons or semipos_buttons:
        pos_html = ' '.join(pos_buttons) if pos_buttons else '<span class="text-xs text-gray-400">None</span>'
        semipos_html = ' '.join(semipos_buttons) if semipos_buttons else '<span class="text-xs text-gray-400">None</span>'
        pano_matches_section = f'''
            <div class="bg-white rounded-lg shadow p-4 mb-4">
                <h3 class="text-sm font-bold mb-2">Matching Panoramas</h3>
                <div class="mb-2">
                    <span class="text-xs font-medium text-gray-500 mr-2">Positive ({len(pos_pano_idxs)}):</span>
                    <div class="inline-flex flex-wrap gap-1">{pos_html}</div>
                </div>
                <div>
                    <span class="text-xs font-medium text-gray-500 mr-2">Semi-positive ({len(semipos_pano_idxs)}):</span>
                    <div class="inline-flex flex-wrap gap-1">{semipos_html}</div>
                </div>
            </div>
        '''

    # Build landmark cards
    lm_cards = []
    for i, lm in enumerate(sat_data['landmark_cards']):
        tags_html = ''.join(
            f'<span class="inline-block bg-gray-200 rounded px-2 py-1 text-xs mr-1 mb-1">{t["key"]}={t["value"]}</span>'
            for t in lm['tags']
        )
        lm_cards.append(f'''
            <div class="bg-white rounded-lg shadow p-3 border-l-4" style="border-left-color: {lm['color']}">
                <div class="text-xs font-medium text-gray-500 mb-1">Landmark {i}</div>
                <div>{tags_html if tags_html else '<span class="text-xs text-gray-400">No tags</span>'}</div>
            </div>
        ''')

    # Panorama comparison section (when pano_id is provided)
    pano_section = ''
    if pano_id is not None:
        pred_idx = CONFIG['pano_id_to_pred_idx'].get(pano_id)
        if pred_idx is not None:
            pred = CONFIG['predictions'][pred_idx]
            pano_key = pred['key']
            p_id, p_lat, p_lon = parse_pano_key(pano_key)

            pano_landmark_cards = []
            for lm_idx, lm in enumerate(pred['landmarks']):
                primary_tag = lm.get('primary_tag', {})
                confidence = lm.get('confidence', 'unknown')
                description = lm.get('description', '')
                conf_color = get_confidence_color(confidence)
                tag_str = f"{primary_tag.get('key', '?')}={primary_tag.get('value', '?')}"

                pano_landmark_cards.append(f'''
                    <div class="bg-white rounded-lg shadow p-3">
                        <div class="flex justify-between items-start mb-1">
                            <span class="inline-block bg-blue-100 text-blue-800 rounded px-2 py-1 text-xs font-medium">{tag_str}</span>
                            <span class="inline-block rounded px-2 py-1 text-xs text-white" style="background-color: {conf_color}">{confidence}</span>
                        </div>
                        <p class="text-xs text-gray-600">{description}</p>
                    </div>
                ''')

            pano_section = f'''
                <div class="mt-8 bg-blue-50 rounded-lg shadow p-4 border border-blue-200">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-bold">Panorama: {p_id[:20]}...</h2>
                        <a href="/view/{pred_idx}?sat_idx={sat_idx}" class="text-blue-600 hover:underline text-sm">
                            View panorama &rarr;
                        </a>
                    </div>
                    <p class="text-sm text-gray-600 mb-2">Location type: {pred['location_type']}</p>

                    <div class="grid grid-cols-4 gap-2 mb-4">
                        <div class="bg-white rounded p-1">
                            <p class="text-center text-xs text-gray-500">0&deg;</p>
                            <img src="/image/{pred_idx}/0" alt="Yaw 0" class="w-full">
                        </div>
                        <div class="bg-white rounded p-1">
                            <p class="text-center text-xs text-gray-500">90&deg;</p>
                            <img src="/image/{pred_idx}/90" alt="Yaw 90" class="w-full">
                        </div>
                        <div class="bg-white rounded p-1">
                            <p class="text-center text-xs text-gray-500">180&deg;</p>
                            <img src="/image/{pred_idx}/180" alt="Yaw 180" class="w-full">
                        </div>
                        <div class="bg-white rounded p-1">
                            <p class="text-center text-xs text-gray-500">270&deg;</p>
                            <img src="/image/{pred_idx}/270" alt="Yaw 270" class="w-full">
                        </div>
                    </div>

                    <h3 class="text-lg font-bold mb-2">Predicted Landmarks ({len(pred['landmarks'])})</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                        {''.join(pano_landmark_cards)}
                    </div>
                </div>
            '''
        else:
            pano_section = f'''
                <div class="mt-4 p-4 bg-yellow-100 text-yellow-700 rounded">
                    Panorama ID "{pano_id}" not found in predictions.
                </div>
            '''

    prev_disabled = "opacity-50 pointer-events-none" if sat_idx == 0 else ""
    next_disabled = "opacity-50 pointer-events-none" if sat_idx >= num_sats - 1 else ""

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Satellite {sat_idx} - OSM Extraction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #sat-map {{ height: 400px; }}
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="flex justify-between items-start mb-6 gap-4">
            <div class="flex-1">
                <a href="/" class="text-blue-600 hover:underline mb-2 inline-block">&larr; Back to map</a>
                <h1 class="text-2xl font-bold">Satellite Patch {sat_idx}</h1>
                <p class="text-gray-600 text-sm">
                    Lat: {sat_lat:.6f}, Lon: {sat_lon:.6f}
                </p>
                <p class="text-gray-400 text-xs font-mono">{sat_data['path']}</p>
                <a href="{google_maps_url}" target="_blank" class="inline-flex items-center gap-1 text-blue-600 hover:underline text-sm mt-1">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    Open in Google Maps
                </a>
            </div>
            <div class="flex flex-col gap-2 items-end">
                <!-- Navigation -->
                <div class="flex gap-2 items-center">
                    <input type="number" id="sat-goto-input" min="0" max="{num_sats - 1}"
                           placeholder="Go to index..."
                           class="px-3 py-2 border border-gray-300 rounded-lg text-sm w-36">
                    <button onclick="window.location.href='/satellite/' + document.getElementById('sat-goto-input').value"
                            class="px-3 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 text-sm">Go</button>
                </div>
                <div class="flex gap-2">
                    <a href="/satellite/{sat_idx-1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {prev_disabled}">&larr; Prev</a>
                    <a href="/satellite/{sat_idx+1}" class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 text-center {next_disabled}">&rarr; Next</a>
                </div>
                <!-- Panorama ID search -->
                <div class="flex gap-2 items-center">
                    <input type="text" id="pano-id-input"
                           placeholder="Panorama ID..."
                           class="px-3 py-2 border border-gray-300 rounded-lg text-sm w-48">
                    <button onclick="window.location.href='/satellite/{sat_idx}?pano_id=' + encodeURIComponent(document.getElementById('pano-id-input').value)"
                            class="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm">Find</button>
                </div>
            </div>
        </div>

        <!-- Satellite image + Map -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">Satellite Image + OSM Landmarks</p>
                <div style="position: relative; display: inline-block; width: 100%;">
                    <img src="/satellite_image/{sat_idx}" alt="Satellite {sat_idx}" style="width: 100%; height: auto; display: block;">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none"
                         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
                        {sat_data['svg_content']}
                    </svg>
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-2">
                <p class="text-center text-sm text-gray-500 mb-1">Map View</p>
                <div id="sat-map"></div>
            </div>
        </div>

        {pano_matches_section}

        <!-- Landmark cards -->
        <h2 class="text-xl font-bold mb-4">OSM Landmarks ({len(sat_data['landmark_cards'])})</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-8">
            {''.join(lm_cards)}
        </div>

        {pano_section}
    </div>

    <script>
        // Initialize map
        const satMap = L.map('sat-map').setView([{sat_lat}, {sat_lon}], 17);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; OSM'
        }}).addTo(satMap);

        // Satellite center marker
        L.circleMarker([{sat_lat}, {sat_lon}], {{
            radius: 6, fillColor: '#3b82f6', color: '#fff', weight: 2, fillOpacity: 0.8
        }}).bindPopup('Satellite patch center').addTo(satMap);

        // Landmark GeoJSON
        const geojsonData = {geojson_json};
        L.geoJSON(geojsonData, {{
            style: function(feature) {{
                return {{
                    color: feature.properties.color,
                    weight: 2,
                    opacity: 0.7,
                    fillColor: feature.properties.color,
                    fillOpacity: 0.3,
                }};
            }},
            pointToLayer: function(feature, latlng) {{
                return L.circleMarker(latlng, {{
                    radius: 6,
                    fillColor: feature.properties.color,
                    color: '#fff',
                    weight: 1,
                    fillOpacity: 0.7,
                }});
            }},
            onEachFeature: function(feature, layer) {{
                if (feature.properties.tooltip) {{
                    layer.bindTooltip(feature.properties.tooltip, {{sticky: true, direction: 'top'}});
                }}
            }},
        }}).addTo(satMap);
    </script>
    {SVG_TOOLTIP_ASSETS}
</body>
</html>'''


class OSMVisualizerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the visualizer."""

    def _send_html(self, html: str):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_image(self, image_path: Path):
        if image_path.exists():
            suffix = image_path.suffix.lower()
            content_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(suffix, 'image/jpeg')
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            with open(image_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, f'Image not found: {image_path}')

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query_params = urllib.parse.parse_qs(parsed_path.query)

        if path == '/' or path == '/index.html':
            self._send_html(generate_index_html())

        elif path.startswith('/view/'):
            try:
                view_key = urllib.parse.unquote(path.split('/')[2])
                # Accept either numeric index or pano_id
                try:
                    idx = int(view_key)
                    # Redirect numeric index to pano_id URL
                    if 0 <= idx < len(CONFIG['predictions']):
                        pano_id, _, _ = parse_pano_key(CONFIG['predictions'][idx]['key'])
                        redirect_url = f'/view/{urllib.parse.quote(pano_id, safe="")}'
                        if parsed_path.query:
                            redirect_url += f'?{parsed_path.query}'
                        self.send_response(302)
                        self.send_header('Location', redirect_url)
                        self.end_headers()
                        return
                    else:
                        self.send_error(404, 'Index out of range')
                        return
                except ValueError:
                    idx = CONFIG['pano_id_to_pred_idx'].get(view_key)
                    if idx is None:
                        self.send_error(404, f'Panorama ID not found: {view_key}')
                        return
                sat_idx = None
                if 'sat_idx' in query_params:
                    sat_idx = int(query_params['sat_idx'][0])
                self._send_html(generate_view_html(idx, sat_idx=sat_idx))
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
                self._send_image(image_path)
            except (ValueError, IndexError, KeyError) as e:
                self.send_error(404, f'Not found: {e}')

        elif path.startswith('/satellite_image/'):
            # Serve satellite image: /satellite_image/{sat_idx}
            try:
                sat_idx = int(path.split('/')[2])
                dataset = CONFIG['vigor_dataset']
                if dataset is None:
                    self.send_error(404, 'VIGOR dataset not loaded')
                    return
                sat_meta = dataset._satellite_metadata.iloc[sat_idx]
                self._send_image(Path(sat_meta['path']))
            except (ValueError, IndexError) as e:
                self.send_error(404, f'Not found: {e}')

        elif path.startswith('/satellite/'):
            # Standalone satellite detail page: /satellite/{sat_idx}?pano_id=...
            try:
                sat_idx = int(path.split('/')[2])
                pano_id = query_params.get('pano_id', [None])[0]
                self._send_html(generate_satellite_view_html(sat_idx, pano_id=pano_id))
            except (ValueError, IndexError):
                self.send_error(404, 'Not found')

        else:
            self.send_error(404, 'Not found')

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    parser = argparse.ArgumentParser(description='Web visualizer for OSM tag extraction results')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Path to directory containing prediction JSONL files (searched recursively)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to directory containing panorama folders with pinhole images')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the server on')
    parser.add_argument('--vigor_dataset_path', type=str, default=None,
                        help='Path to VIGOR dataset (enables satellite patch features)')
    parser.add_argument('--landmark_version', type=str, default='v3',
                        help='Landmark version to load (default: v3)')

    args = parser.parse_args()

    # Set global config
    CONFIG['predictions_dir'] = Path(args.predictions_dir)
    CONFIG['images_dir'] = Path(args.images_dir)

    # Load predictions
    print(f"Loading predictions from {CONFIG['predictions_dir']}...")
    CONFIG['predictions'] = load_predictions(CONFIG['predictions_dir'])
    print(f"Loaded {len(CONFIG['predictions'])} predictions")

    # Build pano_id -> prediction index mapping
    for pred_idx, pred in enumerate(CONFIG['predictions']):
        pano_id, _, _ = parse_pano_key(pred['key'])
        CONFIG['pano_id_to_pred_idx'][pano_id] = pred_idx

    # Optionally load VIGOR dataset for satellite features
    if args.vigor_dataset_path is not None:
        from experimental.overhead_matching.swag.data.vigor_dataset import VigorDataset, VigorDatasetConfig

        print(f"Loading VIGOR dataset from {args.vigor_dataset_path}...")
        vigor_config = VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=True,
            landmark_version=args.landmark_version,
        )
        CONFIG['vigor_dataset'] = VigorDataset(
            Path(args.vigor_dataset_path),
            vigor_config,
        )
        print(f"Loaded VIGOR: {len(CONFIG['vigor_dataset']._satellite_metadata)} satellites, "
              f"{len(CONFIG['vigor_dataset']._panorama_metadata)} panoramas")

    # Start server
    with socketserver.TCPServer(("localhost", args.port), OSMVisualizerHandler) as httpd:
        print(f"Server running at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == '__main__':
    main()
