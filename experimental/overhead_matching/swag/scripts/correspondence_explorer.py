#!/usr/bin/env python3
"""Interactive explorer for correspondence-based similarity scoring.

Loads precomputed P(match) cost matrices (from export_correspondence_similarity --save_raw)
and lets you interactively change matching/aggregation methods to see how satellite scores
change spatially and in distribution.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:correspondence_explorer -- \
        --precomputed_data /tmp/miami_corr_raw.pt \
        --dataset_path /data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach \
        --port 5003
"""

import argparse
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation.correspondence_similarity import (
    AggregationMode,
    MatchingMethod,
    RawCorrespondenceData,
    match_and_aggregate,
)

app = Flask(__name__)

# Global state
VIGOR_DATASET = None
RAW_DATA: RawCorrespondenceData | None = None
PANO_ID_TO_VIGOR_IDX: dict[str, int] = {}
PANO_ID_LIST: list[str] = []  # ordered pano_ids with cost data
# sat_idx → list of column positions in cost matrix (prebuilt for speed)
SAT_TO_COL_POSITIONS: list[list[int]] = []


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Correspondence Explorer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; overflow: hidden; height: 100%; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .header {
            background: white; padding: 12px 20px; border-bottom: 2px solid #ddd;
            display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
        }
        .header h1 { margin: 0; font-size: 18px; white-space: nowrap; }
        .nav-group { display: flex; align-items: center; gap: 8px; }
        .nav-group select { padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; font-size: 13px; max-width: 300px; }
        .nav-group button { padding: 6px 14px; border: none; background: #007bff; color: white; border-radius: 4px; cursor: pointer; font-size: 13px; }
        .nav-group button:hover { background: #0056b3; }
        .controls {
            display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
            background: #f0f8ff; padding: 8px 12px; border-radius: 4px; border: 1px solid #d0e8ff;
        }
        .controls label { font-size: 12px; font-weight: 600; color: #555; }
        .controls select, .controls input[type=range] { font-size: 12px; }
        .metrics-bar {
            display: flex; gap: 16px; font-size: 13px; color: #333;
        }
        .metrics-bar .metric { font-weight: 600; }
        .metric.good { color: #2e7d32; }
        .metric.bad { color: #c62828; }

        .main { display: flex; height: calc(100vh - 60px); overflow: hidden; }
        .left-panel { width: 380px; min-width: 380px; overflow-y: auto; background: white; border-right: 1px solid #ddd; }
        .right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

        .pano-section { padding: 12px; }
        .pano-section h3 { margin: 0 0 8px 0; font-size: 14px; color: #333; }
        .pano-img { width: 100%; border-radius: 4px; border: 1px solid #ddd; }

        .landmark-list { list-style: none; padding: 0; margin: 0; }
        .landmark-item {
            padding: 8px 10px; margin: 4px 0; background: #f9f9f9; border-radius: 4px;
            border: 2px solid transparent; cursor: pointer; font-size: 12px;
            transition: all 0.15s;
        }
        .landmark-item:hover { border-color: #007bff; }
        .landmark-item.selected { border-color: #2e7d32; background: #e8f5e9; }

        .lm-detail { padding: 12px; border-top: 1px solid #ddd; display: none; }
        .lm-detail h4 { margin: 0 0 8px 0; font-size: 13px; color: #555; }
        .lm-match-list { font-size: 12px; }
        .lm-match-item { padding: 4px 0; border-bottom: 1px solid #f0f0f0; display: flex; gap: 8px; }
        .prob-badge {
            display: inline-block; padding: 2px 6px; border-radius: 3px;
            font-size: 11px; font-weight: 600; color: white; white-space: nowrap;
        }
        .prob-badge.high { background: #4caf50; }
        .prob-badge.medium { background: #ff9800; }
        .prob-badge.low { background: #9e9e9e; }

        #map-container { flex: 1; position: relative; }
        #map { height: 100%; }

        #histogram-container {
            height: 200px; border-top: 1px solid #ddd; background: white;
            padding: 8px; position: relative;
        }
        #histogram-canvas { width: 100%; height: 100%; cursor: crosshair; }
        #histogram-tooltip {
            display: none; position: absolute; background: white; border: 1px solid #ccc;
            border-radius: 6px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            max-width: 350px; z-index: 1000; font-size: 12px; pointer-events: none;
        }

        #sat-detail {
            border-top: 2px solid #ddd; background: white; padding: 12px 16px;
            display: none; max-height: 350px; overflow-y: auto;
        }
        #sat-detail h3 { margin: 0 0 10px 0; font-size: 14px; }
        .sat-detail-grid { display: flex; gap: 16px; }
        .sat-detail-grid img { width: 120px; height: 120px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }
        .cost-table { border-collapse: collapse; font-size: 11px; }
        .cost-table th, .cost-table td {
            padding: 4px 6px; border: 1px solid #ddd; text-align: center;
            max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .cost-table th { background: #f5f5f5; font-weight: 600; font-size: 10px; }
        .cost-table td.matched { border: 2px solid #2e7d32; font-weight: 700; }
        .match-list { font-size: 12px; min-width: 200px; }
        .match-list .match-pair { padding: 4px 0; border-bottom: 1px solid #f0f0f0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Correspondence Explorer</h1>
        <div class="nav-group">
            <button onclick="navigate(-1)">&#9664;</button>
            <select id="pano-select" onchange="loadPanorama(this.value)"></select>
            <button onclick="navigate(1)">&#9654;</button>
            <span id="pano-counter" style="font-size:12px;color:#666;"></span>
        </div>
        <div class="controls">
            <label>Match:</label>
            <select id="method-select">
                <option value="hungarian">Hungarian</option>
                <option value="greedy">Greedy</option>
            </select>
            <label>Agg:</label>
            <select id="agg-select">
                <option value="sum">Sum</option>
                <option value="max">Max</option>
                <option value="log_odds">Log Odds</option>
            </select>
            <label>Thresh:</label>
            <input type="range" id="threshold-slider" min="0" max="0.9" step="0.05" value="0.3"
                   oninput="document.getElementById('threshold-val').textContent=this.value">
            <span id="threshold-val" style="font-size:12px;width:30px;">0.3</span>
            <button onclick="recomputeScores()" style="background:#2e7d32;">Recompute</button>
        </div>
        <div class="metrics-bar" id="metrics-bar"></div>
    </div>

    <div class="main">
        <div class="left-panel">
            <div class="pano-section">
                <h3>Panorama <span id="pano-id-display"></span></h3>
                <img id="pano-img" class="pano-img" src="" alt="Panorama">
            </div>
            <div class="pano-section">
                <h3>Pano Landmarks (<span id="lm-count">0</span>)</h3>
                <ul class="landmark-list" id="landmark-list"></ul>
            </div>
            <div class="lm-detail" id="lm-detail">
                <h4>P(match) Distribution for: <span id="lm-detail-name"></span></h4>
                <canvas id="lm-hist-canvas" width="350" height="120" style="width:100%;border:1px solid #ddd;border-radius:4px;"></canvas>
                <h4 style="margin-top:8px;">Top Matches</h4>
                <div class="lm-match-list" id="lm-match-list"></div>
            </div>
        </div>
        <div class="right-panel">
            <div id="map-container"><div id="map"></div></div>
            <div id="histogram-container">
                <canvas id="histogram-canvas"></canvas>
                <div id="histogram-tooltip"></div>
            </div>
            <div id="sat-detail">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <h3 id="sat-detail-title" style="margin:0;"></h3>
                    <button onclick="closeSatDetail()" style="padding:4px 12px;border:1px solid #ccc;border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:12px;">✕ Close</button>
                </div>
                <div class="sat-detail-grid">
                    <img id="sat-detail-img" src="" alt="Satellite">
                    <div style="overflow-x:auto;flex:1;">
                        <table class="cost-table" id="cost-table"></table>
                    </div>
                    <div class="match-list" id="sat-match-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let panoIds = [];
        let currentPanoIdx = 0;
        let currentPanoId = null;
        let selectedLmIdx = null;
        let satMap = null;
        let satMarkers = [];
        let panoMarker = null;
        let allSatScores = null;  // cached from last recompute
        let histogramLayout = null;  // cached layout info for hover

        // --- Init ---
        fetch('/api/panoramas').then(r => r.json()).then(data => {
            panoIds = data.map(p => p.pano_id);
            const sel = document.getElementById('pano-select');
            data.forEach((p, i) => {
                const opt = document.createElement('option');
                opt.value = p.pano_id;
                opt.textContent = `${p.pano_id} (${p.n_landmarks} lm)`;
                sel.appendChild(opt);
            });
            // Init map
            satMap = L.map('map');
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap', maxZoom: 19
            }).addTo(satMap);

            if (panoIds.length > 0) loadPanorama(panoIds[0]);
        });

        function navigate(delta) {
            currentPanoIdx = Math.max(0, Math.min(panoIds.length - 1, currentPanoIdx + delta));
            document.getElementById('pano-select').value = panoIds[currentPanoIdx];
            loadPanorama(panoIds[currentPanoIdx]);
        }

        function loadPanorama(panoId) {
            currentPanoId = panoId;
            currentPanoIdx = panoIds.indexOf(panoId);
            selectedLmIdx = null;
            document.getElementById('lm-detail').style.display = 'none';
            document.getElementById('sat-detail').style.display = 'none';
            document.getElementById('pano-counter').textContent =
                `${currentPanoIdx + 1} / ${panoIds.length}`;
            document.getElementById('pano-id-display').textContent = panoId;
            document.getElementById('pano-img').src = '/api/image/panorama/' + panoId;

            // Load landmarks
            fetch('/api/panorama/' + panoId).then(r => r.json()).then(data => {
                document.getElementById('lm-count').textContent = data.landmarks.length;
                const list = document.getElementById('landmark-list');
                list.innerHTML = '';
                data.landmarks.forEach((lm, i) => {
                    const li = document.createElement('li');
                    li.className = 'landmark-item';
                    const matchInfo = lm.n_strong_matches === 0
                        ? '<span style="color:#c62828;font-size:10px;margin-left:6px;">(no matches)</span>'
                        : `<span style="color:#666;font-size:10px;margin-left:6px;">(${lm.n_strong_matches} matches, max ${(lm.max_match*100).toFixed(0)}%)</span>`;
                    li.innerHTML = `${lm.tags_str} ${matchInfo}`;
                    if (lm.n_strong_matches === 0) li.style.opacity = '0.5';
                    li.addEventListener('click', () => selectLandmark(i, lm.tags_str));
                    list.appendChild(li);
                });
            });

            // Compute satellite scores
            recomputeScores();
        }

        function getAggParams() {
            return {
                method: document.getElementById('method-select').value,
                aggregation: document.getElementById('agg-select').value,
                prob_threshold: parseFloat(document.getElementById('threshold-slider').value),
            };
        }

        function recomputeScores() {
            if (!currentPanoId) return;
            const params = getAggParams();
            fetch('/api/satellite_scores/' + currentPanoId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params),
            }).then(r => r.json()).then(data => {
                allSatScores = data;
                renderMap(data);
                renderHistogram(data);
                renderMetrics(data);
            });
        }

        function renderMetrics(data) {
            const bar = document.getElementById('metrics-bar');
            const m = data.metrics;
            const mrrClass = m.mrr >= 0.5 ? 'good' : m.mrr >= 0.1 ? '' : 'bad';
            bar.innerHTML = `
                <span>MRR: <span class="metric ${mrrClass}">${m.mrr.toFixed(4)}</span></span>
                <span>Best rank: <span class="metric">${m.best_rank >= 0 ? m.best_rank + 1 : 'N/A'}</span></span>
                <span>Non-zero: <span class="metric">${m.pct_nonzero.toFixed(1)}%</span></span>
                <span>Top score: <span class="metric">${m.max_score.toFixed(3)}</span></span>
            `;
        }

        // --- Map ---
        function renderMap(data) {
            // Clear old markers
            satMarkers.forEach(m => satMap.removeLayer(m));
            satMarkers = [];
            if (panoMarker) satMap.removeLayer(panoMarker);

            const sats = data.satellites;
            if (sats.length === 0) return;

            // Find score range for coloring
            const scores = sats.map(s => s.score);
            const maxScore = Math.max(...scores.filter(s => s > 0), 0.001);

            // Add satellite markers
            sats.forEach(s => {
                if (s.score <= 0 && !s.is_positive) return;  // skip zero-score non-positives
                const t = Math.min(s.score / maxScore, 1.0);
                const color = viridisColor(t);
                const radius = s.is_positive ? 7 : 4 + t * 4;
                const opacity = s.is_positive ? 0.9 : 0.3 + t * 0.6;

                const marker = L.circleMarker([s.lat, s.lon], {
                    radius: radius,
                    color: s.is_positive ? '#2e7d32' : color,
                    fillColor: color,
                    fillOpacity: opacity,
                    weight: s.is_positive ? 2 : 1,
                });
                marker.bindTooltip(`#${s.rank+1} score=${s.score.toFixed(3)}${s.is_positive ? ' ✓' : ''} (${s.n_landmarks} lm)`,
                    {direction: 'top'});
                marker.on('click', () => loadSatDetail(s.sat_idx, s.rank, s.score, s.is_positive));
                marker.addTo(satMap);
                satMarkers.push(marker);
            });

            // Panorama marker
            panoMarker = L.circleMarker([data.pano_lat, data.pano_lon], {
                radius: 10, color: '#1565c0', fillColor: '#42a5f5',
                fillOpacity: 0.9, weight: 3,
            }).addTo(satMap);
            panoMarker.bindTooltip('Panorama', {permanent: true, direction: 'top'});

            // Fit bounds
            const allLats = sats.filter(s => s.score > 0 || s.is_positive).map(s => s.lat).concat([data.pano_lat]);
            const allLons = sats.filter(s => s.score > 0 || s.is_positive).map(s => s.lon).concat([data.pano_lon]);
            if (allLats.length > 0) {
                satMap.fitBounds([[Math.min(...allLats), Math.min(...allLons)],
                                  [Math.max(...allLats), Math.max(...allLons)]],
                                 {padding: [30, 30]});
            }
        }

        function viridisColor(t) {
            // Simplified viridis: purple → teal → yellow
            t = Math.max(0, Math.min(1, t));
            const r = Math.round(t < 0.5 ? 68 + t * 2 * (30 - 68) : 30 + (t - 0.5) * 2 * (253 - 30));
            const g = Math.round(t < 0.5 ? 1 + t * 2 * (130 - 1) : 130 + (t - 0.5) * 2 * (231 - 130));
            const b = Math.round(t < 0.5 ? 84 + t * 2 * (129 - 84) : 129 + (t - 0.5) * 2 * (37 - 129));
            return `rgb(${r},${g},${b})`;
        }

        // --- Satellite Score Histogram ---
        function renderHistogram(data) {
            const canvas = document.getElementById('histogram-canvas');
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            ctx.scale(dpr, dpr);

            const W = rect.width, H = rect.height;
            const pad = {top: 20, right: 15, bottom: 35, left: 55};
            const plotW = W - pad.left - pad.right;
            const plotH = H - pad.top - pad.bottom;

            ctx.clearRect(0, 0, W, H);

            const hist = data.histogram;
            if (!hist || hist.bin_edges.length < 2) return;

            const edges = hist.bin_edges;
            const counts = hist.counts;
            const posCounts = hist.positive_counts;
            const nBins = counts.length;
            const maxCount = Math.max(...counts, 1);
            const logMax = Math.log10(maxCount + 1);

            const barW = plotW / nBins;

            // Save layout for hover handler
            histogramLayout = { padLeft: pad.left, barW: barW, nBins: nBins };

            // Draw bars
            for (let i = 0; i < nBins; i++) {
                if (counts[i] === 0) continue;
                const logH = Math.log10(counts[i] + 1) / logMax;
                const bh = logH * plotH;
                const x = pad.left + i * barW;
                const y = pad.top + plotH - bh;

                // All satellites bar
                ctx.fillStyle = '#90caf9';
                ctx.fillRect(x, y, barW - 1, bh);

                // Positive overlay
                if (posCounts[i] > 0) {
                    const posLogH = Math.log10(posCounts[i] + 1) / logMax;
                    const pbh = posLogH * plotH;
                    ctx.fillStyle = '#4caf50';
                    ctx.fillRect(x, pad.top + plotH - pbh, barW - 1, pbh);
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

            // X labels
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            for (let i = 0; i <= nBins; i += Math.max(1, Math.floor(nBins / 6))) {
                const x = pad.left + i * barW;
                ctx.fillText(edges[i].toFixed(2), x, pad.top + plotH + 14);
            }
            ctx.textAlign = 'center';
            ctx.font = '11px Arial';
            ctx.fillText('Aggregated Similarity Score', pad.left + plotW / 2, H - 3);

            // Y labels (log scale)
            ctx.textAlign = 'right';
            ctx.font = '10px Arial';
            for (let exp = 0; exp <= Math.ceil(logMax); exp++) {
                const y = pad.top + plotH - (exp / logMax) * plotH;
                if (y >= pad.top) {
                    ctx.fillText(Math.pow(10, exp).toFixed(0), pad.left - 4, y + 3);
                }
            }
            ctx.save();
            ctx.translate(12, pad.top + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.font = '11px Arial';
            ctx.fillText('Count (log)', 0, 0);
            ctx.restore();

            // Title
            ctx.textAlign = 'left';
            ctx.font = '12px Arial';
            ctx.fillStyle = '#333';
            const totalNonZero = data.metrics.pct_nonzero;
            ctx.fillText(`Score Distribution (${totalNonZero.toFixed(1)}% non-zero)`, pad.left, 14);

            // Legend
            ctx.textAlign = 'right';
            ctx.fillStyle = '#90caf9';
            ctx.fillRect(W - 110, 5, 10, 10);
            ctx.fillStyle = '#4caf50';
            ctx.fillRect(W - 110, 18, 10, 10);
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.fillText('All', W - 15, 14);
            ctx.fillText('Positive', W - 15, 27);
        }

        // --- Landmark selection ---
        function selectLandmark(idx, name) {
            const items = document.querySelectorAll('#landmark-list .landmark-item');
            items.forEach((el, i) => el.classList.toggle('selected', i === idx));

            if (selectedLmIdx === idx) {
                selectedLmIdx = null;
                document.getElementById('lm-detail').style.display = 'none';
                return;
            }
            selectedLmIdx = idx;
            document.getElementById('lm-detail-name').textContent = name;
            document.getElementById('lm-detail').style.display = 'block';

            fetch(`/api/landmark_scores/${currentPanoId}/${idx}`).then(r => r.json()).then(data => {
                renderLmHistogram(data.scores);
                renderLmTopMatches(data.top_k);
            });
        }

        function renderLmHistogram(scores) {
            const canvas = document.getElementById('lm-hist-canvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const W = canvas.width, H = canvas.height;
            const pad = {top: 10, right: 10, bottom: 22, left: 40};

            // Bin scores into 50 bins from 0 to 1
            const nBins = 50;
            const counts = new Array(nBins).fill(0);
            scores.forEach(s => {
                const bin = Math.min(Math.floor(s * nBins), nBins - 1);
                counts[bin]++;
            });

            const maxCount = Math.max(...counts, 1);
            const logMax = Math.log10(maxCount + 1);
            const plotW = W - pad.left - pad.right;
            const plotH = H - pad.top - pad.bottom;
            const barW = plotW / nBins;

            for (let i = 0; i < nBins; i++) {
                if (counts[i] === 0) continue;
                const logH = Math.log10(counts[i] + 1) / logMax;
                const bh = logH * plotH;
                ctx.fillStyle = i >= nBins * 0.5 ? '#ff9800' : '#90caf9';
                ctx.fillRect(pad.left + i * barW, pad.top + plotH - bh, barW - 0.5, bh);
            }

            // Axes
            ctx.strokeStyle = '#999';
            ctx.beginPath();
            ctx.moveTo(pad.left, pad.top + plotH);
            ctx.lineTo(pad.left + plotW, pad.top + plotH);
            ctx.stroke();

            ctx.fillStyle = '#666';
            ctx.font = '9px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('0', pad.left, H - 4);
            ctx.fillText('0.5', pad.left + plotW / 2, H - 4);
            ctx.fillText('1.0', pad.left + plotW, H - 4);

            ctx.textAlign = 'right';
            ctx.fillText('1', pad.left - 3, pad.top + plotH);
            ctx.fillText(maxCount.toString(), pad.left - 3, pad.top + 9);
        }

        function renderLmTopMatches(topK) {
            const list = document.getElementById('lm-match-list');
            list.innerHTML = '';
            topK.forEach(m => {
                const div = document.createElement('div');
                div.className = 'lm-match-item';
                const cls = m.prob >= 0.8 ? 'high' : m.prob >= 0.5 ? 'medium' : 'low';
                div.innerHTML = `<span class="prob-badge ${cls}">${(m.prob * 100).toFixed(1)}%</span>
                    <span style="font-size:11px;">${m.tags_str}</span>`;
                list.appendChild(div);
            });
        }

        // --- Satellite detail ---
        function loadSatDetail(satIdx, rank, score, isPositive) {
            const params = getAggParams();
            params.sat_idx = satIdx;
            fetch(`/api/satellite_detail/${currentPanoId}/${satIdx}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params),
            }).then(r => r.json()).then(data => {
                const panel = document.getElementById('sat-detail');
                panel.style.display = 'block';

                document.getElementById('sat-detail-title').textContent =
                    `Rank #${rank + 1} — Satellite ${satIdx} — Score: ${data.score.toFixed(4)}` +
                    (isPositive ? ' ✓ Positive' : '');
                document.getElementById('sat-detail-img').src = '/api/image/satellite/' + satIdx;

                // Build cost matrix table
                const table = document.getElementById('cost-table');
                const matchedSet = new Set(data.matches.map(m => `${m.pano_lm_idx},${m.osm_lm_idx}`));

                let html = '<thead><tr><th></th>';
                data.osm_tags.forEach((t, j) => {
                    html += `<th title="${t}">${t.substring(0, 20)}</th>`;
                });
                html += '</tr></thead><tbody>';

                data.cost_slice.forEach((row, i) => {
                    html += `<tr><th title="${data.pano_tags[i]}">${data.pano_tags[i].substring(0, 20)}</th>`;
                    row.forEach((val, j) => {
                        const matched = matchedSet.has(`${i},${j}`);
                        const bg = probColor(val);
                        html += `<td style="background:${bg}" class="${matched ? 'matched' : ''}">${val.toFixed(3)}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody>';
                table.innerHTML = html;

                // Match list
                const ml = document.getElementById('sat-match-list');
                if (data.matches.length > 0) {
                    ml.innerHTML = '<strong>Matched Pairs:</strong><br>' +
                        data.matches.map(m => {
                            const cls = m.prob >= 0.8 ? 'high' : m.prob >= 0.5 ? 'medium' : 'low';
                            return `<div class="match-pair">
                                <span class="prob-badge ${cls}">${(m.prob*100).toFixed(1)}%</span>
                                ${m.pano_tags.substring(0, 30)} ↔ ${m.osm_tags.substring(0, 30)}
                            </div>`;
                        }).join('') +
                        `<div style="margin-top:6px;font-weight:600;">Total: ${data.score.toFixed(4)}</div>`;
                } else {
                    ml.innerHTML = '<em style="color:#999;">No matches above threshold</em>';
                }
            });
        }

        function probColor(p) {
            if (p < 0.01) return 'transparent';
            const r = Math.round(255 * Math.min(p * 2, 1));
            const g = Math.round(255 * Math.max(0, 1 - p * 2));
            return `rgba(${r}, ${g}, 0, ${0.2 + p * 0.6})`;
        }

        // --- Close satellite detail ---
        function closeSatDetail() {
            document.getElementById('sat-detail').style.display = 'none';
        }

        // --- Histogram hover ---
        (function() {
            const canvas = document.getElementById('histogram-canvas');
            const tooltip = document.getElementById('histogram-tooltip');

            canvas.addEventListener('mousemove', function(e) {
                if (!histogramLayout || !allSatScores) { tooltip.style.display = 'none'; return; }
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const L = histogramLayout;

                // Which bin?
                const binIdx = Math.floor((x - L.padLeft) / L.barW);
                if (binIdx < 0 || binIdx >= L.nBins || x < L.padLeft) {
                    tooltip.style.display = 'none'; return;
                }

                const hist = allSatScores.histogram;
                const count = hist.counts[binIdx];
                const posCount = hist.positive_counts[binIdx];
                const lo = hist.bin_edges[binIdx].toFixed(3);
                const hi = hist.bin_edges[binIdx + 1].toFixed(3);

                // Find a sample satellite in this bin
                const sample = hist.bin_samples ? hist.bin_samples[binIdx] : null;
                let sampleHtml = '';
                if (sample && sample.sat_idx >= 0) {
                    sampleHtml = `<div style="margin-top:6px;border-top:1px solid #eee;padding-top:6px;">
                        <strong>Sample: Sat #${sample.sat_idx}</strong>
                        ${sample.is_positive ? ' <span style="color:#2e7d32;">✓ Positive</span>' : ''}
                        <span style="color:#666;">(score: ${sample.score.toFixed(3)})</span>
                        <img src="/api/image/satellite/${sample.sat_idx}" style="width:80px;height:80px;object-fit:cover;border-radius:4px;display:block;margin:4px 0;">
                        <div style="font-size:11px;color:#555;">${(sample.landmarks || []).map(l => '<div>' + l + '</div>').join('')}</div>
                    </div>`;
                }

                tooltip.innerHTML = `<strong>Score: ${lo} – ${hi}</strong><br>
                    Count: ${count} satellites${posCount > 0 ? ` (${posCount} positive)` : ''}
                    ${sampleHtml}`;

                // Position tooltip
                const tipX = Math.min(e.clientX - rect.left + 12, rect.width - 360);
                tooltip.style.left = tipX + 'px';
                tooltip.style.top = Math.max(0, y - 30) + 'px';
                tooltip.style.display = 'block';
            });

            canvas.addEventListener('mouseleave', function() {
                tooltip.style.display = 'none';
            });

            canvas.addEventListener('click', function(e) {
                if (!histogramLayout || !allSatScores) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const L = histogramLayout;
                const binIdx = Math.floor((x - L.padLeft) / L.barW);
                if (binIdx < 0 || binIdx >= L.nBins || x < L.padLeft) return;

                const sample = allSatScores.histogram.bin_samples ? allSatScores.histogram.bin_samples[binIdx] : null;
                if (sample && sample.sat_idx >= 0) {
                    tooltip.style.display = 'none';
                    const sats = allSatScores.satellites;
                    const satData = sats.find(s => s.sat_idx === sample.sat_idx);
                    const rank = satData ? satData.rank : 0;
                    const isPos = satData ? satData.is_positive : false;
                    loadSatDetail(sample.sat_idx, rank, sample.score, isPos);
                }
            });
        })();

        // --- Keyboard nav ---
        document.addEventListener('keydown', e => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            if (e.key === 'ArrowLeft') navigate(-1);
            if (e.key === 'ArrowRight') navigate(1);
        });
    </script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/api/panoramas')
def get_panoramas():
    result = []
    for pano_id in PANO_ID_LIST:
        vigor_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
        if vigor_idx is None:
            continue
        row = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]
        n_lm = len(RAW_DATA.pano_id_to_lm_rows.get(pano_id, []))
        result.append({
            'pano_id': pano_id,
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'n_landmarks': n_lm,
        })
    return jsonify(result)


@app.route('/api/panorama/<pano_id>')
def get_panorama(pano_id):
    rows = RAW_DATA.pano_id_to_lm_rows.get(pano_id, [])
    landmarks = []
    for row_idx in rows:
        tags = RAW_DATA.pano_lm_tags[row_idx]
        tags_str = '; '.join(f'{k}={v}' for k, v in tags)
        # Count OSM landmarks with P(match) >= 0.3 for this pano landmark
        scores = RAW_DATA.cost_matrix[row_idx]
        n_strong = int((scores >= 0.3).sum())
        max_score = float(scores.max()) if len(scores) > 0 else 0.0
        landmarks.append({
            'tags_str': tags_str, 'tags': tags,
            'n_strong_matches': n_strong, 'max_match': round(max_score, 3),
        })
    return jsonify({'pano_id': pano_id, 'landmarks': landmarks})


@app.route('/api/image/panorama/<pano_id>')
def get_panorama_image(pano_id):
    vigor_idx = PANO_ID_TO_VIGOR_IDX.get(pano_id)
    if vigor_idx is None:
        return 'Not found', 404
    path = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]['path']
    return send_file(path)


@app.route('/api/image/satellite/<int:sat_idx>')
def get_satellite_image(sat_idx):
    if sat_idx < 0 or sat_idx >= len(VIGOR_DATASET._satellite_metadata):
        return 'Not found', 404
    path = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]['path']
    return send_file(path)


@app.route('/api/landmark_scores/<pano_id>/<int:lm_idx>')
def get_landmark_scores(pano_id, lm_idx):
    rows = RAW_DATA.pano_id_to_lm_rows.get(pano_id)
    if rows is None or lm_idx >= len(rows):
        return jsonify({'error': 'Invalid landmark'}), 404

    row_idx = rows[lm_idx]
    scores = RAW_DATA.cost_matrix[row_idx].tolist()

    # Top-k matches
    sorted_indices = np.argsort(-RAW_DATA.cost_matrix[row_idx])
    top_k = []
    for i in sorted_indices[:20]:
        prob = float(RAW_DATA.cost_matrix[row_idx, i])
        if prob < 0.001:
            break
        tags = RAW_DATA.osm_lm_tags[i]
        tags_str = '; '.join(f'{k}={v}' for k, v in sorted(tags.items()))
        top_k.append({'osm_col': int(i), 'prob': round(prob, 4), 'tags_str': tags_str})

    return jsonify({'scores': scores, 'top_k': top_k})


@app.route('/api/satellite_scores/<pano_id>', methods=['POST'])
def get_satellite_scores(pano_id):
    params = request.get_json(force=True)
    method = MatchingMethod(params.get('method', 'hungarian'))
    aggregation = AggregationMode(params.get('aggregation', 'sum'))
    threshold = float(params.get('prob_threshold', 0.3))

    rows = RAW_DATA.pano_id_to_lm_rows.get(pano_id)
    if rows is None:
        return jsonify({'error': 'Panorama not found'}), 404

    pano_cost = RAW_DATA.cost_matrix[rows]
    vigor_idx = PANO_ID_TO_VIGOR_IDX[pano_id]
    pano_row = VIGOR_DATASET._panorama_metadata.iloc[vigor_idx]
    positive_set = set(pano_row['positive_satellite_idxs']) | set(
        pano_row.get('semipositive_satellite_idxs', []))

    num_sats = len(VIGOR_DATASET._satellite_metadata)
    scores_array = np.zeros(num_sats, dtype=np.float32)

    for sat_idx in range(num_sats):
        cols = SAT_TO_COL_POSITIONS[sat_idx]
        if not cols:
            continue
        sub_cost = pano_cost[:, cols]
        result = match_and_aggregate(sub_cost, method, aggregation, threshold)
        scores_array[sat_idx] = result.similarity_score

    # Build sorted ranking
    ranking = np.argsort(-scores_array)
    rank_of = np.empty(num_sats, dtype=np.int32)
    rank_of[ranking] = np.arange(num_sats)

    # Find best positive rank
    best_rank = num_sats
    for pos_idx in positive_set:
        best_rank = min(best_rank, int(rank_of[pos_idx]))
    mrr = 1.0 / (best_rank + 1) if positive_set else 0.0

    # Build satellite list — return top-500 + all positives + all non-zero
    nonzero_mask = scores_array > 0
    nonzero_count = int(nonzero_mask.sum())
    top_500 = set(ranking[:500].tolist())
    include_set = top_500 | positive_set | set(np.where(nonzero_mask)[0].tolist())

    satellites = []
    for sat_idx in include_set:
        sat_idx = int(sat_idx)
        sat_meta = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]
        n_lm = len(SAT_TO_COL_POSITIONS[sat_idx])
        satellites.append({
            'sat_idx': sat_idx,
            'lat': float(sat_meta['lat']),
            'lon': float(sat_meta['lon']),
            'score': float(scores_array[sat_idx]),
            'rank': int(rank_of[sat_idx]),
            'is_positive': sat_idx in positive_set,
            'n_landmarks': n_lm,
        })

    # Histogram with bin samples
    nonzero_scores = scores_array[nonzero_mask]
    if len(nonzero_scores) > 0:
        bins = np.linspace(0, float(nonzero_scores.max()) * 1.05, 40)
    else:
        bins = np.linspace(0, 1, 40)
    counts, bin_edges = np.histogram(scores_array[scores_array > 0], bins=bins)

    pos_scores = np.array([scores_array[i] for i in positive_set if scores_array[i] > 0])
    pos_counts, _ = np.histogram(pos_scores, bins=bins) if len(pos_scores) > 0 else (
        np.zeros(len(counts)), None)

    # Pick a sample satellite per bin (closest to bin center)
    bin_samples = []
    nonzero_idxs = np.where(nonzero_mask)[0]
    for i in range(len(counts)):
        if counts[i] == 0:
            bin_samples.append({'sat_idx': -1, 'score': 0, 'is_positive': False, 'landmarks': []})
            continue
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        in_bin = nonzero_idxs[(scores_array[nonzero_idxs] >= bin_edges[i]) &
                              (scores_array[nonzero_idxs] < bin_edges[i + 1])]
        if len(in_bin) == 0:
            bin_samples.append({'sat_idx': -1, 'score': 0, 'is_positive': False, 'landmarks': []})
            continue
        closest = in_bin[np.argmin(np.abs(scores_array[in_bin] - bin_center))]
        closest = int(closest)
        # Get landmarks for this satellite
        lm_strs = []
        for col_pos in SAT_TO_COL_POSITIONS[closest][:8]:
            tags = RAW_DATA.osm_lm_tags[col_pos]
            lm_strs.append('; '.join(f'{k}={v}' for k, v in sorted(tags.items())))
        bin_samples.append({
            'sat_idx': closest,
            'score': float(scores_array[closest]),
            'is_positive': closest in positive_set,
            'landmarks': lm_strs,
        })

    return jsonify({
        'pano_id': pano_id,
        'pano_lat': float(pano_row['lat']),
        'pano_lon': float(pano_row['lon']),
        'satellites': satellites,
        'metrics': {
            'mrr': mrr,
            'best_rank': best_rank if positive_set else -1,
            'pct_nonzero': 100.0 * nonzero_count / num_sats,
            'max_score': float(scores_array.max()),
            'n_positives': len(positive_set),
        },
        'histogram': {
            'bin_edges': bin_edges.tolist(),
            'counts': counts.tolist(),
            'positive_counts': pos_counts.tolist(),
            'bin_samples': bin_samples,
        },
    })


@app.route('/api/satellite_detail/<pano_id>/<int:sat_idx>', methods=['POST'])
def get_satellite_detail(pano_id, sat_idx):
    params = request.get_json(force=True)
    method = MatchingMethod(params.get('method', 'hungarian'))
    aggregation = AggregationMode(params.get('aggregation', 'sum'))
    threshold = float(params.get('prob_threshold', 0.3))

    rows = RAW_DATA.pano_id_to_lm_rows.get(pano_id)
    if rows is None:
        return jsonify({'error': 'Panorama not found'}), 404

    cols = SAT_TO_COL_POSITIONS[sat_idx]
    if not cols:
        return jsonify({
            'cost_slice': [], 'matches': [], 'score': 0.0,
            'pano_tags': [], 'osm_tags': [],
        })

    pano_cost = RAW_DATA.cost_matrix[rows]
    sub_cost = pano_cost[:, cols]
    result = match_and_aggregate(sub_cost, method, aggregation, threshold)

    pano_tags = ['; '.join(f'{k}={v}' for k, v in RAW_DATA.pano_lm_tags[r])
                 for r in rows]
    osm_tags = ['; '.join(f'{k}={v}' for k, v in sorted(RAW_DATA.osm_lm_tags[c].items()))
                for c in cols]

    matches = []
    for pi, oi, prob in zip(result.pano_lm_indices, result.osm_lm_indices,
                            result.match_probs):
        matches.append({
            'pano_lm_idx': pi, 'osm_lm_idx': oi,
            'pano_tags': pano_tags[pi], 'osm_tags': osm_tags[oi],
            'prob': round(prob, 4),
        })

    return jsonify({
        'cost_slice': sub_cost.tolist(),
        'matches': matches,
        'score': round(result.similarity_score, 4),
        'pano_tags': pano_tags,
        'osm_tags': osm_tags,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global VIGOR_DATASET, RAW_DATA, PANO_ID_TO_VIGOR_IDX, PANO_ID_LIST
    global SAT_TO_COL_POSITIONS

    parser = argparse.ArgumentParser(description='Correspondence Explorer')
    parser.add_argument('--precomputed_data', type=str, required=True,
                        help='Path to .pt file from export_correspondence_similarity --save_raw')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to VIGOR city directory')
    parser.add_argument('--landmark_version', type=str, default=None,
                        help='Landmark version (default: auto-detect)')
    parser.add_argument('--inflation_factor', type=float, default=1.0,
                        help='Satellite patch inflation factor (must match precompute)')
    parser.add_argument('--port', type=int, default=5003)
    args = parser.parse_args()

    startup_start = time.time()
    dataset_path = Path(args.dataset_path)

    # 1. Load precomputed data
    print("Loading precomputed data...")
    data = torch.load(args.precomputed_data, weights_only=False)
    RAW_DATA = RawCorrespondenceData(
        cost_matrix=data['cost_matrix'],
        pano_id_to_lm_rows=data['pano_id_to_lm_rows'],
        pano_lm_tags=data['pano_lm_tags'],
        osm_lm_indices=data['osm_lm_indices'],
        osm_lm_tags=data['osm_lm_tags'],
    )
    print(f"  Cost matrix: {RAW_DATA.cost_matrix.shape}")
    print(f"  {len(RAW_DATA.pano_id_to_lm_rows)} panoramas, "
          f"{len(RAW_DATA.osm_lm_indices)} OSM landmarks")

    # 2. Load VigorDataset
    landmark_version = args.landmark_version
    if landmark_version is None:
        landmarks_dir = dataset_path / "landmarks"
        feather_files = list(landmarks_dir.glob("*.feather"))
        if len(feather_files) == 1:
            landmark_version = feather_files[0].stem

    print(f"Loading VigorDataset from {dataset_path}...")
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=landmark_version,
        landmark_correspondence_inflation_factor=args.inflation_factor,
    )
    VIGOR_DATASET = vd.VigorDataset(dataset_path, config)
    print(f"  {len(VIGOR_DATASET._panorama_metadata)} panos, "
          f"{len(VIGOR_DATASET._satellite_metadata)} sats")

    # 3. Build mappings
    print("Building index mappings...")
    for idx, (_, row) in enumerate(VIGOR_DATASET._panorama_metadata.iterrows()):
        PANO_ID_TO_VIGOR_IDX[row['pano_id']] = idx

    PANO_ID_LIST = [pid for pid in RAW_DATA.pano_id_to_lm_rows.keys()
                    if pid in PANO_ID_TO_VIGOR_IDX]
    print(f"  {len(PANO_ID_LIST)} panoramas with cost data")

    # 4. Pre-build sat → col positions
    osm_idx_to_col = {idx: col for col, idx in enumerate(RAW_DATA.osm_lm_indices)}
    num_sats = len(VIGOR_DATASET._satellite_metadata)
    SAT_TO_COL_POSITIONS = [[] for _ in range(num_sats)]
    sats_with_lm = 0
    for sat_idx in range(num_sats):
        sat_meta = VIGOR_DATASET._satellite_metadata.iloc[sat_idx]
        lm_idxs = sat_meta.get('landmark_idxs', [])
        if lm_idxs is None:
            continue
        cols = [osm_idx_to_col[i] for i in lm_idxs if i in osm_idx_to_col]
        SAT_TO_COL_POSITIONS[sat_idx] = cols
        if cols:
            sats_with_lm += 1
    print(f"  {sats_with_lm} / {num_sats} satellites have landmarks")

    startup_time = time.time() - startup_start
    print(f"\nStartup: {startup_time:.1f}s")
    print(f"Starting server on http://localhost:{args.port}")
    app.run(debug=True, use_reloader=False, port=args.port, host='0.0.0.0')


if __name__ == '__main__':
    main()
