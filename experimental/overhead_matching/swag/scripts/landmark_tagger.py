"""
Interactive web tool for viewing panoramas and tagging landmarks with OSM tags.

Displays pinhole-reprojected views of equirectangular panoramas and provides
fuzzy search against a city's actual OSM tag vocabulary. Users can build
landmarks interactively and save them in a format compatible with Gemini's
OSMTagExtraction schema.

Usage:
    python landmark_tagger.py \
        --panorama_dir /data/overhead_matching/datasets/VIGOR/Chicago/panorama \
        --feather /data/overhead_matching/datasets/VIGOR/Chicago/landmarks/v4_202001.feather \
        --output /tmp/tagged_landmarks.jsonl
"""

import argparse
import io
import json
import math
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, Response, jsonify, request
from PIL import Image

import shapely

from experimental.overhead_matching.swag.data.vigor_dataset import (
    compute_panorama_from_landmarks,
    load_landmark_geojson,
    load_panorama_metadata,
)
from experimental.overhead_matching.swag.scripts.panorama_to_pinhole import (
    reproject_pinhole,
    spherical_pixel_from_azimuth_elevation,
)
from experimental.overhead_matching.swag.scripts.search_osm_tags import (
    TagSearchIndex,
)

import pandas as pd

app = Flask(__name__)


class FastReprojector:
    """Cached pinhole reprojection using cv2.remap for interactive use.

    Precomputes the normalized direction grid for a given (output_shape, fov)
    so that only the rotation + remap need to run per frame.
    """

    def __init__(self):
        self._dirs = None  # (H, W, 3) normalized ray directions
        self._cache_key = None  # (H, W, fov_x, fov_y)

    def _ensure_dirs(self, output_shape, fov):
        key = (*output_shape, *fov)
        if key == self._cache_key:
            return
        H, W = output_shape
        fx = 1.0 / np.tan(fov[0] / 2.0)
        fy = 1.0 / np.tan(fov[1] / 2.0)
        row = np.linspace(-1, 1, H, dtype=np.float32)
        col = np.linspace(1, -1, W, dtype=np.float32)
        row, col = np.meshgrid(row, col, indexing="ij")
        dirs = np.stack([col, row * (fx / fy), np.full_like(col, fx)], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        self._dirs = dirs
        self._cache_key = key

    def reproject(self, panorama, output_shape, fov, yaw=0.0, pitch=0.0):
        self._ensure_dirs(output_shape, fov)
        cy, sy = np.cos(yaw, dtype=np.float32), np.sin(yaw, dtype=np.float32)
        cp, sp = np.cos(pitch, dtype=np.float32), np.sin(pitch, dtype=np.float32)
        R = np.array([
            [cy, sy * sp, sy * cp],
            [0, cp, -sp],
            [-sy, cy * sp, cy * cp],
        ], dtype=np.float32)
        dirs_rot = self._dirs @ R.T

        azimuth = np.arctan2(dirs_rot[..., 0], dirs_rot[..., 2])
        elevation = np.arcsin(np.clip(dirs_rot[..., 1], -1, 1))

        H_pano, W_pano = panorama.shape[:2]
        col_frac = (np.pi - azimuth) / (2 * np.pi)
        row_frac = elevation / np.pi + 0.5
        map_x = (col_frac * W_pano).astype(np.float32)
        map_y = (row_frac * H_pano).astype(np.float32)

        # cv2.remap handles all channels at once and uses optimized SIMD
        return cv2.remap(
            panorama, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )


# Global state populated at startup
PANORAMAS = []  # List of {"id": str, "lat": float, "lon": float, "path": Path}
PANO_CACHE = {}  # id -> np.ndarray (float32, 0-1)
TAG_INDEX = None  # TagSearchIndex
OUTPUT_PATH = None  # Path to JSONL output
SAVED_LANDMARKS = {}  # pano_id -> list of landmark dicts
REPROJECTOR = FastReprojector()
LANDMARK_METADATA = None  # GeoDataFrame from load_landmark_geojson
PANO_LANDMARK_IDXS = []  # list of list[int], parallel to PANORAMAS
PANO_CATEGORIES_GEMINI = []  # list of str, parallel to PANORAMAS
PANO_CATEGORIES_OSM = []  # list of str, parallel to PANORAMAS
GEMINI_PREDICTIONS = {}  # pano_id -> {"location_type": str, "landmarks": [...]}
PANO_LABELS = {}  # pano_idx -> "useful" | "not_useful"
TAG_OVERRIDES = {}  # (key, val) -> "always_distinctive" | "never_distinctive"
TAG_SCORES = {}  # (key, val) -> P(useful|tag)
PANO_CATEGORIES_LEARNED = []  # parallel to PANORAMAS
PANO_TAGS = []  # list of set[(key, val)], parallel to PANORAMAS — precomputed at startup
LABELS_PATH = None  # Path to labels JSON file

# DINO-based visual distinctiveness prediction
DINO_FEATURES = None     # np.ndarray (N, 3072) or None
DINO_PANO_MAP = {}       # pano_id -> row index in DINO_FEATURES
DINO_MODEL = None        # sklearn LogisticRegression or None
PANO_CATEGORIES_DINO = []  # parallel to PANORAMAS: "distinctive"/"generic"/"no_data"
DINO_CONFIDENCES = []    # parallel to PANORAMAS: float confidence or None

USEFUL_TOP_KEYS = {
    "amenity", "shop", "leisure", "tourism", "historic", "natural",
    "man_made", "office", "healthcare", "craft", "club", "sport",
}


def _categorize_panorama_gemini(pano_id, gemini_predictions):
    """Categorize a panorama based on Gemini's extracted landmarks.

    Returns one of:
      - "distinctive": Gemini found a landmark with a useful top-level tag
      - "generic": Gemini found landmarks but only generic ones (building, highway)
      - "no_predictions": no Gemini predictions for this panorama
    """
    pred = gemini_predictions.get(pano_id)
    if not pred or not pred.get("landmarks"):
        return "no_predictions"

    for lm in pred["landmarks"]:
        pt = lm.get("primary_tag")
        if not isinstance(pt, dict):
            continue
        key = pt.get("key", "")
        if key in USEFUL_TOP_KEYS:
            return "distinctive"

    return "generic"


def _categorize_panorama_osm(pano_idx):
    """Categorize a panorama based on its associated OSM landmarks.

    Returns one of:
      - "distinctive": has at least one landmark with a useful top-level tag
      - "generic": has landmarks but only generic ones (building, highway, etc.)
      - "no_landmarks": no OSM landmarks associated with this panorama
    """
    lm_idxs = PANO_LANDMARK_IDXS[pano_idx]
    if not lm_idxs:
        return "no_landmarks"

    for lm_idx in lm_idxs:
        row = LANDMARK_METADATA.iloc[lm_idx]
        for key, _value in row["pruned_props"]:
            if key in USEFUL_TOP_KEYS:
                return "distinctive"

    return "generic"


def _load_labels():
    """Load panorama labels and tag overrides from the labels JSON file."""
    global PANO_LABELS, TAG_OVERRIDES
    if LABELS_PATH is None or not LABELS_PATH.exists():
        return
    with open(LABELS_PATH) as f:
        data = json.load(f)
    PANO_LABELS = {int(k): v for k, v in data.get("panorama_labels", {}).items()}
    TAG_OVERRIDES = {}
    for tag_str, override in data.get("tag_overrides", {}).items():
        if "=" in tag_str:
            key, value = tag_str.split("=", 1)
            TAG_OVERRIDES[(key, value)] = override


def _save_labels():
    """Save panorama labels and tag overrides to the labels JSON file."""
    if LABELS_PATH is None:
        return
    data = {
        "panorama_labels": {str(k): v for k, v in PANO_LABELS.items()},
        "tag_overrides": {
            f"{k}={v}": override for (k, v), override in TAG_OVERRIDES.items()
        },
    }
    with open(LABELS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _get_pano_tags(pano_idx):
    """Get all (key, value) tags from a panorama's landmarks (precomputed)."""
    return PANO_TAGS[pano_idx] if PANO_TAGS else set()


def _precompute_pano_tags():
    """Build PANO_TAGS cache from LANDMARK_METADATA. Call once at startup."""
    global PANO_TAGS
    PANO_TAGS = []
    for pano_idx in range(len(PANORAMAS)):
        tags = set()
        for lm_idx in PANO_LANDMARK_IDXS[pano_idx]:
            row = LANDMARK_METADATA.iloc[lm_idx]
            for key, value in row["pruned_props"]:
                tags.add((key, value))
        PANO_TAGS.append(tags)
    total_tags = sum(len(t) for t in PANO_TAGS)
    print(f"Precomputed tags: {total_tags} tag instances across {len(PANO_TAGS)} panoramas")


def _recompute_learned_categories():
    """Recompute TAG_SCORES and PANO_CATEGORIES_LEARNED from labels."""
    global TAG_SCORES, PANO_CATEGORIES_LEARNED

    if not PANO_LABELS:
        TAG_SCORES = {}
        PANO_CATEGORIES_LEARNED = ["no_data"] * len(PANORAMAS)
        return

    # Count per-tag occurrences in labeled panoramas
    tag_useful = {}  # (key, val) -> count in "useful" panos
    tag_total = {}   # (key, val) -> count in any labeled pano
    for pano_idx, label in PANO_LABELS.items():
        tags = _get_pano_tags(pano_idx)
        for tag in tags:
            tag_total[tag] = tag_total.get(tag, 0) + 1
            if label == "useful":
                tag_useful[tag] = tag_useful.get(tag, 0) + 1

    # Laplace-smoothed scores
    TAG_SCORES = {}
    for tag, total in tag_total.items():
        useful = tag_useful.get(tag, 0)
        TAG_SCORES[tag] = (useful + 1) / (total + 2)

    # Apply overrides
    for tag, override in TAG_OVERRIDES.items():
        if override == "always_distinctive":
            TAG_SCORES[tag] = 1.0
        elif override == "never_distinctive":
            TAG_SCORES[tag] = 0.0

    # Classify all panoramas
    PANO_CATEGORIES_LEARNED = []
    for pano_idx in range(len(PANORAMAS)):
        lm_idxs = PANO_LANDMARK_IDXS[pano_idx]
        if not lm_idxs:
            PANO_CATEGORIES_LEARNED.append("no_landmarks")
            continue
        tags = _get_pano_tags(pano_idx)
        if not tags:
            PANO_CATEGORIES_LEARNED.append("no_landmarks")
            continue
        max_score = max(
            (TAG_SCORES.get(tag, 0.5) for tag in tags), default=0.0
        )
        if max_score >= 0.6:
            PANO_CATEGORIES_LEARNED.append("distinctive")
        else:
            PANO_CATEGORIES_LEARNED.append("generic")


def _retrain_dino_model():
    """Retrain logistic regression on DINO features from labeled panoramas."""
    global DINO_MODEL, PANO_CATEGORIES_DINO, DINO_CONFIDENCES

    if DINO_FEATURES is None:
        return

    # Collect features and labels for labeled panoramas that have DINO data
    X, y = [], []
    for pano_idx, label in PANO_LABELS.items():
        pano_id = PANORAMAS[pano_idx]["id"]
        row_idx = DINO_PANO_MAP.get(pano_id)
        if row_idx is None:
            continue
        X.append(DINO_FEATURES[row_idx])
        y.append(1 if label == "useful" else 0)

    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos < 5 or n_neg < 5:
        DINO_MODEL = None
        PANO_CATEGORIES_DINO = ["no_data"] * len(PANORAMAS)
        DINO_CONFIDENCES = [None] * len(PANORAMAS)
        return

    from sklearn.linear_model import LogisticRegression

    X = np.array(X)
    y = np.array(y)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    DINO_MODEL = model

    # Predict for all panoramas with features
    PANO_CATEGORIES_DINO = []
    DINO_CONFIDENCES = []
    for pano_idx in range(len(PANORAMAS)):
        pano_id = PANORAMAS[pano_idx]["id"]
        row_idx = DINO_PANO_MAP.get(pano_id)
        if row_idx is None:
            PANO_CATEGORIES_DINO.append("no_data")
            DINO_CONFIDENCES.append(None)
            continue
        feat = DINO_FEATURES[row_idx].reshape(1, -1)
        prob = model.predict_proba(feat)[0]
        p_useful = prob[1] if model.classes_[1] == 1 else prob[0]
        PANO_CATEGORIES_DINO.append("distinctive" if p_useful >= 0.5 else "generic")
        DINO_CONFIDENCES.append(round(float(p_useful), 3))

    from collections import Counter
    counts = Counter(PANO_CATEGORIES_DINO)
    print(f"DINO model retrained ({n_pos}+/{n_neg}-): "
          f"distinctive={counts.get('distinctive', 0)}, "
          f"generic={counts.get('generic', 0)}, "
          f"no_data={counts.get('no_data', 0)}")


def _load_gemini_predictions(predictions_dir):
    """Load Gemini landmark predictions from a sentences directory.

    Expects structure: predictions_dir/sentences/panorama_request_*/*/predictions.jsonl
    Each line is a raw Gemini API response with nested JSON.
    Returns dict mapping pano_id -> {"location_type": str, "landmarks": [...]}.
    """
    import ast

    results = {}
    sentences_dir = predictions_dir / "sentences"
    if not sentences_dir.exists():
        print(f"  No sentences dir at {sentences_dir}")
        return results

    parse_errors = 0
    for request_dir in sorted(sentences_dir.glob("panorama_request_*")):
        for pred_dir in request_dir.iterdir():
            pred_file = pred_dir / "predictions.jsonl"
            if not pred_file.exists():
                continue
            with open(pred_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue

                    key = record.get("key", "")
                    pano_id = key.split(",")[0]
                    if not pano_id:
                        continue

                    try:
                        text = record["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError, TypeError):
                        parse_errors += 1
                        continue

                    try:
                        prediction = json.loads(text)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            prediction = ast.literal_eval(text)
                        except (ValueError, SyntaxError):
                            parse_errors += 1
                            continue

                    landmarks = prediction.get("landmarks", [])
                    if not isinstance(landmarks, list):
                        continue

                    if pano_id not in results:  # keep first occurrence
                        results[pano_id] = {
                            "location_type": prediction.get("location_type", "unknown"),
                            "landmarks": landmarks,
                        }

    if parse_errors:
        print(f"  {parse_errors} parse errors skipped")
    return results


def _load_panorama(idx):
    """Load and cache a panorama as float32 array."""
    pano = PANORAMAS[idx]
    if pano["id"] not in PANO_CACHE:
        img = Image.open(pano["path"])
        PANO_CACHE[pano["id"]] = np.array(img).astype(np.float32) / 255.0
    return PANO_CACHE[pano["id"]]


def _array_to_jpeg_bytes(arr_float):
    """Convert float32 (0-1) array to JPEG bytes."""
    arr_uint8 = np.clip(arr_float * 255.0, 0, 255).astype(np.uint8)
    # cv2.imencode is faster than PIL for JPEG
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<title>Panorama Landmark Tagger</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; }
.top-bar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; }
.top-bar button { padding: 6px 14px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 14px; }
.top-bar button:hover { background: #0f3460; }
.top-bar input { padding: 6px 10px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; width: 80px; font-size: 14px; }
.top-bar .pano-info { color: #a0a0c0; font-size: 13px; }
.pinhole-row { display: flex; gap: 4px; padding: 8px 16px; background: #16213e; justify-content: center; }
.pinhole-row img { height: 180px; border-radius: 4px; border: 2px solid #0f3460; cursor: pointer; transition: border-color 0.15s; }
.pinhole-row img:hover { border-color: #a0c0ff; }
.middle { display: flex; padding: 8px 16px; gap: 16px; justify-content: center; }
.interactive-view { position: relative; width: 768px; flex-shrink: 0; }
.interactive-view img { width: 768px; height: 768px; border-radius: 4px; cursor: grab; display: block; }
#landmark-map { width: 500px; height: 768px; border-radius: 4px; border: 1px solid #0f3460; flex-shrink: 0; }
.interactive-view img:active { cursor: grabbing; }
.interactive-view .overlay { position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #a0c0ff; pointer-events: none; }
.bottom { display: flex; gap: 16px; padding: 8px 16px; min-height: 320px; }
.panel { flex: 1; background: #16213e; border-radius: 8px; padding: 12px; border: 1px solid #0f3460; overflow-y: auto; max-height: 400px; }
.panel h3 { margin-bottom: 8px; color: #a0c0ff; font-size: 14px; }
.search-input { width: 100%; padding: 8px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; margin-bottom: 8px; font-size: 14px; }
.tag-result { padding: 6px 8px; cursor: pointer; border-radius: 4px; font-size: 13px; display: flex; justify-content: space-between; }
.tag-result:hover { background: #0f3460; }
.tag-result .count { color: #a0a0c0; font-size: 12px; }
.badge-primary { font-size: 10px; padding: 1px 5px; background: #2d6a4f; border-radius: 3px; color: #a0e0c0; margin-left: 4px; }
.badge-additional { font-size: 10px; padding: 1px 5px; background: #0f3460; border-radius: 3px; color: #a0a0c0; margin-left: 4px; }
.context-tag { padding: 4px 8px; cursor: pointer; border-radius: 4px; font-size: 12px; display: flex; justify-content: space-between; margin: 2px 0; }
.context-tag:hover { background: #0f3460; }
.context-tag.selected { background: #1b4332; border: 1px solid #2d6a4f; }
.landmark-form { margin-top: 12px; padding-top: 12px; border-top: 1px solid #0f3460; }
.landmark-form label { display: block; font-size: 12px; color: #a0a0c0; margin-bottom: 4px; }
.landmark-form input, .landmark-form select, .landmark-form textarea {
    width: 100%; padding: 6px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; margin-bottom: 8px; font-size: 13px;
}
.landmark-form textarea { resize: vertical; min-height: 50px; }
.btn-primary { padding: 8px 16px; background: #2d6a4f; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
.btn-primary:hover { background: #40916c; }
.btn-danger { padding: 4px 10px; background: #6a2d2d; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; }
.btn-danger:hover { background: #913030; }
.landmark-item { padding: 8px; margin: 4px 0; background: #1a1a2e; border-radius: 4px; border: 1px solid #0f3460; font-size: 13px; display: flex; justify-content: space-between; align-items: start; }
.landmark-item .tags { flex: 1; }
.landmark-item .tag { display: inline-block; padding: 2px 6px; background: #0f3460; border-radius: 3px; margin: 1px; font-size: 11px; }
.landmark-item .primary-tag { background: #2d6a4f; }
.selected-primary { padding: 6px 8px; background: #1b4332; border: 1px solid #2d6a4f; border-radius: 4px; margin-bottom: 8px; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
.selected-primary .clear { cursor: pointer; color: #ff6b6b; font-size: 12px; }
.ctx-link { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }
.ctx-link:hover { color: #a0c0ff; }
.label-btn { padding: 5px 12px; border: 2px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 13px; }
.label-btn:hover { background: #0f3460; }
.label-btn.active-useful { background: #1b4332; border-color: #2d6a4f; color: #a0e0c0; }
.label-btn.active-not-useful { background: #4a1c1c; border-color: #6a2d2d; color: #ff9090; }
.label-stats { color: #a0a0c0; font-size: 12px; margin-left: 4px; }
.override-btn { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; border-radius: 50%; border: 1px solid #555; background: transparent; cursor: pointer; font-size: 12px; margin-right: 6px; flex-shrink: 0; }
.override-btn.always { background: #1b4332; border-color: #2d6a4f; color: #a0e0c0; }
.override-btn.never { background: #4a1c1c; border-color: #6a2d2d; color: #ff6b6b; }
.tag-score-row { display: flex; justify-content: space-between; align-items: center; padding: 3px 8px; font-size: 12px; border-radius: 3px; margin: 1px 0; }
.tag-score-bar { height: 4px; border-radius: 2px; margin-top: 2px; }
</style>
</head>
<body>

<div class="top-bar">
    <button onclick="navigate(-1)">&larr; Prev</button>
    <button onclick="navigate(1)">Next &rarr;</button>
    <span>Go to:</span>
    <input type="number" id="goto-idx" min="0" placeholder="#" onkeydown="if(event.key==='Enter') goToIndex()">
    <span>ID:</span>
    <input type="text" id="search-id" style="width:200px" placeholder="pano ID" onkeydown="if(event.key==='Enter') searchById()">
    <span id="pano-info" class="pano-info"></span>
    <span style="margin-left:12px; border-left:1px solid #0f3460; padding-left:12px; display:flex; align-items:center; gap:6px;">
        <button class="label-btn" id="btn-useful" onclick="setLabel('useful')" title="Useful (u)">Useful (u)</button>
        <button class="label-btn" id="btn-not-useful" onclick="setLabel('not_useful')" title="Not Useful (n)">Not Useful (n)</button>
        <button class="label-btn" id="btn-clear-label" onclick="setLabel(null)" title="Clear (x)">Clear</button>
        <span class="label-stats" id="label-stats"></span>
    </span>
    <span id="dino-prediction" style="margin-left:8px; font-size:12px; color:#a0a0c0;"></span>
    <span style="margin-left:auto; color:#a0a0c0; font-size:12px;" id="save-status"></span>
</div>

<div class="pinhole-row" id="pinhole-row"></div>

<div class="middle">
    <div class="interactive-view">
        <img id="interactive-img" src="" draggable="false">
        <div class="overlay" id="view-overlay">yaw: 0.0  pitch: 0.0  fov: 90</div>
    </div>
    <div id="landmark-map"></div>
</div>

<div class="bottom">
    <div class="panel" id="search-panel">
        <h3>Tag Search</h3>
        <input type="text" class="search-input" id="tag-search" placeholder="Search OSM tags (e.g. tennis, restaurant)..." oninput="debounceSearch()">
        <div id="search-results"></div>

        <div id="context-section" style="display:none; margin-top:12px; padding-top:12px; border-top: 1px solid #0f3460;">
            <h3>Co-occurring tags for <span id="context-tag-label"></span></h3>
            <div id="context-results"></div>
        </div>
    </div>

    <div class="panel" id="builder-panel">
        <h3>Landmark Builder</h3>

        <div id="primary-tag-display" style="display:none;">
            <label>Primary Tag</label>
            <div class="selected-primary">
                <span id="primary-tag-text" class="ctx-link" onclick="getContextForPrimary()"></span>
                <span class="clear" onclick="clearPrimaryTag()">&times; clear</span>
            </div>
        </div>

        <div id="additional-tags-display" style="display:none;">
            <label>Additional Tags</label>
            <div id="additional-tags-list"></div>
        </div>

        <div class="landmark-form" id="landmark-form">
            <label>Description</label>
            <textarea id="lm-description" placeholder="What is this landmark?"></textarea>
            <label>Confidence</label>
            <select id="lm-confidence">
                <option value="high">high</option>
                <option value="medium" selected>medium</option>
                <option value="low">low</option>
            </select>
            <button class="btn-primary" onclick="addLandmark()">Add Landmark</button>
        </div>

        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #0f3460;">
            <h3>Landmarks for this panorama (<span id="landmark-count">0</span>)</h3>
            <div id="landmark-list"></div>
        </div>

    </div>

    <div class="panel" id="predictions-panel">
        <h3>Gemini predictions</h3>
        <div id="gemini-list" style="font-size:12px; color:#a0a0c0;">Loading...</div>

        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #0f3460;">
            <h3 style="cursor:pointer;" onclick="document.getElementById('tag-scores-body').style.display = document.getElementById('tag-scores-body').style.display === 'none' ? 'block' : 'none'">
                Tag Scores <span style="font-size:11px;color:#a0a0c0;">(click to toggle)</span>
            </h3>
            <div id="tag-scores-body" style="display:none; font-size:12px; color:#a0a0c0;">Loading...</div>
        </div>
    </div>
</div>

<script>
// State
let currentIndex = 0;
let totalPanoramas = 0;
let panoramaList = [];
let viewYaw = 0.0;
let viewPitch = 0.0;
let viewFov = 90;
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let dragStartYaw = 0, dragStartPitch = 0;
let primaryTag = null;   // {key, value}
let additionalTags = []; // [{key, value}, ...]
let landmarks = {};      // panoId -> [landmark, ...]
let searchTimeout = null;
let interactiveImg = null;
let loadingReproject = false;
let lmap = null;
let landmarkLayer = null;
let currentLabel = null;  // "useful", "not_useful", or null
let labelStats = {total: 0, n_useful: 0, n_not_useful: 0};

// Init
window.addEventListener('DOMContentLoaded', async () => {
    interactiveImg = document.getElementById('interactive-img');

    // Init Leaflet map
    lmap = L.map('landmark-map', {zoomControl: true}).setView([41.88, -87.63], 18);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OSM', maxZoom: 20,
    }).addTo(lmap);
    landmarkLayer = L.layerGroup().addTo(lmap);

    const resp = await fetch('/api/panorama_list');
    const data = await resp.json();
    panoramaList = data.panoramas;
    totalPanoramas = panoramaList.length;
    const params = new URLSearchParams(window.location.search);
    const gotoIdx = parseInt(params.get('goto'));
    loadPanorama(!isNaN(gotoIdx) && gotoIdx >= 0 && gotoIdx < totalPanoramas ? gotoIdx : 0);
    setupDrag();
});

function setupDrag() {
    const img = interactiveImg;
    img.addEventListener('mousedown', e => {
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        dragStartYaw = viewYaw;
        dragStartPitch = viewPitch;
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        const sensitivity = viewFov / 600;
        viewYaw = dragStartYaw + dx * sensitivity;
        viewPitch = Math.max(-80, Math.min(80, dragStartPitch + dy * sensitivity));
        updateOverlay();
        debouncedReproject();
    });
    window.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            loadReproject();
        }
    });
    img.addEventListener('wheel', e => {
        e.preventDefault();
        viewFov = Math.max(20, Math.min(150, viewFov + e.deltaY * 0.1));
        updateOverlay();
        loadReproject();
    });
}

let reprojectTimeout = null;
function debouncedReproject() {
    if (reprojectTimeout) clearTimeout(reprojectTimeout);
    reprojectTimeout = setTimeout(() => loadReproject(), 80);
}

function updateOverlay() {
    document.getElementById('view-overlay').textContent =
        `yaw: ${viewYaw.toFixed(1)}  pitch: ${viewPitch.toFixed(1)}  fov: ${viewFov.toFixed(0)}`;
}

async function loadPanorama(idx) {
    // Save current landmarks before navigating
    await saveLandmarks();

    currentIndex = idx;
    const pano = panoramaList[idx];
    const mapsUrl = `https://www.google.com/maps/place/${pano.lat},${pano.lon}/@${pano.lat},${pano.lon},18z`;
    document.getElementById('pano-info').innerHTML =
        `${idx + 1}/${totalPanoramas}  |  ${pano.id}  |  <a href="${mapsUrl}" target="_blank" style="color:#a0c0ff;">(${pano.lat.toFixed(6)}, ${pano.lon.toFixed(6)})</a>`;

    // Load 4 pinhole views
    const row = document.getElementById('pinhole-row');
    row.innerHTML = '';
    for (const yaw of [0, 90, 180, 270]) {
        const img = document.createElement('img');
        img.src = `/api/image/pinhole/${idx}/${yaw}`;
        img.title = `${yaw}\u00b0 — click to jump`;
        img.onclick = () => { viewYaw = yaw; viewPitch = 0; viewFov = 90; updateOverlay(); loadReproject(); };
        row.appendChild(img);
    }

    // Reset interactive view
    viewYaw = 0; viewPitch = 0; viewFov = 90;
    updateOverlay();
    loadReproject();

    // Load saved landmarks for this pano
    renderLandmarkList();

    // Load nearby OSM landmarks on map
    loadNearbyLandmarks(idx);
    loadGeminiPredictions(idx);
    loadPanoLabel(idx);
    loadTagScores(idx);
    loadDinoPrediction(idx);
}

async function loadGeminiPredictions(idx) {
    const container = document.getElementById('gemini-list');
    container.innerHTML = '<span style="color:#a0a0c0;">Loading...</span>';
    const resp = await fetch(`/api/gemini_predictions/${idx}`);
    const data = await resp.json();
    if (!data.landmarks || data.landmarks.length === 0) {
        container.innerHTML = '<span style="color:#a0a0c0;">None</span>';
        return;
    }
    const locType = data.location_type ? `<div style="margin-bottom:6px;color:#a0c0ff;">Location: ${data.location_type}</div>` : '';
    container.innerHTML = locType + data.landmarks.map(lm => {
        const pt = lm.primary_tag || {};
        const primary = pt.key ? `<span style="display:inline-block;padding:1px 5px;background:#2d6a4f;border-radius:3px;margin:1px;font-size:11px;">${pt.key}=${pt.value}</span>` : '';
        const addl = (lm.additional_tags || []).map(t =>
            `<span style="display:inline-block;padding:1px 5px;background:#0f3460;border-radius:3px;margin:1px;font-size:11px;">${t.key}=${t.value}</span>`
        ).join('');
        const conf = lm.confidence ? ` <span style="color:#a0a0c0;">[${lm.confidence}]</span>` : '';
        const desc = lm.description ? `<div style="color:#808090;font-size:11px;margin-top:1px;">${lm.description}</div>` : '';
        return `<div style="padding:4px 0;border-bottom:1px solid #0f3460;">${primary} ${addl}${conf}${desc}</div>`;
    }).join('');
}

async function loadNearbyLandmarks(idx) {
    const pano = panoramaList[idx];
    lmap.setView([pano.lat, pano.lon], 18);
    landmarkLayer.clearLayers();

    // Panorama marker
    L.circleMarker([pano.lat, pano.lon], {
        radius: 6, fillColor: '#ff4444', color: '#fff', weight: 2, fillOpacity: 1,
    }).bindPopup('Panorama location').addTo(landmarkLayer);

    const resp = await fetch(`/api/nearby_landmarks/${idx}`);
    const geojson = await resp.json();
    L.geoJSON(geojson, {
        style: { color: '#4fc3f7', weight: 2, fillOpacity: 0.3 },
        pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
            radius: 5, fillColor: '#4fc3f7', color: '#fff', weight: 1, fillOpacity: 0.7,
        }),
        onEachFeature: (feature, layer) => {
            const props = feature.properties;
            const lines = Object.entries(props)
                .filter(([k,v]) => v && !k.startsWith('addr:') && !k.startsWith('tiger:'))
                .slice(0, 12)
                .map(([k,v]) => `<b>${k}</b>: ${v}`);
            if (lines.length) layer.bindPopup(lines.join('<br>'), {maxWidth: 300});
        },
    }).addTo(landmarkLayer);
}

async function loadReproject() {
    if (loadingReproject) return;
    loadingReproject = true;
    const yawRad = viewYaw * Math.PI / 180;
    const pitchRad = viewPitch * Math.PI / 180;
    const fovRad = viewFov * Math.PI / 180;
    const url = `/api/image/reproject/${currentIndex}?yaw=${yawRad}&pitch=${pitchRad}&fov=${fovRad}`;
    try {
        // Use fetch to detect errors before setting src
        const resp = await fetch(url);
        if (resp.ok) {
            const blob = await resp.blob();
            const objectUrl = URL.createObjectURL(blob);
            const oldSrc = interactiveImg.src;
            interactiveImg.src = objectUrl;
            if (oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);
        }
    } finally {
        loadingReproject = false;
    }
}

function navigate(delta) {
    const newIdx = Math.max(0, Math.min(totalPanoramas - 1, currentIndex + delta));
    if (newIdx !== currentIndex) loadPanorama(newIdx);
}

function goToIndex() {
    const idx = parseInt(document.getElementById('goto-idx').value);
    if (!isNaN(idx) && idx >= 0 && idx < totalPanoramas) loadPanorama(idx);
}

function searchById() {
    const id = document.getElementById('search-id').value.trim();
    const idx = panoramaList.findIndex(p => p.id === id);
    if (idx >= 0) loadPanorama(idx);
    else alert('Panorama ID not found');
}

// Top-level OSM keys (only these can be primary tags)
const TOP_LEVEL_KEYS = new Set([
    'advertising', 'aerialway', 'aeroway', 'amenity', 'barrier', 'boundary',
    'building', 'club', 'craft', 'departures_board', 'education', 'emergency',
    'geological', 'healthcare', 'highway', 'historic', 'landcover', 'landuse',
    'leisure', 'man_made', 'military', 'natural', 'office', 'piste:type',
    'place', 'power', 'public_transport', 'railway', 'route', 'shop',
    'telecom', 'tourism', 'waterway',
]);

function isTopLevel(key) { return TOP_LEVEL_KEYS.has(key); }

// Tag search
function debounceSearch() {
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(doSearch, 250);
}

async function doSearch() {
    const q = document.getElementById('tag-search').value.trim();
    if (q.length < 2) { document.getElementById('search-results').innerHTML = ''; return; }
    const resp = await fetch(`/api/search_tags?q=${encodeURIComponent(q)}`);
    const data = await resp.json();
    const container = document.getElementById('search-results');
    container.innerHTML = data.results.map(r => {
        const topLevel = isTopLevel(r.key);
        const action = topLevel
            ? `selectPrimaryTag('${r.key}','${r.value}')`
            : `selectAdditionalFromSearch('${r.key}','${r.value}')`;
        const badge = topLevel ? '<span class="badge-primary">primary</span>' : '<span class="badge-additional">additional</span>';
        const overrideHtml = renderOverrideBtn(r.key, r.value);
        return `<div class="tag-result" onclick="${action}">
            ${overrideHtml}
            <span style="flex:1;">${r.key}=<b>${r.value}</b> ${badge}</span>
            <span class="count">(${r.count})</span>
        </div>`;
    }).join('');
}

function selectAdditionalFromSearch(key, value) {
    if (!additionalTags.find(t => t.key === key && t.value === value)) {
        additionalTags.push({key, value});
        renderAdditionalTags();
    }
    if (additionalTags.length > 0) {
        document.getElementById('additional-tags-display').style.display = 'block';
    }
}

function selectPrimaryTag(key, value) {
    primaryTag = {key, value};
    document.getElementById('primary-tag-display').style.display = 'block';
    document.getElementById('primary-tag-text').textContent = `${key}=${value}`;
    document.getElementById('context-section').style.display = 'none';
}

function clearPrimaryTag() {
    primaryTag = null;
    additionalTags = [];
    document.getElementById('primary-tag-display').style.display = 'none';
    document.getElementById('additional-tags-display').style.display = 'none';
    document.getElementById('context-section').style.display = 'none';
}

function getContextForPrimary() {
    if (primaryTag) getContext(primaryTag.key, primaryTag.value);
}

async function getContext(key, value) {
    const resp = await fetch(`/api/tag_context?key=${encodeURIComponent(key)}&value=${encodeURIComponent(value)}`);
    const data = await resp.json();
    const section = document.getElementById('context-section');
    section.style.display = 'block';
    document.getElementById('context-tag-label').textContent = `${key}=${value} (${data.total} landmarks)`;
    const container = document.getElementById('context-results');
    container.innerHTML = data.results.map(r => {
        const tagId = `${r.key}=${r.value}`;
        const topLevel = isTopLevel(r.key);
        const alreadyPrimary = primaryTag && primaryTag.key === r.key && primaryTag.value === r.value;
        const alreadyAdditional = additionalTags.some(t => t.key === r.key && t.value === r.value);
        const selected = alreadyPrimary || alreadyAdditional ? ' selected' : '';
        let action;
        if (topLevel) {
            action = `selectPrimaryTag('${r.key}','${r.value}')`;
        } else {
            action = `toggleAdditionalTag('${r.key}','${r.value}', this)`;
        }
        const badge = topLevel ? '<span class="badge-primary">primary</span>' : '';
        const overrideHtml = renderOverrideBtn(r.key, r.value);
        return `<div class="context-tag${selected}" id="ctx-${CSS.escape(tagId)}" onclick="${action}">
            ${overrideHtml}
            <span style="flex:1;">${r.key}=<b>${r.value}</b> ${badge}</span>
            <span class="count">(${r.count}/${data.total})</span>
        </div>`;
    }).join('');
}

function toggleAdditionalTag(key, value, elem) {
    const idx = additionalTags.findIndex(t => t.key === key && t.value === value);
    if (idx >= 0) {
        additionalTags.splice(idx, 1);
        elem.classList.remove('selected');
    } else {
        additionalTags.push({key, value});
        elem.classList.add('selected');
    }
    renderAdditionalTags();
}

function removeAdditionalTag(key, value) {
    additionalTags = additionalTags.filter(t => !(t.key === key && t.value === value));
    // Un-highlight in context list
    const tagId = `${key}=${value}`;
    const elem = document.getElementById(`ctx-${CSS.escape(tagId)}`);
    if (elem) elem.classList.remove('selected');
    renderAdditionalTags();
}

function renderAdditionalTags() {
    const display = document.getElementById('additional-tags-display');
    const list = document.getElementById('additional-tags-list');
    if (additionalTags.length === 0) {
        display.style.display = 'none';
        return;
    }
    display.style.display = 'block';
    list.innerHTML = additionalTags.map(t =>
        `<span style="display:inline-flex; align-items:center; gap:4px; padding:2px 6px; background:#0f3460; border-radius:3px; margin:2px; font-size:12px;">` +
        `<span class="ctx-link" onclick="getContext('${t.key}','${t.value}')">${t.key}=${t.value}</span>` +
        `<span style="cursor:pointer; color:#ff6b6b;" onclick="removeAdditionalTag('${t.key}','${t.value}')">&times;</span></span>`
    ).join('');
}

function addLandmark() {
    const desc = document.getElementById('lm-description').value.trim();
    if (!primaryTag && additionalTags.length === 0 && !desc) return;
    const panoId = panoramaList[currentIndex].id;
    if (!landmarks[panoId]) landmarks[panoId] = [];
    const lm = {
        additional_tags: [...additionalTags],
        description: desc,
        confidence: document.getElementById('lm-confidence').value,
    };
    if (primaryTag) lm.primary_tag = {...primaryTag};
    landmarks[panoId].push(lm);
    // Reset builder
    clearPrimaryTag();
    document.getElementById('lm-description').value = '';
    document.getElementById('lm-confidence').value = 'medium';
    document.getElementById('tag-search').value = '';
    document.getElementById('search-results').innerHTML = '';
    renderLandmarkList();
}

function removeLandmark(idx) {
    const panoId = panoramaList[currentIndex].id;
    if (landmarks[panoId]) {
        landmarks[panoId].splice(idx, 1);
        if (landmarks[panoId].length === 0) delete landmarks[panoId];
    }
    renderLandmarkList();
}

function renderLandmarkList() {
    const panoId = panoramaList[currentIndex].id;
    const lms = landmarks[panoId] || [];
    document.getElementById('landmark-count').textContent = lms.length;
    const container = document.getElementById('landmark-list');
    if (lms.length === 0) {
        container.innerHTML = '<div style="color:#a0a0c0; font-size:12px; padding:8px;">No landmarks yet</div>';
        return;
    }
    container.innerHTML = lms.map((lm, i) => {
        const allTags = [
            ...(lm.primary_tag ? [`<span class="tag primary-tag">${lm.primary_tag.key}=${lm.primary_tag.value}</span>`] : []),
            ...lm.additional_tags.map(t => `<span class="tag">${t.key}=${t.value}</span>`)
        ].join('');
        const desc = lm.description ? `<div style="font-size:11px;color:#a0a0c0;margin-top:2px;">${lm.description}</div>` : '';
        return `<div class="landmark-item">
            <div class="tags">${allTags} <span style="font-size:11px;color:#a0a0c0;">[${lm.confidence}]</span>${desc}</div>
            <button class="btn-danger" onclick="removeLandmark(${i})">&times;</button>
        </div>`;
    }).join('');
}

async function saveLandmarks() {
    const panoId = panoramaList[currentIndex]?.id;
    if (!panoId) return;
    const lms = landmarks[panoId];
    if (!lms || lms.length === 0) return;
    const pano = panoramaList[currentIndex];
    const payload = {
        key: `${pano.id},${pano.lat},${pano.lon},`,
        landmarks: lms,
    };
    try {
        const resp = await fetch('/api/save_landmarks', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
        });
        if (resp.ok) {
            document.getElementById('save-status').textContent = `Saved ${lms.length} landmarks for ${panoId}`;
            setTimeout(() => document.getElementById('save-status').textContent = '', 3000);
        }
    } catch (e) {
        console.error('Save failed:', e);
    }
}

// --- Panorama labeling ---

async function loadPanoLabel(idx) {
    const resp = await fetch(`/api/pano_label/${idx}`);
    const data = await resp.json();
    currentLabel = data.label || null;
    updateLabelButtons();
}

async function setLabel(label) {
    // Toggle: if clicking the same label, clear it
    if (label === currentLabel) label = null;
    const resp = await fetch('/api/pano_label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: currentIndex, label: label}),
    });
    const data = await resp.json();
    currentLabel = label;
    labelStats = {total: data.total_labels, n_useful: data.n_useful, n_not_useful: data.n_not_useful};
    updateLabelButtons();
    loadTagScores(currentIndex);
    loadDinoPrediction(currentIndex);
}

function updateLabelButtons() {
    const btnU = document.getElementById('btn-useful');
    const btnN = document.getElementById('btn-not-useful');
    btnU.className = 'label-btn' + (currentLabel === 'useful' ? ' active-useful' : '');
    btnN.className = 'label-btn' + (currentLabel === 'not_useful' ? ' active-not-useful' : '');
    document.getElementById('label-stats').textContent =
        labelStats.total > 0 ? `${labelStats.n_useful}/${labelStats.total} labeled useful` : '';
}

// --- Tag scores ---

async function loadTagScores(idx) {
    const container = document.getElementById('tag-scores-body');
    const resp = await fetch(`/api/tag_scores/${idx}`);
    const data = await resp.json();
    if (!data.tags || data.tags.length === 0) {
        container.innerHTML = '<span>No tags</span>';
        return;
    }
    container.innerHTML = data.tags.map(t => {
        const pct = (t.score * 100).toFixed(0);
        const barColor = t.score >= 0.6 ? '#2d6a4f' : t.score >= 0.4 ? '#e6a817' : '#6a2d2d';
        const overrideIcon = t.override === 'always_distinctive' ? ' <span style="color:#a0e0c0;">&#10003;</span>'
            : t.override === 'never_distinctive' ? ' <span style="color:#ff6b6b;">&#10007;</span>' : '';
        return `<div class="tag-score-row">
            <span>${t.key}=<b>${t.value}</b>${overrideIcon}</span>
            <span>${pct}%</span>
        </div>
        <div class="tag-score-bar" style="width:${pct}%;background:${barColor};"></div>`;
    }).join('');
}

// --- DINO prediction ---

async function loadDinoPrediction(idx) {
    const el = document.getElementById('dino-prediction');
    try {
        const resp = await fetch(`/api/dino_prediction/${idx}`);
        const data = await resp.json();
        if (data.category === 'no_data' || data.confidence === null) {
            el.textContent = '';
        } else {
            const pct = (data.confidence * 100).toFixed(0);
            const color = data.category === 'distinctive' ? '#a0e0c0' : '#e6a817';
            el.innerHTML = `DINO: <span style="color:${color};font-weight:bold;">${pct}% ${data.category}</span>`;
        }
    } catch(e) {
        el.textContent = '';
    }
}

// --- Tag overrides ---

async function cycleOverride(key, value, btn) {
    const states = [null, 'always_distinctive', 'never_distinctive'];
    const labels = ['\u25CB', '\u2713', '\u2717'];
    const classes = ['', 'always', 'never'];
    let curIdx = 0;
    if (btn.classList.contains('always')) curIdx = 1;
    else if (btn.classList.contains('never')) curIdx = 2;
    const nextIdx = (curIdx + 1) % 3;
    const override = states[nextIdx];
    await fetch('/api/tag_override', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({key, value, override}),
    });
    btn.className = 'override-btn ' + classes[nextIdx];
    btn.textContent = labels[nextIdx];
    loadTagScores(currentIndex);
}

function renderOverrideBtn(key, value) {
    // Returns inline HTML for the override toggle button
    return `<button class="override-btn" onclick="event.stopPropagation(); cycleOverride('${key}','${value}', this)" title="Cycle override">\u25CB</button>`;
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') { navigate(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navigate(1); e.preventDefault(); }
    if (e.key === 'u') { setLabel('useful'); e.preventDefault(); }
    if (e.key === 'n') { setLabel('not_useful'); e.preventDefault(); }
    if (e.key === 'x') { setLabel(null); e.preventDefault(); }
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(HTML_PAGE, content_type="text/html")


@app.route("/api/panorama_list")
def panorama_list():
    return jsonify(
        {
            "panoramas": [
                {"id": p["id"], "lat": p["lat"], "lon": p["lon"]}
                for p in PANORAMAS
            ]
        }
    )


@app.route("/api/image/pinhole/<int:index>/<int:yaw_deg>")
def pinhole_image(index, yaw_deg):
    if index < 0 or index >= len(PANORAMAS):
        return "Index out of range", 404
    pano_arr = _load_panorama(index)
    fov = math.pi / 2.0  # 90 deg
    yaw_rad = math.radians(yaw_deg)
    out = REPROJECTOR.reproject(
        pano_arr,
        output_shape=(256, 256),
        fov=(fov, fov),
        yaw=yaw_rad,
        pitch=0.0,
    )
    jpeg = _array_to_jpeg_bytes(out)
    return Response(jpeg, content_type="image/jpeg")


@app.route("/api/image/reproject/<int:index>")
def reproject_image(index):
    if index < 0 or index >= len(PANORAMAS):
        return "Index out of range", 404
    yaw = float(request.args.get("yaw", 0.0))
    pitch = float(request.args.get("pitch", 0.0))
    fov = float(request.args.get("fov", math.pi / 2.0))
    pano_arr = _load_panorama(index)
    out = REPROJECTOR.reproject(
        pano_arr,
        output_shape=(768, 768),
        fov=(fov, fov),
        yaw=yaw,
        pitch=pitch,
    )
    jpeg = _array_to_jpeg_bytes(out)
    return Response(jpeg, content_type="image/jpeg")


@app.route("/api/search_tags")
def search_tags():
    q = request.args.get("q", "")
    if len(q) < 2:
        return jsonify({"results": []})
    results = TAG_INDEX.search_tags(q, limit=20)
    return jsonify(
        {
            "results": [
                {"key": k, "value": v, "count": c} for k, v, c in results
            ]
        }
    )


@app.route("/api/tag_context")
def tag_context():
    key = request.args.get("key", "")
    value = request.args.get("value", "")
    if not key or not value:
        return jsonify({"results": [], "total": 0})
    results, total = TAG_INDEX.get_tag_context(key, value, limit=20)
    return jsonify(
        {
            "results": [
                {"key": k, "value": v, "count": c} for k, v, c in results
            ],
            "total": total,
        }
    )


@app.route("/api/save_landmarks", methods=["POST"])
def save_landmarks():
    data = request.get_json()
    if not data or "key" not in data:
        return "Missing key", 400

    record = {
        "key": data["key"],
        "landmarks": data.get("landmarks", []),
    }

    # Append to output file
    with open(OUTPUT_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return jsonify({"ok": True})


def _geom_to_geojson(geom):
    """Convert a Shapely geometry to a GeoJSON dict."""
    return shapely.geometry.mapping(geom)


@app.route("/api/nearby_landmarks/<int:index>")
def nearby_landmarks(index):
    if index < 0 or index >= len(PANORAMAS) or LANDMARK_METADATA is None:
        return jsonify({"type": "FeatureCollection", "features": []})

    lm_idxs = PANO_LANDMARK_IDXS[index]
    features = []
    for lm_idx in lm_idxs:
        row = LANDMARK_METADATA.iloc[lm_idx]
        geom = row["geometry"]
        props = {}
        for k, v in row["pruned_props"]:
            props[k] = v
        features.append({
            "type": "Feature",
            "geometry": _geom_to_geojson(geom),
            "properties": props,
        })
    return jsonify({"type": "FeatureCollection", "features": features})


@app.route("/api/gemini_predictions/<int:index>")
def gemini_predictions(index):
    if index < 0 or index >= len(PANORAMAS):
        return jsonify({"landmarks": []})
    pano_id = PANORAMAS[index]["id"]
    pred = GEMINI_PREDICTIONS.get(pano_id)
    if pred is None:
        return jsonify({"landmarks": [], "location_type": None})
    return jsonify(pred)


@app.route("/api/pano_label", methods=["POST"])
def set_pano_label():
    data = request.get_json()
    if not data or "pano_idx" not in data:
        return "Missing pano_idx", 400
    pano_idx = int(data["pano_idx"])
    label = data.get("label")  # "useful", "not_useful", or null to clear
    if label:
        PANO_LABELS[pano_idx] = label
    elif pano_idx in PANO_LABELS:
        del PANO_LABELS[pano_idx]
    _recompute_learned_categories()
    _retrain_dino_model()
    _save_labels()
    n_useful = sum(1 for v in PANO_LABELS.values() if v == "useful")
    n_not_useful = sum(1 for v in PANO_LABELS.values() if v == "not_useful")
    return jsonify({
        "ok": True,
        "total_labels": len(PANO_LABELS),
        "n_useful": n_useful,
        "n_not_useful": n_not_useful,
    })


@app.route("/api/pano_label/<int:index>")
def get_pano_label(index):
    label = PANO_LABELS.get(index)
    return jsonify({"label": label})


@app.route("/api/tag_override", methods=["POST"])
def set_tag_override():
    data = request.get_json()
    if not data or "key" not in data or "value" not in data:
        return "Missing key/value", 400
    key = data["key"]
    value = data["value"]
    override = data.get("override")  # "always_distinctive", "never_distinctive", or null
    tag = (key, value)
    if override:
        TAG_OVERRIDES[tag] = override
    elif tag in TAG_OVERRIDES:
        del TAG_OVERRIDES[tag]
    _recompute_learned_categories()
    _save_labels()
    score = TAG_SCORES.get(tag, 0.5)
    return jsonify({"ok": True, "score": score})


@app.route("/api/tag_overrides")
def get_tag_overrides():
    result = []
    for (key, value), override in TAG_OVERRIDES.items():
        score = TAG_SCORES.get((key, value), 0.5)
        result.append({
            "key": key, "value": value,
            "override": override, "score": score,
        })
    return jsonify({"overrides": result})


@app.route("/api/tag_scores/<int:index>")
def get_tag_scores(index):
    if index < 0 or index >= len(PANORAMAS):
        return jsonify({"tags": []})
    tags = _get_pano_tags(index)
    result = []
    for key, value in sorted(tags):
        score = TAG_SCORES.get((key, value), 0.5)
        override = TAG_OVERRIDES.get((key, value))
        result.append({
            "key": key, "value": value,
            "score": round(score, 3), "override": override,
        })
    result.sort(key=lambda t: -t["score"])
    return jsonify({"tags": result})


@app.route("/api/dino_prediction/<int:index>")
def dino_prediction(index):
    if index < 0 or index >= len(PANORAMAS) or not PANO_CATEGORIES_DINO:
        return jsonify({"category": "no_data", "confidence": None})
    return jsonify({
        "category": PANO_CATEGORIES_DINO[index],
        "confidence": DINO_CONFIDENCES[index] if DINO_CONFIDENCES else None,
    })


# ---------------------------------------------------------------------------
# Overview map page
# ---------------------------------------------------------------------------

MAP_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<title>Panorama Overview Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
#map { width: 100vw; height: 100vh; }
.legend {
    background: white; padding: 10px 14px; border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 13px; line-height: 22px;
}
.legend-dot {
    display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle; border: 1px solid #555;
}
.info-panel {
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: white; padding: 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); max-width: 400px; max-height: 90vh;
    overflow-y: auto; font-size: 13px; display: none;
}
.info-panel h3 { margin-bottom: 8px; }
.info-panel .close { float: right; cursor: pointer; font-size: 18px; color: #888; }
.info-panel .close:hover { color: #333; }
.info-panel table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.info-panel td { padding: 3px 6px; border-bottom: 1px solid #eee; vertical-align: top; }
.info-panel td:first-child { font-weight: bold; color: #555; white-space: nowrap; }
.info-panel .tag { display: inline-block; padding: 1px 6px; background: #e8f4fd; border-radius: 3px; margin: 1px; font-size: 11px; }
.info-panel .lm-item { padding: 6px 0; border-bottom: 1px solid #eee; }
.info-panel .lm-name { font-weight: bold; color: #333; }
.info-panel a { color: #2563eb; }
.info-panel .btn { display: inline-block; margin-top: 8px; padding: 6px 12px; background: #2563eb; color: white; border-radius: 4px; text-decoration: none; font-size: 12px; }
.info-panel .btn:hover { background: #1d4ed8; }
.counts { margin-top: 6px; padding: 4px 0; font-size: 12px; color: #666; }
</style>
</head>
<body>
<div id="map"></div>
<div class="info-panel" id="info-panel">
    <span class="close" onclick="document.getElementById('info-panel').style.display='none'">&times;</span>
    <div id="info-content"></div>
</div>

<script>
const COLORS_GEMINI = {distinctive: '#2d9e2d', generic: '#e6a817', no_predictions: '#d43d3d'};
const LABELS_GEMINI = {distinctive: 'Distinctive', generic: 'Generic only', no_predictions: 'No predictions'};

const COLORS_OSM = {distinctive: '#2d9e2d', generic: '#e6a817', no_landmarks: '#d43d3d'};
const LABELS_OSM = {distinctive: 'Distinctive', generic: 'Generic only', no_landmarks: 'No landmarks'};

const COLORS_LEARNED = {distinctive: '#2d9e2d', generic: '#e6a817', no_landmarks: '#d43d3d', no_data: '#999999'};
const LABELS_LEARNED = {distinctive: 'Distinctive', generic: 'Generic', no_landmarks: 'No landmarks', no_data: 'No data'};

const COLORS_DINO = {distinctive: '#2d9e2d', generic: '#e6a817', no_data: '#999999'};
const LABELS_DINO = {distinctive: 'Distinctive', generic: 'Generic', no_data: 'No data'};

let currentSource = 'gemini';
let labelStatsData = {total: 0, n_useful: 0, n_not_useful: 0};
function getColors() {
    if (currentSource === 'learned') return COLORS_LEARNED;
    if (currentSource === 'dino') return COLORS_DINO;
    return currentSource === 'gemini' ? COLORS_GEMINI : COLORS_OSM;
}
function getLabels() {
    if (currentSource === 'learned') return LABELS_LEARNED;
    if (currentSource === 'dino') return LABELS_DINO;
    return currentSource === 'gemini' ? LABELS_GEMINI : LABELS_OSM;
}
function getCat(p) {
    if (currentSource === 'learned') return p.category_learned;
    if (currentSource === 'dino') return p.category_dino;
    return currentSource === 'gemini' ? p.category_gemini : p.category_osm;
}

const map = L.map('map').setView([41.88, -87.63], 13);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OSM', maxZoom: 20,
}).addTo(map);

// Legend
const legend = L.control({position: 'bottomleft'});
legend.onAdd = function() {
    const div = L.DomUtil.create('div', 'legend');
    div.innerHTML = `
        <div style="margin-bottom:8px;">
            <b>Category source:</b><br>
            <label style="cursor:pointer;margin-right:10px;">
                <input type="radio" name="cat-source" value="gemini" checked onchange="switchSource('gemini')"> Gemini
            </label>
            <label style="cursor:pointer;margin-right:10px;">
                <input type="radio" name="cat-source" value="osm" onchange="switchSource('osm')"> OSM
            </label>
            <label style="cursor:pointer;margin-right:10px;">
                <input type="radio" name="cat-source" value="learned" onchange="switchSource('learned')"> Learned
            </label>
            <label style="cursor:pointer;">
                <input type="radio" name="cat-source" value="dino" onchange="switchSource('dino')"> DINO
            </label>
        </div>
        <div id="legend-items"></div>
        <span id="legend-counts"></span>
        <div id="label-stats-legend" style="margin-top:6px;font-size:12px;color:#666;"></div>
        <div style="margin-top:6px;">
            <button onclick="refreshData()" style="padding:4px 10px;font-size:12px;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5;">Refresh scores</button>
        </div>`;
    return div;
};
legend.addTo(map);

let allPanoramas = [];
let allMarkers = [];
const markersGroup = L.layerGroup().addTo(map);

function updateLegendItems() {
    const colors = getColors();
    const labels = getLabels();
    document.getElementById('legend-items').innerHTML =
        Object.entries(colors).map(([k, c]) =>
            `<span class="legend-dot" style="background:${c}"></span>${labels[k]}<br>`
        ).join('');
}

function updateCounts() {
    const labels = getLabels();
    const counts = {};
    allPanoramas.forEach(p => {
        const cat = getCat(p);
        counts[cat] = (counts[cat] || 0) + 1;
    });
    const total = allPanoramas.length;
    document.getElementById('legend-counts').innerHTML =
        Object.entries(labels).map(([k, label]) => {
            const c = counts[k] || 0;
            return `${label}: ${c} (${(c/total*100).toFixed(1)}%)`;
        }).join('<br>');
}

function recolorMarkers() {
    const colors = getColors();
    allMarkers.forEach((marker, i) => {
        const p = allPanoramas[i];
        const cat = getCat(p);
        let borderColor = '#fff';
        let borderWeight = 1;
        if (p.label === 'useful') { borderColor = '#2d9e2d'; borderWeight = 3; }
        else if (p.label === 'not_useful') { borderColor = '#d43d3d'; borderWeight = 3; }
        marker.setStyle({ fillColor: colors[cat], color: borderColor, weight: borderWeight });
    });
    updateLegendItems();
    updateCounts();
    updateLabelStats();
}

function switchSource(source) {
    currentSource = source;
    recolorMarkers();
}

function updateLabelStats() {
    const el = document.getElementById('label-stats-legend');
    if (labelStatsData.total > 0) {
        el.innerHTML = `Labels: ${labelStatsData.total} (${labelStatsData.n_useful} useful, ${labelStatsData.n_not_useful} not useful)`;
    } else {
        el.innerHTML = '';
    }
}

async function refreshData() {
    const resp = await fetch('/api/panorama_map_data');
    const data = await resp.json();
    allPanoramas = data.panoramas;
    if (data.label_stats) labelStatsData = data.label_stats;
    recolorMarkers();
}

// Load panoramas
fetch('/api/panorama_map_data').then(r => r.json()).then(data => {
    allPanoramas = data.panoramas;

    if (data.label_stats) labelStatsData = data.label_stats;

    data.panoramas.forEach((p, i) => {
        const cat = getCat(p);
        let borderColor = '#fff';
        let borderWeight = 1;
        if (p.label === 'useful') { borderColor = '#2d9e2d'; borderWeight = 3; }
        else if (p.label === 'not_useful') { borderColor = '#d43d3d'; borderWeight = 3; }
        const marker = L.circleMarker([p.lat, p.lon], {
            radius: 5,
            fillColor: getColors()[cat],
            color: borderColor,
            weight: borderWeight,
            fillOpacity: 0.75,
        });
        marker.on('click', () => showPanoInfo(i, p));
        markersGroup.addLayer(marker);
        allMarkers.push(marker);
    });

    // Fit bounds
    const lats = data.panoramas.map(p => p.lat);
    const lons = data.panoramas.map(p => p.lon);
    map.fitBounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]], {padding: [30, 30]});

    updateLegendItems();
    updateCounts();
    updateLabelStats();
});

async function showPanoInfo(idx, pano) {
    const panel = document.getElementById('info-panel');
    const content = document.getElementById('info-content');
    panel.style.display = 'block';

    const catLabel = getLabels()[getCat(pano)];
    const catColor = getColors()[getCat(pano)];
    const mapsUrl = `https://www.google.com/maps/place/${pano.lat},${pano.lon}/@${pano.lat},${pano.lon},18z`;

    const geminiLabel = LABELS_GEMINI[pano.category_gemini];
    const geminiColor = COLORS_GEMINI[pano.category_gemini];
    const osmLabel = LABELS_OSM[pano.category_osm];
    const osmColor = COLORS_OSM[pano.category_osm];
    const learnedLabel = LABELS_LEARNED[pano.category_learned] || 'No data';
    const learnedColor = COLORS_LEARNED[pano.category_learned] || '#999';
    const dinoLabel = LABELS_DINO[pano.category_dino] || 'No data';
    const dinoColor = COLORS_DINO[pano.category_dino] || '#999';
    const userLabel = pano.label ? (pano.label === 'useful' ? '<span style="color:#2d9e2d;font-weight:bold;">Useful</span>' : '<span style="color:#d43d3d;font-weight:bold;">Not useful</span>') : '<span style="color:#999;">Unlabeled</span>';

    content.innerHTML = `
        <h3>${pano.id}</h3>
        <table>
            <tr><td>Index</td><td>${idx}</td></tr>
            <tr><td>Location</td><td><a href="${mapsUrl}" target="_blank">${pano.lat.toFixed(6)}, ${pano.lon.toFixed(6)}</a></td></tr>
            <tr><td>Gemini</td><td><span style="color:${geminiColor};font-weight:bold;">${geminiLabel}</span></td></tr>
            <tr><td>OSM</td><td><span style="color:${osmColor};font-weight:bold;">${osmLabel}</span></td></tr>
            <tr><td>Learned</td><td><span style="color:${learnedColor};font-weight:bold;">${learnedLabel}</span></td></tr>
            <tr><td>DINO</td><td><span style="color:${dinoColor};font-weight:bold;">${dinoLabel}</span></td></tr>
            <tr><td>User label</td><td>${userLabel}</td></tr>
            <tr><td>OSM landmarks</td><td>${pano.n_landmarks}</td></tr>
        </table>
        <a class="btn" href="/?goto=${idx}" target="_blank">Open in tagger</a>
        <div id="gemini-section" style="margin-top:12px;"><em>Loading...</em></div>
        <div id="osm-section" style="margin-top:12px;"></div>`;

    // Fetch both in parallel
    const [osmResp, geminiResp] = await Promise.all([
        fetch(`/api/nearby_landmarks/${idx}`),
        fetch(`/api/gemini_predictions/${idx}`),
    ]);
    const geojson = await osmResp.json();
    const gemini = await geminiResp.json();

    // Gemini predictions
    const geminiSection = document.getElementById('gemini-section');
    if (gemini.landmarks && gemini.landmarks.length > 0) {
        const locType = gemini.location_type ? `<div class="counts">Location type: <b>${gemini.location_type}</b></div>` : '';
        const geminiHtml = gemini.landmarks.map(lm => {
            const pt = lm.primary_tag || {};
            const primaryStr = pt.key ? `<span class="tag" style="background:#d4edda;">${pt.key}=${pt.value}</span>` : '';
            const addlStr = (lm.additional_tags || [])
                .map(t => `<span class="tag">${t.key}=${t.value}</span>`).join(' ');
            const conf = lm.confidence || '';
            const desc = lm.description || '';
            return `<div class="lm-item">
                <div>${primaryStr} ${addlStr} <span style="color:#888;font-size:11px;">[${conf}]</span></div>
                ${desc ? `<div style="color:#666;font-size:11px;margin-top:2px;">${desc}</div>` : ''}
            </div>`;
        }).join('');
        geminiSection.innerHTML = `<h4 style="margin-bottom:6px;">Gemini predictions (${gemini.landmarks.length})</h4>${locType}${geminiHtml}`;
    } else {
        geminiSection.innerHTML = '<h4 style="margin-bottom:6px;">Gemini predictions</h4><em>None</em>';
    }

    // OSM landmarks
    const osmSection = document.getElementById('osm-section');
    let lmHtml = '';
    if (geojson.features.length === 0) {
        lmHtml = '<em>No landmarks</em>';
    } else {
        lmHtml = geojson.features.map(f => {
            const p = f.properties;
            const name = p.name || '';
            const tags = Object.entries(p)
                .filter(([k]) => k !== 'name' && !k.startsWith('addr:') && !k.startsWith('tiger:') && !k.startsWith('gnis:'))
                .slice(0, 8)
                .map(([k,v]) => `<span class="tag">${k}=${v}</span>`)
                .join(' ');
            return `<div class="lm-item">
                ${name ? `<div class="lm-name">${name}</div>` : ''}
                <div>${tags}</div>
            </div>`;
        }).join('');
    }
    osmSection.innerHTML = `<h4 style="margin-bottom:6px;">Nearby OSM landmarks (${geojson.features.length})</h4>` + lmHtml;
}
</script>
</body>
</html>
"""


@app.route("/map")
def overview_map():
    return Response(MAP_PAGE, content_type="text/html")


@app.route("/api/panorama_map_data")
def panorama_map_data():
    n_useful = sum(1 for v in PANO_LABELS.values() if v == "useful")
    n_not_useful = sum(1 for v in PANO_LABELS.values() if v == "not_useful")
    return jsonify({
        "panoramas": [
            {
                "id": p["id"],
                "lat": p["lat"],
                "lon": p["lon"],
                "category_gemini": PANO_CATEGORIES_GEMINI[i],
                "category_osm": PANO_CATEGORIES_OSM[i],
                "category_learned": PANO_CATEGORIES_LEARNED[i] if PANO_CATEGORIES_LEARNED else "no_data",
                "category_dino": PANO_CATEGORIES_DINO[i] if PANO_CATEGORIES_DINO else "no_data",
                "label": PANO_LABELS.get(i),
                "n_landmarks": len(PANO_LANDMARK_IDXS[i]),
            }
            for i, p in enumerate(PANORAMAS)
        ],
        "label_stats": {
            "total": len(PANO_LABELS),
            "n_useful": n_useful,
            "n_not_useful": n_not_useful,
        },
    })


def _parse_panorama_filename(filename):
    """Parse panorama filename like '{pano_id},{lat},{lon},.jpg'."""
    stem = Path(filename).stem
    parts = stem.split(",")
    if len(parts) < 3:
        return None
    try:
        return {
            "id": parts[0],
            "lat": float(parts[1]),
            "lon": float(parts[2]),
        }
    except ValueError:
        return None


def main():
    global PANORAMAS, TAG_INDEX, OUTPUT_PATH, LANDMARK_METADATA, PANO_LANDMARK_IDXS
    global PANO_CATEGORIES_GEMINI, PANO_CATEGORIES_OSM, LABELS_PATH
    global DINO_FEATURES, DINO_PANO_MAP, PANO_CATEGORIES_DINO, DINO_CONFIDENCES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panorama_dir",
        type=Path,
        required=True,
        help="Directory containing panorama images",
    )
    parser.add_argument(
        "--feather",
        type=Path,
        required=True,
        help="Feather file with OSM landmark data for tag search",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/tagged_landmarks.jsonl"),
        help="Output JSONL file for saved landmarks",
    )
    parser.add_argument(
        "--landmark_radius_px",
        type=float,
        default=640,
        help="Radius in web mercator pixels for panorama-landmark association",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=None,
        help="Directory with Gemini predictions (contains sentences/panorama_request_*/*/predictions.jsonl)",
    )
    parser.add_argument(
        "--dino_cache",
        type=Path,
        default=None,
        help="Path to .npz file with DINO CLS tokens (from extract_dino_cls_tokens.py)",
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    OUTPUT_PATH = args.output

    # Load panorama list
    print(f"Scanning panoramas from {args.panorama_dir}...")
    pano_dir = args.panorama_dir
    image_files = sorted(
        f
        for f in pano_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for f in image_files:
        parsed = _parse_panorama_filename(f.name)
        if parsed:
            parsed["path"] = f
            PANORAMAS.append(parsed)
    print(f"Found {len(PANORAMAS)} panoramas")

    # Build tag search index
    print(f"Loading OSM data from {args.feather}...")
    df = pd.read_feather(args.feather)
    print(f"Building tag search index from {len(df)} landmarks...")
    TAG_INDEX = TagSearchIndex(df)
    print(f"Index ready: {len(TAG_INDEX._values)} unique tag values")

    # Load landmark geometries and compute panorama associations
    print("Loading landmark geometries...")
    LANDMARK_METADATA = load_landmark_geojson(args.feather, zoom_level=20)
    print(f"Loaded {len(LANDMARK_METADATA)} landmarks with geometries")

    print("Computing panorama-landmark associations...")
    pano_metadata = load_panorama_metadata(args.panorama_dir, zoom_level=20)
    correspondences = compute_panorama_from_landmarks(
        pano_metadata, LANDMARK_METADATA, args.landmark_radius_px,
    )
    PANO_LANDMARK_IDXS = correspondences.landmark_idxs_from_pano_idx
    total_assoc = sum(len(idxs) for idxs in PANO_LANDMARK_IDXS)
    print(f"Computed {total_assoc} panorama-landmark associations")

    print("Precomputing per-panorama tags...")
    _precompute_pano_tags()

    # Load Gemini predictions if provided
    if args.predictions_dir:
        print(f"Loading Gemini predictions from {args.predictions_dir}...")
        GEMINI_PREDICTIONS.update(_load_gemini_predictions(args.predictions_dir))
        pano_ids_with_preds = sum(
            1 for p in PANORAMAS if p["id"] in GEMINI_PREDICTIONS
        )
        print(f"Loaded predictions for {len(GEMINI_PREDICTIONS)} panoramas "
              f"({pano_ids_with_preds} matching loaded panoramas)")

    # Categorize panoramas
    from collections import Counter

    print("Categorizing panoramas (OSM)...")
    PANO_CATEGORIES_OSM = [
        _categorize_panorama_osm(i) for i in range(len(PANORAMAS))
    ]
    osm_counts = Counter(PANO_CATEGORIES_OSM)
    for cat in ["distinctive", "generic", "no_landmarks"]:
        c = osm_counts.get(cat, 0)
        print(f"  {cat}: {c} ({c / len(PANORAMAS) * 100:.1f}%)")

    print("Categorizing panoramas (Gemini)...")
    PANO_CATEGORIES_GEMINI = [
        _categorize_panorama_gemini(p["id"], GEMINI_PREDICTIONS)
        for p in PANORAMAS
    ]
    gemini_counts = Counter(PANO_CATEGORIES_GEMINI)
    for cat in ["distinctive", "generic", "no_predictions"]:
        c = gemini_counts.get(cat, 0)
        print(f"  {cat}: {c} ({c / len(PANORAMAS) * 100:.1f}%)")

    # Load any existing landmarks from output file
    if args.output.exists():
        print(f"Loading existing landmarks from {args.output}...")
        count = 0
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pano_id = record["key"].split(",")[0]
                    SAVED_LANDMARKS[pano_id] = record.get("landmarks", [])
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Loaded landmarks for {count} panoramas")

    # Load panorama labels and tag overrides
    LABELS_PATH = args.output.parent / (args.output.stem + ".labels.json")
    print(f"Loading labels from {LABELS_PATH}...")
    _load_labels()
    print(f"Loaded {len(PANO_LABELS)} panorama labels, {len(TAG_OVERRIDES)} tag overrides")

    # Compute learned categories
    print("Computing learned categories...")
    _recompute_learned_categories()
    if PANO_LABELS:
        learned_counts = Counter(PANO_CATEGORIES_LEARNED)
        for cat in ["distinctive", "generic", "no_landmarks", "no_data"]:
            c = learned_counts.get(cat, 0)
            print(f"  {cat}: {c} ({c / len(PANORAMAS) * 100:.1f}%)")
    else:
        print("  No labels yet, all panoramas set to no_data")

    # Load DINO features if provided
    if args.dino_cache and args.dino_cache.exists():
        print(f"Loading DINO features from {args.dino_cache}...")
        data = np.load(args.dino_cache, allow_pickle=True)
        DINO_FEATURES = data["features"]
        dino_pano_ids = data["pano_ids"]
        DINO_PANO_MAP = {str(pid): i for i, pid in enumerate(dino_pano_ids)}
        n_matched = sum(1 for p in PANORAMAS if p["id"] in DINO_PANO_MAP)
        print(f"Loaded {len(dino_pano_ids)} DINO features ({DINO_FEATURES.shape[1]}-dim), "
              f"{n_matched} matching loaded panoramas")
        PANO_CATEGORIES_DINO = ["no_data"] * len(PANORAMAS)
        DINO_CONFIDENCES = [None] * len(PANORAMAS)
        if PANO_LABELS:
            print("Training initial DINO model from existing labels...")
            _retrain_dino_model()
    elif args.dino_cache:
        print(f"Warning: DINO cache not found at {args.dino_cache}")

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Output: {args.output}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
