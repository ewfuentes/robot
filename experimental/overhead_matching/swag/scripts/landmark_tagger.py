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
PANO_LABELS = {}  # pano_id -> "useful" | "not_useful"
TAG_OVERRIDES = {}  # (key, val) -> "always_distinctive" | "never_distinctive"
TAG_SCORES = {}  # (key, val) -> P(useful|tag)
PANO_CATEGORIES_LEARNED = []  # parallel to PANORAMAS
PANO_TAGS = []  # list of set[(key, val)], parallel to PANORAMAS — precomputed at startup
LABELS_PATH = None  # Path to labels JSON file
PANO_ID_TO_IDX = {}  # pano_id -> pano_idx, built once at startup
PANO_PX = []  # list of shapely.Point in pixel coords, parallel to PANORAMAS

BAD_GPS_PANOS = set()  # set of pano_id strings flagged as bad GPS

# Landmark annotation: which OSM landmarks are the "distinctive thing" for each panorama
RELEVANT_LANDMARKS = {}  # pano_id -> list of osm_id strings ([] = reviewed, no relevant)
OSM_ID_TO_LM_IDX = {}  # osm_id string -> index in LANDMARK_METADATA

# DINO-based visual distinctiveness prediction
DINO_FEATURES = None     # np.ndarray (N, 3072) or None
DINO_PANO_MAP = {}       # pano_id -> row index in DINO_FEATURES
DINO_MODEL = None        # sklearn LogisticRegression or None
PANO_CATEGORIES_DINO = []  # parallel to PANORAMAS: "distinctive"/"generic"/"no_data"
DINO_CONFIDENCES = []    # parallel to PANORAMAS: float confidence or None

# Active learning queue for prioritizing which panoramas to label next
ACTIVE_LEARNING_QUEUE = []    # [(score, pano_idx), ...] sorted descending by score
SPATIAL_K = 5
AL_W_UNCERTAINTY = 0.5
AL_W_DISAGREEMENT = 0.25
AL_W_ISOLATION = 0.25

# Landmark relevance model (tag-based logistic regression)
LANDMARK_RELEVANCE_MODEL = None   # sklearn LogisticRegression or None
LANDMARK_RELEVANCE_VOCAB = {}     # feature_name -> column index
LANDMARK_RELEVANCE_DIRTY = False  # set True on annotation change, cleared after refit

# Gemini evaluation: correspondences between Gemini-extracted and OSM landmarks
GEMINI_EVAL = {}           # pano_id -> {"reviewed": bool, "matches": {...}}
GEMINI_EVAL_QUEUE = []     # list of pano_idx for panoramas with Gemini preds + "useful" label

# Auto-labeling generic Gemini landmarks via sklearn classifier
GENERIC_CLF = None         # trained RandomForestClassifier (or None if not enough data)
GENERIC_VEC = None         # fitted DictVectorizer

# Tag featurization constants for landmark relevance model
# Primary category tags: full key=value features
PRIMARY_TAG_KEYS = {
    "amenity", "shop", "tourism", "leisure", "historic", "natural",
    "man_made", "building", "highway", "office", "healthcare", "craft",
    "sport", "railway", "power", "landuse", "emergency", "public_transport",
    "barrier", "waterway", "place", "aeroway", "military", "club",
    "advertising", "geological",
}
# Tags where presence matters but value is free-text
PRESENCE_ONLY_KEYS = {
    "name", "brand", "cuisine", "religion", "denomination",
    "height", "building:levels", "colour", "building:colour",
    "architect", "material", "building:material", "roof:material",
    "description", "inscription", "note",
    "addr:housenumber", "addr:street",
    "phone", "operator", "ref",
}
# Keys to skip entirely
SKIP_TAG_KEYS = {
    "source", "source:name", "source_ref", "created_by",
    "image", "facebook", "contact:website", "contact:facebook",
    "contact:twitter", "contact:email", "contact:fax", "contact:phone",
    "brand:website", "brand:wikidata", "brand:wikipedia",
    "wikidata", "wikipedia", "artist:wikidata",
    "import_uuid",
}

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


def _extract_landmark_features(glm):
    """Extract feature dict from a single Gemini landmark for the generic classifier.

    Returns a dict suitable for sklearn DictVectorizer (mix of one-hot string
    features and numeric features).
    """
    import re

    pt = glm.get("primary_tag") or {}
    key = pt.get("key", "")
    value = pt.get("value", "")
    addl = glm.get("additional_tags") or []
    desc = glm.get("description", "") or ""
    confidence = glm.get("confidence", "medium")
    bboxes = glm.get("bounding_boxes") or []

    has_name = any(t.get("key") == "name" for t in addl)
    # Check for proper nouns in description: words starting with uppercase
    # that aren't at the start of a sentence
    words = desc.split()
    has_proper_nouns = any(
        w[0].isupper() and not re.match(r"^[A-Z]\.", w)
        for w in words[1:]
        if w and w[0].isalpha()
    ) if len(words) > 1 else False

    feat = {
        f"key={key}": 1,
        f"value={value}": 1,
        f"conf={confidence}": 1,
        "has_name": int(has_name),
        "has_proper_nouns": int(has_proper_nouns),
        "n_additional_tags": len(addl),
        "desc_length": len(desc),
        "n_bboxes": len(bboxes),
    }
    return feat


def _train_generic_classifier():
    """Train a RandomForest to predict generic landmarks from reviewed GEMINI_EVAL data.

    Sets globals GENERIC_CLF and GENERIC_VEC. No-ops if insufficient labeled data.
    """
    global GENERIC_CLF, GENERIC_VEC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction import DictVectorizer

    features = []
    labels = []

    for pano_id, ev in GEMINI_EVAL.items():
        if not ev.get("reviewed"):
            continue
        pred = GEMINI_PREDICTIONS.get(pano_id, {})
        landmarks = pred.get("landmarks", [])
        for gi_str, match in ev.get("matches", {}).items():
            status = match.get("status", "unreviewed")
            if status == "unreviewed":
                continue
            gi = int(gi_str)
            if gi >= len(landmarks):
                continue
            feat = _extract_landmark_features(landmarks[gi])
            labels.append(1 if status == "generic" else 0)
            features.append(feat)

    n_generic = sum(labels)
    n_total = len(labels)

    if n_total < 20 or n_generic < 5:
        GENERIC_CLF = None
        GENERIC_VEC = None
        print(f"Generic classifier: insufficient data ({n_total} total, {n_generic} generic) — skipping")
        return

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(features)
    y = np.array(labels)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    # Simple training accuracy for logging
    train_acc = clf.score(X, y)
    GENERIC_CLF = clf
    GENERIC_VEC = vec
    print(f"Generic classifier: trained on {n_total} landmarks "
          f"({n_generic} generic), accuracy {train_acc:.2f}")


def _load_labels():
    """Load panorama labels and tag overrides from the labels JSON file.

    Returns True if migration from old (pano_idx) format was performed.
    """
    global PANO_LABELS, TAG_OVERRIDES, RELEVANT_LANDMARKS, BAD_GPS_PANOS, GEMINI_EVAL
    if LABELS_PATH is None or not LABELS_PATH.exists():
        return False
    with open(LABELS_PATH) as f:
        data = json.load(f)

    raw_labels = data.get("panorama_labels", {})
    migrated = False

    # Detect old format: keys are numeric strings like "123"
    is_old_format = raw_labels and all(k.isdigit() for k in raw_labels)
    if is_old_format and PANORAMAS:
        print("Migrating labels from pano_idx to pano_id format...")
        import shutil
        backup_path = Path(str(LABELS_PATH) + ".bak")
        shutil.copy2(LABELS_PATH, backup_path)
        print(f"  Backed up old labels to {backup_path}")
        PANO_LABELS = {}
        for k, v in raw_labels.items():
            idx = int(k)
            if 0 <= idx < len(PANORAMAS):
                PANO_LABELS[PANORAMAS[idx]["id"]] = v
            else:
                print(f"  Warning: skipping out-of-range pano_idx {idx}")
        print(f"  Migrated {len(PANO_LABELS)} labels")
        migrated = True
    else:
        PANO_LABELS = dict(raw_labels)

    TAG_OVERRIDES = {}
    for tag_str, override in data.get("tag_overrides", {}).items():
        if "=" in tag_str:
            key, value = tag_str.split("=", 1)
            TAG_OVERRIDES[(key, value)] = override

    RELEVANT_LANDMARKS = data.get("relevant_landmarks", {})
    BAD_GPS_PANOS = set(data.get("bad_gps", []))
    GEMINI_EVAL = data.get("gemini_eval", {})
    return migrated


def _save_labels():
    """Save panorama labels, tag overrides, and relevant landmarks to JSON."""
    if LABELS_PATH is None:
        return
    data = {
        "panorama_labels": dict(PANO_LABELS),
        "tag_overrides": {
            f"{k}={v}": override for (k, v), override in TAG_OVERRIDES.items()
        },
        "relevant_landmarks": RELEVANT_LANDMARKS,
        "bad_gps": sorted(BAD_GPS_PANOS),
        "gemini_eval": GEMINI_EVAL,
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


LANDMARK_RADIUS_PX = 640  # search radius in pixel coords (matches compute_panorama_from_landmarks)


def _featurize_landmark(row):
    """Extract sparse binary feature set from a landmark row's pruned_props.

    Returns a set of feature name strings using hybrid featurization:
      - Primary category tags (amenity, shop, etc.): "key=value"
      - Presence-only tags (name, brand, etc.): "has:key"
      - Metadata tags (source, wikidata, tiger:*, etc.): skipped
      - Other tags: "key=value" (default)
    """
    features = set()
    for key, value in row["pruned_props"]:
        if key in SKIP_TAG_KEYS or key.startswith("tiger:") or key.startswith("gnis:"):
            continue
        if key in PRIMARY_TAG_KEYS:
            features.add(f"{key}={value}")
        elif key in PRESENCE_ONLY_KEYS:
            features.add(f"has:{key}")
        else:
            features.add(f"{key}={value}")
    return features


def _landmark_distance_px(pano_idx, lm_idx):
    """Compute pixel distance from panorama to nearest point of a landmark geometry."""
    pano_pt = PANO_PX[pano_idx]
    lm_geom = LANDMARK_METADATA.iloc[lm_idx]["geometry_px"]
    return pano_pt.distance(lm_geom)


def _retrain_landmark_relevance_model():
    """Retrain logistic regression on OSM tag features + distance from landmark annotations."""
    global LANDMARK_RELEVANCE_MODEL, LANDMARK_RELEVANCE_VOCAB, LANDMARK_RELEVANCE_DIRTY
    LANDMARK_RELEVANCE_DIRTY = False

    if not RELEVANT_LANDMARKS:
        LANDMARK_RELEVANCE_MODEL = None
        LANDMARK_RELEVANCE_VOCAB = {}
        return

    # Collect training examples: for each annotated panorama, each nearby landmark
    feature_sets = []
    distances = []
    labels = []
    for pano_id, osm_ids in RELEVANT_LANDMARKS.items():
        pano_idx = PANO_ID_TO_IDX.get(pano_id)
        if pano_idx is None:
            continue
        relevant_set = set(osm_ids)
        for lm_idx in PANO_LANDMARK_IDXS[pano_idx]:
            row = LANDMARK_METADATA.iloc[lm_idx]
            osm_id = str(row["id"])
            feats = _featurize_landmark(row)
            if not feats:
                continue
            feature_sets.append(feats)
            distances.append(_landmark_distance_px(pano_idx, lm_idx))
            labels.append(1 if osm_id in relevant_set else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos < 10 or n_neg < 10:
        LANDMARK_RELEVANCE_MODEL = None
        LANDMARK_RELEVANCE_VOCAB = {}
        return

    # Build vocabulary from all observed features
    all_features = set()
    for fs in feature_sets:
        all_features.update(fs)
    vocab = {feat: i for i, feat in enumerate(sorted(all_features))}
    LANDMARK_RELEVANCE_VOCAB = vocab

    # Build feature matrix: sparse tag columns + 2 dense distance columns
    from scipy.sparse import hstack, lil_matrix
    from sklearn.linear_model import LogisticRegression

    n = len(labels)
    n_tag_feats = len(vocab)
    X_tags = lil_matrix((n, n_tag_feats), dtype=np.float32)
    for i, fs in enumerate(feature_sets):
        for feat in fs:
            if feat in vocab:
                X_tags[i, vocab[feat]] = 1.0

    # Distance features: normalized distance and inverse distance
    dist_arr = np.array(distances, dtype=np.float32)
    norm_dist = dist_arr / LANDMARK_RADIUS_PX  # 0 to ~1
    inv_dist = 1.0 / (1.0 + dist_arr)  # 1 at distance 0, decays toward 0
    X_dense = np.column_stack([norm_dist, inv_dist])

    X = hstack([X_tags.tocsr(), X_dense]).tocsr()
    y = np.array(labels)

    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    LANDMARK_RELEVANCE_MODEL = model
    print(f"Landmark relevance model retrained ({n_pos}+/{n_neg}-): "
          f"{n_tag_feats} tag features + 2 distance features")


def _auto_match_gemini_osm(pano_idx):
    """Auto-match Gemini landmarks to OSM landmarks for a panorama.

    Uses tag key/value matching and name similarity. Greedy 1:1 assignment
    with minimum threshold. Returns dict of gemini_idx -> {osm_ids, status}.
    """
    import difflib

    pano_id = PANORAMAS[pano_idx]["id"]
    pred = GEMINI_PREDICTIONS.get(pano_id)
    if not pred or not pred.get("landmarks"):
        return {}

    gemini_landmarks = pred["landmarks"]
    lm_idxs = PANO_LANDMARK_IDXS[pano_idx]

    # Build OSM landmark info list
    osm_info = []
    for lm_idx in lm_idxs:
        row = LANDMARK_METADATA.iloc[lm_idx]
        osm_id = str(row["id"]) if "id" in row.index else ""
        props = dict(row["pruned_props"])
        name = props.get("name", "")
        # Find primary tag key
        primary_key = None
        primary_value = None
        for k in PRIMARY_TAG_KEYS:
            if k in props:
                primary_key = k
                primary_value = props[k]
                break
        osm_info.append({
            "osm_id": osm_id,
            "props": props,
            "name": name,
            "primary_key": primary_key,
            "primary_value": primary_value,
        })

    # Score all pairs
    scores = []
    for gi, glm in enumerate(gemini_landmarks):
        pt = glm.get("primary_tag") or {}
        g_key = pt.get("key", "")
        g_value = pt.get("value", "")
        g_addl = {(t.get("key", ""), t.get("value", "")) for t in (glm.get("additional_tags") or [])}
        g_name = ""
        for t in (glm.get("additional_tags") or []):
            if t.get("key") == "name":
                g_name = t.get("value", "")
                break
        if not g_name:
            g_name = glm.get("description", "")

        for oi, osm in enumerate(osm_info):
            score = 0.0
            # Primary tag key match
            if g_key and osm["primary_key"] and g_key == osm["primary_key"]:
                score += 2.0
                # Primary tag value match
                if g_value and osm["primary_value"]:
                    if g_value.lower() == osm["primary_value"].lower():
                        score += 3.0
                    elif difflib.SequenceMatcher(None, g_value.lower(), osm["primary_value"].lower()).ratio() > 0.6:
                        score += 1.5

            # Name match
            if g_name and osm["name"]:
                ratio = difflib.SequenceMatcher(None, g_name.lower(), osm["name"].lower()).ratio()
                score += ratio * 5.0

            # Additional tag overlap
            for ak, av in g_addl:
                if ak in osm["props"] and osm["props"][ak].lower() == av.lower():
                    score += 1.0

            if score > 0:
                scores.append((score, gi, oi))

    # Greedy 1:1 assignment
    scores.sort(reverse=True)
    used_gemini = set()
    used_osm = set()
    matches = {}
    min_threshold = 2.0

    for score, gi, oi in scores:
        if gi in used_gemini or oi in used_osm:
            continue
        if score < min_threshold:
            break
        used_gemini.add(gi)
        used_osm.add(oi)
        matches[str(gi)] = {"osm_ids": [osm_info[oi]["osm_id"]], "status": "matched"}

    # Unmatched Gemini landmarks: auto-label obvious generics, rest unreviewed
    for gi in range(len(gemini_landmarks)):
        if str(gi) not in matches:
            status = "unreviewed"
            if GENERIC_CLF is not None:
                feat = _extract_landmark_features(gemini_landmarks[gi])
                X = GENERIC_VEC.transform([feat])
                proba = GENERIC_CLF.predict_proba(X)[0]
                generic_idx = list(GENERIC_CLF.classes_).index(1)
                if proba[generic_idx] > 0.8:
                    status = "generic"
            matches[str(gi)] = {"osm_ids": [], "status": status}

    return matches


def _build_gemini_eval_queue():
    """Build queue of panoramas eligible for Gemini evaluation.

    Includes panoramas with both Gemini predictions AND 'useful' label.
    """
    global GEMINI_EVAL_QUEUE
    GEMINI_EVAL_QUEUE = []
    for i, p in enumerate(PANORAMAS):
        pano_id = p["id"]
        if PANO_LABELS.get(pano_id) != "useful":
            continue
        pred = GEMINI_PREDICTIONS.get(pano_id)
        if not pred or not pred.get("landmarks"):
            continue
        GEMINI_EVAL_QUEUE.append(i)


def _compute_gemini_eval_stats():
    """Compute aggregate statistics from GEMINI_EVAL across reviewed panoramas."""
    reviewed_panos = []
    for pano_id, ev in GEMINI_EVAL.items():
        if ev.get("reviewed"):
            reviewed_panos.append((pano_id, ev))

    total_in_queue = len(GEMINI_EVAL_QUEUE)
    n_reviewed = len(reviewed_panos)

    n_matched = 0
    n_hallucination = 0
    n_novel = 0
    n_misidentified = 0
    n_generic = 0
    n_out_of_range = 0
    n_unreviewed = 0
    conf_bins = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}  # [matched, total]

    for pano_id, ev in reviewed_panos:
        pred = GEMINI_PREDICTIONS.get(pano_id, {})
        landmarks = pred.get("landmarks", [])
        for gi_str, match in ev.get("matches", {}).items():
            status = match.get("status", "unreviewed")
            gi = int(gi_str)
            conf = landmarks[gi].get("confidence", "medium") if gi < len(landmarks) else "medium"
            if status == "matched":
                n_matched += 1
                if conf in conf_bins:
                    conf_bins[conf][0] += 1
                    conf_bins[conf][1] += 1
            elif status == "hallucination":
                n_hallucination += 1
                if conf in conf_bins:
                    conf_bins[conf][1] += 1
            elif status == "novel":
                n_novel += 1
                if conf in conf_bins:
                    conf_bins[conf][0] += 1
                    conf_bins[conf][1] += 1
            elif status == "misidentified":
                n_misidentified += 1
                if conf in conf_bins:
                    conf_bins[conf][1] += 1
            elif status == "generic":
                n_generic += 1
            elif status == "out_of_range":
                n_out_of_range += 1
                if conf in conf_bins:
                    conf_bins[conf][0] += 1
                    conf_bins[conf][1] += 1
            else:
                n_unreviewed += 1

    total_landmarks = n_matched + n_hallucination + n_novel + n_misidentified + n_generic + n_out_of_range + n_unreviewed

    # Relevant recall: of user-marked relevant OSM landmarks in evaluated panos,
    # fraction with a Gemini match
    relevant_total = 0
    relevant_matched = 0
    matched_osm_ids = set()
    for pano_id, ev in reviewed_panos:
        for match in ev.get("matches", {}).values():
            if match.get("status") in ("matched", "misidentified"):
                for oid in match.get("osm_ids", []):
                    matched_osm_ids.add((pano_id, oid))

    for pano_id, ev in reviewed_panos:
        for osm_id in RELEVANT_LANDMARKS.get(pano_id, []):
            relevant_total += 1
            if (pano_id, osm_id) in matched_osm_ids:
                relevant_matched += 1

    # Confidence calibration
    calibration = {}
    for level, (m, t) in conf_bins.items():
        calibration[level] = {"matched": m, "total": t, "rate": m / t if t > 0 else None}

    return {
        "progress": {"reviewed": n_reviewed, "total": total_in_queue},
        "landmarks": {
            "total": total_landmarks,
            "matched": n_matched,
            "hallucination": n_hallucination,
            "novel": n_novel,
            "misidentified": n_misidentified,
            "generic": n_generic,
            "out_of_range": n_out_of_range,
            "unreviewed": n_unreviewed,
            "match_rate": n_matched / total_landmarks if total_landmarks > 0 else None,
            "hallucination_rate": n_hallucination / total_landmarks if total_landmarks > 0 else None,
            "misidentified_rate": n_misidentified / total_landmarks if total_landmarks > 0 else None,
        },
        "relevant_recall": {
            "matched": relevant_matched,
            "total": relevant_total,
            "recall": relevant_matched / relevant_total if relevant_total > 0 else None,
        },
        "calibration": calibration,
    }


def _score_landmark_relevance(row, pano_idx, lm_idx):
    """Score a landmark's relevance probability from its OSM tags and distance.

    Returns P(relevant | tags, distance) from the fitted model, or 0.5 if no model.
    """
    if LANDMARK_RELEVANCE_MODEL is None:
        return 0.5
    feats = _featurize_landmark(row)
    if not feats:
        return 0.5
    from scipy.sparse import hstack, lil_matrix
    n_tag_feats = len(LANDMARK_RELEVANCE_VOCAB)
    x_tags = lil_matrix((1, n_tag_feats), dtype=np.float32)
    for feat in feats:
        col = LANDMARK_RELEVANCE_VOCAB.get(feat)
        if col is not None:
            x_tags[0, col] = 1.0
    dist = _landmark_distance_px(pano_idx, lm_idx)
    x_dense = np.array([[dist / LANDMARK_RADIUS_PX, 1.0 / (1.0 + dist)]], dtype=np.float32)
    x = hstack([x_tags.tocsr(), x_dense]).tocsr()
    prob = LANDMARK_RELEVANCE_MODEL.predict_proba(x)[0]
    cls = LANDMARK_RELEVANCE_MODEL.classes_
    p_relevant = prob[1] if cls[1] == 1 else prob[0]
    return float(p_relevant)


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
    for pano_id, label in PANO_LABELS.items():
        pano_idx = PANO_ID_TO_IDX.get(pano_id)
        if pano_idx is None:
            continue
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
    global DINO_MODEL, PANO_CATEGORIES_DINO, DINO_CONFIDENCES, ACTIVE_LEARNING_QUEUE

    if DINO_FEATURES is None:
        return

    # Collect features and labels for labeled panoramas that have DINO data
    X, y = [], []
    for pano_id, label in PANO_LABELS.items():
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
        ACTIVE_LEARNING_QUEUE = []
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

    _recompute_active_learning_queue()


def _recompute_active_learning_queue():
    """Recompute the active learning priority queue from DINO confidences.

    Uses three signals:
    - Uncertainty: how close the model's confidence is to 0.5
    - Spatial disagreement: model prediction vs. nearby labeled panoramas
    - Spatial isolation: distance from nearest labeled panorama
    """
    global ACTIVE_LEARNING_QUEUE

    if DINO_MODEL is None or not DINO_CONFIDENCES:
        ACTIVE_LEARNING_QUEUE = []
        return

    from scipy.spatial import cKDTree

    labeled_idxs = sorted(
        PANO_ID_TO_IDX[pid] for pid in PANO_LABELS if pid in PANO_ID_TO_IDX
    )
    if not labeled_idxs:
        ACTIVE_LEARNING_QUEUE = []
        return

    # Build KDTree from labeled panorama coordinates
    labeled_coords = np.array([
        [PANORAMAS[i]["lat"], PANORAMAS[i]["lon"]] for i in labeled_idxs
    ])
    tree = cKDTree(labeled_coords)

    # Identify unlabeled panoramas that have DINO data
    labeled_pano_ids = set(PANO_LABELS.keys())
    unlabeled = []
    unlabeled_coords = []
    unlabeled_confs = []
    for i in range(len(PANORAMAS)):
        if PANORAMAS[i]["id"] in labeled_pano_ids:
            continue
        conf = DINO_CONFIDENCES[i]
        if conf is None:
            continue
        unlabeled.append(i)
        unlabeled_coords.append([PANORAMAS[i]["lat"], PANORAMAS[i]["lon"]])
        unlabeled_confs.append(conf)

    if not unlabeled:
        ACTIVE_LEARNING_QUEUE = []
        return

    unlabeled_coords = np.array(unlabeled_coords)
    unlabeled_confs = np.array(unlabeled_confs)

    # Query K nearest labeled neighbors for each unlabeled panorama
    k = min(SPATIAL_K, len(labeled_idxs))
    distances, neighbor_indices = tree.query(unlabeled_coords, k=k)
    # Ensure 2D shape even when k=1
    if k == 1:
        distances = distances.reshape(-1, 1)
        neighbor_indices = neighbor_indices.reshape(-1, 1)

    # Uncertainty: 1 = maximally uncertain (conf=0.5), 0 = confident
    uncertainty = 1.0 - 2.0 * np.abs(unlabeled_confs - 0.5)

    # Spatial disagreement: compare model confidence to fraction of useful neighbors
    frac_useful = np.zeros(len(unlabeled))
    for j in range(len(unlabeled)):
        n_useful = sum(
            1 for ni in neighbor_indices[j]
            if PANO_LABELS.get(PANORAMAS[labeled_idxs[ni]]["id"]) == "useful"
        )
        frac_useful[j] = n_useful / k
    spatial_disagreement = np.abs(unlabeled_confs - frac_useful)

    # Spatial isolation: distance to nearest labeled panorama
    nearest_dist = distances[:, 0]
    p95 = np.percentile(nearest_dist, 95) if len(nearest_dist) > 0 else 1.0
    if p95 > 0:
        spatial_isolation = np.clip(nearest_dist / p95, 0.0, 1.0)
    else:
        spatial_isolation = np.zeros(len(unlabeled))

    # Combined score
    scores = (AL_W_UNCERTAINTY * uncertainty
              + AL_W_DISAGREEMENT * spatial_disagreement
              + AL_W_ISOLATION * spatial_isolation)

    # Sort descending by score
    order = np.argsort(-scores)
    ACTIVE_LEARNING_QUEUE = [(float(scores[j]), unlabeled[j]) for j in order]

    print(f"Active learning queue: {len(ACTIVE_LEARNING_QUEUE)} candidates, "
          f"top score={ACTIVE_LEARNING_QUEUE[0][0]:.3f}")


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
.label-btn.active-bad-gps { background: #4a3a1c; border-color: #6a5a2d; color: #e6c860; }
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
        <button class="label-btn" id="btn-useful" onclick="setLabel('useful')" title="Good (g)">Good (g)</button>
        <button class="label-btn" id="btn-not-useful" onclick="setLabel('not_useful')" title="Bad (b)">Bad (b)</button>
        <button class="label-btn" id="btn-clear-label" onclick="setLabel(null)" title="Clear (x)">Clear</button>
        <button class="label-btn" id="btn-bad-gps" onclick="toggleBadGps()" title="Bad GPS (p)" style="margin-left:4px;">Bad GPS (p)</button>
        <button class="label-btn" id="btn-next-al" onclick="goToNextLabel()" title="Next to label (n)" style="margin-left:4px;background:#1a2a4e;">Next (n)</button>
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
let badGps = false;
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
    badGps = data.bad_gps || false;
    updateLabelButtons();
}

async function setLabel(label) {
    const idx = currentIndex;
    const resp = await fetch('/api/pano_label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: idx, label: label}),
    });
    const data = await resp.json();
    // Only update UI if we're still on the same panorama
    if (currentIndex !== idx) return;
    currentLabel = label;
    labelStats = {total: data.total_labels, n_useful: data.n_useful, n_not_useful: data.n_not_useful};
    updateLabelButtons();
    loadTagScores(idx);
    loadDinoPrediction(idx);
}

async function toggleBadGps() {
    const idx = currentIndex;
    const newVal = !badGps;
    const resp = await fetch('/api/bad_gps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: idx, bad_gps: newVal}),
    });
    const data = await resp.json();
    if (currentIndex !== idx) return;
    badGps = data.bad_gps;
    updateLabelButtons();
}

function updateLabelButtons() {
    const btnU = document.getElementById('btn-useful');
    const btnN = document.getElementById('btn-not-useful');
    const btnG = document.getElementById('btn-bad-gps');
    btnU.className = 'label-btn' + (currentLabel === 'useful' ? ' active-useful' : '');
    btnN.className = 'label-btn' + (currentLabel === 'not_useful' ? ' active-not-useful' : '');
    btnG.className = 'label-btn' + (badGps ? ' active-bad-gps' : '');
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

// Active learning: go to most informative unlabeled panorama
async function goToNextLabel() {
    try {
        const resp = await fetch('/api/next_to_label');
        const data = await resp.json();
        if (data.pano_idx !== null && data.pano_idx !== undefined) {
            const conf = data.confidence !== null ? (data.confidence * 100).toFixed(0) + '%' : '?';
            document.getElementById('save-status').textContent =
                `Next: #${data.pano_idx} (score=${data.score}, conf=${conf}, queue=${data.queue_size})`;
            setTimeout(() => document.getElementById('save-status').textContent = '', 5000);
            loadPanorama(data.pano_idx);
        } else {
            const reason = data.reason === 'no_model' ? 'Need 5+ labels per class for DINO model' : 'No candidates';
            document.getElementById('save-status').textContent = reason;
            setTimeout(() => document.getElementById('save-status').textContent = '', 4000);
        }
    } catch(e) {
        console.error('Failed to get next label:', e);
    }
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft') { navigate(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navigate(1); e.preventDefault(); }
    if (e.key === 'g') { setLabel('useful'); e.preventDefault(); }
    if (e.key === 'b') { setLabel('not_useful'); e.preventDefault(); }
    if (e.key === 'x') { setLabel(null); e.preventDefault(); }
    if (e.key === 'p') { toggleBadGps(); e.preventDefault(); }
    if (e.key === 'n') { goToNextLabel(); e.preventDefault(); }
});
</script>
</body>
</html>
"""


LANDING_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<title>Panorama Tools</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
.container { max-width: 600px; width: 100%; padding: 24px; }
h1 { color: #a0c0ff; margin-bottom: 24px; font-size: 28px; }
.card { display: block; padding: 20px; margin: 12px 0; background: #16213e; border: 1px solid #0f3460; border-radius: 8px; text-decoration: none; color: #e0e0e0; transition: all 0.15s; }
.card:hover { border-color: #a0c0ff; background: #1a2a4e; }
.card h2 { color: #a0c0ff; font-size: 18px; margin-bottom: 6px; }
.card p { color: #a0a0c0; font-size: 14px; line-height: 1.4; }
.stats { margin-top: 24px; padding: 16px; background: #16213e; border-radius: 8px; border: 1px solid #0f3460; font-size: 13px; color: #a0a0c0; }
.stats span { color: #a0c0ff; font-weight: bold; }
</style>
</head>
<body>
<div class="container">
    <h1>Panorama Tools</h1>
    <a class="card" href="/tagger">
        <h2>Panorama Tagger</h2>
        <p>Browse panoramas, label them as useful/not useful, search OSM tags, and build landmark annotations with Gemini predictions.</p>
    </a>
    <a class="card" href="/map">
        <h2>Overview Map</h2>
        <p>View all panoramas on a map colored by category (Gemini, OSM, Learned, DINO). Click panoramas to inspect details.</p>
    </a>
    <a class="card" href="/annotate">
        <h2>Landmark Annotation</h2>
        <p>For each useful-labeled panorama, review nearby OSM landmarks and mark which ones are the distinctive feature.</p>
    </a>
    <a class="card" href="/gemini_eval">
        <h2>Gemini Evaluation</h2>
        <p>Evaluate Gemini landmark extraction quality by matching predictions to OSM landmarks. Track hallucinations, novel finds, and confidence calibration.</p>
    </a>
    <div class="stats" id="stats">Loading stats...</div>
</div>
<script>
fetch('/api/annotate_queue').then(r => r.json()).then(aq => {
    fetch('/api/panorama_map_data').then(r => r.json()).then(md => {
        const s = md.label_stats;
        document.getElementById('stats').innerHTML =
            `<span>${md.panoramas.length}</span> panoramas | ` +
            `<span>${s.n_useful}</span> useful, <span>${s.n_not_useful}</span> not useful (${s.total} labeled) | ` +
            `<span>${aq.done}</span>/${aq.total} landmarks annotated`;
    });
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return Response(LANDING_PAGE, content_type="text/html")


@app.route("/tagger")
def tagger_page():
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
    fov_y = float(request.args.get("fov_y", fov))
    w = 768
    h = int(w * math.tan(fov_y / 2.0) / math.tan(fov / 2.0)) if fov > 0 else w
    h = max(1, h)
    pano_arr = _load_panorama(index)
    out = REPROJECTOR.reproject(
        pano_arr,
        output_shape=(h, w),
        fov=(fov, fov_y),
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
    pano_id = PANORAMAS[pano_idx]["id"]
    label = data.get("label")  # "useful", "not_useful", or null to clear
    if label:
        PANO_LABELS[pano_id] = label
    elif pano_id in PANO_LABELS:
        del PANO_LABELS[pano_id]
    _recompute_learned_categories()
    _retrain_dino_model()
    _build_gemini_eval_queue()
    _save_labels()
    n_useful = sum(1 for v in PANO_LABELS.values() if v == "useful")
    n_not_useful = sum(1 for v in PANO_LABELS.values() if v == "not_useful")
    result = {
        "ok": True,
        "total_labels": len(PANO_LABELS),
        "n_useful": n_useful,
        "n_not_useful": n_not_useful,
    }
    if ACTIVE_LEARNING_QUEUE:
        score, next_idx = ACTIVE_LEARNING_QUEUE[0]
        result["next_to_label"] = next_idx
        result["next_to_label_score"] = round(score, 3)
    return jsonify(result)


@app.route("/api/pano_label/<int:index>")
def get_pano_label(index):
    pano_id = PANORAMAS[index]["id"] if 0 <= index < len(PANORAMAS) else None
    label = PANO_LABELS.get(pano_id) if pano_id else None
    bad_gps = pano_id in BAD_GPS_PANOS if pano_id else False
    return jsonify({"label": label, "bad_gps": bad_gps})


@app.route("/api/bad_gps", methods=["POST"])
def toggle_bad_gps():
    data = request.get_json()
    if not data or "pano_idx" not in data:
        return "Missing pano_idx", 400
    pano_idx = int(data["pano_idx"])
    pano_id = PANORAMAS[pano_idx]["id"]
    if data.get("bad_gps", True):
        BAD_GPS_PANOS.add(pano_id)
    else:
        BAD_GPS_PANOS.discard(pano_id)
    _save_labels()
    return jsonify({"ok": True, "bad_gps": pano_id in BAD_GPS_PANOS})


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


@app.route("/api/next_to_label")
def next_to_label():
    if DINO_MODEL is None:
        return jsonify({"pano_idx": None, "reason": "no_model"})
    if not ACTIVE_LEARNING_QUEUE:
        return jsonify({"pano_idx": None, "reason": "no_candidates"})
    score, pano_idx = ACTIVE_LEARNING_QUEUE[0]
    conf = DINO_CONFIDENCES[pano_idx] if DINO_CONFIDENCES else None
    return jsonify({
        "pano_idx": pano_idx,
        "score": round(score, 3),
        "confidence": conf,
        "queue_size": len(ACTIVE_LEARNING_QUEUE),
    })


# ---------------------------------------------------------------------------
# Gemini evaluation endpoints
# ---------------------------------------------------------------------------


@app.route("/api/gemini_eval_queue")
def gemini_eval_queue():
    """Return eval queue with review status."""
    items = []
    n_reviewed = 0
    for pano_idx in GEMINI_EVAL_QUEUE:
        pano_id = PANORAMAS[pano_idx]["id"]
        ev = GEMINI_EVAL.get(pano_id, {})
        reviewed = ev.get("reviewed", False)
        if reviewed:
            n_reviewed += 1
        items.append({
            "pano_idx": pano_idx,
            "pano_id": pano_id,
            "reviewed": reviewed,
        })
    return jsonify({"queue": items, "total": len(items), "reviewed": n_reviewed})


@app.route("/api/gemini_eval_data/<int:index>")
def gemini_eval_data(index):
    """Return Gemini + OSM landmarks + match state for one panorama.

    Auto-matches if visiting for the first time.
    """
    if index < 0 or index >= len(PANORAMAS):
        return jsonify({"error": "Index out of range"}), 404

    pano_id = PANORAMAS[index]["id"]

    # Auto-match if no existing eval data
    if pano_id not in GEMINI_EVAL:
        matches = _auto_match_gemini_osm(index)
        GEMINI_EVAL[pano_id] = {"reviewed": False, "matches": matches}
        _save_labels()

    ev = GEMINI_EVAL[pano_id]

    # Gemini landmarks
    pred = GEMINI_PREDICTIONS.get(pano_id, {})
    gemini_landmarks = pred.get("landmarks", [])

    # OSM landmarks with osm_id
    lm_idxs = PANO_LANDMARK_IDXS[index]
    osm_landmarks = []
    for lm_idx in lm_idxs:
        row = LANDMARK_METADATA.iloc[lm_idx]
        osm_id = str(row["id"]) if "id" in row.index else ""
        props = {}
        for k, v in row["pruned_props"]:
            props[k] = v
        osm_landmarks.append({"osm_id": osm_id, "props": props})

    # Relevant OSM IDs
    relevant_osm_ids = RELEVANT_LANDMARKS.get(pano_id, [])

    return jsonify({
        "pano_id": pano_id,
        "reviewed": ev.get("reviewed", False),
        "matches": ev.get("matches", {}),
        "gemini_landmarks": gemini_landmarks,
        "osm_landmarks": osm_landmarks,
        "relevant_osm_ids": relevant_osm_ids,
        "location_type": pred.get("location_type"),
    })


@app.route("/api/gemini_eval_match", methods=["POST"])
def gemini_eval_match():
    """Set/update a match for a Gemini landmark.

    For status="matched", toggles osm_id in the osm_ids list.
    For other statuses (hallucination/novel/unreviewed), clears osm_ids.
    """
    data = request.get_json()
    if not data or "pano_idx" not in data or "gemini_idx" not in data:
        return "Missing pano_idx or gemini_idx", 400

    pano_idx = int(data["pano_idx"])
    pano_id = PANORAMAS[pano_idx]["id"]
    gemini_idx = str(data["gemini_idx"])
    osm_id = data.get("osm_id")
    status = data.get("status", "matched")

    if pano_id not in GEMINI_EVAL:
        matches = _auto_match_gemini_osm(pano_idx)
        GEMINI_EVAL[pano_id] = {"reviewed": False, "matches": matches}

    current = GEMINI_EVAL[pano_id]["matches"].get(gemini_idx, {"osm_ids": [], "status": "unreviewed"})
    osm_ids = current.get("osm_ids", [])
    # Migrate old single osm_id format if present
    if "osm_id" in current and "osm_ids" not in current:
        osm_ids = [current["osm_id"]] if current["osm_id"] else []

    keep_links = data.get("keep_links", False)

    if status == "matched" and osm_id:
        # Toggle: add if not present, remove if already there
        if osm_id in osm_ids:
            osm_ids.remove(osm_id)
        else:
            osm_ids.append(osm_id)
        # If all OSM IDs removed, revert to unreviewed
        final_status = "matched" if osm_ids else "unreviewed"
        GEMINI_EVAL[pano_id]["matches"][gemini_idx] = {
            "osm_ids": osm_ids,
            "status": final_status,
        }
    elif keep_links:
        # Change status but preserve existing osm_ids (e.g. misidentified)
        GEMINI_EVAL[pano_id]["matches"][gemini_idx] = {
            "osm_ids": osm_ids,
            "status": status,
        }
    else:
        # hallucination / novel / unreviewed — clear osm_ids
        GEMINI_EVAL[pano_id]["matches"][gemini_idx] = {
            "osm_ids": [],
            "status": status,
        }

    _save_labels()
    return jsonify({"ok": True, "matches": GEMINI_EVAL[pano_id]["matches"]})


@app.route("/api/gemini_eval_reviewed", methods=["POST"])
def gemini_eval_reviewed():
    """Toggle a panorama's reviewed status."""
    data = request.get_json()
    if not data or "pano_idx" not in data:
        return "Missing pano_idx", 400

    pano_idx = int(data["pano_idx"])
    pano_id = PANORAMAS[pano_idx]["id"]

    if pano_id not in GEMINI_EVAL:
        matches = _auto_match_gemini_osm(pano_idx)
        GEMINI_EVAL[pano_id] = {"reviewed": False, "matches": matches}

    reviewed = data.get("reviewed")
    if reviewed is None:
        # Toggle
        reviewed = not GEMINI_EVAL[pano_id].get("reviewed", False)
    GEMINI_EVAL[pano_id]["reviewed"] = reviewed
    _save_labels()
    _train_generic_classifier()
    return jsonify({"ok": True, "reviewed": reviewed})


@app.route("/api/gemini_eval_stats")
def gemini_eval_stats():
    """Return aggregate evaluation statistics."""
    return jsonify(_compute_gemini_eval_stats())


# ---------------------------------------------------------------------------
# Gemini evaluation page
# ---------------------------------------------------------------------------

GEMINI_EVAL_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<title>Gemini Evaluation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; }
.top-bar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }
.top-bar button { padding: 6px 14px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 14px; }
.top-bar button:hover { background: #0f3460; }
.top-bar .progress { color: #a0c0ff; font-size: 14px; font-weight: bold; }
.top-bar .pano-info { color: #a0a0c0; font-size: 13px; }
.stats-bar { display: flex; gap: 16px; padding: 4px 16px; background: #12192e; font-size: 12px; color: #a0a0c0; border-bottom: 1px solid #0f3460; }
.stats-bar .stat { display: flex; gap: 4px; }
.stats-bar .stat-val { color: #a0c0ff; font-weight: bold; }
.main-content { display: flex; height: calc(100vh - 90px); }
.left-panel { width: 60%; display: flex; flex-direction: column; overflow: hidden; }
.right-panel { width: 40%; display: flex; flex-direction: column; overflow: hidden; border-left: 1px solid #0f3460; }
.pinhole-row { display: flex; gap: 4px; padding: 8px; background: #16213e; justify-content: center; flex-shrink: 0; }
.pinhole-container { position: relative; display: inline-block; }
.pinhole-container img { height: 140px; border-radius: 4px; border: 2px solid #0f3460; cursor: pointer; transition: border-color 0.15s; display: block; }
.pinhole-container img:hover { border-color: #a0c0ff; }
.bbox-overlay { position: absolute; border: 2px solid #ff6b6b; background: rgba(255,107,107,0.15); pointer-events: none; border-radius: 2px; }
.interactive-view { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; padding: 8px; }
.interactive-view img { max-width: 100%; max-height: 100%; border-radius: 4px; cursor: grab; }
.interactive-view img:active { cursor: grabbing; }
.interactive-view .overlay { position: absolute; top: 16px; left: 16px; background: rgba(0,0,0,0.6); padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #a0c0ff; pointer-events: none; }
.panel-section { flex: 1; overflow-y: auto; padding: 8px; min-height: 0; }
.panel-section h3 { color: #a0c0ff; font-size: 14px; margin-bottom: 8px; padding: 0 4px; flex-shrink: 0; }
.gemini-card { padding: 8px 10px; margin: 4px 0; background: #16213e; border-radius: 6px; border: 2px solid #555; cursor: pointer; transition: all 0.15s; }
.gemini-card:hover { border-color: #a0c0ff; }
.gemini-card.selected { border-color: #4fc3f7 !important; background: #1a2a4e; box-shadow: 0 0 12px rgba(79,195,247,0.5), inset 0 0 0 1px rgba(79,195,247,0.3); }
.gemini-card.selected .gc-idx { background: #4fc3f7; color: #1a1a2e; border-radius: 3px; padding: 0 4px; }
.gemini-card.status-matched { border-color: #2d6a4f; }
.gemini-card.status-hallucination { border-color: #6a2d2d; }
.gemini-card.status-novel { border-color: #6a5a2d; }
.gemini-card.status-misidentified { border-color: #6a4a2d; }
.gemini-card.status-generic { border-color: #4a4a4a; }
.gemini-card.status-out_of_range { border-color: #2d4a6a; }
.gemini-card.status-unreviewed { border-color: #555; }
.gemini-card .gc-header { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
.gemini-card .gc-idx { color: #a0a0c0; font-size: 11px; font-weight: bold; min-width: 18px; }
.gemini-card .gc-status { font-size: 10px; padding: 1px 6px; border-radius: 3px; font-weight: bold; }
.gc-status.matched { background: #1b4332; color: #a0e0c0; }
.gc-status.hallucination { background: #4a1c1c; color: #ff9090; }
.gc-status.novel { background: #4a3a1c; color: #e6c860; }
.gc-status.misidentified { background: #4a3420; color: #e6a860; }
.gc-status.generic { background: #3a3a3a; color: #aaa; }
.gc-status.out_of_range { background: #1a3450; color: #80b0e0; }
.gc-status.unreviewed { background: #333; color: #999; }
.gemini-card .gc-conf { font-size: 11px; color: #a0a0c0; margin-left: auto; }
.gemini-card .gc-match-info { font-size: 11px; color: #a0e0c0; margin-top: 4px; padding: 3px 6px; background: #1b4332; border-radius: 3px; }
.tag-badge-primary { display: inline-block; padding: 1px 5px; background: #2d6a4f; border-radius: 3px; margin: 1px; font-size: 11px; color: #a0e0c0; }
.tag-badge-addl { display: inline-block; padding: 1px 5px; background: #0f3460; border-radius: 3px; margin: 1px; font-size: 11px; color: #a0a0c0; }
.gc-desc { color: #808090; font-size: 11px; margin-top: 2px; }
.osm-card { padding: 8px 10px; margin: 4px 0; background: #16213e; border-radius: 6px; border: 2px solid #0f3460; cursor: pointer; transition: all 0.15s; }
.osm-card:hover { border-color: #a0c0ff; }
.osm-card.matched { border-color: #2d6a4f; background: #1b4332; }
.osm-card .osm-name { font-weight: bold; font-size: 13px; margin-bottom: 4px; }
.osm-card .osm-tags { font-size: 11px; color: #a0a0c0; }
.osm-card .osm-tag { display: inline-block; padding: 1px 5px; background: #0f3460; border-radius: 3px; margin: 1px; }
.osm-card.matched .osm-tag { background: #2d6a4f; }
.osm-card .osm-id { font-size: 10px; color: #666; margin-top: 3px; }
.osm-card .relevant-star { color: #e6c860; margin-left: 4px; }
.divider { border-top: 1px solid #0f3460; margin: 4px 0; }
#ge-landmark-map { height: 280px; flex-shrink: 0; border-bottom: 1px solid #0f3460; }
.btn-reviewed-active { background: #1b4332 !important; border-color: #2d6a4f !important; color: #a0e0c0 !important; }
.help-btn { position: fixed; bottom: 16px; left: 16px; z-index: 1000; padding: 6px 12px; border: 1px solid #0f3460; background: #16213e; color: #a0a0c0; border-radius: 4px; cursor: pointer; font-size: 13px; }
.help-btn:hover { background: #0f3460; color: #e0e0e0; }
.help-panel { position: fixed; bottom: 50px; left: 16px; z-index: 1000; background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 14px 18px; display: none; min-width: 240px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
.help-panel.visible { display: block; }
.help-panel h4 { color: #a0c0ff; margin-bottom: 8px; font-size: 14px; }
.help-panel table { font-size: 13px; }
.help-panel td { padding: 2px 0; }
.help-panel td:first-child { color: #a0c0ff; font-weight: bold; padding-right: 14px; white-space: nowrap; }
</style>
</head>
<body>

<div class="top-bar">
    <button onclick="geNavigate(-1)" title="Prev (Left)">&#8592; Prev</button>
    <button onclick="geNavigate(1)" title="Next (Right)">Next &#8594;</button>
    <button onclick="geSkipUnreviewed()" title="Next unreviewed (s)">Skip unreviewed (s)</button>
    <button onclick="geToggleReviewed()" title="Toggle reviewed (r)" id="ge-btn-reviewed">Mark reviewed (r)</button>
    <span class="progress" id="ge-progress">Loading...</span>
    <span class="pano-info" id="ge-pano-info"></span>
    <span style="margin-left:auto;">
        <a href="/tagger" style="color:#a0c0ff;font-size:13px;">Tagger</a>
        &nbsp;|&nbsp;
        <a href="/annotate" style="color:#a0c0ff;font-size:13px;">Annotate</a>
        &nbsp;|&nbsp;
        <a href="/map" style="color:#a0c0ff;font-size:13px;">Map</a>
    </span>
</div>
<div class="stats-bar" id="ge-stats-bar">Loading stats...</div>

<div class="main-content">
    <div class="left-panel">
        <div class="pinhole-row" id="ge-pinhole-row"></div>
        <div class="interactive-view">
            <img id="ge-interactive-img" src="" draggable="false">
            <div class="overlay" id="ge-view-overlay">yaw: 0.0  pitch: 0.0  fov: 90</div>
        </div>
    </div>
    <div class="right-panel">
        <div class="panel-section" style="border-bottom: 1px solid #0f3460;">
            <h3>Gemini Landmarks (<span id="ge-gemini-count">0</span>)</h3>
            <div id="ge-gemini-cards"></div>
        </div>
        <div id="ge-landmark-map"></div>
        <div class="panel-section">
            <h3>OSM Landmarks (<span id="ge-osm-count">0</span>)</h3>
            <div id="ge-osm-cards"></div>
        </div>
    </div>
</div>

<script>
let geQueue = [];
let geQueuePos = 0;
let geCurrentIdx = -1;
let gePanoList = [];
let geData = null;       // current eval data from API
let geGeoJson = null;    // GeoJSON FeatureCollection from nearby_landmarks
let geSelectedGemini = null;  // currently selected gemini index (int or null)

// Map state
let geMap = null;
let geLandmarkLayer = null;

// View state
let geYaw = 0, gePitch = 0, geFov = 90;
let geDragging = false, geDsX = 0, geDsY = 0, geDsYaw = 0, geDsPitch = 0;
let geRpTimer = null, geLoadingRP = false;

window.addEventListener('DOMContentLoaded', async () => {
    // Init map
    geMap = L.map('ge-landmark-map', {zoomControl: true}).setView([41.88, -87.63], 18);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OSM', maxZoom: 20,
    }).addTo(geMap);
    geLandmarkLayer = L.layerGroup().addTo(geMap);

    const pResp = await fetch('/api/panorama_list');
    const pData = await pResp.json();
    gePanoList = pData.panoramas;

    await geRefreshQueue();

    // Start at first unreviewed
    const firstUnreviewed = geQueue.findIndex(q => !q.reviewed);
    geQueuePos = firstUnreviewed >= 0 ? firstUnreviewed : 0;
    if (geQueue.length > 0) geLoadPano(geQueuePos);

    geSetupDrag();
    geLoadStats();
});

async function geRefreshQueue() {
    const resp = await fetch('/api/gemini_eval_queue');
    const data = await resp.json();
    geQueue = data.queue;
    geUpdateProgress(data.reviewed, data.total);
}

function geUpdateProgress(reviewed, total) {
    document.getElementById('ge-progress').textContent = `${reviewed}/${total} reviewed`;
}

async function geLoadStats() {
    const resp = await fetch('/api/gemini_eval_stats');
    const stats = await resp.json();
    const lm = stats.landmarks;
    const rr = stats.relevant_recall;
    const cal = stats.calibration;
    let html = '';
    html += `<div class="stat">Progress: <span class="stat-val">${stats.progress.reviewed}/${stats.progress.total}</span></div>`;
    if (lm.total > 0) {
        html += `<div class="stat">Matched: <span class="stat-val">${lm.matched}</span> (${(lm.match_rate*100).toFixed(0)}%)</div>`;
        html += `<div class="stat">Hallucinations: <span class="stat-val">${lm.hallucination}</span> (${(lm.hallucination_rate*100).toFixed(0)}%)</div>`;
        html += `<div class="stat">Misidentified: <span class="stat-val">${lm.misidentified}</span> (${(lm.misidentified_rate*100).toFixed(0)}%)</div>`;
        html += `<div class="stat">Generic: <span class="stat-val">${lm.generic}</span></div>`;
        html += `<div class="stat">Novel: <span class="stat-val">${lm.novel}</span></div>`;
        html += `<div class="stat">Out of range: <span class="stat-val">${lm.out_of_range}</span></div>`;
    }
    if (rr.total > 0) {
        html += `<div class="stat">Relevant recall: <span class="stat-val">${rr.matched}/${rr.total}</span> (${(rr.recall*100).toFixed(0)}%)</div>`;
    }
    for (const [level, c] of Object.entries(cal)) {
        if (c.total > 0) {
            html += `<div class="stat">${level}: <span class="stat-val">${c.matched}/${c.total}</span> (${(c.rate*100).toFixed(0)}%)</div>`;
        }
    }
    document.getElementById('ge-stats-bar').innerHTML = html || 'No stats yet';
}

async function geLoadPano(queuePos) {
    if (queuePos < 0 || queuePos >= geQueue.length) return;
    geQueuePos = queuePos;
    const entry = geQueue[queuePos];
    geCurrentIdx = entry.pano_idx;
    geSelectedGemini = null;
    const pano = gePanoList[geCurrentIdx];

    // Update info
    const mapsUrl = `https://www.google.com/maps/place/${pano.lat},${pano.lon}/@${pano.lat},${pano.lon},18z`;
    document.getElementById('ge-pano-info').innerHTML =
        `#${queuePos + 1} of ${geQueue.length} | idx=${geCurrentIdx} | ${entry.pano_id} | ` +
        `<a href="${mapsUrl}" target="_blank" style="color:#a0c0ff;">(${pano.lat.toFixed(5)}, ${pano.lon.toFixed(5)})</a>` +
        ` | <a href="/tagger?goto=${geCurrentIdx}" target="_blank" style="color:#a0c0ff;">tagger</a>`;

    // Load pinholes
    const row = document.getElementById('ge-pinhole-row');
    row.innerHTML = '';
    for (const yaw of [0, 90, 180, 270]) {
        const container = document.createElement('div');
        container.className = 'pinhole-container';
        container.dataset.yaw = yaw;
        const img = document.createElement('img');
        img.src = `/api/image/pinhole/${geCurrentIdx}/${yaw}`;
        img.title = `${yaw}\u00b0`;
        img.onclick = () => { geYaw = yaw; gePitch = 0; geFov = 90; geUpdateOverlay(); geLoadReproject(); };
        container.appendChild(img);
        row.appendChild(container);
    }

    // Reset view
    geYaw = 0; gePitch = 0; geFov = 90;
    geUpdateOverlay();
    geLoadReproject();

    // Load eval data and GeoJSON in parallel
    const [evalResp, geoResp] = await Promise.all([
        fetch(`/api/gemini_eval_data/${geCurrentIdx}`),
        fetch(`/api/nearby_landmarks/${geCurrentIdx}`),
    ]);
    geData = await evalResp.json();
    geGeoJson = await geoResp.json();
    geRenderAll();
}

function geRenderAll() {
    if (!geData) return;
    geRenderGeminiCards();
    geRenderOsmCards();
    geRenderBboxes();
    geRenderMap();
    geUpdateReviewedBtn();
}

function geRenderGeminiCards() {
    const container = document.getElementById('ge-gemini-cards');
    const landmarks = geData.gemini_landmarks || [];
    const matches = geData.matches || {};
    document.getElementById('ge-gemini-count').textContent = landmarks.length;

    if (landmarks.length === 0) {
        container.innerHTML = '<div style="color:#a0a0c0;font-size:13px;padding:8px;">No Gemini landmarks</div>';
        return;
    }

    container.innerHTML = landmarks.map((lm, i) => {
        const m = matches[String(i)] || {status: 'unreviewed'};
        const status = m.status || 'unreviewed';
        const selected = geSelectedGemini === i ? ' selected' : '';
        const pt = lm.primary_tag || {};
        const primary = pt.key ? `<span class="tag-badge-primary">${pt.key}=${pt.value}</span>` : '';
        const addl = (lm.additional_tags || []).map(t =>
            `<span class="tag-badge-addl">${t.key}=${t.value}</span>`
        ).join('');
        const conf = lm.confidence || '';
        const desc = lm.description ? `<div class="gc-desc">${lm.description}</div>` : '';

        // Show matched OSM info
        let matchInfo = '';
        const osmIds = m.osm_ids || [];
        if ((status === 'matched' || status === 'misidentified') && osmIds.length > 0) {
            const names = osmIds.map(oid => {
                const osmLm = (geData.osm_landmarks || []).find(o => o.osm_id === oid);
                return osmLm ? (osmLm.props.name || oid) : oid;
            });
            matchInfo = `<div class="gc-match-info">&#8594; ${names.join(', ')}</div>`;
        }

        return `<div class="gemini-card status-${status}${selected}" data-gi="${i}" onclick="geSelectGemini(${i})">
            <div class="gc-header">
                <span class="gc-idx">${i + 1}</span>
                ${primary} ${addl}
                <span class="gc-status ${status}">${status}</span>
                <span class="gc-conf">[${conf}]</span>
            </div>
            ${desc}${matchInfo}
        </div>`;
    }).join('');
}

function geRenderOsmCards() {
    const container = document.getElementById('ge-osm-cards');
    const osmLandmarks = geData.osm_landmarks || [];
    const matches = geData.matches || {};
    const relevantIds = geData.relevant_osm_ids || [];
    document.getElementById('ge-osm-count').textContent = osmLandmarks.length;

    // Build set of highlighted OSM IDs: only the selected Gemini landmark's matches,
    // or all matched/misidentified if nothing is selected
    const linkedStatuses = new Set(['matched', 'misidentified']);
    const highlightedOsmIds = new Set();
    if (geSelectedGemini !== null) {
        const selMatch = matches[String(geSelectedGemini)];
        if (selMatch && linkedStatuses.has(selMatch.status)) {
            for (const oid of (selMatch.osm_ids || [])) highlightedOsmIds.add(oid);
        }
    } else {
        for (const [gi, m] of Object.entries(matches)) {
            if (linkedStatuses.has(m.status)) {
                for (const oid of (m.osm_ids || [])) highlightedOsmIds.add(oid);
            }
        }
    }

    if (osmLandmarks.length === 0) {
        container.innerHTML = '<div style="color:#a0a0c0;font-size:13px;padding:8px;">No OSM landmarks</div>';
        return;
    }

    // Sort: relevant first, then rest
    const sorted = [...osmLandmarks].sort((a, b) => {
        const aR = relevantIds.includes(a.osm_id) ? 0 : 1;
        const bR = relevantIds.includes(b.osm_id) ? 0 : 1;
        return aR - bR;
    });

    container.innerHTML = sorted.map((osm, i) => {
        const isMatched = highlightedOsmIds.has(osm.osm_id);
        const isRelevant = relevantIds.includes(osm.osm_id);
        const name = osm.props.name || '';
        const tags = Object.entries(osm.props)
            .filter(([k]) => k !== 'name' && !k.startsWith('tiger:') && !k.startsWith('gnis:'))
            .slice(0, 10)
            .map(([k, v]) => `<span class="osm-tag">${k}=${v}</span>`)
            .join(' ');
        const star = isRelevant ? '<span class="relevant-star">&#9733;</span>' : '';
        return `<div class="osm-card${isMatched ? ' matched' : ''}" data-osm-id="${osm.osm_id}" onclick="geMatchToOsm(this.dataset.osmId)">
            ${name ? `<div class="osm-name">${name}${star}</div>` : (star ? `<div>${star}</div>` : '')}
            <div class="osm-tags">${tags}</div>
            <div class="osm-id">${osm.osm_id}</div>
        </div>`;
    }).join('');
}

function geRenderBboxes() {
    // Clear existing bboxes
    document.querySelectorAll('.bbox-overlay').forEach(el => el.remove());

    if (geSelectedGemini === null || !geData) return;
    const lm = (geData.gemini_landmarks || [])[geSelectedGemini];
    if (!lm || !lm.bounding_boxes) return;

    const yawMap = {'0': 0, '90': 1, '180': 2, '270': 3};
    const containers = document.querySelectorAll('.pinhole-container');

    for (const bb of lm.bounding_boxes) {
        const yawKey = String(bb.yaw_angle);
        const containerIdx = yawMap[yawKey];
        if (containerIdx === undefined || containerIdx >= containers.length) continue;
        const container = containers[containerIdx];
        const img = container.querySelector('img');
        if (!img) continue;

        const div = document.createElement('div');
        div.className = 'bbox-overlay';
        // Coordinates are normalized 0-1000
        const scale = img.offsetHeight / 1000;
        div.style.left = (bb.xmin * scale) + 'px';
        div.style.top = (bb.ymin * scale) + 'px';
        div.style.width = ((bb.xmax - bb.xmin) * scale) + 'px';
        div.style.height = ((bb.ymax - bb.ymin) * scale) + 'px';
        container.appendChild(div);
    }
}

function geRenderMap() {
    if (!geMap || !geGeoJson) return;
    const pano = gePanoList[geCurrentIdx];
    geMap.setView([pano.lat, pano.lon], 18);
    geLandmarkLayer.clearLayers();

    // Panorama marker
    L.circleMarker([pano.lat, pano.lon], {
        radius: 6, fillColor: '#ff4444', color: '#fff', weight: 2, fillOpacity: 1,
    }).bindPopup('Panorama').addTo(geLandmarkLayer);

    // Build sets for styling
    const matches = (geData && geData.matches) || {};
    const relevantIds = (geData && geData.relevant_osm_ids) || [];
    const linkedStatuses = new Set(['matched', 'misidentified']);
    const matchedOsmIds = new Set();
    for (const [gi, m] of Object.entries(matches)) {
        if (linkedStatuses.has(m.status)) {
            for (const oid of (m.osm_ids || [])) matchedOsmIds.add(oid);
        }
    }
    // Selected Gemini's linked OSM IDs
    const selectedOsmIds = new Set();
    if (geSelectedGemini !== null) {
        const selMatch = matches[String(geSelectedGemini)];
        if (selMatch && linkedStatuses.has(selMatch.status)) {
            for (const oid of (selMatch.osm_ids || [])) selectedOsmIds.add(oid);
        }
    }

    // Map osm_id from eval data to GeoJSON features via tag matching
    const osmLandmarks = (geData && geData.osm_landmarks) || [];

    geGeoJson.features.forEach((f, i) => {
        const osmId = osmLandmarks[i] ? osmLandmarks[i].osm_id : '';
        const isRelevant = relevantIds.includes(osmId);
        const isSelected = selectedOsmIds.has(osmId);
        const isMatched = matchedOsmIds.has(osmId);

        let color = '#4fc3f7';
        let weight = 2;
        let fillOpacity = 0.3;
        let radius = 5;
        if (isSelected) {
            color = '#4fc3f7'; weight = 4; fillOpacity = 0.7; radius = 8;
        } else if (isMatched) {
            color = '#2d9e2d'; weight = 3; fillOpacity = 0.5; radius = 7;
        } else if (isRelevant) {
            color = '#e6c860'; weight = 3; fillOpacity = 0.5; radius = 7;
        }

        const layer = L.geoJSON(f, {
            style: { color, weight, fillColor: color, fillOpacity },
            pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
                radius, fillColor: color, color: '#fff',
                weight: isSelected ? 3 : (isMatched || isRelevant ? 2 : 1),
                fillOpacity: isSelected ? 0.9 : fillOpacity,
            }),
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                const lines = Object.entries(props)
                    .filter(([k,v]) => v && !k.startsWith('addr:') && !k.startsWith('tiger:'))
                    .slice(0, 12)
                    .map(([k,v]) => `<b>${k}</b>: ${v}`);
                if (lines.length) layer.bindPopup(lines.join('<br>'), {maxWidth: 300});
            },
        });
        layer.addTo(geLandmarkLayer);
    });

    // Invalidate size in case map was hidden
    setTimeout(() => geMap.invalidateSize(), 100);
}

function geSelectGemini(idx) {
    if (geSelectedGemini === idx) {
        geSelectedGemini = null;
    } else {
        geSelectedGemini = idx;
    }
    geRenderAll();
}

async function geMatchToOsm(osmId) {
    if (geSelectedGemini === null) return;
    const resp = await fetch('/api/gemini_eval_match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            pano_idx: geCurrentIdx,
            gemini_idx: geSelectedGemini,
            osm_id: osmId,
            status: 'matched',
        }),
    });
    const data = await resp.json();
    geData.matches = data.matches;
    geRenderAll();
}

async function geSetStatus(status) {
    if (geSelectedGemini === null) return;
    const resp = await fetch('/api/gemini_eval_match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            pano_idx: geCurrentIdx,
            gemini_idx: geSelectedGemini,
            osm_id: null,
            status: status,
        }),
    });
    const data = await resp.json();
    geData.matches = data.matches;
    geAdvanceSelection();
    geRenderAll();
}

async function geSetStatusKeepLinks(status) {
    // Change status but preserve existing osm_ids links
    if (geSelectedGemini === null) return;
    const resp = await fetch('/api/gemini_eval_match', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            pano_idx: geCurrentIdx,
            gemini_idx: geSelectedGemini,
            status: status,
            keep_links: true,
        }),
    });
    const data = await resp.json();
    geData.matches = data.matches;
    geAdvanceSelection();
    geRenderAll();
}

function geAdvanceSelection() {
    // Move to next unreviewed gemini landmark after current
    const landmarks = geData.gemini_landmarks || [];
    const matches = geData.matches || {};
    for (let i = geSelectedGemini + 1; i < landmarks.length; i++) {
        const m = matches[String(i)];
        if (!m || m.status === 'unreviewed') {
            geSelectedGemini = i;
            return;
        }
    }
    geSelectedGemini = null;
}

async function geToggleReviewed() {
    const resp = await fetch('/api/gemini_eval_reviewed', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: geCurrentIdx}),
    });
    const result = await resp.json();
    const nowReviewed = result.reviewed;
    geQueue[geQueuePos].reviewed = nowReviewed;
    geData.reviewed = nowReviewed;
    const reviewed = geQueue.filter(q => q.reviewed).length;
    geUpdateProgress(reviewed, geQueue.length);
    geUpdateReviewedBtn();
    geLoadStats();
    if (nowReviewed) geNavigate(1);
}

function geUpdateReviewedBtn() {
    const btn = document.getElementById('ge-btn-reviewed');
    if (!btn || !geData) return;
    if (geData.reviewed) {
        btn.classList.add('btn-reviewed-active');
        btn.textContent = 'Reviewed (r)';
    } else {
        btn.classList.remove('btn-reviewed-active');
        btn.textContent = 'Mark reviewed (r)';
    }
}

function geNavigate(delta) {
    const newPos = geQueuePos + delta;
    if (newPos >= 0 && newPos < geQueue.length) geLoadPano(newPos);
}

async function geSkipUnreviewed() {
    await geRefreshQueue();
    for (let i = geQueuePos + 1; i < geQueue.length; i++) {
        if (!geQueue[i].reviewed) { geLoadPano(i); return; }
    }
    for (let i = 0; i < geQueuePos; i++) {
        if (!geQueue[i].reviewed) { geLoadPano(i); return; }
    }
    geNavigate(1);
}

// --- Drag/zoom for interactive view ---
function geSetupDrag() {
    const img = document.getElementById('ge-interactive-img');
    img.addEventListener('mousedown', e => {
        geDragging = true; geDsX = e.clientX; geDsY = e.clientY;
        geDsYaw = geYaw; geDsPitch = gePitch; e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!geDragging) return;
        const dx = e.clientX - geDsX, dy = e.clientY - geDsY;
        const sensitivity = geFov / 600;
        geYaw = geDsYaw + dx * sensitivity;
        gePitch = Math.max(-80, Math.min(80, geDsPitch + dy * sensitivity));
        geUpdateOverlay();
        if (geRpTimer) clearTimeout(geRpTimer);
        geRpTimer = setTimeout(() => geLoadReproject(), 80);
    });
    window.addEventListener('mouseup', () => { if (geDragging) { geDragging = false; geLoadReproject(); } });
    img.addEventListener('wheel', e => {
        e.preventDefault();
        geFov = Math.max(20, Math.min(150, geFov + e.deltaY * 0.1));
        geUpdateOverlay(); geLoadReproject();
    });
}

function geUpdateOverlay() {
    document.getElementById('ge-view-overlay').textContent =
        `yaw: ${geYaw.toFixed(1)}  pitch: ${gePitch.toFixed(1)}  fov: ${geFov.toFixed(0)}`;
}

async function geLoadReproject() {
    if (geLoadingRP) return;
    geLoadingRP = true;
    const yR = geYaw * Math.PI / 180, pR = gePitch * Math.PI / 180, fR = geFov * Math.PI / 180;
    try {
        const resp = await fetch(`/api/image/reproject/${geCurrentIdx}?yaw=${yR}&pitch=${pR}&fov=${fR}`);
        if (resp.ok) {
            const blob = await resp.blob();
            const img = document.getElementById('ge-interactive-img');
            const oldSrc = img.src;
            img.src = URL.createObjectURL(blob);
            if (oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);
        }
    } finally { geLoadingRP = false; }
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { geNavigate(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { geNavigate(1); e.preventDefault(); }
    if (e.key === 's') { geSkipUnreviewed(); e.preventDefault(); }
    if (e.key === 'r') { geToggleReviewed(); e.preventDefault(); }
    if (e.key === 'h') { geSetStatus('hallucination'); e.preventDefault(); }
    if (e.key === 'n') { geSetStatus('novel'); e.preventDefault(); }
    if (e.key === 'm') { geSetStatusKeepLinks('misidentified'); e.preventDefault(); }
    if (e.key === 'g') { geSetStatus('generic'); e.preventDefault(); }
    if (e.key === 'o') { geSetStatus('out_of_range'); e.preventDefault(); }
    if (e.key === 'x') { geSetStatus('unreviewed'); e.preventDefault(); }
    if (e.key === 'Escape') { geSelectedGemini = null; geRenderAll(); e.preventDefault(); }
    // 1-9 to select gemini landmarks
    if (e.key >= '1' && e.key <= '9') {
        const idx = parseInt(e.key) - 1;
        if (geData && idx < (geData.gemini_landmarks || []).length) {
            geSelectGemini(idx);
            e.preventDefault();
        }
    }
});
</script>

<button class="help-btn" onclick="document.getElementById('help-panel').classList.toggle('visible')">? Keys</button>
<div class="help-panel" id="help-panel">
    <h4>Keyboard Shortcuts</h4>
    <table>
        <tr><td>1-9</td><td>Select Gemini landmark</td></tr>
        <tr><td>Esc</td><td>Deselect</td></tr>
        <tr><td>h</td><td>Mark hallucination</td></tr>
        <tr><td>n</td><td>Mark novel</td></tr>
        <tr><td>m</td><td>Mark misidentified</td></tr>
        <tr><td>g</td><td>Mark generic</td></tr>
        <tr><td>o</td><td>Mark out of range</td></tr>
        <tr><td>x</td><td>Clear to unreviewed</td></tr>
        <tr><td>r</td><td>Toggle reviewed</td></tr>
        <tr><td>s</td><td>Skip to next unreviewed</td></tr>
        <tr><td>&larr; &rarr;</td><td>Prev / Next</td></tr>
    </table>
</div>

</body>
</html>
"""


@app.route("/gemini_eval")
def gemini_eval_page():
    return Response(GEMINI_EVAL_PAGE, content_type="text/html")


# ---------------------------------------------------------------------------
# Landmark annotation endpoints
# ---------------------------------------------------------------------------


@app.route("/api/nearby_landmarks_annotate/<int:index>")
def nearby_landmarks_annotate(index):
    """Like nearby_landmarks but includes osm_id and relevant status.

    Landmarks are sorted by predicted relevance (most likely relevant first)
    using the tag-based relevance model when available.
    """
    global LANDMARK_RELEVANCE_DIRTY
    if index < 0 or index >= len(PANORAMAS) or LANDMARK_METADATA is None:
        return jsonify({"features": [], "relevant_osm_ids": []})

    # Retrain relevance model if annotations changed since last fit
    if LANDMARK_RELEVANCE_DIRTY:
        _retrain_landmark_relevance_model()

    pano_id = PANORAMAS[index]["id"]
    lm_idxs = PANO_LANDMARK_IDXS[index]
    relevant_osm_ids = RELEVANT_LANDMARKS.get(pano_id, [])
    features = []
    for lm_idx in lm_idxs:
        row = LANDMARK_METADATA.iloc[lm_idx]
        geom = row["geometry"]
        osm_id = str(row["id"]) if "id" in row.index else ""
        relevance_score = _score_landmark_relevance(row, index, lm_idx)
        props = {"osm_id": osm_id, "relevance_score": round(relevance_score, 3)}
        for k, v in row["pruned_props"]:
            props[k] = v
        features.append({
            "type": "Feature",
            "geometry": _geom_to_geojson(geom),
            "properties": props,
        })
    # Sort by predicted relevance descending
    features.sort(key=lambda f: -f["properties"]["relevance_score"])
    return jsonify({"features": features, "relevant_osm_ids": relevant_osm_ids})


@app.route("/api/relevant_landmark", methods=["POST"])
def toggle_relevant_landmark():
    """Toggle whether an OSM landmark is relevant for a panorama."""
    data = request.get_json()
    if not data or "pano_idx" not in data or "osm_id" not in data:
        return "Missing pano_idx or osm_id", 400
    pano_idx = int(data["pano_idx"])
    pano_id = PANORAMAS[pano_idx]["id"]
    osm_id = data["osm_id"]
    relevant = data.get("relevant", True)

    current = RELEVANT_LANDMARKS.get(pano_id, [])
    if relevant and osm_id not in current:
        current.append(osm_id)
    elif not relevant and osm_id in current:
        current.remove(osm_id)
    if current:
        RELEVANT_LANDMARKS[pano_id] = current
    else:
        # Don't store empty list from toggling — that's reserved for explicit
        # "none relevant" via mark_no_relevant. Remove the entry so the panorama
        # reverts to unannotated state.
        RELEVANT_LANDMARKS.pop(pano_id, None)
    _save_labels()
    global LANDMARK_RELEVANCE_DIRTY
    LANDMARK_RELEVANCE_DIRTY = True
    return jsonify({"ok": True, "relevant_osm_ids": current})


@app.route("/api/mark_no_relevant", methods=["POST"])
def mark_no_relevant():
    """Mark a panorama as having no relevant OSM landmarks."""
    data = request.get_json()
    if not data or "pano_idx" not in data:
        return "Missing pano_idx", 400
    pano_idx = int(data["pano_idx"])
    pano_id = PANORAMAS[pano_idx]["id"]
    RELEVANT_LANDMARKS[pano_id] = []
    _save_labels()
    global LANDMARK_RELEVANCE_DIRTY
    LANDMARK_RELEVANCE_DIRTY = True
    return jsonify({"ok": True})


@app.route("/api/annotate_queue")
def annotate_queue():
    """Return list of useful-labeled panoramas with annotation status."""
    useful_panos = []
    n_annotated = 0
    for i, p in enumerate(PANORAMAS):
        if PANO_LABELS.get(p["id"]) != "useful":
            continue
        annotated = p["id"] in RELEVANT_LANDMARKS
        if annotated:
            n_annotated += 1
        useful_panos.append({
            "pano_idx": i,
            "pano_id": p["id"],
            "annotated": annotated,
        })
    return jsonify({
        "queue": useful_panos,
        "total": len(useful_panos),
        "done": n_annotated,
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
        <a class="btn" href="/tagger?goto=${idx}" target="_blank">Open in tagger</a>
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
                "label": PANO_LABELS.get(p["id"]),
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


# ---------------------------------------------------------------------------
# Landmark annotation page
# ---------------------------------------------------------------------------

ANNOTATE_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<title>Landmark Annotation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; }
.top-bar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; flex-wrap: wrap; }
.top-bar button { padding: 6px 14px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; border-radius: 4px; cursor: pointer; font-size: 14px; }
.top-bar button:hover { background: #0f3460; }
.top-bar .progress { color: #a0c0ff; font-size: 14px; font-weight: bold; }
.top-bar .pano-info { color: #a0a0c0; font-size: 13px; }
.btn-none { padding: 6px 14px; border: 2px solid #6a2d2d; background: #1a1a2e; color: #ff9090; border-radius: 4px; cursor: pointer; font-size: 14px; }
.btn-none:hover { background: #4a1c1c; }
.btn-none.active { background: #4a1c1c; border-color: #ff6b6b; }
.btn-none.active-bad-gps { background: #4a3a1c; border-color: #e6c860; color: #e6c860; }
.main-content { display: flex; height: calc(100vh - 50px); }
.left-panel { width: 60%; display: flex; flex-direction: column; overflow: hidden; }
.right-panel { width: 40%; display: flex; flex-direction: column; overflow: hidden; border-left: 1px solid #0f3460; }
.pinhole-row { display: flex; gap: 4px; padding: 8px; background: #16213e; justify-content: center; flex-shrink: 0; }
.pinhole-row img { height: 140px; border-radius: 4px; border: 2px solid #0f3460; cursor: pointer; transition: border-color 0.15s; }
.pinhole-row img:hover { border-color: #a0c0ff; }
.interactive-view { flex: 1; position: relative; display: flex; align-items: center; justify-content: center; padding: 8px; }
.interactive-view img { max-width: 100%; max-height: 100%; border-radius: 4px; cursor: grab; }
.interactive-view img:active { cursor: grabbing; }
.interactive-view .overlay { position: absolute; top: 16px; left: 16px; background: rgba(0,0,0,0.6); padding: 4px 8px; border-radius: 4px; font-size: 12px; color: #a0c0ff; pointer-events: none; }
#landmark-map { height: 45%; min-height: 250px; flex-shrink: 0; }
.landmark-list { flex: 1; overflow: hidden; padding: 8px; display: flex; flex-direction: column; }
#lm-cards { flex: 1; overflow-y: auto; }
.landmark-list h3 { color: #a0c0ff; font-size: 14px; margin-bottom: 8px; padding: 0 4px; }
.lm-card { padding: 8px 10px; margin: 4px 0; background: #16213e; border-radius: 6px; border: 2px solid #0f3460; cursor: pointer; transition: all 0.15s; position: relative; }
.lm-card .lm-score { position: absolute; top: 6px; right: 8px; font-size: 11px; color: #a0a0c0; font-weight: bold; }
.lm-card:hover { border-color: #a0c0ff; }
.lm-card.relevant { border-color: #2d6a4f; background: #1b4332; }
.lm-card .lm-name { font-weight: bold; font-size: 13px; margin-bottom: 4px; }
.lm-card .lm-tags { font-size: 11px; color: #a0a0c0; }
.lm-card .lm-tag { display: inline-block; padding: 1px 5px; background: #0f3460; border-radius: 3px; margin: 1px; }
.lm-card.relevant .lm-tag { background: #2d6a4f; }
.lm-card .lm-id { font-size: 10px; color: #666; margin-top: 3px; }
</style>
</head>
<body>

<div class="top-bar">
    <button onclick="navigateAnnotate(-1)" title="Prev (←)">&#8592; Prev</button>
    <button onclick="navigateAnnotate(1)" title="Next (→)">Next &#8594;</button>
    <button onclick="skipAnnotate()" title="Next unannotated (s)">Next unannotated (s)</button>
    <button class="btn-none" id="btn-none-relevant" onclick="markNoneRelevant()" title="None relevant (0)">None relevant (0)</button>
    <button class="btn-none" id="ann-btn-bad-gps" onclick="annToggleBadGps()" title="Bad GPS (p)" style="border-color:#6a5a2d;">Bad GPS (p)</button>
    <span class="progress" id="ann-progress">Loading...</span>
    <span class="pano-info" id="ann-pano-info"></span>
    <span style="margin-left:auto;">
        <a href="/tagger" style="color:#a0c0ff;font-size:13px;">Tagger</a>
        &nbsp;|&nbsp;
        <a href="/map" style="color:#a0c0ff;font-size:13px;">Map</a>
    </span>
</div>

<div class="main-content">
    <div class="left-panel">
        <div class="pinhole-row" id="ann-pinhole-row"></div>
        <div class="interactive-view">
            <img id="ann-interactive-img" src="" draggable="false">
            <div class="overlay" id="ann-view-overlay">yaw: 0.0  pitch: 0.0  fov: 90</div>
            <div class="overlay" style="top:8px;left:auto;right:8px;pointer-events:auto;cursor:pointer;">
                <label style="cursor:pointer;"><input type="checkbox" id="lock-street" onchange="onLockStreet()"> Street view</label>
            </div>
        </div>
    </div>
    <div class="right-panel">
        <div id="landmark-map"></div>
        <div class="landmark-list" id="landmark-list-container">
            <h3>Nearby Landmarks (<span id="lm-count">0</span>)</h3>
            <input type="text" id="lm-filter" placeholder="Filter landmarks..." oninput="renderLandmarkCards()" style="width:100%;padding:6px 8px;border:1px solid #0f3460;background:#1a1a2e;color:#e0e0e0;border-radius:4px;margin-bottom:8px;font-size:13px;">
            <div id="lm-cards"></div>
        </div>
    </div>
</div>

<script>
let annQueue = [];        // [{pano_idx, pano_id, annotated}, ...]
let annQueuePos = 0;      // current position in queue
let annTotal = 0;
let annDone = 0;
let annCurrentIdx = -1;   // current pano_idx
let annBadGps = false;    // current panorama bad GPS flag
let annFeatures = [];     // current landmarks
let annRelevantIds = [];   // current relevant osm_ids
let annPanoList = [];     // full panorama list for image endpoints

// View state
let vYaw = 0, vPitch = 0, vFov = 90;
let dragging = false, dStartX = 0, dStartY = 0, dStartYaw = 0, dStartPitch = 0;
let reprojectTimer = null;
let loadingRP = false;
let amap = null, aLandmarkLayer = null;

window.addEventListener('DOMContentLoaded', async () => {
    // Init map
    amap = L.map('landmark-map', {zoomControl: true}).setView([41.88, -87.63], 18);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OSM', maxZoom: 20,
    }).addTo(amap);
    aLandmarkLayer = L.layerGroup().addTo(amap);

    // Load panorama list (for coordinates)
    const pResp = await fetch('/api/panorama_list');
    const pData = await pResp.json();
    annPanoList = pData.panoramas;

    // Load annotation queue
    await refreshQueue();

    // Find first unannotated or start at 0
    const firstUnannotated = annQueue.findIndex(q => !q.annotated);
    annQueuePos = firstUnannotated >= 0 ? firstUnannotated : 0;
    if (annQueue.length > 0) loadAnnotatePano(annQueuePos);

    setupAnnotateDrag();
});

async function refreshQueue() {
    const resp = await fetch('/api/annotate_queue');
    const data = await resp.json();
    annQueue = data.queue;
    annTotal = data.total;
    annDone = data.done;
    updateProgress();
}

function updateProgress() {
    document.getElementById('ann-progress').textContent =
        `${annDone}/${annTotal} annotated`;
}

async function loadAnnotatePano(queuePos) {
    if (queuePos < 0 || queuePos >= annQueue.length) return;
    annQueuePos = queuePos;
    const entry = annQueue[queuePos];
    annCurrentIdx = entry.pano_idx;
    const pano = annPanoList[annCurrentIdx];

    // Update info
    const mapsUrl = `https://www.google.com/maps/place/${pano.lat},${pano.lon}/@${pano.lat},${pano.lon},18z`;
    document.getElementById('ann-pano-info').innerHTML =
        `#${queuePos + 1} of ${annQueue.length} | idx=${annCurrentIdx} | ${entry.pano_id} | ` +
        `<a href="${mapsUrl}" target="_blank" style="color:#a0c0ff;">(${pano.lat.toFixed(5)}, ${pano.lon.toFixed(5)})</a>` +
        ` | <a href="/tagger?goto=${annCurrentIdx}" target="_blank" style="color:#a0c0ff;">tagger</a>`;

    // Load pinholes
    const row = document.getElementById('ann-pinhole-row');
    row.innerHTML = '';
    for (const yaw of [0, 90, 180, 270]) {
        const img = document.createElement('img');
        img.src = `/api/image/pinhole/${annCurrentIdx}/${yaw}`;
        img.title = `${yaw}\u00b0`;
        img.onclick = () => { vYaw = yaw; vPitch = 0; vFov = 90; updateAnnOverlay(); loadAnnReproject(); };
        row.appendChild(img);
    }

    // Reset view
    vYaw = 0; vPitch = 0; vFov = 90;
    updateAnnOverlay();
    loadAnnReproject();

    // Load landmarks
    const filterInput = document.getElementById('lm-filter');
    if (filterInput) filterInput.value = '';
    await loadAnnotateLandmarks(annCurrentIdx);
    // Load bad GPS flag
    const labelResp = await fetch(`/api/pano_label/${annCurrentIdx}`);
    const labelData = await labelResp.json();
    annBadGps = labelData.bad_gps || false;
    updateAnnBadGpsBtn();
}

async function loadAnnotateLandmarks(idx) {
    const pano = annPanoList[idx];
    amap.setView([pano.lat, pano.lon], 18);
    aLandmarkLayer.clearLayers();

    // Panorama marker
    L.circleMarker([pano.lat, pano.lon], {
        radius: 6, fillColor: '#ff4444', color: '#fff', weight: 2, fillOpacity: 1,
    }).bindPopup('Panorama').addTo(aLandmarkLayer);

    const resp = await fetch(`/api/nearby_landmarks_annotate/${idx}`);
    const data = await resp.json();
    annFeatures = data.features;
    annRelevantIds = data.relevant_osm_ids || [];

    renderLandmarkCards();
    renderLandmarksOnMap();
    updateNoneBtn();
}

function renderLandmarkCards() {
    const container = document.getElementById('lm-cards');
    const filterText = (document.getElementById('lm-filter')?.value || '').toLowerCase();
    const filtered = annFeatures.filter(f => {
        if (!filterText) return true;
        const p = f.properties;
        const searchable = Object.entries(p)
            .map(([k, v]) => `${k} ${v}`)
            .join(' ')
            .toLowerCase();
        return searchable.includes(filterText);
    });
    document.getElementById('lm-count').textContent =
        filterText ? `${filtered.length}/${annFeatures.length}` : `${annFeatures.length}`;
    if (filtered.length === 0) {
        container.innerHTML = `<div style="color:#a0a0c0;font-size:13px;padding:8px;">${annFeatures.length === 0 ? 'No nearby landmarks' : 'No matches'}</div>`;
        return;
    }
    container.innerHTML = filtered.map((f, i) => {
        const p = f.properties;
        const osmId = p.osm_id || '';
        const isRelevant = annRelevantIds.includes(osmId);
        const name = p.name || '';
        const tags = Object.entries(p)
            .filter(([k]) => k !== 'osm_id' && k !== 'name' && k !== 'relevance_score' && !k.startsWith('tiger:'))
            .slice(0, 10)
            .map(([k,v]) => `<span class="lm-tag">${k}=${v}</span>`)
            .join(' ');
        const score = p.relevance_score != null ? `${Math.round(p.relevance_score * 100)}%` : '';
        return `<div class="lm-card${isRelevant ? ' relevant' : ''}" data-osm-id="${osmId}" onclick="toggleLandmark(this.dataset.osmId)">
            ${score ? `<div class="lm-score">${score}</div>` : ''}
            ${name ? `<div class="lm-name">${name}</div>` : ''}
            <div class="lm-tags">${tags}</div>
            <div class="lm-id">${osmId}</div>
        </div>`;
    }).join('');
}

function renderLandmarksOnMap() {
    // Remove existing GeoJSON layers (keep panorama marker)
    aLandmarkLayer.eachLayer(l => {
        if (l._isGeoJSON) aLandmarkLayer.removeLayer(l);
    });

    annFeatures.forEach(f => {
        const osmId = f.properties.osm_id || '';
        const isRelevant = annRelevantIds.includes(osmId);
        const color = isRelevant ? '#2d9e2d' : '#4fc3f7';
        const fillColor = isRelevant ? '#2d9e2d' : '#4fc3f7';
        const weight = isRelevant ? 3 : 2;

        const layer = L.geoJSON(f, {
            style: { color, weight, fillColor, fillOpacity: isRelevant ? 0.5 : 0.3 },
            pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
                radius: isRelevant ? 7 : 5, fillColor, color: '#fff',
                weight: isRelevant ? 3 : 1, fillOpacity: isRelevant ? 0.8 : 0.7,
            }),
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                const lines = Object.entries(props)
                    .filter(([k,v]) => v && k !== 'osm_id' && !k.startsWith('addr:') && !k.startsWith('tiger:'))
                    .slice(0, 12)
                    .map(([k,v]) => `<b>${k}</b>: ${v}`);
                if (lines.length) layer.bindPopup(lines.join('<br>'), {maxWidth: 300});
            },
        });
        layer._isGeoJSON = true;
        layer.addTo(aLandmarkLayer);
    });
}

async function toggleLandmark(osmId) {
    if (!osmId) return;
    const isCurrentlyRelevant = annRelevantIds.includes(osmId);
    const resp = await fetch('/api/relevant_landmark', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: annCurrentIdx, osm_id: osmId, relevant: !isCurrentlyRelevant}),
    });
    const data = await resp.json();
    annRelevantIds = data.relevant_osm_ids;
    renderLandmarkCards();
    renderLandmarksOnMap();
    // Mark annotated only if there are selected landmarks; toggling all off
    // reverts to unannotated (use "None relevant" for explicit empty).
    annQueue[annQueuePos].annotated = annRelevantIds.length > 0;
    annDone = annQueue.filter(q => q.annotated).length;
    updateNoneBtn();
    updateProgress();
}

async function markNoneRelevant() {
    const resp = await fetch('/api/mark_no_relevant', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: annCurrentIdx}),
    });
    annRelevantIds = [];
    annQueue[annQueuePos].annotated = true;
    annDone = annQueue.filter(q => q.annotated).length;
    renderLandmarkCards();
    renderLandmarksOnMap();
    updateNoneBtn();
    updateProgress();
}

function updateNoneBtn() {
    const btn = document.getElementById('btn-none-relevant');
    const panoId = annQueue[annQueuePos]?.pano_id;
    // "None relevant" is active when the entry exists and is empty
    const isNone = annRelevantIds.length === 0 && annQueue[annQueuePos]?.annotated;
    btn.className = 'btn-none' + (isNone ? ' active' : '');
}

async function annToggleBadGps() {
    const newVal = !annBadGps;
    const resp = await fetch('/api/bad_gps', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pano_idx: annCurrentIdx, bad_gps: newVal}),
    });
    const data = await resp.json();
    annBadGps = data.bad_gps;
    updateAnnBadGpsBtn();
}

function updateAnnBadGpsBtn() {
    const btn = document.getElementById('ann-btn-bad-gps');
    btn.className = 'btn-none' + (annBadGps ? ' active-bad-gps' : '');
}

function navigateAnnotate(delta) {
    const newPos = annQueuePos + delta;
    if (newPos >= 0 && newPos < annQueue.length) loadAnnotatePano(newPos);
}

async function skipAnnotate() {
    await refreshQueue();
    // Move to next unannotated from current position
    for (let i = annQueuePos + 1; i < annQueue.length; i++) {
        if (!annQueue[i].annotated) { loadAnnotatePano(i); return; }
    }
    // Wrap around
    for (let i = 0; i < annQueuePos; i++) {
        if (!annQueue[i].annotated) { loadAnnotatePano(i); return; }
    }
    // All annotated, just go next
    navigateAnnotate(1);
}

// --- Drag/zoom for interactive view ---

function setupAnnotateDrag() {
    const img = document.getElementById('ann-interactive-img');
    img.addEventListener('mousedown', e => {
        dragging = true; dStartX = e.clientX; dStartY = e.clientY;
        dStartYaw = vYaw; dStartPitch = vPitch; e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        const dx = e.clientX - dStartX, dy = e.clientY - dStartY;
        const sensitivity = vFov / 600;
        vYaw = dStartYaw + dx * sensitivity;
        const locked = document.getElementById('lock-street')?.checked;
        vPitch = locked ? 0 : Math.max(-80, Math.min(80, dStartPitch + dy * sensitivity));
        updateAnnOverlay();
        if (reprojectTimer) clearTimeout(reprojectTimer);
        reprojectTimer = setTimeout(() => loadAnnReproject(), 80);
    });
    window.addEventListener('mouseup', () => { if (dragging) { dragging = false; loadAnnReproject(); } });
    img.addEventListener('wheel', e => {
        e.preventDefault();
        const locked = document.getElementById('lock-street')?.checked;
        if (!locked) {
            vFov = Math.max(20, Math.min(150, vFov + e.deltaY * 0.1));
        }
        updateAnnOverlay(); loadAnnReproject();
    });
}

function onLockStreet() {
    const locked = document.getElementById('lock-street').checked;
    if (locked) { vPitch = 0; vFov = 90; }
    updateAnnOverlay();
    loadAnnReproject();
}

function updateAnnOverlay() {
    document.getElementById('ann-view-overlay').textContent =
        `yaw: ${vYaw.toFixed(1)}  pitch: ${vPitch.toFixed(1)}  fov: ${vFov.toFixed(0)}`;
}

async function loadAnnReproject() {
    if (loadingRP) return;
    loadingRP = true;
    const locked = document.getElementById('lock-street')?.checked;
    const yR = vYaw * Math.PI / 180, pR = vPitch * Math.PI / 180, fR = vFov * Math.PI / 180;
    const fovYR = locked ? (45 * Math.PI / 180) : fR;
    try {
        const resp = await fetch(`/api/image/reproject/${annCurrentIdx}?yaw=${yR}&pitch=${pR}&fov=${fR}&fov_y=${fovYR}`);
        if (resp.ok) {
            const blob = await resp.blob();
            const img = document.getElementById('ann-interactive-img');
            const oldSrc = img.src;
            img.src = URL.createObjectURL(blob);
            if (oldSrc.startsWith('blob:')) URL.revokeObjectURL(oldSrc);
        }
    } finally { loadingRP = false; }
}

// Keyboard shortcuts
window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { navigateAnnotate(-1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { navigateAnnotate(1); e.preventDefault(); }
    if (e.key === '0') { markNoneRelevant(); e.preventDefault(); }
    if (e.key === 'p') { annToggleBadGps(); e.preventDefault(); }
    if (e.key === 's') { skipAnnotate(); e.preventDefault(); }
});
</script>
</body>
</html>
"""


@app.route("/annotate")
def annotate_page():
    return Response(ANNOTATE_PAGE, content_type="text/html")


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

    # Build pano_id -> pano_idx reverse lookup
    PANO_ID_TO_IDX.update({p["id"]: i for i, p in enumerate(PANORAMAS)})


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

    # Build OSM ID reverse lookup for landmark annotation
    if "id" in LANDMARK_METADATA.columns:
        OSM_ID_TO_LM_IDX.update({
            str(LANDMARK_METADATA.iloc[i]["id"]): i
            for i in range(len(LANDMARK_METADATA))
        })
        print(f"Built OSM ID index: {len(OSM_ID_TO_LM_IDX)} entries")

    print("Computing panorama-landmark associations...")
    pano_metadata = load_panorama_metadata(args.panorama_dir, zoom_level=20)
    # Build panorama pixel coordinate points for distance calculations
    for _, row in pano_metadata.iterrows():
        PANO_PX.append(shapely.Point(row["web_mercator_x"], row["web_mercator_y"]))
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
    migrated = _load_labels()
    if migrated:
        _save_labels()
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

    # Train landmark relevance model from existing annotations
    if RELEVANT_LANDMARKS:
        print("Training initial landmark relevance model from existing annotations...")
        _retrain_landmark_relevance_model()
    else:
        print("No landmark annotations yet, skipping relevance model")

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

    # Build Gemini evaluation queue
    _build_gemini_eval_queue()
    n_ge_reviewed = sum(1 for i in GEMINI_EVAL_QUEUE
                        if GEMINI_EVAL.get(PANORAMAS[i]["id"], {}).get("reviewed"))
    print(f"Gemini eval queue: {len(GEMINI_EVAL_QUEUE)} panoramas "
          f"({n_ge_reviewed} reviewed)")

    _train_generic_classifier()

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Output: {args.output}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
