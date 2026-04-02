"""Dataset for landmark correspondence classification.

Parses Gemini batch JSONL responses to extract (pano_tags, osm_tags, label) pairs.
Each JSONL line contains a prompt with Set 1 (pano) and Set 2 (OSM) tag bundles,
plus a Gemini response identifying matches and negatives.

Yields:
  - Positive pairs: pano landmark matched to OSM landmark
  - Hard negatives: same category but different landmark
  - Easy negatives: completely different type
"""

import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    BOOLEAN_KEYS,
    BooleanValue,
    HOUSENUMBER_KEY,
    NUMERIC_KEYS,
    NUM_TAG_KEYS,
    TAG_KEY_TO_IDX,
    ValueType,
    encode_housenumber_value,
    encode_numeric_value,
    key_type,
    parse_boolean,
    parse_housenumber,
    parse_maxheight,
    parse_numeric,
)


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def parse_tag_string(tag_str: str) -> dict[str, str]:
    """Parse 'key=value; key=value' into a dict."""
    tags = {}
    for part in tag_str.split("; "):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            tags[k.strip()] = v.strip()
    return tags


def parse_prompt_landmarks(prompt_text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse the prompt to extract Set 1 and Set 2 landmark tag bundles.

    The prompt format is:
        Set 1 (street-level observations):
         0. key=value; key=value
         1. key=value
        ...
        Set 2 (map database):
         0. key=value; key=value
        ...

    Returns: (set1_landmarks, set2_landmarks) where each is a list of tag dicts.
    """
    # Split into Set 1 and Set 2 sections
    set2_marker = "\n\nSet 2 (map database):\n"
    if set2_marker not in prompt_text:
        raise ValueError("Could not find Set 2 marker in prompt")

    set1_text, set2_text = prompt_text.split(set2_marker, 1)

    # Remove Set 1 header
    set1_header = "Set 1 (street-level observations):\n"
    if set1_text.startswith(set1_header):
        set1_text = set1_text[len(set1_header):]

    def parse_landmarks(text: str) -> list[dict[str, str]]:
        landmarks = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Match " N. tags..." pattern
            match = re.match(r"\d+\.\s+(.*)", line)
            if match:
                landmarks.append(parse_tag_string(match.group(1)))
        return landmarks

    return parse_landmarks(set1_text), parse_landmarks(set2_text)


@dataclass
class CorrespondencePair:
    """A single training pair."""
    pano_tags: dict[str, str]
    osm_tags: dict[str, str]
    label: float  # 1.0 = positive, 0.0 = negative
    difficulty: str  # "positive", "hard", "easy"
    uniqueness_score: int | None  # 1-5, from Gemini; None if not provided
    pano_id: str


def parse_jsonl_line(line_data: dict) -> list[CorrespondencePair]:
    """Parse one JSONL line into correspondence pairs.

    Returns list of CorrespondencePair for all matches and negatives.
    """
    pano_id = line_data["key"]

    # Parse prompt to get tag bundles
    prompt_text = line_data["request"]["contents"][0]["parts"][0]["text"]
    set1_landmarks, set2_landmarks = parse_prompt_landmarks(prompt_text)

    # Parse response
    response = line_data.get("response", {})
    candidates = response.get("candidates", [])
    if not candidates:
        return []

    try:
        resp_text = candidates[0]["content"]["parts"][0]["text"]
        resp = json.loads(resp_text)
    except (KeyError, IndexError, json.JSONDecodeError):
        return []

    pairs = []
    for match in resp.get("matches", []):
        set1_id = match.get("set_1_id")
        if set1_id is None or set1_id < 0 or set1_id >= len(set1_landmarks):
            continue

        pano_tags = set1_landmarks[set1_id]
        uniqueness = match.get("uniqueness_score", None)

        # Positive pairs
        for set2_id in match.get("set_2_matches", []):
            if set2_id < 0 or set2_id >= len(set2_landmarks):
                continue
            pairs.append(CorrespondencePair(
                pano_tags=pano_tags,
                osm_tags=set2_landmarks[set2_id],
                label=1.0,
                difficulty="positive",
                uniqueness_score=uniqueness,
                pano_id=pano_id,
            ))

        # Negative pairs
        for neg in match.get("negatives", []):
            set2_id = neg.get("set_2_id")
            if set2_id is None or set2_id < 0 or set2_id >= len(set2_landmarks):
                continue
            difficulty = neg.get("difficulty")
            if difficulty is None:
                continue
            pairs.append(CorrespondencePair(
                pano_tags=pano_tags,
                osm_tags=set2_landmarks[set2_id],
                label=0.0,
                difficulty=difficulty,
                uniqueness_score=uniqueness,
                pano_id=pano_id,
            ))

    return pairs


def load_pairs_from_directory(data_dir: Path) -> list[CorrespondencePair]:
    """Load all correspondence pairs from a city's response directory.

    Recursively finds all predictions.jsonl files under data_dir.
    """
    pairs = []
    jsonl_files = list(data_dir.rglob("predictions.jsonl"))
    if not jsonl_files:
        jsonl_files = list(data_dir.rglob("*.jsonl"))

    skipped = 0
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    pairs.extend(parse_jsonl_line(data))
                except (json.JSONDecodeError, ValueError):
                    skipped += 1
                    continue
    if skipped:
        print(f"WARNING: Skipped {skipped} unparseable JSONL lines across {len(jsonl_files)} files")
    return pairs


# ---------------------------------------------------------------------------
# Cross-pair feature computation
# ---------------------------------------------------------------------------

NUM_CROSS_FEATURES = 13


def compute_cross_features(
    pano_tags: dict[str, str],
    osm_tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor] | None = None,
) -> list[float]:
    """Compute cross-pair features between two tag bundles.

    Features (13 total):
      - Key Jaccard similarity (1)
      - Number of shared keys / 10 (1)
      - Exact value match count / 10 (1)
      - Text cosine similarity: max, mean, name-specific (3)
      - Numeric proximity per numeric key (6, sorted: building:levels, heritage,
        lanes, levels, maxheight, maxspeed)
      - Housenumber range overlap (1)
    """
    pano_keys = set(pano_tags.keys())
    osm_keys = set(osm_tags.keys())
    shared_keys = pano_keys & osm_keys
    union_keys = pano_keys | osm_keys

    features = []

    # Jaccard similarity
    jaccard = len(shared_keys) / max(len(union_keys), 1)
    features.append(jaccard)

    # Shared key count (normalized)
    features.append(len(shared_keys) / 10.0)

    # Exact value match count
    exact_matches = sum(1 for k in shared_keys if pano_tags[k] == osm_tags[k])
    features.append(exact_matches / 10.0)

    # Text cosine similarities (3 features)
    if text_embeddings is not None:
        text_sims = []
        name_sim = 0.0
        for k in shared_keys:
            if key_type(k) != ValueType.TEXT:
                continue
            pv = pano_tags[k]
            ov = osm_tags[k]
            if pv in text_embeddings and ov in text_embeddings:
                pe = text_embeddings[pv]
                oe = text_embeddings[ov]
                sim = F.cosine_similarity(pe.unsqueeze(0), oe.unsqueeze(0)).item()
                text_sims.append(sim)
                if k == "name":
                    name_sim = sim
        features.append(max(text_sims) if text_sims else 0.0)  # max
        features.append(sum(text_sims) / max(len(text_sims), 1) if text_sims else 0.0)  # mean
        features.append(name_sim)  # name-specific
    else:
        features.extend([0.0, 0.0, 0.0])

    # Numeric proximity (6 features, one per numeric key)
    numeric_keys_ordered = sorted(NUMERIC_KEYS)
    for nk in numeric_keys_ordered:
        if nk in pano_tags and nk in osm_tags:
            if nk == "maxheight":
                a = parse_maxheight(pano_tags[nk])
                b = parse_maxheight(osm_tags[nk])
            else:
                a, _ = parse_numeric(pano_tags[nk], key=nk)
                b, _ = parse_numeric(osm_tags[nk], key=nk)
            if not (math.isnan(a) or math.isnan(b)):
                scale = 10.0 if nk in ("maxspeed",) else 2.0
                features.append(math.exp(-abs(a - b) / scale))
            else:
                features.append(0.0)
        else:
            features.append(0.0)

    # Housenumber range overlap (1 feature)
    if HOUSENUMBER_KEY in pano_tags and HOUSENUMBER_KEY in osm_tags:
        plo, phi = parse_housenumber(pano_tags[HOUSENUMBER_KEY])
        olo, ohi = parse_housenumber(osm_tags[HOUSENUMBER_KEY])
        if not any(math.isnan(x) for x in (plo, phi, olo, ohi)):
            # Check if either number falls in the other's range
            overlap = (plo >= olo and plo <= ohi) or (olo >= plo and olo <= phi)
            features.append(1.0 if overlap else 0.0)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    assert len(features) == NUM_CROSS_FEATURES, f"Expected {NUM_CROSS_FEATURES} cross features, got {len(features)}"
    return features


# ---------------------------------------------------------------------------
# Tag encoding for model input
# ---------------------------------------------------------------------------

def encode_tag_bundle(
    tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor] | None,
    text_input_dim: int = 768,
) -> dict:
    """Encode a tag bundle into tensors for TagBundleEncoder.

    Returns dict with:
      key_indices: list[int]
      value_types: list[int]  (ValueType enum: 0=bool, 1=numeric, 2=housenumber, 3=text)
      boolean_values: list[int]  (BooleanValue enum: 0=true, 1=false, 2=unknown)
      numeric_values: list[list[float]]  (each is 4-dim)
      numeric_nan_mask: list[bool]
      housenumber_values: list[list[float]]  (each is 4-dim)
      housenumber_nan_mask: list[bool]
      text_embeddings: list[torch.Tensor]  (each is text_input_dim-dim)
    """
    key_indices = []
    value_types = []
    boolean_values = []
    numeric_values = []
    numeric_nan_masks = []
    housenumber_vals = []
    housenumber_nan_masks = []
    text_embs = []

    zero_text = torch.zeros(text_input_dim)

    for k, v in tags.items():
        if k not in TAG_KEY_TO_IDX:
            continue

        key_indices.append(TAG_KEY_TO_IDX[k])
        kt = key_type(k)

        if kt == ValueType.BOOLEAN:
            value_types.append(ValueType.BOOLEAN)
            bval = parse_boolean(v)
            boolean_values.append(
                BooleanValue.TRUE if bval > 0.7
                else (BooleanValue.FALSE if bval < 0.3 else BooleanValue.UNKNOWN)
            )
            numeric_values.append([0.0, 0.0, 0.0, 0.0])
            numeric_nan_masks.append(False)
            housenumber_vals.append([0.0, 0.0, 0.0, 0.0])
            housenumber_nan_masks.append(False)
            text_embs.append(zero_text)

        elif kt == ValueType.NUMERIC:
            value_types.append(ValueType.NUMERIC)
            boolean_values.append(BooleanValue.UNKNOWN)
            if k == "maxheight":
                x = parse_maxheight(v)
                is_lower_bound = False
            else:
                x, is_lower_bound = parse_numeric(v, key=k)
            is_nan = math.isnan(x)
            enc = encode_numeric_value(x, is_lower_bound)
            assert all(abs(e) <= 30 for e in enc), (
                f"Numeric encoding magnitude >30 for key={k!r}, value={v!r}: {enc}"
            )
            numeric_values.append(enc)
            numeric_nan_masks.append(is_nan)
            housenumber_vals.append([0.0, 0.0, 0.0, 0.0])
            housenumber_nan_masks.append(False)
            text_embs.append(zero_text)

        elif kt == ValueType.HOUSENUMBER:
            value_types.append(ValueType.HOUSENUMBER)
            boolean_values.append(BooleanValue.UNKNOWN)
            numeric_values.append([0.0, 0.0, 0.0, 0.0])
            numeric_nan_masks.append(False)
            lo, hi = parse_housenumber(v)
            is_nan = math.isnan(lo) or math.isnan(hi)
            enc = encode_housenumber_value(lo, hi)
            assert all(abs(e) <= 30 for e in enc), (
                f"Housenumber encoding magnitude >30 for value={v!r}: {enc}"
            )
            housenumber_vals.append(enc)
            housenumber_nan_masks.append(is_nan)
            text_embs.append(zero_text)

        else:  # text
            value_types.append(ValueType.TEXT)
            boolean_values.append(BooleanValue.UNKNOWN)
            numeric_values.append([0.0, 0.0, 0.0, 0.0])
            numeric_nan_masks.append(False)
            housenumber_vals.append([0.0, 0.0, 0.0, 0.0])
            housenumber_nan_masks.append(False)
            if text_embeddings is None:
                raise ValueError(
                    f"text_embeddings is required but not provided "
                    f"(key={k!r}, value={v!r})"
                )
            if v not in text_embeddings:
                raise KeyError(
                    f"Text value {v!r} for key {k!r} not found in "
                    f"text_embeddings ({len(text_embeddings)} entries)"
                )
            text_embs.append(text_embeddings[v])

    return {
        "key_indices": key_indices,
        "value_types": value_types,
        "boolean_values": boolean_values,
        "numeric_values": numeric_values,
        "numeric_nan_mask": numeric_nan_masks,
        "housenumber_values": housenumber_vals,
        "housenumber_nan_mask": housenumber_nan_masks,
        "text_embeddings": text_embs,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LandmarkCorrespondenceDataset(Dataset):
    """PyTorch dataset yielding dicts with encoded pano/osm tag bundles, cross-features, label, and difficulty."""

    def __init__(
        self,
        pairs: list[CorrespondencePair],
        text_embeddings: dict[str, torch.Tensor] | None = None,
        text_input_dim: int = 768,
        include_difficulties: tuple[str, ...] = ("positive", "easy"),
    ):
        """Initialize dataset.

        Args:
            pairs: List of CorrespondencePair from JSONL parsing
            text_embeddings: Pre-computed {value_string: tensor} for text keys
            text_input_dim: Dimension of pre-computed text embeddings
            include_difficulties: Which pair types to include
        """
        self.pairs = [p for p in pairs if p.difficulty in include_difficulties]
        self.text_embeddings = text_embeddings
        self.text_input_dim = text_input_dim

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pano_encoded = encode_tag_bundle(
            pair.pano_tags, self.text_embeddings, self.text_input_dim
        )
        osm_encoded = encode_tag_bundle(
            pair.osm_tags, self.text_embeddings, self.text_input_dim
        )
        cross_feats = compute_cross_features(
            pair.pano_tags, pair.osm_tags, self.text_embeddings,
        )
        return {
            "pano": pano_encoded,
            "osm": osm_encoded,
            "cross_features": cross_feats,
            "label": pair.label,
            "difficulty": pair.difficulty,
        }


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------

@dataclass
class CorrespondenceBatch:
    """Collated batch for CorrespondenceClassifier."""
    # Pano side (all B × max_pano_tags)
    pano_key_indices: torch.Tensor
    pano_value_type: torch.Tensor
    pano_boolean_values: torch.Tensor
    pano_numeric_values: torch.Tensor
    pano_numeric_nan_mask: torch.Tensor
    pano_housenumber_values: torch.Tensor
    pano_housenumber_nan_mask: torch.Tensor
    pano_text_embeddings: torch.Tensor
    pano_tag_mask: torch.Tensor
    # OSM side (all B × max_osm_tags)
    osm_key_indices: torch.Tensor
    osm_value_type: torch.Tensor
    osm_boolean_values: torch.Tensor
    osm_numeric_values: torch.Tensor
    osm_numeric_nan_mask: torch.Tensor
    osm_housenumber_values: torch.Tensor
    osm_housenumber_nan_mask: torch.Tensor
    osm_text_embeddings: torch.Tensor
    osm_tag_mask: torch.Tensor
    # Cross features (B × num_cross)
    cross_features: torch.Tensor
    # Labels (B,)
    labels: torch.Tensor

    def to(self, device: torch.device) -> "CorrespondenceBatch":
        """Move all tensors to device."""
        return CorrespondenceBatch(**{
            name: getattr(self, name).to(device) for name in self.__dataclass_fields__
        })

    def pin_memory(self) -> "CorrespondenceBatch":
        """Pin all tensors to memory for faster GPU transfer."""
        return CorrespondenceBatch(**{
            name: getattr(self, name).pin_memory() for name in self.__dataclass_fields__
        })


def _pad_encoded(encoded_list: list[dict], text_input_dim: int) -> dict[str, torch.Tensor]:
    """Pad and stack encoded tag bundles into batch tensors.

    Args:
        encoded_list: List of dicts from encode_tag_bundle
        text_input_dim: Dimension of text embeddings

    Returns:
        Dict of tensors, all with shape (B, max_tags, ...)
    """
    B = len(encoded_list)
    max_tags = max(len(e["key_indices"]) for e in encoded_list) if encoded_list else 1
    max_tags = max(max_tags, 1)  # At least 1 to avoid empty tensors

    key_indices = torch.zeros(B, max_tags, dtype=torch.long)
    value_types = torch.zeros(B, max_tags, dtype=torch.long)
    boolean_values = torch.full((B, max_tags), 2, dtype=torch.long)  # default: unknown
    numeric_values = torch.zeros(B, max_tags, 4)
    numeric_nan_mask = torch.zeros(B, max_tags, dtype=torch.bool)
    housenumber_values = torch.zeros(B, max_tags, 4)
    housenumber_nan_mask = torch.zeros(B, max_tags, dtype=torch.bool)
    text_embeddings = torch.zeros(B, max_tags, text_input_dim)
    tag_mask = torch.zeros(B, max_tags, dtype=torch.bool)

    for i, enc in enumerate(encoded_list):
        n = len(enc["key_indices"])
        if n == 0:
            continue
        tag_mask[i, :n] = True
        key_indices[i, :n] = torch.tensor(enc["key_indices"], dtype=torch.long)
        value_types[i, :n] = torch.tensor(enc["value_types"], dtype=torch.long)
        boolean_values[i, :n] = torch.tensor(enc["boolean_values"], dtype=torch.long)
        numeric_values[i, :n] = torch.tensor(enc["numeric_values"], dtype=torch.float)
        numeric_nan_mask[i, :n] = torch.tensor(enc["numeric_nan_mask"], dtype=torch.bool)
        housenumber_values[i, :n] = torch.tensor(enc["housenumber_values"], dtype=torch.float)
        housenumber_nan_mask[i, :n] = torch.tensor(enc["housenumber_nan_mask"], dtype=torch.bool)
        if enc["text_embeddings"]:
            text_embeddings[i, :n] = torch.stack(enc["text_embeddings"])

    return {
        "key_indices": key_indices,
        "value_type": value_types,
        "boolean_values": boolean_values,
        "numeric_values": numeric_values,
        "numeric_nan_mask": numeric_nan_mask,
        "housenumber_values": housenumber_values,
        "housenumber_nan_mask": housenumber_nan_mask,
        "text_embeddings": text_embeddings,
        "tag_mask": tag_mask,
    }


def collate_correspondence(samples: list[dict]) -> CorrespondenceBatch:
    """Collate function for DataLoader."""
    # Infer text embedding dim from first sample with a text embedding
    text_input_dim = None
    for s in samples:
        for emb in s["pano"]["text_embeddings"]:
            if emb.shape[0] > 0:
                text_input_dim = emb.shape[0]
                break
        if text_input_dim is not None:
            break
        for emb in s["osm"]["text_embeddings"]:
            if emb.shape[0] > 0:
                text_input_dim = emb.shape[0]
                break
        if text_input_dim is not None:
            break
    if text_input_dim is None:
        raise ValueError(
            "Cannot infer text_input_dim: no samples have text embeddings"
        )

    pano_encoded = [s["pano"] for s in samples]
    osm_encoded = [s["osm"] for s in samples]

    pano_tensors = _pad_encoded(pano_encoded, text_input_dim)
    osm_tensors = _pad_encoded(osm_encoded, text_input_dim)

    cross_features = torch.tensor([s["cross_features"] for s in samples], dtype=torch.float)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.float)

    return CorrespondenceBatch(
        pano_key_indices=pano_tensors["key_indices"],
        pano_value_type=pano_tensors["value_type"],
        pano_boolean_values=pano_tensors["boolean_values"],
        pano_numeric_values=pano_tensors["numeric_values"],
        pano_numeric_nan_mask=pano_tensors["numeric_nan_mask"],
        pano_housenumber_values=pano_tensors["housenumber_values"],
        pano_housenumber_nan_mask=pano_tensors["housenumber_nan_mask"],
        pano_text_embeddings=pano_tensors["text_embeddings"],
        pano_tag_mask=pano_tensors["tag_mask"],
        osm_key_indices=osm_tensors["key_indices"],
        osm_value_type=osm_tensors["value_type"],
        osm_boolean_values=osm_tensors["boolean_values"],
        osm_numeric_values=osm_tensors["numeric_values"],
        osm_numeric_nan_mask=osm_tensors["numeric_nan_mask"],
        osm_housenumber_values=osm_tensors["housenumber_values"],
        osm_housenumber_nan_mask=osm_tensors["housenumber_nan_mask"],
        osm_text_embeddings=osm_tensors["text_embeddings"],
        osm_tag_mask=osm_tensors["tag_mask"],
        cross_features=cross_features,
        labels=labels,
    )


def load_text_embeddings(path: Path) -> dict[str, torch.Tensor]:
    """Load pre-computed text embeddings from pickle file.

    Expected format: {value_string: numpy_array_or_tensor}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Convert numpy arrays to tensors if needed
    result = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v
        else:
            result[k] = torch.tensor(v, dtype=torch.float32)
    return result


# ---------------------------------------------------------------------------
# Parse failure reporting
# ---------------------------------------------------------------------------

PARSE_REPORT_PATH = Path("/tmp/landmark_correspondence_parse_report.txt")


@dataclass
class ParseFailure:
    """A single parse failure."""
    key: str
    raw_value: str
    key_type_str: str  # "boolean", "numeric", "housenumber"
    failure_reason: str  # "nan", "unknown_boolean"


def scan_parse_failures(
    pairs: list[CorrespondencePair],
    text_embeddings: dict[str, torch.Tensor] | None = None,
) -> list[ParseFailure]:
    """Scan all pairs and collect parse failures for non-text keys.

    Failures are:
      - Boolean: value parsed to 0.5 (unknown/unrecognized)
      - Numeric: value parsed to NaN
      - Housenumber: value parsed to (NaN, NaN)
      - Text: value not in text_embeddings dict (missing embedding)
    """
    failures = []
    seen = set()  # (key, raw_value) to avoid duplicate reports

    for pair in pairs:
        for tags in [pair.pano_tags, pair.osm_tags]:
            for k, v in tags.items():
                if (k, v) in seen:
                    continue
                seen.add((k, v))

                kt = key_type(k)

                if kt == ValueType.BOOLEAN:
                    parsed = parse_boolean(v)
                    if parsed == 0.5:
                        failures.append(ParseFailure(
                            key=k, raw_value=v, key_type_str="boolean",
                            failure_reason="unknown_boolean",
                        ))

                elif kt == ValueType.NUMERIC:
                    if k == "maxheight":
                        parsed = parse_maxheight(v)
                    else:
                        parsed, _ = parse_numeric(v, key=k)
                    if math.isnan(parsed):
                        failures.append(ParseFailure(
                            key=k, raw_value=v, key_type_str="numeric",
                            failure_reason="nan",
                        ))

                elif kt == ValueType.HOUSENUMBER:
                    lo, hi = parse_housenumber(v)
                    if math.isnan(lo) or math.isnan(hi):
                        failures.append(ParseFailure(
                            key=k, raw_value=v, key_type_str="housenumber",
                            failure_reason="nan",
                        ))

                elif kt == ValueType.TEXT:
                    if text_embeddings is not None and v not in text_embeddings:
                        failures.append(ParseFailure(
                            key=k, raw_value=v, key_type_str="text",
                            failure_reason="missing_embedding",
                        ))

    return failures


def write_parse_report(
    failures: list[ParseFailure],
    pairs: list[CorrespondencePair],
    report_path: Path = PARSE_REPORT_PATH,
) -> None:
    """Write parse failure report to file.

    Groups failures by key and type, shows raw values and counts.
    """
    # Count total values per key type to compute failure rates
    type_counts: Counter = Counter()
    key_value_counts: Counter = Counter()
    seen = set()
    for pair in pairs:
        for tags in [pair.pano_tags, pair.osm_tags]:
            for k, v in tags.items():
                if (k, v) in seen:
                    continue
                seen.add((k, v))
                kt = key_type(k)
                type_counts[kt] += 1
                key_value_counts[k] += 1

    # Group failures
    by_key: defaultdict[str, list[ParseFailure]] = defaultdict(list)
    for f in failures:
        by_key[f.key].append(f)

    by_type: defaultdict[str, list[ParseFailure]] = defaultdict(list)
    for f in failures:
        by_type[f.key_type_str].append(f)

    lines = []
    lines.append("=" * 80)
    lines.append("PARSE FAILURE REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Summary
    lines.append("SUMMARY BY TYPE")
    lines.append("-" * 40)
    _type_name_to_enum = {
        "boolean": ValueType.BOOLEAN, "numeric": ValueType.NUMERIC,
        "housenumber": ValueType.HOUSENUMBER, "text": ValueType.TEXT,
    }
    for kt in ["boolean", "numeric", "housenumber", "text"]:
        n_fail = len(by_type.get(kt, []))
        n_total = type_counts.get(_type_name_to_enum[kt], 0)
        pct = (n_fail / n_total * 100) if n_total > 0 else 0
        lines.append(f"  {kt:15s}: {n_fail:5d} / {n_total:5d} unique values failed ({pct:.1f}%)")
    lines.append(f"  {'TOTAL':15s}: {len(failures):5d} / {sum(type_counts.values()):5d}")
    lines.append("")

    # Per-key breakdown
    lines.append("FAILURES BY KEY")
    lines.append("-" * 40)
    for key in sorted(by_key.keys()):
        key_failures = by_key[key]
        n_total = key_value_counts.get(key, 0)
        n_fail = len(key_failures)
        pct = (n_fail / n_total * 100) if n_total > 0 else 0
        lines.append(f"\n  {key} ({key_failures[0].key_type_str}): "
                      f"{n_fail}/{n_total} failed ({pct:.1f}%)")

        # Show all failing values (sorted by value)
        for f in sorted(key_failures, key=lambda x: x.raw_value):
            lines.append(f"    {f.failure_reason:20s}  {f.raw_value!r}")

    lines.append("")
    lines.append("=" * 80)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Parse failure report written to {report_path}")
