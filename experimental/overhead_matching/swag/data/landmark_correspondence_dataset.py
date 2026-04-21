"""Dataset for landmark correspondence classification.

Parses Gemini batch JSONL responses to extract (pano_tags, osm_tags, label)
pairs. Each JSONL line contains a prompt with Set 1 (pano) and Set 2 (OSM)
tag bundles plus a Gemini response identifying matches and negatives.

All tag values are encoded as text via pre-computed embeddings. Values
missing from the embeddings dict raise `KeyError` by default; callers that
explicitly want silent fallback to zero vectors can pass
`allow_missing_text_embeddings=True`.

Yields:
  - Positive pairs: pano landmark matched to OSM landmark
  - Hard negatives: same category but different landmark
  - Easy negatives: completely different type
"""

import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    NUM_CROSS_FEATURES,
    TAG_KEY_TO_IDX,
)


HOUSENUMBER_KEY = "addr:housenumber"


# ---------------------------------------------------------------------------
# Value parsing helpers
# ---------------------------------------------------------------------------

def parse_housenumber(value) -> tuple[float, float]:
    """Parse addr:housenumber to (low, high) range.

    "665-667" → (665, 667), "1858" → (1858, 1858). Any parse failure
    (non-string input, alphabetic suffix, etc.) returns (nan, nan).
    """
    try:
        v = value.strip()
        if "-" in v:
            parts = v.split("-", 1)
            try:
                return (float(parts[0].strip()), float(parts[1].strip()))
            except (ValueError, TypeError):
                pass
        return (float(v), float(v))
    except (ValueError, TypeError, AttributeError):
        return (float("nan"), float("nan"))


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


def parse_prompt_landmarks(
    prompt_text: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
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
    set2_marker = "\n\nSet 2 (map database):\n"
    if set2_marker not in prompt_text:
        raise ValueError("Could not find Set 2 marker in prompt")

    set1_text, set2_text = prompt_text.split(set2_marker, 1)

    set1_header = "Set 1 (street-level observations):\n"
    if set1_text.startswith(set1_header):
        set1_text = set1_text[len(set1_header):]

    def parse_landmarks(text: str) -> list[dict[str, str]]:
        landmarks = []
        for line in text.strip().split("\n"):
            line = line.strip()
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

    prompt_text = line_data["request"]["contents"][0]["parts"][0]["text"]
    set1_landmarks, set2_landmarks = parse_prompt_landmarks(prompt_text)

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
        print(f"WARNING: Skipped {skipped} unparseable JSONL lines across "
              f"{len(jsonl_files)} files")
    return pairs


# ---------------------------------------------------------------------------
# Text embedding loading
# ---------------------------------------------------------------------------

def load_text_embeddings(path: Path) -> dict[str, torch.Tensor]:
    """Load pre-computed text embeddings from pickle file.

    Expected format: {value_string: numpy_array_or_tensor}. All entries are
    converted to float32 tensors and required to share a single embedding
    dimension; otherwise a ValueError is raised up front so the caller sees
    a meaningful error rather than a shape mismatch deep inside a forward pass.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    result: dict[str, torch.Tensor] = {}
    dim: int | None = None
    for k, v in data.items():
        t = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        if t.ndim != 1:
            raise ValueError(
                f"Text embedding for {k!r} has shape {tuple(t.shape)}; "
                f"expected 1-D vector."
            )
        if dim is None:
            dim = t.shape[0]
        elif t.shape[0] != dim:
            raise ValueError(
                f"Text embedding dim mismatch at key {k!r}: "
                f"got {t.shape[0]}, expected {dim} (inferred from earlier entries)."
            )
        result[k] = t.to(torch.float32)
    return result


# ---------------------------------------------------------------------------
# Per-pair encoding
# ---------------------------------------------------------------------------

def encode_tag_bundle(
    tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int,
    allow_missing_text_embeddings: bool = False,
) -> dict:
    """Encode a tag bundle into tensors. All values encoded as text.

    Returns dict with `key_indices` (list[int]) and `text_embeddings`
    (list[torch.Tensor]). The two lists are parallel and padded/stacked
    downstream by `_pad_side`.

    Raises KeyError if a value is not in `text_embeddings` unless
    `allow_missing_text_embeddings=True`, in which case missing values fall
    back to a zero vector.
    """
    key_indices = []
    text_embs = []
    zero_text = torch.zeros(text_input_dim)

    for k, v in tags.items():
        if k not in TAG_KEY_TO_IDX:
            continue
        if v in text_embeddings:
            text_embs.append(text_embeddings[v])
        elif allow_missing_text_embeddings:
            text_embs.append(zero_text)
        else:
            raise KeyError(
                f"Text value {v!r} for key {k!r} not found in "
                f"text_embeddings ({len(text_embeddings)} entries). "
                f"Pass allow_missing_text_embeddings=True to fall back to zeros."
            )
        key_indices.append(TAG_KEY_TO_IDX[k])

    return {
        "key_indices": key_indices,
        "text_embeddings": text_embs,
    }


def compute_cross_features(
    pano_tags: dict[str, str],
    osm_tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor] | None = None,
) -> list[float]:
    """Compute 4 cross-pair features.

    Layout: [text_max, text_mean, text_name, housenumber_overlap].
    When `text_embeddings` is None, the three text features fall back to 0.0.
    """
    features: list[float] = []

    shared_keys = set(pano_tags.keys()) & set(osm_tags.keys())
    if text_embeddings is not None:
        text_sims: list[float] = []
        name_sim = 0.0
        for k in shared_keys:
            pv, ov = pano_tags[k], osm_tags[k]
            if pv in text_embeddings and ov in text_embeddings:
                sim = F.cosine_similarity(
                    text_embeddings[pv].unsqueeze(0),
                    text_embeddings[ov].unsqueeze(0),
                ).item()
                text_sims.append(sim)
                if k == "name":
                    name_sim = sim
        features.append(max(text_sims) if text_sims else 0.0)
        features.append(
            sum(text_sims) / len(text_sims) if text_sims else 0.0
        )
        features.append(name_sim)
    else:
        features.extend([0.0, 0.0, 0.0])

    # Housenumber range overlap (1 feature)
    if HOUSENUMBER_KEY in pano_tags and HOUSENUMBER_KEY in osm_tags:
        plo, phi = parse_housenumber(pano_tags[HOUSENUMBER_KEY])
        olo, ohi = parse_housenumber(osm_tags[HOUSENUMBER_KEY])
        if not any(math.isnan(x) for x in (plo, phi, olo, ohi)):
            overlap = (plo >= olo and plo <= ohi) or (olo >= plo and olo <= phi)
            features.append(1.0 if overlap else 0.0)
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    assert len(features) == NUM_CROSS_FEATURES
    return features


# ---------------------------------------------------------------------------
# Batch dataclass and collation
# ---------------------------------------------------------------------------

@dataclass
class CorrespondenceBatch:
    pano_key_indices: torch.Tensor       # (B, T) int
    pano_text_embeddings: torch.Tensor   # (B, T, text_dim) float
    pano_tag_mask: torch.Tensor          # (B, T) bool
    osm_key_indices: torch.Tensor        # (B, T) int
    osm_text_embeddings: torch.Tensor    # (B, T, text_dim) float
    osm_tag_mask: torch.Tensor           # (B, T) bool
    cross_features: torch.Tensor         # (B, NUM_CROSS_FEATURES) float
    labels: torch.Tensor                 # (B,) float

    def to(self, device: torch.device | str) -> "CorrespondenceBatch":
        return CorrespondenceBatch(**{
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        })

    def pin_memory(self) -> "CorrespondenceBatch":
        return CorrespondenceBatch(**{
            name: getattr(self, name).pin_memory()
            for name in self.__dataclass_fields__
        })


def _pad_side(encoded_list: List[dict], text_input_dim: int):
    """Pad a list of encoded tag bundles to uniform length.

    Returns (key_indices, text_embeddings, tag_mask) stacked across the batch.
    Every sample ends up with at least T=1 slot (padded if empty); the mask
    marks padded positions as False.
    """
    max_tags = max(len(e["key_indices"]) for e in encoded_list)
    max_tags = max(max_tags, 1)

    key_indices = []
    text_embs = []
    masks = []

    for e in encoded_list:
        n = len(e["key_indices"])
        pad = max_tags - n

        key_indices.append(
            torch.tensor(e["key_indices"] + [0] * pad, dtype=torch.long)
        )
        if n > 0:
            stacked = torch.stack(e["text_embeddings"])
        else:
            stacked = torch.zeros(0, text_input_dim)
        if pad > 0:
            stacked = torch.cat([stacked, torch.zeros(pad, text_input_dim)])
        text_embs.append(stacked)
        masks.append(
            torch.tensor([True] * n + [False] * pad, dtype=torch.bool)
        )

    return (
        torch.stack(key_indices),
        torch.stack(text_embs),
        torch.stack(masks),
    )


def collate_correspondence(
    batch: list[dict],
    text_input_dim: int,
) -> CorrespondenceBatch:
    """Collate function for DataLoader.

    `text_input_dim` must match the dimension of the text embeddings used by
    the dataset; bind it via `functools.partial` when constructing the
    DataLoader (see `LandmarkCorrespondenceDataset.text_input_dim`).
    """
    pano_keys, pano_text, pano_mask = _pad_side(
        [b["pano"] for b in batch], text_input_dim
    )
    osm_keys, osm_text, osm_mask = _pad_side(
        [b["osm"] for b in batch], text_input_dim
    )
    cross = torch.tensor([b["cross_features"] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    return CorrespondenceBatch(
        pano_key_indices=pano_keys,
        pano_text_embeddings=pano_text,
        pano_tag_mask=pano_mask,
        osm_key_indices=osm_keys,
        osm_text_embeddings=osm_text,
        osm_tag_mask=osm_mask,
        cross_features=cross,
        labels=labels,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LandmarkCorrespondenceDataset(Dataset):
    """Correspondence pairs encoded as text-embedding bundles.

    At construction time, scans all pairs to summarize text-embedding
    coverage and zero-tag rates. If any value string is missing from
    `text_embeddings` and `allow_missing_text_embeddings` is False, raises
    KeyError listing a handful of examples so the caller can spot bad data
    early rather than degrading silently during training.
    """

    def __init__(
        self,
        pairs: list[CorrespondencePair],
        text_embeddings: dict[str, torch.Tensor],
        text_input_dim: int,
        include_difficulties: tuple[str, ...] = ("positive", "easy", "hard"),
        allow_missing_text_embeddings: bool = False,
    ):
        self.pairs = [p for p in pairs if p.difficulty in include_difficulties]
        self.text_embeddings = text_embeddings
        self.text_input_dim = text_input_dim
        self.allow_missing_text_embeddings = allow_missing_text_embeddings

        self._log_coverage()

    def _log_coverage(self) -> None:
        """Summarize text-embedding coverage and zero-tag rates on init."""
        unique_values: set[str] = set()
        zero_tag_pano = 0
        zero_tag_osm = 0
        for pair in self.pairs:
            pano_keys = [k for k in pair.pano_tags if k in TAG_KEY_TO_IDX]
            osm_keys = [k for k in pair.osm_tags if k in TAG_KEY_TO_IDX]
            if not pano_keys:
                zero_tag_pano += 1
            if not osm_keys:
                zero_tag_osm += 1
            for k in pano_keys:
                unique_values.add(pair.pano_tags[k])
            for k in osm_keys:
                unique_values.add(pair.osm_tags[k])

        missing = [v for v in unique_values if v not in self.text_embeddings]
        total = len(unique_values)
        pct = 100.0 * len(missing) / total if total else 0.0
        print(
            f"LandmarkCorrespondenceDataset: {len(self.pairs)} pairs, "
            f"{total} unique text values, {len(missing)} missing "
            f"from embeddings ({pct:.2f}%)."
        )
        if zero_tag_pano or zero_tag_osm:
            print(
                f"  Zero-tag pairs: pano={zero_tag_pano}, osm={zero_tag_osm} "
                f"(all tags filtered by TAG_KEY_TO_IDX whitelist)."
            )
        if missing and not self.allow_missing_text_embeddings:
            sample = sorted(missing)[:5]
            raise KeyError(
                f"{len(missing)} of {total} unique text values missing from "
                f"text_embeddings ({pct:.2f}%). Examples: {sample}. "
                f"Pass allow_missing_text_embeddings=True to fall back to "
                f"zero vectors."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pano_encoded = encode_tag_bundle(
            pair.pano_tags, self.text_embeddings, self.text_input_dim,
            allow_missing_text_embeddings=self.allow_missing_text_embeddings,
        )
        osm_encoded = encode_tag_bundle(
            pair.osm_tags, self.text_embeddings, self.text_input_dim,
            allow_missing_text_embeddings=self.allow_missing_text_embeddings,
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
