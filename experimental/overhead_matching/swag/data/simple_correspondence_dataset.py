"""Simplified dataset for landmark correspondence classifier.

Compared to landmark_correspondence_dataset.py, this removes all value type
routing — every tag value is encoded as text via pre-computed embeddings.

Cross features are reduced to 4: text cosine sim (max, mean, name) + housenumber overlap.
"""

import math
from dataclasses import dataclass
from typing import List

import common.torch.load_torch_deps  # noqa: F401

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from experimental.overhead_matching.swag.model.simple_correspondence_model import (
    TAG_KEY_TO_IDX,
    NUM_CROSS_FEATURES,
    parse_housenumber,
)


HOUSENUMBER_KEY = "addr:housenumber"


def load_text_embeddings(path) -> dict[str, torch.Tensor]:
    """Load pre-computed text embeddings from pickle file."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


def encode_tag_bundle(
    tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor],
    text_input_dim: int = 768,
) -> dict:
    """Encode a tag bundle into tensors. All values encoded as text.

    Returns dict with key_indices, text_embeddings, (lists to be padded later).
    """
    key_indices = []
    text_embs = []
    zero_text = torch.zeros(text_input_dim)

    for k, v in tags.items():
        if k not in TAG_KEY_TO_IDX:
            continue
        key_indices.append(TAG_KEY_TO_IDX[k])
        text_embs.append(text_embeddings.get(v, zero_text))

    return {
        "key_indices": key_indices,
        "text_embeddings": text_embs,
    }


def compute_cross_features(
    pano_tags: dict[str, str],
    osm_tags: dict[str, str],
    text_embeddings: dict[str, torch.Tensor] | None = None,
) -> list[float]:
    """Compute 4 cross-pair features: 3 text cosine sims + housenumber overlap."""
    features = []

    # Text cosine similarities (3 features)
    shared_keys = set(pano_tags.keys()) & set(osm_tags.keys())
    if text_embeddings is not None:
        text_sims = []
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
        features.append(sum(text_sims) / max(len(text_sims), 1) if text_sims else 0.0)
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
class SimpleCorrespondenceBatch:
    pano_key_indices: torch.Tensor       # (B, T) int
    pano_text_embeddings: torch.Tensor   # (B, T, text_dim) float
    pano_tag_mask: torch.Tensor          # (B, T) bool
    osm_key_indices: torch.Tensor        # (B, T) int
    osm_text_embeddings: torch.Tensor    # (B, T, text_dim) float
    osm_tag_mask: torch.Tensor           # (B, T) bool
    cross_features: torch.Tensor         # (B, 4) float
    labels: torch.Tensor                 # (B,) float

    def to(self, device):
        return SimpleCorrespondenceBatch(**{
            k: v.to(device) for k, v in self.__dict__.items()
        })


def _pad_side(encoded_list: List[dict], text_input_dim: int):
    """Pad a list of encoded tag bundles to uniform length."""
    max_tags = max(len(e["key_indices"]) for e in encoded_list)
    max_tags = max(max_tags, 1)  # at least 1

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


def collate_correspondence(batch: list[dict]) -> SimpleCorrespondenceBatch:
    """Collate function for DataLoader."""
    text_dim = 768  # infer from first sample
    if batch[0]["pano"]["text_embeddings"]:
        text_dim = batch[0]["pano"]["text_embeddings"][0].shape[0]

    pano_keys, pano_text, pano_mask = _pad_side(
        [b["pano"] for b in batch], text_dim
    )
    osm_keys, osm_text, osm_mask = _pad_side(
        [b["osm"] for b in batch], text_dim
    )
    cross = torch.tensor([b["cross_features"] for b in batch], dtype=torch.float32)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    return SimpleCorrespondenceBatch(
        pano_key_indices=pano_keys,
        pano_text_embeddings=pano_text,
        pano_tag_mask=pano_mask,
        osm_key_indices=osm_keys,
        osm_text_embeddings=osm_text,
        osm_tag_mask=osm_mask,
        cross_features=cross,
        labels=labels,
    )


# Re-export pair loading from the original dataset module
from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (  # noqa: E402
    CorrespondencePair,
    load_pairs_from_directory,
    scan_parse_failures as _scan_parse_failures_orig,
    PARSE_REPORT_PATH,
)


class SimpleLandmarkCorrespondenceDataset(Dataset):
    """Dataset that encodes all tag values as text."""

    def __init__(
        self,
        pairs: list[CorrespondencePair],
        text_embeddings: dict[str, torch.Tensor],
        text_input_dim: int = 768,
        include_difficulties: tuple[str, ...] = ("positive", "easy", "hard"),
    ):
        self.pairs = [p for p in pairs if p.difficulty in include_difficulties]
        self.text_embeddings = text_embeddings
        self.text_input_dim = text_input_dim

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        pano_encoded = encode_tag_bundle(
            pair.pano_tags, self.text_embeddings, self.text_input_dim,
        )
        osm_encoded = encode_tag_bundle(
            pair.osm_tags, self.text_embeddings, self.text_input_dim,
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
