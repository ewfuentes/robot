"""Landmark correspondence classifier: twin encoder + interaction classifier.

Architecture:
    Pano tags → TagBundleEncoder → pano_repr (D)
                                                \
                                  [pano; osm; pano*osm; cross_feats] → MLP → P(match)
                                                /
    OSM tags  → TagBundleEncoder → osm_repr  (D)

TagBundleEncoder processes each key=value tag in a bundle:
  1. Key embedding (learned, 112 keys × key_dim)
  2. Value encoding by type:
     - Boolean: single learned embedding (true/false/unknown → 8-dim)
     - Numeric: [log(1+|x|), x/1000, 1.0, is_lower_bound] → linear → 16-dim
     - Housenumber: [log(1+lo), log(1+hi), log(1+hi-lo), mean/2000] → linear → 16-dim
     - Text: pre-computed embedding → linear projection → text_proj_dim
  3. Per-tag MLP: concat(key_emb, value_emb) → 64-dim
  4. Pooling: mean + max → 128-dim representation
"""

import enum
import math
from dataclasses import dataclass, field

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    _TAGS_TO_KEEP,
)


# ---------------------------------------------------------------------------
# Enums for value types
# ---------------------------------------------------------------------------

class ValueType(enum.IntEnum):
    BOOLEAN = 0
    NUMERIC = 1
    HOUSENUMBER = 2
    TEXT = 3


class BooleanValue(enum.IntEnum):
    TRUE = 0
    FALSE = 1
    UNKNOWN = 2

# ---------------------------------------------------------------------------
# Tag key classification
# ---------------------------------------------------------------------------

BOOLEAN_KEYS = frozenset([
    "outdoor_seating", "drive_through", "embankment", "handrail",
    "fire_escape", "fenced", "oneway", "covered",
])

NUMERIC_KEYS = frozenset([
    "building:levels", "levels", "lanes", "maxspeed",
    "maxheight", "heritage",
])

HOUSENUMBER_KEY = "addr:housenumber"

# Everything else is TEXT
# Build key → index mapping from _TAGS_TO_KEEP
TAG_KEY_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_TAGS_TO_KEEP)}
NUM_TAG_KEYS = len(_TAGS_TO_KEEP)


def key_type(key: str) -> ValueType:
    """Return the encoding type for a tag key."""
    if key == HOUSENUMBER_KEY:
        return ValueType.HOUSENUMBER
    if key in BOOLEAN_KEYS:
        return ValueType.BOOLEAN
    if key in NUMERIC_KEYS:
        return ValueType.NUMERIC
    return ValueType.TEXT


# ---------------------------------------------------------------------------
# Value parsing utilities
# ---------------------------------------------------------------------------

def parse_boolean(value: str) -> float:
    """Parse boolean tag value to float. yes→1.0, no→0.0, unknown→0.5."""
    v = value.strip().lower()
    if v in ("yes", "true", "1"):
        return 1.0
    if v in ("no", "false", "0", "-1"):
        return 0.0
    return 0.5


_LEVEL_KEYS = frozenset(["building:levels", "levels"])


def parse_numeric(value: str, key: str | None = None) -> tuple[float, bool]:
    """Parse numeric tag value to (float, is_lower_bound).

    The is_lower_bound flag is True when the raw value had a '+' suffix
    (e.g. "20+" means "20 or more").
    """
    v = value.strip().lower()
    # Strip common suffixes
    for suffix in (" mph", " km/h", " m", " ft"):
        if v.endswith(suffix):
            v = v[:-len(suffix)]
    # Detect and strip "+" (lower-bound indicator, e.g. "20+")
    is_lower_bound = v.endswith("+")
    if is_lower_bound:
        v = v[:-1].strip()
    # Key-aware special text values: "high"→20, "multi"→5 only for level keys
    special: dict[str, float] = {"yes": 1.0, "no": 0.0}
    if key is not None and key in _LEVEL_KEYS:
        special["high"] = 20.0
        special["multi"] = 5.0
    if v in special:
        return (special[v], is_lower_bound)
    try:
        return (float(v), is_lower_bound)
    except ValueError:
        return (float("nan"), False)


def parse_maxheight(value: str) -> float:
    """Parse maxheight, handling feet'inches\" format."""
    v = value.strip().lower()
    if v in ("default", "none", "no_sign", "below_default", "no"):
        return float("nan")
    # Try feet'inches" format
    if "'" in v:
        parts = v.replace('"', '').split("'")
        try:
            feet = float(parts[0])
            inches = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
            return feet + inches / 12.0
        except ValueError:
            return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def parse_housenumber(value: str) -> tuple[float, float]:
    """Parse addr:housenumber to (low, high) range.

    "665-667" → (665, 667)
    "1858" → (1858, 1858)
    Returns (nan, nan) on failure.
    """
    v = value.strip()
    if "-" in v:
        parts = v.split("-", 1)
        try:
            lo = float(parts[0].strip())
            hi = float(parts[1].strip())
            return (lo, hi)
        except ValueError:
            pass
    try:
        n = float(v)
        return (n, n)
    except ValueError:
        return (float("nan"), float("nan"))


def encode_numeric_value(x: float, is_lower_bound: bool = False) -> list[float]:
    """Encode numeric value as [log(1+|x|), x/1000, 1.0, is_lower_bound].

    The third element (1.0) is a presence flag: it distinguishes a valid zero
    encoding [0, 0, 1.0, ...] from the NaN encoding [0, 0, 0, 0].
    """
    if math.isnan(x):
        return [0.0, 0.0, 0.0, 0.0]  # Will use NaN embedding instead
    return [math.log1p(abs(x)), x / 1000.0, 1.0, 1.0 if is_lower_bound else 0.0]


def encode_housenumber_value(lo: float, hi: float) -> list[float]:
    """Encode housenumber range as [log(1+lo), log(1+hi), log(1+hi-lo), mean/2000]."""
    if math.isnan(lo) or math.isnan(hi):
        return [0.0, 0.0, 0.0, 0.0]  # Will use NaN embedding instead
    # Ensure lo <= hi and both non-negative for log1p
    lo, hi = max(lo, 0.0), max(hi, 0.0)
    if lo > hi:
        lo, hi = hi, lo
    return [math.log1p(lo), math.log1p(hi), math.log1p(hi - lo), (lo + hi) / 4000.0]


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class TagBundleEncoderConfig:
    key_dim: int = 32
    boolean_dim: int = 8
    numeric_dim: int = 16
    housenumber_dim: int = 16
    text_input_dim: int = 768
    text_proj_dim: int = 128
    per_tag_dim: int = 64

    @property
    def repr_dim(self) -> int:
        """Output dimension: mean + max pool of per_tag_dim."""
        return 2 * self.per_tag_dim


@dataclass
class CorrespondenceClassifierConfig:
    encoder: TagBundleEncoderConfig = field(default_factory=TagBundleEncoderConfig)
    mlp_hidden_dim: int = 128
    dropout: float = 0.1
    num_cross_features: int = 13  # Jaccard(1) + shared_keys(1) + exact_matches(1) + text_sim(3) + numeric(6) + housenumber(1)


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------

class TagBundleEncoder(nn.Module):
    """Encode a bundle of key=value tags into a fixed-size representation.

    Processes each tag independently, then pools over all tags.
    """

    def __init__(self, config: TagBundleEncoderConfig):
        super().__init__()
        self.config = config

        # Key embeddings
        self.key_embedding = nn.Embedding(NUM_TAG_KEYS, config.key_dim)

        # Boolean encoder: 3 embeddings (true, false, unknown)
        self.boolean_embedding = nn.Embedding(3, config.boolean_dim)

        # Numeric encoder: linear 4 → numeric_dim
        self.numeric_linear = nn.Linear(4, config.numeric_dim)
        self.numeric_nan_embedding = nn.Parameter(torch.randn(config.numeric_dim) * 0.01)

        # Housenumber encoder: linear 4 → housenumber_dim
        self.housenumber_linear = nn.Linear(4, config.housenumber_dim)
        self.housenumber_nan_embedding = nn.Parameter(
            torch.randn(config.housenumber_dim) * 0.01
        )

        # Text encoder: linear projection from pre-computed embeddings
        self.text_projection = nn.Linear(config.text_input_dim, config.text_proj_dim)

        # Absent value embeddings (one per type)
        self.absent_embedding = nn.Parameter(torch.randn(config.text_proj_dim) * 0.01)

        # Per-tag MLP: concat(key_emb, value_emb) → per_tag_dim
        # Value dim varies by type; use the max (text_proj_dim) and pad smaller ones
        max_value_dim = config.text_proj_dim
        self.per_tag_mlp = nn.Sequential(
            nn.Linear(config.key_dim + max_value_dim, config.per_tag_dim),
            nn.ReLU(),
            nn.Linear(config.per_tag_dim, config.per_tag_dim),
        )

        # Value dim adapters: project each type's value to max_value_dim
        self.boolean_proj = nn.Linear(config.boolean_dim, max_value_dim)
        self.numeric_proj = nn.Linear(config.numeric_dim, max_value_dim)
        self.housenumber_proj = nn.Linear(config.housenumber_dim, max_value_dim)
        # text_projection already outputs text_proj_dim = max_value_dim

    def forward(
        self,
        key_indices: torch.Tensor,      # (B, max_tags) int
        value_type: torch.Tensor,        # (B, max_tags) int: 0=bool, 1=numeric, 2=housenum, 3=text
        boolean_values: torch.Tensor,    # (B, max_tags) int: 0=true, 1=false, 2=unknown
        numeric_values: torch.Tensor,    # (B, max_tags, 4) float
        numeric_nan_mask: torch.Tensor,  # (B, max_tags) bool: True if NaN
        housenumber_values: torch.Tensor,  # (B, max_tags, 4) float
        housenumber_nan_mask: torch.Tensor,  # (B, max_tags) bool: True if NaN
        text_embeddings: torch.Tensor,   # (B, max_tags, text_input_dim) float
        tag_mask: torch.Tensor,          # (B, max_tags) bool: True for real tags
    ) -> torch.Tensor:
        """Encode tag bundles to fixed-size representations.

        Returns: (B, repr_dim) tensor
        """
        B, T = key_indices.shape

        # 1. Key embeddings: (B, T, key_dim)
        key_emb = self.key_embedding(key_indices)

        # 2. Value embeddings by type
        # Boolean: (B, T, boolean_dim) → (B, T, max_value_dim)
        bool_emb = self.boolean_proj(self.boolean_embedding(boolean_values))

        # Numeric: (B, T, 4) → (B, T, numeric_dim)
        num_emb = self.numeric_linear(numeric_values)
        # Replace NaN entries with learned NaN embedding
        nan_expand = self.numeric_nan_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        num_emb = torch.where(numeric_nan_mask.unsqueeze(-1), nan_expand, num_emb)
        num_emb = self.numeric_proj(num_emb)  # → (B, T, max_value_dim)

        # Housenumber: (B, T, 4) → (B, T, housenumber_dim)
        hn_emb = self.housenumber_linear(housenumber_values)
        hn_nan_expand = self.housenumber_nan_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        hn_emb = torch.where(housenumber_nan_mask.unsqueeze(-1), hn_nan_expand, hn_emb)
        hn_emb = self.housenumber_proj(hn_emb)  # → (B, T, max_value_dim)

        # Text: (B, T, text_input_dim) → (B, T, text_proj_dim = max_value_dim)
        txt_emb = self.text_projection(text_embeddings)

        # Select value embedding based on type
        # type: 0=bool, 1=numeric, 2=housenumber, 3=text
        type_expanded = value_type.unsqueeze(-1)  # (B, T, 1)
        max_vd = bool_emb.shape[-1]
        value_emb = torch.zeros(B, T, max_vd, device=key_emb.device, dtype=key_emb.dtype)
        value_emb = torch.where(type_expanded == 0, bool_emb, value_emb)
        value_emb = torch.where(type_expanded == 1, num_emb, value_emb)
        value_emb = torch.where(type_expanded == 2, hn_emb, value_emb)
        value_emb = torch.where(type_expanded == 3, txt_emb, value_emb)

        # 3. Per-tag representation: concat(key, value) → MLP
        tag_repr = self.per_tag_mlp(torch.cat([key_emb, value_emb], dim=-1))  # (B, T, per_tag_dim)

        # 4. Masked pooling: mean + max → repr_dim
        # Mask out padding
        mask_f = tag_mask.unsqueeze(-1).float()  # (B, T, 1)
        tag_repr_masked = tag_repr * mask_f

        # Mean pool
        tag_count = mask_f.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pool = tag_repr_masked.sum(dim=1) / tag_count  # (B, per_tag_dim)

        # Max pool (set padding to -inf)
        tag_repr_for_max = tag_repr_masked + (1 - mask_f) * (-1e9)
        max_pool = tag_repr_for_max.max(dim=1).values  # (B, per_tag_dim)

        # Concat mean and max → repr_dim
        return torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2 * per_tag_dim = repr_dim)


class CorrespondenceClassifier(nn.Module):
    """Twin encoder + cross-pair features + MLP classifier.

    Takes two tag bundles (pano, OSM) and predicts P(same physical object).
    """

    def __init__(self, config: CorrespondenceClassifierConfig):
        super().__init__()
        self.config = config

        # Shared twin encoder
        self.encoder = TagBundleEncoder(config.encoder)

        repr_dim = config.encoder.repr_dim  # 128
        # Input to MLP: [pano; osm; pano*osm; cross_features]
        mlp_input_dim = repr_dim * 3 + config.num_cross_features

        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, config.mlp_hidden_dim),
            nn.BatchNorm1d(config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, 1),
        )

    def forward(
        self,
        pano_key_indices: torch.Tensor,
        pano_value_type: torch.Tensor,
        pano_boolean_values: torch.Tensor,
        pano_numeric_values: torch.Tensor,
        pano_numeric_nan_mask: torch.Tensor,
        pano_housenumber_values: torch.Tensor,
        pano_housenumber_nan_mask: torch.Tensor,
        pano_text_embeddings: torch.Tensor,
        pano_tag_mask: torch.Tensor,
        osm_key_indices: torch.Tensor,
        osm_value_type: torch.Tensor,
        osm_boolean_values: torch.Tensor,
        osm_numeric_values: torch.Tensor,
        osm_numeric_nan_mask: torch.Tensor,
        osm_housenumber_values: torch.Tensor,
        osm_housenumber_nan_mask: torch.Tensor,
        osm_text_embeddings: torch.Tensor,
        osm_tag_mask: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Returns: (B, 1) logits
        """
        pano_repr = self.encoder(
            pano_key_indices, pano_value_type, pano_boolean_values,
            pano_numeric_values, pano_numeric_nan_mask,
            pano_housenumber_values, pano_housenumber_nan_mask,
            pano_text_embeddings, pano_tag_mask,
        )
        osm_repr = self.encoder(
            osm_key_indices, osm_value_type, osm_boolean_values,
            osm_numeric_values, osm_numeric_nan_mask,
            osm_housenumber_values, osm_housenumber_nan_mask,
            osm_text_embeddings, osm_tag_mask,
        )

        # Elementwise product captures multiplicative interaction between pano
        # and osm representations. Motivated by ESIM (Chen et al. 2016,
        # arXiv:1609.06038) and InferSent (Conneau et al. 2017, arXiv:1705.02364).
        combined = torch.cat([
            pano_repr,
            osm_repr,
            pano_repr * osm_repr,
            cross_features,
        ], dim=-1)

        return self.classifier(combined)
