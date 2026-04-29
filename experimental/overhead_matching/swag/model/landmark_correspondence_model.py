"""Landmark correspondence classifier: text-only twin encoder.

Architecture:
    Pano tags → TagBundleEncoder → pano_repr (128)
                                               \\
                 [pano; osm; pano*osm; cross_feats(4)] → MLP → P(match)
                                               /
    OSM tags  → TagBundleEncoder → osm_repr  (128)

Cross features (4):
  - Text cosine similarity: max, mean, name-specific (3)
  - Housenumber range overlap (1)

Tag values are encoded as text via pre-computed embeddings. Unknown values
are rejected at dataset construction time unless the dataset is explicitly
constructed with `allow_missing_text_embeddings=True`.
"""

from dataclasses import dataclass, field

import common.torch.load_torch_deps  # noqa: F401

import torch
import torch.nn as nn

from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    _TAGS_TO_KEEP,
)

TAG_KEY_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_TAGS_TO_KEEP)}
NUM_TAG_KEYS = len(_TAGS_TO_KEEP)
NUM_CROSS_FEATURES = 4


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

@dataclass
class TagBundleEncoderConfig:
    key_dim: int = 32
    text_input_dim: int = 768
    text_proj_dim: int = 128
    per_tag_dim: int = 64

    @property
    def repr_dim(self) -> int:
        return 2 * self.per_tag_dim


@dataclass
class CorrespondenceClassifierConfig:
    encoder: TagBundleEncoderConfig = field(
        default_factory=TagBundleEncoderConfig
    )
    mlp_hidden_dim: int = 128
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Model modules
# ---------------------------------------------------------------------------

class TagBundleEncoder(nn.Module):
    """Encode a bundle of key=value tags into a fixed-size representation."""

    def __init__(self, config: TagBundleEncoderConfig):
        super().__init__()
        self.config = config
        self.repr_dim = config.repr_dim
        self.key_embedding = nn.Embedding(NUM_TAG_KEYS, config.key_dim)
        self.text_projection = nn.Linear(config.text_input_dim, config.text_proj_dim)
        self.per_tag_mlp = nn.Sequential(
            nn.Linear(config.key_dim + config.text_proj_dim, config.per_tag_dim),
            nn.ReLU(),
            nn.Linear(config.per_tag_dim, config.per_tag_dim),
        )

    def forward(
        self,
        key_indices: torch.Tensor,     # (B, T) int
        text_embeddings: torch.Tensor,  # (B, T, text_input_dim) float
        tag_mask: torch.Tensor,         # (B, T) bool: True for real tags
    ) -> torch.Tensor:
        """Returns: (B, repr_dim) tensor."""
        key_emb = self.key_embedding(key_indices)       # (B, T, key_dim)
        txt_emb = self.text_projection(text_embeddings)  # (B, T, text_proj_dim)

        tag_repr = self.per_tag_mlp(
            torch.cat([key_emb, txt_emb], dim=-1)
        )  # (B, T, per_tag_dim)

        mask_f = tag_mask.unsqueeze(-1).float()
        tag_repr_masked = tag_repr * mask_f

        tag_count = mask_f.sum(dim=1).clamp(min=1)
        mean_pool = tag_repr_masked.sum(dim=1) / tag_count

        tag_repr_for_max = tag_repr_masked + (1 - mask_f) * (-1e9)
        max_pool = tag_repr_for_max.max(dim=1).values

        # Samples with zero real tags: emit zeros from max-pool too (symmetric
        # with mean-pool, which is already zero via clamp(min=1)). Prevents
        # -1e9 from propagating into the classifier MLP when a landmark has
        # no tags matching _TAGS_TO_KEEP.
        has_tags = tag_mask.any(dim=1).unsqueeze(-1).float()
        max_pool = max_pool * has_tags

        return torch.cat([mean_pool, max_pool], dim=-1)


class CorrespondenceClassifier(nn.Module):
    """Twin encoder + cross-pair features + MLP classifier."""

    def __init__(self, config: CorrespondenceClassifierConfig):
        super().__init__()
        self.config = config
        self.encoder = TagBundleEncoder(config.encoder)

        repr_dim = config.encoder.repr_dim
        mlp_input_dim = repr_dim * 3 + NUM_CROSS_FEATURES

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
        pano_text_embeddings: torch.Tensor,
        pano_tag_mask: torch.Tensor,
        osm_key_indices: torch.Tensor,
        osm_text_embeddings: torch.Tensor,
        osm_tag_mask: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns: (B, 1) logits."""
        pano_repr = self.encoder(
            pano_key_indices, pano_text_embeddings, pano_tag_mask,
        )
        osm_repr = self.encoder(
            osm_key_indices, osm_text_embeddings, osm_tag_mask,
        )
        return self.classify_from_reprs(pano_repr, osm_repr, cross_features)

    def classify_from_reprs(
        self,
        pano_repr: torch.Tensor,
        osm_repr: torch.Tensor,
        cross_features: torch.Tensor,
    ) -> torch.Tensor:
        """Run just the MLP classifier on pre-computed encoder representations."""
        combined = torch.cat([
            pano_repr,
            osm_repr,
            pano_repr * osm_repr,
            cross_features,
        ], dim=-1)
        return self.classifier(combined)
