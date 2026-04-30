
import os
from pathlib import Path
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision
from dataclasses import dataclass
from typing import Union
from enum import StrEnum, auto
import msgspec
from experimental.overhead_matching.swag.model.swag_config_types import ExtractorDataRequirement


class BackboneType(StrEnum):
    VGG16 = auto()
    DINOV2 = auto()


class VGGConfig(msgspec.Struct, tag=True, tag_field='kind'):
    type: BackboneType = BackboneType.VGG16


class DinoConfig(msgspec.Struct, tag=True, tag_field='kind'):
    project: int | None
    model: str = "dinov2_vitb14"
    type: BackboneType = BackboneType.DINOV2


BackboneConfig = Union[VGGConfig, DinoConfig]


class AggregationType(StrEnum):
    SAFA = auto()
    MEAN_POOL_MLP = auto()
    CLS_CROSS_ATTENTION = auto()
    SIGMOID_CROSS_ATTENTION = auto()
    SAFA_BIASED_CROSS_ATTENTION = auto()


class WagPatchEmbeddingConfig(msgspec.Struct, tag=True, tag_field='kind'):
    patch_dims: tuple[int, int]
    num_aggregation_heads: int
    backbone_config: BackboneConfig
    aggregation_type: AggregationType = AggregationType.SAFA
    cls_cross_attention_d_k: int = 64
    safa_init_model_path: str | None = None


class DinoFeatureExtractor(torch.nn.Module):
    def __init__(self, model_str: str, project: int | None):
        super().__init__()
        # Model strings are like "dinov2_vitb14" or "dinov3_vitb16";
        # the repo name is the prefix before the first underscore.
        repo_name = f"facebookresearch/{model_str.split('_')[0]}"
        hub_dir = torch.hub.get_dir()
        checkpoint_dir = os.path.join(hub_dir, "checkpoints")
        # Find a cached checkpoint matching this model string to avoid downloading
        cached_checkpoint = None
        if os.path.isdir(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.startswith(model_str) and f.endswith(".pth"):
                    cached_checkpoint = os.path.join(checkpoint_dir, f)
                    break
        if cached_checkpoint:
            print(f"Loading {model_str} weights from cached checkpoint: {cached_checkpoint}")
            self.dino = torch.hub.load(repo_name, model_str, pretrained=False)
            state_dict = torch.load(cached_checkpoint, map_location="cpu", weights_only=True)
            self.dino.load_state_dict(state_dict, strict=True)
        else:
            print(f"No cached checkpoint found for '{model_str}' in {checkpoint_dir}")
            if os.path.isdir(checkpoint_dir):
                print(f"  Files in checkpoint dir: {os.listdir(checkpoint_dir)}")
            else:
                print(f"  Checkpoint directory does not exist: {checkpoint_dir}")
            print(f"  Falling back to torch.hub.load (requires network access)")
            self.dino = torch.hub.load(repo_name, model_str)
        self.dino.eval()
        for param in self.dino.parameters():
            param.requires_grad = False

        if project is None:
            self.project = torch.nn.Identity()
        else:
            self.project = torch.nn.Conv2d(
                in_channels=self.dino.num_features,
                out_channels=project,
                kernel_size=1)

    def forward(self, x):
        # Patch tokens are batch x n_tokens x embedding_dim
        # the tokens go from left to right, then top to bottom
        batch_size, n_channels, img_height, img_width = x.shape
        patch_height, patch_width = self.dino.patch_embed.patch_size
        num_patch_cols = int((img_width + patch_width / 2) // patch_width)
        num_patch_rows = int((img_height + patch_height / 2) // patch_height)

        x = torchvision.transforms.functional.normalize(
                x,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

        with torch.no_grad():
            patch_tokens = self.dino.forward_features(x)["x_norm_patchtokens"]
        # Swap the embedding dim and the token dim and reshape to have the same
        # aspect ratio as the original image
        out = patch_tokens.transpose(-1, -2).reshape(
                batch_size, -1, num_patch_rows, num_patch_cols)
        out = self.project(out)
        return out


def load_backbone(backbone_config: BackboneConfig):
    match backbone_config:
        case VGGConfig():
            model = torchvision.models.vgg16()
        case DinoConfig(project, model_str):
            model = DinoFeatureExtractor(model_str, project)
    model.backbone_type = backbone_config.type
    return model


def vgg16_feature_extraction(backbone, x):
    features = backbone.features(x)
    return features


def dino_feature_extraction(backbone, x):
    return backbone(x)


def compute_safa_input_dims(backbone: torch.nn.Module, patch_dims: tuple[int, int]):
    with torch.no_grad():
        test_input = torch.empty(1, 3, *patch_dims)
        result = extract_features(backbone, test_input)
        return result.shape[-3], result.shape[-2] * result.shape[-1]


def extract_features(backbone, x):
    if backbone.backbone_type == BackboneType.VGG16:
        return vgg16_feature_extraction(backbone, x)
    elif backbone.backbone_type == BackboneType.DINOV2:
        return dino_feature_extraction(backbone, x)


class ClsCrossAttention(torch.nn.Module):
    def __init__(self, n_channels, n_spatial, n_heads, d_k, use_sigmoid=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.use_sigmoid = use_sigmoid
        self.cls = torch.nn.Parameter(torch.randn(n_heads, d_k) * 0.02)
        self.W_k = torch.nn.Parameter(torch.randn(n_heads, n_channels, d_k) * 0.02)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 1, n_spatial, d_k) * 0.02)

    def forward(self, features):
        B, C, H, W = features.shape
        tokens = features.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        K = torch.einsum('bnc,hcd->bhnd', tokens, self.W_k)      # (B, heads, N, d_k)
        K = K + self.pos_embed
        logits = torch.einsum('hd,bhnd->bhn', self.cls, K) / (self.d_k ** 0.5)
        if self.use_sigmoid:
            attn = torch.sigmoid(logits)                           # (B, heads, N)
        else:
            attn = torch.softmax(logits, dim=-1)                   # (B, heads, N)
        out = torch.einsum('bhn,bnc->bhc', attn, tokens)          # (B, heads, C)
        return out.reshape(B, -1)                                  # (B, heads * C)


class SafaBiasedCrossAttention(torch.nn.Module):
    """Hybrid SAFA + transformer cross-attention.

    Uses SAFA's global spatial reasoning as an additive bias to transformer
    attention logits, with a learnable gate to balance the two paths.
    Sigmoid activation (not softmax) preserves per-position independence.
    """
    def __init__(self, n_channels, n_spatial, n_heads, d_k):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.n_spatial = n_spatial

        # SAFA path: collapsed two-layer affine (N -> N)
        self.safa_W = torch.nn.Parameter(
            torch.randn(n_heads, n_spatial, n_spatial) * 0.005)
        self.safa_b = torch.nn.Parameter(
            torch.ones(1, n_heads, n_spatial) * 0.1)

        # Transformer path: CLS cross-attention
        self.cls = torch.nn.Parameter(torch.randn(n_heads, d_k) * 0.02)
        self.W_k = torch.nn.Parameter(torch.randn(n_heads, n_channels, d_k) * 0.02)
        self.pos_embed = torch.nn.Parameter(torch.randn(1, 1, n_spatial, d_k) * 0.02)

        # Learnable gate, initialized to favor SAFA: sigmoid(2.0) ≈ 0.88
        self.safa_gate = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, features, extra_tokens=None):
        B, C, H, W = features.shape
        N_img = H * W
        tokens = features.reshape(B, C, N_img).permute(0, 2, 1)  # (B, N, C)

        # SAFA path: channel-max → learned affine
        channel_max = features.max(dim=1).values.reshape(B, 1, N_img)  # (B, 1, N)
        safa_logits = torch.einsum('bdi,dij->bdj', channel_max, self.safa_W) + self.safa_b
        # safa_logits: (B, n_heads, N_img)

        if extra_tokens is not None:
            tokens = torch.cat([tokens, extra_tokens], dim=1)  # (B, N_img+M, C)
            # SAFA bias is 0 for extra tokens (no spatial prior)
            safa_logits = F.pad(safa_logits, (0, extra_tokens.shape[1]))

        # Transformer path: CLS cross-attention
        K = torch.einsum('bnc,hcd->bhnd', tokens, self.W_k)  # (B, heads, N, d_k)
        # Position embedding only applies to image tokens
        K[:, :, :N_img, :] = K[:, :, :N_img, :] + self.pos_embed
        xattn_logits = torch.einsum('hd,bhnd->bhn', self.cls, K) / (self.d_k ** 0.5)

        # Combine with learnable gate
        gate = torch.sigmoid(self.safa_gate)
        logits = gate * safa_logits + (1 - gate) * xattn_logits

        # Sigmoid activation (per-position independence, like SAFA)
        attn = torch.sigmoid(logits)
        out = torch.einsum('bhn,bnc->bhc', attn, tokens)  # (B, heads, C)
        return out.reshape(B, -1)  # (B, heads * C)


class WagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: WagPatchEmbeddingConfig):
        super().__init__()
        self._backbone = load_backbone(config.backbone_config)
        self._patch_dims = config.patch_dims
        self._aggregation_type = config.aggregation_type

        n_channels, n_spatial = compute_safa_input_dims(self._backbone, config.patch_dims)
        n_heads = config.num_aggregation_heads
        self._output_dim = n_channels * n_heads

        match self._aggregation_type:
            case AggregationType.SAFA:
                input_safa_dim = n_spatial
                safa_dims = [input_safa_dim // 2, input_safa_dim]
                self._safa_params = []
                for layer_idx, output_safa_dim in enumerate(safa_dims):
                    safa_weight = torch.nn.Parameter(
                            torch.randn((n_heads, input_safa_dim, output_safa_dim)) * 0.005)
                    safa_bias = torch.nn.Parameter(torch.ones((1, n_heads, output_safa_dim)) * 0.1)
                    self.register_parameter(f"safa_{layer_idx}_weight", safa_weight)
                    self.register_parameter(f"safa_{layer_idx}_bias", safa_bias)
                    self._safa_params.append((safa_weight, safa_bias))
                    input_safa_dim = output_safa_dim

            case AggregationType.MEAN_POOL_MLP:
                self._pool_mlp = torch.nn.Sequential(
                    torch.nn.Linear(n_channels, n_channels * 2),
                    torch.nn.GELU(),
                    torch.nn.Linear(n_channels * 2, self._output_dim),
                )

            case AggregationType.CLS_CROSS_ATTENTION:
                self._cls_cross_attn = ClsCrossAttention(
                    n_channels, n_spatial, n_heads, config.cls_cross_attention_d_k)

            case AggregationType.SIGMOID_CROSS_ATTENTION:
                self._cls_cross_attn = ClsCrossAttention(
                    n_channels, n_spatial, n_heads, config.cls_cross_attention_d_k,
                    use_sigmoid=True)

            case AggregationType.SAFA_BIASED_CROSS_ATTENTION:
                self._safa_biased_cross_attn = SafaBiasedCrossAttention(
                    n_channels, n_spatial, n_heads, config.cls_cross_attention_d_k)

        if config.safa_init_model_path is not None:
            if self._aggregation_type != AggregationType.SAFA_BIASED_CROSS_ATTENTION:
                raise ValueError(
                    "safa_init_model_path is only supported with SAFA_BIASED_CROSS_ATTENTION")
            raw_weights = torch.load(
                Path(config.safa_init_model_path) / "model_weights.pt",
                map_location="cpu", weights_only=True)
            # Strip _orig_mod. prefix from torch.compile'd checkpoints
            safa_weights = {k.removeprefix('_orig_mod.'): v for k, v in raw_weights.items()}

            # Transfer the learned projection layer
            self._backbone.project.weight.data.copy_(safa_weights['_backbone.project.weight'])
            self._backbone.project.bias.data.copy_(safa_weights['_backbone.project.bias'])

            # Collapse two-layer SAFA (no nonlinearity!) into single matrix
            # out = (x @ W0 + b0) @ W1 + b1 = x @ (W0 @ W1) + (b0 @ W1 + b1)
            W0 = safa_weights['safa_0_weight']   # (heads, N, N/2)
            b0 = safa_weights['safa_0_bias']     # (1, heads, N/2)
            W1 = safa_weights['safa_1_weight']   # (heads, N/2, N)
            b1 = safa_weights['safa_1_bias']     # (1, heads, N)

            collapsed_W = W0 @ W1                                       # (heads, N, N)
            collapsed_b = torch.einsum('bdi,dij->bdj', b0, W1) + b1    # (1, heads, N)

            self._safa_biased_cross_attn.safa_W.data.copy_(collapsed_W)
            self._safa_biased_cross_attn.safa_b.data.copy_(collapsed_b)

            print(f"Initialized SAFA path from {config.safa_init_model_path}")

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def num_embeddings(self):
        return 1

    @property
    def patch_dims(self):
        return self._patch_dims

    def cache_info(self):
        return None

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.IMAGES]

    def safa(self, x):
        batch_size = x.shape[0]
        (out, _) = torch.max(x, dim=-3)
        out = torch.reshape(out, (batch_size, 1, -1))

        for (weight, bias) in self._safa_params:
            out = torch.einsum('bdi, dij->bdj', out, weight) + bias
        return out

    def model_input_from_batch(self, batch_item):
        if self._patch_dims[0] != self._patch_dims[1]:
            return batch_item.panorama
        else:
            return batch_item.satellite

    def forward(self, x, landmark_dropout_scheduler=None):
        assert landmark_dropout_scheduler is None, "WAG does not support landmark dropout"
        features = extract_features(self._backbone, x)
        batch_size, num_channels, _, _ = features.shape

        match self._aggregation_type:
            case AggregationType.SAFA:
                attention = self.safa(features)
                vectorized_features = torch.reshape(features, (batch_size, num_channels, -1))
                per_head_embedding = torch.einsum('bci,bdi->bdc', vectorized_features, attention)
                embedding = torch.reshape(per_head_embedding, (batch_size, -1))

            case AggregationType.MEAN_POOL_MLP:
                pooled = features.mean(dim=(-2, -1))
                embedding = self._pool_mlp(pooled)

            case AggregationType.CLS_CROSS_ATTENTION:
                embedding = self._cls_cross_attn(features)

            case AggregationType.SIGMOID_CROSS_ATTENTION:
                embedding = self._cls_cross_attn(features)

            case AggregationType.SAFA_BIASED_CROSS_ATTENTION:
                embedding = self._safa_biased_cross_attn(features)

        return F.normalize(embedding, dim=-1).unsqueeze(1), {}
