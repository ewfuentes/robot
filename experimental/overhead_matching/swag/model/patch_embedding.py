
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision
from dataclasses import dataclass
from typing import Tuple, Callable
from enum import StrEnum, auto


class BackboneType(StrEnum):
    VGG16 = auto()
    DINOV2_B14 = auto()


@dataclass
class WagPatchEmbeddingConfig:
    patch_dims: Tuple[int, int]
    num_aggregation_heads: int
    backbone_type: BackboneType


def load_backbone(backbone_type: BackboneType):
    if backbone_type == BackboneType.VGG16:
        model = torchvision.models.vgg16()
    elif backbone_type == BackboneType.DINOV2_B14:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model.eval()
    model.backbone_type = backbone_type
    return model


def vgg16_feature_extraction(backbone, x):
    features = backbone.features(x)
    return features


def dino_feature_extraction(backbone, x):
    # Patch tokens are batch x n_tokens x embedding_dim
    # the tokens go from left to right, then top to bottom
    batch_size, n_channels, img_height, img_width = x.shape
    patch_height, patch_width = backbone.patch_embed.patch_size
    num_patch_cols = int((img_width + patch_width / 2) // patch_width)
    num_patch_rows = int((img_height + patch_height / 2) // patch_height)

    x = torchvision.transforms.functional.normalize(
            x,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        patch_tokens = backbone.forward_features(x)["x_norm_patchtokens"]
    # Swap the embedding dim and the token dim and reshape to have the same
    # aspect ratio as the original image
    out = patch_tokens.transpose(-1, -2).reshape(
            batch_size, -1, num_patch_rows, num_patch_cols)
    return out


def compute_safa_input_dims(backbone: torch.nn.Module, patch_dims: Tuple[int, int]):
    with torch.no_grad():
        test_input = torch.empty(1, 3, *patch_dims)
        result = extract_features(backbone, test_input)
        return result.shape[-3], result.shape[-2] * result.shape[-1]


def extract_features(backbone, x):
    if backbone.backbone_type == BackboneType.VGG16:
        return vgg16_feature_extraction(backbone, x)
    elif backbone.backbone_type == BackboneType.DINOV2_B14:
        return dino_feature_extraction(backbone, x)


class WagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: WagPatchEmbeddingConfig):
        super().__init__()
        self.backbone = load_backbone(config.backbone_type)

        n_channels, input_safa_dim = compute_safa_input_dims(self.backbone, config.patch_dims)
        safa_dims = [input_safa_dim // 2, input_safa_dim]
        n_heads = config.num_aggregation_heads
        self._output_dim = n_channels * n_heads

        self._safa_params = []
        for layer_idx, output_safa_dim in enumerate(safa_dims):
            safa_weight = torch.nn.Parameter(
                    torch.randn((n_heads, input_safa_dim, output_safa_dim)) * 0.005)
            safa_bias = torch.nn.Parameter(torch.ones((1, n_heads, output_safa_dim)) * 0.1)
            self.register_parameter(f"safa_{layer_idx}_weight", safa_weight)
            self.register_parameter(f"safa_{layer_idx}_bias", safa_bias)
            self._safa_params.append((safa_weight, safa_bias))
            input_safa_dim = output_safa_dim

    @property
    def output_dim(self):
        return self._output_dim

    def safa(self, x):
        batch_size = x.shape[0]
        (out, _) = torch.max(x, dim=-3)
        out = torch.reshape(out, (batch_size, 1, -1))

        for (weight, bias) in self._safa_params:
            out = torch.einsum('bdi, dij->bdj', out, weight) + bias
        return out

    def forward(self, x):
        features = extract_features(self.backbone, x)
        batch_size, num_channels, _, _ = features.shape
        attention = self.safa(features)
        vectorized_features = torch.reshape(features, (batch_size, num_channels, -1))
        per_head_embedding = torch.einsum('bci,bdi->bdc', vectorized_features, attention)
        embedding = torch.reshape(per_head_embedding, (batch_size, -1))
        return F.normalize(embedding)
