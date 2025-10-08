
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


class WagPatchEmbeddingConfig(msgspec.Struct, tag=True, tag_field='kind'):
    patch_dims: tuple[int, int]
    num_aggregation_heads: int
    backbone_config: BackboneConfig


class DinoFeatureExtractor(torch.nn.Module):
    def __init__(self, model_str: str, project: int | None):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", model_str)
        self.dino.eval()

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


class WagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: WagPatchEmbeddingConfig):
        super().__init__()
        self._backbone = load_backbone(config.backbone_config)
        self._patch_dims = config.patch_dims

        n_channels, input_safa_dim = compute_safa_input_dims(self._backbone, config.patch_dims)
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

    def forward(self, x):
        features = extract_features(self._backbone, x)
        batch_size, num_channels, _, _ = features.shape
        attention = self.safa(features)
        vectorized_features = torch.reshape(features, (batch_size, num_channels, -1))
        per_head_embedding = torch.einsum('bci,bdi->bdc', vectorized_features, attention)
        embedding = torch.reshape(per_head_embedding, (batch_size, -1))
        return F.normalize(embedding)
