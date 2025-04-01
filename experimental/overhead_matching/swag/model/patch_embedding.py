
import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import torchvision
from dataclasses import dataclass
from typing import Tuple


@dataclass
class WagPatchEmbeddingConfig:
    patch_dims: Tuple[int, int]
    num_aggregation_heads: int


def feature_extraction(backbone, x):
    features = backbone.features(x)
    # If the dimension size is odd, then ceil_mode=True will pad by one on the right/bottom
    max_pooled = F.max_pool2d(features, kernel_size=2, stride=2, padding=0, ceil_mode=True)
    return max_pooled

def compute_safa_input_dims(backbone: torch.nn.Module, patch_dims: Tuple[int, int]):
    with torch.no_grad():
        test_input = torch.empty(1, 3, *patch_dims)
        result = feature_extraction(backbone, test_input)
        return result.shape[-2] * result.shape[-1]


class WagPatchEmbedding(torch.nn.Module):
    def __init__(self, config: WagPatchEmbeddingConfig):
        super().__init__()
        self.backbone = torchvision.models.vgg19()

        safa_dim = compute_safa_input_dims(self.backbone, config.patch_dims)
        n_heads = config.num_aggregation_heads
        self._safa_params = [
            (torch.nn.Parameter(torch.randn((n_heads, safa_dim, safa_dim // 2)) * 0.005),
             torch.nn.Parameter(torch.ones((1, n_heads, safa_dim // 2)) * 0.1)),
            (torch.nn.Parameter(torch.randn((n_heads, safa_dim // 2, safa_dim)) * 0.005),
             torch.nn.Parameter(torch.ones((1, n_heads, safa_dim)) * 0.1)),
        ]

    def safa(self, x):
        batch_size = x.shape[0]
        (out, _) = torch.max(x, dim=-3)
        out = torch.reshape(out, (batch_size, 1, -1))

        for (weight, bias) in self._safa_params:
            out = torch.einsum('bdi, dij->bdj', out, weight) + bias
        return out

    def forward(self, x):
        features = feature_extraction(self.backbone, x)
        batch_size, num_channels, _, _ = features.shape
        attention = self.safa(features)
        vectorized_features = torch.reshape(features, (batch_size, num_channels, -1))
        per_head_embedding = torch.einsum('bci,bdi->bdc', vectorized_features, attention)
        embedding = torch.reshape(per_head_embedding, (batch_size, -1))
        return F.normalize(embedding)
