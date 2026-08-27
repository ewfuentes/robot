"""PyTorch port of the CrossLocate VGG16-MAC descriptor network.

Faithful to the released ``AlpsPhotosToDepthCompact_31_2`` checkpoint
(Tomesek et al., WACV 2022), whose exact training config was published with
the weights. Release-audited facts this port encodes (see
``dem_baseline_grounding.md`` and the released ``train_config.py``):

- Shared single-branch VGG16 conv stack (13 convs, 4 max-pools; there is no
  pool after conv13 when global pooling is used).
- conv13 has NO ReLU (``withRelu=(pooling is NONE)`` in the release).
- Aggregation: per-pixel L2 normalization over channels, MAC (spatial max),
  then final L2 normalization -> 512-D descriptor.
- ``NetPreprocessing.NONE``: inputs are raw float32. RGB queries are 0-255
  RGB; database inputs are raw *metric* depth (meters) replicated to three
  channels. No mean subtraction, no scaling.
- Native input resolution 500x500 (the net is fully convolutional; other
  sizes work but the checkpoint was trained at 500).
- Retrieval metric: squared Euclidean distance. Descriptors are unit-norm so
  Euclidean ranking == cosine ranking; ``descriptor_distances`` keeps the
  release's convention anyway so raw values are comparable.

TF/torch equivalences relied on: TF 'SAME' 3x3 stride-1 conv == torch
padding=1; TF 'SAME' 2x2/2 max-pool == torch ``MaxPool2d(2, 2,
ceil_mode=True)`` (both pad bottom/right when odd, and padding cannot win a
max); TF kernels are (kh, kw, cin, cout) -> torch (cout, cin, kh, kw).

Weights load from an .npz produced by ``convert_checkpoint.py`` (TF variable
names like ``conv1/kernels``); nothing here imports TensorFlow.
"""

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch
import torch.nn.functional as F

import numpy as np
from pathlib import Path

DESCRIPTOR_DIM = 512
NATIVE_INPUT_HW = (500, 500)

# (in_channels, out_channels) per conv layer, 1-indexed names conv1..conv13.
_CONV_CHANNELS = [
    (3, 64), (64, 64),
    (64, 128), (128, 128),
    (128, 256), (256, 256), (256, 256),
    (256, 512), (512, 512), (512, 512),
    (512, 512), (512, 512), (512, 512),
]
# Max-pool after these (1-indexed) conv layers. None after conv13.
_POOL_AFTER = {2, 4, 7, 10}


class CrossLocateVGG16MAC(torch.nn.Module):
    """VGG16 conv stack -> per-pixel L2 -> MAC -> L2, as released."""

    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(cin, cout, kernel_size=3, padding=1)
            for cin, cout in _CONV_CHANNELS
        ])
        self.pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) raw-valued float input -> (N, 512) unit descriptors."""
        for i, conv in enumerate(self.convs, start=1):
            x = conv(x)
            if i < len(self.convs):  # conv13 has no ReLU in the release
                x = F.relu(x, inplace=True)
            if i in _POOL_AFTER:
                x = self.pool(x)
        # Pre-feature L2 norm over channels at each spatial position. The TF
        # release uses default epsilon 1e-12 inside l2_normalize.
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        # MAC: global spatial max.
        x = torch.amax(x, dim=(2, 3))
        return F.normalize(x, p=2, dim=1, eps=1e-12)


def load_converted_weights(model: CrossLocateVGG16MAC,
                           npz_path: Path) -> None:
    """Load a converted-checkpoint .npz (TF names) into the torch model.

    The conversion tool preserves TF variable names (``conv{i}/kernels``,
    ``conv{i}/biases``) and raw TF layouts; the (kh, kw, cin, cout) ->
    (cout, cin, kh, kw) permute happens here so the .npz stays a faithful
    dump of the checkpoint.
    """
    data = np.load(npz_path)
    for i, conv in enumerate(model.convs, start=1):
        kernels = torch.from_numpy(data[f"conv{i}/kernels"])
        biases = torch.from_numpy(data[f"conv{i}/biases"])
        expected = (conv.out_channels, conv.in_channels, 3, 3)
        permuted = kernels.permute(3, 2, 0, 1).contiguous()
        if tuple(permuted.shape) != expected:
            raise ValueError(
                f"conv{i}: checkpoint kernel shape {tuple(kernels.shape)} "
                f"does not map to {expected}")
        with torch.no_grad():
            conv.weight.copy_(permuted)
            conv.bias.copy_(biases)


def rgb_query_tensor(image_rgb_u8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (3, h, w) float32 network input.

    Raw 0-255 values, resized to the native 500x500 with area interpolation
    exactly like the release (cv2.INTER_AREA == adaptive average pooling for
    integer factors; torchvision antialiased bilinear is the closest
    maintained equivalent and matched rankings must be verified against the
    released results, not assumed).
    """
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {image_rgb_u8.shape}")
    tensor = torch.from_numpy(
        np.ascontiguousarray(image_rgb_u8)).permute(2, 0, 1).float()
    return _resize_area(tensor, NATIVE_INPUT_HW)


def depth_render_tensor(depth_m: np.ndarray, sky_fill_m: float) -> torch.Tensor:
    """(H, W) metric depth with +inf sky -> (3, h, w) float32 network input.

    The released database EXRs store raw metric depth in the R channel and
    are fed to the network unchanged (replicated to 3 channels). ``sky_fill_m``
    is the finite value standing in for sky/no-return; it must match the
    release convention (measured from released EXRs, recorded in the render
    manifest), not be invented per run.
    """
    depth = torch.from_numpy(np.ascontiguousarray(depth_m)).float()
    depth = torch.where(torch.isfinite(depth), depth,
                        torch.full_like(depth, sky_fill_m))
    depth = _resize_area(depth.unsqueeze(0), NATIVE_INPUT_HW)
    return depth.expand(3, -1, -1).contiguous()


def _resize_area(chw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    if chw.shape[-2:] == out_hw:
        return chw
    return F.interpolate(chw.unsqueeze(0), size=out_hw, mode="area").squeeze(0)


def descriptor_distances(queries: torch.Tensor,
                         database: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances, (n_q, d) x (n_db, d) -> (n_q, n_db).

    For unit-norm descriptors this is 2 - 2 * cosine; kept in the release's
    distance convention so thresholds/margins are directly comparable.
    """
    return torch.cdist(queries, database, p=2).square()
