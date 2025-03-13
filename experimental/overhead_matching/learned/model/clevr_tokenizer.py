import common.torch.load_torch_deps
import torch
from torchvision.ops.misc import Conv2dNormActivation
from typing import Callable
from dataclasses import dataclass
from experimental.overhead_matching.learned.model import perspective_projection
import experimental.overhead_matching.learned.data.clevr_dataset as cd


def create_scene_tokens(scene_list_batch, vocabulary):
    """
    A batch is a list of scenes
    A scene is a list of items

    A vocabulary is a dictionary from strings to list of strings
    where the keys correspond to keys of an item and vocabulary[key]
    corresponds to all possible values that the key can take.

    This function assigns a unique token id for each object in the scene

    This function also returns a mask where elements that are true correspond
    to tokens that should be ignored
    """

    vocab_keys = sorted(vocabulary)
    num_scenes = len(scene_list_batch)
    max_tokens_per_scene = max([len(x) for x in scene_list_batch])
    tokens = torch.zeros((num_scenes, max_tokens_per_scene), dtype=torch.int32)
    mask = torch.ones((num_scenes, max_tokens_per_scene), dtype=torch.bool)

    for i, scene in enumerate(scene_list_batch):
        for j, item in enumerate(scene):
            embedding = 0
            for key in vocab_keys:
                options = vocabulary[key]
                embedding = embedding * len(options) + options.index(item[key])

            tokens[i, j] = embedding
            mask[i, j] = False

    return {"tokens": tokens, "mask": mask}


def get_xy_from_batch(scene_list_batch):
    num_scenes = len(scene_list_batch)
    max_tokens_per_scene = max([len(x) for x in scene_list_batch])
    xy_pos = torch.zeros((num_scenes, max_tokens_per_scene, 2), dtype=torch.float32)
    for i, scene in enumerate(scene_list_batch):
        for j, item in enumerate(scene):
            xy_pos[i, j, 0] = item["3d_coords"][0]
            xy_pos[i, j, 1] = item["3d_coords"][1]
    return xy_pos

def create_position_embeddings(
    scene_list_batch, *, min_scale: float = 1e-3, scale_step: float = 2.0, embedding_size: int
):
    assert embedding_size % 4 == 0
    xy_pos = get_xy_from_batch(scene_list_batch)

    out = torch.zeros(
        (xy_pos.shape[0], xy_pos.shape[1], embedding_size), dtype=torch.float32
    )
    num_scales = embedding_size // 4
    for scale_idx in range(num_scales):
        embedding_idx_start = 4 * scale_idx
        scale = min_scale * scale_step**scale_idx / (2 * torch.pi)

        out[:, :, embedding_idx_start + 0] = torch.sin(xy_pos[:, :, 0] / scale)
        out[:, :, embedding_idx_start + 1] = torch.cos(xy_pos[:, :, 0] / scale)
        out[:, :, embedding_idx_start + 2] = torch.sin(xy_pos[:, :, 1] / scale)
        out[:, :, embedding_idx_start + 3] = torch.cos(xy_pos[:, :, 1] / scale)

    return out


def create_spherical_embeddings(
        scene_list_batch, *, scale_step: float = 2.0, embedding_size: int
):
    assert embedding_size % 4 == 0
    num_scenes = len(scene_list_batch)
    max_tokens_per_scene = max([len(x) for x in scene_list_batch])
    xyz_pos = torch.zeros((num_scenes, max_tokens_per_scene, 3), dtype=torch.float32)
    for i, scene in enumerate(scene_list_batch):
        for j, item in enumerate(scene):
            xyz_pos[i, j, 0] = item["3d_coords"][0]
            xyz_pos[i, j, 1] = item["3d_coords"][1]
            xyz_pos[i, j, 2] = item["3d_coords"][2]

    spherical_coords = perspective_projection.spherical_projection(xyz_pos)
    inclination_azimuth = spherical_coords[..., 1:]

    out = torch.zeros(
        (num_scenes, max_tokens_per_scene, embedding_size), dtype=torch.float32
    )
    num_scales = embedding_size // 4
    for scale_idx in range(num_scales):
        embedding_idx_start = 4 * scale_idx
        scale = scale_step ** scale_idx

        out[:, :, embedding_idx_start + 0] = torch.sin(inclination_azimuth[:, :, 0] * scale)
        out[:, :, embedding_idx_start + 1] = torch.cos(inclination_azimuth[:, :, 0] * scale)
        out[:, :, embedding_idx_start + 2] = torch.sin(inclination_azimuth[:, :, 1] * scale)
        out[:, :, embedding_idx_start + 3] = torch.cos(inclination_azimuth[:, :, 1] * scale)
    return out


@dataclass
class ConvStemConfig:
    num_conv_layers: int  # conv dim will be [3, 32, 64, ...]
    kernel_size: int
    stride: int
    norm_layer: Callable[..., torch.nn.Module] = torch.nn.BatchNorm2d
    activation_layer: Callable[..., torch.nn.Module] = torch.nn.ReLU


@dataclass
class ImageToTokensConfig:
    embedding_dim: int
    image_shape: tuple[int, int]  # (h, w)
    patch_size_or_conv_stem_config: int | ConvStemConfig


class ImageToTokens(torch.nn.Module):
    # Based in part on https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
    def __init__(self, config: ImageToTokensConfig):
        super().__init__()
        # As per https://arxiv.org/abs/2106.14881
        self.config = config

        self.overhead_token_positions = None

        if isinstance(self.config.patch_size_or_conv_stem_config, ConvStemConfig):
            seq_proj = torch.nn.Sequential()
            prev_channels = 3
            for i in range(config.patch_size_or_conv_stem_config.num_conv_layers):
                out_channels = 32 if prev_channels == 3 else 2 * prev_channels
                seq_proj.add_module(
                    f"conv_bn_act_{i}",
                    Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=out_channels,
                        kernel_size=self.config.patch_size_or_conv_stem_config.kernel_size,
                        stride=self.config.patch_size_or_conv_stem_config.stride,
                        norm_layer=self.config.patch_size_or_conv_stem_config.norm_layer,
                        activation_layer=self.config.patch_size_or_conv_stem_config.activation_layer,
                    ),
                )
                prev_channels = out_channels
            seq_proj.add_module(
                "conv_last", torch.nn.Conv2d(in_channels=prev_channels,
                                             out_channels=self.config.embedding_dim, kernel_size=1)
            )
            self.conv_proj = seq_proj
        else:
            self.conv_proj = torch.nn.Conv2d(in_channels=3, out_channels=self.config.embedding_dim,
                                             kernel_size=self.config.patch_size_or_conv_stem_config, stride=self.config.patch_size_or_conv_stem_config)
            uu, vv = torch.meshgrid(torch.arange(0, self.config.image_shape[1] // self.config.patch_size_or_conv_stem_config, dtype=torch.float32), torch.arange(0, self.config.image_shape[0] // self.config.patch_size_or_conv_stem_config, dtype=torch.float32), indexing="xy")
            all_patch_centers_pixels = torch.stack([uu, vv]).reshape(2, -1)
            all_patch_centers_pixels = all_patch_centers_pixels * self.config.patch_size_or_conv_stem_config
            all_patch_centers_pixels += self.config.patch_size_or_conv_stem_config / 2
            all_patch_centers_m = cd.project_overhead_pixels_to_ground_plane(all_patch_centers_pixels)
            all_patch_centers_m = all_patch_centers_m[:2, :].T  # drop z values, transpose to be num_tokens, (x,y)
            self.overhead_token_positions = all_patch_centers_m
            


        with torch.no_grad():
            dummy_input = torch.zeros((5, 3, *self.config.image_shape))
            output_shape = self.conv_proj(dummy_input).shape
            sequence_length = output_shape[2] * output_shape[3]

        self.position_encoding = torch.nn.Parameter(torch.empty(
            1, sequence_length, self.config.embedding_dim).normal_(0.2))

    def forward(self, image):
        assert image.shape[2] == self.config.image_shape[0] and image.shape[3] == self.config.image_shape[
            1], f"Image shape received {image.shape}, expected {self.config.image_shape} for last two dims"
        x = self.conv_proj(image)
        x = x.reshape(image.shape[0], self.config.embedding_dim, -1)  # flatten image dimension
        x = x.permute(0, 2, 1)  # batch x seq_length x hidden dim
        x = x + self.position_encoding
        return x
