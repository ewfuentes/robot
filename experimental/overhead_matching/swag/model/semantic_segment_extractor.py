
import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.model.swag_model_input import ModelInput
from experimental.overhead_matching.swag.model.swag_config_types import (
        SemanticSegmentExtractorConfig)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
from PIL import Image
import open_clip
import ipdb


def sample_pts(num_pts):
    gen = np.random.default_rng(0)
    rvs = gen.random((num_pts, 2))
    elevation_rad = np.arcsin(2 * rvs[:, 0] - 1)
    row_frac = 1.0/2.0 - elevation_rad / np.pi
    col_frac = rvs[:, 1]
    return np.stack((col_frac, row_frac), axis=1)


class SemanticSegmentExtractor(torch.nn.Module):
    def __init__(self, config: SemanticSegmentExtractorConfig):
        super().__init__()
        self._clip_model, self._clip_preprocess = open_clip.create_model_from_pretrained(
                config.clip_model_str)
        self._clip_model.eval()

        self._mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
                config.sam_model_str,
                points_per_side=None,
                points_per_batch=128,
                point_grids=[sample_pts(config.num_query_pts)])

    def forward(self, model_input: ModelInput):
        images = model_input.image.permute(0, 2, 3, 1).cpu().numpy()
        batch_features = []
        for batch_idx in range(model_input.image.shape[0]):
            with torch.no_grad(), torch.autocast("cuda"):
                image_features = []
                segments = self._mask_generator.generate(images[batch_idx])
                for segment in segments:
                    start_col, start_row, width, height = segment["bbox"]
                    start_row = int(start_row)
                    start_col = int(start_col)
                    end_row = int(start_row + height)
                    end_col = int(start_col + width)

                    # TODO: Can we preprocess the image once and extract crops?
                    # (and probably resize)
                    pil_image = Image.fromarray(np.uint8(
                        images[batch_idx, start_row:end_row, start_col:end_col, :] * 255))
                    clip_image = self._clip_preprocess(pil_image).unsqueeze(0)
                    image_features.append({
                        'bbox': segment["bbox"],
                        'clip_feature': self._clip_model.encode_image(clip_image)
                    })
            batch_features.append(image_features)

        max_num_segments = max([len(x) for x in batch_features])
        batch_size = model_input.image.shape[0]
        dev = model_input.image.device
        CLIP_FEATURE_DIM = 512

        semantic_positions = torch.empty((batch_size, max_num_segments, 2))
        semantic_tokens = torch.empty((batch_size, max_num_segments, CLIP_FEATURE_DIM))
        semantic_mask = torch.ones((batch_size, max_num_segments))
        for batch_idx, image_features in enumerate(batch_features):
            semantic_mask[batch_idx, :len(image_features)] = 0
            for segment_idx, segment_feature in enumerate(image_features):
                start_col, start_row, width, height = segment_feature["bbox"]
                semantic_positions[batch_idx, segment_idx, 0] = start_row + height / 2.0
                semantic_positions[batch_idx, segment_idx, 1] = start_col + width / 2.0
                semantic_tokens[batch_idx, segment_idx, :] = segment_feature["clip_feature"]
        return semantic_positions.to(dev), semantic_tokens.to(dev), semantic_mask.to(dev)
