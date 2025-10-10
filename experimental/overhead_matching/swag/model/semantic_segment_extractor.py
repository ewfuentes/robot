
import common.torch.load_torch_deps
import torch

from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput, SemanticTokenExtractorOutput
from experimental.overhead_matching.swag.model.swag_config_types import (
        SemanticSegmentExtractorConfig)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
from PIL import Image
import open_clip


class SemanticSegmentExtractor(torch.nn.Module):
    def __init__(self, config: SemanticSegmentExtractorConfig):
        super().__init__()
        self._clip_model, self._clip_preprocess = open_clip.create_model_from_pretrained(
                config.clip_model_str)
        self._clip_model.eval()

        self._mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
            config.sam_model_str,
            points_per_side=32,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.85,
            stability_score_offset=1.0,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=512/1500,
            crop_n_points_downscale_factor=1,
            point_grids=None,
            min_mask_region_area=200)

    def forward(self, model_input: ModelInput) -> SemanticTokenExtractorOutput:
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

                    if width == 0 or height == 0:
                        continue

                    # TODO: Can we preprocess the image once and extract crops?
                    # (and probably resize)
                    pil_image = Image.fromarray(np.uint8(
                        images[batch_idx, start_row:end_row, start_col:end_col, :] * 255))
                    clip_image = self._clip_preprocess(pil_image).unsqueeze(0).to(model_input.image.device)
                    image_features.append({
                        'bbox': segment["bbox"],
                        'clip_feature': self._clip_model.encode_image(clip_image)
                    })
            batch_features.append(image_features)

        max_num_segments = max([len(x) for x in batch_features])
        batch_size = model_input.image.shape[0]
        dev = model_input.image.device

        semantic_positions = torch.zeros((batch_size, max_num_segments, self.num_position_outputs, 2))
        semantic_tokens = torch.zeros((batch_size, max_num_segments, self.output_dim))
        semantic_mask = torch.ones((batch_size, max_num_segments), dtype=torch.bool)
        for batch_idx, image_features in enumerate(batch_features):
            semantic_mask[batch_idx, :len(image_features)] = False
            for segment_idx, segment_feature in enumerate(image_features):
                start_col, start_row, width, height = segment_feature["bbox"]
                semantic_positions[batch_idx, segment_idx, 0, 0] = start_row + height / 2.0
                semantic_positions[batch_idx, segment_idx, 0, 1] = start_col + width / 2.0
                semantic_tokens[batch_idx, segment_idx, :] = segment_feature["clip_feature"]
        return SemanticTokenExtractorOutput(
            positions=semantic_positions.to(dev),
            features=semantic_tokens.to(dev),
            mask=semantic_mask.to(dev))

    @property
    def output_dim(self):
        return self._clip_model.visual.output_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def num_position_outputs(self):
        return 1
