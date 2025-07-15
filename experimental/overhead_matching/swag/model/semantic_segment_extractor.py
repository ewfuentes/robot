
import common.torch.load_torch_deps
import torch
from pathlib import Path

from experimental.overhead_matching.swag.model.swag_model_input import ModelInput
from experimental.overhead_matching.swag.model.swag_config_types import (
        SemanticSegmentExtractorConfig)
from x_segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from x_segment_anything.build_sam import sam_model_urls


def load_model(model_str: str):
    assert model_str in sam_model_urls
    model_file = Path(f'~/.cache/robot/xsam_models/{model_str}.pt').expanduser()
    if not model_file.exists():
        model_file.parent.mkdir(parents=True, exist_ok=True)
        import requests
        print(f"Fetching {model_str} from {sam_model_urls[model_str]}")
        response = requests.get(sam_model_urls[model_str])
        model_file.write_bytes(response.content)
        print(f"Model written to {model_file}")
    print(f"Loading model from {model_file}")
    return sam_model_registry[model_str](checkpoint=model_file)


class SemanticSegmentExtractor(torch.nn.Module):
    def __init__(self, config: SemanticSegmentExtractorConfig):
        super().__init__()
        self._sam = load_model(config.sam_model_str)
        self._mask_generator = SamAutomaticMaskGenerator(self._sam)

    def forward(self, model_input: ModelInput):
        results = self._mask_generator.generate(model_input)
        return results
