
import common.torch.load_torch_deps
import torch
import json
import msgspec

from pathlib import Path

from experimental.overhead_matching.swag.model.swag_config_types import SafaExtractorConfig, ExtractorDataRequirement
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model import patch_embedding
from experimental.overhead_matching.swag.model.patch_embedding import WagPatchEmbedding


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile'd OptimizedModule to get the original module."""
    if hasattr(model, '_orig_mod'):
        print(f"[SafaExtractor] Unwrapping torch.compile'd {type(model).__name__} "
              f"-> {type(model._orig_mod).__name__}")
        return model._orig_mod
    return model


def _load_inner_model(model_path: Path) -> torch.nn.Module:
    try:
        model = torch.load(model_path / "model.pt", map_location="cpu", weights_only=False)
        model = _unwrap_compiled(model)
        print(f"[SafaExtractor] Loaded model.pt: type={type(model).__name__}")
        if hasattr(model, '_config'):
            print(f"[SafaExtractor]   skip_aggregation={model._config.skip_aggregation}, "
                  f"num_embeddings={model._config.num_embeddings}")
        return model
    except Exception as e:
        print(f"[SafaExtractor] Failed to load model.pt, falling back to config+weights: {e}")

    config_field = "sat_model_config" if "satellite" in model_path.name else "pano_model_config"
    config_json_path = model_path.parent / "config.json"
    config_yaml_path = model_path.parent / "train_config.yaml"
    if config_json_path.exists():
        training_config = json.loads(config_json_path.read_text())
    elif config_yaml_path.exists():
        import yaml
        training_config = yaml.safe_load(config_yaml_path.read_bytes())
    else:
        raise FileNotFoundError(f"No config.json or train_config.yaml found in {model_path.parent}")

    # Lazy import to avoid circular dependency
    from experimental.overhead_matching.swag.model import swag_patch_embedding

    model_config_json = training_config[config_field]
    config = msgspec.json.decode(
        json.dumps(model_config_json),
        type=patch_embedding.WagPatchEmbeddingConfig | swag_patch_embedding.SwagPatchEmbeddingConfig)

    model_weights = torch.load(model_path / "model_weights.pt", weights_only=True)
    model_weights = {k.removeprefix("_orig_mod."): v for k, v in model_weights.items()}

    model_type = (WagPatchEmbedding
                  if isinstance(config, patch_embedding.WagPatchEmbeddingConfig)
                  else swag_patch_embedding.SwagPatchEmbedding)
    model = model_type(config)
    model.load_state_dict(model_weights)
    return model


class SafaExtractor(torch.nn.Module):
    def __init__(self, config: SafaExtractorConfig):
        super().__init__()
        self._inner_model = _load_inner_model(Path(config.model_path))
        self._is_wag = isinstance(self._inner_model, WagPatchEmbedding)

        if config.freeze:
            for param in self._inner_model.parameters():
                param.requires_grad = False
            self._inner_model.eval()

        self._freeze = config.freeze

    def train(self, mode=True):
        super().train(mode)
        if self._freeze:
            self._inner_model.eval()
        return self

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if self._is_wag:
            embeddings, _ = self._inner_model(model_input.image)
        else:
            embeddings, _ = self._inner_model(model_input)

        if not hasattr(self, '_shape_logged'):
            print(f"[SafaExtractor] forward output shape: {embeddings.shape}")
            self._shape_logged = True

        batch_size, num_embeddings, output_dim = embeddings.shape
        positions = torch.zeros(batch_size, num_embeddings, 1, 2,
                                device=embeddings.device)
        mask = torch.zeros(batch_size, num_embeddings,
                           dtype=torch.bool, device=embeddings.device)

        return ExtractorOutput(
            features=embeddings,
            positions=positions,
            mask=mask)

    @property
    def output_dim(self):
        return self._inner_model.output_dim

    @property
    def num_position_outputs(self):
        return 1

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.IMAGES]
