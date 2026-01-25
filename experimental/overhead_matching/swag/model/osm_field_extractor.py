
import common.torch.load_torch_deps
import torch
import pickle
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMFieldExtractorConfig, ExtractorDataRequirement, LandmarkType)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    compute_landmark_pano_positions, compute_landmark_sat_positions)


class OSMFieldExtractor(torch.nn.Module):
    """Generic extractor for OSM field embeddings.

    Extracts embeddings for a single configurable tag type (name, brand,
    amenity, street_address, or osm_sentences) from OSM landmarks.

    This works with the v2.0 pickle format where each tag type has its own
    pre-computed embeddings indexed by custom_id (SHA256 hash of pruned_props).
    """

    def __init__(self, config: OSMFieldExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self.files_loaded = False
        self.embeddings_tensor = None  # [num_items, embedding_dim]
        self.custom_id_to_idx = None   # dict: custom_id -> index

    def load_files(self):
        """Load embeddings for the configured tag from v2.0 pickle."""
        pickle_path = (self.semantic_embedding_base_path /
                       self.config.embedding_version / "embeddings" / "embeddings.pkl")

        if not pickle_path.exists():
            raise FileNotFoundError(f"Embeddings pickle not found at {pickle_path}")

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        if not isinstance(data, dict) or data.get("version") != "2.0":
            raise RuntimeError(
                f"Expected v2.0 pickle format, got: {type(data)}, "
                f"version={data.get('version') if isinstance(data, dict) else 'N/A'}")

        if self.config.tag not in data:
            available_tags = [k for k in data.keys() if k != "version"]
            raise KeyError(
                f"Tag '{self.config.tag}' not found in pickle. "
                f"Available tags: {available_tags}")

        tag_data = data[self.config.tag]
        # Both tuple and list formats are supported
        if isinstance(tag_data, (tuple, list)) and len(tag_data) >= 2:
            self.embeddings_tensor, self.custom_id_to_idx = tag_data[0], tag_data[1]
        else:
            raise RuntimeError(
                f"Unexpected format for tag '{self.config.tag}': {type(tag_data)}")

        # Crop to requested embedding size if needed
        if self.config.openai_embedding_size < self.embeddings_tensor.shape[1]:
            self.embeddings_tensor = self.embeddings_tensor[:, :self.config.openai_embedding_size]

        # Normalize embeddings
        self.embeddings_tensor = (self.embeddings_tensor /
                                   torch.norm(self.embeddings_tensor, dim=-1, keepdim=True))

        self.files_loaded = True
        print(f"OSMFieldExtractor[{self.config.tag}]: Loaded {len(self.custom_id_to_idx)} embeddings")

    def _filter_landmark(self, landmark) -> bool:
        """Return True if landmark passes the geometry filter (if any)."""
        if self.config.landmark_type is None:
            return True
        return landmark['geometry'].geom_type.lower() == self.config.landmark_type.value.lower()

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self.files_loaded:
            self.load_files()

        batch_size = len(model_input.metadata)
        is_panorama = 'pano_id' in model_input.metadata[0]

        # Count valid landmarks (those with embeddings for this tag and passing geometry filter)
        valid_counts = []
        for item in model_input.metadata:
            count = 0
            for lm in item["landmarks"]:
                if not self._filter_landmark(lm):
                    continue
                custom_id = custom_id_from_props(lm['pruned_props'])
                if custom_id in self.custom_id_to_idx:
                    count += 1
            valid_counts.append(count)

        max_landmarks = max(valid_counts) if valid_counts else 0

        if max_landmarks == 0:
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.output_dim), device=model_input.image.device),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=model_input.image.device),
                positions=torch.zeros((batch_size, 0, 2, 2), device=model_input.image.device),
                debug={})

        # Build output tensors
        features = torch.zeros((batch_size, max_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_landmarks, 2, 2))
        mask = torch.ones((batch_size, max_landmarks), dtype=torch.bool)

        for i, item in enumerate(model_input.metadata):
            # Build mask for landmarks that pass geometry filter AND have embeddings
            landmark_mask = []
            for lm in item["landmarks"]:
                passes_filter = self._filter_landmark(lm)
                has_embedding = False
                if passes_filter:
                    custom_id = custom_id_from_props(lm['pruned_props'])
                    has_embedding = custom_id in self.custom_id_to_idx
                landmark_mask.append(passes_filter and has_embedding)

            landmark_mask_tensor = torch.tensor(landmark_mask, dtype=torch.bool)

            # Compute positions for valid landmarks
            num_valid = sum(landmark_mask)
            if num_valid > 0:
                if is_panorama:
                    computed_positions = compute_landmark_pano_positions(
                        item, model_input.image.shape[-2:], landmark_mask=landmark_mask_tensor)
                else:
                    computed_positions = compute_landmark_sat_positions(
                        item, landmark_mask=landmark_mask_tensor)
                positions[i, :num_valid] = computed_positions

            # Extract features for valid landmarks
            token_idx = 0
            for lm_idx, lm in enumerate(item["landmarks"]):
                if not landmark_mask[lm_idx]:
                    continue

                custom_id = custom_id_from_props(lm['pruned_props'])
                idx = self.custom_id_to_idx[custom_id]
                features[i, token_idx] = self.embeddings_tensor[idx]
                mask[i, token_idx] = False
                token_idx += 1

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug={})

    @property
    def output_dim(self):
        return self.config.openai_embedding_size

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.LANDMARKS]
