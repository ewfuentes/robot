import common.torch.load_torch_deps
import torch
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMFieldExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    compute_landmark_pano_positions, compute_landmark_sat_positions)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    load_v2_pickle, iter_city_directories, normalize_cropped_embeddings)


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
        self.embeddings = None  # dict: custom_id -> tensor

    def _load_from_pickle(self, pickle_path: Path, label: str):
        """Load embeddings for the configured tag from a single v2.0 pickle.

        Args:
            pickle_path: Path to the embeddings.pkl file.
            label: Label for logging (e.g. city name or "flat").
        """
        data = load_v2_pickle(pickle_path)
        if data is None:
            print(f"  Warning: {pickle_path} missing or not v2.0 format, skipping")
            return

        if self.config.tag not in data:
            print(f"  Warning: Tag '{self.config.tag}' not in {pickle_path}, skipping")
            return

        tag_data = data[self.config.tag]
        if isinstance(tag_data, (tuple, list)) and len(tag_data) >= 2:
            tensor, id_to_idx = tag_data[0], tag_data[1]
        else:
            raise RuntimeError(
                f"Unexpected format for tag '{self.config.tag}': {type(tag_data)}")

        # Crop to requested embedding size
        if self.config.openai_embedding_size < tensor.shape[1]:
            tensor = tensor[:, :self.config.openai_embedding_size]

        # Convert to dict: custom_id -> tensor row
        for custom_id, idx in id_to_idx.items():
            self.embeddings[custom_id] = tensor[idx]

        print(f"  OSMFieldExtractor[{self.config.tag}]: Loaded {len(id_to_idx)} embeddings for {label}")

    def load_files(self):
        """Load embeddings for the configured tag from v2.0 pickles.

        Tries flat layout first (base_path/embeddings/embeddings.pkl), which is
        standard for OSM embeddings. Falls back to multi-city layout with per-city
        subdirectories for compatibility.
        """
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        self.embeddings = {}

        # Try flat layout first (standard for OSM embeddings)
        flat_pickle = base_path / "embeddings" / "embeddings.pkl"
        if flat_pickle.exists():
            self._load_from_pickle(flat_pickle, "flat")
        else:
            # Fall back to multi-city layout
            for city_name, city_dir in iter_city_directories(base_path):
                pickle_path = city_dir / "embeddings" / "embeddings.pkl"
                self._load_from_pickle(pickle_path, city_name)

        assert len(self.embeddings) > 0, (
            f"Failed to load any embeddings for tag '{self.config.tag}' from {base_path}")

        self.files_loaded = True
        print(f"OSMFieldExtractor[{self.config.tag}]: Total {len(self.embeddings)} embeddings loaded")

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
                if custom_id in self.embeddings:
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
                    has_embedding = custom_id in self.embeddings
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
                features[i, token_idx] = self.embeddings[custom_id]
                mask[i, token_idx] = False
                token_idx += 1

        # Re-normalize embeddings after cropping
        features = normalize_cropped_embeddings(features, mask)

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
