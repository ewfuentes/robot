"""Tag token extractors for OSM and panorama landmarks.

Encodes (key, value) OSM tag pairs as individual tokens for cross-attention
between panorama and OSM tag sets via the transformer encoder distance function.

Each tag becomes one token: Linear(concat(key_emb, ngram_hash(value)[, desc_proj])) + landmark_idx_emb.
When description embeddings are enabled, they are concatenated into the projection input
for each tag rather than emitted as separate tokens.
"""

import common.torch.load_torch_deps
import torch
import torch.nn as nn
from pathlib import Path

from experimental.overhead_matching.swag.model.swag_config_types import (
    OSMTagTokenExtractorConfig,
    PanoTagTokenExtractorConfig,
    ExtractorDataRequirement,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput,
)
from experimental.overhead_matching.swag.model.additional_panorama_extractors import (
    yaw_angles_to_binary_vector,
    extract_yaw_angles_from_bboxes,
    load_v2_tags_pickle,
    iter_city_directories,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    custom_id_from_props, load_embeddings,
)
from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
    compute_landmark_sat_positions,
    compute_landmark_pano_positions,
)


class CharNgramHasher(nn.Module):
    """Encode arbitrary strings into fixed-dim vectors via character n-gram hashing."""

    def __init__(self, num_buckets: int = 10_000, embedding_dim: int = 64):
        super().__init__()
        self.num_buckets = num_buckets
        self.embedding_bag = nn.EmbeddingBag(num_buckets, embedding_dim, mode='mean')

    def _extract_ngrams(self, text: str, n: int = 3) -> list[int]:
        padded = f"<{text.lower()}>"
        return [hash(padded[i:i+n]) % self.num_buckets for i in range(len(padded) - n + 1)]

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            Tensor of shape [len(texts), embedding_dim].
        """
        if not texts:
            return torch.zeros(
                (0, self.embedding_bag.embedding_dim),
                device=self.embedding_bag.weight.device,
            )

        all_indices = []
        offsets = [0]
        for text in texts:
            ngrams = self._extract_ngrams(text)
            if not ngrams:
                ngrams = [0]
            all_indices.extend(ngrams)
            offsets.append(offsets[-1] + len(ngrams))

        indices = torch.tensor(all_indices, dtype=torch.long,
                               device=self.embedding_bag.weight.device)
        offsets_t = torch.tensor(offsets[:-1], dtype=torch.long,
                                 device=self.embedding_bag.weight.device)
        return self.embedding_bag(indices, offsets_t)


def _load_key_vocabulary(vocabulary_file: str) -> list[str]:
    """Load key vocabulary from a text file (one key per line)."""
    path = Path(vocabulary_file)
    if not path.exists():
        raise FileNotFoundError(f"Key vocabulary file not found: {path}")
    with open(path) as f:
        keys = [line.strip() for line in f if line.strip()]
    return keys


class OSMTagTokenExtractor(nn.Module):
    """Extract tag tokens from OSM landmark metadata in the dataset.

    For each landmark's pruned_props, each (key, value) pair becomes one token.
    When description embeddings are enabled, they are concatenated into the
    projection input for each tag token.
    """

    def __init__(self, config: OSMTagTokenExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()

        # Load key vocabulary
        key_vocab = _load_key_vocabulary(config.key_vocabulary_file)
        self._key_to_idx = {k: i + 1 for i, k in enumerate(key_vocab)}  # 0 = padding/unknown
        num_keys = len(key_vocab) + 1

        # Learnable key embeddings
        self._key_embedding = nn.Embedding(num_keys, config.key_embedding_dim, padding_idx=0)

        # Character n-gram hasher for values
        self._value_hasher = CharNgramHasher(
            num_buckets=config.ngram_bucket_size,
            embedding_dim=config.value_embedding_dim,
        )

        # Description embedding projection (if enabled)
        tag_proj_input_dim = config.key_embedding_dim + config.value_embedding_dim
        if config.include_description_embeddings:
            self._desc_projection = nn.Linear(
                config.description_embedding_dim, config.description_projection_dim)
            tag_proj_input_dim += config.description_projection_dim

        # Project concatenated (key_emb, value_emb[, desc_proj]) to token_dim
        self._tag_projection = nn.Linear(tag_proj_input_dim, config.token_dim)

        # Landmark index embeddings (OSM-specific)
        self._landmark_idx_embedding = nn.Embedding(config.max_landmarks, config.token_dim)

        # Lazy-loaded embeddings
        self.files_loaded = False
        self.all_embeddings_tensor = None
        self.landmark_id_to_idx = None

    def load_files(self):
        """Load sentence embeddings for OSM landmarks (flat tuple pickle format)."""
        if self.config.include_description_embeddings:
            embedding_directory = (
                self.semantic_embedding_base_path
                / self.config.embedding_version
                / "embeddings"
            )
            self.all_embeddings_tensor, self.landmark_id_to_idx = load_embeddings(
                embedding_directory
            )
            print(f"OSMTagTokenExtractor: loaded {len(self.landmark_id_to_idx)} description embeddings")
        self.files_loaded = True

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self.files_loaded:
            self.load_files()

        dev = self._key_embedding.weight.device

        if not model_input.metadata:
            return ExtractorOutput(
                features=torch.zeros((0, 0, self.config.token_dim), device=dev),
                positions=torch.zeros((0, 0, 2, 2), device=dev),
                mask=torch.ones((0, 0), dtype=torch.bool, device=dev),
                debug={},
            )

        batch_size = len(model_input.metadata)
        is_panorama = 'pano_id' in model_input.metadata[0]

        # First pass: collect token data and desc projections per batch item
        batch_tokens_data = []
        batch_desc_projections = []
        max_tokens = 0

        for item in model_input.metadata:
            tokens = []
            desc_projections = {}  # landmark_idx -> projected desc embedding
            landmarks = item.get("landmarks", [])

            for lm_idx, landmark in enumerate(landmarks):
                if lm_idx >= self.config.max_landmarks:
                    break
                props = landmark.get('pruned_props', {})
                if isinstance(props, frozenset):
                    props = dict(props)

                # One token per (key, value) pair
                for key, value in (props.items() if isinstance(props, dict) else props):
                    key_idx = self._key_to_idx.get(key, 0)
                    if key_idx == 0:
                        continue  # Unknown key, skip
                    tokens.append({
                        'key_idx': key_idx,
                        'value': str(value),
                        'landmark_idx': lm_idx,
                    })

                # Precompute description projection for this landmark
                if self.config.include_description_embeddings and self.all_embeddings_tensor is not None:
                    landmark_id = custom_id_from_props(props)
                    if landmark_id in self.landmark_id_to_idx:
                        emb_idx = self.landmark_id_to_idx[landmark_id]
                        desc_emb = self.all_embeddings_tensor[emb_idx].to(dev)
                        desc_projections[lm_idx] = self._desc_projection(desc_emb)

            batch_tokens_data.append(tokens)
            batch_desc_projections.append(desc_projections)
            max_tokens = max(max_tokens, len(tokens))

        if max_tokens == 0:
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.config.token_dim), device=dev),
                positions=torch.zeros((batch_size, 0, 2, 2), device=dev),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=dev),
                debug={},
            )

        # Build output tensors
        features = torch.zeros((batch_size, max_tokens, self.config.token_dim), device=dev)
        positions = torch.zeros((batch_size, max_tokens, 2, 2), device=dev)
        mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=dev)

        for i, (item, tokens, desc_projections) in enumerate(
                zip(model_input.metadata, batch_tokens_data, batch_desc_projections)):
            landmarks = item.get("landmarks", [])

            # Compute positions for all landmarks; token loop already caps lm_idx
            # to max_landmarks via the first-pass break.
            if landmarks:
                if is_panorama:
                    lm_positions = compute_landmark_pano_positions(
                        item, model_input.image.shape[-2:])
                else:
                    lm_positions = compute_landmark_sat_positions(item)
            else:
                lm_positions = torch.zeros((0, 2, 2))

            # Batch-hash all tag values
            tag_values = [t['value'] for t in tokens]
            hashed_values = self._value_hasher(tag_values) if tag_values else None

            for j, token_data in enumerate(tokens):
                lm_idx = token_data['landmark_idx']
                landmark_idx_emb = self._landmark_idx_embedding(
                    torch.tensor(lm_idx, device=dev))

                key_emb = self._key_embedding(
                    torch.tensor(token_data['key_idx'], device=dev))
                val_emb = hashed_values[j]

                if self.config.include_description_embeddings:
                    desc_proj = desc_projections.get(
                        lm_idx,
                        torch.zeros(self.config.description_projection_dim, device=dev))
                    token_vec = self._tag_projection(
                        torch.cat([key_emb, val_emb, desc_proj]))
                else:
                    token_vec = self._tag_projection(
                        torch.cat([key_emb, val_emb]))

                features[i, j] = token_vec + landmark_idx_emb

                # Set position from landmark positions
                if lm_idx < lm_positions.shape[0]:
                    positions[i, j] = lm_positions[lm_idx].to(dev)

                mask[i, j] = False

        return ExtractorOutput(
            features=features,
            positions=positions.to(dev),
            mask=mask.to(dev),
            debug={},
        )

    @property
    def output_dim(self):
        return self.config.token_dim

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.LANDMARKS]


class PanoTagTokenExtractor(nn.Module):
    """Extract tag tokens from panorama image landmarks (2.0_tags pickle format).

    For each landmark's primary_tag and additional_tags, each tag becomes one token.
    When description embeddings are enabled, they are concatenated into the
    projection input for each tag token.
    """

    def __init__(self, config: PanoTagTokenExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()

        # Load key vocabulary
        key_vocab = _load_key_vocabulary(config.key_vocabulary_file)
        self._key_to_idx = {k: i + 1 for i, k in enumerate(key_vocab)}  # 0 = padding/unknown
        num_keys = len(key_vocab) + 1

        # Learnable key embeddings
        self._key_embedding = nn.Embedding(num_keys, config.key_embedding_dim, padding_idx=0)

        # Character n-gram hasher for values
        self._value_hasher = CharNgramHasher(
            num_buckets=config.ngram_bucket_size,
            embedding_dim=config.value_embedding_dim,
        )

        # Description embedding projection (if enabled)
        tag_proj_input_dim = config.key_embedding_dim + config.value_embedding_dim
        if config.include_description_embeddings:
            self._desc_projection = nn.Linear(
                config.description_embedding_dim, config.description_projection_dim)
            tag_proj_input_dim += config.description_projection_dim

        # Project concatenated (key_emb, value_emb[, desc_proj]) to token_dim
        self._tag_projection = nn.Linear(tag_proj_input_dim, config.token_dim)

        # Landmark index embeddings (panorama-specific, separate from OSM)
        self._landmark_idx_embedding = nn.Embedding(config.max_landmarks, config.token_dim)

        # Lazy-loaded pickle data
        self.files_loaded = False
        self.panorama_data = None  # pano_id -> {location_type, landmarks}
        self._desc_embeddings_dict = None  # {custom_id: tensor}
        self._pano_id_to_full_prefix = None  # {clean_pano_id: full_prefix_with_coords}

    def load_files(self):
        """Load tag data from 2.0_tags pickle files across cities."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        self.panorama_data = {}
        all_desc_embeddings = {}

        for city_name, city_dir in iter_city_directories(base_path):
            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            data = load_v2_tags_pickle(pickle_path)
            if data is None:
                print(f"  Warning: {pickle_path} missing or not v2.0_tags format, skipping")
                continue

            # Load panorama tag data
            for pano_id, pano_data in data.get("panoramas", {}).items():
                pano_id_clean = pano_id.split(",")[0]
                self.panorama_data[pano_id_clean] = pano_data

            # Load description embeddings if available
            if (self.config.include_description_embeddings
                    and "description_embeddings" in data
                    and "description_id_to_idx" in data):
                tensor = data["description_embeddings"]
                id_to_idx = data["description_id_to_idx"]
                for custom_id, idx in id_to_idx.items():
                    all_desc_embeddings[custom_id] = tensor[idx]

            print(f"  Loaded {len(data.get('panoramas', {}))} panoramas for {city_name}")

        # Store description embeddings as dict for lookup
        self._desc_embeddings_dict = all_desc_embeddings

        # Precompute clean pano_id -> full pano_id (with coords) prefix mapping
        self._pano_id_to_full_prefix = {}
        for custom_id in all_desc_embeddings:
            full_prefix = custom_id.split("__landmark_")[0]
            clean_id = full_prefix.split(",")[0]
            if clean_id not in self._pano_id_to_full_prefix:
                self._pano_id_to_full_prefix[clean_id] = full_prefix

        self.files_loaded = True
        print(f"PanoTagTokenExtractor: loaded {len(self.panorama_data)} panoramas, "
              f"{len(all_desc_embeddings)} description embeddings")

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self.files_loaded:
            self.load_files()

        dev = self._key_embedding.weight.device
        batch_size = len(model_input.metadata)

        # Validate panorama data
        for item in model_input.metadata:
            if 'pano_id' not in item:
                raise ValueError(
                    "PanoTagTokenExtractor requires panorama data with 'pano_id' field.")

        # First pass: collect token data and desc projections per batch item
        batch_tokens_data = []
        batch_desc_projections = []
        max_tokens = 0

        for item in model_input.metadata:
            pano_id = item['pano_id']
            tokens = []
            desc_projections = {}  # landmark_idx -> projected desc embedding

            if pano_id not in self.panorama_data:
                batch_tokens_data.append(tokens)
                batch_desc_projections.append(desc_projections)
                continue

            pano_info = self.panorama_data[pano_id]

            # Look up the full pano_id (with coords) from precomputed mapping
            pano_id_with_coords = (
                self._pano_id_to_full_prefix.get(pano_id)
                if self.config.include_description_embeddings
                else None
            )

            for lm_idx, landmark in enumerate(pano_info.get("landmarks", [])):
                if lm_idx >= self.config.max_landmarks:
                    break

                # Primary tag
                primary_tag = landmark.get("primary_tag", {})
                if primary_tag:
                    key = primary_tag.get("key", "")
                    value = primary_tag.get("value", "")
                    key_idx = self._key_to_idx.get(key, 0)
                    if key_idx > 0:
                        tokens.append({
                            'key_idx': key_idx,
                            'value': value,
                            'landmark_idx': lm_idx,
                        })

                # Additional tags
                for tag in landmark.get("additional_tags", []):
                    key = tag.get("key", "")
                    value = tag.get("value", "")
                    key_idx = self._key_to_idx.get(key, 0)
                    if key_idx > 0:
                        tokens.append({
                            'key_idx': key_idx,
                            'value': value,
                            'landmark_idx': lm_idx,
                        })

                # Precompute description projection for this landmark
                if self.config.include_description_embeddings:
                    landmark_idx_val = landmark.get("landmark_idx", lm_idx)
                    if pano_id_with_coords:
                        custom_id = f"{pano_id_with_coords}__landmark_{landmark_idx_val}"
                    else:
                        custom_id = f"{pano_id}__landmark_{landmark_idx_val}"
                    if custom_id in self._desc_embeddings_dict:
                        desc_emb = self._desc_embeddings_dict[custom_id].to(dev)
                        desc_projections[lm_idx] = self._desc_projection(desc_emb)

            batch_tokens_data.append(tokens)
            batch_desc_projections.append(desc_projections)
            max_tokens = max(max_tokens, len(tokens))

        if max_tokens == 0:
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.config.token_dim), device=dev),
                positions=torch.zeros((batch_size, 0, 2, 2), device=dev),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=dev),
                debug={},
            )

        # Build output tensors
        features = torch.zeros((batch_size, max_tokens, self.config.token_dim), device=dev)
        positions = torch.zeros((batch_size, max_tokens, 2, 2), device=dev)
        mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=dev)

        for i, (item, tokens, desc_projections) in enumerate(
                zip(model_input.metadata, batch_tokens_data, batch_desc_projections)):
            pano_id = item['pano_id']

            if pano_id not in self.panorama_data:
                continue

            pano_info = self.panorama_data[pano_id]

            # Precompute yaw positions per landmark
            lm_positions = {}
            for lm_idx, landmark in enumerate(pano_info.get("landmarks", [])):
                if lm_idx >= self.config.max_landmarks:
                    break
                yaw_angles = extract_yaw_angles_from_bboxes(
                    landmark.get("bounding_boxes", []))
                yaw_vector = (yaw_angles_to_binary_vector(yaw_angles)
                              if yaw_angles else [0.0, 0.0, 0.0, 0.0])
                lm_positions[lm_idx] = torch.tensor([
                    [yaw_vector[0], yaw_vector[1]],
                    [yaw_vector[2], yaw_vector[3]],
                ])

            # Batch-hash all tag values
            tag_values = [t['value'] for t in tokens]
            hashed_values = self._value_hasher(tag_values) if tag_values else None

            for j, token_data in enumerate(tokens):
                lm_idx = token_data['landmark_idx']
                landmark_idx_emb = self._landmark_idx_embedding(
                    torch.tensor(lm_idx, device=dev))

                key_emb = self._key_embedding(
                    torch.tensor(token_data['key_idx'], device=dev))
                val_emb = hashed_values[j]

                if self.config.include_description_embeddings:
                    desc_proj = desc_projections.get(
                        lm_idx,
                        torch.zeros(self.config.description_projection_dim, device=dev))
                    token_vec = self._tag_projection(
                        torch.cat([key_emb, val_emb, desc_proj]))
                else:
                    token_vec = self._tag_projection(
                        torch.cat([key_emb, val_emb]))

                features[i, j] = token_vec + landmark_idx_emb

                # Set position from precomputed yaw
                if lm_idx in lm_positions:
                    positions[i, j] = lm_positions[lm_idx].to(dev)

                mask[i, j] = False

        return ExtractorOutput(
            features=features,
            positions=positions.to(dev),
            mask=mask.to(dev),
            debug={},
        )

    @property
    def output_dim(self):
        return self.config.token_dim

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return []
