"""Composable CLIP/Hash-bit text encoder for panorama landmarks."""

import common.torch.load_torch_deps
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    ComposableCLIPPanoExtractorConfig,
    CLIPEmbedMode,
    TextEncoderType,
    ExtractorDataRequirement,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.panorama_semantic_landmark_extractor import (
    yaw_angles_to_binary_vector, _extract_yaw_angles_from_bboxes, _iter_city_directories)
from experimental.overhead_matching.swag.model.composable_clip_osm_extractor import (
    init_clip_encoder, encode_texts_clip, encode_texts_hash)


class ComposableCLIPPanoExtractor(torch.nn.Module):
    """Composable text encoder for panorama landmarks.

    Supports multiple embedding modes:
    - proper_nouns_only: Encode proper nouns with CLIP/hash, sentences use pre-computed
    - sentences_only: Encode full sentences with CLIP/hash
    - both: Encode both proper nouns and sentences with CLIP/hash

    Supports two encoder types:
    - CLIP: Trainable CLIP text encoder with learnable type embeddings
    - Hash-bit: Fixed SHA256-based embeddings
    """

    def __init__(self, config: ComposableCLIPPanoExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()

        # Initialize encoder based on type
        self._use_clip = config.encoder_type == TextEncoderType.CLIP

        if self._use_clip:
            self.tokenizer, self.text_encoder = init_clip_encoder(config)
            # Learnable type embeddings: 0=proper_noun, 1=sentence
            hidden_size = self.text_encoder.config.hidden_size
            self.type_embedding = nn.Embedding(2, hidden_size)
        else:
            print(f"Using hash-bit encoder with dim={config.hash_bit_dim}")
            self.tokenizer = None
            self.text_encoder = None
            self.type_embedding = None

        # Learned projection for precomputed sentence embeddings when using hash_bit + proper_nouns_only
        # Projects from precomputed_embedding_size to hash_bit_dim instead of truncating
        if (not self._use_clip and
                config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY and
                config.precomputed_embedding_size != config.hash_bit_dim and
                config.use_sentence_projection):
            print(f"Creating sentence projection layer: {config.precomputed_embedding_size} -> {config.hash_bit_dim}")
            self.sentence_projection = nn.Linear(config.precomputed_embedding_size, config.hash_bit_dim)
        else:
            self.sentence_projection = None

        # Data storage
        self.panorama_data = {}  # pano_id -> list of {description, proper_nouns, yaw_angles}
        self.precomputed_tensor = None  # For sentence fallback in proper_nouns_only mode
        self.precomputed_id_to_idx = {}  # custom_id -> tensor row index
        self.files_loaded = False

    def load_files(self):
        """Load panorama data from v2.0 pickle format."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        self.panorama_data = {}
        precomputed_embeddings = []
        precomputed_ids = []

        for city_name, city_dir in _iter_city_directories(base_path):
            print(f"Loading composable CLIP pano data for city: {city_name}")

            pickle_path = city_dir / "embeddings" / "embeddings.pkl"
            if not pickle_path.exists():
                raise FileNotFoundError(f"Required embeddings.pkl not found at {pickle_path}")

            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)

            if not isinstance(data, dict) or data.get("version") != "2.0":
                raise ValueError(
                    f"{pickle_path} is not v2.0 format. "
                    f"Got version: {data.get('version') if isinstance(data, dict) else 'not a dict'}")

            # panoramas dict is required
            if "panoramas" not in data:
                raise KeyError(f"Missing 'panoramas' key in {pickle_path}")

            # Extract panorama data (descriptions + proper_nouns)
            landmark_count = 0
            for pano_id, pano_data in data["panoramas"].items():
                # pano_id may be "panoid,lat,lng," format - extract just the panoid for key
                pano_id_clean = pano_id.split(",")[0]

                if pano_id_clean not in self.panorama_data:
                    self.panorama_data[pano_id_clean] = []

                # landmarks list is required
                if "landmarks" not in pano_data:
                    raise KeyError(f"Missing 'landmarks' key for pano_id {pano_id}")

                for landmark in pano_data["landmarks"]:
                    # landmark_idx and bounding_boxes are required
                    landmark_idx = landmark["landmark_idx"]
                    bounding_boxes = landmark["bounding_boxes"]

                    # description and proper_nouns may be empty but should exist
                    description = landmark.get("description", "")
                    proper_nouns = landmark.get("proper_nouns", [])

                    if not description and not proper_nouns:
                        continue

                    yaw_angles = _extract_yaw_angles_from_bboxes(bounding_boxes)
                    # Use full pano_id for custom_id to match description_id_to_idx format
                    custom_id = f"{pano_id}__landmark_{landmark_idx}"

                    self.panorama_data[pano_id_clean].append({
                        "landmark_idx": landmark_idx,
                        "description": description,
                        "proper_nouns": proper_nouns,
                        "yaw_angles": yaw_angles,
                        "custom_id": custom_id,
                    })
                    landmark_count += 1

            print(f"  Loaded {landmark_count} landmarks")

            # Load pre-computed embeddings for sentence fallback (proper_nouns_only mode)
            if self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY:
                if "description_embeddings" in data and "description_id_to_idx" in data:
                    desc_tensor = data["description_embeddings"]
                    desc_id_to_idx = data["description_id_to_idx"]

                    offset = len(precomputed_ids)
                    for custom_id, idx in desc_id_to_idx.items():
                        self.precomputed_id_to_idx[custom_id] = idx + offset
                    precomputed_embeddings.append(desc_tensor)
                    precomputed_ids.extend(desc_id_to_idx.keys())
                    print(f"  Loaded {len(desc_id_to_idx)} pre-computed sentence embeddings")
                else:
                    print(f"  Warning: No pre-computed embeddings in {pickle_path}")

        if precomputed_embeddings:
            self.precomputed_tensor = torch.cat(precomputed_embeddings, dim=0)
            # Crop to configured size if needed
            if self.precomputed_tensor.shape[1] > self.config.precomputed_embedding_size:
                self.precomputed_tensor = self.precomputed_tensor[:, :self.config.precomputed_embedding_size]
            # Normalize
            self.precomputed_tensor = self.precomputed_tensor / torch.norm(
                self.precomputed_tensor, dim=-1, keepdim=True)

        self.files_loaded = True
        print(f"Total panoramas with landmarks: {len(self.panorama_data)}")

    def _encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts using configured encoder."""
        if self._use_clip:
            return encode_texts_clip(
                texts, device, self.tokenizer, self.text_encoder,
                self.config.max_text_length, self.config.freeze_encoder)
        else:
            return encode_texts_hash(texts, device, self.config.hash_bit_dim)

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """Extract embeddings based on configured mode."""
        if not self.files_loaded:
            self.load_files()

        batch_size = len(model_input.metadata)
        device = model_input.image.device

        # Validate panorama data
        for item in model_input.metadata:
            if 'pano_id' not in item:
                raise ValueError(
                    "ComposableCLIPPanoExtractor requires panorama data with 'pano_id' field.")

        # Collect all texts and metadata
        all_texts = []  # (text, type_id) where type_id: 0=proper_noun, 1=sentence
        text_info = []  # (batch_idx, local_token_idx, type_id, yaw_angles, custom_id)

        # Count max tokens per batch item for output tensor sizing
        max_tokens_per_item = []
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']
            landmarks = self.panorama_data.get(pano_id, [])

            token_count = 0
            for lm in landmarks:
                # These fields are set by load_files, so direct access is safe
                yaw_angles = lm["yaw_angles"]
                custom_id = lm["custom_id"]
                proper_nouns = lm["proper_nouns"]
                description = lm["description"]

                if self.config.embed_mode in [CLIPEmbedMode.PROPER_NOUNS_ONLY, CLIPEmbedMode.BOTH]:
                    for proper_noun in proper_nouns:
                        all_texts.append((proper_noun, 0))
                        text_info.append((i, token_count, 0, yaw_angles, custom_id))
                        token_count += 1

                if self.config.embed_mode in [CLIPEmbedMode.SENTENCES_ONLY, CLIPEmbedMode.BOTH]:
                    if description:
                        all_texts.append((description, 1))
                        text_info.append((i, token_count, 1, yaw_angles, custom_id))
                        token_count += 1
                elif self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY:
                    # Sentence uses pre-computed, but still counts as a token
                    if description:
                        text_info.append((i, token_count, 1, yaw_angles, custom_id))
                        token_count += 1

            max_tokens_per_item.append(token_count)

        max_tokens = max(max_tokens_per_item) if max_tokens_per_item else 0

        if max_tokens == 0:
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.output_dim), device=device),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=device),
                positions=torch.zeros((batch_size, 0, 2, 2), device=device),
                debug={})

        # Initialize output tensors
        mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)
        features = torch.zeros((batch_size, max_tokens, self.output_dim), device=device)
        positions = torch.zeros((batch_size, max_tokens, 2, 2), device=device)

        # Encode texts that need encoding (not pre-computed sentences in proper_nouns_only mode)
        # Track which text_info indices need encoding and map to encoded position
        texts_to_encode = []
        text_info_idx_to_enc_pos = {}  # text_info index -> position in encoded_embeddings

        for text_info_idx, (batch_idx, local_token_idx, type_id, yaw_angles, custom_id) in enumerate(text_info):
            # In proper_nouns_only mode, sentences (type_id=1) use pre-computed
            if self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY and type_id == 1:
                continue
            # Get the text from all_texts - need to find matching entry
            # Since we track text_info_idx, we need the corresponding all_texts entry
            # In PROPER_NOUNS_ONLY: all_texts only has proper nouns, text_info has both
            # In other modes: all_texts and text_info are aligned
            if self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY:
                # Count how many proper noun entries are before this text_info_idx
                all_texts_idx = sum(1 for i in range(text_info_idx) if text_info[i][2] == 0)
            else:
                all_texts_idx = text_info_idx
            text, _ = all_texts[all_texts_idx]
            text_info_idx_to_enc_pos[text_info_idx] = len(texts_to_encode)
            texts_to_encode.append((text, type_id))

        if texts_to_encode:
            encoded_embeddings = self._encode_texts([t for t, _ in texts_to_encode], device)

            # Add type embeddings for CLIP mode
            if self._use_clip:
                for enc_idx, (_, type_id) in enumerate(texts_to_encode):
                    type_emb = self.type_embedding(torch.tensor(type_id, device=device))
                    encoded_embeddings[enc_idx] = encoded_embeddings[enc_idx] + type_emb

                # Re-normalize after adding type embedding
                encoded_embeddings = encoded_embeddings / torch.norm(
                    encoded_embeddings, dim=-1, keepdim=True)

        # Fill output tensors and track token type counts
        proper_noun_count = 0
        sentence_count = 0

        for idx, (batch_idx, local_token_idx, type_id, yaw_angles, custom_id) in enumerate(text_info):
            # Get embedding
            if self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY and type_id == 1:
                # Use pre-computed embedding for sentence
                if custom_id not in self.precomputed_id_to_idx:
                    print(f"Warning: No pre-computed embedding for {custom_id}")
                    continue
                emb_idx = self.precomputed_id_to_idx[custom_id]
                embedding = self.precomputed_tensor[emb_idx].to(device)
                # Apply learned projection if available, otherwise truncate/pad
                if self.sentence_projection is not None:
                    embedding = self.sentence_projection(embedding)
                elif embedding.shape[0] > self.output_dim:
                    embedding = embedding[:self.output_dim]
                elif embedding.shape[0] < self.output_dim:
                    # Pad with zeros if pre-computed is smaller
                    padded = torch.zeros(self.output_dim, device=device)
                    padded[:embedding.shape[0]] = embedding
                    embedding = padded
            else:
                # Use encoded embedding
                if idx not in text_info_idx_to_enc_pos:
                    raise RuntimeError(
                        f"Internal error: text_info idx {idx} not found in text_info_idx_to_enc_pos. "
                        f"embed_mode={self.config.embed_mode}, type_id={type_id}")
                enc_pos = text_info_idx_to_enc_pos[idx]
                embedding = encoded_embeddings[enc_pos]

            features[batch_idx, local_token_idx, :] = embedding
            mask[batch_idx, local_token_idx] = False

            # Track token type counts
            if type_id == 0:
                proper_noun_count += 1
            else:
                sentence_count += 1

            # Position encoding from yaw angles
            yaw_vector = yaw_angles_to_binary_vector(yaw_angles) if yaw_angles else [0.0, 0.0, 0.0, 0.0]
            positions[batch_idx, local_token_idx, 0, :] = torch.tensor(
                [yaw_vector[0], yaw_vector[1]], device=device)
            positions[batch_idx, local_token_idx, 1, :] = torch.tensor(
                [yaw_vector[2], yaw_vector[3]], device=device)

        return ExtractorOutput(
            features=features,
            mask=mask,
            positions=positions,
            debug={
                'proper_noun_count': proper_noun_count,
                'sentence_count': sentence_count,
            })

    @property
    def output_dim(self):
        if self._use_clip:
            return self.text_encoder.config.hidden_size
        else:
            return self.config.hash_bit_dim

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return []
