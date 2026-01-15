"""Composable CLIP/Hash-bit text encoder for OSM landmarks.

Also contains shared CLIP encoder utilities used by ComposableCLIPPanoExtractor.
"""

import common.torch.load_torch_deps
import torch
import torch.nn as nn
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel
from experimental.overhead_matching.swag.model.swag_config_types import (
    ComposableCLIPOSMExtractorConfig,
    CLIPEmbedMode,
    TextEncoderType,
    ExtractorDataRequirement,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_sentence_dict_from_json,
    custom_id_from_props, load_embeddings, string_to_hash_embedding)


# Geometry type mapping
GEOMETRY_TYPE_TO_IDX = {
    'point': 0,
    'linestring': 1,
    'polygon': 2,
    'multipolygon': 3,
}


def init_clip_encoder(config, print_prefix: str = "") -> tuple[CLIPTokenizer, CLIPTextModel]:
    """Initialize CLIP tokenizer and text encoder with training configuration.

    Args:
        config: Config object with model_name, freeze_encoder, use_gradient_checkpointing
        print_prefix: Prefix for print statements (e.g. "  ")

    Returns:
        Tuple of (tokenizer, text_encoder)
    """
    print(f"{print_prefix}Loading CLIP text encoder: {config.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(config.model_name)
    text_encoder = CLIPTextModel.from_pretrained(config.model_name)

    if config.freeze_encoder:
        print(f"{print_prefix}  Freezing CLIP text encoder")
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False
    else:
        print(f"{print_prefix}  CLIP text encoder is trainable")
        text_encoder.train()
        for param in text_encoder.parameters():
            param.requires_grad = True
        if config.use_gradient_checkpointing:
            print(f"{print_prefix}  Enabling gradient checkpointing")
            text_encoder.gradient_checkpointing_enable()

    return tokenizer, text_encoder


def encode_texts_clip(
    texts: list[str],
    device: torch.device,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    max_text_length: int,
    freeze_encoder: bool,
) -> torch.Tensor:
    """Encode texts using CLIP text encoder.

    Args:
        texts: List of strings to encode (will be lowercased)
        device: Target device for output tensor
        tokenizer: CLIP tokenizer
        text_encoder: CLIP text encoder model
        max_text_length: Maximum token length for truncation
        freeze_encoder: Whether encoder is frozen (use no_grad)

    Returns:
        Normalized embeddings tensor of shape (len(texts), hidden_size)
    """
    if not texts:
        return torch.zeros((0, text_encoder.config.hidden_size), device=device)

    # Lowercase all text for consistent embeddings
    texts = [t.lower() for t in texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if freeze_encoder:
        with torch.no_grad():
            outputs = text_encoder(**inputs)
    else:
        outputs = text_encoder(**inputs)

    embeddings = outputs.pooler_output
    embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    return embeddings


def encode_texts_hash(
    texts: list[str],
    device: torch.device,
    hash_bit_dim: int,
) -> torch.Tensor:
    """Encode texts using hash-bit embeddings.

    Args:
        texts: List of strings to encode (will be lowercased)
        device: Target device for output tensor
        hash_bit_dim: Dimension of hash embedding

    Returns:
        Hash embeddings tensor of shape (len(texts), hash_bit_dim)
    """
    if not texts:
        return torch.zeros((0, hash_bit_dim), device=device)

    embeddings = []
    for text in texts:
        # Lowercase for consistent embeddings
        text = text.lower()
        emb = string_to_hash_embedding(text, hash_bit_dim)
        embeddings.append(torch.from_numpy(emb))
    return torch.stack(embeddings).to(device)


def _compute_landmark_position(landmark, sat_metadata) -> list[list[float]]:
    """Compute position of a single landmark relative to satellite center.

    Returns:
        List of 2 positions, each [y_offset, x_offset] from satellite center.

    Raises:
        KeyError: If required fields are missing from landmark or sat_metadata
        ValueError: If geometry type is not recognized
    """
    sat_y = sat_metadata["web_mercator_y"]
    sat_x = sat_metadata["web_mercator_x"]
    geometry = landmark["geometry_px"]

    if geometry.geom_type == "Point":
        return [
            [geometry.y - sat_y, geometry.x - sat_x],
            [geometry.y - sat_y, geometry.x - sat_x]]

    elif geometry.geom_type == "LineString":
        x, y = geometry.xy
        return [
            [y[0] - sat_y, x[0] - sat_x],
            [y[-1] - sat_y, x[-1] - sat_x]]

    elif geometry.geom_type in ["Polygon", "MultiPolygon"]:
        x_min, y_min, x_max, y_max = geometry.bounds
        return [
            [y_min - sat_y, x_min - sat_x],
            [y_max - sat_y, x_max - sat_x]]

    else:
        raise ValueError(
            f"Unknown geometry type: {geometry.geom_type}. "
            f"Expected Point, LineString, Polygon, or MultiPolygon."
        )


class ComposableCLIPOSMExtractor(torch.nn.Module):
    """Composable text encoder for OSM landmarks.

    Supports multiple embedding modes:
    - proper_nouns_only: Encode names/addresses with CLIP/hash, sentences use pre-computed
    - sentences_only: Encode full sentences with CLIP/hash
    - both: Encode both names/addresses and sentences with CLIP/hash

    Extracts name from 'name' tag, address from 'addr:street' and 'addr:housenumber'.
    Supports all geometry types with learned type embeddings (CLIP mode only).
    """

    def __init__(self, config: ComposableCLIPOSMExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()

        # Initialize encoder based on type
        self._use_clip = config.encoder_type == TextEncoderType.CLIP

        if self._use_clip:
            self.tokenizer, self.text_encoder = init_clip_encoder(config)
            hidden_size = self.text_encoder.config.hidden_size

            # Token type embeddings: 0=name, 1=address, 2=sentence
            self.token_type_embedding = nn.Embedding(3, hidden_size)

            # Geometry type embeddings: point, linestring, polygon, multipolygon
            self.geometry_type_embedding = nn.Embedding(4, hidden_size)
        else:
            print(f"Using hash-bit encoder with dim={config.hash_bit_dim}")
            self.tokenizer = None
            self.text_encoder = None
            self.token_type_embedding = None
            self.geometry_type_embedding = None

        # Data storage
        self.all_sentences = {}  # custom_id -> sentence
        self.precomputed_tensor = None  # For sentence fallback in proper_nouns_only mode
        self.precomputed_id_to_idx = {}  # custom_id -> tensor row index
        self.files_loaded = False

    def load_files(self):
        """Load sentence data and pre-computed embeddings."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        # Load sentences (required)
        sentence_directory = base_path / "sentences"
        if not sentence_directory.exists():
            raise FileNotFoundError(f"Required sentences directory not found at {sentence_directory}")
        self.all_sentences, _ = make_sentence_dict_from_json(
            load_all_jsonl_from_folder(sentence_directory))
        print(f"Loaded {len(self.all_sentences)} OSM sentences")

        # Load pre-computed embeddings (required)
        embedding_directory = base_path / "embeddings"
        if not embedding_directory.exists():
            raise FileNotFoundError(f"Required embeddings directory not found at {embedding_directory}")
        self.precomputed_tensor, self.precomputed_id_to_idx = load_embeddings(
            embedding_directory,
            output_dim=self.config.precomputed_embedding_size,
            normalize=True)
        print(f"Loaded {len(self.precomputed_id_to_idx)} pre-computed sentence embeddings")

        self.files_loaded = True

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

        # Collect all token info
        # token_info: (batch_idx, local_token_idx, token_type, geom_type_idx, position, custom_id, text_or_none)
        # token_type: 0=name, 1=address, 2=sentence
        # text_or_none: text to encode, or None if using pre-computed
        token_info_list = []

        for i, item in enumerate(model_input.metadata):
            local_token_idx = 0
            for landmark in item["landmarks"]:
                # geometry_px and pruned_props are required
                geometry = landmark["geometry_px"]
                geom_type_str = geometry.geom_type.lower()
                if geom_type_str not in GEOMETRY_TYPE_TO_IDX:
                    raise ValueError(
                        f"Unknown geometry type: {geom_type_str}. "
                        f"Expected one of {list(GEOMETRY_TYPE_TO_IDX.keys())}")
                geom_type_idx = GEOMETRY_TYPE_TO_IDX[geom_type_str]

                position = _compute_landmark_position(landmark, item)
                custom_id = custom_id_from_props(landmark["pruned_props"])

                if self.config.embed_mode in [CLIPEmbedMode.PROPER_NOUNS_ONLY, CLIPEmbedMode.BOTH]:
                    # Extract and encode name
                    name = landmark.get('name')
                    if name:
                        token_info_list.append((
                            i, local_token_idx, 0, geom_type_idx, position, custom_id, name))
                        local_token_idx += 1

                    # Extract and encode address
                    street = landmark.get('addr:street', '')
                    housenumber = landmark.get('addr:housenumber', '')
                    if street or housenumber:
                        address = f"{housenumber} {street}".strip()
                        token_info_list.append((
                            i, local_token_idx, 1, geom_type_idx, position, custom_id, address))
                        local_token_idx += 1

                # Handle sentence embedding
                if self.config.embed_mode == CLIPEmbedMode.PROPER_NOUNS_ONLY:
                    # Use pre-computed embedding
                    if custom_id in self.precomputed_id_to_idx:
                        token_info_list.append((
                            i, local_token_idx, 2, geom_type_idx, position, custom_id, None))
                        local_token_idx += 1
                elif self.config.embed_mode in [CLIPEmbedMode.SENTENCES_ONLY, CLIPEmbedMode.BOTH]:
                    # Encode sentence with CLIP/hash
                    sentence = self.all_sentences.get(custom_id)
                    if sentence:
                        token_info_list.append((
                            i, local_token_idx, 2, geom_type_idx, position, custom_id, sentence))
                        local_token_idx += 1
                    else:
                        print(f"Failed to get sentence for {custom_id}")

        if not token_info_list:
            return ExtractorOutput(
                features=torch.zeros((batch_size, 0, self.output_dim), device=device),
                mask=torch.ones((batch_size, 0), dtype=torch.bool, device=device),
                positions=torch.zeros((batch_size, 0, 2, 2), device=device),
                debug={})

        # Compute max tokens per batch item
        max_token_idx = {}
        for batch_idx, local_token_idx, *_ in token_info_list:
            max_token_idx[batch_idx] = max(max_token_idx.get(batch_idx, 0), local_token_idx + 1)
        max_tokens = max(max_token_idx.values()) if max_token_idx else 0

        # Initialize output tensors
        mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)
        features = torch.zeros((batch_size, max_tokens, self.output_dim), device=device)
        positions = torch.zeros((batch_size, max_tokens, 2, 2), device=device)

        # Collect texts to encode
        texts_to_encode = []
        encode_info_indices = []  # Which token_info entries need encoding

        for idx, (batch_idx, local_token_idx, token_type, geom_type_idx, position, custom_id, text) in enumerate(token_info_list):
            if text is not None:
                texts_to_encode.append(text)
                encode_info_indices.append(idx)

        # Encode texts
        if texts_to_encode:
            encoded_embeddings = self._encode_texts(texts_to_encode, device)

            # Add type embeddings for CLIP mode
            if self._use_clip:
                for enc_idx, info_idx in enumerate(encode_info_indices):
                    _, _, token_type, geom_type_idx, _, _, _ = token_info_list[info_idx]
                    token_type_emb = self.token_type_embedding(
                        torch.tensor(token_type, device=device))
                    geom_type_emb = self.geometry_type_embedding(
                        torch.tensor(geom_type_idx, device=device))
                    encoded_embeddings[enc_idx] = (
                        encoded_embeddings[enc_idx] + token_type_emb + geom_type_emb)

                # Re-normalize after adding type embeddings
                encoded_embeddings = encoded_embeddings / torch.norm(
                    encoded_embeddings, dim=-1, keepdim=True)

        # Fill output tensors
        encode_ptr = 0
        for idx, (batch_idx, local_token_idx, token_type, geom_type_idx, position, custom_id, text) in enumerate(token_info_list):
            if text is not None:
                # Use encoded embedding
                enc_pos = encode_info_indices.index(idx)
                embedding = encoded_embeddings[enc_pos]
            else:
                # Use pre-computed embedding for sentence
                if custom_id in self.precomputed_id_to_idx:
                    emb_idx = self.precomputed_id_to_idx[custom_id]
                    embedding = self.precomputed_tensor[emb_idx].to(device)
                    # Adjust dimensions if needed
                    if embedding.shape[0] > self.output_dim:
                        embedding = embedding[:self.output_dim]
                    elif embedding.shape[0] < self.output_dim:
                        padded = torch.zeros(self.output_dim, device=device)
                        padded[:embedding.shape[0]] = embedding
                        embedding = padded
                else:
                    print(f"Failed to find sentence embedding for {custom_id}")

            features[batch_idx, local_token_idx, :] = embedding
            mask[batch_idx, local_token_idx] = False

            # Set position
            positions[batch_idx, local_token_idx, 0, :] = torch.tensor(position[0], device=device)
            positions[batch_idx, local_token_idx, 1, :] = torch.tensor(position[1], device=device)

        return ExtractorOutput(
            features=features,
            mask=mask,
            positions=positions,
            debug={})

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
        return [ExtractorDataRequirement.LANDMARKS]
