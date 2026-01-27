
import common.torch.load_torch_deps
import torch
import pickle
import math
import json
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_config_types import (
    PanoramaSemanticLandmarkExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_embedding_dict_from_json, make_sentence_dict_from_pano_jsons,
    string_to_hash_embedding, parse_thing_and_place_from_sentence)


def yaw_angles_to_binary_vector(yaw_degrees: list[int]) -> list[float]:
    """
    Convert yaw angles to a 4D binary vector indicating which yaws are present.

    Since only yaws 0° (north), 90° (west), 180°, 270° are possible, this creates a 4-element
    vector where each element is 1.0 if that yaw is present, 0.0 otherwise.

    Args:
        yaw_degrees: List of yaw angles (each should be 0, 90, 180, or 270)

    Returns:
        4-element list of floats: [yaw_0_present, yaw_90_present, yaw_180_present, yaw_270_present]

    Examples:
        [0] -> [1.0, 0.0, 0.0, 0.0]
        [0, 90] -> [1.0, 1.0, 0.0, 0.0]
        [90, 270] -> [0.0, 1.0, 0.0, 1.0]
        [] -> [0.0, 0.0, 0.0, 0.0]
    """
    # Initialize all to 0.0
    vector = [0.0, 0.0, 0.0, 0.0]

    # Map yaw degrees to indices
    yaw_to_idx = {0: 0, 90: 1, 180: 2, 270: 3}

    for yaw in yaw_degrees:
        if yaw in yaw_to_idx:
            vector[yaw_to_idx[yaw]] = 1.0
        else:
            raise ValueError(f"Invalid yaw angle: {yaw}. Must be 0, 90, 180, or 270.")

    return vector


class PanoramaSemanticLandmarkExtractor(torch.nn.Module):
    """
    Extractor for panorama-based semantic landmarks.

    Unlike SemanticLandmarkExtractor which matches landmarks by OSM properties,
    this extractor works with landmarks extracted directly from panorama images
    by vision models. Landmarks are associated with panoramas by ID alone.
    """

    def __init__(self, config: PanoramaSemanticLandmarkExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self.files_loaded = False
        self.all_embeddings = None
        self.all_sentences = None
        self.panorama_metadata = None  # Maps pano_id -> list of (landmark_idx, custom_id, yaw_angles)

        # Check for hash-bit mode
        self._use_hash_bit = config.hash_bit_config is not None
        if self._use_hash_bit:
            self._hash_bit_dim = config.hash_bit_config.embedding_dim

    def load_files(self):
        """Load embeddings, sentences, and metadata from multi-city directory structure."""
        base_path = self.semantic_embedding_base_path / self.config.embedding_version

        if not base_path.exists():
            raise FileNotFoundError(f"Embedding base path does not exist: {base_path}")

        # Find all city directories (e.g., Chicago, Seattle, etc.)
        city_dirs = [d for d in base_path.iterdir() if d.is_dir()]

        if not city_dirs:
            raise FileNotFoundError(f"No city directories found in {base_path}")

        self.all_embeddings = {}
        self.all_sentences = {}
        self.panorama_metadata = {}
        for city_dir in city_dirs:
            city_name = city_dir.name
            print(f"Loading panorama landmarks for city: {city_name}")

            # Load embeddings (skip if using hash-bit mode)
            if not self._use_hash_bit:
                embedding_dir = city_dir / "embeddings"
                if embedding_dir.exists():
                    pickle_path = embedding_dir / "embeddings.pkl"
                    if pickle_path.exists():
                        with open(pickle_path, 'rb') as f:
                            embedding_data = pickle.load(f)
                            tensor, id_to_idx = embedding_data
                            city_embeddings = {}
                            for custom_id, idx in id_to_idx.items():
                                city_embeddings[custom_id] = tensor[idx].tolist()
                        print(f"  Loaded {len(city_embeddings)} embeddings from pickle")
                    else:
                        # Fall back to JSONL loading
                        city_embeddings = make_embedding_dict_from_json(
                            load_all_jsonl_from_folder(embedding_dir))
                        print(f"  Loaded {len(city_embeddings)} embeddings from JSONL")

                    self.all_embeddings.update(city_embeddings)

            # Load sentences (optional)
            sentence_dir = city_dir / "sentences"
            metadata_from_sentences = None
            if sentence_dir.exists():
                city_sentences, metadata_from_sentences, _ = make_sentence_dict_from_pano_jsons(
                    load_all_jsonl_from_folder(sentence_dir))
                self.all_sentences.update(city_sentences)
                print(f"  Loaded {len(city_sentences)} sentences")
            else:
                print(f"  Sentences dir {sentence_dir} does not exist, skipping loading sentences")

            # Load panorama metadata
            metadata_file = city_dir / "embedding_requests" / "panorama_metadata.jsonl"
            if metadata_file.exists():
                new_metadata = {}
                with open(metadata_file, 'r') as f:
                    for line in f:
                        meta = json.loads(line)
                        pano_id = meta["panorama_id"]
                        landmark_idx = meta["landmark_idx"]
                        custom_id = meta["custom_id"]
                        yaw_angles = meta.get("yaw_angles", [])

                        if pano_id not in new_metadata:
                            new_metadata[pano_id] = []

                        new_metadata[pano_id].append({
                            "landmark_idx": landmark_idx,
                            "custom_id": custom_id,
                            "yaw_angles": yaw_angles
                        })
                new_pano_metadata_len = len(new_metadata)
                old_metadata_size = len(self.panorama_metadata)
                print(f"  Loaded metadata for {len(new_metadata)} panoramas")
                if metadata_from_sentences is not None:
                    assert metadata_from_sentences == new_metadata
                self.panorama_metadata.update(new_metadata)
                assert len(self.panorama_metadata) == old_metadata_size + new_pano_metadata_len

        # Validate embeddings (skip if using hash-bit mode)
        if not self._use_hash_bit:
            assert len(self.all_embeddings) > 0, f"Failed to load any embeddings from {base_path}"
            assert len(next(iter(self.all_embeddings.values()))) >= self.config.openai_embedding_size, \
                f"Requested embedding length ({self.config.openai_embedding_size}) longer than available ({len(next(iter(self.all_embeddings.values())))})"

        # remove coordinates from keys in self.panorama_metadata:
        self.panorama_metadata = {k.split(",")[0]: v for k, v in self.panorama_metadata.items()}

        self.files_loaded = True

        print(f"Total embeddings loaded: {len(self.all_embeddings)}")
        print(f"Total panoramas with landmarks: {len(self.panorama_metadata)}")

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        """Extract panorama-based semantic landmark features."""
        if not self.files_loaded:
            self.load_files()

        batch_size = len(model_input.metadata)

        # Validate that this is panorama data
        for item in model_input.metadata:
            if 'pano_id' not in item:
                raise ValueError(
                    "PanoramaSemanticLandmarkExtractor requires panorama data with 'pano_id' field. "
                    "This extractor should only be used with panorama images, not satellite images.")

        # Determine max number of landmarks across batch
        valid_landmarks = []
        for item in model_input.metadata:
            pano_id = item['pano_id']
            if pano_id not in self.panorama_metadata:
                valid_landmarks.append(0)
                continue
            matching_landmarks = self.panorama_metadata[pano_id]

            num_landmarks = len(matching_landmarks)
            valid_landmarks.append(num_landmarks)

        max_num_landmarks = max(valid_landmarks) if valid_landmarks else 0

        # In hash-bit mode, each landmark produces 2 tokens (thing + place)
        tokens_per_landmark = 2 if self._use_hash_bit else 1
        max_num_tokens = max_num_landmarks * tokens_per_landmark

        # Initialize output tensors
        mask = torch.ones((batch_size, max_num_tokens), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_tokens, self.output_dim))
        positions = torch.zeros((batch_size, max_num_tokens, 2, 2))

        max_description_length = 0

        # Process each batch item
        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']

            # Find matching panorama metadata
            if pano_id not in self.panorama_metadata:
                print(f"Failed to find pano id {pano_id} in panorama metadata! Skipping landmark")
                continue
            matching_landmarks = self.panorama_metadata[pano_id]

            # Sort by landmark_idx to ensure consistent ordering
            matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

            for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                custom_id = landmark_meta["custom_id"]
                yaw_angles = landmark_meta["yaw_angles"]

                # Convert yaw angles to binary presence vector
                yaw_vector = yaw_angles_to_binary_vector(yaw_angles)

                if self._use_hash_bit:
                    # Hash-bit mode: parse thing/place from sentence and create 2 tokens
                    sentence = self.all_sentences.get(custom_id, "")
                    if not sentence:
                        print(f"Warning: missing sentence for {custom_id}")
                        continue

                    thing, place = parse_thing_and_place_from_sentence(sentence)

                    token_base_idx = landmark_idx * 2

                    # Token 0: thing (business name)
                    if thing is not None:
                        thing_emb = string_to_hash_embedding(thing, self._hash_bit_dim)
                        features[i, token_base_idx, :] = torch.from_numpy(thing_emb)
                        mask[i, token_base_idx] = False

                    # Token 1: place (address)
                    if place is not None:
                        place_emb = string_to_hash_embedding(place, self._hash_bit_dim)
                        features[i, token_base_idx + 1, :] = torch.from_numpy(place_emb)
                        mask[i, token_base_idx + 1] = False

                    # Duplicate position for both tokens
                    positions[i, token_base_idx, 0, :] = torch.tensor([yaw_vector[0], yaw_vector[1]])
                    positions[i, token_base_idx, 1, :] = torch.tensor([yaw_vector[2], yaw_vector[3]])
                    positions[i, token_base_idx + 1, 0, :] = torch.tensor([yaw_vector[0], yaw_vector[1]])
                    positions[i, token_base_idx + 1, 1, :] = torch.tensor([yaw_vector[2], yaw_vector[3]])

                    max_description_length = max(max_description_length, len(sentence.encode("utf-8")))

                else:
                    # Original mode: look up pre-computed embedding
                    if custom_id not in self.all_embeddings:
                        print(f"Warning: missing embedding for {custom_id}")
                        continue

                    # Crop embedding if needed
                    features[i, landmark_idx, :] = torch.tensor(
                        self.all_embeddings[custom_id])[:self.config.openai_embedding_size]

                    # Position format: [batch, num_landmarks, 2, 2]
                    # Split the 4D binary vector across 2 positions:
                    # - position 0: [yaw_0_present, yaw_90_present]
                    # - position 1: [yaw_180_present, yaw_270_present]
                    positions[i, landmark_idx, 0, :] = torch.tensor([yaw_vector[0], yaw_vector[1]])
                    positions[i, landmark_idx, 1, :] = torch.tensor([yaw_vector[2], yaw_vector[3]])

                    # Mark as valid (False = not masked)
                    mask[i, landmark_idx] = False

                    # Track max description length for debug tensor
                    max_description_length = max(
                        max_description_length,
                        len(self.all_sentences.get(custom_id, "").encode("utf-8")))

        # Re-normalize embeddings if we cropped them (only in non-hash-bit mode)
        if not self._use_hash_bit:
            features[~mask] = features[~mask] / torch.norm(features[~mask], dim=-1).unsqueeze(-1)

        debug = {}

        # Create debug tensor for sentences
        debug_tokens = max_num_tokens if self._use_hash_bit else max_num_landmarks
        sentence_debug = torch.zeros(
            (batch_size, debug_tokens, max_description_length), dtype=torch.uint8)

        for i, item in enumerate(model_input.metadata):
            pano_id = item['pano_id']
            if pano_id not in self.panorama_metadata:
                continue

            # Find matching panorama metadata
            matching_landmarks = self.panorama_metadata[pano_id]

            matching_landmarks = sorted(matching_landmarks, key=lambda x: x["landmark_idx"])

            for landmark_idx, landmark_meta in enumerate(matching_landmarks):
                custom_id = landmark_meta["custom_id"]
                sentence_bytes = self.all_sentences.get(custom_id, "").encode('utf-8')
                sentence_tensor = torch.tensor(list(sentence_bytes), dtype=torch.uint8)
                if self._use_hash_bit:
                    # Duplicate sentence for both tokens
                    token_base_idx = landmark_idx * 2
                    sentence_debug[i, token_base_idx, :len(sentence_bytes)] = sentence_tensor
                    sentence_debug[i, token_base_idx + 1, :len(sentence_bytes)] = sentence_tensor
                else:
                    sentence_debug[i, landmark_idx, :len(sentence_bytes)] = sentence_tensor
        debug["sentences"] = sentence_debug.to(model_input.image.device)

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug=debug)

    @property
    def output_dim(self):
        if self._use_hash_bit:
            return self._hash_bit_dim
        return self.config.openai_embedding_size

    @property
    def num_position_outputs(self):
        return 2

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        # This extractor doesn't use landmarks from the dataset
        # (it uses vision-extracted landmarks stored separately)
        return []
